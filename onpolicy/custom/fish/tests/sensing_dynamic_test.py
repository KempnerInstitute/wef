import sys, os; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cfg
import electric
from cfg import AGENT_PARAMS, AGENT_ELECTRIC_PARAMS, m_to_cm

from electric_scene import ElectricScene
from sensing import calculate_uniform_angles
from sensing import SensingParams, SensorModesEpisode
from sensing import DynamicBaselineModel
from sensing import _food_within_radius_mask


def get_electric_scene_fish_inputs(num_agents, fish_eods):
    assert len(fish_eods) == num_agents
    fish_monopole_positions = np.tile(
        AGENT_ELECTRIC_PARAMS["monopole_positions_ego"][None, ...], (num_agents, 1, 1)
    )
    fish_monopole_charges = np.tile(
        AGENT_ELECTRIC_PARAMS["monopole_charges"][None, ...], (num_agents, 1)
    )
    fish_contrasts = np.array(
        [cfg.ELECTRIC_CONSTANTS["fish_contrast"] for _ in range(num_agents)]
    )
    fish_contrasts = fish_contrasts * (~fish_eods).astype(float)
    fish_radii = np.array(
        [AGENT_PARAMS["body_radius_cm"] for _ in range(num_agents)]
    )
    fish_intrinsic_dipole_moments_ego = np.tile(
        np.asarray(AGENT_ELECTRIC_PARAMS["fish_intrinsic_dipole_moment"], dtype=float)[None, ...],
        (num_agents, 1),
    )
    return (
        fish_monopole_positions,
        fish_monopole_charges,
        fish_contrasts,
        fish_radii,
        fish_intrinsic_dipole_moments_ego,
    )


def plot_sensing_grid_from_data(
    data,
    outfile="ampullary_food_sensing_grid.png",
):
    """
    Plot/save a sensing grid from precomputed data dict.
    Supports ampullary and knollen (and can be extended later).
    """
    grid_x = data["grid_x"]
    grid_y = data["grid_y"]
    grid_vals = data["grid_vals"]

    with np.errstate(divide="ignore"):
        grid_z = np.clip(grid_vals, a_min=1e-30, a_max=None)
        if data.get("do_log_z", True):
            grid_z = np.log10(grid_z)

    fig, ax = plt.subplots(figsize=(7, 7))
    cs = ax.contourf(grid_x, grid_y, grid_z, levels=50, cmap="hot")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cbar = fig.colorbar(cs, cax=cax)

    # --- minimal sensor switch: choose min/max keys + label ---
    min_key, max_key = None, None
    if "sensor_min" in data and "sensor_max" in data:
        min_key, max_key = "sensor_min", "sensor_max"
        cbar.set_label(data.get("sensor_label", "log10(sensor value)"))
    else:
        cbar.set_label("log10(sensor value)")

    # --- optional threshold xcontours (only if we found a min/max pair) ---
    if min_key is not None:
        # contour_levels = np.log10([data[min_key], data[max_key]]) if data.get("do_log_z", True) else [data[min_key], data[max_key]]
        contour_levels = np.log10([data[min_key], data[max_key]]) if data.get("do_log_z", True) else [0, 1] # abs(data) goes from 0 to 1 after clipping
        contour_lines = ax.contour(
            grid_x,
            grid_y,
            grid_z,
            levels=contour_levels,
            colors="white",
            linewidths=0.9,
        )
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt="%.2f")

    ax.add_patch(
        plt.Circle(
            data["center"],
            data["body_radius_cm"],
            color="lightblue",
            label="fish",
        )
    )

    if data.get("detection_cm", None) is not None:
        ax.add_patch(
            plt.Circle(
                data["center"],
                data["detection_cm"],
                color="green",
                fill=False,
                linestyle="--",
                linewidth=1.0,
                label="Expected range",
            )
        )

    ax.set_title(data.get("title", "Sensing grid"))
    ax.set_xlabel("X Position (cm)")
    ax.set_ylabel("Y Position (cm)")
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize="small", framealpha=1.0)

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print("Saved", outfile)
    return outfile


## TEST USING SENSING.py
class _DummyEnv:
    def __init__(self, arena_size_cm, num_agents=1):
        self.np_random = np.random.default_rng(0)
        self.arena_size = arena_size_cm
        self.num_agents = num_agents

        # used by set_sensor_modes_episode()
        self.mormyromast_mode = 1
        self.ampullary_mode = 1
        self.knollen_mode = 1
        self.morm_selfimage_mode = 1
        self.morm_consimage_mode = 1
        self.active_agent_ids = list(range(num_agents))
        self.agent_objects = [
            _DummyAgent(
                agent_id=i,
                agent_size=i/num_agents,
            ) for i in range(num_agents)
        ]

class _DummyAgent:
    def __init__(self, agent_id, agent_size):
        self.agent_id = agent_id
        self.agent_size = agent_size


def test_max_food_sensing_radius_mask():
    fish_positions = np.array([[0.0, 0.0]])
    food_positions = np.array([[0.0, 0.0], [5.0, 0.0], [15.0, 0.0]])
    mask = _food_within_radius_mask(fish_positions, food_positions, max_radius_cm=10.0)
    assert mask.tolist() == [True, True, False]


def _threshold_verdict(signal, threshold, range_cm):
    ratio = signal / threshold if threshold > 0 else float("inf")
    if ratio > 3.0:
        verdict = f"threshold LOW  — detects well past {range_cm:.0f} cm; consider raising threshold"
    elif ratio >= 0.33:
        verdict = f"threshold OK   — signal {ratio:.2f}× threshold at {range_cm:.0f} cm"
    else:
        verdict = f"threshold HIGH — signal only {ratio:.3f}× threshold; undetectable at {range_cm:.0f} cm"
    print(f"    signal={signal:.3e}  threshold={threshold:.3e}  ratio={ratio:.2f}  → {verdict}")


def test_sensing_thresholds():
    """Single-point calibration: raw signal vs. threshold at stated max detection range."""
    arena_size_cm = (300, 300)
    center = np.array(arena_size_cm, dtype=float) / 2
    fish_orientation = np.pi / 2  # upward (+y); lateral = x-offset, axial = y-offset

    base_params = SensingParams(
        knollen_metadata_mode="absolute",
        collective_sensing_mode=1,
        morm_selfimage_mode=1,
        morm_consimage_mode=1,
        noise_frac_morm=0.0,
        noise_frac_amp=0.0,
        noise_frac_knollen=0.0,
        noise_frac_knollen_metadata=0.0,
        general_sensor_min=1e-30,
        do_reflection=False,
        ampullary_intrinsic_only=True,
        max_food_sensing_radius=None,
    )

    # [A food] Ampullary — food placed laterally at amp_food_detection_range
    amp_food_range_cm = AGENT_PARAMS["amp_food_detection_range_m"] * m_to_cm
    food_pos_af = center + np.array([amp_food_range_cm, 0.0])
    env_af = _DummyEnv(arena_size_cm=arena_size_cm, num_agents=1)
    model_af = DynamicBaselineModel(sensing_params=base_params, env=env_af)
    model_af.sensor_modes_episode = SensorModesEpisode(ampullary_mode=1)
    fish_pos_af = center[None, :]
    fish_ori_af = np.array([fish_orientation])
    fish_eods_af = np.array([False])
    mp, mc, fc, fr, fid = get_electric_scene_fish_inputs(1, fish_eods_af)
    model_af.electric_scene.update(
        fish_positions=fish_pos_af, fish_orientations=fish_ori_af, fish_eods=fish_eods_af,
        fish_monopole_positions=mp, fish_monopole_charges=mc,
        fish_dipole_positions=np.zeros((1, 0, 2)), fish_dipole_moments=np.zeros((1, 0, 2)),
        conductor_positions=food_pos_af[None, :],
        conductor_contrasts=np.array([cfg.ELECTRIC_CONSTANTS["food_contrast"]]),
        conductor_radii=np.array([cfg.ENV_PARAMS["food_radius_cm"]]),
        conductor_orientations=np.zeros((1,)),
        fish_intrinsic_dipole_moments_ego=fid,
        conductor_intrinsic_dipole_moments_ego=np.asarray(
            cfg.ENV_PARAMS["food_intrinsic_dipole_moment"], dtype=float)[None, ...],
        fish_contrasts=fc, fish_radii=fr,
    )
    _, _, _, _, raw_amp_food = model_af.sense_ampullary_all(fish_pos_af, fish_ori_af, fish_eods_af, env_af.np_random)
    amp_food_signal = float(np.max(np.abs(raw_amp_food[0])))

    # [A agent] Ampullary — other agent placed laterally at amp_fish_detection_range
    amp_agent_range_cm = AGENT_PARAMS["amp_fish_detection_range_m"] * m_to_cm
    other_pos_aa = center + np.array([amp_agent_range_cm, 0.0])
    env_aa = _DummyEnv(arena_size_cm=arena_size_cm, num_agents=2)
    model_aa = DynamicBaselineModel(sensing_params=base_params, env=env_aa)
    model_aa.sensor_modes_episode = SensorModesEpisode(ampullary_mode=1)
    fish_pos_aa = np.stack([center, other_pos_aa])
    fish_ori_aa = np.full(2, fish_orientation)
    fish_eods_aa = np.array([False, False])
    mp, mc, fc, fr, fid = get_electric_scene_fish_inputs(2, fish_eods_aa)
    model_aa.electric_scene.update(
        fish_positions=fish_pos_aa, fish_orientations=fish_ori_aa, fish_eods=fish_eods_aa,
        fish_monopole_positions=mp, fish_monopole_charges=mc,
        fish_dipole_positions=np.zeros((2, 0, 2)), fish_dipole_moments=np.zeros((2, 0, 2)),
        conductor_positions=np.zeros((0, 2), dtype=float),
        conductor_contrasts=np.zeros((0,), dtype=float),
        conductor_radii=np.zeros((0,), dtype=float),
        conductor_orientations=np.zeros((0,), dtype=float),
        fish_intrinsic_dipole_moments_ego=fid,
        conductor_intrinsic_dipole_moments_ego=np.zeros((0, 2), dtype=float),
        fish_contrasts=fc, fish_radii=fr,
    )
    _, _, _, _, raw_amp_agent = model_aa.sense_ampullary_all(fish_pos_aa, fish_ori_aa, fish_eods_aa, env_aa.np_random)
    amp_agent_signal = float(np.max(np.abs(raw_amp_agent[0])))

    # [M food] Mormyromast — food placed laterally at morm_food_detection_range
    morm_food_range_cm = AGENT_PARAMS["morm_food_detection_range_m"] * m_to_cm
    food_pos_mf = center + np.array([morm_food_range_cm, 0.0])
    env_mf = _DummyEnv(arena_size_cm=arena_size_cm, num_agents=1)
    model_mf = DynamicBaselineModel(sensing_params=base_params, env=env_mf)
    model_mf.sensor_modes_episode = SensorModesEpisode(mormyromast_mode=1)
    num_morm_real_mf = model_mf.agent_electrics.num_mormyromast_sensors_real
    fish_pos_mf = center[None, :]
    fish_ori_mf = np.array([fish_orientation])
    fish_eods_mf = np.array([True])
    mp, mc, fc, fr, fid = get_electric_scene_fish_inputs(1, fish_eods_mf)
    model_mf.electric_scene.update(
        fish_positions=fish_pos_mf, fish_orientations=fish_ori_mf, fish_eods=fish_eods_mf,
        fish_monopole_positions=mp, fish_monopole_charges=mc,
        fish_dipole_positions=np.zeros((1, 0, 2)), fish_dipole_moments=np.zeros((1, 0, 2)),
        conductor_positions=food_pos_mf[None, :],
        conductor_contrasts=np.array([cfg.ELECTRIC_CONSTANTS["food_contrast"]]),
        conductor_radii=np.array([cfg.ENV_PARAMS["food_radius_cm"]]),
        conductor_orientations=np.zeros((1,)),
        fish_intrinsic_dipole_moments_ego=fid,
        conductor_intrinsic_dipole_moments_ego=np.zeros((1, 2), dtype=float),
        fish_contrasts=fc, fish_radii=fr,
    )
    _, _, raw_morm_food = model_mf.sense_mormyromast_all(
        fish_pos_mf, fish_ori_mf, fish_eods_mf, env_mf.np_random,
        conductor_positions=food_pos_mf[None, :],
        conductor_contrasts=np.array([cfg.ELECTRIC_CONSTANTS["food_contrast"]]),
        conductor_radii=np.array([cfg.ENV_PARAMS["food_radius_cm"]]),
        conductor_orientations=np.zeros((1,)),
        conductor_intrinsic_dipole_moments_ego=np.zeros((1, 2), dtype=float),
    )
    morm_food_signal = float(np.max(np.abs(raw_morm_food[0, :num_morm_real_mf])))

    # [M agent] Mormyromast — other agent placed laterally at morm_agent_detection_range
    morm_agent_range_cm = AGENT_PARAMS["morm_agent_detection_range_m"] * m_to_cm
    other_pos_ma = center + np.array([morm_agent_range_cm, 0.0])
    env_ma = _DummyEnv(arena_size_cm=arena_size_cm, num_agents=2)
    model_ma = DynamicBaselineModel(sensing_params=base_params, env=env_ma)
    model_ma.sensor_modes_episode = SensorModesEpisode(mormyromast_mode=1)
    num_morm_real_ma = model_ma.agent_electrics.num_mormyromast_sensors_real
    fish_pos_ma = np.stack([center, other_pos_ma])
    fish_ori_ma = np.full(2, fish_orientation)
    fish_eods_ma = np.array([True, False])  # receiver EODs; other body distorts self-image
    mp, mc, fc, fr, fid = get_electric_scene_fish_inputs(2, fish_eods_ma)
    model_ma.electric_scene.update(
        fish_positions=fish_pos_ma, fish_orientations=fish_ori_ma, fish_eods=fish_eods_ma,
        fish_monopole_positions=mp, fish_monopole_charges=mc,
        fish_dipole_positions=np.zeros((2, 0, 2)), fish_dipole_moments=np.zeros((2, 0, 2)),
        conductor_positions=np.zeros((0, 2), dtype=float),
        conductor_contrasts=np.zeros((0,), dtype=float),
        conductor_radii=np.zeros((0,), dtype=float),
        conductor_orientations=np.zeros((0,), dtype=float),
        fish_intrinsic_dipole_moments_ego=fid,
        conductor_intrinsic_dipole_moments_ego=np.zeros((0, 2), dtype=float),
        fish_contrasts=fc, fish_radii=fr,
    )
    _, _, raw_morm_agent = model_ma.sense_mormyromast_all(
        fish_pos_ma, fish_ori_ma, fish_eods_ma, env_ma.np_random,
        conductor_positions=np.zeros((0, 2), dtype=float),
        conductor_contrasts=np.zeros((0,), dtype=float),
        conductor_radii=np.zeros((0,), dtype=float),
        conductor_orientations=np.zeros((0,), dtype=float),
        conductor_intrinsic_dipole_moments_ego=np.zeros((0, 2), dtype=float),
    )
    morm_agent_signal = float(np.max(np.abs(raw_morm_agent[0, :num_morm_real_ma])))

    # [K agent] Knollen — other agent placed axially at knollen_agent_detection_range
    knollen_range_cm = AGENT_PARAMS["knollen_agent_detection_range_m"] * m_to_cm
    other_pos_k = center + np.array([0.0, knollen_range_cm])  # axial = along +y
    env_k = _DummyEnv(arena_size_cm=arena_size_cm, num_agents=2)
    model_k = DynamicBaselineModel(sensing_params=base_params, env=env_k)
    model_k.sensor_modes_episode = SensorModesEpisode(knollen_mode=1)
    fish_pos_k = np.stack([center, other_pos_k])
    fish_ori_k = np.full(2, fish_orientation)
    fish_eods_k = np.array([False, True])  # other fish EODs; receiver detects it
    mp, mc, fc, fr, fid = get_electric_scene_fish_inputs(2, fish_eods_k)
    model_k.electric_scene.update(
        fish_positions=fish_pos_k, fish_orientations=fish_ori_k, fish_eods=fish_eods_k,
        fish_monopole_positions=mp, fish_monopole_charges=mc,
        fish_dipole_positions=np.zeros((2, 0, 2)), fish_dipole_moments=np.zeros((2, 0, 2)),
        conductor_positions=np.zeros((0, 2), dtype=float),
        conductor_contrasts=np.zeros((0,), dtype=float),
        conductor_radii=np.zeros((0,), dtype=float),
        conductor_orientations=np.zeros((0,), dtype=float),
        fish_intrinsic_dipole_moments_ego=fid,
        conductor_intrinsic_dipole_moments_ego=np.zeros((0, 2), dtype=float),
        fish_contrasts=fc, fish_radii=fr,
    )
    _, _, raw_knollen, _ = model_k.sense_knollen_all(
        fish_pos_k, fish_ori_k, fish_eods_k, env_k.np_random,
        env_k.active_agent_ids,
        np.array([a.agent_size for a in env_k.agent_objects]),
    )
    knollen_signal = float(np.max(np.abs(raw_knollen[0])))

    print("=== Dynamic Sensing Threshold Calibration ===")
    print(f"  [A food]  Ampullary   food lateral   at {amp_food_range_cm:.1f} cm:")
    _threshold_verdict(amp_food_signal, model_af.sensing_params.amp_sensor_min, amp_food_range_cm)
    print(f"  [A agent] Ampullary   agent lateral  at {amp_agent_range_cm:.1f} cm:")
    _threshold_verdict(amp_agent_signal, model_aa.sensing_params.amp_sensor_min, amp_agent_range_cm)
    print(f"  [M food]  Mormyromast food lateral   at {morm_food_range_cm:.1f} cm:")
    _threshold_verdict(morm_food_signal, model_mf.morm_sensor_min, morm_food_range_cm)
    print(f"  [M agent] Mormyromast agent lateral  at {morm_agent_range_cm:.1f} cm:")
    _threshold_verdict(morm_agent_signal, model_ma.morm_sensor_min, morm_agent_range_cm)
    print(f"  [K agent] Knollen     agent axial    at {knollen_range_cm:.1f} cm:")
    _threshold_verdict(knollen_signal, model_k.sensing_params.knollen_sensor_min, knollen_range_cm)
    print()


# ------------ MORMYROMAST TESTS ------------
def compute_mormyromast_food_sensing_grid_model(
    num_points=50,
    arena_size_cm=(100, 100),
    buffer_cm=10,
    fish_orientation=np.pi / 2,
    raw_readings=False,
):
    center = np.array(arena_size_cm, dtype=float) / 2
    dummy_env = _DummyEnv(arena_size_cm=arena_size_cm, num_agents=1)

    sensing_params = SensingParams(
        knollen_metadata_mode="absolute",
        collective_sensing_mode=1,
        morm_selfimage_mode=1,
        morm_consimage_mode=1,
        noise_frac_morm=0.0,
        noise_frac_amp=0.0,
        noise_frac_knollen=0.0,
        noise_frac_knollen_metadata=0.0,
        general_sensor_min=1e-30,
        do_reflection=False,
        ampullary_intrinsic_only=True,
        max_food_sensing_radius=cfg.ENV_PARAMS.get("max_food_sensing_radius", None),
        max_charge_allowed=cfg.ENV_PARAMS.get("max_charge_allowed", None),
        body_radius_cm=AGENT_PARAMS["body_radius_cm"],
    )

    model = DynamicBaselineModel(sensing_params=sensing_params, env=dummy_env)
    model.sensor_modes_episode = SensorModesEpisode(mormyromast_mode=1)
    num_morm_real = model.agent_electrics.num_mormyromast_sensors_real
    morm_sensor_min = model.morm_sensor_min
    morm_sensor_max = model.morm_sensor_max

    food_x = np.linspace(center[0] - buffer_cm, center[0] + buffer_cm, num_points)
    food_y = np.linspace(center[1] - buffer_cm, center[1] + buffer_cm, num_points)
    food_pos_grid = np.array(np.meshgrid(food_x, food_y)).T.reshape(-1, 2)

    fish_positions = center[None, :]
    fish_orientations = np.array([fish_orientation], dtype=float)
    fish_eods = np.array([True], dtype=bool)

    (
        fish_monopole_positions,
        fish_monopole_charges,
        fish_contrasts,
        fish_radii,
        fish_intrinsic_dipole_moments_ego,
    ) = get_electric_scene_fish_inputs(num_agents=1, fish_eods=fish_eods)

    max_readings = np.zeros((len(food_pos_grid),), dtype=float)

    for i, food_pos in enumerate(food_pos_grid):
        model.electric_scene.update(
            fish_positions=fish_positions,
            fish_orientations=fish_orientations,
            fish_eods=fish_eods,
            fish_monopole_positions=fish_monopole_positions,
            fish_monopole_charges=fish_monopole_charges,
            fish_dipole_positions=np.zeros((1, 0, 2)),
            fish_dipole_moments=np.zeros((1, 0, 2)),
            conductor_positions=np.array([food_pos]),
            conductor_contrasts=np.array([cfg.ELECTRIC_CONSTANTS["food_contrast"]]),
            conductor_radii=np.array([cfg.ENV_PARAMS["food_radius_cm"]]),
            conductor_orientations=np.zeros((1,)),
            conductor_intrinsic_dipole_moments_ego=np.zeros((1, 2), dtype=float),
            fish_intrinsic_dipole_moments_ego=fish_intrinsic_dipole_moments_ego,
            fish_contrasts=fish_contrasts,
            fish_radii=fish_radii,
        )

        morm_obs, center_field_morm, raw_morm = model.sense_mormyromast_all(
            fish_positions,
            fish_orientations,
            fish_eods,
            dummy_env.np_random,
            conductor_positions=np.array([food_pos]),
            conductor_contrasts=np.array([cfg.ELECTRIC_CONSTANTS["food_contrast"]]),
            conductor_radii=np.array([cfg.ENV_PARAMS["food_radius_cm"]]),
            conductor_orientations=np.zeros((1,)),
            conductor_intrinsic_dipole_moments_ego=np.zeros((1, 2), dtype=float),
        )

        receiver_fish_idx = 0
        if raw_readings:
            max_readings[i] = np.max(np.abs(raw_morm[receiver_fish_idx, :num_morm_real])) # look at only self-images; cons-images are hard to interpret
        else:
            max_readings[i] = np.max(np.abs(morm_obs[receiver_fish_idx, :num_morm_real]))

    grid_x = food_pos_grid[:, 0].reshape(num_points, num_points)
    grid_y = food_pos_grid[:, 1].reshape(num_points, num_points)
    grid_vals = max_readings.reshape(num_points, num_points)
    sensor_min = 0 if not raw_readings else morm_sensor_min
    sensor_max = 1 if not raw_readings else morm_sensor_max

    return {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "grid_vals": grid_vals,
        "center": center,
        "body_radius_cm": AGENT_PARAMS["body_radius_cm"],
        "sensor_min": sensor_min,
        "sensor_max": sensor_max,
        "sensor_label": "log10(max |mormyromast reading|) (V/m)",
        "detection_cm": AGENT_PARAMS.get("morm_food_detection_range", 0.0) * m_to_cm,
        "title": "Mormyromast food sensing via sense_mormyromast_all" + (" (raw)" if raw_readings else ""),
        "do_log_z": True if raw_readings else False,
    }

def test_mormyromast_food_sensing_grid_model(
    outfile="mormyromast_food_sensing_grid_model.png",
    raw_readings=False,
):
    data = compute_mormyromast_food_sensing_grid_model(raw_readings=raw_readings)
    return plot_sensing_grid_from_data(data, outfile=outfile)

def compute_mormyromast_otherfish_sensing_grid_model(
    num_points=50,
    arena_size_cm=(100, 100),
    buffer_cm=15,
    fish_orientation=np.pi / 2,
    other_fish_orientation=np.pi / 2,
    raw_readings=False,
):
    center = np.array(arena_size_cm, dtype=float) / 2
    dummy_env = _DummyEnv(arena_size_cm=arena_size_cm, num_agents=2)

    sensing_params = SensingParams(
        knollen_metadata_mode="absolute",
        collective_sensing_mode=1,
        morm_selfimage_mode=1,
        morm_consimage_mode=1,
        noise_frac_morm=0.0,
        noise_frac_amp=0.0,
        noise_frac_knollen=0.0,
        noise_frac_knollen_metadata=0.0,
        general_sensor_min=1e-30,
        do_reflection=False,
        ampullary_intrinsic_only=True,
        max_food_sensing_radius=cfg.ENV_PARAMS.get("max_food_sensing_radius", None),
        max_charge_allowed=cfg.ENV_PARAMS.get("max_charge_allowed", None),
        body_radius_cm=AGENT_PARAMS["body_radius_cm"],
    )

    model = DynamicBaselineModel(sensing_params=sensing_params, env=dummy_env)
    model.sensor_modes_episode = SensorModesEpisode(mormyromast_mode=1)
    num_morm_real = model.agent_electrics.num_mormyromast_sensors_real
    morm_sensor_min = model.morm_sensor_min
    morm_sensor_max = model.morm_sensor_max

    xs = np.linspace(center[0] - buffer_cm, center[0] + buffer_cm, num_points)
    ys = np.linspace(center[1] - buffer_cm, center[1] + buffer_cm, num_points)
    other_pos_grid = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)

    max_readings = np.zeros((len(other_pos_grid),), dtype=float)
    receiver_idx = 0

    for i, other_pos in enumerate(other_pos_grid):
        fish_positions = np.stack([center, other_pos], axis=0)
        fish_orientations = np.array([fish_orientation, other_fish_orientation], dtype=float)
        fish_eods = np.array([True, False], dtype=bool)
        (
            fish_monopole_positions,
            fish_monopole_charges,
            fish_contrasts,
            fish_radii,
            fish_intrinsic_dipole_moments_ego,
        ) = get_electric_scene_fish_inputs(num_agents=len(fish_eods), fish_eods=fish_eods)

        model.electric_scene.update(
            fish_positions=fish_positions,
            fish_orientations=fish_orientations,
            fish_eods=fish_eods,
            fish_monopole_positions=fish_monopole_positions,
            fish_monopole_charges=fish_monopole_charges,
            fish_dipole_positions=np.zeros((2, 0, 2), dtype=float),
            fish_dipole_moments=np.zeros((2, 0, 2), dtype=float),
            conductor_positions=np.zeros((0, 2), dtype=float),
            conductor_contrasts=np.zeros((0,), dtype=float),
            conductor_radii=np.zeros((0,), dtype=float),
            conductor_orientations=np.zeros((0,), dtype=float),
            conductor_intrinsic_dipole_moments_ego=np.zeros((0, 2), dtype=float),
            fish_intrinsic_dipole_moments_ego=fish_intrinsic_dipole_moments_ego,
            fish_contrasts=fish_contrasts,
            fish_radii=fish_radii,
        )

        morm_obs, center_field_morm, raw_morm = model.sense_mormyromast_all(
            fish_positions,
            fish_orientations,
            fish_eods,
            dummy_env.np_random,
            conductor_positions=np.zeros((0, 2), dtype=float),
            conductor_contrasts=np.zeros((0,), dtype=float),
            conductor_radii=np.zeros((0,), dtype=float),
            conductor_orientations=np.zeros((0,), dtype=float),
            conductor_intrinsic_dipole_moments_ego=np.zeros((0, 2), dtype=float),
        )
        if raw_readings:
            max_readings[i] = np.max(np.abs(raw_morm[receiver_idx, :num_morm_real])) # look at only self-images; cons-images are hard to interpret
        else:
            max_readings[i] = np.max(np.abs(morm_obs[receiver_idx, :num_morm_real]))

    grid_x = other_pos_grid[:, 0].reshape(num_points, num_points)
    grid_y = other_pos_grid[:, 1].reshape(num_points, num_points)
    grid_vals = max_readings.reshape(num_points, num_points)
    sensor_min = 0 if not raw_readings else morm_sensor_min
    sensor_max = 1 if not raw_readings else morm_sensor_max

    return {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "grid_vals": grid_vals,
        "center": center,
        "body_radius_cm": AGENT_PARAMS["body_radius_cm"],
        "sensor_min": sensor_min,
        "sensor_max": sensor_max,
        "sensor_label": "log10(max |mormyromast reading|) (V/m)",
        "detection_cm": AGENT_PARAMS.get("morm_agent_detection_range", 0.0) * m_to_cm
            if "morm_agent_detection_range_m" in AGENT_PARAMS else None,
        "title": "Mormyromast sensing via sense_mormyromast_all" + (" (raw)" if raw_readings else ""),
        "do_log_z": True if raw_readings else False,
    }


def test_mormyromast_otherfish_sensing_grid_model(
    outfile="mormyromast_otherfish_sensing_grid_model.png",
    raw_readings=False,
):
    data = compute_mormyromast_otherfish_sensing_grid_model(raw_readings=raw_readings)
    return plot_sensing_grid_from_data(data, outfile=outfile)


if __name__ == "__main__":
    test_sensing_thresholds()

    # Mormyromast
    ext = ["png", "pdf"][1]
    test_mormyromast_otherfish_sensing_grid_model(outfile=f"mormyromast_otherfish_sensing_grid_model_dynamicbaseline_chk.{ext}", raw_readings=False)
    test_mormyromast_otherfish_sensing_grid_model(outfile=f"mormyromast_otherfish_sensing_grid_model_raw_dynamicbaseline_chk.{ext}", raw_readings=True)
    test_mormyromast_food_sensing_grid_model(outfile=f"mormyromast_food_sensing_grid_model_dynamicbaseline_chk.{ext}", raw_readings=False)
    test_mormyromast_food_sensing_grid_model(outfile=f"mormyromast_food_sensing_grid_model_raw_dynamicbaseline_chk.{ext}", raw_readings=True)

