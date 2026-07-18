from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial.distance import cdist

import cfg
import electric
from electric_scene import ElectricScene


# ---------------------------------------------------------------------------
# EMA filter
# ---------------------------------------------------------------------------

def compute_ema_alpha(cutoff_freq_hz, sampling_rate_hz):
    """Alpha parameter for EMA smoothing at the given cutoff frequency."""
    dt = 1 / sampling_rate_hz
    tau = 1 / (2 * np.pi * cutoff_freq_hz)
    return dt / (tau + dt)


# Cutoff used for ampullary EMA: 40 Hz at simulation frame rate
DEFAULT_AMPULLARY_ALPHA = compute_ema_alpha(40, cfg.ENV_PARAMS["fps_sim"])


# ---------------------------------------------------------------------------
# Sensor geometry helpers
# ---------------------------------------------------------------------------

def calculate_mormyromast_angles(num_real, num_chin, num_rest, asym_rays, chin_angle):
    """Calculate the fixed angles for each mormyromast sensor."""
    mormyromast_angles = {}
    if not asym_rays:
        for i in range(num_real):
            mormyromast_angles[i] = i * (2 * np.pi / num_real)
        return np.array(list(mormyromast_angles.values()))

    if num_chin > 0:
        chin_angles = np.linspace(-chin_angle / 2, chin_angle / 2, num_chin)
        for i, angle in enumerate(chin_angles):
            mormyromast_angles[i] = angle

    if num_rest > 0:
        remaining_angle = 2 * np.pi - chin_angle
        rest_angles = np.linspace(chin_angle / 2, chin_angle / 2 + remaining_angle, num_rest + 1)[:-1]
        for i, angle in enumerate(rest_angles):
            mormyromast_angles[i + num_chin] = angle

    return np.array(list(mormyromast_angles.values()))


def calculate_uniform_angles(num_sensors):
    angles = {}
    for i in range(num_sensors):
        angles[i] = i * (2 * np.pi / num_sensors)
    return np.array(list(angles.values()))


# ---------------------------------------------------------------------------
# FracRand processing helpers
# ---------------------------------------------------------------------------

def make_shared_morm_virtual_baselines(
    mormyromast_cd,
    num_morm_sets,
    num_mormyromast_sensors_real,
    mag_lower_limit_multiplier=2e-4,
    grid_style="linear_in_distance",
    shuffle_seed=42,
    exp=3,
):
    mag_max = np.max(np.abs(mormyromast_cd))
    mag_min = mag_lower_limit_multiplier * mag_max
    assert num_morm_sets > 0

    n = num_morm_sets * num_mormyromast_sensors_real

    if grid_style == "linear_in_logspace":
        grid = np.exp(np.linspace(np.log(mag_max), np.log(mag_min), n))
    elif grid_style == "linear":
        grid = np.linspace(mag_max, mag_min, n)
    elif grid_style == "log":
        grid = np.logspace(np.log10(mag_max), np.log10(mag_min), n)
    elif grid_style == "linear_in_distance":
        r_min = mag_max ** (-1.0 / exp)
        r_max = mag_min ** (-1.0 / exp)
        r = np.linspace(r_min, r_max, n)
        grid = r ** -exp
    else:
        grid = np.linspace(mag_max, mag_min, n)

    rng = np.random.default_rng(shuffle_seed)
    rng.shuffle(grid)

    mags = grid.reshape(num_morm_sets, num_mormyromast_sensors_real)
    signs = np.ones_like(mags)
    signs.flat[1::2] = -1.0
    morm_virtual_baselines = signs * mags
    morm_virtual_baselines[0] = mormyromast_cd
    return morm_virtual_baselines


def process_morm_fixed_normalization_range(
    raw_minus_baseline,
    baselines,
    max_mult=0.25,
    min_mult=0.25e-6,
    log_scale=True,
    do_clip=True,
    do_scale=True,
):
    """
    Shared-normalization version:
    - If do_clip: clip |v| to [min_mult * max|B|, max_mult * max|B|]
    - If do_scale: log/linear normalize to [0, 1] and restore sign to [-1, 1]
    - If not do_scale: return signed raw/clipped magnitudes (no normalization)
    """
    v = np.asarray(raw_minus_baseline)
    B = np.asarray(baselines)
    assert v.shape == B.shape, (v.shape, B.shape)

    Babs_max = np.max(np.abs(B))  # one max for all sensors
    sensor_mag_max = max_mult * Babs_max
    sensor_mag_min = min_mult * Babs_max

    signs = np.sign(v)
    mags = np.abs(v)

    if do_clip:
        clipped = np.clip(mags, sensor_mag_min, sensor_mag_max)
    else:
        clipped = mags

    if do_scale:
        if log_scale:
            log_min = np.log10(sensor_mag_min)
            log_max = np.log10(sensor_mag_max)
            log_range = log_max - log_min
            assert log_range > 0, log_range
            log_vals = np.log10(clipped)
            norm = (log_vals - log_min) / log_range
        else:
            norm = (clipped - sensor_mag_min) / (sensor_mag_max - sensor_mag_min)
        signed = signs * norm
    else:
        base = clipped if do_clip else mags
        signed = signs * base

    return signed


# ---------------------------------------------------------------------------
# Dynamic / shared processing helpers
# ---------------------------------------------------------------------------

def process_sensor_readings(
    sensor_values,
    sensor_min,
    sensor_max,
    log_scale=True,
    min_magnitude=1e-25,
    do_clip=True,
    do_scale=True,
):
    """
    Post corollary-discharge subtraction sensor processing pipeline.

    If do_scale: clip → log/linear scale → normalize to [-1, 1].
    If not do_scale: optionally clip, return signed raw/clipped magnitudes.
    """
    assert sensor_min < sensor_max, "Sensor min must be less than sensor max"
    assert sensor_min > 0, "Sensor min must be greater than 0"
    assert min_magnitude <= sensor_min, "Min magnitude must be <= sensor min"

    signs = np.sign(sensor_values)
    magnitudes = np.abs(sensor_values)

    if do_clip:
        clipped_magnitudes = np.clip(magnitudes, sensor_min, sensor_max)
    else:
        clipped_magnitudes = magnitudes

    if do_scale:
        if log_scale:
            log_min = np.log10(sensor_min)
            log_max = np.log10(sensor_max)
            log_range = log_max - log_min
            log_magnitudes = np.log10(np.maximum(clipped_magnitudes, min_magnitude))
            normalized_magnitudes = (log_magnitudes - log_min) / log_range
        else:
            normalized_magnitudes = (clipped_magnitudes - sensor_min) / (sensor_max - sensor_min)
        normalized_readings = signs * normalized_magnitudes
    else:
        base_magnitudes = clipped_magnitudes if do_clip else magnitudes
        normalized_readings = signs * base_magnitudes

    return normalized_readings


def log_scale_with_sign(x, min_magnitude=1e-25):
    """Apply log scaling while preserving signs. Small values clipped to min_magnitude."""
    signs = np.sign(x)
    magnitudes = np.abs(x)
    log_magnitudes = np.log10(np.maximum(magnitudes, min_magnitude))
    log_magnitudes += -np.log10(min_magnitude)  # shift to [0, inf)
    return signs * log_magnitudes


def clip_to_sensor_range(sensor_values, sensor_min, sensor_max):
    if sensor_min is None and sensor_max is None:
        return sensor_values
    abs_sensor_values = np.abs(sensor_values)
    sensor_values_signs = np.sign(sensor_values)
    clipped_sensor_values = np.clip(abs_sensor_values, a_min=sensor_min, a_max=sensor_max)
    scaled_sensor_values = (clipped_sensor_values - sensor_min) / (sensor_max - sensor_min)
    return sensor_values_signs * scaled_sensor_values


def binarize_sensor_values(sensor_values, threshold, sensor_min):
    abs_sensor_values = np.abs(sensor_values)
    sensor_values_signs = np.sign(sensor_values)
    within_sensing_range = abs_sensor_values > sensor_min
    sensor_values_signs[~within_sensing_range] = 0
    binarized_sensor_values = np.where(abs_sensor_values > threshold, 1, 0)
    return sensor_values_signs * binarized_sensor_values


def _food_within_radius_mask(fish_positions_cm, food_positions_cm, max_radius_cm):
    if max_radius_cm is None:
        return np.ones((food_positions_cm.shape[0],), dtype=bool)
    if food_positions_cm.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    distances = cdist(fish_positions_cm, food_positions_cm)
    min_distances = np.min(distances, axis=0)
    return min_distances < max_radius_cm


@dataclass
class AgentElectrics:
    """
    Episode-persistent, agent-specific electric calibration + geometry.
    """
    # TODO: add angles?
    # geometry (ego frame)
    # morm_pos_ego: np.ndarray       # (m,2)
    # morm_norm_ego: np.ndarray      # (m,2)  # TODO: possibly clarify names?
    # amp_pos_ego:  np.ndarray       # (a,2)
    # amp_norm_ego:  np.ndarray      # (a,2)
    # kno_pos_ego:  np.ndarray       # (k,2)  # per-agent knollen ring
    # kno_norm_ego:  np.ndarray      # (k,2)

    mormyromast_positions_ego: np.ndarray    # (m,2)
    mormyromast_normals_ego: np.ndarray      # (m,2)
    ampullary_positions_ego: np.ndarray      # (a,2)
    ampullary_normals_ego: np.ndarray        # (a,2)
    knollen_positions_ego: np.ndarray        # (k,2)
    knollen_normals_ego: np.ndarray          # (k,2)
    num_mormyromast_sensors_real: int
    num_mormyromast_sensors_virtual: int
    num_morm_sets: int
    num_ampullary_sensors: int
    num_knollen_sensors_real: int
    num_knollen_sensors_total: int
    mormyromast_indices_real: np.ndarray     # (m,)
    mormyromast_indices_virtual: np.ndarray  # (m,)
    ampullary_indices: np.ndarray            # (a,)
    knollen_indices: np.ndarray              # (k,)
    mormyromast_angles: np.ndarray           # (m,)
    ampullary_angles: np.ndarray             # (a,)
    knollen_angles: np.ndarray               # (k,)
    monopole_positions_ego: np.ndarray       # (p,2)

    monopole_charges: np.ndarray             # (p,)

    # calibration/baselines (episode-persistent)
    mormyromast_cd: np.ndarray # (m,)
    ampullary_cd: np.ndarray  # (a,)
    amp_intrinsic_baseline: np.ndarray   # (a,)
    morm_virtual_baselines: np.ndarray   # (num_sets, m)

    # center-field baselines (episode-persistent)
    center_morm_cd: np.ndarray           # (2,)
    center_amp_cd: np.ndarray            # (2,)
    center_amp_intrinsic_baseline: np.ndarray  # (2,)



@dataclass(frozen=True)
class SensingParams:
    """
    Does not change across episodes
    """
    # core sensing toggles
    knollen_metadata_mode: str  # "absolute" | "relative"
    # collective_sensing_mode controls which EOD sources are included in the raw mormyromast
    collective_sensing_mode: int  # 0=self-image only, 1=self/cons-images, 2=cons-image only
    morm_selfimage_mode: int  # 0=off, 1=on
    morm_consimage_mode: int  # 0=off, 1=on

    # noise
    noise_frac_morm: float
    noise_frac_amp: float
    noise_frac_knollen: float
    noise_frac_knollen_metadata: float

    # runtime flags
    ampullary_intrinsic_only: bool
    max_food_sensing_radius: float

    # sensor min/max — morm values are owned by each model subclass, not SensingParams
    general_sensor_min: float = cfg.AGENT_ELECTRIC_PARAMS["general_sensor_min"]
    amp_sensor_min: float = cfg.AGENT_ELECTRIC_PARAMS["ampullary_sensor_min"]
    amp_sensor_max: float = cfg.AGENT_ELECTRIC_PARAMS["ampullary_sensor_max"]
    knollen_sensor_min: float = cfg.AGENT_ELECTRIC_PARAMS["knollen_sensor_min"]
    knollen_binarize_threshold: float = cfg.AGENT_ELECTRIC_PARAMS["knollen_binarize_threshold"]
    knollen_sensor_max_log: float = cfg.AGENT_ELECTRIC_PARAMS["knollen_sensor_max_log"]
    max_charge_allowed: float = cfg.AGENT_ELECTRIC_PARAMS["max_charge_allowed"]
    body_radius_cm: float = cfg.AGENT_GEOMETRY_PARAMS["body_radius_cm"]
    num_morm_sets: int = cfg.AGENT_GEOMETRY_PARAMS["num_morm_sets"]  # overridden per model
    do_reflection: bool = cfg.ELECTRIC_CONSTANTS["do_wall_reflection"]

    # applicable only to dynamic baselines:
    subtract_cons_baseline: bool = True

    # ampullary noise when any conspecific EOD is emitting
    noise_frac_amp_cons_eod: float = 0.5
    # when True and cons-EOD fires, replace amp with sign(amp) — models complete EOD swamping
    amp_cons_eod_sign_only: bool = False

    # ampullary dynamics
    ampullary_ema: bool = False
    ampullary_alpha: float = DEFAULT_AMPULLARY_ALPHA


@dataclass
class SensorModesEpisode:
    """
    Changes per-episode; also can be overridden per-agent.
    """
    mormyromast_mode: int = 1
    ampullary_mode: int = 1
    knollen_mode: int = 1
    knollen_processing: str = "binarize"  # "binarize" | "log"


class SensingModel:
    """
    Abstracts sensor calibration + postprocessing policy.
    ElectricScene gives raw projected readings; SensingModel turns them into observations.

    Init builds ONE shared AgentElectrics (geometry + baselines) to reuse across all agents.
    Assumptions:
    - All agents share the same sensor geometry + monopole geometry (pulled from env.agent_objects[0]).
    - Corollary discharge / intrinsic baseline are computed using a single representative agent.
    - Virtual morm baselines (if relevant) are shared.
    """
    def __init__(self, sensing_params: SensingParams, env):
        self.sensing_params = sensing_params
        self.electric_scene = None
        self.agent_electrics = None
        self.num_agents = None

        # initialize the electric scene (used in corollary discharge calculation)
        self.electric_scene = ElectricScene(arena_size_cm=env.arena_size, max_charge_allowed=self.sensing_params.max_charge_allowed)
        geometry_params = {**cfg.AGENT_GEOMETRY_PARAMS, "num_morm_sets": self.sensing_params.num_morm_sets}
        self.agent_electrics = self._build_agent_electrics(
            agent_geometry_params=geometry_params,
            electric_params=cfg.AGENT_ELECTRIC_PARAMS,
            num_agents=env.num_agents,
            arena_size=env.arena_size
        )
        self.num_agents = env.num_agents
        self.ampullary_ema_buffer = np.zeros(
            (self.num_agents, self.agent_electrics.num_ampullary_sensors),
            dtype=float,
        )

    def _get_source_names(self, sensor_type):
        if sensor_type == "ampullary":
            if self.sensing_params.ampullary_intrinsic_only:
                # only intrinsic dipoles (fish + food)
                source_names = ["intrinsic"]
            else:
                # everything: eod + induced + intrinsic
                source_names = ["eod", "induced_food", "induced_fish", "intrinsic"]
        elif sensor_type == "mormyromast":
            source_names = ["eod", "induced_food", "induced_fish"]
        elif sensor_type == "knollen" or sensor_type == "morm_cons_baseline":
            source_names = ["eod"]
        else:
            raise ValueError(f"Unknown sensor type: {sensor_type}")

        return source_names

    def _build_sensor_geometry(self, agent_geometry_params, num_agents):
        num_mormyromast_sensors_real = agent_geometry_params["num_rays"]
        num_mormyromast_sensors_chin = agent_geometry_params["num_rays_chin"]
        num_mormyromast_sensors_rest = agent_geometry_params["num_rays_rest"]
        num_morm_sets = agent_geometry_params["num_morm_sets"]
        asym_rays = agent_geometry_params["asym_rays"]
        chin_angle = agent_geometry_params["chin_angle"]
        num_ampullary_sensors = agent_geometry_params["num_ampullary_sensors"]
        num_knollen_sensors = agent_geometry_params["num_knollen_sensors"]
        body_radius = agent_geometry_params["body_radius_cm"]
        num_mormyromast_sensors_virtual = num_mormyromast_sensors_real * num_morm_sets

        mormyromast_angles = calculate_mormyromast_angles(
            num_mormyromast_sensors_real,
            num_mormyromast_sensors_chin,
            num_mormyromast_sensors_rest,
            asym_rays,
            chin_angle,
        )
        ampullary_angles = calculate_uniform_angles(num_ampullary_sensors)
        knollen_angles = calculate_uniform_angles(num_knollen_sensors)
        num_knollen_sensors_total = num_knollen_sensors * (num_agents - 1)

        mormyromast_indices_real = np.arange(0, num_mormyromast_sensors_real).astype(int)
        mormyromast_indices_virtual = np.arange(0, num_mormyromast_sensors_virtual).astype(int)
        ampullary_indices = (len(mormyromast_indices_real) + np.arange(0, num_ampullary_sensors)).astype(int)
        knollen_indices = (
            len(mormyromast_indices_real)
            + len(ampullary_indices)
            + np.arange(0, num_knollen_sensors_total).astype(int)
        )

        mormyromast_positions_ego = np.array(
            [[body_radius * np.cos(angle), body_radius * np.sin(angle)] for angle in mormyromast_angles]
        )
        knollen_positions_ego = np.array(
            [[body_radius * np.cos(angle), body_radius * np.sin(angle)] for angle in knollen_angles]
        )
        ampullary_positions_ego = np.array(
            [[body_radius * np.cos(angle), body_radius * np.sin(angle)] for angle in ampullary_angles]
        )
        mormyromast_normals_ego = np.array([[np.cos(angle), np.sin(angle)] for angle in mormyromast_angles])
        knollen_normals_ego = np.array([[np.cos(angle), np.sin(angle)] for angle in knollen_angles])
        ampullary_normals_ego = np.array([[np.cos(angle), np.sin(angle)] for angle in ampullary_angles])

        return {
            "mormyromast_positions_ego": mormyromast_positions_ego,
            "mormyromast_normals_ego": mormyromast_normals_ego,
            "ampullary_positions_ego": ampullary_positions_ego,
            "ampullary_normals_ego": ampullary_normals_ego,
            "knollen_positions_ego": knollen_positions_ego,
            "knollen_normals_ego": knollen_normals_ego,
            "num_mormyromast_sensors_real": num_mormyromast_sensors_real,
            "num_mormyromast_sensors_virtual": num_mormyromast_sensors_virtual,
            "num_morm_sets": num_morm_sets,
            "num_ampullary_sensors": num_ampullary_sensors,
            "num_knollen_sensors_real": num_knollen_sensors,
            "num_knollen_sensors_total": num_knollen_sensors_total,
            "mormyromast_indices_real": mormyromast_indices_real,
            "mormyromast_indices_virtual": mormyromast_indices_virtual,
            "ampullary_indices": ampullary_indices,
            "knollen_indices": knollen_indices,
            "mormyromast_angles": mormyromast_angles,
            "ampullary_angles": ampullary_angles,
            "knollen_angles": knollen_angles,
            "monopole_positions_ego": agent_geometry_params["monopole_positions_ego"],
        }

    def _calculate_corollary_discharge(self, sensor_geometry, electric_params, arena_size):  # TODO add electric params from cfg or make decision
        # dummy agent position and orientation (in middle of the arena) for CD
        # NOTE: this is different from previously where we used each agent's actual position,
        # where walls could affect the CD calculation depending on initial position.
        agent_position = np.array([arena_size[0] / 2, arena_size[1] / 2], dtype=float)
        agent_orientation = 0.0
        agent_eod = True  # assume EOD is on for CD calculation

        # sense at sensors and center
        morm_query_positions = np.concatenate(
            [
                sensor_geometry["mormyromast_positions_ego"],
                np.zeros((1, 2)),   # center in ego frame
            ],
            axis=0,
        )  # (m + 1, 2)
        amp_query_positions = np.concatenate(
            [
                sensor_geometry["ampullary_positions_ego"],
                np.zeros((1, 2)),   # center in ego frame
            ],
            axis=0,
        )  # (a + 1, 2)

        # map ego -> world
        morm_query_pos_world = electric.rotate_and_translate(
            morm_query_positions,
            np.array([agent_orientation]),
            agent_position[None, :],
        )
        amp_query_pos_world = electric.rotate_and_translate(
            amp_query_positions,
            np.array([agent_orientation]),
            agent_position[None, :],
        )
        morm_normals_world = electric.rotate_and_translate(
            sensor_geometry["mormyromast_normals_ego"],
            np.array([agent_orientation]),
            translations_cm=0.0,
        )
        amp_normals_world = electric.rotate_and_translate(
            sensor_geometry["ampullary_normals_ego"],
            np.array([agent_orientation]),
            translations_cm=0.0,
        )

        # Choose which sources each sensor type senses
        morm_source_names = self._get_source_names("mormyromast")
        amp_source_names = self._get_source_names("ampullary")

        # Build electric scene for CD calculation (EOD only; no intrinsics in scene)
        self.electric_scene.update(
            fish_positions=np.array([agent_position]),
            fish_orientations=np.array([agent_orientation]),
            fish_eods=np.array([agent_eod]),
            fish_monopole_positions=sensor_geometry["monopole_positions_ego"][None, ...],
            fish_monopole_charges=electric_params["monopole_charges"][None, ...],
            fish_dipole_positions=np.zeros((1,0,2)),
            fish_dipole_moments=np.zeros((1,0,2)),
            conductor_positions=np.zeros((0,2)),
            conductor_contrasts=np.array([]),
            conductor_radii=np.array([]),
        )

        # Electric field vectors at query positions
        morm_field_vecs = self.electric_scene.get_field_at_positions(
            source_names=morm_source_names,
            positions_world=morm_query_pos_world,
            use_reflections=self.sensing_params.do_reflection,

        )
        amp_field_vecs = self.electric_scene.get_field_at_positions(
            source_names=amp_source_names,
            positions_world=amp_query_pos_world,
            use_reflections=self.sensing_params.do_reflection,
        )

        # Project onto sensor normals => raw scalar readings
        morm_cd = self.electric_scene.project_field_on_normals(
            morm_field_vecs[:-1, :],  # exclude center
            morm_normals_world,
        )

        amp_cd = self.electric_scene.project_field_on_normals(
            amp_field_vecs[:-1, :],  # exclude center
            amp_normals_world,
        )

        # Build scene for intrinsic baseline (fish intrinsic only; no EOD in scene)
        self.electric_scene.update(
            fish_positions=np.array([agent_position]),
            fish_orientations=np.array([agent_orientation]),
            fish_eods=np.array([agent_eod]),
            fish_monopole_positions=sensor_geometry["monopole_positions_ego"][None, ...],
            fish_monopole_charges=electric_params["monopole_charges"][None, ...],
            fish_dipole_positions=np.zeros((1,0,2)),
            fish_dipole_moments=np.zeros((1,0,2)),
            conductor_positions=np.zeros((0,2)),
            conductor_contrasts=np.array([]),
            conductor_radii=np.array([]),
            fish_intrinsic_dipole_moments_ego=np.array([electric_params["fish_intrinsic_dipole_moment"]]),  # TODO add this to electric params
        )
        amp_baseline_field_vecs = self.electric_scene.get_field_at_positions(
            source_names=["intrinsic"],
            positions_world=amp_query_pos_world,
            use_reflections=self.sensing_params.do_reflection,
        )
        amp_intrinsic_baseline = self.electric_scene.project_field_on_normals(
            amp_baseline_field_vecs[:-1, :],  # exclude center
            amp_normals_world,
        )
        # extract center readings
        center_morm_cd = morm_field_vecs[-1, :]  # (2,)
        center_amp_cd = amp_field_vecs[-1, :]    # (2,)
        center_amp_intrinsic_baseline = amp_baseline_field_vecs[-1, :]  # (2,)

        return {
            "mormyromast_cd": morm_cd,
            "ampullary_cd": amp_cd,
            "amp_intrinsic_baseline": amp_intrinsic_baseline,
            "center_morm_cd": center_morm_cd,
            "center_amp_cd": center_amp_cd,
            "center_amp_intrinsic_baseline": center_amp_intrinsic_baseline,
        }

    def _build_agent_electrics(self, agent_geometry_params, electric_params, num_agents, arena_size):   # NOTE: could also be a builder function on AgentElectrics if you pass in scene
        sensor_geometry = self._build_sensor_geometry(agent_geometry_params, num_agents)

        cd_amp_intrinsic_baselines = self._calculate_corollary_discharge(
            sensor_geometry,
            electric_params,
            arena_size)

        morm_virtual_baselines = make_shared_morm_virtual_baselines(
            cd_amp_intrinsic_baselines["mormyromast_cd"],
            sensor_geometry["num_morm_sets"],
            sensor_geometry["num_mormyromast_sensors_real"],
        )

        agent_electrics = AgentElectrics(
            **sensor_geometry,
            **cd_amp_intrinsic_baselines,
            morm_virtual_baselines=morm_virtual_baselines,
            monopole_charges=electric_params["monopole_charges"],
        )
        return agent_electrics

    def set_sensor_modes_episode(self, sensor_modes_episode):
        self.sensor_modes_episode = sensor_modes_episode
        self.ampullary_ema_buffer[:] = 0.0

    def _add_agent_electric_info(self, fish_info):
        assert "fish_positions_cm" in fish_info
        assert "fish_orientations" in fish_info
        assert "fish_eods" in fish_info
        assert "agent_sizes" in fish_info
        # TODO maybe also add in some checks on the shape of fish (or maybe that's being done elsewhere?)

        fish_contrasts = np.array([cfg.ELECTRIC_CONSTANTS["fish_contrast"] for _ in range(self.num_agents)])  # TODO put in agent electrics?
        fish_contrasts = fish_contrasts * (~fish_info["fish_eods"]).astype(float)
        fish_intrinsic_dipole_moments_ego = np.asarray(cfg.AGENT_ELECTRIC_PARAMS["fish_intrinsic_dipole_moment"], dtype=float)
        fish_radii = np.array([cfg.AGENT_PARAMS["body_radius_cm"] for _ in range(self.num_agents)])

        fish_info["fish_contrasts"] = fish_contrasts
        fish_info["fish_intrinsic_dipole_moments_ego"] = fish_intrinsic_dipole_moments_ego
        fish_info["fish_radii"] = fish_radii
        return fish_info

    def _add_food_electric_info(self, food_info):
        assert "food_positions_cm" in food_info
        assert "food_orientations" in food_info
        assert "food_radius" in food_info

        num_food = food_info["food_positions_cm"].shape[0]
        food_radii_cm = np.ones(num_food) * food_info["food_radius"]
        food_contrasts = np.ones(num_food) * cfg.ELECTRIC_CONSTANTS["food_contrast"]
        food_intrinsic_dipole_moments_ego = np.asarray(cfg.ENV_PARAMS["food_intrinsic_dipole_moment"], dtype=float)  # TODO: possibly move out of ENV_PARAMS
        food_info["food_radii_cm"] = food_radii_cm
        food_info["food_contrasts"] = food_contrasts
        food_info["food_intrinsic_dipole_moments_ego"] = food_intrinsic_dipole_moments_ego

        return food_info

    # specialty collective sensing function for computing self-EOD-only or cons-EOD-only scenes
    def _sense_morm_raw_masked_eods(
        self,
        fish_positions: np.ndarray,      # (f,2)
        fish_orientations: np.ndarray,   # (f,)
        fish_eods: np.ndarray,           # (f,)
        morm_pos_world: np.ndarray,      # (f,m,2)
        morm_norm_world: np.ndarray,     # (f,m,2)
        source_names,
        emitter_mode: str,               # "self" | "cons"
        conductor_positions=None,
        conductor_contrasts=None,
        conductor_radii=None,
        conductor_intrinsic_dipole_moments_ego=None,
        conductor_orientations=None,
        fish_radii=None,
    ):
        ae = self.agent_electrics
        f = fish_positions.shape[0]
        m = ae.num_mormyromast_sensors_real

        fish_contrasts_base = np.array(
            [cfg.ELECTRIC_CONSTANTS["fish_contrast"] for _ in range(self.num_agents)]
        )

        if fish_radii is None:
            fish_radii = np.array([cfg.AGENT_PARAMS["body_radius_cm"] for _ in range(self.num_agents)])

        raw_morm = np.zeros((f, m), dtype=float)
        center_field_vecs = np.zeros((f, 2), dtype=float)

        local_scene = ElectricScene(
            arena_size_cm=self.electric_scene.arena_size_cm,
            max_charge_allowed=self.sensing_params.max_charge_allowed,
        )

        for i in range(f):
            masked_eods = np.zeros_like(fish_eods, dtype=bool)
            if emitter_mode == "self":
                if fish_eods[i]:
                    masked_eods[i] = True
            elif emitter_mode == "cons":
                masked_eods[:] = fish_eods
                masked_eods[i] = False
            else:
                raise ValueError(f"Unknown emitter_mode: {emitter_mode}")
            if not np.any(masked_eods):
                continue

            # zero contrast for emitting fish, nonzero for non-emitting fish
            fish_contrasts = fish_contrasts_base * (~masked_eods).astype(float)

            local_scene.update(
                fish_positions=fish_positions,
                fish_orientations=fish_orientations,
                fish_eods=masked_eods,
                fish_monopole_positions=ae.monopole_positions_ego[None, ...],
                fish_monopole_charges=ae.monopole_charges[None, ...],
                fish_dipole_positions=np.zeros((1, 0, 2)),
                fish_dipole_moments=np.zeros((1, 0, 2)),
                conductor_positions=conductor_positions,
                conductor_contrasts=conductor_contrasts,
                conductor_radii=conductor_radii,
                conductor_intrinsic_dipole_moments_ego=conductor_intrinsic_dipole_moments_ego,
                conductor_orientations=conductor_orientations,
                fish_contrasts=fish_contrasts,
                fish_radii=fish_radii,
            )

            query_i = np.concatenate([morm_pos_world[i], fish_positions[i][None, :]], axis=0)
            field_vecs_i = local_scene.get_field_at_positions(
                source_names=source_names,
                positions_world=query_i,
                use_reflections=self.sensing_params.do_reflection,
            )

            center_field_vecs[i] = field_vecs_i[-1, :]
            raw_morm[i] = local_scene.project_field_on_normals(
                field_vecs_i[:-1, :],
                morm_norm_world[i],
            )

        return raw_morm, center_field_vecs


    def sense_step(
            self,
            fish_info,
            food_info,
            active_agent_ids,
            arena_size_cm,
            rng,
            ):  # equivalent to all_agents_sense
        """
        Compute and assign observations to all agents in the env.

        assumes fish_info has keys:
            - fish_positions_cm
            - fish_orientations
            - fish_eods
        assumes food_info has keys:
            - food_positions_cm
            - food_orientations

        mormyromast:
            - EODs
            - induced charges on food (+ from EOD reflections if ElectricScene.induce_with_reflections)
            - induced charges on fish (+ from EOD reflections if ElectricScene.induce_with_reflections)
            - reflections of all of the above (if ElectricScene.use_reflections)

        ampullary:
            - intrinsic charges on food/fish (if SensingParams.ampullary_intrinsic_only)
            - ampullary_intrinsic_only should be default/maybe deprecate case where it's False. if False, then also:
                - EODs
                - induced charges on food (+ from EOD reflections if ElectricScene.induce_with_reflections)
                - induced charges on fish (+ from EOD reflections if ElectricScene.induce_with_reflections)
            - reflections of all of the above (if ElectricScene.use_reflections)

        knollen:
            - EODs only
            - reflections of EODs (if ElectricScene.use_reflections) -- NOTE: currently omitted/commented out below. If SHS agrees, kill this line (and the commented-out code)
        """
        self.electric_scene.arena_size_cm = arena_size_cm
        # NOTE: set_sensor_modes_episode must be called at the start of the episode
        # 1) gather world state
        fish_info = self._add_agent_electric_info(fish_info)
        food_info = self._add_food_electric_info(food_info)
        if self.sensing_params.max_food_sensing_radius is not None:
            food_mask = _food_within_radius_mask(
                fish_info["fish_positions_cm"],
                food_info["food_positions_cm"],
                self.sensing_params.max_food_sensing_radius,
            )
            if food_mask.shape[0] > 0 and not np.all(food_mask):
                food_info["food_positions_cm"] = food_info["food_positions_cm"][food_mask]
                food_info["food_orientations"] = food_info["food_orientations"][food_mask]
                food_info["food_radii_cm"] = food_info["food_radii_cm"][food_mask]
                food_info["food_contrasts"] = food_info["food_contrasts"][food_mask]

        # 2) build electric scene (and its multiple source sets)
        self.electric_scene.update(
            fish_positions=fish_info["fish_positions_cm"],
            fish_orientations=fish_info["fish_orientations"],
            fish_eods=fish_info["fish_eods"],
            fish_monopole_positions=self.agent_electrics.monopole_positions_ego[None, ...],
            fish_monopole_charges=self.agent_electrics.monopole_charges[None, ...],
            fish_dipole_positions=np.zeros((1,0,2)),
            fish_dipole_moments=np.zeros((1,0,2)),
            conductor_positions=food_info["food_positions_cm"],
            conductor_contrasts=food_info["food_contrasts"],
            conductor_radii=food_info["food_radii_cm"],
            fish_contrasts=fish_info["fish_contrasts"],
            fish_radii=fish_info["fish_radii"],
            conductor_intrinsic_dipole_moments_ego=food_info["food_intrinsic_dipole_moments_ego"],
            conductor_orientations=food_info["food_orientations"],
            fish_intrinsic_dipole_moments_ego=fish_info["fish_intrinsic_dipole_moments_ego"],
        )

        # 3) sense each electroreceptor type (vectorized across all fish)
        center_fields = {}

        ampullary_obs, center_fields["ampullary"], _, _, _ = self.sense_ampullary_all(
            fish_info["fish_positions_cm"], fish_info["fish_orientations"], fish_info["fish_eods"], rng
        )

        morm_obs, center_fields["mormyromast"], morm_obs_raw = self.sense_mormyromast_all(
            fish_info["fish_positions_cm"],
            fish_info["fish_orientations"],
            fish_info["fish_eods"],
            rng,
            food_info["food_positions_cm"],
            food_info["food_contrasts"],
            food_info["food_radii_cm"],
            food_info["food_intrinsic_dipole_moments_ego"],
            food_info["food_orientations"],
        )

        knollen_obs, center_fields["knollen"], raw_knollen_obs, knollen_metadata = self.sense_knollen_all(
            fish_info["fish_positions_cm"], fish_info["fish_orientations"], fish_info["fish_eods"], rng, active_agent_ids, fish_info["agent_sizes"]
        )

        return ampullary_obs, morm_obs, knollen_obs, center_fields, knollen_metadata

    def sense_ampullary_all(
        self,
        fish_positions: np.ndarray,      # (f,2) cm
        fish_orientations: np.ndarray,   # (f,) radians
        fish_eods: np.ndarray,           # (f,) bool
        rng,
    ):
        """
        Compute ampullary observations for all agents from the given ElectricScene.

        Returns
        -------
        amp_obs : np.ndarray
            (f, num_ampullary) processed ampullary observations (post baseline/noise/clip/scale).
        center_field_amp : np.ndarray
            (f, 2) raw field vectors at fish centers (with the same baseline/CD adjustments applied).
        raw_amp : np.ndarray
            (f, num_ampullary) raw sensor projections before baseline subtraction.
        center_field_vecs : np.ndarray
            (f, 2) raw field vectors at fish centers before baseline subtraction.
        raw_amp_minus_baseline : np.ndarray
            (f, num_ampullary) raw projections after baseline subtraction, before noise/log/scale.
        """
        ae = self.agent_electrics
        # helpful shapes
        f = fish_positions.shape[0]
        num_amp = ae.num_ampullary_sensors

        # 1) map ego -> world for ALL agents
        # Shapes: (f,a,2)
        amp_pos_world = electric.rotate_and_translate(
            ae.ampullary_positions_ego[None, :, :],
            fish_orientations[:, None],
            fish_positions[:, None, :],
        )
        # Normals rotate only (no translation): (f,a,2)
        amp_norm_world = electric.rotate_and_translate(
            ae.ampullary_normals_ego[None, :, :],
            fish_orientations[:, None],
        )

        # Measure at sensors AND fish centers in one call for efficiency
        query_positions = np.concatenate(
            [amp_pos_world.reshape(-1, 2), fish_positions],
            axis=0,
        )  # (f*a + f, 2)

        # Choose which sources ampullary senses
        source_names = self._get_source_names("ampullary")

        # Electric field vectors at query positions: (f*a + f, 2)
        field_vecs = self.electric_scene.get_field_at_positions(
            source_names=source_names,
            positions_world=query_positions,
            use_reflections=self.sensing_params.do_reflection,
        )

        sensor_field_vecs = field_vecs[: f * num_amp].reshape(f, num_amp, 2)  # (f,a,2)
        center_field_vecs = field_vecs[f * num_amp :]                         # (f,2)

        # Project onto sensor normals => raw scalar readings (f,a)
        raw_amp = self.electric_scene.project_field_on_normals(
            sensor_field_vecs.reshape(-1, 2),
            amp_norm_world.reshape(-1, 2),
        ).reshape(f, num_amp)

        # ---- per-agent baseline/CD + noise + scaling ----
        amp_obs = np.zeros((f, num_amp), dtype=float)
        center_field_amp = np.zeros((f, 2), dtype=float)
        raw_amp_minus_baseline = np.zeros((f, num_amp), dtype=float)

        for i in range(len(fish_eods)):
            amp_i = raw_amp[i].copy()
            center_i = center_field_vecs[i].copy()

            # Subtract intrinsic baseline and CD always
            amp_i = amp_i - self.agent_electrics.amp_intrinsic_baseline
            center_i = center_i - self.agent_electrics.center_amp_intrinsic_baseline

            if not self.sensing_params.ampullary_intrinsic_only and fish_eods[i]:
                amp_i = amp_i - self.agent_electrics.ampullary_cd
                center_i = center_i - self.agent_electrics.center_amp_cd

            raw_amp_minus_baseline[i] = amp_i

            # If another fish emitted an EOD, this fish may receive either a noisy
            # ampullary magnitude signal or a fully swamped sign-only signal.
            cons_eod_emitting = (fish_eods.sum() - fish_eods[i]) > 0 # is there at least one conspecific emitter?
            amp_swamped_by_cons_eod = (
                cons_eod_emitting and self.sensing_params.amp_cons_eod_sign_only
            )

            # In sign-only swamping mode, skip magnitude noise for the swamped
            # case because the magnitude will be discarded after normalization.
            # Otherwise, keep the usual multiplicative noise path.
            if not amp_swamped_by_cons_eod:
                noise_frac_amp = (
                    self.sensing_params.noise_frac_amp_cons_eod
                    if cons_eod_emitting
                    else self.sensing_params.noise_frac_amp
                )
                if noise_frac_amp > 0.0:
                    amp_i *= rng.uniform(
                        1 - noise_frac_amp,
                        1 + noise_frac_amp,
                        size=amp_i.shape
                    )

            # Clip/log/normalize to [-1,1]
            amp_i = process_sensor_readings(
                amp_i,
                self.sensing_params.amp_sensor_min,
                self.sensing_params.amp_sensor_max,
                log_scale=True,
                min_magnitude=self.sensing_params.general_sensor_min,
                do_clip=True,
                do_scale=True,
            )

            if amp_swamped_by_cons_eod:
                # Model complete EOD swamping: discard all ampullary information (set to zero).
                amp_i = np.zeros_like(amp_i)

                # Model complete EOD swamping: preserve polarity, discard magnitude.
                # amp_i = np.sign(amp_i)


            # Zero if ampullary is disabled
            if self.sensor_modes_episode.ampullary_mode == 0:
                amp_i[:] = 0.0
                # NOTE: consider also zeroing center_i?

            amp_obs[i] = amp_i
            center_field_amp[i] = center_i

        if self.sensing_params.ampullary_ema:
            if self.sensor_modes_episode.ampullary_mode == 0:
                self.ampullary_ema_buffer[:] = 0.0
            else:
                alpha = self.sensing_params.ampullary_alpha
                self.ampullary_ema_buffer = (
                    alpha * amp_obs + (1 - alpha) * self.ampullary_ema_buffer
                )
                amp_obs = self.ampullary_ema_buffer.copy()

        return amp_obs, center_field_amp, raw_amp, center_field_vecs, raw_amp_minus_baseline

    def _emit_slot(self, i_sense, j_emit, k):
        # index of emitter j in the flattened (f-1)*k vector for sensing agent i
        slot = j_emit if j_emit < i_sense else (j_emit - 1)
        return slice(slot * k, (slot + 1) * k)

    def sense_knollen_all(
        self,
        fish_positions: np.ndarray,      # (f,2) cm
        fish_orientations: np.ndarray,   # (f,) radians
        fish_eods: np.ndarray,           # (f,) bool
        rng,
        active_agent_ids,  # TODO maybe put this elsewhere?
        agent_sizes,              # (f,)
    ):
        """
        Compute knollen observations for all agents from the given ElectricScene.

        Returns
        -------
        knollen_obs : np.ndarray
            (f, num_knollen_total) processed knollen observations (post baseline/noise/clip/scale).
        center_field_knollen : np.ndarray
            (f, f, 2) raw field vectors at fish centers (with the same baseline/CD adjustments applied).
        raw_knollen_all : np.ndarray
            (f, num_knollen_total) raw knollen readings before processing.
        knollen_metadata : dict
            per-agent dict of metadata about sensed knollen signals (e.g. emitter sizes).
        """
        f = fish_positions.shape[0]
        k = self.agent_electrics.num_knollen_sensors_real
        num_knollen_total = self.agent_electrics.num_knollen_sensors_total  # (k * (f - 1))
        knollen_obs = np.zeros((f, num_knollen_total), dtype=float)
        raw_knollen_all = np.zeros((f, num_knollen_total), dtype=float)
        center_field_knollen = np.zeros((f, f, 2), dtype=float)  # TODO: possibly switch to (f, f-1, 2) depending
        knollen_metadata = {
            i: {j: {"size": -1} for j in range(f) if j != i}
            for i in range(f)
        }
        if self.sensor_modes_episode.knollen_mode == 0:
            return knollen_obs, center_field_knollen, raw_knollen_all, knollen_metadata

        # map ego -> world for ALL agents
        kno_pos_world = electric.rotate_and_translate(
            self.agent_electrics.knollen_positions_ego[None, :, :],
            fish_orientations[:, None],
            fish_positions[:, None, :],
        )
        # Normals rotate only (no translation): (f,k,2)
        kno_norm_world = electric.rotate_and_translate(
            self.agent_electrics.knollen_normals_ego[None, :, :],
            fish_orientations[:, None],
        )
        query_positions = np.concatenate(
            [kno_pos_world.reshape(-1, 2), fish_positions],
            axis=0,
        )  # (f*k + f, 2)

        local_scene = ElectricScene(
            arena_size_cm=self.electric_scene.arena_size_cm,
            max_charge_allowed=self.sensing_params.max_charge_allowed,
        )

        for emitting_agent_id in range(self.num_agents):
            if emitting_agent_id not in active_agent_ids:
                continue
            if not fish_eods[emitting_agent_id]:
                continue  # TODO need to make sure initialization with zeros so that non-emitting agents are handled
            knollen_only_eods_onehot = np.zeros_like(fish_eods, dtype=bool)
            knollen_only_eods_onehot[emitting_agent_id] = True
            # rebuild electric scene with only one fish emitting EOD
            local_scene.update(
                fish_positions=fish_positions,
                fish_orientations=fish_orientations,
                fish_eods=knollen_only_eods_onehot,
                fish_monopole_positions=self.agent_electrics.monopole_positions_ego[None, ...],
                fish_monopole_charges=self.agent_electrics.monopole_charges[None, ...],
                fish_dipole_positions=np.zeros((1,0,2)),
                fish_dipole_moments=np.zeros((1,0,2)),
                conductor_positions=None,
                conductor_contrasts=None,
                conductor_radii=None,
            )

            # measure at sensors and fish centers
            source_names = self._get_source_names("knollen")
            field_vecs = local_scene.get_field_at_positions(
                source_names=source_names,
                positions_world=query_positions,
                # use_reflections=self.sensing_params.do_reflection,
                use_reflections=False,  # NOTE: omitting reflections for knollen
            )
            sensor_field_vecs = field_vecs[: f * k].reshape(f, k, 2)  # (f,k,2)
            center_field_vecs = field_vecs[f * k :]                   # (f,2)

            raw_knollen = local_scene.project_field_on_normals(
                sensor_field_vecs.reshape(-1, 2),
                kno_norm_world.reshape(-1, 2),
            ).reshape(f, k)

            for sensing_agent_id in range(self.num_agents):
                if sensing_agent_id not in active_agent_ids:
                    continue
                if sensing_agent_id == emitting_agent_id:
                    continue

                slot = self._emit_slot(sensing_agent_id, emitting_agent_id, k)
                raw_knollen_all[sensing_agent_id, slot] = raw_knollen[sensing_agent_id].copy()
                sensing_agent_knollen_obs = raw_knollen[sensing_agent_id].copy()
                sensing_agent_knollen_obs *= rng.uniform(
                    1 - self.sensing_params.noise_frac_knollen,
                    1 + self.sensing_params.noise_frac_knollen,
                    size=sensing_agent_knollen_obs.shape,
                )

                if self.sensor_modes_episode.knollen_processing == "log":
                    sensing_agent_knollen_obs = process_sensor_readings(
                        sensing_agent_knollen_obs,
                        sensor_min=self.sensing_params.knollen_sensor_min,
                        sensor_max=self.sensing_params.knollen_sensor_max_log,
                        log_scale=True,
                        min_magnitude=self.sensing_params.general_sensor_min,
                    )
                else:
                    sensing_agent_knollen_obs = binarize_sensor_values(
                        sensing_agent_knollen_obs,
                        self.sensing_params.knollen_binarize_threshold,
                        self.sensing_params.knollen_sensor_min,
                    )

                knollen_obs[sensing_agent_id, slot] = sensing_agent_knollen_obs

                if np.any(sensing_agent_knollen_obs != 0):
                    # TODO: possibly pass in agent sizes instead of using env
                    if self.sensing_params.knollen_metadata_mode == "relative":
                        knollen_metadata_value = agent_sizes[sensing_agent_id] - agent_sizes[emitting_agent_id]
                    elif self.sensing_params.knollen_metadata_mode == "absolute":
                        knollen_metadata_value = agent_sizes[emitting_agent_id]

                    if self.sensing_params.knollen_metadata_mode is not None:
                        knollen_metadata_value += rng.uniform(
                            -self.sensing_params.noise_frac_knollen_metadata,
                            self.sensing_params.noise_frac_knollen_metadata,
                        )
                        if self.sensing_params.knollen_metadata_mode == "absolute":
                            knollen_metadata_value = np.clip(knollen_metadata_value, 0.0, 1.0)
                        if self.sensing_params.knollen_metadata_mode == "relative":
                            knollen_metadata_value = np.clip(knollen_metadata_value, -1.0, 1.0)

                    knollen_metadata[sensing_agent_id][emitting_agent_id] = {"size": knollen_metadata_value}


                center_field_knollen[sensing_agent_id][emitting_agent_id] = center_field_vecs[sensing_agent_id]

        return knollen_obs, center_field_knollen, raw_knollen_all, knollen_metadata

    def sense_field_at_positions(
        self,
        positions_cm: np.ndarray,    # (n,2)
        source_names=['eod', 'induced_food', 'induced_fish'],
        use_reflections=True
    ) -> np.ndarray:
        """
        Utility to sense electric field vectors at arbitrary world positions from the shared ElectricScene.

        Parameters
        ----------
        positions_cm : np.ndarray
            (n,2) world positions to read field at.
        source_names : List[str]
            names of source sets to include in field calculation.
        use_reflections : bool
            whether to include reflections from conductors in the scene.

        Returns
        -------
        field_vecs : np.ndarray
            (n,2) electric field vectors at the query positions.
        """
        field_vecs = self.electric_scene.get_field_at_positions(
            source_names=source_names,
            positions_world=positions_cm,
            use_reflections=use_reflections,
        )
        return field_vecs


# =============================================== #
# =========== FRACRAND VIRTUAL BASELINES ======== #
# =============================================== #

class FracRandModel(SensingModel):

    def __init__(self, sensing_params: SensingParams, env):
        super().__init__(
            replace(sensing_params, num_morm_sets=cfg.SENSING_PARAMS_FRAC["num_morm_sets"]),
            env,
        )
        self.morm_sensor_min = cfg.SENSING_PARAMS_FRAC["mormyromast_sensor_min"]
        self.morm_sensor_max = cfg.SENSING_PARAMS_FRAC["mormyromast_sensor_max"]

    def sense_mormyromast_all(
        self,
        fish_positions: np.ndarray,      # (f,2) cm
        fish_orientations: np.ndarray,   # (f,) radians
        fish_eods: np.ndarray,           # (f,) bool
        rng,
        conductor_positions: np.ndarray,
        conductor_contrasts: np.ndarray,
        conductor_radii: np.ndarray,
        conductor_intrinsic_dipole_moments_ego: np.ndarray,
        conductor_orientations: np.ndarray,
    ):
        """
        Compute mormyromast observations for all agents from the given ElectricScene.

        Returns
        -------
        morm_obs_virtual : np.ndarray
            (f, num_mormyromast_virtual) processed mormyromast virtual observations (post baseline/noise/clip/scale).
        center_field_morm : np.ndarray
            (f, 2) raw field vectors at fish centers (with the same baseline/CD adjustments applied).
        morm_obs_virtual_raw : np.ndarray
            (f, num_mormyromast_virtual) raw virtual mormyromast readings before processing.
        """
        ae = self.agent_electrics

        f = fish_positions.shape[0]
        num_morm_real = ae.num_mormyromast_sensors_real
        num_morm_virtual = ae.num_mormyromast_sensors_virtual

        # 1) map ego -> world for ALL agents
        morm_pos_world = electric.rotate_and_translate(
            ae.mormyromast_positions_ego[None, :, :],
            fish_orientations[:, None],
            fish_positions[:, None, :],
        )
        morm_norm_world = electric.rotate_and_translate(
            ae.mormyromast_normals_ego[None, :, :],
            fish_orientations[:, None],
        )

        query_positions = np.concatenate(
            [morm_pos_world.reshape(-1, 2), fish_positions],
            axis=0,
        )  # (f*m + f, 2)

        # Choose which sources mormyromast senses
        source_names = self._get_source_names("mormyromast")

        if self.sensing_params.collective_sensing_mode == 0:
            raw_morm, center_field_vecs = self._sense_morm_raw_masked_eods(
                fish_positions,
                fish_orientations,
                fish_eods,
                morm_pos_world,
                morm_norm_world,
                source_names,
                emitter_mode="self",
                conductor_positions=conductor_positions,
                conductor_contrasts=conductor_contrasts,
                conductor_radii=conductor_radii,
                conductor_intrinsic_dipole_moments_ego=conductor_intrinsic_dipole_moments_ego,
                conductor_orientations=conductor_orientations,
            )

        elif self.sensing_params.collective_sensing_mode == 1:
            field_vecs = self.electric_scene.get_field_at_positions(
                source_names=source_names,
                positions_world=query_positions,
                use_reflections=self.sensing_params.do_reflection,
            )
            sensor_field_vecs = field_vecs[: f * num_morm_real].reshape(f, num_morm_real, 2)
            center_field_vecs = field_vecs[f * num_morm_real :]

            raw_morm = self.electric_scene.project_field_on_normals(
                sensor_field_vecs.reshape(-1, 2),
                morm_norm_world.reshape(-1, 2),
            ).reshape(f, num_morm_real)
        elif self.sensing_params.collective_sensing_mode == 2:
            raw_morm, center_field_vecs = self._sense_morm_raw_masked_eods(
                fish_positions,
                fish_orientations,
                fish_eods,
                morm_pos_world,
                morm_norm_world,
                source_names,
                emitter_mode="cons",
                conductor_positions=conductor_positions,
                conductor_contrasts=conductor_contrasts,
                conductor_radii=conductor_radii,
                conductor_intrinsic_dipole_moments_ego=conductor_intrinsic_dipole_moments_ego,
                conductor_orientations=conductor_orientations,
            )
        else:
            raise ValueError(f"Unknown collective_sensing_mode: {self.sensing_params.collective_sensing_mode}")

        morm_obs_virtual = np.zeros((f, num_morm_virtual), dtype=float)
        morm_obs_virtual_raw = np.zeros((f, num_morm_virtual), dtype=float)
        center_field_morm = np.zeros((f, 2), dtype=float)

        for i in range(len(fish_eods)):
            center_i = center_field_vecs[i].copy()
            if fish_eods[i] and self.sensing_params.collective_sensing_mode in [0, 1]:
                center_i = center_i - ae.center_morm_cd

            morm_real = raw_morm[i].copy()
            stacked = np.tile(morm_real[None, :], (ae.num_morm_sets, 1))
            if self.sensing_params.collective_sensing_mode in [0, 1]:
                values_after_baseline = stacked - ae.morm_virtual_baselines
                virtual_baseline_used = ae.morm_virtual_baselines
            elif self.sensing_params.collective_sensing_mode == 2:
                # mask out the self part of the baseline so that only the cons baseline is subtracted in cons-only mode
                # TODO: double-check this logic. If we're always subtracting out morm CD when self/cons EOD both enabled,
                # then maybe we don't need to do this masking.
                morm_virtual_baselines_cons_only = ae.morm_virtual_baselines.copy()
                morm_virtual_baselines_cons_only[0, :] = 0.0
                values_after_baseline = stacked - morm_virtual_baselines_cons_only
                virtual_baseline_used = morm_virtual_baselines_cons_only

            if self.sensing_params.noise_frac_morm > 0.0:
                values_after_baseline *= rng.uniform(
                    1 - self.sensing_params.noise_frac_morm,
                    1 + self.sensing_params.noise_frac_morm,
                    size=values_after_baseline.shape,
                )

            stacked_norm = process_morm_fixed_normalization_range(
                raw_minus_baseline=values_after_baseline,
                baselines=virtual_baseline_used,
                log_scale=True,
                do_clip=True,
                do_scale=True,
            )
            morm_virtual = stacked_norm.reshape(-1)

            # Zero if mormyromast [or subset] is disabled
            if self.sensing_params.morm_selfimage_mode == 0:
                morm_virtual[:num_morm_real] = 0.0
            if self.sensing_params.morm_consimage_mode == 0:
                morm_virtual[num_morm_real:] = 0.0
            if self.sensor_modes_episode.mormyromast_mode == 0:
                morm_virtual[:] = 0.0

            morm_obs_virtual[i] = morm_virtual
            morm_obs_virtual_raw[i] = values_after_baseline.reshape(-1)
            center_field_morm[i] = center_i

        return morm_obs_virtual, center_field_morm, morm_obs_virtual_raw


# =============================================== #
# =============== DYNAMIC BASELINES ============= #
# =============================================== #

class DynamicBaselineModel(SensingModel):

    def __init__(self, sensing_params: SensingParams, env):
        super().__init__(
            replace(sensing_params, num_morm_sets=cfg.SENSING_PARAMS_DYNAMIC["num_morm_sets"]),
            env,
        )
        self.morm_sensor_min = cfg.SENSING_PARAMS_DYNAMIC["mormyromast_sensor_min"]
        self.morm_sensor_max = cfg.SENSING_PARAMS_DYNAMIC["mormyromast_sensor_max"]

    def sense_mormyromast_all(
        self,
        fish_positions: np.ndarray,      # (f,2) cm
        fish_orientations: np.ndarray,   # (f,) radians
        fish_eods: np.ndarray,           # (f,) bool
        rng,
        conductor_positions: np.ndarray,
        conductor_contrasts: np.ndarray,
        conductor_radii: np.ndarray,
        conductor_intrinsic_dipole_moments_ego: np.ndarray,
        conductor_orientations: np.ndarray,
    ):
        """
        Compute mormyromast observations for all agents from the given ElectricScene.

        Returns
        -------
        morm_obs_virtual : np.ndarray
            (f, num_mormyromast_virtual) processed mormyromast virtual observations (post baseline/noise/clip/scale).
        center_field_morm : np.ndarray
            (f, 2) raw field vectors at fish centers (with the same baseline/CD adjustments applied).
        morm_obs_virtual_raw : np.ndarray
            (f, num_mormyromast_virtual) raw virtual mormyromast readings before processing.
        """
        # first half of this function is the same as in FracRandModel
        ae = self.agent_electrics

        f = fish_positions.shape[0]
        num_morm_real = ae.num_mormyromast_sensors_real
        num_morm_virtual = ae.num_mormyromast_sensors_virtual  # inclusive; any beyond num_morm_real will be zeroed out
        # NOTE: I think this departs from the old implementation, which made the observation space smaller.

        # 1) map ego -> world for ALL agents
        morm_pos_world = electric.rotate_and_translate(
            ae.mormyromast_positions_ego[None, :, :],
            fish_orientations[:, None],
            fish_positions[:, None, :],
        )
        morm_norm_world = electric.rotate_and_translate(
            ae.mormyromast_normals_ego[None, :, :],
            fish_orientations[:, None],
        )

        query_positions = np.concatenate(
            [morm_pos_world.reshape(-1, 2), fish_positions],
            axis=0,
        )  # (f*m + f, 2)

        # Choose which sources mormyromast senses
        source_names = self._get_source_names("mormyromast")

        # --- Step 1: Raw field acquisition ---
        # Mode 0 (self-only): each fish senses only its own EOD + conductors.
        # Mode 1 (self+cons): full scene already in ElectricScene; read directly.
        # Mode 2 (cons-only): each fish senses only the EODs of other fish + conductors.
        if self.sensing_params.collective_sensing_mode == 0:  # self only
            raw_morm, center_field_vecs = self._sense_morm_raw_masked_eods(
                fish_positions,
                fish_orientations,
                fish_eods,
                morm_pos_world,
                morm_norm_world,
                source_names,
                emitter_mode="self",
                conductor_positions=conductor_positions,
                conductor_contrasts=conductor_contrasts,
                conductor_radii=conductor_radii,
                conductor_intrinsic_dipole_moments_ego=conductor_intrinsic_dipole_moments_ego,
                conductor_orientations=conductor_orientations,
            )
        elif self.sensing_params.collective_sensing_mode == 1:  # self and cons
            # Already computed in the main ElectricScene, so just read directly without masking
            field_vecs = self.electric_scene.get_field_at_positions(
                source_names=source_names,
                positions_world=query_positions,
                use_reflections=self.sensing_params.do_reflection,
            )
            sensor_field_vecs = field_vecs[: f * num_morm_real].reshape(f, num_morm_real, 2)
            center_field_vecs = field_vecs[f * num_morm_real :]

            raw_morm = self.electric_scene.project_field_on_normals(
                sensor_field_vecs.reshape(-1, 2),
                morm_norm_world.reshape(-1, 2),
            ).reshape(f, num_morm_real)
        elif self.sensing_params.collective_sensing_mode == 2:  # cons only
            raw_morm, center_field_vecs = self._sense_morm_raw_masked_eods(
                fish_positions,
                fish_orientations,
                fish_eods,
                morm_pos_world,
                morm_norm_world,
                source_names,
                emitter_mode="cons",
                conductor_positions=conductor_positions,
                conductor_contrasts=conductor_contrasts,
                conductor_radii=conductor_radii,
                conductor_intrinsic_dipole_moments_ego=conductor_intrinsic_dipole_moments_ego,
                conductor_orientations=conductor_orientations,
            )
        else:
            raise ValueError(f"Invalid collective_sensing_mode: {self.sensing_params.collective_sensing_mode}")

        # This is where the implementation diverges from frac rand!
        morm_obs_virtual = np.zeros((f, num_morm_virtual), dtype=float)
        morm_obs_virtual_raw = np.zeros((f, num_morm_virtual), dtype=float)
        center_field_morm = np.zeros((f, 2), dtype=float)
        morm_obs_real = raw_morm.copy()

        # --- Step 2: Compute cons-baseline (EOD-only field, no induced charges) ---
        # Used in modes 1 and 2 to isolate the object-induced perturbation (cons-image).
        # Mode 1: read from the already-computed ElectricScene (EOD-only sources).
        # Mode 2: re-run the cons-only EOD scene without conductors.
        source_names_cons = self._get_source_names("morm_cons_baseline")
        if self.sensing_params.collective_sensing_mode == 1:
            field_vecs_no_induced = self.electric_scene.get_field_at_positions(
                source_names=source_names_cons,
                positions_world=query_positions,
                use_reflections=self.sensing_params.do_reflection,
            )
            sensor_field_vecs_no_induced = field_vecs_no_induced[: f * num_morm_real].reshape(f, num_morm_real, 2)
            center_field_vecs_no_induced = field_vecs_no_induced[f * num_morm_real :]
            morm_no_induced = self.electric_scene.project_field_on_normals(
                sensor_field_vecs_no_induced.reshape(-1, 2),
                morm_norm_world.reshape(-1, 2),
            ).reshape(f, num_morm_real)
        elif self.sensing_params.collective_sensing_mode == 2:
            morm_no_induced, center_field_vecs_no_induced = self._sense_morm_raw_masked_eods(
                fish_positions,
                fish_orientations,
                fish_eods,
                morm_pos_world,
                morm_norm_world,
                source_names_cons,
                emitter_mode="cons",
                conductor_positions=None,  # no conductors — gives pure cons-EOD baseline
                conductor_contrasts=None,
                conductor_radii=None,
                conductor_intrinsic_dipole_moments_ego=None,
                conductor_orientations=None,
            )

        # --- Step 3: Per-agent baseline subtraction and normalization ---
        for i in range(len(fish_eods)):
            center_i = center_field_vecs[i].copy()

            if self.sensing_params.collective_sensing_mode == 0:
                # Self-image only: subtract pre-computed self-EOD corollary discharge.
                # Residual = induced-charge distortion in agent's own EOD field.
                if fish_eods[i]:
                    morm_obs_real[i] -= ae.mormyromast_cd
                    center_i -= ae.center_morm_cd

            elif self.sensing_params.collective_sensing_mode == 1:
                # Self + cons scene. Two options:
                if self.sensing_params.subtract_cons_baseline:
                    # Subtract full EOD-only field (self + cons if emitting, cons-only if not).
                    # Residual = induced-charge perturbations from all conductors.
                    morm_obs_real[i] -= morm_no_induced[i]
                    center_i -= center_field_vecs_no_induced[i]
                else:
                    # No cons baseline subtraction: subtract self-CD only (if emitting).
                    # Unlike mode 0, raw_morm[i] came from the full scene, so the residual
                    # still contains the raw cons-EOD signal — expect saturation.
                    if fish_eods[i]:
                        morm_obs_real[i] -= ae.mormyromast_cd
                        center_i -= ae.center_morm_cd

            elif self.sensing_params.collective_sensing_mode == 2:
                # Cons-only scene: the raw reading IS the cons-EOD + its induced perturbations.
                if self.sensing_params.subtract_cons_baseline:
                    # Subtract cons-EOD-only field → residual = cons-image (object-induced
                    # perturbations in the cons-EOD field). Without subtraction the raw
                    # cons-EOD dominates and no image is recoverable.
                    morm_obs_real[i] -= morm_no_induced[i]
                    center_i -= center_field_vecs_no_induced[i]

            else:
                raise ValueError(f"Invalid collective_sensing_mode: {self.sensing_params.collective_sensing_mode}")

            values_after_baseline = morm_obs_real[i].copy()

            # Scale up the residual when it is purely cons-induced (tiny, attenuated ~x^{-3}).
            # This happens when:
            #   - mode 2 always (cons-only scene, no self-EOD contribution), or
            #   - mode 1 when this fish is NOT emitting (morm_no_induced = cons-EODs only,
            #     so residual = cons-induced only).
            # When this fish IS emitting in mode 1, the residual includes self-induced charges
            # at self-EOD scale — no scaling needed, sensor range applies directly.
            cons_only_residual = (
                self.sensing_params.collective_sensing_mode == 2
                or (self.sensing_params.collective_sensing_mode == 1 and not fish_eods[i])
            )
            if self.sensing_params.subtract_cons_baseline and cons_only_residual:
                values_after_baseline *= cfg.SENSING_PARAMS_DYNAMIC["mormyromast_sensor_cons_multiplier"]

            if self.sensing_params.noise_frac_morm > 0.0:
                values_after_baseline *= rng.uniform(
                    1 - self.sensing_params.noise_frac_morm,
                    1 + self.sensing_params.noise_frac_morm,
                    size=values_after_baseline.shape,
                )

            morm_processed = process_sensor_readings(
                sensor_values=values_after_baseline,
                sensor_min=self.morm_sensor_min,
                sensor_max=self.morm_sensor_max,
                log_scale=True,
                do_clip=True,
                do_scale=True,
            )
            morm_processed = morm_processed.reshape(-1)

            if self.sensor_modes_episode.mormyromast_mode == 0:
                morm_processed[:] = 0.0

            morm_obs_virtual[i, :num_morm_real] = morm_processed
            morm_obs_virtual_raw[i, :num_morm_real] = values_after_baseline.reshape(-1)
            center_field_morm[i] = center_i

        return morm_obs_virtual, center_field_morm, morm_obs_virtual_raw
