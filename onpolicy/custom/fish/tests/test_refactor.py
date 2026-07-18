"""
Tests for the cfg/sensing/obs-partition refactor.

Test types:
  1. cfg_integrity       — pure dict checks, no model instantiation
  2. sensing_params      — SensingParams defaults and per-model overrides
  3. agent_electrics     — model builds AgentElectrics with correct virtual counts
  4. obs_partitions_unit — get_obs_partitions with synthetic metadata, index structure
  5. obs_partitions_err  — get_obs_partitions raises on missing required keys
  6. obs_shape_integration — mormyromast obs shape matches virtual count end-to-end
"""

import sys, os; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np

import cfg
from sensing import (
    FracRandModel, DynamicBaselineModel, SensingParams, SensorModesEpisode,
)
from utils_sensors import get_obs_partitions


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

class _DummyAgent:
    def __init__(self, agent_id, agent_size=1.0):
        self.agent_id  = agent_id
        self.agent_size = agent_size


class _DummyEnv:
    def __init__(self, num_agents=2, arena_size_cm=(100, 100)):
        self.np_random     = np.random.default_rng(0)
        self.arena_size    = arena_size_cm
        self.num_agents    = num_agents
        self.mormyromast_mode = 1
        self.ampullary_mode   = 1
        self.knollen_mode     = 1
        self.morm_selfimage_mode  = 1
        self.morm_consimage_mode  = 1
        self.active_agent_ids = list(range(num_agents))
        self.agent_objects    = [_DummyAgent(i, i / num_agents) for i in range(num_agents)]


def _base_params(**overrides):
    defaults = dict(
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
    defaults.update(overrides)
    return SensingParams(**defaults)


def _synthetic_metadata(num_morm_virtual, num_morm_real, num_agents=4, **all_args_overrides):
    all_args = dict(
        feedback_action=True,
        enable_bite_action=True,
        use_bite_cooldown=True,
        agent_size_mode=None,
        feedback_displacement=False,
        num_agents=num_agents,
    )
    all_args.update(all_args_overrides)
    return {
        "all_args": all_args,
        "agent_args": {
            "num_mormyromast_sensors_virtual": num_morm_virtual,
            "num_mormyromast_sensors_real":    num_morm_real,
            "num_ampullary_sensors":           cfg.AGENT_GEOMETRY_PARAMS["num_ampullary_sensors"],
            "num_knollen_sensors":             cfg.AGENT_GEOMETRY_PARAMS["num_knollen_sensors"],
        },
    }


# ---------------------------------------------------------------------------
# 1. cfg integrity
# ---------------------------------------------------------------------------

def test_cfg_integrity():
    print("1. cfg_integrity")

    frac = cfg.SENSING_PARAMS_FRAC
    dyn  = cfg.SENSING_PARAMS_DYNAMIC

    assert "num_morm_sets" in frac,  "SENSING_PARAMS_FRAC missing num_morm_sets"
    assert "num_morm_sets" in dyn,   "SENSING_PARAMS_DYNAMIC missing num_morm_sets"
    assert frac["num_morm_sets"] == 4, f"expected 4, got {frac['num_morm_sets']}"
    assert dyn["num_morm_sets"]  == 1, f"expected 1, got {dyn['num_morm_sets']}"

    num_rays = cfg.AGENT_GEOMETRY_PARAMS["num_rays"]
    assert frac["num_morm_sets"] * num_rays == 144, "FracRand virtual count should be 144"
    assert dyn["num_morm_sets"]  * num_rays == 36,  "Dynamic virtual count should be 36"

    # AGENT_GEOMETRY_PARAMS still has a sane default
    assert cfg.AGENT_GEOMETRY_PARAMS["num_morm_sets"] == 1

    print("   PASS")


# ---------------------------------------------------------------------------
# 2. SensingParams defaults and per-model overrides
# ---------------------------------------------------------------------------

def test_sensing_params():
    print("2. sensing_params")

    base = _base_params()

    # Default num_morm_sets comes from AGENT_GEOMETRY_PARAMS (= 1)
    assert base.num_morm_sets == 1, f"default num_morm_sets should be 1, got {base.num_morm_sets}"

    env  = _DummyEnv(num_agents=2)
    frac = FracRandModel(base, env)
    dyn  = DynamicBaselineModel(base, env)

    assert frac.sensing_params.num_morm_sets == 4, \
        f"FracRand num_morm_sets should be 4, got {frac.sensing_params.num_morm_sets}"
    assert dyn.sensing_params.num_morm_sets  == 1, \
        f"Dynamic num_morm_sets should be 1, got {dyn.sensing_params.num_morm_sets}"

    # Morm thresholds: FracRand uses SENSING_PARAMS_FRAC value
    expected_frac_min = cfg.SENSING_PARAMS_FRAC["mormyromast_sensor_min"]
    assert frac.sensing_params.morm_sensor_min == expected_frac_min, \
        f"FracRand morm_sensor_min mismatch: {frac.sensing_params.morm_sensor_min} != {expected_frac_min}"

    # Dynamic uses AGENT_ELECTRIC_PARAMS default (no override in SENSING_PARAMS_DYNAMIC)
    expected_dyn_min = cfg.AGENT_ELECTRIC_PARAMS["mormyromast_sensor_min"]
    assert dyn.sensing_params.morm_sensor_min == expected_dyn_min, \
        f"Dynamic morm_sensor_min mismatch: {dyn.sensing_params.morm_sensor_min} != {expected_dyn_min}"

    print("   PASS")


# ---------------------------------------------------------------------------
# 3. AgentElectrics virtual sensor counts
# ---------------------------------------------------------------------------

def test_agent_electrics():
    print("3. agent_electrics")

    base = _base_params()
    env  = _DummyEnv(num_agents=2)
    frac = FracRandModel(base, env)
    dyn  = DynamicBaselineModel(base, env)

    num_rays = cfg.AGENT_GEOMETRY_PARAMS["num_rays"]

    assert frac.agent_electrics.num_mormyromast_sensors_virtual == 4 * num_rays, \
        f"FracRand virtual={frac.agent_electrics.num_mormyromast_sensors_virtual}, expected {4*num_rays}"
    assert frac.agent_electrics.num_morm_sets == 4

    assert dyn.agent_electrics.num_mormyromast_sensors_virtual  == 1 * num_rays, \
        f"Dynamic virtual={dyn.agent_electrics.num_mormyromast_sensors_virtual}, expected {num_rays}"
    assert dyn.agent_electrics.num_morm_sets  == 1

    # virtual baselines shape: (num_morm_sets, num_rays)
    assert frac.agent_electrics.morm_virtual_baselines.shape == (4, num_rays), \
        f"FracRand baselines shape {frac.agent_electrics.morm_virtual_baselines.shape}"
    assert dyn.agent_electrics.morm_virtual_baselines.shape  == (1, num_rays), \
        f"Dynamic baselines shape {dyn.agent_electrics.morm_virtual_baselines.shape}"

    print("   PASS")


# ---------------------------------------------------------------------------
# 4. get_obs_partitions: index structure (unit)
# ---------------------------------------------------------------------------

def test_obs_partitions_unit():
    print("4. obs_partitions_unit")

    num_rays = cfg.AGENT_GEOMETRY_PARAMS["num_rays"]
    num_amp  = cfg.AGENT_GEOMETRY_PARAMS["num_ampullary_sensors"]
    num_kn   = cfg.AGENT_GEOMETRY_PARAMS["num_knollen_sensors"]
    num_agents = 4

    for label, num_morm_virtual, num_morm_real in [
        ("FracRand", 4 * num_rays, num_rays),
        ("Dynamic",  1 * num_rays, num_rays),
    ]:
        meta = _synthetic_metadata(num_morm_virtual, num_morm_real, num_agents=num_agents)
        p = get_obs_partitions(meta)

        # morm block starts at 0
        assert p["mormyromast"][0]      == 0,                   f"{label}: morm start"
        assert len(p["mormyromast"])    == num_morm_virtual,     f"{label}: morm length"
        assert len(p["mormyromast_self"]) == num_morm_real,      f"{label}: morm_self length"
        assert len(p["mormyromast_cons"]) == num_morm_virtual - num_morm_real, \
            f"{label}: morm_cons length"

        # amp immediately follows morm
        assert p["ampullary"][0]        == num_morm_virtual,     f"{label}: amp start"
        assert len(p["ampullary"])      == num_amp,              f"{label}: amp length"

        # knollen immediately follows amp
        assert p["knollen"][0]          == num_morm_virtual + num_amp, f"{label}: kn start"
        assert len(p["knollen"])        == num_kn * (num_agents - 1),  f"{label}: kn length"

        # no index appears in more than one sensor block
        sensor_indices = np.concatenate([p["mormyromast"], p["ampullary"], p["knollen"]])
        assert len(sensor_indices) == len(set(sensor_indices.tolist())), f"{label}: overlapping indices"

        # obs_dim is a positive int
        assert isinstance(p["obs_dim"], int) and p["obs_dim"] > 0, f"{label}: obs_dim"

        # obs_dim equals the sum of the non-overlapping (top-level) partitions
        # (mormyromast_self/cons are sub-partitions of mormyromast, so excluded)
        top_level = ["mormyromast", "ampullary", "knollen", "knollen_metadata",
                     "actions", "fatigue", "bitten", "agent_size",
                     "bite_cooldown", "feedback_displacement"]
        counted = sum(len(p[k]) for k in top_level if k in p)
        assert counted == p["obs_dim"], \
            f"{label}: obs_dim {p['obs_dim']} != sum of top-level partitions {counted}"

    print("   PASS")


# ---------------------------------------------------------------------------
# 5. get_obs_partitions: no silent fallback on missing key
# ---------------------------------------------------------------------------

def test_obs_partitions_error():
    print("5. obs_partitions_error")

    good_agent_args = {
        "num_mormyromast_sensors_virtual": 36,
        "num_mormyromast_sensors_real":    36,
        "num_ampullary_sensors":           24,
        "num_knollen_sensors":             12,
    }
    all_args = dict(
        feedback_action=True, enable_bite_action=True, use_bite_cooldown=True,
        agent_size_mode=None, feedback_displacement=False, num_agents=4,
    )

    for missing_key in ["num_mormyromast_sensors_virtual", "num_mormyromast_sensors_real",
                        "num_ampullary_sensors", "num_knollen_sensors"]:
        agent_args = {k: v for k, v in good_agent_args.items() if k != missing_key}
        meta = {"all_args": all_args, "agent_args": agent_args}
        try:
            get_obs_partitions(meta)
            assert False, f"should have raised KeyError for missing '{missing_key}'"
        except KeyError:
            pass

    print("   PASS")


# ---------------------------------------------------------------------------
# 6. Mormyromast obs shape integration test
# ---------------------------------------------------------------------------

def test_obs_shape_integration():
    print("6. obs_shape_integration")

    from sensing import SensorModesEpisode
    from electric_scene import ElectricScene
    from sensing_fracrand_test import get_electric_scene_fish_inputs

    num_rays     = cfg.AGENT_GEOMETRY_PARAMS["num_rays"]
    arena        = (100, 100)
    center       = np.array(arena, dtype=float) / 2
    fish_ori     = np.pi / 2
    num_agents   = 2

    base = _base_params()
    env  = _DummyEnv(num_agents=num_agents, arena_size_cm=arena)

    for label, ModelCls, expected_virtual in [
        ("FracRand", FracRandModel,        4 * num_rays),
        ("Dynamic",  DynamicBaselineModel, 1 * num_rays),
    ]:
        model = ModelCls(base, env)
        model.sensor_modes_episode = SensorModesEpisode(mormyromast_mode=1)

        other_pos    = center + np.array([10.0, 0.0])
        fish_pos     = np.stack([center, other_pos])
        fish_ori_arr = np.full(num_agents, fish_ori)
        fish_eods    = np.array([True, False])

        mp, mc, fc, fr, fid = get_electric_scene_fish_inputs(num_agents, fish_eods)
        model.electric_scene.update(
            fish_positions=fish_pos, fish_orientations=fish_ori_arr, fish_eods=fish_eods,
            fish_monopole_positions=mp, fish_monopole_charges=mc,
            fish_dipole_positions=np.zeros((num_agents, 0, 2)),
            fish_dipole_moments=np.zeros((num_agents, 0, 2)),
            conductor_positions=np.zeros((0, 2)),
            conductor_contrasts=np.zeros((0,)),
            conductor_radii=np.zeros((0,)),
            conductor_orientations=np.zeros((0,)),
            fish_intrinsic_dipole_moments_ego=fid,
            conductor_intrinsic_dipole_moments_ego=np.zeros((0, 2)),
            fish_contrasts=fc, fish_radii=fr,
        )
        morm_obs, _, raw_morm = model.sense_mormyromast_all(
            fish_pos, fish_ori_arr, fish_eods, env.np_random,
            conductor_positions=np.zeros((0, 2)),
            conductor_contrasts=np.zeros((0,)),
            conductor_radii=np.zeros((0,)),
            conductor_orientations=np.zeros((0,)),
            conductor_intrinsic_dipole_moments_ego=np.zeros((0, 2)),
        )

        assert morm_obs.shape == (num_agents, expected_virtual), \
            f"{label}: morm_obs shape {morm_obs.shape} != ({num_agents}, {expected_virtual})"
        assert raw_morm.shape == (num_agents, expected_virtual), \
            f"{label}: raw_morm shape {raw_morm.shape} != ({num_agents}, {expected_virtual})"

        # also verify model.agent_electrics stays consistent
        assert model.agent_electrics.num_mormyromast_sensors_virtual == expected_virtual

    print("   PASS")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_cfg_integrity,
        test_sensing_params,
        test_agent_electrics,
        test_obs_partitions_unit,
        test_obs_partitions_error,
        test_obs_shape_integration,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"   FAIL: {e}")
            failed += 1
    print(f"\n{passed}/{passed+failed} passed")
    sys.exit(0 if failed == 0 else 1)
