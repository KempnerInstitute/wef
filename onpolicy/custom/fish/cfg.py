# Configurations/constants
# All distances in centimeters (unless otherwise specified)

import os as _os
import getpass as _getpass
import numpy as np
from features import FEATURE_METADATA, FEATURE_TYPE_COLORMAP, EXCLUDE_FROM_DECODING # for quick backwards compatibility

# Object types -- Backwards compatibility, Robin!
OBJECT_TYPES = {
    "NONE": 0.0,
    "FOOD": 0.33,
    "WALL": 1.0,
    "AGENT": 0.66,
    "AGENT_MIN": 0.5,
    "AGENT_MAX": 0.75,
}

NORMALIZE_REWARDS = False # For testing PPO stability

m_to_cm = 100.0  # Conversion factor from meters to centimeters
cm_to_m = 1 / m_to_cm  # Conversion factor from centimeters to meters

# Fish
FISH_CONSTANTS = {
    "percentile_to_motion_params": {
        # velocities drawn from %ile of real fish velocities "within normal range (-100, 100)"
        # accel drawn from 85th %ile of real fish accelerations
        # all calculated using calculate_real_fish_motion_subset.py
        # 15th, 85th percentile
        85: {
            "min_linear_velocity_m": -0.01,  # m/s
            "min_angular_velocity": -1,  # rad/s  # match to max angular velocity
            "max_linear_velocity_m": 0.25,  # m/s
            "max_angular_velocity": 1,  # rad/s
            "max_linear_acceleration_m": 3,  # m/s^2
            "max_angular_acceleration": 66,  # rad/s^2
        },
        # 10th, 90th percentile
        90: {
            "min_linear_velocity_m": -0.05,  # m/s
            "min_angular_velocity": -1.8,  # rad/s
            "max_linear_velocity_m": 0.35,  # m/s
            "max_angular_velocity": 1.8,  # rad/s
            "max_linear_acceleration_m": 6.5,  # m/s^2
            "max_angular_acceleration": 131,  # rad/s^2
        },
        # 5th, 95th percentile
        95: {
            "min_linear_velocity_m": -0.11,  # m/s
            "min_angular_velocity": -3.5,  # rad/s
            "max_linear_velocity_m": 0.5,  # m/s
            "max_angular_velocity": 3.6,  # rad/s
            "max_linear_acceleration_m": 14.9,  # m/s^2
            "max_angular_acceleration": 318,  # rad/s^2
        },
    },
}


# Electric
ELECTRIC_CONSTANTS = {
    "food_contrast": -0.5,  # Negative contrast for insulators, Chen eq 6
    "fish_contrast": -0.5,
    "k_coulomb": 8.99e9,  # coulomb's constant 8.99 × 10^9 N·m²/C²
    "epsilon_0": 8.854e-12,  # Vacuum permittivity (F/m) 8.854 × 10-12 m^-3 kg^-1 s^4 A^2
    "eps": 1e-25,  # Small number to avoid division by zero in field calculations
    "mVcm_to_Vm": 0.1,  # mV/cm to V/m
    "do_wall_reflection": True,
    "reflection_scale": 0.95,  # Reflection coefficient for electric images
    "flip_on_reflection": False,  # Whether to flip monopole/dipole charges on reflection
    # Since we have non-conducting walls, we don't need to flip charges
    "electric_backend": _os.environ.get("ELECTRIC_BACKEND") or "original",  # "original" | "numba"
}
ELECTRIC_CONSTANTS["Vm_to_mVcm"] = 1 / ELECTRIC_CONSTANTS["mVcm_to_Vm"]

# Environment parameters
ENV_PARAMS = {
    "food_radius_cm": 0.25,  # cm 2025-03-10
    # "food_radius_cm": 0.5,
    # "food_radius_cm": 1.0,
    # "arena_size_max_cm": (80, 80),
    # "arena_size_max_cm": (200, 200),
    "arena_size_min_cm": (40, 40),  # cm
    "arena_size_max_cm": (300, 300),  # cm
    "1p_arena_size_cm": (150, 150),
    "homing_arena_size_min": (75, 75), # cm
    "homing_arena_size_max": (150, 150), # cm
    # "homing_arena_size_min": (70, 70), # cm
    # "homing_arena_size_max": (70, 70), # cm
    "arena_size_bias_alpha": 3,  # Bias towards smaller arenas if a > 1
    # "fps_video": 21,  # ~ 0.25X
    "fps_video": 42,  # ~ 0.25X
    "fps_sim": 83,
    "render_figsize": (8, 16),
    "render_dpi": 72,
    # "food_intrinsic_charge": 0 # 1e-19, # TODO units
    # "food_intrinsic_charge": 1e-22,  # TODO units
    # "food_intrinsic_charge": 0.5e-16,  # Coulombs
    "food_intrinsic_dipole_moment": np.array(
        # [0.8e-18, 0.0]
        # [1.8e-17, 0]  # 2025-02-28
        # [1.11e-24, 0.0]  # 2025-10-03 analytic_frac_rand_shs
        [0, 1.11e-24]  # 2026-01-09 (same as last one but pointing upwards)
    ),  # Units: Coulomb-meter (C*m) (x-directed baseline)
    "1p_d_mean_cm": 12,  # one- arena patch diameter in cm
    # "step_food_density": 0.001,  # used for all patchy arenas (not uniform)
    "step_food_density": 0.0,  # 2025-11-24 (Decided to move to No Renewal)
    "step_food_decay": 0.0,
    "reset_patch_density": 0.001,  # ~4 patches / 60x60cm arena for standard density
    "step_patch_density": 0,
    "max_patch_density": 0.001,  # ~4 patches / 60x60cm arena for standard density
    "stockpile_density": float("inf"),
    "food_drift": 0.0,
    "food_drag": 0.0,
    "uniform_reset_food_density": 0.0045,
    # "uniform_step_food_density": 0.00001,
    "uniform_step_food_density": 0.0, # 2025-11-24 (Decided to move to No Renewal)
    "uniform_max_food_density": 0.005,
    "patchy_reset_food_density": 0.15,
    "patchy_max_food_density": 0.2,
    "feeder_reset_food_density": 0.04,
    "feeder_max_food_density": 0.06,
    "npatch_reset_food_density": 0.15,
    "npatch_max_food_density": 0.2,
}


# Agent parameters
mVcm_to_Vm = ELECTRIC_CONSTANTS["mVcm_to_Vm"]  # shortcut
AGENT_PARAMS = {
    # Sensing ranges
    "amp_food_detection_range_m": 4e-2,  # m
    "amp_fish_detection_range_m": 8e-2,  # m
    "morm_food_detection_range_m": 5e-2,  # m
    "morm_agent_detection_range_m": 10e-2,  # m
    "knollen_agent_detection_range_m": 100e-2,  # m
    # Sensor params

    "num_rays": 36,  # TODO: change to num_mormyromast?
    "num_morm_sets": 1,  # Number of sets of virtual mormyromast sensors (multiple of num_rays)
    "num_knollen_sensors": 12,
    "num_ampullary_sensors": 24,

    # Half the above values
    # "num_rays": 18,  # TODO: change to num_mormyromast?
    # "num_morm_sets": 1,  # Number of sets of virtual mormyromast sensors (multiple of num_rays)
    # "num_knollen_sensors": 6,
    # "num_ampullary_sensors": 12,

    # Spatial params
    "chin_angle": np.pi / 3,
    "eating_angle": np.pi / 4,  # default eating angle when curriculum is disabled; also used as the initial value
    "eating_angle_start": np.pi,  # starting eating angle for curriculum
    "eating_angle_end": np.pi / 4,  # final eating angle for curriculum; should match eating_angle when curriculum completes
    "eating_radius_cm": 2,
    "body_radius_cm": 1.0,
    "biting_radius_cm": 3,
    "expanded_angle": np.pi / 6,
    "falloff_rate": 2.0,
    # Reward/Internal variable params
    "energy_minimum": 0,
    "energy_maximum": 100,
    # "collective_sensing_radius_cm": 15,
    "collective_sensing_radius_cm": 10,  # temp HACK
    "energy_per_step": 0.2,
    "energy_per_eod": 0.1,
    "energy_per_bite": 5,
    "energy_food": 10,
    "percentile_motion": 90,
    "linear_drag_factor": 0.95,
    "angular_drag_factor": 0.95,
    "bite_cooldown_rate": 1/5, # need 1/bite_cooldown_rate timesteps between bites
    "eat_cooldown_rate": 1/3,  # need 1/eat_cooldown_rate timesteps between eating events
    "asym_eating": True,
    # "asym_eating": False,
    "asym_rays": True,  # mormyromasts use different number of rays for chin and rest of the body
    "agent_size_grid_num_options": 6,  # if using grid sampling for agent size, how many options to include between agent_size_min and agent_size_max
}
# AGENT_PARAMS["food_sensing_radius"] = AGENT_PARAMS["body_radius_cm"] + 3 # Needed?
AGENT_PARAMS["num_rays_chin"] = int(AGENT_PARAMS["num_rays"] * 0.3)
AGENT_PARAMS["num_rays_rest"] = AGENT_PARAMS["num_rays"] - AGENT_PARAMS["num_rays_chin"]
AGENT_PARAMS["num_mormyromast_sensors_real"] = AGENT_PARAMS["num_rays"]
AGENT_PARAMS["num_mormyromast_sensors_virtual"] = AGENT_PARAMS["num_rays"] * AGENT_PARAMS["num_morm_sets"]

motion_params = FISH_CONSTANTS["percentile_to_motion_params"][
    AGENT_PARAMS["percentile_motion"]
]
AGENT_PARAMS["max_linear_velocity_baseline"] = (
    motion_params["max_linear_velocity_m"]
    * m_to_cm
    / ENV_PARAMS["fps_sim"]
)  # cm/timestep
AGENT_PARAMS["min_linear_velocity_baseline"] = (
    motion_params["min_linear_velocity_m"]
    * m_to_cm
    / ENV_PARAMS["fps_sim"]
)  # cm/timestep
AGENT_PARAMS["max_linear_acceleration_baseline"] = (
    motion_params["max_linear_acceleration_m"]
    * m_to_cm
    / ENV_PARAMS["fps_sim"] ** 2
)  # cm/timestep^2

motion_params = FISH_CONSTANTS["percentile_to_motion_params"][
    95
]  # OVERRIDE: use 95th percentile for angular velocities
AGENT_PARAMS["max_angular_velocity_baseline"] = (
    motion_params["max_angular_velocity"] / ENV_PARAMS["fps_sim"]
)  # rad/timestep
AGENT_PARAMS["min_angular_velocity_baseline"] = (
    motion_params["min_angular_velocity"] / ENV_PARAMS["fps_sim"]
)  # rad/timestep
AGENT_PARAMS["max_angular_acceleration_baseline"] = (
    motion_params["max_angular_acceleration"] / ENV_PARAMS["fps_sim"] ** 2
)  # rad/timestep^2

AGENT_GEOMETRY_PARAMS = {
    # copy values from AGENT_PARAMS; I intend to remove them from AGENT_PARAMS later
    "num_rays": 36,  # TODO: change to num_mormyromast?
    "num_morm_sets": 1,  # Number of sets of virtual mormyromast sensors (multiple of num_rays)
    "num_knollen_sensors": 12,
    "num_ampullary_sensors": 24,
    "monopole_positions_ego": np.array([[0.5, 0], [-0.5, 0]]),  # w.r.t. agent's center  # NOTE: duplicated for now; wasn't sure where it belongs
    "chin_angle": np.pi / 3,
    "eating_angle": np.pi / 4,  # default eating angle when curriculum is disabled; also used as the initial value
    "eating_angle_start": np.pi,  # starting eating angle for curriculum
    "eating_angle_end": np.pi / 4,  # final eating angle for curriculum; should match eating_angle when curriculum completes
    "eating_radius_cm": 2,
    "body_radius_cm": 1.0,
    "biting_radius_cm": 3,
    "expanded_angle": np.pi / 6,
    "falloff_rate": 2.0,
    "asym_eating": True,
    "asym_rays": True,  # mormyromasts use different number of rays for chin and rest of the body
}
AGENT_GEOMETRY_PARAMS["num_rays_chin"] = int(AGENT_GEOMETRY_PARAMS["num_rays"] * 0.3)
AGENT_GEOMETRY_PARAMS["num_rays_rest"] = AGENT_GEOMETRY_PARAMS["num_rays"] - AGENT_GEOMETRY_PARAMS["num_rays_chin"]
AGENT_GEOMETRY_PARAMS["num_mormyromast_sensors_real"] = AGENT_GEOMETRY_PARAMS["num_rays"]
AGENT_GEOMETRY_PARAMS["num_mormyromast_sensors_virtual"] = AGENT_GEOMETRY_PARAMS["num_rays"] * AGENT_GEOMETRY_PARAMS["num_morm_sets"]

AGENT_ELECTRIC_PARAMS = {
    # copy values from AGENT_PARAMS; I intend to remove them from AGENT_PARAMS later
    "monopole_positions_ego": np.array([[0.5, 0], [-0.5, 0]]),  # w.r.t. agent's center  # NOTE: duplicated for now; wasn't sure where it belongs
    "monopole_charges": np.array([1, -1])
    * 1.11e-15,  # C 2025-08-20 (from Chen et al 2005)
    "fish_intrinsic_dipole_moment": np.array(
        [1.11e-23, 0.0]  # 2025-10-03 analytic_frac_rand_shs
    ),  # Units: Coulomb-meter (C*m) (x-directed baseline)
    "general_sensor_min": 1e-25,
    "ampullary_sensor_min": 2e-9 * mVcm_to_Vm,  # V/m 2025-10-03 analytic_frac_rand_shs
    "ampullary_sensor_max": 2e-7 * mVcm_to_Vm,  # V/m 2025-10-03 analytic_frac_rand_shs
    "knollen_sensor_min": 2e-6 * mVcm_to_Vm,  # V/m 2025-10-03 analytic_frac_rand_shs # empirically verified 2026-01-09
    # "knollen_sensor_min": 7e-9 * mVcm_to_Vm,  # V/m 2025-10-07 calibrate_coll_sense (empirical)
    "knollen_sensor_max": 1
    * mVcm_to_Vm
    * 100,  # V/m 2025-03-10 # Doesn't matter, as knollen is binary
    # TODO: get rid of MAX
}
AGENT_ELECTRIC_PARAMS["knollen_binarize_threshold"] = AGENT_ELECTRIC_PARAMS["knollen_sensor_min"]
# log processing: sign-log transform; 3 orders of magnitude → saturation at ~10 cm (vs detection at ~100 cm)
AGENT_ELECTRIC_PARAMS["knollen_sensor_max_log"] = AGENT_ELECTRIC_PARAMS["knollen_sensor_min"] * 1e3
AGENT_ELECTRIC_PARAMS["max_charge_allowed"] = AGENT_ELECTRIC_PARAMS["monopole_charges"][0]
ENV_PARAMS["max_charge_allowed"] = AGENT_ELECTRIC_PARAMS["monopole_charges"][0]  # C

SENSING_PARAMS_FRAC = {
    "num_morm_sets": 4,
    "morm_max_multiplier": 0.25,  # max mormyromast amplitude is this fraction of baseline
    "morm_min_multiplier": 0.25e-6,  # min mormyromast amplitude is this fraction of baseline
    # TODO remove?
    "mormyromast_sensor_min": 2 * 0.25 * 1e-6 * mVcm_to_Vm,
    "mormyromast_sensor_max": 2 * 0.25 * mVcm_to_Vm,
}
SENSING_PARAMS_DYNAMIC = {
    "num_morm_sets": 1,
    "mormyromast_sensor_min": 2 * 0.25 * 1e-6 * mVcm_to_Vm,
    "mormyromast_sensor_max": 2 * 0.25 * mVcm_to_Vm,
    "mormyromast_sensor_cons_multiplier": 100,  # Scales readings after cons-EOD baseline subtraction
}


# Rewards
REWARDS = {
    "timestep": -0.0,  # Small negative reward each timestep to encourage efficiency
    "eat": 10, # Primary
    "proximity_shaping": 1.0, # Encourage getting closer to food
    "bitten": -5, # Base penalty for being bitten (scaled by size difference when agent_size_mode=True)
    "bite": -0.001, # discourage excessive biting actions
    # "collision": 0,
    "collision": -0.5, # 1/10th biting mean penalty
    "effort_over": -0.1,  # Penalize movement/turning effort over threshold
    "homing": 10,
    "homing_shaping": 0.01,
    "homing_time_penalty": -0.05,
    "wall_proximity": 0,  # Negative reward for coming too close to a wall
}
max_abs = max(abs(v) for v in REWARDS.values() if v != 0)
REWARDS_NORM = {k: v / max_abs for k, v in REWARDS.items()}
if NORMALIZE_REWARDS:
    REWARDS = REWARDS_NORM


# Rendering color configuration
COLORS = {
    "agent_bitten": "red",  # when agent is bitten
    "agent_biting": "black",  # when agent is biting
    "eating_other": "red",  # eating cirle color when agent is biting
    "eating_food": "black",  # eating cirle color when agent is eating
    "agent_eating_radius": "black",  # Eating radius indicator color -- unused?
    "food": "green",  # Food color
    "ray_default": "lightgray",  # Default ray color
    "ray_intersecting_food": "red",  # Ray color when intersecting with food
    "ray_intersecting_agent": "yellow",  # Ray color when intersecting with another agent
    "ray_observing_empty": "black",  # Ray color when observing nothing
    "ray_contrast": "purple",  # Ray color when observing nothing
    "knollen": "black",  # Line color for knollen sensing visualization
    "eating_cone": "cyan",  # Eating cone color
    "ampullary": "orange",  # Line color for ampullary sensing visualization
    "bounded_walk_circ": "blue",  # Bounded walk circumference color
}

SCALABLE_FOOD_PARAMS = [
    "step_food_density",
    "stockpile_density",
    "reset_patch_density",
    "step_patch_density",
    "max_patch_density",
    "uniform_max_food_density",
    "uniform_reset_food_density",
    "uniform_step_food_density",
    # "patchy_max_food_density",  # patches stay same size but num patches scales
    # "patchy_reset_food_density",
    # "feeder_max_food_density",  # same with feeder
    # "feeder_reset_food_density",
]



DOMINANCE_LABEL_CONVERSIONS = {
    "AltB": "A < B",
    "AeqB": "A = B",
    "AgtB": "A > B",
    "Bonly": "B only",
    "Aonly": "A only",
}

MAX_CONDITION_LABEL_LEN = 4

# ── Analysis constants ────────────────────────────────────────────────────────
SIM_FPS = ENV_PARAMS["fps_sim"]          # simulation frames per second (83 Hz)
TIME_STEP_MS = 1000.0 / SIM_FPS          # ms per simulation step (≈12.05 ms at 83 Hz)
EOD_RATE_WINDOW = int(SIM_FPS / 4)       # 20 steps ≈ 240 ms — MA window for eod_rate_*

# Overlay individual points on box/violin plots only when every group has fewer
# than this many observations.  Above this threshold the overlay is skipped to
# avoid overloaded PDFs (e.g. timestep-level power distributions).
PLOT_POINTS_BELOW_N = 50

# Agent colours (Agents 1–4) used in trajectory and analysis plots.
AGENT_COLORS = ["#ABA7EC", "#FEC82E", "#598386", "#C3616F"]

# Semantic colour map for analysis plots.  Keys are reused consistently across
# all analysis scripts so related metrics share the same hue.
SEMANTIC_COLORS = {
    "agent":     AGENT_COLORS[0],   # lavender — agent proximity, social context
    "eod":       AGENT_COLORS[1],   # yellow   — EOD rate (emit/receive)
    "food":      AGENT_COLORS[2],   # teal     — food consumption, eating events
    "dominance": AGENT_COLORS[3],   # pink/red — dominance, win ratio, displacement
}

# Real-fish IDI data (nighttime SPIs, filtered 0.1–1.0 s).
# Update cluster path once confirmed on /n/holylfs06.
REAL_FISH_IDI_CSV = {
    "local": _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                           "real_fish_nighttime_spis_min_0.1_max_1.0.csv"),
    "cluster": f"/n/holylfs06/LABS/krajan_lab/Lab/{_getpass.getuser()}/marl_fish_storage/"
               "real_fish_nighttime_spis_min_0.1_max_1.0.csv",
}
