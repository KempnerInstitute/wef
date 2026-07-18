import matplotlib.cm as cm
import numpy as np

############### FEATURES ###############
# Custom colormap
twilight = cm.get_cmap("twilight", 256)
twilight_rotated = np.roll(
    twilight(np.linspace(0, 1, 256)), 64, axis=0
)  # Roll it by 64/256 so that 0 is no longer grey
twilight_rotated = cm.colors.ListedColormap(twilight_rotated)

# Contains color, type, range metadata for each feature.
# "core": True  — column is NaN-checked by preprocess_features.py.
# "is_event": True — column gets event counters (time_since_last_*) in add_event_counters.
FEATURE_METADATA = {
    # Circular features — all in radians
    "orientation": {
        "color_mode": "circular",
        "feature_type": "circular",
        "unit": "rad",
        "name": "Agent Orientation",
        "short_name": r"$\theta$",
    },
    "knollen_error_angle_nearest": {
        "color_mode": "circular",
        "feature_type": "circular",
        "unit": "rad",
        "name": "Knollen Error Angle",
        "short_name": r"$\alpha^K_{\mathrm{agent}}$",
    },
    "ampullary_error_angle": {
        "color_mode": "circular",
        "feature_type": "circular",
        "unit": "rad",
        "name": "Ampullary Error Angle",
        "short_name": r"$\alpha^A_{\mathrm{agent}}$",
    },
    "mormyromast_error_angle": {
        "color_mode": "circular",
        "feature_type": "circular",
        "unit": "rad",
        "name": "Mormyromast Error Angle",
        "short_name": r"$\alpha^M_{\mathrm{agent}}$",
    },
    "knollen_field_direction_nearest": {
        "color_mode": "circular",
        "feature_type": "circular",
        "unit": "rad",
        "name": "Knollen Field Direction",
        "short_name": r"$\theta^K_{\mathrm{agent}}$",
    },
    "angle_to_closest_food": {
        "color_mode": "circular",
        "cmap": twilight_rotated,
        "feature_type": "circular",
        "unit": "rad",
        "core": True,
        "name": "Angle to Closest Food",
        "short_name": r"$\alpha_{\mathrm{food}}$",
    },
    "angle_to_closest_agent": {
        "color_mode": "circular",
        "feature_type": "circular",
        "unit": "rad",
        "name": "Angle to Closest Agent",
        "short_name": r"$\alpha_{\mathrm{agent}}$",
    },
    "knollen_field_direction": {
        "color_mode": "circular",
        "feature_type": "circular",
        "unit": "rad",
        "name": "Knollenorgan Field Direction",
        "short_name": r"$\theta^K_E$",
    },
    "mormyromast_field_direction": {
        "color_mode": "circular",
        "feature_type": "circular",
        "unit": "rad",
        "name": "Mormyromast Field Direction",
        "short_name": r"$\theta^M_E$",
    },
    "ampullary_field_direction": {
        "color_mode": "circular",
        "feature_type": "circular",
        "unit": "rad",
        "name": "Ampullary Field Direction",
        "short_name": r"$\theta^A_E$",
    },
    # Electric field magnitudes
    "knollen_field_magnitude": {
        "color_mode": "scalar",
        "feature_type": "scalar",
        "unit": "V/m",
        "name": "Knollenorgan Field Magnitude",
        "short_name": r"$|\vec{E}_{\mathrm{K}}|$",
    },
    "mormyromast_field_magnitude": {
        "color_mode": "scalar",
        "feature_type": "scalar",
        "unit": "V/m",
        "name": "Mormyromast Field Magnitude",
        "short_name": r"$|\vec{E}_{\mathrm{M}}|$",
    },
    "ampullary_field_magnitude": {
        "color_mode": "scalar",
        "feature_type": "scalar",
        "unit": "V/m",
        "name": "Ampullary Field Magnitude",
        "short_name": r"$|\vec{E}_{\mathrm{A}}|$",
    },
    "knollen_field_magnitude_log": {
        "color_mode": "scalar",
        "feature_type": "scalar",
        "unit": None,
        "name": "Log Knollenorgan Field Magnitude",
        "short_name": r"$\log |\vec{E}_{\mathrm{K}}|$",
    },
    "mormyromast_field_magnitude_log": {
        "color_mode": "scalar",
        "feature_type": "scalar",
        "unit": None,
        "name": "Log Mormyromast Field Magnitude",
        "short_name": r"$\log |\vec{E}_{\mathrm{M}}|$",
    },
    "ampullary_field_magnitude_log": {
        "color_mode": "scalar",
        "feature_type": "scalar",
        "unit": None,
        "name": "Log Ampullary Field Magnitude",
        "short_name": r"$\log |\vec{E}_{\mathrm{A}}|$",
    },
    # Inverted distance features — all in cm
    "distance_to_closest_food": {
        "color_mode": "inverted",
        "feature_type": "scalar",
        "unit": "cm",
        "core": True,
        "name": "Distance to Closest Food",
        "short_name": r"$d_{\mathrm{food}}$",
    },
    "distance_to_nearest_agent": {
        "color_mode": "inverted",
        "feature_type": "scalar",
        "unit": "cm",
        "core": True,
        "name": "Distance to Nearest Agent",
        "short_name": r"$d_{\mathrm{agent}}$",
    },
    "distance_to_second_nearest_agent": {
        "color_mode": "inverted",
        "feature_type": "scalar",
        "unit": "cm",
        "core": True,
        "name": "Distance to Second Nearest Agent",
        "short_name": r"$d_{\mathrm{agent,2}}$",
    },
    "distance_to_wall": {
        "color_mode": "inverted",
        "feature_type": "scalar",
        "unit": "cm",
        "name": "Distance to Wall",
        "short_name": r"$d_{\mathrm{wall}}$",
    },
    # Inverted time_since features — in simulation timesteps
    "time_since_last_eating_event": {
        "color_mode": "inverted",
        "feature_type": "scalar",
        "unit": "steps",
        "name": "Time Since Last Eating",
        "short_name": r"$t_{\mathrm{eat}}$",
    },
    "time_since_last_emit_eod": {
        "color_mode": "inverted",
        "feature_type": "scalar",
        "unit": "steps",
        "name": "Time Since Last EOD Emission",
        "short_name": r"$t_{\mathrm{EOD}}$",
    },
    "time_since_last_meeting_event": {
        "color_mode": "inverted",
        "feature_type": "scalar",
        "unit": "steps",
        "name": "Time Since Last Agent Encounter",
        "short_name": r"$t_{\mathrm{meet}}$",
    },
    "time_since_last_wall_observed": {
        "color_mode": "inverted",
        "feature_type": "scalar",
        "unit": "steps",
        "name": "Time Since Wall Observed",
        "short_name": r"$t_{\mathrm{wall}}$",
    },
    # EOD rate features — Hz, computed via rolling MA × fps_sim
    "eod_rate_causal": {
        "color_mode": "probability",
        "feature_type": "rate",
        "unit": "Hz",
        "name": "EOD Rate — Causal MA",
        "short_name": r"$r_{\mathrm{EOD}}^{-}$ (Hz)",
    },
    "eod_rate_centered": {
        "color_mode": "probability",
        "feature_type": "rate",
        "unit": "Hz",
        "name": "EOD Rate — Centred MA",
        "short_name": r"$r_{\mathrm{EOD}}^{0}$ (Hz)",
    },
    "eod_rate_predictive": {
        "color_mode": "probability",
        "feature_type": "rate",
        "unit": "Hz",
        "name": "EOD Rate — Predictive MA",
        "short_name": r"$r_{\mathrm{EOD}}^{+}$ (Hz)",
    },
    # Divergent features
    "move_forward": {
        "color_mode": "divergent",
        "feature_type": "scalar",
        "unit": None,
        "core": True,
        "name": "Forward Movement Command",
        "short_name": r"$a_{\parallel}$",
    },
    "turn_angle": {
        "color_mode": "divergent",
        "feature_type": "scalar",
        "unit": None,
        "core": True,
        "name": "Turn Command",
        "short_name": r"$a_\theta$",
    },
    "actual_turn": {
        "color_mode": "divergent",
        "feature_type": "scalar",
        "unit": "rad/step",
        "name": "Actual Turn",
        "short_name": r"$\omega$",
    },
    "linear_velocity": {
        "color_mode": "default",
        "feature_type": "scalar",
        "unit": "cm/step",
        "name": "Linear Velocity",
        "short_name": r"$v$",
    },
    "angular_velocity": {
        "color_mode": "divergent",
        "feature_type": "scalar",
        "unit": "rad/step",
        "name": "Angular Velocity",
        "short_name": r"$\omega_{\mathrm{phys}}$",
    },
    "approach_nearest_velocity": {
        "color_mode": "divergent",
        "vmin": -5.0,
        "vmax": 5.0,
        "feature_type": "scalar",
        "unit": "cm/step",
        "name": "Velocity Toward Nearest Agent",
        "short_name": r"$v_{\mathrm{approach}}$",
    },
    # Other continuous features
    "agent_size": {
        "color_mode": "default",
        "vmin": 0.0,
        "vmax": 1.0,
        "feature_type": "scalar",
        "unit": None,
        "name": "Agent Size",
        "short_name": r"$s_{\mathrm{agent}}$",
    },
    "displacement": {
        "color_mode": "default",
        "vmin": 0.0,
        "vmax": 0.6,
        "feature_type": "scalar",
        "unit": "cm/step",
        "name": "Displacement per Timestep",
        "short_name": r"$\|\Delta r\|$",
    },
    "displacement_x": {
        "feature_type": "scalar",
        "unit": "cm/step",
        "name": "X Displacement per Timestep",
        "short_name": r"$\Delta x$",
    },
    "displacement_y": {
        "feature_type": "scalar",
        "unit": "cm/step",
        "name": "Y Displacement per Timestep",
        "short_name": r"$\Delta y$",
    },
    "position_x": {
        "feature_type": "scalar",
        "unit": "cm",
        "name": "X Position",
        "short_name": r"$x$",
    },
    "position_y": {
        "feature_type": "scalar",
        "unit": "cm",
        "name": "Y Position",
        "short_name": r"$y$",
    },
    "size_of_nearest_agent": {
        "feature_type": "scalar",
        "unit": None,
        "core": True,
        "name": "Size of Nearest Agent",
        "short_name": r"$s_{\mathrm{agent,1}}$",
    },
    "size_of_second_nearest_agent": {
        "feature_type": "scalar",
        "unit": None,
        "core": True,
        "name": "Size of 2nd Nearest Agent",
        "short_name": r"$s_{\mathrm{agent,2}}$",
    },
    # Boolean / event features — dimensionless
    "eating_event": {
        "mode": "boolean",
        "feature_type": "boolean",
        "unit": None,
        "core": True,
        "is_event": True,
        "name": "Eating Event",
        "short_name": r"$I_{\mathrm{eat}}$",
    },
    "meeting_event": {
        "mode": "boolean",
        "feature_type": "boolean",
        "unit": None,
        "is_event": True,
        "name": "Agent Encounter Event (rising edge of has_nearby)",
        "short_name": r"$I_{\mathrm{meet}}$",
    },
    "has_nearby": {
        "mode": "boolean",
        "feature_type": "boolean",
        "unit": None,
        "core": True,
        "name": "Agent Nearby",
        "short_name": r"$I^M_{\mathrm{agent}}$",
    },
    "nearby_emitting": {
        "mode": "boolean",
        "feature_type": "boolean",
        "unit": None,
        "core": True,
        "name": "EOD Emitted by Nearby Agent",
        "short_name": r"$I^M_{\mathrm{EOD\,sensed}}$",
    },
    "was_bitten": {
        "mode": "boolean",
        "feature_type": "boolean",
        "unit": None,
        "name": "Was Bitten",
        "short_name": r"$I_{\mathrm{bitten}}$",
    },
    "bite_other_fish": {
        "mode": "boolean",
        "feature_type": "boolean",
        "unit": None,
        "name": "Bit Another Fish",
        "short_name": r"$I_{\mathrm{bite}}$",
    },
    "bite_action": {
        "feature_type": "boolean",
        "mode": "boolean",
        "unit": None,
        "name": "Bite Action Taken",
        "short_name": r"$I_{\mathrm{bite\_action}}$",
    },
    "emit_eod": {
        "feature_type": "boolean",
        "mode": "boolean",
        "unit": None,
        "core": True,
        "is_event": True,
        "name": "Emitted EOD",
        "short_name": r"$I_{\mathrm{emit\,EOD}}$",
    },
    "has_agent_in_knollen_range": {
        "feature_type": "boolean",
        "mode": "boolean",
        "unit": None,
        "core": True,
        "name": "Agent in Knollen Range",
        "short_name": r"$I^K_{\mathrm{agent}}$",
    },
    "has_food_in_food_sensing_range": {
        "feature_type": "boolean",
        "mode": "boolean",
        "unit": None,
        "core": True,
        "name": "Food in Sensing Range",
        "short_name": r"$I^M_{\mathrm{food}}$",
    },
    # Count features — integer counts
    "food_back_5cm": {
        "feature_type": "count",
        "unit": "count",
        "core": True,
        "name": "Food Count Behind (5cm)",
        "short_name": r"$n_{\mathrm{food,back}}$",
    },
    "food_count_5cm": {
        "feature_type": "count",
        "unit": "count",
        "core": True,
        "name": "Food Count Centered (5cm)",
        "short_name": r"$n^M_{\mathrm{food}}$",
    },
    "food_front_5cm": {
        "feature_type": "count",
        "unit": "count",
        "core": True,
        "name": "Food Count Front (5cm)",
        "short_name": r"$n_{\mathrm{food,front}}$",
    },
    "food_left_5cm": {
        "feature_type": "count",
        "unit": "count",
        "core": True,
        "name": "Food Count Left (5cm)",
        "short_name": r"$n_{\mathrm{food,left}}$",
    },
    "food_right_5cm": {
        "feature_type": "count",
        "unit": "count",
        "core": True,
        "name": "Food Count Right (5cm)",
        "short_name": r"$n_{\mathrm{food,right}}$",
    },
    "num_agents_in_knollen_range": {
        "feature_type": "count",
        "unit": "count",
        "core": True,
        "name": "Agents in Knollen Range",
        "short_name": r"$n^K_{\mathrm{agent}}$",
    },
    # Misc
    "position": {
        "feature_type": "vector",
        "unit": "cm",
        "name": "Position (x,y)",
        "short_name": r"$(x,y)$",
    },
}
FEATURE_METADATA["displacement_magnitude"] = FEATURE_METADATA["displacement"].copy()

# Derived convenience lists
CORE_FEATURES = [k for k, v in FEATURE_METADATA.items() if v.get("core")]
EVENT_FEATURES = [k for k, v in FEATURE_METADATA.items() if v.get("is_event")]

FEATURE_TYPE_COLORMAP = {
    "scalar": "#3b84b8",
    "boolean": "#e27a1f",
    "vector": "#2ca02c",
    "circular": "#c02323",
    "count": "#a16cd2",
    "probability": "#a1665b",
    "unknown": "#7f7f7f",
}

EXCLUDE_FROM_DECODING = [
    "bite_action",
    "move_forward",
    "turn_angle",
    "emit_eod",
    "eating_event",
    "displacement",  # Also an input now
    "mormyromast_field_magnitude",  # Plot log though
    "ampullary_field_magnitude",  # Plot log though
    "knollen_field_magnitude",  # Plot log though
    # analysis_rnn_decoding derives raw linear magnitudes named *_field_mag (not
    # *_field_magnitude); exclude those too so only the *_field_mag_log versions
    # are decoded (raw-linear-magnitude OLS targets are numerically unstable).
    "mormyromast_field_mag",
    "ampullary_field_mag",
    "time_since_last_angle_to_closest_agent_observed",  # Bogus
]
