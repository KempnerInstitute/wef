import sys, os; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy import stats

from electric import fish_forward
from cfg import ELECTRIC_CONSTANTS, AGENT_PARAMS, ENV_PARAMS
from MAEFish import EFishAgent
from sensing import clip_to_sensor_range, log_scale_with_sign
import utils_report as ru
# ### Passive Ampullary sensing of FOOD and other AGENTS/FISH
# Free parameter in cfg: food_intrinsic_dipole_moment
# Free parameter in cfg: fish_intrinsic_dipole_moment

# TODO: Rotate the dipole moment around (and average, or take max?)
def compute_ampullary_dipole_field(intrinsic_dipole_moments_amplitudes, fixed_distance, fixed_angle):
    """
    Computes the sensed electric field at a fixed distance for passive (ampullary) sensing.
    """

    # Create a dummy agent to access sensor positions and parameters
    dummy_agent = EFishAgent(
        num_agents=1,
        arena_size=(200, 200),  # Dummy arena size
        max_step=0,  # Dummy values
        max_turn=0,
        agent_id=0,
    )
    
    c = np.cos(fixed_angle)
    s = np.sin(fixed_angle)
    
    max_abs_sensor_values = []
    for intrinsic_dipole_moments_amplitude in intrinsic_dipole_moments_amplitudes:
        intrinsic_dipole_moments = np.array(
            [[c, s]]
        ) * intrinsic_dipole_moments_amplitude

        # Food with intrinsic charge
        food_pos = np.array(
            [[50 + fixed_distance, 50]])

        sensor_values = fish_forward(
            fish_positions=np.array([[50, 50]]),
            fish_orientations=np.array([0]),
            fish_monopole_positions=np.zeros((1, 0, 2)),
            fish_monopole_charges=np.zeros((1, 0)),
            fish_dipole_positions=np.zeros((1, 0, 2)),
            fish_dipole_moments=np.zeros((1, 0, 2)),
            fish_eods=np.array([0]),  # No EOD for passive sensing
            fish_sensor_positions=dummy_agent.ampullary_positions_ego[None, ...],
            fish_sensor_normals=dummy_agent.ampullary_normals_ego[None, ...],
            fish_sensor_type_mask=None,
            conductor_positions=food_pos,
            conductor_contrasts=np.array([ELECTRIC_CONSTANTS["food_contrast"]]),
            conductor_radii=np.array([ENV_PARAMS.get("food_radius_cm", 1)]),
            arena_size=(200, 200),
            arena_clipping_distance=200,
            do_reflection=False,
            do_induced=False,
            do_intrinsics=True,
            conductor_intrinsic_dipole_moments=intrinsic_dipole_moments,
            measurement_noise=0,
            eps=ELECTRIC_CONSTANTS["eps"],
            units=0.01,
        )  # Return shape: (num_fish, num_sensors)
        max_abs_sensor_values.append(np.max(np.abs(sensor_values)))

    return np.array(max_abs_sensor_values)


def compare_nearby_passive(
    intrinsic_dipole_moment_amplitude,
    fish_position,
    fish_angle,
    object_position,
    object_angle,
    search_distance,
    search_angle,
    distance_steps,
    angle_steps,
    sensor_min,
    sensor_max,
):
    """
    Computes the sensed electric field at a fixed distance for passive (ampullary) sensing.
    """

    # Create a dummy agent to access sensor positions and parameters
    dummy_agent = EFishAgent(
        num_agents=1,
        arena_size=(200, 200),  # Dummy arena size
        max_step=0,  # Dummy values
        max_turn=0,
        agent_id=0,
    )
    
    cf = np.cos(object_angle)
    sf = np.sin(object_angle)
    
    fixed_intrinsic_dipole_moments = np.array(
        [[cf, sf]]
    ) * intrinsic_dipole_moment_amplitude

    # Food with intrinsic charge
    #fixed_food_pos = np.array(
    #    [[50, 50 + fixed_distance]])

    fixed_sensor_values = fish_forward(
        fish_positions=np.array([fish_position]),
        fish_orientations=np.array([fish_angle]),
        fish_monopole_positions=np.zeros((1, 0, 2)),
        fish_monopole_charges=np.zeros((1, 0)),
        fish_dipole_positions=np.zeros((1, 0, 2)),
        fish_dipole_moments=np.zeros((1, 0, 2)),
        fish_eods=np.array([0]),  # No EOD for passive sensing
        fish_sensor_positions=dummy_agent.ampullary_positions_ego[None, ...],
        fish_sensor_normals=dummy_agent.ampullary_normals_ego[None, ...],
        fish_sensor_type_mask=None,
        conductor_positions=np.array([object_position]),
        conductor_contrasts=np.array([ELECTRIC_CONSTANTS["food_contrast"]]),
        conductor_radii=np.array([ENV_PARAMS.get("food_radius_cm", 1)]),
        arena_size=(200, 200),
        arena_clipping_distance=200,
        do_reflection=False,
        do_induced=False,
        do_intrinsics=True,
        conductor_intrinsic_dipole_moments=fixed_intrinsic_dipole_moments,
        measurement_noise=0,
        eps=ELECTRIC_CONSTANTS["eps"],
        units=0.01,
    )  # Return shape: (num_fish, num_sensors)
    
    #print('dot product', fixed_sensor_values)
    fixed_sensor_values = clip_to_sensor_range(
        fixed_sensor_values, sensor_min, sensor_max)
    clamped_sensor_values = fixed_sensor_values
    #print('clamped', fixed_sensor_values)
    #fixed_sensor_values = log_scale_with_sign(
    #    fixed_sensor_values, dummy_agent.general_sensor_min)
    #print('log', fixed_sensor_values)
    #print('min', dummy_agent.general_sensor_min)
    
    num_sensors = fixed_sensor_values.shape[-1]
    varied_food_positions = np.zeros(
        (distance_steps, distance_steps, 2))
    varied_food_angles = np.zeros((angle_steps,))
    varied_sensor_values = np.zeros(
        (distance_steps, distance_steps, angle_steps, num_sensors))
    for i, x in enumerate(
        np.linspace(-search_distance, search_distance, distance_steps
    )):
        for j, y in enumerate(
            np.linspace(-search_distance, search_distance, distance_steps
        )):
            varied_position = object_position + np.array([[x,y]])
            for k, theta in enumerate(
                np.linspace(-search_angle, search_angle, angle_steps
            )):
                #print(theta, varied_position)
                varied_angle = object_angle + theta
                #varied_food_positions.append(varied_position)
                #varied_food_angles.append(varied_angle)
                varied_food_positions[i,j] = varied_position
                varied_food_angles[k] = varied_angle
                
                cv = np.cos(varied_angle)
                sv = np.sin(varied_angle)
                varied_intrinsic_dipole_moment = np.array(
                    [[cv, sv]]
                ) * intrinsic_dipole_moment_amplitude
                
                sensor_values = fish_forward(
                    fish_positions=np.array([[50, 50]]),
                    fish_orientations=np.array([0]),
                    fish_monopole_positions=np.zeros((1, 0, 2)),
                    fish_monopole_charges=np.zeros((1, 0)),
                    fish_dipole_positions=np.zeros((1, 0, 2)),
                    fish_dipole_moments=np.zeros((1, 0, 2)),
                    fish_eods=np.array([0]),  # No EOD for passive sensing
                    fish_sensor_positions=
                        dummy_agent.ampullary_positions_ego[None, ...],
                    fish_sensor_normals=
                        dummy_agent.ampullary_normals_ego[None, ...],
                    fish_sensor_type_mask=None,
                    conductor_positions=varied_position,
                    conductor_contrasts=
                        np.array([ELECTRIC_CONSTANTS["food_contrast"]]),
                    conductor_radii=
                        np.array([ENV_PARAMS.get("food_radius_cm", 1)]),
                    arena_size=(200, 200),
                    arena_clipping_distance=200,
                    do_reflection=False,
                    do_induced=False,
                    do_intrinsics=True,
                    conductor_intrinsic_dipole_moments=
                        varied_intrinsic_dipole_moment,
                    measurement_noise=0,
                    eps=ELECTRIC_CONSTANTS["eps"],
                    units=0.01,
                )  # Return shape: (num_fish, num_sensors)
                sensor_values = clip_to_sensor_range(
                    sensor_values, sensor_min, sensor_max)
                #sensor_values = log_scale_with_sign(
                #    sensor_values, dummy_agent.general_sensor_min)
                
                #varied_sensor_values.append(sensor_values)
                varied_sensor_values[i,j,k] = sensor_values[0]
    
    return (
        fixed_sensor_values,
        clamped_sensor_values,
        varied_food_positions,
        varied_food_angles,
        varied_sensor_values,
        dummy_agent.ampullary_positions_ego,
        dummy_agent.ampullary_normals_ego,
    )

### Active/Mormyromast sensing of food
def compute_active_field(monopole_charges, fixed_distance):
    """
    Computes the sensed electric field at a fixed distance for active (mormyromast) sensing.
    """
    # Create a dummy agent to access sensor positions and parameters
    dummy_agent = EFishAgent(
        num_agents=1,
        arena_size=(200, 200),  # Dummy arena size
        max_step=0,  # Dummy values
        max_turn=0,
        agent_id=0,
    )

    max_sensor_values = []
    for monopole_charge in monopole_charges:
        # Setup monopoles (EOD-based active sensing)
        monopole_positions = AGENT_PARAMS["monopole_positions_ego"]
        monopole_charges_array = np.array(
            [monopole_charge, -monopole_charge]
        )  # Ensure charge balance

        dummy_agent.monopole_charges = monopole_charges_array
        dummy_agent.reset() # Recalculate CD

        # Food (conductors) setup
        food_pos = np.array([[50 + fixed_distance, 50]])
        food_contrast = np.array([ELECTRIC_CONSTANTS["food_contrast"]])
        food_radius = np.array([ENV_PARAMS.get("food_radius_cm", 1)])

        field = fish_forward(
            fish_positions=np.array([[50, 50]]),
            fish_orientations=np.array([0]),
            fish_monopole_positions=monopole_positions[
                None, ...
            ],  # Shape: (1, num_monopoles, 2)
            fish_monopole_charges=monopole_charges_array[
                None, ...
            ],  # Shape: (1, num_monopoles)
            fish_dipole_positions=np.zeros((1, 0, 2)),  # No dipoles
            fish_dipole_moments=np.zeros((1, 0, 2)),
            fish_eods=np.array([1]),  # EOD is ON for active sensing
            fish_sensor_positions=dummy_agent.mormyromast_positions_ego[None, ...],
            fish_sensor_normals=dummy_agent.mormyromast_positions_ego[None, ...],
            fish_sensor_type_mask=None,
            conductor_positions=food_pos,
            conductor_contrasts=food_contrast,
            conductor_radii=food_radius,
            arena_size=(200, 200),
            arena_clipping_distance=200,
            do_reflection=True,
            do_induced=True,  # Induced dipole effect is crucial for active sensing
            do_intrinsics=False,  # No intrinsic charge for active sensing
            measurement_noise=0,
            eps=ELECTRIC_CONSTANTS["eps"],
            units=0.01,
        )  # Return shape: (num_fish, num_sensors)

        # Subtract out CD
        field -= dummy_agent.mormyromast_cd

        # Save max amplitude to list
        max_sensor_values.append(np.max(np.abs(field)))  # Get max amplitude sensor value

    return np.array(max_sensor_values)

# # ### Knollen
def compute_knollen_field(monopole_charges, knollen_distance):
    """
    Computes max-knollen-organ readings for a receiving agent 100cm away from an EOD-emitting conspecific.
    """

    # Create two dummy agents to access sensor positions and parameters
    dummy_sender = EFishAgent(
        num_agents=1,
        arena_size=(200, 200),  # Dummy arena size
        max_step=0,  # Dummy values
        max_turn=0,
        agent_id=0,
    )

    dummy_receiver = EFishAgent(
        num_agents=1,
        arena_size=(200, 200),  # Dummy arena size
        max_step=0,  # Dummy values
        max_turn=0,
        agent_id=1,
    )

    max_knollen_values = []
    for monopole_charge in monopole_charges:
        # Setup monopoles for knollen sensing (EOD-emitting conspecific at 100cm)
        monopole_positions = AGENT_PARAMS["monopole_positions_ego"]
        monopole_charges_array = np.array(
            [monopole_charge, -monopole_charge]
        )  # Ensure charge balance

        field_knollen = fish_forward(
            fish_positions=np.array([[50, 50], [50 + knollen_distance, 50]]),
            fish_orientations=np.array([0, 0]),
            fish_monopole_positions=monopole_positions[None, ...],
            fish_monopole_charges=monopole_charges_array[None, ...],
            fish_dipole_positions=np.zeros((1, 0, 2)),
            fish_dipole_moments=np.zeros((1, 0, 2)),
            fish_eods=np.array([1, 0]),  # Only sender emits EOD
            fish_sensor_positions=dummy_receiver.knollen_positions_ego[None, ...],
            fish_sensor_normals=dummy_receiver.knollen_positions_ego[None, ...],
            fish_sensor_type_mask=None,
            conductor_positions=np.zeros((0, 2)),
            conductor_contrasts=np.array([]),
            conductor_radii=np.array([]),
            arena_size=(200, 200),
            arena_clipping_distance=200,
            do_reflection=False,
            do_induced=False,
            do_intrinsics=False,
            measurement_noise=0,
            eps=ELECTRIC_CONSTANTS["eps"],
            units=0.01,
        )  # Return shape: (num_fish, num_sensors)
        # max_knollen_values.append(np.max(np.abs(field_knollen)))
        max_knollen_values.append(
            np.max(np.abs(field_knollen[1]))
        )  # Only analyze the receiving fish

    return np.array(max_knollen_values)


# ## Active/Mormyromast sensing of food
# ### Calibrate "Fish Induced Dipole" in active mormyromast sensing (TODO: Pull from SJY)
def compute_active_field_over_distances(distances):
    """
    Computes the sensed electric field at a fixed distance for active (mormyromast) sensing.
    """
    max_sensor_values = []
    for distance in distances:
        # Setup monopoles (EOD-based active sensing)
        monopole_positions = AGENT_PARAMS["monopole_positions_ego"]
        monopole_charges_array = np.array([1, -1]) * 0.5e-12  # Ensure charge balance

        # Fish conductor setup
        fish_pos = np.array([[50 + distance, 50]])
        fish_contrast = np.array([ELECTRIC_CONSTANTS["fish_contrast"]])
        # fish_radius = np.array([AGENT_PARAMS.get("body_radius", 1)])
        fish_radius = np.array([7])

        field = fish_forward(
            fish_positions=np.array([[50, 50]]),
            fish_orientations=np.array([0]),
            fish_monopole_positions=monopole_positions[
                None, ...
            ],  # Shape: (1, num_monopoles, 2)
            fish_monopole_charges=monopole_charges_array[
                None, ...
            ],  # Shape: (1, num_monopoles)
            fish_dipole_positions=np.zeros((1, 0, 2)),  # No dipoles
            fish_dipole_moments=np.zeros((1, 0, 2)),
            fish_eods=np.array([1]),  # EOD is ON for active sensing
            fish_sensor_positions=dummy_agent.mormyromast_positions_ego[None, ...],
            fish_sensor_normals=dummy_agent.mormyromast_positions_ego[None, ...],
            fish_sensor_baselines=None,
            fish_sensor_thresholds=None,
            fish_sensor_type_mask=None,
            conductor_positions=fish_pos,
            conductor_contrasts=fish_contrast,
            conductor_radii=fish_radius,
            arena_size=(200, 200),
            arena_clipping_distance=200,
            do_reflection=True,
            do_induced=True,  # Induced dipole effect is crucial for active sensing
            do_intrinsics=False,  # No intrinsic charge for active sensing
            conductor_intrinsic_charge=0,  # Not used since do_intrinsics=False
            measurement_noise=0,
            eps=ELECTRIC_CONSTANTS["eps"],
            units=0.01,
        )  # Return shape: (num_fish, num_sensors)

        field -= dummy_sender.mormyromast_cd
        max_sensor_values.append(np.max(np.abs(field)))  # Get max sensor value

    return np.array(max_sensor_values)


# for operation in ['max']:  # ['max', 'mean', 'min']
#             plot_dist_to_objects_vs_sensors_eod_conditioned_aggregate(dff, save_path=outfile_base + f'_dist_to_objects_vs_sensors_{operation}.png', operation=operation)
def plot_dist_to_objects_vs_sensors_eod_conditioned_aggregate(
    dff,
    save_path=None,
    agent_distance_threshold=30,
    operation='max',
):
    """
    Produces:
      (A) EOD-Conditioned Subplots:
          - Food: 2×2 (top row = EOD=True, bottom row = EOD=False)
          - Agents: 2×3 (top row = EOD=True, bottom row = EOD=False)

      (B) Aggregate (EOD-agnostic) Plots:
          - Food: 1×2 row (Mormyromast, Ampullary)
          - Agents: 1×3 row (Mormyromast, Knollen, Ampullary)

      (C) Self+Cons EOD Conditioned Plots:
          - Food: 4×2 (rows = 4 combos of (self_eod, cons_eod), cols = Mormyromast/Ampullary)
          - Agents: 4×3 (rows = 4 combos, cols = Mormyromast/Knollen/Ampullary)

      The variable `cons_eod` = True if ANY OTHER agent in the same group/time-step
      has `emit_eod=True`. Achieved via a per-group distance-matrix approach for speed.
    """

    # ----------------------------------------------------------------
    # 1. Setup: sensor bounds & sensor extraction
    # ----------------------------------------------------------------
    sensor_bounds = ru.get_sensor_bounds(dff)
    log_scaled_sensor_bounds = {
        k: (log_scale_with_sign(v, sensor_bounds['general_sensor_min']) if v is not None else None)
        for k, v in sensor_bounds.items()
    }

    mormyromast_indices, ampullary_indices, knollen_indices = ru.get_sensor_indices(dff)

    # Define aggregation for sensor data
    def max_abs(x):
        return np.max(np.abs(x))
    def mean_abs(x):
        return np.mean(np.abs(x))
    def min_abs(x):
        return np.min(np.abs(x))

    if operation == 'max':
        sensor_fn = max_abs
    elif operation == 'mean':
        sensor_fn = mean_abs
    elif operation == 'min':
        sensor_fn = min_abs
    else:
        raise ValueError(f"Unknown operation: {operation}")

    # ----------------------------------------------------
    # 2A. Containers: EOD=TRUE, EOD=FALSE, plus AGGREGATE
    # ----------------------------------------------------

    # --- Food ---
    food_distances_true = []
    mormyromast_food_true = []
    ampullary_food_true = []

    food_distances_false = []
    mormyromast_food_false = []
    ampullary_food_false = []

    food_distances_all = []
    mormyromast_food_all = []
    ampullary_food_all = []

    # --- Agents ---
    agent_distances_true = []
    mormyromast_agent_true = []
    knollen_agent_true = []
    ampullary_agent_true = []

    agent_distances_false = []
    mormyromast_agent_false = []
    knollen_agent_false = []
    ampullary_agent_false = []

    agent_distances_all = []
    mormyromast_agent_all = []
    knollen_agent_all = []
    ampullary_agent_all = []

    # ----------------------------------------------------------------
    # 2B. NEW Containers: self+cons EOD combos
    #
    # We'll store them in dicts keyed by (self_eod, cons_eod),
    # each entry containing sub-lists for distance, mormyromast, knollen, ampullary.
    #
    # Food sensors only need: dist, mormyromast, ampullary
    # Agent sensors need: dist, mormyromast, knollen, ampullary
    # ----------------------------------------------------------------
    # All 4 combos:
    combos = [(False,False), (True,False), (False,True), (True,True)]

    # Food data: each (self_e, cons_e) => {'dist':[], 'mormy':[], 'amp':[]}
    food_data = {
        c: {'dist': [], 'mormy': [], 'amp': []}
        for c in combos
    }

    # Agent data: each (self_e, cons_e) => {'dist':[], 'mormy':[], 'knol':[], 'amp':[]}
    agent_data = {
        c: {'dist': [], 'mormy': [], 'knol': [], 'amp': []}
        for c in combos
    }

    # ----------------------------------------------------------------
    # 3. Speedup via groupby + distance matrix
    # ----------------------------------------------------------------
    grouped = dff.groupby(['env_id', 'episode_index', 'time_step'], sort=False)

    for (env_id, ep_idx, t), group in grouped:
        # Convert positions to a NumPy array (N, 2)
        positions = np.vstack(group['position'])

        # Build the distance matrix (N×N)
        dist_matrix = cdist(positions, positions)
        np.fill_diagonal(dist_matrix, np.inf)  # Exclude self-distance

        # Check how many EODs in total => needed for cons_eod
        total_eods = group['emit_eod'].sum()

        # Find agent_0 for food
        agent_0_row = group[group['agent_id'] == 0].head(1)
        if not agent_0_row.empty:
            food_positions = agent_0_row.iloc[0]['food_positions']
        else:
            food_positions = []

        # Precompute distances to food for each agent
        food_positions = np.array(food_positions)
        if food_positions.size > 0:
            food_dist_matrix = cdist(positions, food_positions)
            # nearest food distance for each agent row_i
            nearest_food_dists = food_dist_matrix.min(axis=1)
        else:
            nearest_food_dists = None

        # Iterate over each row in this group
        for row_i, (df_index, row_data) in enumerate(group.iterrows()):
            # (a) Nearest agent distance
            nearest_agent_distance = dist_matrix[row_i].min()

            # (b) self_eod
            self_eod = bool(row_data['emit_eod'])

            # (c) cons_eod = True if any OTHER agent has emit_eod=True
            #     We can do this by subtracting 1 if self_eod is True
            #     then checking if >0
            other_eods = total_eods - (1 if self_eod else 0)
            cons_eod = (other_eods > 0)

            # For convenience, define a tuple to index the new dict
            combo_key = (self_eod, cons_eod)

            # (d) Food logic (only if agent is beyond threshold from another agent)
            if (nearest_food_dists is not None) and (nearest_agent_distance >= agent_distance_threshold):
                nf_dist = nearest_food_dists[row_i]

                # Aggregate
                food_distances_all.append(nf_dist)
                mormyromast_food_all.append(sensor_fn(row_data['observations'][mormyromast_indices]))
                ampullary_food_all.append(sensor_fn(row_data['observations'][ampullary_indices]))

                # EOD-conditioned
                if self_eod:
                    food_distances_true.append(nf_dist)
                    mormyromast_food_true.append(sensor_fn(row_data['observations'][mormyromast_indices]))
                    ampullary_food_true.append(sensor_fn(row_data['observations'][ampullary_indices]))
                else:
                    food_distances_false.append(nf_dist)
                    mormyromast_food_false.append(sensor_fn(row_data['observations'][mormyromast_indices]))
                    ampullary_food_false.append(sensor_fn(row_data['observations'][ampullary_indices]))

                # Self+Cons EOD Conditioned
                food_data[combo_key]['dist'].append(nf_dist)
                food_data[combo_key]['mormy'].append(
                    sensor_fn(row_data['observations'][mormyromast_indices])
                )
                food_data[combo_key]['amp'].append(
                    sensor_fn(row_data['observations'][ampullary_indices])
                )

            # (e) Agent logic (always included if there's another agent)
            if np.isfinite(nearest_agent_distance):
                # Aggregate
                agent_distances_all.append(nearest_agent_distance)
                mormyromast_agent_all.append(sensor_fn(row_data['observations'][mormyromast_indices]))
                knollen_agent_all.append(sensor_fn(row_data['observations'][knollen_indices]))
                ampullary_agent_all.append(sensor_fn(row_data['observations'][ampullary_indices]))

                # EOD-conditioned
                if self_eod:
                    agent_distances_true.append(nearest_agent_distance)
                    mormyromast_agent_true.append(sensor_fn(row_data['observations'][mormyromast_indices]))
                    knollen_agent_true.append(sensor_fn(row_data['observations'][knollen_indices]))
                    ampullary_agent_true.append(sensor_fn(row_data['observations'][ampullary_indices]))
                else:
                    agent_distances_false.append(nearest_agent_distance)
                    mormyromast_agent_false.append(sensor_fn(row_data['observations'][mormyromast_indices]))
                    knollen_agent_false.append(sensor_fn(row_data['observations'][knollen_indices]))
                    ampullary_agent_false.append(sensor_fn(row_data['observations'][ampullary_indices]))

                # Self+Cons EOD Conditioned
                agent_data[combo_key]['dist'].append(nearest_agent_distance)
                agent_data[combo_key]['mormy'].append(
                    sensor_fn(row_data['observations'][mormyromast_indices])
                )
                agent_data[combo_key]['knol'].append(
                    sensor_fn(row_data['observations'][knollen_indices])
                )
                agent_data[combo_key]['amp'].append(
                    sensor_fn(row_data['observations'][ampullary_indices])
                )

    # ----------------------------------------------------------------
    # 4. Plotting Helpers
    # ----------------------------------------------------------------
    def plot_scatter_and_regression(
        x, y,
        title,
        x_label,
        y_label,
        sensor_min=None,
        sensor_max=None,
        binarization_threshold=None,
        subplot=None
    ):
        """Plot a scatter + regression with correlation in the title."""
        if subplot:
            plt.subplot(*subplot)
        sns.regplot(x=x, y=y, scatter_kws={'alpha': 0.1}, line_kws={'color': 'red'})

        # Optional sensor bounds
        if sensor_min is not None:
            plt.axhline(sensor_min, color='orange', linestyle='--', alpha=0.5, label='Sensor Min')
        if sensor_max is not None:
            plt.axhline(sensor_max, color='orange', linestyle='--', alpha=0.5, label='Sensor Max')
        if binarization_threshold is not None:
            plt.axhline(
                binarization_threshold, color='red', linestyle='--', alpha=0.5,
                label='Binarization Threshold'
            )

        # correlation
        try:
            corr, p_val = stats.pearsonr(x, y)
            plt.title(f"{title}\nr={corr:.3f}, p={p_val:.3e}")
        except Exception as exc:
            print(f"Correlation error: {exc}")

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid()

    # ----------------------------------------------------------------
    # (A) EOD-CONDITIONED SUBPLOTS
    # ----------------------------------------------------------------

    #  A.1: Food Distances => 2x2
    plt.figure(figsize=(10, 8))

    # Row 1, Col 1 => EOD=True, Mormyromast
    plot_scatter_and_regression(
        x=food_distances_true,
        y=mormyromast_food_true,
        title="Food vs. Mormyromast (EOD=True)",
        x_label="Distance to Food (cm)",
        y_label=f"{operation.capitalize()} Abs Sensor",
        sensor_min=log_scaled_sensor_bounds["mormyromast_sensor_min"],
        sensor_max=log_scaled_sensor_bounds["mormyromast_sensor_max"],
        subplot=(2, 2, 1)
    )

    # Row 1, Col 2 => EOD=True, Ampullary
    plot_scatter_and_regression(
        x=food_distances_true,
        y=ampullary_food_true,
        title="Food vs. Ampullary (EOD=True)",
        x_label="Distance to Food (cm)",
        y_label=f"{operation.capitalize()} Abs Sensor",
        sensor_min=log_scaled_sensor_bounds["ampullary_sensor_min"],
        sensor_max=log_scaled_sensor_bounds["ampullary_sensor_max"],
        subplot=(2, 2, 2)
    )

    # Row 2, Col 1 => EOD=False, Mormyromast
    plot_scatter_and_regression(
        x=food_distances_false,
        y=mormyromast_food_false,
        title="Food vs. Mormyromast (EOD=False)",
        x_label="Distance to Food (cm)",
        y_label=f"{operation.capitalize()} Abs Sensor",
        sensor_min=log_scaled_sensor_bounds["mormyromast_sensor_min"],
        sensor_max=log_scaled_sensor_bounds["mormyromast_sensor_max"],
        subplot=(2, 2, 3)
    )

    # Row 2, Col 2 => EOD=False, Ampullary
    plot_scatter_and_regression(
        x=food_distances_false,
        y=ampullary_food_false,
        title="Food vs. Ampullary (EOD=False)",
        x_label="Distance to Food (cm)",
        y_label=f"{operation.capitalize()} Abs Sensor",
        sensor_min=log_scaled_sensor_bounds["ampullary_sensor_min"],
        sensor_max=log_scaled_sensor_bounds["ampullary_sensor_max"],
        subplot=(2, 2, 4)
    )

    plt.tight_layout()
    if save_path:
        conditioned_food_path = save_path.split(".png")[0] + "_food_eod_conditioned.png"
        plt.savefig(conditioned_food_path, dpi=300)

    #  A.2: Agent Distances => 2x3
    plt.figure(figsize=(15, 10))

    # Row 1, Col 1 => EOD=True, Mormyromast
    plot_scatter_and_regression(
        x=agent_distances_true,
        y=mormyromast_agent_true,
        title="Agent vs. Mormyromast (EOD=True)",
        x_label="Distance to Agent (cm)",
        y_label=f"{operation.capitalize()} Abs Sensor",
        sensor_min=log_scaled_sensor_bounds["mormyromast_sensor_min"],
        sensor_max=log_scaled_sensor_bounds["mormyromast_sensor_max"],
        subplot=(2, 3, 1)
    )

    # Row 1, Col 2 => EOD=True, Knollen
    plot_scatter_and_regression(
        x=agent_distances_true,
        y=knollen_agent_true,
        title="Agent vs. Knollen (EOD=True)",
        x_label="Distance to Agent (cm)",
        y_label=f"{operation.capitalize()} Abs Sensor",
        sensor_min=log_scaled_sensor_bounds["knollen_sensor_min"],
        sensor_max=log_scaled_sensor_bounds["knollen_sensor_max"],
        binarization_threshold=log_scaled_sensor_bounds["knollen_binarize_threshold"],
        subplot=(2, 3, 2)
    )

    # Row 1, Col 3 => EOD=True, Ampullary
    plot_scatter_and_regression(
        x=agent_distances_true,
        y=ampullary_agent_true,
        title="Agent vs. Ampullary (EOD=True)",
        x_label="Distance to Agent (cm)",
        y_label=f"{operation.capitalize()} Abs Sensor",
        sensor_min=log_scaled_sensor_bounds["ampullary_sensor_min"],
        sensor_max=log_scaled_sensor_bounds["ampullary_sensor_max"],
        subplot=(2, 3, 3)
    )

    # Row 2, Col 1 => EOD=False, Mormyromast
    plot_scatter_and_regression(
        x=agent_distances_false,
        y=mormyromast_agent_false,
        title="Agent vs. Mormyromast (EOD=False)",
        x_label="Distance to Agent (cm)",
        y_label=f"{operation.capitalize()} Abs Sensor",
        sensor_min=log_scaled_sensor_bounds["mormyromast_sensor_min"],
        sensor_max=log_scaled_sensor_bounds["mormyromast_sensor_max"],
        subplot=(2, 3, 4)
    )

    # Row 2, Col 2 => EOD=False, Knollen
    plot_scatter_and_regression(
        x=agent_distances_false,
        y=knollen_agent_false,
        title="Agent vs. Knollen (EOD=False)",
        x_label="Distance to Agent (cm)",
        y_label=f"{operation.capitalize()} Abs Sensor",
        sensor_min=log_scaled_sensor_bounds["knollen_sensor_min"],
        sensor_max=log_scaled_sensor_bounds["knollen_sensor_max"],
        binarization_threshold=log_scaled_sensor_bounds["knollen_binarize_threshold"],
        subplot=(2, 3, 5)
    )

    # Row 2, Col 3 => EOD=False, Ampullary
    plot_scatter_and_regression(
        x=agent_distances_false,
        y=ampullary_agent_false,
        title="Agent vs. Ampullary (EOD=False)",
        x_label="Distance to Agent (cm)",
        y_label=f"{operation.capitalize()} Abs Sensor",
        sensor_min=log_scaled_sensor_bounds["ampullary_sensor_min"],
        sensor_max=log_scaled_sensor_bounds["ampullary_sensor_max"],
        subplot=(2, 3, 6)
    )

    plt.tight_layout()
    if save_path:
        conditioned_agent_path = save_path.split(".png")[0] + "_agent_eod_conditioned.png"
        plt.savefig(conditioned_agent_path, dpi=300)


    # ----------------------------------------------------------------
    # (B) AGGREGATE (EOD-agnostic) PLOTS
    # ----------------------------------------------------------------
    #  B.1: Food Distances => 1×2
    plt.figure(figsize=(10, 5))

    # Col 1 => Mormyromast (All EOD)
    plot_scatter_and_regression(
        x=food_distances_all,
        y=mormyromast_food_all,
        title="Food vs. Mormyromast (All EOD)",
        x_label="Distance to Food (cm)",
        y_label=f"{operation.capitalize()} Abs Sensor",
        sensor_min=log_scaled_sensor_bounds["mormyromast_sensor_min"],
        sensor_max=log_scaled_sensor_bounds["mormyromast_sensor_max"],
        subplot=(1, 2, 1)
    )

    # Col 2 => Ampullary (All EOD)
    plot_scatter_and_regression(
        x=food_distances_all,
        y=ampullary_food_all,
        title="Food vs. Ampullary (All EOD)",
        x_label="Distance to Food (cm)",
        y_label=f"{operation.capitalize()} Abs Sensor",
        sensor_min=log_scaled_sensor_bounds["ampullary_sensor_min"],
        sensor_max=log_scaled_sensor_bounds["ampullary_sensor_max"],
        subplot=(1, 2, 2)
    )

    plt.tight_layout()
    if save_path:
        food_agg_path = save_path.split(".png")[0] + "_food_aggregate.png"
        plt.savefig(food_agg_path, dpi=300)


    #  B.2: Agent Distances => 1×3
    plt.figure(figsize=(15, 5))

    # Col 1 => Mormyromast (All EOD)
    plot_scatter_and_regression(
        x=agent_distances_all,
        y=mormyromast_agent_all,
        title="Agent vs. Mormyromast (All EOD)",
        x_label="Distance to Agent (cm)",
        y_label=f"{operation.capitalize()} Abs Sensor",
        sensor_min=log_scaled_sensor_bounds["mormyromast_sensor_min"],
        sensor_max=log_scaled_sensor_bounds["mormyromast_sensor_max"],
        subplot=(1, 3, 1)
    )

    # Col 2 => Knollen (All EOD)
    plot_scatter_and_regression(
        x=agent_distances_all,
        y=knollen_agent_all,
        title="Agent vs. Knollen (All EOD)",
        x_label="Distance to Agent (cm)",
        y_label=f"{operation.capitalize()} Abs Sensor",
        sensor_min=log_scaled_sensor_bounds["knollen_sensor_min"],
        sensor_max=log_scaled_sensor_bounds["knollen_sensor_max"],
        binarization_threshold=log_scaled_sensor_bounds["knollen_binarize_threshold"],
        subplot=(1, 3, 2)
    )

    # Col 3 => Ampullary (All EOD)
    plot_scatter_and_regression(
        x=agent_distances_all,
        y=ampullary_agent_all,
        title="Agent vs. Ampullary (All EOD)",
        x_label="Distance to Agent (cm)",
        y_label=f"{operation.capitalize()} Abs Sensor",
        sensor_min=log_scaled_sensor_bounds["ampullary_sensor_min"],
        sensor_max=log_scaled_sensor_bounds["ampullary_sensor_max"],
        subplot=(1, 3, 3)
    )

    plt.tight_layout()
    if save_path:
        agent_agg_path = save_path.split(".png")[0] + "_agent_aggregate.png"
        plt.savefig(agent_agg_path, dpi=300)


    # ----------------------------------------------------------------
    # (C) NEW: Self+Cons EOD Conditioned
    # ----------------------------------------------------------------
    # We'll produce two new figures:
    #   - Food => 4×2
    #   - Agents => 4×3
    #
    # The 4 rows correspond to:
    #   Row 1: self_eod=False, cons_eod=False
    #   Row 2: self_eod=True,  cons_eod=False
    #   Row 3: self_eod=False, cons_eod=True
    #   Row 4: self_eod=True,  cons_eod=True
    #
    # The columns are the sensor types (2 for Food, 3 for Agents).
    # We'll match them with the same dictionaries we set up.
    # ----------------------------------------------------------------

    combos = [(False,False), (True,False), (False,True), (True,True)]
    row_labels = [
        "self_eod=False, cons_eod=False",
        "self_eod=True,  cons_eod=False",
        "self_eod=False, cons_eod=True",
        "self_eod=True,  cons_eod=True",
    ]

    # (C.1) Food => 4×2
    plt.figure(figsize=(12, 14))  # taller since 4 rows

    for i, combo_key in enumerate(combos):
        # Grab data from food_data dictionary
        dist_vals = food_data[combo_key]['dist']
        mormy_vals = food_data[combo_key]['mormy']
        amp_vals = food_data[combo_key]['amp']
        row_title = row_labels[i]

        # col 1 => Mormyromast
        subplot_index = (4, 2, i*2 + 1)  # row i, col 1
        plot_scatter_and_regression(
            x=dist_vals,
            y=mormy_vals,
            title=f"Food vs. Mormyromast\n{row_title}",
            x_label="Distance to Food (cm)",
            y_label=f"{operation.capitalize()} Abs Sensor",
            sensor_min=log_scaled_sensor_bounds["mormyromast_sensor_min"],
            sensor_max=log_scaled_sensor_bounds["mormyromast_sensor_max"],
            subplot=subplot_index
        )

        # col 2 => Ampullary
        subplot_index = (4, 2, i*2 + 2)  # row i, col 2
        plot_scatter_and_regression(
            x=dist_vals,
            y=amp_vals,
            title=f"Food vs. Ampullary\n{row_title}",
            x_label="Distance to Food (cm)",
            y_label=f"{operation.capitalize()} Abs Sensor",
            sensor_min=log_scaled_sensor_bounds["ampullary_sensor_min"],
            sensor_max=log_scaled_sensor_bounds["ampullary_sensor_max"],
            subplot=subplot_index
        )

    plt.tight_layout()
    if save_path:
        new_food_path = save_path.split(".png")[0] + "_food_self_cons_eod.png"
        plt.savefig(new_food_path, dpi=300)


    # (C.2) Agents => 4×3
    plt.figure(figsize=(18, 18))  # bigger for 4 rows × 3 columns

    for i, combo_key in enumerate(combos):
        # Grab data from agent_data dictionary
        dist_vals = agent_data[combo_key]['dist']
        mormy_vals = agent_data[combo_key]['mormy']
        knol_vals = agent_data[combo_key]['knol']
        amp_vals = agent_data[combo_key]['amp']
        row_title = row_labels[i]

        # col 1 => Mormyromast
        subplot_index = (4, 3, i*3 + 1)
        plot_scatter_and_regression(
            x=dist_vals,
            y=mormy_vals,
            title=f"Agent vs. Mormyromast\n{row_title}",
            x_label="Distance to Agent (cm)",
            y_label=f"{operation.capitalize()} Abs Sensor",
            sensor_min=log_scaled_sensor_bounds["mormyromast_sensor_min"],
            sensor_max=log_scaled_sensor_bounds["mormyromast_sensor_max"],
            subplot=subplot_index
        )

        # col 2 => Knollen
        subplot_index = (4, 3, i*3 + 2)
        plot_scatter_and_regression(
            x=dist_vals,
            y=knol_vals,
            title=f"Agent vs. Knollen\n{row_title}",
            x_label="Distance to Agent (cm)",
            y_label=f"{operation.capitalize()} Abs Sensor",
            sensor_min=log_scaled_sensor_bounds["knollen_sensor_min"],
            sensor_max=log_scaled_sensor_bounds["knollen_sensor_max"],
            binarization_threshold=log_scaled_sensor_bounds["knollen_binarize_threshold"],
            subplot=subplot_index
        )

        # col 3 => Ampullary
        subplot_index = (4, 3, i*3 + 3)
        plot_scatter_and_regression(
            x=dist_vals,
            y=amp_vals,
            title=f"Agent vs. Ampullary\n{row_title}",
            x_label="Distance to Agent (cm)",
            y_label=f"{operation.capitalize()} Abs Sensor",
            sensor_min=log_scaled_sensor_bounds["ampullary_sensor_min"],
            sensor_max=log_scaled_sensor_bounds["ampullary_sensor_max"],
            subplot=subplot_index
        )

    plt.tight_layout()
    if save_path:
        new_agent_path = save_path.split(".png")[0] + "_agent_self_cons_eod.png"
        plt.savefig(new_agent_path, dpi=300)


# FIXME: out of date since we switched to 0-1 normalization
    # try:
    #     plot_sensor_histograms(dff, save_path=outfile_base + '_sensor_histograms.png')
    #     plot_sensor_histograms(dff, save_path=outfile_base + '_sensor_histograms_log.png', log_scale=True)
    # except Exception as e:
    #     print(f"Error in sensor histograms: {e}")
def plot_sensor_histograms(dff, save_path=None, log_scale=False, xlim=None, num_bins=30):
    sensor_bounds = ru.get_sensor_bounds(dff)
    mormyromast_indices, ampullary_indices, knollen_indices = ru.get_sensor_indices(dff)

    observations = np.vstack(dff['observations'].values)
    mormyromast_observations = observations[:, mormyromast_indices]
    ampullary_observations = observations[:, ampullary_indices]
    knollen_observations = observations[:, knollen_indices]
    
    # plot histograms for each sensor type
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    for i, (sensor_data, sensor_name) in enumerate(zip([mormyromast_observations, knollen_observations, ampullary_observations], ['Mormyromast', 'Knollen', 'Ampullary'])):
        ax = axes[i]
        ax.set_title(f'{sensor_name} Sensor Observations')
        ax.set_xlabel('Sensor Observation')
        ax.set_ylabel('Count')
        bins = None
        if log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('abs(Sensor Observation)')
            sensor_data = sensor_data[sensor_data > 0]
            bins = np.logspace(np.log10(sensor_data.min()), np.log10(sensor_data.max()), num_bins)

        if xlim:
            ax.set_xlim(-xlim, xlim)
            sensor_data = np.clip(sensor_data, -xlim, xlim)

        if bins is None:
            bins = num_bins
        ax.hist(sensor_data.flatten(), bins=bins)
        sensor_min = sensor_bounds[f'{sensor_name.lower()}_sensor_min']
        sensor_max = sensor_bounds[f'{sensor_name.lower()}_sensor_max']
        if sensor_min is not None:
            ax.axvline(log_scale_with_sign(sensor_min, sensor_bounds['general_sensor_min']), color='orange', linestyle='--', label='Min Sensor Value')
        if sensor_max is not None:
            ax.axvline(log_scale_with_sign(sensor_max, sensor_bounds['general_sensor_min']), color='orange', linestyle='--', label='Max Sensor Value')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

