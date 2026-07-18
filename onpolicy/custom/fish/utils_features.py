# These utilities typically operate on the flattened dff

import traceback

import numpy as np
import pandas as pd
import tqdm
from cfg import AGENT_PARAMS, ENV_PARAMS, m_to_cm

_FOOD_SENSING_RADIUS_CM = AGENT_PARAMS['morm_food_detection_range_m'] * m_to_cm
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def add_distance_to_wall(df, feature_names):
    # Make sure that arena_size is available on each row
    df = df.sort_values(by=["env_id", "episode_index", "time_step"])
    df["arena_size"] = df.groupby(["env_id", "episode_index"])["arena_size"].ffill()

    if {"position_x", "position_y", "arena_size"}.issubset(df.columns):
        try:
            df["distance_to_wall"] = df.apply(
                lambda row: (
                    min(
                        row["position_x"],
                        row["arena_size"][0] - row["position_x"],
                        row["position_y"],
                        row["arena_size"][1] - row["position_y"],
                    )
                    if row["arena_size"] is not None and not (np.isscalar(row["arena_size"]) and pd.isna(row["arena_size"]))
                    else np.nan
                ),
                axis=1,
            )
            feature_names.append("distance_to_wall")
        except Exception as e:
            print(f"Could not calculate distance to wall: {e}")
            print("Ensure 'position_x', 'position_y', and 'arena_size' are present in the DataFrame.")
    return df, feature_names


def add_distance_angle_to_closest_food(df, feature_names, add_binned_features=True):
    if {"distance_to_closest_food", "angle_to_closest_food"}.issubset(df.columns):
        feature_names.extend(["distance_to_closest_food", "angle_to_closest_food"])

        # angle_to_closest_food is in radians [-π, π]; derive cos/sin directly.
        df["angle_to_closest_food_cos"] = np.cos(df["angle_to_closest_food"])
        df["angle_to_closest_food_sin"] = np.sin(df["angle_to_closest_food"])
        feature_names.append("angle_to_closest_food_circular")

        if add_binned_features:
            df["distance_to_food_binned"] = pd.cut(
                df["distance_to_closest_food"],
                bins=[0, 10, 15, 40, 100, np.inf],
                labels=range(1, 6),
            ).astype(int)
            feature_names.append("distance_to_food_binned")
            df["angle_to_food_binned"] = pd.cut(
                df["angle_to_closest_food"],
                bins=np.linspace(-np.pi, np.pi, 9),
                labels=range(1, 9),
                include_lowest=True,
            ).astype(int)
            feature_names.append("angle_to_food_binned")
    return df, feature_names


def add_velocity_to_nearest_agent(df, feature_names):
    df["approach_nearest_velocity"] = (
        df.groupby(["episode_index", "env_id", "agent_id"])["distance_to_nearest_agent"]
        .diff()
        .fillna(0)
    )
    feature_names.append("approach_nearest_velocity")
    return df, feature_names


def add_knollen_by_dist(df, feature_names):
    # 1) Stack your knollen fields into an (N, A, 2) array
    knollen = np.stack(df["center_field.knollen"].to_numpy())      # shape = (n_rows, n_agents, 2)

    # 2) Grab the per‐row sort‐orders: an (N, A) int array
    #    Each row is a permutation of agent indices
    sort_idx = np.stack(df["agent_ids_by_dist"].to_numpy())        # shape = (n_rows, n_agents)

    # 3) Use advanced indexing to pull out the (x,y) vectors in sorted order
    #    vects[i, j, :] = knollen[i, sort_idx[i,j], :]
    rows = np.arange(len(df))[:, None]                             # shape = (n_rows, 1)
    vects_sorted = knollen[rows, sort_idx, :]                      # shape = (n_rows, n_agents, 2)

    # 4) Compute field directions per‐neighbor: arctan2(y, x)
    #    vects_sorted[...,0] is x, vects_sorted[...,1] is y
    dirs_by_dist = np.arctan2(vects_sorted[...,1], vects_sorted[...,0])

    # 5) Normalize zero‐vectors to NaN if you still want
    zero_mask = (vects_sorted[...,0] == 0) & (vects_sorted[...,1] == 0)
    dirs_by_dist[zero_mask] = np.nan

    # 6) Compute error angles: subtract agent’s orientation, normalize to [-π, π]
    ori = df["orientation"].to_numpy()                             # shape = (n_rows,)
    # broadcast ori[:,None] across the second axis
    errs_by_dist = np.mod(ori[:, None] - dirs_by_dist + np.pi, 2 * np.pi) - np.pi
    # propagate NaN where direction was NaN
    errs_by_dist[np.isnan(dirs_by_dist)] = np.nan

    df = df.assign(
        knollen_field_direction_by_dist = list(dirs_by_dist),
        knollen_error_angle_by_dist     = list(errs_by_dist)
    )
    # feature_names.extend([
    #     "knollen_field_direction_by_dist",
    #     "knollen_error_angle_by_dist",
    # ])  # for now, don't use this since they're lists
    return df, feature_names


def extract_knollen_direction_error_angle(df, feature_names):
    df['knollen_field_direction_nearest'] = df['knollen_field_direction_by_dist'].apply(lambda x: x[1] if len(x) > 1 else np.nan)
    df['knollen_error_angle_nearest'] = df['knollen_error_angle_by_dist'].apply(lambda x: x[1] if len(x) > 1 else np.nan)
    df['knollen_field_direction_nearest'] = df['knollen_field_direction_nearest'].apply(
        lambda x: np.nan if x == 0.0 else x
    )
    df['knollen_error_angle_nearest'] = df['knollen_error_angle_nearest'].apply(
        lambda x: np.nan if x == 0.0 else x
    )
    feature_names.append('knollen_field_direction_nearest')
    feature_names.append('knollen_error_angle_nearest')
    return df, feature_names


def add_morm_amp_field_features(df, feature_names):
    sensor_types = ["mormyromast", "ampullary"]  # ["mormyromast", "ampullary", "knollen"]
    for sensor_type in sensor_types:
        sensor_col = f"center_field.{sensor_type}"
        if sensor_col in df.columns:
            try:
                # Extract x and y components of the field
                df[f"{sensor_type}_field_x"] = df[sensor_col].apply(lambda x: x[0])
                df[f"{sensor_type}_field_y"] = df[sensor_col].apply(lambda x: x[1])

                # Gradient magnitude and log-magnitude
                df[f"{sensor_type}_field_magnitude"] = np.hypot(
                    df[f"{sensor_type}_field_x"], df[f"{sensor_type}_field_y"]
                )
                feature_names.append(f"{sensor_type}_field_magnitude")

                df[f"{sensor_type}_field_magnitude_log"] = df[f"{sensor_type}_field_magnitude"].apply(
                    lambda x: np.log10(x) if x > 0 else 0
                )
                feature_names.append(f"{sensor_type}_field_magnitude_log")

                # Field direction (angle)
                df[f"{sensor_type}_field_direction"] = df.apply(
                    lambda row: (
                        np.arctan2(
                            row[f"{sensor_type}_field_y"],
                            row[f"{sensor_type}_field_x"],
                        )
                        if not (
                            row[f"{sensor_type}_field_x"] == 0
                            and row[f"{sensor_type}_field_y"] == 0
                        )
                        else np.nan
                    ),
                    axis=1,
                )
                # normalize to [-π, π]
                df[f"{sensor_type}_field_direction"] = (
                    df[f"{sensor_type}_field_direction"] + np.pi
                ) % (2 * np.pi) - np.pi

                feature_names.append(f"{sensor_type}_field_direction")

                # Error angle between field direction and agent orientation
                df[f"{sensor_type}_error_angle"] = df.apply(
                    lambda row: (
                        np.arctan2(
                            np.sin(row[f"{sensor_type}_field_direction"] - row["orientation"]),
                            np.cos(row[f"{sensor_type}_field_direction"] - row["orientation"])
                        )
                        if not pd.isna(row[f"{sensor_type}_field_direction"]) else np.nan
                    ),
                    axis=1,
                )
                # normalize to [-π, π]
                df[f"{sensor_type}_error_angle"] = (
                    df[f"{sensor_type}_error_angle"] + np.pi
                ) % (2 * np.pi) - np.pi
                feature_names.append(f"{sensor_type}_error_angle")

            except Exception as e:
                print(f"Could not process {sensor_type} gradients: {e}")
    return df, feature_names


def prepare_features_for_decoding(df, normalize=False, behavior_mode="homing"):
    """Engineer the decoding feature columns and return (df, feature_names).

    Ported from the now-deleted utils_decoding.py so it lives next to the
    helpers it orchestrates.  For ``behavior_mode="homing"`` it adds the
    knollen/mormyromast/ampullary error angles, field magnitudes, and
    distance-to-wall; for ``"foraging"`` it additionally adds food distance/angle,
    velocity-to-nearest-agent, EOD rolling windows, and pulls in preprocessed
    social/food columns already present on the frame.

    The homing decoding (utils_homing) calls this; the archived predict_action
    analysis (archive/analysis_predict_action.py) also used it.
    """
    df = df.copy()
    feature_names = []

    try:
        df, feature_names = add_knollen_by_dist(df, feature_names)
    except Exception as e:
        print("[ERROR]", e)
        traceback.print_exc()

    try:
        df, feature_names = extract_knollen_direction_error_angle(df, feature_names)
    except Exception as e:
        print("[ERROR]", e)

    # Field magnitude/direction/error-angle for sensors with directional info
    try:
        df, feature_names = add_morm_amp_field_features(df, feature_names)
    except Exception as e:
        print("[ERROR]", e)

    try:
        df, feature_names = add_distance_to_wall(df, feature_names)
    except Exception as e:
        print("[ERROR]", e)

    if behavior_mode == "foraging":
        try:
            df, feature_names = add_distance_angle_to_closest_food(df, feature_names)
        except Exception as e:
            print("[ERROR]", e)
        try:
            df, feature_names = add_velocity_to_nearest_agent(df, feature_names)
        except Exception as e:
            print("[ERROR]", e)
        try:
            df, eod_rate_features = add_eod_rolling_windows(df)
            feature_names.extend(eod_rate_features)
        except Exception as e:
            print("[ERROR]", e)

        PREPROCESSED_FEATURES_TO_ADD = [
            "distance_to_nearest_agent",
            "size_of_nearest_agent",
            "distance_to_second_nearest_agent",
            "size_of_second_nearest_agent",
            "food_count_5cm",
            "food_front_5cm",
            "food_back_5cm",
            "food_left_5cm",
            "food_right_5cm",
        ]
        for existent_feature in PREPROCESSED_FEATURES_TO_ADD:
            if existent_feature in df.columns:
                feature_names.append(existent_feature)

    if normalize:
        original_features = list(feature_names)
        for feature_name in original_features:
            if feature_name.endswith("_binned") or feature_name.endswith("_circular"):
                continue
            try:
                df[f"{feature_name}_normalized"] = (
                    df[feature_name] - df[feature_name].mean()
                ) / df[feature_name].std()
                feature_names.append(f"{feature_name}_normalized")
            except Exception as e:
                print(f"Could not normalize {feature_name}: {e}")

    return df, feature_names


def get_df_with_group_agent_distance_lists(dff):
    """
    For each agent (row) in dff, compute and attach:
      • agent_ids_by_dist:    list of all n agent_ids sorted by distance from this agent
      • distances_by_dist:    list of n distances (including self=0.0), sorted
      • emit_eod_by_dist:     list of n emit_eod flags, sorted by distance
      • agent_sizes_by_dist:  list of n agent_size values, sorted by distance
      • angles_by_dist:       list of n angles (radians, normalised to [-π, π])

    NOTE: use index 1 to get the nearest non-self agent.

    Vectorised implementation: sorts the whole DataFrame once, reshapes to
    (n_groups, n_agents, features), computes pairwise distances via numpy
    broadcasting, and avoids the per-timestep Python groupby loop entirely.
    ~14× faster than the original groupby implementation.
    """
    sort_cols  = ['env_id', 'episode_index', 'time_step', 'agent_id']
    orig_index = dff.index
    dff_s      = dff.sort_values(sort_cols)

    # Infer uniform agent count from the first timestep group
    A = int(dff_s.groupby(
        ['env_id', 'episode_index', 'time_step'], sort=False
    ).size().iloc[0])
    N = len(dff_s)
    G = N // A

    pos   = np.column_stack([dff_s['position_x'].values,
                              dff_s['position_y'].values]).astype(np.float32)
    ori   = dff_s['orientation'].values.astype(np.float32)
    a_ids = dff_s['agent_id'].values
    emit  = dff_s['emit_eod'].values.astype(bool)
    sizes = dff_s['agent_size'].values.astype(np.float32)

    pos_3d  = pos.reshape(G, A, 2)
    ori_2d  = ori.reshape(G, A)
    ids_2d  = a_ids.reshape(G, A)
    emit_2d = emit.reshape(G, A)
    size_2d = sizes.reshape(G, A)

    # Pairwise distances: (G, A, A)
    diff     = pos_3d[:, :, None, :] - pos_3d[:, None, :, :]
    dist_mat = np.sqrt((diff ** 2).sum(axis=-1))
    order    = np.argsort(dist_mat, axis=2)                       # (G, A, A)

    sorted_dist = np.take_along_axis(dist_mat, order, axis=2)

    def _gather(arr_2d):
        return np.take_along_axis(
            np.broadcast_to(arr_2d[:, None, :], (G, A, A)).copy(),
            order, axis=2,
        )

    sorted_ids  = _gather(ids_2d)
    sorted_emit = _gather(emit_2d)
    sorted_size = _gather(size_2d)

    # Angles: vector from each agent to its k-th nearest neighbour
    g_idx      = np.arange(G).reshape(G, 1, 1)
    sorted_pos = pos_3d[g_idx, order, :]                         # (G, A, A, 2)
    vectors    = sorted_pos - pos_3d[:, :, None, :]              # (G, A, A, 2)
    raw_angles = np.arctan2(vectors[..., 1], vectors[..., 0])    # (G, A, A)
    angle_diff = raw_angles - ori_2d[:, :, None]
    angles_norm = (angle_diff + np.pi) % (2 * np.pi) - np.pi    # (G, A, A)

    # Flatten to (N, A) and convert to per-row lists for pandas
    result = dff_s.copy()
    result['agent_ids_by_dist']   = sorted_ids.reshape(N, A).tolist()
    result['distances_by_dist']   = sorted_dist.reshape(N, A).tolist()
    result['emit_eod_by_dist']    = sorted_emit.reshape(N, A).tolist()
    result['agent_sizes_by_dist'] = sorted_size.reshape(N, A).tolist()
    result['angles_by_dist']      = angles_norm.reshape(N, A).tolist()

    return result.loc[orig_index]


def extract_nearest_agent_info(dff):
    """
    Extract nearest/second-nearest agent scalars from the sorted list columns.
    Converts each object column to a (N, A) float array once and slices by column index,
    avoiding per-row Python iteration.
    """
    A = len(dff['distances_by_dist'].iloc[0])
    if A == 1:
        for col in ('distance_to_nearest_agent', 'size_of_nearest_agent',
                    'angle_to_closest_agent', 'distance_to_second_nearest_agent',
                    'size_of_second_nearest_agent', 'angle_to_second_closest_agent'):
            dff[col] = np.nan
        dff['nearest_agent_id'] = None
        dff['second_nearest_agent_id'] = None
        return dff

    # Convert object columns to (N, A) arrays in one pass
    dist_arr  = np.array(dff['distances_by_dist'].tolist(),   dtype=np.float32)
    sizes_arr = np.array(dff['agent_sizes_by_dist'].tolist(), dtype=np.float32)
    ang_arr   = np.array(dff['angles_by_dist'].tolist(),      dtype=np.float32)
    ids_arr   = np.array(dff['agent_ids_by_dist'].tolist(),   dtype=object)

    dff['distance_to_nearest_agent'] = dist_arr[:, 1]
    dff['size_of_nearest_agent']     = sizes_arr[:, 1]
    dff['nearest_agent_id']          = ids_arr[:, 1]
    dff['angle_to_closest_agent']    = ang_arr[:, 1]

    if A > 2:
        dff['distance_to_second_nearest_agent'] = dist_arr[:, 2]
        dff['size_of_second_nearest_agent']     = sizes_arr[:, 2]
        dff['second_nearest_agent_id']          = ids_arr[:, 2]
        dff['angle_to_second_closest_agent']    = ang_arr[:, 2]
    else:
        for col in ('distance_to_second_nearest_agent', 'size_of_second_nearest_agent',
                    'angle_to_second_closest_agent'):
            dff[col] = np.nan
        dff['second_nearest_agent_id'] = None

    return dff


# Function to calculate distance to closest food
def distance_to_closest_food(row):
    agent_position = np.array(row["position"])
    food_positions = np.array(row["food_positions"])
    if food_positions.size == 0:
        return np.nan
    distances = np.linalg.norm(food_positions - agent_position, axis=1)
    return np.min(distances)


# Function to calculate angle to closest food
# TODO switch to radians, combine with distance_to_closest_food, or possibly vectorize
def angle_to_closest_food(row):
    agent_position = np.array(row["position"])
    agent_orientation = row["orientation"]
    food_positions = np.array(row["food_positions"])
    if food_positions.size == 0:
        return np.nan
    distances = np.linalg.norm(food_positions - agent_position, axis=1)
    closest_food_position = food_positions[np.argmin(distances)]
    vector_to_food = closest_food_position - agent_position
    angle = np.arctan2(vector_to_food[1], vector_to_food[0]) - agent_orientation
    angle_deg = np.degrees(angle) % 360
    # Convert from [0, 360] to [-180, 180]
    return (angle_deg + 180) % 360 - 180


def food_count_5cm(row):
    """
    Returns the total number of food items within 5 cm of the agent.
    """
    agent_position = np.array(row["position"])
    food_positions = np.array(row["food_positions"])

    if food_positions.size == 0:
        return 0  # No food at all

    # Distances from agent to each food item
    distances = np.linalg.norm(food_positions - agent_position, axis=1)

    return np.sum(distances <= _FOOD_SENSING_RADIUS_CM)


def quadrant_food_count_5cm(row):
    """
    Returns a pd.Series with the count of food items in each quadrant
    (front, back, left, right), all within 5 cm of the agent.

    Quadrants are defined relative to agent_orientation = 0:
    - front: angle in [-45°, 45°)
    - right: angle in [-135°, -45°)
    - left: angle in [45°, 135°)
    - back: angle in [135°, 180) U [-180°, -135°)
    """
    agent_position = np.array(row["position"])
    agent_orientation = row["orientation"]  # in radians
    food_positions = np.array(row["food_positions"])

    # If no food, just return zero for all quadrants
    if food_positions.size == 0:
        return pd.Series(
            [0, 0, 0, 0],
            index=[
                "food_front_5cm",
                "food_back_5cm",
                "food_left_5cm",
                "food_right_5cm",
            ],
        )

    # Distances from agent to each food item
    distances = np.linalg.norm(food_positions - agent_position, axis=1)
    within_5_mask = distances <= _FOOD_SENSING_RADIUS_CM
    relevant_food_positions = food_positions[within_5_mask]

    # If none within 5 cm, all quadrant counts are zero
    if len(relevant_food_positions) == 0:
        return pd.Series(
            [0, 0, 0, 0],
            index=[
                "food_front_5cm",
                "food_back_5cm",
                "food_left_5cm",
                "food_right_5cm",
            ],
        )

    # Vectors from agent to each food item (within 5 cm)
    vectors = relevant_food_positions - agent_position

    # Compute angles (relative to agent orientation)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0]) - agent_orientation
    # Convert angles to degrees in [-180, 180]
    angles_deg = np.degrees(angles) % 360
    angles_deg = (angles_deg + 180) % 360 - 180

    # Define quadrant masks
    # front: [-45, 45)
    front_mask = (angles_deg >= -45) & (angles_deg < 45)
    # right: [-135, -45)
    right_mask = (angles_deg >= -135) & (angles_deg < -45)
    # left: [45, 135)
    left_mask = (angles_deg >= 45) & (angles_deg < 135)
    # back: everything else => [135, 180) or [-180, -135)
    back_mask = (angles_deg >= 135) | (angles_deg < -135)

    front_count = np.sum(front_mask)
    back_count = np.sum(back_mask)
    left_count = np.sum(left_mask)
    right_count = np.sum(right_mask)

    return pd.Series(
        [front_count, back_count, left_count, right_count],
        index=["food_front_5cm", "food_back_5cm", "food_left_5cm", "food_right_5cm"],
    )


# def check_observations(observations, object_type, num_rays=30):
#     if object_type == OBJECT_TYPES["NONE"]:
#         # Check if all observations are "NONE"
#         return all(
#             observations[i + 1] == OBJECT_TYPES["NONE"]
#             for i in range(0, num_rays * 2, 2)
#         )
#     elif object_type == OBJECT_TYPES["AGENT"]:
#         # Check if any observation is within the agent range
#         return any(
#             OBJECT_TYPES["AGENT_MIN"]
#             <= observations[i + 1]
#             <= OBJECT_TYPES["AGENT_MAX"]
#             for i in range(0, num_rays * 2, 2)
#         )
#     else:
#         # For other objects, check exact match as before
#         return any(
#             observations[i + 1] == object_type for i in range(0, num_rays * 2, 2)
#         )

def add_event_counters(df, event_columns):
    group_cols = ["env_id", "episode_index", "agent_id"]
    event_counter_colnames = []
    tmp = "_tmp_event_t"
    for event in event_columns:
        col = f"time_since_last_{event}"
        event_counter_colnames.append(col)
        df[tmp] = df["time_step"].where(df[event].astype(bool))
        df[tmp] = df.groupby(group_cols, sort=False)[tmp].ffill()
        df[col] = df["time_step"] - df[tmp]
        df.drop(columns=[tmp], inplace=True)
    return df, event_counter_colnames


try:
    from numba import njit as _njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False


if _NUMBA_AVAILABLE:
    @_njit(cache=True)
    def _food_features_numba(positions, orientations,
                              food_flat, food_offsets, food_counts,
                              food_radius):
        """
        Numba JIT core for add_food_features_fast.
        food_flat    : (total_food, 2) float64 — all food positions concatenated
        food_offsets : (n,) int64         — start index into food_flat per row
        food_counts  : (n,) int64         — number of food items per row
        """
        n = len(positions)
        dist_to_food  = np.full(n, np.nan)
        angle_to_food = np.full(n, np.nan)
        food_front    = np.zeros(n, dtype=np.int64)
        food_back     = np.zeros(n, dtype=np.int64)
        food_left     = np.zeros(n, dtype=np.int64)
        food_right    = np.zeros(n, dtype=np.int64)
        pi4  = np.pi / 4
        pi34 = 3.0 * np.pi / 4

        for i in range(n):
            cnt = food_counts[i]
            if cnt == 0:
                continue
            off = food_offsets[i]
            px, py  = positions[i, 0], positions[i, 1]
            ori     = orientations[i]
            min_d   = 1e18
            min_ang = 0.0

            for j in range(cnt):
                fx = food_flat[off + j, 0]
                fy = food_flat[off + j, 1]
                dx = fx - px
                dy = fy - py
                d  = np.sqrt(dx * dx + dy * dy)

                if d < min_d:
                    min_d = d
                    raw   = np.arctan2(dy, dx) - ori
                    min_ang = (raw + np.pi) % (2.0 * np.pi) - np.pi

                if d <= food_radius:
                    a = np.arctan2(dy, dx) - ori
                    a = (a + np.pi) % (2.0 * np.pi) - np.pi
                    if -pi4 <= a < pi4:
                        food_front[i] += 1
                    elif -pi34 <= a < -pi4:
                        food_right[i] += 1
                    elif pi4 <= a < pi34:
                        food_left[i] += 1
                    else:
                        food_back[i] += 1

            dist_to_food[i]  = min_d
            angle_to_food[i] = min_ang

        return dist_to_food, angle_to_food, food_front, food_back, food_left, food_right


def add_food_features_fast(merged_df):
    """
    Compute distance/angle to closest food and quadrant food counts per row.

    Uses a Numba JIT kernel when numba is available (typically 20-50× faster
    than the pure-Python loop on 1M+ rows); falls back to pure numpy otherwise.
    """
    positions_arr = np.stack(merged_df["position"].values).astype(np.float64)
    orientations  = merged_df["orientation"].values.astype(np.float64)
    food_pos_list = merged_df["food_positions"].values
    n = len(merged_df)

    if _NUMBA_AVAILABLE:
        # Build ragged food array: flatten all food positions with offsets
        food_arrays  = [np.asarray(fp, dtype=np.float64).reshape(-1, 2)
                        for fp in food_pos_list]
        food_counts  = np.array([len(fa) for fa in food_arrays], dtype=np.int64)
        food_offsets = np.zeros(n, dtype=np.int64)
        food_offsets[1:] = np.cumsum(food_counts[:-1])
        total = int(food_counts.sum())
        food_flat = np.empty((max(total, 1), 2), dtype=np.float64)
        for i, fa in enumerate(food_arrays):
            if len(fa) > 0:
                food_flat[food_offsets[i]: food_offsets[i] + len(fa)] = fa

        dist_to_food, angle_to_food, ff, fb, fl, fr = _food_features_numba(
            positions_arr, orientations, food_flat, food_offsets, food_counts,
            float(_FOOD_SENSING_RADIUS_CM),
        )
    else:
        dist_to_food  = np.full(n, np.nan)
        angle_to_food = np.full(n, np.nan)
        ff = np.zeros(n, dtype=np.int64)
        fb = np.zeros(n, dtype=np.int64)
        fl = np.zeros(n, dtype=np.int64)
        fr = np.zeros(n, dtype=np.int64)
        for i in range(n):
            fp = np.asarray(food_pos_list[i])
            if fp.size == 0:
                continue
            diffs = fp - positions_arr[i]
            dists = np.linalg.norm(diffs, axis=1)
            min_idx = np.argmin(dists)
            dist_to_food[i] = dists[min_idx]
            v   = diffs[min_idx]
            ori = orientations[i]
            raw = np.arctan2(v[1], v[0]) - ori
            angle_to_food[i] = (raw + np.pi) % (2 * np.pi) - np.pi
            in_range = dists <= _FOOD_SENSING_RADIUS_CM
            if in_range.any():
                ang  = (np.arctan2(diffs[in_range, 1], diffs[in_range, 0]) - ori)
                ang  = (ang + np.pi) % (2 * np.pi) - np.pi
                pi4  = np.pi / 4;  pi34 = 3 * np.pi / 4
                ff[i] = np.sum((ang >= -pi4)  & (ang < pi4))
                fr[i] = np.sum((ang >= -pi34) & (ang < -pi4))
                fl[i] = np.sum((ang >= pi4)   & (ang < pi34))
                fb[i] = np.sum((ang >= pi34)  | (ang < -pi34))

    merged_df["distance_to_closest_food"] = dist_to_food
    merged_df["angle_to_closest_food"]    = angle_to_food
    merged_df["food_front_5cm"]           = ff
    merged_df["food_back_5cm"]            = fb
    merged_df["food_left_5cm"]            = fl
    merged_df["food_right_5cm"]           = fr
    merged_df["food_count_5cm"]           = ff + fb + fl + fr
    return merged_df


# # Function to prepare the auxiliary DataFrame for sensor readings conditioned on the queried object type
# def prepare_sensor_df_conditioned(df, num_rays=30, object_type="FOOD"):
#     from MAFish import OBJECT_TYPES
#     queried_object = OBJECT_TYPES.get(object_type)

#     if queried_object is None:
#         raise ValueError(f"Unknown object type: {object_type}")

#     # Initialize an empty DataFrame with sensor columns
#     sensor_cols = [f'sensor_{i}' for i in range(num_rays)]
#     sensor_df = pd.DataFrame(index=df.index, columns=sensor_cols)

#     # Populate the sensor_df with distances for each ray, conditioned on the queried object
#     for idx, row in df.iterrows():
#         observations = row['observations'][:num_rays*2:2]  # Extract distances
#         ray_objects = row['observations'][1:num_rays*2:2]  # Extract objects

#         conditioned_dists = []
#         for dist, obj in zip(observations, ray_objects):
#             if obj == queried_object:
#                 conditioned_dists.append(dist)  # Keep the original distance if object matches
#             else:
#                 conditioned_dists.append(1.0)  # Assign max distance (1.0) if object doesn't match

#         sensor_df.loc[idx] = conditioned_dists

#     return sensor_df


# Main function to calculate EMA and highest EMA angle for knollen sensors
def add_highest_ema_knollen_sensors(df, alpha=0.3, time_window=10):
    #     num_rays = dff['metadata'].iloc[0]['agent_args']['num_rays']
    # # num_knollen_sensors = dff['metadata'].iloc[0]['agent_args']['num_knollen_sensors']
    # # num_ampullary_sensors = dff['metadata'].iloc[0]['agent_args']['num_ampullary_sensors']
    num_rays = 30
    num_knollen_sensors = 12  # Number of knollen sensors
    angle_per_sensor = (
        2 * np.pi / num_knollen_sensors
    )  # Equal angle spacing for knollen sensors

    # Prepare an auxiliary DataFrame for knollen sensor distances
    sensor_cols = [f"knollen_sensor_{i}" for i in range(num_knollen_sensors)]
    sensor_df = pd.DataFrame(index=df.index, columns=sensor_cols)

    # Populate the sensor_df with distances from the knollen sensors
    for idx, row in df.iterrows():
        knollen_distances = row["observations"][
            num_rays : num_rays + num_knollen_sensors
        ]  # idxs 60-71 are knollen
        sensor_df.loc[idx] = knollen_distances

    # Compute the EMA over time for each knollen sensor using pandas' ewm
    ema_df = sensor_df.ewm(span=time_window, adjust=False).mean()

    def calculate_knollen_angle_and_ema(row, ema_row):
        # Find the index of the highest EMA
        highest_ema_idx = np.argmax(ema_row)
        highest_ema_value = ema_row[highest_ema_idx]

        # If the highest EMA value is 1, return -1 for the angle
        if highest_ema_value == 1.0:
            return -1, highest_ema_value

        # Calculate the corresponding angle for the knollen sensor
        angle = highest_ema_idx * angle_per_sensor
        return (
            angle % (2 * np.pi),
            highest_ema_value,
        )  # Normalize the angle and return the value

    # Apply the calculation for each row and ema_row
    results = df.apply(
        lambda row: calculate_knollen_angle_and_ema(row, ema_df.loc[row.name]), axis=1
    )

    # Add both the angle and the EMA value to the DataFrame
    df["highest_ema_angle_knollen"] = [result[0] for result in results]
    df["highest_ema_value_knollen"] = [result[1] for result in results]
    df["summed_ema_values_knollen"] = ema_df.sum(axis=1)

    return df


# Example usage
# merged_df = add_highest_ema_knollen_sensors(merged_df, alpha=0.3, time_window=10)


# Function to get the number of agents within the knollen range
def get_num_agents_in_knollen_range_optimized(merged_df, knollen_range=100):
    """
    Count conspecifics within knollen_range cm for each row.

    Uses the already-computed distances_by_dist column (sorted distances to all
    agents including self at index 0) rather than re-computing pairwise distances
    via a 288K-group Python loop.  48× faster on 1.15M-row DataFrames.
    """
    dist_arr = np.array(merged_df['distances_by_dist'].tolist(), dtype=np.float32)
    # index 0 is self (distance = 0); count agents at indices 1..A-1 within range
    return np.sum(dist_arr[:, 1:] < knollen_range, axis=1)

################ EOD related ################
_EOD_RATE_WINDOW = int(ENV_PARAMS["fps_sim"] / 4)  # 20 steps ≈ 240 ms


def add_eod_rolling_windows(df, eod_rate_window=None):
    """
    Adds three EOD-rate columns (Hz) via a moving-average window of size `eod_rate_window`.

    Rate formula: rolling_mean(emit_eod, W) * fps_sim  →  pulses per second (Hz)

    Columns added:
      eod_rate_causal     — MA over the previous W steps (causal filter)
      eod_rate_centered   — symmetric MA centred on the current step
      eod_rate_predictive — MA over the next W steps (look-ahead)

    Requirements: 'env_id', 'episode_index', 'agent_id', 'time_step', 'emit_eod'.
    """
    if eod_rate_window is None:
        eod_rate_window = _EOD_RATE_WINDOW
    fps = ENV_PARAMS["fps_sim"]

    df = df.sort_values(["env_id", "episode_index", "agent_id", "time_step"])
    group_cols = ["env_id", "episode_index", "agent_id"]
    grouped = df.groupby(group_cols)["emit_eod"]

    # causal: rows up to i-1 (shift so current is excluded)
    df["eod_rate_causal"] = (
        grouped.rolling(window=eod_rate_window, min_periods=1)
        .mean()
        .shift(1)
        .values
    ) * fps

    # centered: symmetric window around row i
    df["eod_rate_centered"] = (
        grouped.rolling(window=eod_rate_window, center=True, min_periods=1)
        .mean()
        .values
    ) * fps

    # predictive: next W steps (shift back)
    df["eod_rate_predictive"] = (
        grouped.rolling(window=eod_rate_window, min_periods=1)
        .mean()
        .shift(-eod_rate_window + 1)
        .values
    ) * fps

    eod_rate_features = ["eod_rate_causal", "eod_rate_centered", "eod_rate_predictive"]
    return df, eod_rate_features


################ RNN related ################
def get_rnn_state_deltas(dff, pad_before=True):
    # first sort by (env_id, episode_index, agent_id, time_step)
    dff = dff.sort_values(
        by=["env_id", "episode_index", "agent_id", "time_step"], ignore_index=True
    )

    # For each (env_id, episode_index, agent_id), calculate delata_(rnn_state) per timestep
    def calculate_rnn_deltas(group):
        rnn_states = np.vstack(group["rnn_states"].tolist())
        deltas = np.diff(rnn_states, axis=0)
        deltas = np.linalg.norm(
            deltas, axis=1
        ).tolist()  # Calculate the norm of the deltas
        # Pad with zeros at the beginning to match original length
        if pad_before:
            deltas = [0.0] + deltas  # First timestep has no delta, so we pad with 0
        else:
            deltas = deltas + [0.0]
        assert len(deltas) == len(group["rnn_states"])
        return pd.Series(deltas, index=group.index)  # needed for transform()

    # Apply the function groupwise
    rnn_deltas = dff.groupby(
        ["env_id", "episode_index", "agent_id"], group_keys=False
    ).apply(calculate_rnn_deltas)
    return rnn_deltas


def knn_distance_analysis(X, k=5, plot=True, pca_dim=-1, verbose=False):
    """
    Computes k-nearest neighbor distances for clumpiness analysis.

    Parameters:
    - X: ndarray of shape (N, d), your dataset.
    - k: int, which nearest neighbor distance to compute (default: 5).
    - plot: bool, whether to show a histogram.

    Returns:
    - distances: ndarray of shape (N,), k-NN distances for each point.
    """
    if pca_dim > 0:
        pca_dim = min(pca_dim, X.shape[1])
        if verbose:
            print(f"Reducing {X.shape[1]} to {pca_dim} PCA components for efficency.")
        pca = PCA(n_components=pca_dim)
        X = pca.fit_transform(X)

    if verbose:
        print(f"Computing {k}-NN distances in {X.shape[1]}-dimensional space...")
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X)
    distances, _ = nbrs.kneighbors(X)

    # distances[:, 0] is distance to self (0), so take the k-th neighbor
    k_distances = distances[:, k]

    if plot:
        plt.figure(figsize=(6, 3))
        plt.hist(k_distances, bins=40, density=True, edgecolor="k")
        plt.title(f"Histogram of {k}-Nearest Neighbor Distances")
        plt.xlabel("Distance")
        plt.ylabel("Density")
        plt.axvline(np.mean(k_distances), color="r", linestyle="dashed", linewidth=2)
        plt.axvline(np.median(k_distances), color="g", linestyle="dashed", linewidth=2)
        plt.legend(["Mean", "Median"])
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return k_distances


# State clumpiness considering dynamical data & episodes
def dynamic_knn_clumpiness(X, window_size=50, k=5, pca_dim=-1):
    T = X.shape[0]
    scores = []

    for t in range(T - window_size + 1):
        window = X[t : t + window_size]
        dists = knn_distance_analysis(window, k=k, plot=False, pca_dim=pca_dim)
        scores.append(dists)
    return scores


# Compute clumpiness scores for each (env_id, agent_id, episode_index)
def compute_clumpiness_scores_df(dff, window_size=50, k=5, pca_dim=-1):
    clumpiness_scores = []

    for (env_id, agent_id, episode_index), group in dff.groupby(
        ["env_id", "agent_id", "episode_index"]
    ):
        rnn_states = np.vstack(group["rnn_states"].tolist())
        scores = dynamic_knn_clumpiness(
            rnn_states,
            window_size=window_size,
            k=k,
            pca_dim=pca_dim,
        )
        clumpiness_scores.append(
            pd.DataFrame(
                {
                    "env_id": env_id,
                    "agent_id": agent_id,
                    "episode_index": episode_index,
                    "clumpiness_score": scores,
                }
            )
        )

    return pd.concat(clumpiness_scores, ignore_index=True)


################ Attention related ################
def calculate_entropy(data_matrix):
    """
    Calculate the entropy of each row in a 2D numpy array.
    Each row represents attention weights for a specific time step.
    """
    if data_matrix.size == 0:
        return np.nan

    # Normalize the weights
    normalized_weights = data_matrix / np.sum(data_matrix, axis=1, keepdims=True)
    normalized_weights = np.nan_to_num(normalized_weights)  # Handle NaNs

    # Calculate entropy for each row
    entropy = -np.nansum(
        normalized_weights * np.log2(normalized_weights + 1e-10), axis=1
    )
    return entropy


def add_attention_entropy_columns(dff, sensor_types, colname="attn_mask"):
    """
    Add columns to dff for the entropy of attention weights for each sensor type.
    """
    data_matrix = np.stack(dff[colname].values)
    print("data_matrix.shape", data_matrix.shape)
    dff["attn_entropy"] = calculate_entropy(data_matrix)

    for sensor_name, indices in sensor_types:
        sensor_data = data_matrix[:, indices]
        col_name = f"attn_{sensor_name.lower()}_entropy"
        dff[col_name] = calculate_entropy(sensor_data)

    return dff


################ Uitls ################
def calculate_column_feature_correlations(dff, features, col="rnn_deltas"):
    """
    Calculate the correlation of a column with a list of features in a DataFrame.
    Args:
        dff (pd.DataFrame): DataFrame containing the data.
        features (list): List of feature names to calculate correlations with.
        col (str): The column name to correlate with the features.
    Returns:
        pd.DataFrame: DataFrame with features as index and their correlation with the specified column.
    """
    correlations = {}
    for idx in tqdm.tqdm(range(len(features)), desc="Calculating correlations"):
        feature = features[idx]
        if feature in dff.columns:
            corr = np.corrcoef(dff[col], dff[feature])[0, 1]
            correlations[feature] = corr
    return pd.DataFrame.from_dict(correlations, orient="index", columns=["corr"])


def compute_wall_distance(df):
    """
    Vectorised wall distance (cm). arena_size column must be a list [w, h].
    """
    arena_w = df["arena_size"].apply(lambda s: s[0])
    arena_h = df["arena_size"].apply(lambda s: s[1])
    dx = np.minimum(df["position_x"] - 1, arena_w - df["position_x"])
    dy = np.minimum(df["position_y"] - 1, arena_h - df["position_y"])
    return np.minimum(dx, dy)


def add_size_advantage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `size_advantage` column = self agent_size − mean opponent agent_size per
    (env_id, episode_index, agent_id).  Constant within an episode; NaN when
    agent_size is missing or only one agent present.

    The sign mirrors the knollen_metadata "relative" encoding: positive means
    the agent is larger than its conspecific(s), negative means smaller.
    """
    if "agent_size" not in df.columns:
        return df
    if "size_advantage" in df.columns:
        return df
    ep_sizes = (
        df[["env_id", "episode_index", "agent_id", "agent_size"]]
        .drop_duplicates(subset=["env_id", "episode_index", "agent_id"])
    )
    records = []
    for (env_id, ep_idx), g in ep_sizes.groupby(["env_id", "episode_index"], sort=False):
        for _, row in g.iterrows():
            others = g.loc[g["agent_id"] != row["agent_id"], "agent_size"]
            adv = float(row["agent_size"] - others.mean()) if len(others) > 0 else np.nan
            records.append({"env_id": env_id, "episode_index": ep_idx,
                            "agent_id": row["agent_id"], "size_advantage": adv})
    if not records:
        return df
    adv_df = pd.DataFrame(records)
    return df.merge(adv_df, on=["env_id", "episode_index", "agent_id"], how="left")


def compute_idis(df):
    """
    Return all IDIs (ms) pooled across all (env, episode, agent) groups.
    df must have: env_id, episode_index, agent_id, time_step, emit_eod.
    """
    from cfg import TIME_STEP_MS
    idis = []
    for _, grp in df.groupby(["env_id", "episode_index", "agent_id"]):
        ts = grp.loc[grp["emit_eod"].astype(bool), "time_step"].values
        if len(ts) > 1:
            idis.append(np.diff(np.sort(ts)) * TIME_STEP_MS)
    return np.concatenate(idis) if idis else np.array([])
