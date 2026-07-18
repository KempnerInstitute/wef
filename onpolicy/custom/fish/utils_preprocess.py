# TODO: redundant computation across below?
#   calculate_nearest_agent_info
#   angle_to_closest_agent

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import cfg
import getpass
import traceback
from contextlib import contextmanager

from features import FEATURE_METADATA, CORE_FEATURES, EVENT_FEATURES
from utils_features import (
    extract_nearest_agent_info,
    add_event_counters,
    add_food_features_fast,
    get_num_agents_in_knollen_range_optimized,
    get_df_with_group_agent_distance_lists
)

FISH_CONSTANTS = cfg.FISH_CONSTANTS
OBJECT_TYPES = cfg.OBJECT_TYPES
ENV_PARAMS = cfg.ENV_PARAMS
AGENT_PARAMS = cfg.AGENT_PARAMS
REWARDS = cfg.REWARDS
COLORS = cfg.COLORS


def get_df_with_candidate_vars(
    dff,
    drop_no_food_rows=True,
    debug=False,
    include_counters=True,
    food_detection_range_cm=None,
    knollen_detection_range_cm=None,
):
    # R3: sensing ranges passed explicitly; fall back to globals with a warning
    if food_detection_range_cm is None:
        food_detection_range_cm = AGENT_PARAMS["morm_food_detection_range_m"] * 100
        print(f"NOTE: food_detection_range_cm not provided; using global AGENT_PARAMS ({food_detection_range_cm:.1f} cm)")
    if knollen_detection_range_cm is None:
        knollen_detection_range_cm = AGENT_PARAMS["knollen_agent_detection_range_m"] * 100
        print(f"NOTE: knollen_detection_range_cm not provided; using global AGENT_PARAMS ({knollen_detection_range_cm:.1f} cm)")

    pre_columns = dff.columns.tolist()
    dff = dff.sort_values(
        by=["env_id", "episode_index", "agent_id", "time_step"]
    ).reset_index(drop=True)

    # Simple features
    print("Adding simple features...")
    # meeting_event fires only on the rising edge of has_nearby (start of each encounter).
    nearby = dff.groupby(["env_id", "episode_index", "agent_id"])["has_nearby"].transform(
        lambda s: s.astype(bool) & ~s.astype(bool).shift(1, fill_value=False)
    )
    dff["meeting_event"] = nearby
    dff["position_x"] = dff["position"].apply(lambda x: x[0])
    dff["position_y"] = dff["position"].apply(lambda x: x[1])
    dff = add_displacement_components(dff)

    # Nearest agent distances and sizes
    print("Adding distance-sorted lists of distances, sizes, ids, emit eods, angles to all agents...")
    dff = get_df_with_group_agent_distance_lists(dff)

    print("Extracting nearest agent distances, sizes, ids, and angles...")
    dff = extract_nearest_agent_info(dff)

    print("Adding angle to closest *observed* agent...")
    dff["angle_to_closest_agent_observed"] = np.where(
        dff["has_nearby"],
        dff["angle_to_closest_agent"],
        np.nan
    )

    # R4: food_positions is joined transiently by preprocess_features.py from arena.pkl.
    # Just check it's present — no extract-from-agent_0 needed.
    print("Adding food related features...")
    if "food_positions" not in dff.columns:
        print("WARNING: food_positions not in DataFrame — skipping food features")
        merged_df = dff
    else:
        merged_df = dff

        if debug:
            print(
                "Food position matrix, unique shapes (before dropping broken rows):",
                pd.Series([np.array(fp).shape for fp in merged_df["food_positions"]]).unique(),
            )

        if drop_no_food_rows:
            print("Dropping rows with no food positions...")
            rows_without_food = merged_df[
                merged_df["food_positions"].apply(
                    lambda x: np.array(x).shape == () or np.array(x).shape[0] == 0
                )
            ]
            before_rows = merged_df.shape[0]
            merged_df = merged_df.drop(rows_without_food.index)
            after_rows = merged_df.shape[0]
            print(f"Dropped {before_rows - after_rows} rows with no food. Remaining: {after_rows}")

        print("Calculating food features (distance, angle, quadrant counts)...")
        merged_df = add_food_features_fast(merged_df)

    # R7: derive event columns from FEATURE_METADATA registry
    event_columns = [col for col in EVENT_FEATURES if col in merged_df.columns]
    # include any suffix-detected columns not yet in the registry (forward compat)
    for col in merged_df.columns:
        if col not in event_columns and (col.endswith("_event") or col.endswith("_observed")):
            if col != "angle_to_closest_agent_observed":
                event_columns.append(col)

    if include_counters:
        print("Adding counters for event columns:", event_columns)
        merged_df, event_counter_colnames = add_event_counters(
            merged_df, event_columns=event_columns
        )
    else:
        print("Skipping event counters (include_counters=False)")

    ################### IN-RANGE FLAGS ###################
    print("Adding in-range flags for electric sensing...")
    print(f"  Mormyromast food detection range: {food_detection_range_cm:.1f} cm")
    merged_df['has_food_in_food_sensing_range'] = (
        merged_df['distance_to_closest_food'] < food_detection_range_cm
    )
    merged_df['food_observed'] = merged_df['has_food_in_food_sensing_range']

    print(f"  Knollen agent detection range: {knollen_detection_range_cm:.1f} cm")
    merged_df['num_agents_in_knollen_range'] = get_num_agents_in_knollen_range_optimized(
        merged_df, knollen_range=knollen_detection_range_cm
    )
    merged_df['has_agent_in_knollen_range'] = merged_df['num_agents_in_knollen_range'] > 0

    ################### R1 + R2: NaN check from registry, no -1 fill ###################
    print("Checking core features for NaN values...")
    candidate_variables = [col for col in CORE_FEATURES if col in merged_df.columns]
    missing = [col for col in CORE_FEATURES if col not in merged_df.columns]
    if missing:
        print(f"WARNING: core features not found in DataFrame (will be absent downstream): {missing}")

    for col in candidate_variables:
        if merged_df[col].dtype not in ('float64', 'int64', 'bool'):
            continue
        pct_na = merged_df[col].isna().mean() * 100
        if pct_na > 0.0:
            print(f"NOTE: {col} has {pct_na:.1f}% NaN — leaving as NaN")

    print("Final DataFrame shape:", merged_df.shape)
    post_columns = merged_df.columns.tolist()
    print("Columns added:", set(post_columns) - set(pre_columns))

    return merged_df


def add_displacement_components(dff, group_cols=None):
    # displacement_ground is already logged per step; extract x/y directly.
    # group_cols kept for API compatibility but is no longer used.
    if "displacement_ground" in dff.columns:
        dff["displacement_x"] = dff["displacement_ground"].apply(
            lambda v: float(v[0]) if isinstance(v, (np.ndarray, list)) and len(v) >= 2 else np.nan
        )
        dff["displacement_y"] = dff["displacement_ground"].apply(
            lambda v: float(v[1]) if isinstance(v, (np.ndarray, list)) and len(v) >= 2 else np.nan
        )
    else:
        if group_cols is None:
            group_cols = ["env_id", "episode_index", "agent_id"]
        dff["displacement_x"] = dff.groupby(group_cols)["position_x"].diff()
        dff["displacement_y"] = dff.groupby(group_cols)["position_y"].diff()
    return dff


if __name__ == "__main__":
    test_pkl = f"/srv/marl/{getpass.getuser()}/marl_fish/WD0.000001_Attn2Mxhx/20250526_174244/outputs/MAFish_neural_20250526_174244_dO8B1noY_agg_flattened.pkl"
    test_pkl = f"/srv/marl/{getpass.getuser()}/marl_fish/short/20250310_152603/outputs/MAFish_neural_20250310_152603_rHBdRtOi_agg_flattened.pkl"
    print("Testing with file:", test_pkl)
    dff = pd.read_pickle(test_pkl)
    print("Loaded DataFrame with shape:", dff.shape)
    dff = get_df_with_candidate_vars(dff)
