"""
Flatten raw eval output into a single per-(env, episode, agent, step) DataFrame.

Expects new Recorder bundle format: outputs/*_behavior.pkl

Usage:
    python preprocess_flatten.py <run_dir> [--delete_raw] [--force]
"""

import glob
import json
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from eval_fish import read_args_from_file, get_old_cfg_args
import cfg

FISH_CONSTANTS = cfg.FISH_CONSTANTS
OBJECT_TYPES   = cfg.OBJECT_TYPES
ENV_PARAMS     = cfg.ENV_PARAMS
AGENT_PARAMS   = cfg.AGENT_PARAMS
REWARDS        = cfg.REWARDS
COLORS         = cfg.COLORS


# ---------------------------------------------------------------------------
# New-format loader (Recorder bundle: *_behavior.pkl)
# ---------------------------------------------------------------------------

def _load_new_format(outputs_folder):
    """Load and merge Recorder bundle files; return flat DataFrame."""
    behavior_files = sorted(glob.glob(os.path.join(outputs_folder, "*_behavior.pkl")))
    episode_files  = sorted(glob.glob(os.path.join(outputs_folder, "*_episodes.json")))

    print(f"Behavior pkls ({len(behavior_files)}): {[os.path.basename(f) for f in behavior_files]}")

    dff = pd.concat([pd.read_pickle(f) for f in behavior_files], ignore_index=True)
    print(f"Loaded {len(dff)} behavior rows.")

    episodes = []
    for ef in episode_files:
        with open(ef) as fp:
            episodes.extend(json.load(fp))
    episodes_df = pd.DataFrame(episodes) if episodes else pd.DataFrame()

    return dff, episodes_df, behavior_files


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def _process_df(dff, episodes_df, active_agent_ids):
    """Apply all per-row transformations to a flat DataFrame. Returns processed df."""
    # Cast list columns to numpy
    array_cols = ["actions", "position",
                  "actions_postactivation", "displacement_ground", "displacement_ego"]
    for col in array_cols:
        if col in dff.columns:
            dff[col] = dff[col].apply(lambda x: np.array(x) if isinstance(x, list) else x)

    # R5/S2: expand center_field dict → named columns; drop the opaque dict column.
    if "center_field" in dff.columns:
        for src_key in ("mormyromast", "ampullary", "knollen"):
            dff[f"center_field.{src_key}"] = dff["center_field"].apply(
                lambda x, k=src_key: np.array(x[k]) if isinstance(x, dict) and k in x else None)
        dff.drop(columns=["center_field"], inplace=True)

    # R8: join arena_size, arena_type, and rw_* columns from episodes.json;
    # set trial_start = first timestep
    if not episodes_df.empty:
        ep_keys = ["env_id", "episode_index"]
        ep_idx = episodes_df.set_index(ep_keys)
        _passthrough = ["arena_size", "arena_type"] + [c for c in ep_idx.columns if c.startswith("rw_")]
        for _ep_col in _passthrough:
            if _ep_col in ep_idx.columns:
                dff[_ep_col] = dff.set_index(ep_keys).index.map(ep_idx[_ep_col]).values
    if "curr_time" in dff.columns:
        dff["trial_start"] = dff["curr_time"] == 0
    else:
        dff["trial_start"] = dff["time_step"] == 0

    dff["episode_index"] = pd.to_numeric(dff["episode_index"])
    dff["time_step"]     = pd.to_numeric(dff["time_step"])
    dff = dff.sort_values(["env_id", "episode_index", "agent_id", "time_step"]).reset_index(drop=True)

    # R6: derive action columns from actions_postactivation.
    # emit_eod is now logged directly in infos (MAEFish step()) so it arrives
    # as its own column and is not derived here.
    ACTION_NAMES    = ["move_forward", "turn_angle", "bite_action"]
    ACTION_INDICES  = [0,              1,             3            ]
    ACTION_TYPES    = [float,          float,         bool         ]
    ACTION_DEFAULTS = [None,           None,          False        ]
    for i, (name, typ, default) in enumerate(zip(ACTION_NAMES, ACTION_TYPES, ACTION_DEFAULTS)):
        idx = ACTION_INDICES[i]
        dff[name] = dff["actions_postactivation"].apply(
            lambda x, idx=idx, typ=typ, d=default:
                typ(x[idx]) if isinstance(x, (np.ndarray, list)) and len(x) > idx else d
        )
    n_actions = dff["actions_postactivation"].apply(
        lambda x: len(x) if isinstance(x, (np.ndarray, list)) else 0).max()
    if n_actions < 4:
        print(f"NOTE: action vector has {n_actions} elements — bite_action defaulted to False")

    dff["eating_event"] = dff["food_eaten_count"] > 0

    # M2: displacement_ground is already logged in infos; use it directly.
    if "displacement_ground" in dff.columns:
        dff["displacement"] = dff["displacement_ground"].apply(
            lambda x: np.linalg.norm(x) if isinstance(x, np.ndarray) else np.nan)

    max_turn_angle = AGENT_PARAMS["max_angular_velocity_baseline"]
    dff["actual_turn"] = dff["turn_angle"] * max_turn_angle

    # Filter inactive agents
    if active_agent_ids is not None:
        inactive = set(dff["agent_id"].unique()) - set(active_agent_ids)
        if inactive:
            print(f"Dropping inactive agent rows: {inactive}")
        dff = dff[dff["agent_id"].isin(active_agent_ids)]

    return dff.sort_values(["env_id", "episode_index", "agent_id", "time_step"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(run_dir, outputs_folder=None, force=False, delete_raw=False, no_aggregate=False):
    outputs_folder = outputs_folder or os.path.join(run_dir, "outputs")
    run_dir  = Path(run_dir)
    log_dir  = run_dir / "logs"
    print(f"Outputs: {outputs_folder}")
    print(f"Run dir: {run_dir}")

    # Load cfg overrides from the run
    all_args     = read_args_from_file(log_dir)
    env_args     = read_args_from_file(log_dir, "env_args", return_dict=True)
    old_cfg_args = get_old_cfg_args(env_args)
    ENV_PARAMS.update(old_cfg_args["ENV_PARAMS"])
    AGENT_PARAMS.update(old_cfg_args["AGENT_PARAMS"])
    REWARDS.update(old_cfg_args["REWARDS"])
    OBJECT_TYPES.update(old_cfg_args["OBJECT_TYPES"])
    if "FISH_CONSTANTS" in old_cfg_args:
        FISH_CONSTANTS.update(old_cfg_args["FISH_CONSTANTS"])

    # ---------------------------------------------------------------------------
    # Load
    # ---------------------------------------------------------------------------
    behavior_files = sorted(glob.glob(os.path.join(outputs_folder, "*_behavior.pkl")))
    if not behavior_files:
        raise FileNotFoundError(f"No *_behavior.pkl found in {outputs_folder}")

    if no_aggregate:
        out_paths = [
            os.path.join(outputs_folder,
                         os.path.basename(f).replace("_behavior.pkl", "_flat.pkl"))
            for f in behavior_files
        ]
        if all(os.path.exists(p) for p in out_paths) and not force:
            raise FileExistsError(
                f"Split flat pkls already exist (use --force): {outputs_folder}")
    else:
        flattened_pkl_file = os.path.join(outputs_folder, "agg_flat.pkl")
        if os.path.exists(flattened_pkl_file) and not force:
            raise FileExistsError(
                f"Flat pkl already exists (use --force): {flattened_pkl_file}")
        print(f"Output: {flattened_pkl_file}")

    # Load episodes metadata
    episode_files = sorted(glob.glob(os.path.join(outputs_folder, "*_episodes.json")))
    episodes = []
    for ef in episode_files:
        with open(ef) as fp:
            episodes.extend(json.load(fp))
    episodes_df = pd.DataFrame(episodes) if episodes else pd.DataFrame()

    # Resolve active_agent_ids from episodes metadata
    active_agent_ids = None
    if not episodes_df.empty and "active_agent_ids" in episodes_df.columns:
        active_agent_ids = episodes_df["active_agent_ids"].iloc[0]

    # ---------------------------------------------------------------------------
    # Process and save
    # ---------------------------------------------------------------------------
    if no_aggregate:
        for f, out_path in zip(behavior_files, out_paths):
            print(f"Processing {os.path.basename(f)} → {os.path.basename(out_path)}")
            df = pd.read_pickle(f)
            df = _process_df(df, episodes_df, active_agent_ids)
            df.to_pickle(out_path)
            print(f"  Saved {len(df)} rows → {out_path}")
    else:
        print(f"Behavior pkls ({len(behavior_files)}): "
              f"{[os.path.basename(f) for f in behavior_files]}")
        dff = pd.concat([pd.read_pickle(f) for f in behavior_files], ignore_index=True)
        print(f"Loaded {len(dff)} behavior rows.")
        dff = _process_df(dff, episodes_df, active_agent_ids)
        print(f"Saving → {flattened_pkl_file}")
        dff.to_pickle(flattened_pkl_file)

    print("Done.")

    if delete_raw:
        for p in behavior_files:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
        print("Deleted raw input files.")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--outputs_folder", default=None,
                        help="Override the outputs folder (default: run_dir/outputs)")
    parser.add_argument("--force",          action="store_true", default=False)
    parser.add_argument("--delete_raw",     action="store_true", default=False)
    parser.add_argument("--no-aggregate",   action="store_true", default=False,
                        help="Save one *_flat.pkl per input behavior pkl instead of agg_flat.pkl")
    args = parser.parse_args(argv)
    run(args.run_dir, outputs_folder=args.outputs_folder, force=args.force,
        delete_raw=args.delete_raw, no_aggregate=args.no_aggregate)


if __name__ == "__main__":
    main()
