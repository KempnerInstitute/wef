"""
Derive per-episode event-time arrays from per_env_ep_agent_step.pkl and save
as derived/event_times.pkl.  This small file (~KB) is loaded by
load_consumption_times / load_biting_times in utils_event_times.py, avoiding
repeated reads of the 186–300 MB step pkl during analysis.

Usage (standalone):
    python preprocess_event_times.py <spec_dir>

Pipeline:
    Called from pipeline.py after summaries, once per spec.
"""

import os
import pickle
import sys

import pandas as pd

from utils_event_times import consumption_times_from_step_df, biting_times_from_step_df


OUTPUT_FILENAME = "event_times.pkl"


def run(spec_dir, force=False):
    step_pkl = os.path.join(spec_dir, "derived", "per_env_ep_agent_step.pkl")
    out_path = os.path.join(spec_dir, "derived", OUTPUT_FILENAME)

    if not os.path.exists(step_pkl):
        print(f"  [SKIP] {spec_dir}: no per_env_ep_agent_step.pkl")
        return

    if os.path.exists(out_path) and not force:
        print(f"  [SKIP] event_times.pkl already exists")
        return

    keep = ["env_id", "episode_index", "time_step", "eating_event", "was_bitten"]
    df_full = pd.read_pickle(step_pkl)
    df = df_full[[c for c in keep if c in df_full.columns]]
    del df_full

    def _ep_keys(event_col):
        """Sorted (env_id, episode_index) for episodes with ≥1 event — parallel to event arrays."""
        if event_col not in df.columns:
            return []
        events = df[df[event_col].astype(bool)][["env_id", "episode_index"]]
        return list(events.groupby(["env_id", "episode_index"]).groups.keys())

    ct = consumption_times_from_step_df(df)
    bt = biting_times_from_step_df(df)
    data = {
        "consumption_times":   ct,
        "biting_times":        bt,
        "consumption_ep_keys": _ep_keys("eating_event"),
        "biting_ep_keys":      _ep_keys("was_bitten"),
    }

    with open(out_path, "wb") as f:
        pickle.dump(data, f, protocol=4)

    print(f"  [DONE] event_times.pkl  "
          f"({len(data['consumption_times'])} consumption, "
          f"{len(data['biting_times'])} biting episodes)")


def run_split(features_pkls, spec_dir, force=False):
    """Produce event_times.pkl by concatenating split *_features_step.pkl files."""
    out_path = os.path.join(spec_dir, "derived", OUTPUT_FILENAME)
    if os.path.exists(out_path) and not force:
        print(f"  [SKIP] event_times.pkl already exists")
        return

    keep = ["env_id", "episode_index", "time_step", "eating_event", "was_bitten"]
    frames = []
    for f in features_pkls:
        df = pd.read_pickle(f)
        frames.append(df[[c for c in keep if c in df.columns]])
    df = pd.concat(frames, ignore_index=True)

    def _ep_keys(event_col):
        if event_col not in df.columns:
            return []
        events = df[df[event_col].astype(bool)][["env_id", "episode_index"]]
        return list(events.groupby(["env_id", "episode_index"]).groups.keys())

    ct = consumption_times_from_step_df(df)
    bt = biting_times_from_step_df(df)
    data = {
        "consumption_times":   ct,
        "biting_times":        bt,
        "consumption_ep_keys": _ep_keys("eating_event"),
        "biting_ep_keys":      _ep_keys("was_bitten"),
    }

    with open(out_path, "wb") as f:
        pickle.dump(data, f, protocol=4)

    print(f"  [DONE] event_times.pkl  "
          f"({len(data['consumption_times'])} consumption, "
          f"{len(data['biting_times'])} biting episodes)")


def run_all(run_dir, force=False):
    """Process all spec dirs under run_dir/evals/ in one Python process."""
    evals_dir = os.path.join(run_dir, "evals")
    if not os.path.isdir(evals_dir):
        print(f"  [SKIP] no evals/ dir under {run_dir}")
        return
    for name in sorted(os.listdir(evals_dir)):
        spec_dir = os.path.join(evals_dir, name)
        if os.path.isdir(spec_dir):
            run(spec_dir, force=force)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python preprocess_event_times.py <spec_dir_or_run_dir> [--force]\n"
                 "  Pass a run_dir (containing evals/) to process all specs at once.")
    path = sys.argv[1]
    force = "--force" in sys.argv
    if os.path.isdir(os.path.join(path, "evals")):
        run_all(path, force=force)
    else:
        run(path, force=force)
