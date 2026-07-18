"""
Feature engineering step: takes the flat output of preprocess_flatten.py and
adds candidate variables (distances, food features, event counters, in-range
flags) via get_df_with_candidate_vars.

Usage:
    python preprocess_features.py <flat_pkl>  [--skip_counters] [--force]

  <flat_pkl>   Path to *_agg_flat.pkl (new Recorder format) or
               *_agg_flattened.pkl (old Observer format).

Output: <flat_pkl stem>_per_env_ep_agent_step.pkl written to --output_dir (or alongside input).

This step is separate from preprocess_flatten.py so that format conversion
failures and feature-engineering failures are always visible as distinct errors,
and so feature engineering can be re-run independently without re-flattening.
"""
import argparse
import glob
import os
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

from utils_preprocess import get_df_with_candidate_vars
from eval_fish import read_args_from_file, get_old_cfg_args
import cfg

FISH_CONSTANTS = cfg.FISH_CONSTANTS
OBJECT_TYPES   = cfg.OBJECT_TYPES
ENV_PARAMS     = cfg.ENV_PARAMS
AGENT_PARAMS   = cfg.AGENT_PARAMS
REWARDS        = cfg.REWARDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cfg_for_flat(flat_pkl: Path):
    """Load cfg overrides from the training run associated with flat_pkl."""
    run_dir = flat_pkl.parent
    for _ in range(6):
        if (run_dir / "logs").is_dir():
            break
        run_dir = run_dir.parent
    log_dir = run_dir / "logs"
    if not log_dir.is_dir():
        warnings.warn(
            f"Could not find logs/ directory in any ancestor of {flat_pkl}. "
            "Cfg overrides will NOT be applied — sensing ranges and reward params "
            "will use current cfg.py defaults, which may not match the training run.",
            stacklevel=2,
        )
        return
    try:
        all_args     = read_args_from_file(log_dir)
        env_args     = read_args_from_file(log_dir, "env_args", return_dict=True)
        old_cfg_args = get_old_cfg_args(env_args)
        ENV_PARAMS.update(old_cfg_args["ENV_PARAMS"])
        AGENT_PARAMS.update(old_cfg_args["AGENT_PARAMS"])
        REWARDS.update(old_cfg_args["REWARDS"])
        OBJECT_TYPES.update(old_cfg_args["OBJECT_TYPES"])
        if "FISH_CONSTANTS" in old_cfg_args:
            FISH_CONSTANTS.update(old_cfg_args["FISH_CONSTANTS"])
    except Exception as e:
        warnings.warn(
            f"Could not load run cfg overrides from {log_dir}: {e}. "
            "Sensing ranges and reward params will use current cfg.py defaults.",
            stacklevel=2,
        )


def _apply_features(dff: pd.DataFrame, raw_dir: Path, skip_counters: bool) -> pd.DataFrame:
    """Join food_positions and run feature engineering. Returns processed DataFrame."""
    arena_files = sorted(glob.glob(str(raw_dir / "*_arena.pkl")))
    food_joined = False
    if arena_files and "food_positions" not in dff.columns:
        arena_df = pd.concat([pd.read_pickle(f) for f in arena_files], ignore_index=True)
        if "food_positions" in arena_df.columns:
            food_cols = arena_df[["env_id", "episode_index", "time_step", "food_positions"]]
            dff = dff.merge(food_cols, on=["env_id", "episode_index", "time_step"], how="left")
            food_joined = True
            print(f"  Temporarily joined food_positions from {len(arena_files)} arena file(s).")
    if not food_joined and "food_positions" not in dff.columns:
        print("  Warning: food_positions not available — food features will be skipped.")

    food_detection_range_cm    = AGENT_PARAMS["morm_food_detection_range_m"] * 100
    knollen_detection_range_cm = AGENT_PARAMS["knollen_agent_detection_range_m"] * 100

    dff = get_df_with_candidate_vars(
        dff,
        drop_no_food_rows=False,
        include_counters=not skip_counters,
        food_detection_range_cm=food_detection_range_cm,
        knollen_detection_range_cm=knollen_detection_range_cm,
    )

    if "food_positions" in dff.columns:
        dff.drop(columns=["food_positions"], inplace=True)

    return dff.sort_values(["env_id", "episode_index", "agent_id", "time_step"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(flat_pkl, output_dir=None, skip_counters=False, force=False):
    flat_pkl = Path(flat_pkl).resolve()
    if not flat_pkl.exists():
        raise FileNotFoundError(f"Flat pkl not found: {flat_pkl}")

    out_dir = Path(output_dir) if output_dir else flat_pkl.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    features_pkl = out_dir / "per_env_ep_agent_step.pkl"

    if features_pkl.exists() and not force:
        raise FileExistsError(f"Features pkl already exists (use --force): {features_pkl}")

    _load_cfg_for_flat(flat_pkl)

    print(f"Loading: {flat_pkl}")
    dff = pd.read_pickle(flat_pkl)
    print(f"Loaded {len(dff)} rows, {len(dff.columns)} columns.")

    dff = _apply_features(dff, flat_pkl.parent, skip_counters)

    print(f"Saving → {features_pkl}")
    dff.to_pickle(features_pkl)
    print("Done.")


def run_split(flat_pkls, output_dir, skip_counters=False, force=False):
    """Process each flat pkl separately; write one *_features_step.pkl per input.

    Flat pkls are expected to live in the same raw/ directory (same run).
    Cfg overrides are loaded once from the first file's ancestor logs/ dir.
    Output files are named by replacing _flat.pkl → _features_step.pkl.
    """
    flat_pkls = [Path(f).resolve() for f in flat_pkls]
    if not flat_pkls:
        print("  run_split: no flat pkls provided — skipped")
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = flat_pkls[0].parent

    _load_cfg_for_flat(flat_pkls[0])

    for flat_pkl in flat_pkls:
        stem = flat_pkl.name.replace("_flat.pkl", "")
        out_path = out_dir / f"{stem}_features_step.pkl"
        if out_path.exists() and not force:
            print(f"  [SKIP] {out_path.name} exists")
            continue
        print(f"  Processing {flat_pkl.name} ...")
        dff = pd.read_pickle(flat_pkl)
        dff = _apply_features(dff, raw_dir, skip_counters)
        dff.to_pickle(out_path)
        print(f"  Saved {len(dff)} rows → {out_path.name}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("flat_pkl", help="Path to the flat pkl produced by preprocess_flatten.py")
    parser.add_argument("--output_dir", default=None,
                        help="Directory to write features pkl (default: alongside input)")
    parser.add_argument("--skip_counters", action="store_true", default=False)
    parser.add_argument("--force", action="store_true", default=False,
                        help="Overwrite existing features pkl")
    args = parser.parse_args(argv)
    run(args.flat_pkl, output_dir=args.output_dir, skip_counters=args.skip_counters, force=args.force)


if __name__ == "__main__":
    main()
