"""Homing analysis: trajectory plots, episode outcomes, error-angle rasters, decoding.

Usage:
    python analysis_homing.py --spec_dir path/to/evals/homing

Inputs:
    {spec_dir}/derived/per_env_ep_agent_step.pkl  — preferred (has agent_ids_by_dist etc.)
    {spec_dir}/raw/agg_flat.pkl                   — fallback if derived pkl not yet present

Outputs to {spec_dir}/analyses/homing/
    homing_episode_outcomes.csv
    homing_trajectory_{env_id}_{episode_idx}.pdf  — successful episode trajectories
    {col}_histogram.pdf       — error-angle histograms
    {col}_raster.pdf          — error-angle rasters (timestep-normalized)
    decoding_perf.pdf         — sensor decoding performance

Data layout note:
    agent_id=0 is the stationary target; agent_id=1 is the homing agent.
    Episodes are variable-length: successful homing terminates early
    (MAEFish homing_success_counter >= required_homing_steps); timed-out ones
    run to episode_length. All analysis functions handle this via groupby.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

OUT_SUBDIR = "analyses/homing"


# ---------------------------------------------------------------------------
# Homing-specific preprocessing
# Replicates utils_report.load_flat_pkl_file(task="homing") — keep in sync.
# ---------------------------------------------------------------------------

def _prepare_homing_dff(dff: pd.DataFrame) -> pd.DataFrame:
    """Filter flat pkl to the homing agent and attach target coordinates.

    Steps (must match utils_report.load_flat_pkl_file task='homing'):
    1. Extract position_x/y from the raw position array (added by preprocess_features
       but needed here since we load from raw/agg_flat.pkl directly).
    2. Extract agent 0 positions as target_x / target_y per (env, episode, step).
    3. Keep only agent_id=1 rows (homing agent).
    4. Trim each episode to before the second trial_start (= first homing trial only).
    5. Merge target coordinates back in.
    """
    if "position_x" not in dff.columns:
        dff = dff.copy()
        dff["position_x"] = dff["position"].apply(lambda x: x[0])
        dff["position_y"] = dff["position"].apply(lambda x: x[1])

    target_df = (
        dff.loc[
            dff["agent_id"] == 0,
            ["env_id", "episode_index", "time_step", "position_x", "position_y"],
        ].rename(columns={"position_x": "target_x", "position_y": "target_y"})
    )

    dff = dff[dff["agent_id"] == 1].copy()
    dff = dff.sort_values(["env_id", "agent_id", "episode_index", "time_step"])

    def _keep_until_second_trial_start(group):
        trial_start_indices = group.index[group["trial_start"]].tolist()
        if len(trial_start_indices) < 2:
            return group
        return group.loc[: trial_start_indices[1] - 1]

    dff = dff.groupby(["env_id", "agent_id", "episode_index"], group_keys=False).apply(
        _keep_until_second_trial_start
    )

    dff = pd.merge(dff, target_df, on=["env_id", "episode_index", "time_step"], how="left")
    return dff


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(spec_dir: str) -> None:
    # Prefer features pkl (has agent_ids_by_dist etc. needed by prepare_features_for_decoding).
    # Fall back to raw flat pkl if features step hasn't run yet.
    features_pkl = os.path.join(spec_dir, "derived", "per_env_ep_agent_step.pkl")
    flat_pkl = os.path.join(spec_dir, "raw", "agg_flat.pkl")
    if os.path.exists(features_pkl):
        source_pkl = features_pkl
    elif os.path.exists(flat_pkl):
        source_pkl = flat_pkl
    else:
        raise FileNotFoundError(f"Neither features pkl nor agg_flat.pkl found under {spec_dir}")

    out_dir = os.path.join(spec_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    outfile_prefix = os.path.join(out_dir, "")

    print(f"[analysis_homing] Loading {source_pkl}")
    dff_raw = pd.read_pickle(source_pkl)
    print(f"[analysis_homing] Loaded {len(dff_raw)} rows. Applying homing preprocessing ...")
    dff = _prepare_homing_dff(dff_raw.copy())
    print(f"[analysis_homing] After preprocessing: {len(dff)} rows, "
          f"agent_ids={sorted(dff['agent_id'].unique())}")

    # Keep untrimmed agent-1 rows for RNN decoding: ep*_rnn.npy has T steps for
    # the full episode, but dff is trimmed to the first successful homing trial.
    dff_full = dff_raw[dff_raw["agent_id"] == 1].copy()

    raw_dir = os.path.join(spec_dir, "raw")
    from utils_homing import run_homing_report
    run_homing_report(dff, outfile_prefix=outfile_prefix, raw_dir=raw_dir, dff_full=dff_full)
    print(f"[analysis_homing] Done. Outputs in {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec_dir", required=True,
                   help="Path to evals/{spec_key}/ directory")
    args = p.parse_args(argv)
    run(args.spec_dir)


if __name__ == "__main__":
    main()
