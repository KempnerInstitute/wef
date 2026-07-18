"""
Compute aggregated summaries from a *_per_env_ep_agent_step.pkl and save three files:

  *_per_env_ep_agent.pkl  — one row per (env_id, episode_index, agent_id)
  *_per_env_ep_step.pkl   — one row per (env_id, episode_index, time_step), agents collapsed
  *_per_env_ep.pkl        — one row per (env_id, episode_index), agents+steps collapsed

All report scripts and comparison scripts should load these instead of re-aggregating
from the full step-level pkl.

Usage:
    python preprocess_summaries.py <features_pkl> [--force]

<features_pkl>  Path to *_per_env_ep_agent_step.pkl produced by preprocess_features.py.
"""
import argparse
from pathlib import Path

import pandas as pd

from utils_stats_general import build_agent_run_df, build_run_df, build_step_df


def _derive_output_paths(features_pkl: Path, out_dir: Path = None):
    parent = out_dir if out_dir else features_pkl.parent
    return (
        parent / "per_env_ep_agent.pkl",
        parent / "per_env_ep_step.pkl",
        parent / "per_env_ep.pkl",
    )


def run_split(features_pkls, output_dir, force=False):
    """Concat split *_features_step.pkl files and produce summary pkls in output_dir."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    agent_pkl = out_dir / "per_env_ep_agent.pkl"
    step_pkl  = out_dir / "per_env_ep_step.pkl"
    run_pkl   = out_dir / "per_env_ep.pkl"

    if not force:
        existing = [p for p in (agent_pkl, step_pkl, run_pkl) if p.exists()]
        if existing:
            raise FileExistsError(
                f"Summary pkls already exist (use --force): {[str(p) for p in existing]}"
            )

    print(f"Concatenating {len(features_pkls)} split features pkls ...")
    dff = pd.concat([pd.read_pickle(f) for f in features_pkls], ignore_index=True)
    print(f"  {len(dff):,} rows, {len(dff.columns)} columns")

    print("Building per-(env, episode, agent) summary ...")
    df_agent = build_agent_run_df(dff)
    print(f"  → {len(df_agent):,} rows")
    df_agent.to_pickle(agent_pkl)
    print(f"Saving → {agent_pkl}")

    print("Building per-(env, episode, time_step) summary ...")
    df_step = build_step_df(dff)
    print(f"  → {len(df_step):,} rows")
    df_step.to_pickle(step_pkl)
    print(f"Saving → {step_pkl}")

    print("Building per-(env, episode) summary ...")
    df_run = build_run_df(dff)
    print(f"  → {len(df_run):,} rows")
    df_run.to_pickle(run_pkl)
    print(f"Saving → {run_pkl}")

    print("Done.")


def run(features_pkl, output_dir=None, force=False):
    features_pkl = Path(features_pkl).resolve()
    if not features_pkl.exists():
        raise FileNotFoundError(f"Features pkl not found: {features_pkl}")

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = features_pkl.parent

    agent_pkl, step_pkl, run_pkl = _derive_output_paths(features_pkl, out_dir)

    if not force:
        existing = [p for p in (agent_pkl, step_pkl, run_pkl) if p.exists()]
        if existing:
            raise FileExistsError(
                f"Summary pkls already exist (use --force): {[str(p) for p in existing]}"
            )

    print(f"Loading: {features_pkl}")
    dff = pd.read_pickle(features_pkl)
    print(f"  {len(dff):,} rows, {len(dff.columns)} columns")

    print("Building per-(env, episode, agent) summary ...")
    df_agent = build_agent_run_df(dff)
    print(f"  → {len(df_agent):,} rows")
    print(f"Saving → {agent_pkl}")
    df_agent.to_pickle(agent_pkl)

    print("Building per-(env, episode, time_step) summary ...")
    df_step = build_step_df(dff)
    print(f"  → {len(df_step):,} rows")
    print(f"Saving → {step_pkl}")
    df_step.to_pickle(step_pkl)

    print("Building per-(env, episode) summary ...")
    df_run = build_run_df(dff)
    print(f"  → {len(df_run):,} rows")
    print(f"Saving → {run_pkl}")
    df_run.to_pickle(run_pkl)

    print("Done.")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("features_pkl", help="Path to *_per_env_ep_agent_step.pkl")
    parser.add_argument("--output_dir", default=None,
                        help="Directory to write summary pkls (default: alongside input)")
    parser.add_argument("--force", action="store_true", default=False,
                        help="Overwrite existing summary pkls")
    args = parser.parse_args(argv)
    run(args.features_pkl, output_dir=args.output_dir, force=args.force)


if __name__ == "__main__":
    main()
