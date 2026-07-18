import argparse
import os
import sys; sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import utils_report as ru
from utils_behavior import plot_agent_trajectories_together
import tqdm


def str2bool(v):
    if isinstance(v, bool):
        return v
    val = str(v).strip().lower()
    if val in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value.")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Plot agent trajectories for episodes in a flattened pkl using "
            "utils_behavior.plot_agent_trajectories_together()."
        )
    )
    parser.add_argument(
        "--flat_pkl_file",
        "--fpkl",
        dest="flat_pkl_file",
        type=str,
        required=True,
        help="Path to flattened pickle file.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="foraging",
        help="Task used by load_flat_pkl_file (foraging, 2f1p, homing, ...).",
    )
    parser.add_argument(
        "--env_ids",
        nargs="+",
        type=int,
        default=None,
        help="Optional env_id filter list. Default: all envs in file.",
    )
    parser.add_argument(
        "--episode_indices",
        nargs="+",
        type=int,
        default=None,
        help="Optional episode_index filter list. Default: all episodes per env.",
    )
    parser.add_argument(
        "--max_episodes_per_env",
        type=int,
        default=None,
        help="Optional cap on number of episodes plotted per env.",
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default="behavior",
        help="Subfolder under outputs folder for trajectory plots.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Explicit output directory. Overrides --output_subdir if set.",
    )
    parser.add_argument(
        "--no_mark_eating",
        action="store_true",
        help="Disable red X markers for eating events.",
    )
    parser.add_argument(
        "--csvs",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help=(
            "Whether to generate summary CSVs (*_run_summary.csv and "
            "*_run_agent_summary.csv). Default: true."
        ),
    )
    return parser.parse_args(argv)


def get_target_location_for_episode(subset):
    if "target_x" not in subset.columns or "target_y" not in subset.columns:
        return None

    target_rows = subset[["target_x", "target_y"]].dropna()
    if target_rows.empty:
        return None

    row = target_rows.iloc[0]
    return [row["target_x"], row["target_y"]]


def iter_env_episode_pairs(dff, env_ids=None, episode_indices=None, max_episodes_per_env=None):
    all_env_ids = sorted(dff["env_id"].dropna().unique().tolist())
    selected_env_ids = all_env_ids if env_ids is None else [eid for eid in env_ids if eid in all_env_ids]

    for env_id in selected_env_ids:
        env_subset = dff[dff["env_id"] == env_id]
        env_episode_indices = sorted(env_subset["episode_index"].dropna().unique().tolist())

        if episode_indices is not None:
            env_episode_indices = [ep for ep in env_episode_indices if ep in episode_indices]

        if max_episodes_per_env is not None:
            env_episode_indices = env_episode_indices[:max_episodes_per_env]

        for episode_index in env_episode_indices:
            yield env_id, episode_index


def save_summary_csvs(dff, output_dir, output_base):
    from utils_stats_general import get_summary_dfs

    summary_agent_df, summary_run_df = get_summary_dfs(dff)
    run_summary_path = os.path.join(output_dir, f"{output_base}_run_summary.csv")
    run_agent_summary_path = os.path.join(output_dir, f"{output_base}_run_agent_summary.csv")
    summary_run_df.to_csv(run_summary_path, sep="\t", index=False)
    summary_agent_df.to_csv(run_agent_summary_path, sep="\t", index=False)
    print(f"--> Wrote: {run_summary_path}")
    print(f"--> Wrote: {run_agent_summary_path}")


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)

    dff, outputs_folder, pkl_str = ru.load_flat_pkl_file(args.flat_pkl_file, task=args.task)
    print(f"Loaded data with {len(dff)} rows from {args.flat_pkl_file}")

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(outputs_folder, args.output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"pkl_str: {pkl_str}")

    if args.csvs:
        save_summary_csvs(dff, output_dir, pkl_str)

    pairs = list(
        iter_env_episode_pairs(
            dff,
            env_ids=args.env_ids,
            episode_indices=args.episode_indices,
            max_episodes_per_env=args.max_episodes_per_env,
        )
    )

    if not pairs:
        print("No matching (env_id, episode_index) pairs found. Nothing to plot.")
        return

    plotted = 0
    for env_id, episode_index in tqdm.tqdm(pairs, desc="Plotting episodes"):
        subset = dff[(dff["env_id"] == env_id) & (dff["episode_index"] == episode_index)]
        if subset.empty:
            continue

        target_location = get_target_location_for_episode(subset)

        print(f"Plotting env {env_id}, episode {episode_index}")
        plot_agent_trajectories_together(
            dff,
            env_id=env_id,
            episode_idx=episode_index,
            mark_eating=not args.no_mark_eating,
            output_dir=output_dir,
            output_str=pkl_str,
            target_location=target_location,
        )
        plotted += 1

    print(f"Done. Plotted {plotted} episode trajectories.")


if __name__ == "__main__":
    main()
