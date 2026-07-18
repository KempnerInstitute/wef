"""
Batch seed ranking across all completed foraging experiment groups.
Reads from both cluster mount and local results.

Usage:
    cd onpolicy/custom/fish/notebooks
    python rank_all_groups.py
"""

import sys
import getpass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import utils_seeds as us

_USER = getpass.getuser()
RESULTS_ROOTS = [
    Path(f"/home/{_USER}/cluster_lab/{_USER}/marl_fish_storage/results"),
    Path(f"/home/{_USER}/kr/mfrefactor/onpolicy/custom/fish/results"),
]

MAIN_EVAL = "m1a1k1_patchy_square"
LAST_N = 5

# Only rank foraging groups (not homing)
GROUP_SUFFIX_FILTER = "_For"


def decode_group(name):
    """Return a short human-readable label from abbreviated group name."""
    parts = name.split("_")
    # e.g. 20260611_Dyn_F00_Kb_For
    # idx:  0         1    2    3    4
    model   = parts[1] if len(parts) > 1 else "?"
    mfo     = f"MFO={parts[2][1:]}" if len(parts) > 2 else "?"
    knollen = parts[3] if len(parts) > 3 else "?"
    return f"{model}/{mfo}/{knollen}"


def rank_group(results_dir, group_name):
    run_dirs = us.discover_run_dirs(results_dir, group_name)
    if not run_dirs:
        return None, None
    df = us.load_ranking_df(run_dirs, main_eval=MAIN_EVAL, last_n=LAST_N)
    # Filter to seeds that have OLS data (else rank is meaningless)
    existing = [c for c in ["size_food_slope", "food_mean"] if c in df.columns]
    if not existing:
        return None, None
    df = df.dropna(subset=existing)
    if df.empty:
        return None, None
    df = us.rank_seeds(df)
    return df, run_dirs


def main():
    all_results = []

    for root in RESULTS_ROOTS:
        if not root.is_dir():
            continue
        groups = sorted(d.name for d in root.iterdir()
                        if d.is_dir() and GROUP_SUFFIX_FILTER in d.name)
        for group_name in groups:
            label = f"[{root.name}] {group_name}"
            df, run_dirs = rank_group(root, group_name)
            if df is None:
                print(f"{label}: no ranked seeds (OLS missing?)")
                continue

            best = df.sort_values("combined_rank").iloc[0]
            runner_up = df.sort_values("combined_rank").iloc[1] if len(df) > 1 else None

            all_results.append({
                "group": group_name,
                "root": str(root),
                "n_seeds": len(df),
                "best_seed": int(best["seed"]),
                "best_slope": best["size_food_slope"],
                "best_food": best["food_mean"],
                "best_r2": best["size_food_r2"],
                "best_rank": int(best["combined_rank"]),
                "runner_up_seed": int(runner_up["seed"]) if runner_up is not None else None,
            })

            print(f"\n{'='*72}")
            print(f"Group : {group_name}  ({decode_group(group_name)})")
            print(f"Root  : {root}")
            print(f"Seeds : {len(df)}  ranked")
            print()
            header = f"{'Seed':>5}  {'slope':>8}  {'R²':>6}  {'food':>7}  {'rank':>5}"
            print(header)
            print("-" * len(header))
            for _, row in df.sort_values("combined_rank").iterrows():
                marker = " <-- BEST" if row["combined_rank"] == df["combined_rank"].min() else ""
                print(f"  {int(row['seed']):3d}  {row['size_food_slope']:+8.3f}  "
                      f"{row['size_food_r2']:6.3f}  {row['food_mean']:7.2f}  "
                      f"{int(row['combined_rank']):5d}{marker}")

    # Summary table
    if all_results:
        import pandas as pd
        summary = pd.DataFrame(all_results)
        print("\n\n" + "=" * 72)
        print("SUMMARY — Best seed per group")
        print("=" * 72)
        print(summary[["group", "n_seeds", "best_seed", "best_slope", "best_food",
                        "best_r2", "runner_up_seed"]].to_string(index=False))


if __name__ == "__main__":
    main()
