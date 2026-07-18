"""
Compare three seed ranking schemes across all foraging groups.

  (1) current:  rank_food + rank_slope + rank_r2      (equal weight)
  (2) food:     rank_food                              (food only)
  (3) balanced: rank_food + 0.5*rank_slope + 0.5*rank_r2

Prints per-group top-seed comparison and a summary diff vs. scheme (1).
"""

import sys
import getpass
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import utils_seeds as us

_USER = getpass.getuser()
RESULTS_ROOTS = [
    Path(f"/home/{_USER}/cluster_lab/{_USER}/marl_fish_storage/results"),
]
MAIN_EVAL = "m1a1k1_patchy_square"
GROUP_SUFFIX_FILTER = "_For"
LAST_N = 5

SCHEMES = {
    "1_full":     {"size_food_slope": 1.0, "food_mean": 1.0, "size_food_r2": 1.0},
    "2_food":     {"size_food_slope": 0.0, "food_mean": 1.0, "size_food_r2": 0.0},
    "3_balanced": {"size_food_slope": 0.5, "food_mean": 1.0, "size_food_r2": 0.5},
}


def rank_weighted(df, weights):
    """Return df with combined_rank using per-metric weights (0 = excluded)."""
    df = df.copy()
    combined = pd.Series(0.0, index=df.index)
    for col, w in weights.items():
        if w == 0 or col not in df.columns:
            continue
        r = df[col].rank(method="min", ascending=False)  # higher = better
        combined += w * r
    df["combined_rank"] = combined
    return df


def best_seed(df):
    row = df.loc[df["combined_rank"].idxmin()]
    return int(row["seed"]), row["food_mean"], row["size_food_slope"], row["size_food_r2"]


def main():
    all_rows = []

    for root in RESULTS_ROOTS:
        if not root.is_dir():
            continue
        groups = sorted(d.name for d in root.iterdir()
                        if d.is_dir() and GROUP_SUFFIX_FILTER in d.name)

        for grp in groups:
            run_dirs = us.discover_run_dirs(root, grp)
            if not run_dirs:
                continue
            raw = us.load_ranking_df(run_dirs, main_eval=MAIN_EVAL, last_n=LAST_N)
            raw = raw.dropna(subset=[c for c in ["size_food_slope", "food_mean", "size_food_r2"]
                                     if c in raw.columns])
            if raw.empty:
                continue

            results = {}
            for scheme_name, weights in SCHEMES.items():
                df_r = rank_weighted(raw, weights)
                seed, food, slope, r2 = best_seed(df_r)
                results[scheme_name] = {"seed": seed, "food": food, "slope": slope, "r2": r2}

            # Detect disagreements vs. scheme 1
            s1_seed = results["1_full"]["seed"]
            diffs = {k: v["seed"] for k, v in results.items() if v["seed"] != s1_seed}

            row = {"group": grp, "n_seeds": len(raw)}
            for k, v in results.items():
                row[f"{k}_seed"] = v["seed"]
                row[f"{k}_food"] = v["food"]
                row[f"{k}_slope"] = v["slope"]
            row["any_diff"] = bool(diffs)
            all_rows.append(row)

            # Per-group detail
            print(f"\n{'='*70}")
            print(f"Group: {grp}  ({len(raw)} seeds ranked)")
            print(f"  {'Scheme':<12}  {'Best':>4}  {'food':>6}  {'slope':>8}  {'r2':>6}  note")
            print(f"  {'-'*58}")
            for k, v in results.items():
                diff_note = "" if v["seed"] == s1_seed else f"  <- differs from (1)"
                print(f"  {k:<12}  S{v['seed']:2d}  {v['food']:6.2f}  {v['slope']:+8.3f}"
                      f"  {v['r2']:6.3f}{diff_note}")

    # Summary table
    df_summary = pd.DataFrame(all_rows)
    n_groups = len(df_summary)
    n_diff_2 = (df_summary["1_full_seed"] != df_summary["2_food_seed"]).sum()
    n_diff_3 = (df_summary["1_full_seed"] != df_summary["3_balanced_seed"]).sum()

    print(f"\n\n{'='*70}")
    print("SUMMARY — best seed per scheme")
    print(f"{'='*70}")
    cols = ["group", "n_seeds", "1_full_seed", "2_food_seed", "3_balanced_seed", "any_diff"]
    print(df_summary[cols].to_string(index=False))
    print(f"\nGroups where (2) food-only disagrees with (1): {n_diff_2}/{n_groups}")
    print(f"Groups where (3) balanced disagrees with (1):  {n_diff_3}/{n_groups}")

    # Show the differing groups in detail
    diff_rows = df_summary[df_summary["any_diff"]]
    if not diff_rows.empty:
        print(f"\nDisagreeing groups:")
        for _, r in diff_rows.iterrows():
            print(f"  {r['group']}:")
            if r["1_full_seed"] != r["2_food_seed"]:
                print(f"    (2) food-only picks S{r['2_food_seed']} "
                      f"(food={r['2_food_food']:.2f}) vs (1) S{r['1_full_seed']} "
                      f"(food={r['1_full_food']:.2f}, slope={r['1_full_slope']:+.3f})")
            if r["1_full_seed"] != r["3_balanced_seed"]:
                print(f"    (3) balanced picks S{r['3_balanced_seed']} "
                      f"vs (1) S{r['1_full_seed']}")


if __name__ == "__main__":
    main()
