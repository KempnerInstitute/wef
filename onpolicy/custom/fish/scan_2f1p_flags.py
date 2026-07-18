"""
Scan all completed 2f1p multi_eval directories (local + cluster-mounted) and flag:
  (1) B eats more alone than in social conditions (U_stat < n_social*n_alone/2)
  (2) B eats pct is not monotonically decreasing across B>A → A=B → A>B

Usage: python scan_2f1p_flags.py
"""

import os
import sys
import glob
import getpass
import pandas as pd
import numpy as np

_USER = getpass.getuser()
RESULTS_ROOTS = [
    f"/home/{_USER}/cluster_lab/{_USER}/marl_fish_storage/results",
    f"/home/{_USER}/kr/mfrefactor/onpolicy/custom/fish/results",
]

SPEC_ORDER = ["2f1p_AltB", "2f1p_AeqB", "2f1p_AgtB"]  # expected decreasing B eat pct


def check_b_eats_pct(run_dir):
    """
    Returns dict {spec: pct} for B eating ≥1 food per condition.
    Returns None if any spec is missing derived data.
    """
    pcts = {}
    for spec in SPEC_ORDER:
        pkl = os.path.join(run_dir, "evals", spec, "derived", "per_env_ep_agent.pkl")
        if not os.path.exists(pkl):
            return None
        try:
            df = pd.read_pickle(pkl)
        except Exception:
            return None
        b = df[df["role"] == "B"] if "role" in df.columns else df[df["agent_id"] == 1]
        if len(b) == 0:
            return None
        pct = (b.groupby(["env_id", "episode_index"])["food_eaten"].sum() > 0).mean() * 100
        pcts[spec] = pct
    return pcts


def check_social_vs_alone(multi2f1p_dir):
    """
    Reads social_vs_alone_mwu.csv.
    Returns (b_alone_gt_social: bool, U, p) or (None, None, None) if missing.
    """
    csv = os.path.join(multi2f1p_dir, "social_vs_alone_mwu.csv")
    if not os.path.exists(csv):
        return None, None, None
    try:
        df = pd.read_csv(csv)
    except Exception:
        return None, None, None
    b_row = df[df["role"] == "B"]
    if len(b_row) == 0:
        return None, None, None
    r = b_row.iloc[0]
    # U = count of times soc_vals > alone_vals; U < n1*n2/2 means alone > social
    midpoint = r["n_social"] * r["n_alone"] / 2
    b_alone_gt = r["U_stat"] < midpoint
    return b_alone_gt, r["U_stat"], r["p"]


def find_run_dirs(root):
    """Yield (run_dir, multi2f1p_dir) for completed 2f1p multi_eval dirs."""
    # Structure: results/{exp_name}/{seed_name}/{timestamp}/
    pattern = os.path.join(root, "*", "*", "*", "multi_eval", "2f1p")
    for d in glob.glob(pattern):
        run_dir = os.path.dirname(os.path.dirname(d))  # strip multi_eval/2f1p
        yield run_dir, d


def short_label(run_dir):
    parts = run_dir.split(os.sep)
    # find the timestamp part (8-digit date pattern)
    for i, p in enumerate(parts):
        if len(p) == 15 and p[8] == "_":  # YYYYMMDD_HHMMSS
            seed_name = parts[i - 1]
            exp_name = parts[i - 2]
            return f"{exp_name}/{seed_name}/{p}"
    return run_dir


def main():
    flags = []

    for root in RESULTS_ROOTS:
        if not os.path.isdir(root):
            continue
        for run_dir, multi2f1p_dir in find_run_dirs(root):
            b_alone_gt, U, p = check_social_vs_alone(multi2f1p_dir)
            pcts = check_b_eats_pct(run_dir)

            flag1 = b_alone_gt  # B eats more alone
            flag2 = None
            if pcts is not None:
                vals = [pcts[s] for s in SPEC_ORDER]
                flag2 = not (vals[0] >= vals[1] >= vals[2])  # not monotone decreasing

            if flag1 or flag2:
                flags.append({
                    "run": short_label(run_dir),
                    "path": run_dir,
                    "flag1_b_alone_gt_social": flag1,
                    "U_stat": round(U, 1) if U is not None else None,
                    "p_mwu": round(p, 4) if p is not None else None,
                    "flag2_not_monotone": flag2,
                    "pct_BgtA": round(pcts["2f1p_AltB"], 1) if pcts else None,
                    "pct_AeqB": round(pcts["2f1p_AeqB"], 1) if pcts else None,
                    "pct_AgtB": round(pcts["2f1p_AgtB"], 1) if pcts else None,
                })

    if not flags:
        print("No flags found across all results.")
        return

    df = pd.DataFrame(flags)
    df = df.sort_values(["flag1_b_alone_gt_social", "flag2_not_monotone"], ascending=False)

    print(f"\nFound {len(flags)} flagged runs:\n")
    for _, row in df.iterrows():
        f1 = "[FLAG1: B alone>social]" if row["flag1_b_alone_gt_social"] else ""
        f2 = "[FLAG2: non-monotone]" if row["flag2_not_monotone"] else ""
        pct_str = ""
        if row["pct_BgtA"] is not None:
            pct_str = f"  B-eat-pct: B>A={row['pct_BgtA']}% A=B={row['pct_AeqB']}% A>B={row['pct_AgtB']}%"
        u_str = f"  U={row['U_stat']} p={row['p_mwu']}" if row["U_stat"] is not None else ""
        print(f"{f1} {f2}")
        print(f"  {row['run']}{u_str}{pct_str}")
        print()


if __name__ == "__main__":
    main()
