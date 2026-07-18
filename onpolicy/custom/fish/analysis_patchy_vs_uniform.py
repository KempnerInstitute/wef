"""
Patchy vs uniform food-distribution comparison (multi-spec).

Compares two foraging specs that differ only in how food is spatially arranged
while keeping the full sensor complement (m1a1k1):

  m1a1k1_patchy_square   — food clustered into patches
  m1a1k1_uniform_square  — food spread uniformly

The question is whether spatial structure of the food (patchy vs uniform) changes
foraging efficiency, competition, and social aggregation when total food and sensors
are held fixed.

Usage:
    python analysis_patchy_vs_uniform.py \\
        --evals_dir <run_dir>/evals \\
        --out_dir   <run_dir>/multi_eval/patchy_vs_uniform

Outputs under out_dir/:
  {metric}.pdf   — per-metric boxplot+strip, patchy vs uniform, with MWU annotation
  summary.csv    — per-spec mean/SEM/N for every metric
  stats.csv      — Mann-Whitney U + BH-corrected p-value per metric
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from analysis_style import COND_PALETTE, condition_plot, panel, save, set_style, format_n_fish

# spec_key → display label
SPECS = {
    "m1a1k1_patchy_square":  "Patchy",
    "m1a1k1_uniform_square": "Uniform",
}
SPEC_COLORS = {
    "m1a1k1_patchy_square":  COND_PALETTE[0],
    "m1a1k1_uniform_square": COND_PALETTE[1],
}

# Episode-level metrics (column in per_env_ep.pkl, axis label). Patchy distributions
# are expected to raise food inequality, biting competition, and aggregation.
_METRICS = [
    # foraging
    ("food_eaten",            "Food eaten / episode"),
    ("food_eaten_theil",      "Food inequality (Theil)"),
    ("p_near_food",           "P(near food)"),
    ("distance_to_closest_food", "Dist to closest food (cm)"),
    # social / aggregation
    ("num_biting_events",     "Biting events"),
    ("mean_nn_distance_cm",   "Mean NN distance (cm)"),
    ("average_pairwise_distances", "Mean pairwise distance (cm)"),
    ("p_near_other_fish",     "P(near other fish)"),
    ("num_interactions",      "Interactions"),
    ("polarization",          "Polarization"),
    # electrocommunication
    ("p_emit_eod",            "P(emit EOD)"),
    ("p_eod_interactions",    "P(EOD interaction)"),
    # movement / space use
    ("displacement_20_step",  "Displacement (20-step)"),
    ("p_near_wall",           "P(near wall)"),
]


def _sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def _load_ep(evals_dir, spec_key):
    p = os.path.join(evals_dir, spec_key, "derived", "per_env_ep.pkl")
    if not os.path.exists(p):
        return None
    df = pd.read_pickle(p)
    df = df.copy()
    df["spec_key"] = spec_key
    df["cond_label"] = SPECS[spec_key]
    return df


def _bh_correct(pvalues):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values in original order."""
    m = len(pvalues)
    if m == 0:
        return np.array([])
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    p_sorted = p[order]
    adjusted = p_sorted * m / np.arange(1, m + 1)
    for i in range(m - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    adjusted = np.minimum(adjusted, 1.0)
    result = np.empty(m)
    result[order] = adjusted
    return result


def _sig_str(p):
    return ("***" if p < 0.001 else "**" if p < 0.01 else
            "*" if p < 0.05 else "n.s.")


def plot_metric(ep_dfs, col, ylabel, out_dir):
    """Per-metric boxplot+strip (patchy vs uniform). Returns a stats row dict or None."""
    specs = [s for s in SPECS if ep_dfs.get(s) is not None and col in ep_dfs[s].columns]
    if len(specs) < 1:
        return None
    long = pd.concat([ep_dfs[s][["cond_label", col]] for s in specs], ignore_index=True)
    order = [SPECS[s] for s in specs]
    palette = {SPECS[s]: SPEC_COLORS[s] for s in specs}

    n_fish_vals = set()
    for s in specs:
        if "num_agents" in ep_dfs[s].columns:
            n_fish_vals.update(ep_dfs[s]["num_agents"].dropna().unique())
    n_fish = format_n_fish(n_fish_vals) if n_fish_vals else None

    set_style()
    fig, ax = panel()
    condition_plot(ax, long, "cond_label", col, palette, order=order, ylabel=ylabel,
                   count_label="title", n_fish=n_fish)

    row = None
    if len(specs) == 2:
        a = ep_dfs[specs[0]][col].dropna().values
        b = ep_dfs[specs[1]][col].dropna().values
        if len(a) > 2 and len(b) > 2:
            U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            ax.set_title(_sig_str(p), fontsize=8)
            row = {
                "metric": col,
                f"{SPECS[specs[0]]}_mean": float(np.mean(a)),
                f"{SPECS[specs[1]]}_mean": float(np.mean(b)),
                "U_stat": float(U), "p_raw": float(p),
            }
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, f"{col}.pdf"))
    return row


def save_summary(ep_dfs, out_dir):
    rows = []
    for spec, df in ep_dfs.items():
        if df is None:
            continue
        for col, _ in _METRICS:
            if col not in df.columns:
                continue
            rows.append({
                "spec_key": spec,
                "cond_label": SPECS[spec],
                "metric": col,
                "mean": float(df[col].mean()),
                "sem": _sem(df[col].values),
                "n_episodes": int(df[col].notna().sum()),
            })
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "summary.csv"), index=False)


def run(evals_dir, out_dir, plot_only=False):
    ep_dfs = {s: _load_ep(evals_dir, s) for s in SPECS}
    found = [s for s in SPECS if ep_dfs[s] is not None]
    print(f"  patchy_vs_uniform: found {len(found)}/{len(SPECS)} specs: {found}", flush=True)
    if not found:
        print("  [SKIP] no patchy/uniform specs found in evals_dir", flush=True)
        return
    os.makedirs(out_dir, exist_ok=True)

    stat_rows = []
    for col, ylabel in _METRICS:
        r = plot_metric(ep_dfs, col, ylabel, out_dir)
        if r is not None:
            stat_rows.append(r)

    if stat_rows:
        adj = _bh_correct([r["p_raw"] for r in stat_rows])
        for r, q in zip(stat_rows, adj):
            r["p_bh"] = float(q)
            r["sig"] = _sig_str(q)
        pd.DataFrame(stat_rows).to_csv(os.path.join(out_dir, "stats.csv"), index=False)

    save_summary(ep_dfs, out_dir)
    print("  patchy_vs_uniform: done", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args(argv)
    out_dir = args.out_dir or os.path.normpath(
        os.path.join(args.evals_dir, "..", "multi_eval", "patchy_vs_uniform"))
    run(args.evals_dir, out_dir)


if __name__ == "__main__":
    main()
