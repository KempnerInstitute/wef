"""
Side-by-side comparison of two PLSC formulations on the same eval data.

  "ours"      — SVD-based, row-permutation null, independent component testing
                (compute_plsc_gpu from analysis_rnn_plsc.py)

  "zhang-phi" — SVD-based, circular-shift null, dual criterion (SV + corr),
                contiguous counting, z-scored inputs
                (ports hongw-lab/code_for_2024_zhang-phi via plsc/plsc_analysis.py)

Usage
-----
    python analysis_rnn_plsc_compare.py --spec_dir path/to/evals/2fish_m1a1k1_uniform_wide

Outputs to {spec_dir}/analyses/rnn_plsc/
    rnn_plsc_compare_sig_dims.pdf
    rnn_plsc_compare_plsc1_by_range.pdf
    rnn_plsc_compare.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_style import set_style, save
from analysis_rnn_plsc import (
    N_COMPONENTS, N_SHUFFLES, MIN_SAMPLES, OUT_SUBDIR,
    iter_rnn_episodes,
    compute_plsc_gpu_zhangphi,
)
from sklearn.decomposition import PCA


# ── run both methods over episodes ───────────────────────────────────────────

def compute_sig_dims_both(raw_dir, dff, morm_range_cm=10.0,
                          max_episodes=None, pca_dim=0):
    """
    Run both PLSC methods on every valid (episode, env, condition) triple.
    Returns a combined DataFrame with a 'method' column ('ours' | 'zhang-phi').
    """
    from analysis_rnn_plsc import compute_plsc_gpu

    records = []
    n_eps = 0
    for k, rnn_arr, dff_ep in iter_rnn_episodes(raw_dir, dff):
        if max_episodes is not None and n_eps >= max_episodes:
            break
        T, E, A, H = rnn_arr.shape
        if A < 2:
            continue

        for e in range(E):
            mask_a0  = (dff_ep["env_id"] == e) & (dff_ep["agent_id"] == 0)
            dist_vals = dff_ep.loc[mask_a0, "distance_to_nearest_agent"].values
            if len(dist_vals) != T:
                continue

            rnn_a0 = rnn_arr[:, e, 0, :]
            rnn_a1 = rnn_arr[:, e, 1, :]

            if pca_dim > 0 and pca_dim < H:
                rnn_a0 = PCA(n_components=pca_dim).fit_transform(rnn_a0)
                rnn_a1 = PCA(n_components=pca_dim).fit_transform(rnn_a1)

            for label, idx_mask in [("in_range",  dist_vals <= morm_range_cm),
                                     ("out_range", dist_vals >  morm_range_cm)]:
                idx = np.where(idx_mask)[0]
                if len(idx) < MIN_SAMPLES * 2:
                    continue

                Xa, Ya = rnn_a0[idx], rnn_a1[idx]

                for method, fn in [("ours",      compute_plsc_gpu),
                                    ("zhang-phi", compute_plsc_gpu_zhangphi)]:
                    result = fn(Xa, Ya, n_components=N_COMPONENTS, n_shuffles=N_SHUFFLES)
                    records.append({
                        "episode_index": k,
                        "env_id":        e,
                        "condition":     label,
                        "method":        method,
                        "num_sig":       result["num_sig"],
                        "top_corr":      result["top_corr"],
                    })
        n_eps += 1

    return pd.DataFrame(records)


# ── comparison plots ──────────────────────────────────────────────────────────

_CMP_PALETTE = {"ours": "#2C5AA0", "zhang-phi": "#E36B2E"}
_METHOD_ORDER = ["ours", "zhang-phi"]
_COND_LABELS  = {"in_range": "In range", "out_range": "Out of range"}
_COND_ORDER   = ["In range", "Out of range"]


def _comparison_violin(combined_df, y_col, ylabel, outfile):
    plot_df = combined_df.copy()
    plot_df["label"] = plot_df["condition"].map(_COND_LABELS)
    plot_df = plot_df.dropna(subset=["label", y_col])

    fig, ax = plt.subplots(figsize=(3.6, 2.24))
    sns.violinplot(data=plot_df, x="label", y=y_col, hue="method",
                   order=_COND_ORDER, hue_order=_METHOD_ORDER,
                   palette=_CMP_PALETTE, inner="quart", cut=0,
                   linewidth=0.8, split=False, dodge=True, ax=ax)

    n_total = len(plot_df)
    ax.text(0.03, 0.97, f"N={n_total}", transform=ax.transAxes,
            ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="none"))
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.legend(title="Method", fontsize=7, title_fontsize=7,
              loc="upper right", framealpha=0.7)
    sns.despine(ax=ax)
    plt.tight_layout()
    save(fig, outfile)


def plot_comparison_sig_dims(combined_df, outfile_base):
    _comparison_violin(combined_df, "num_sig",
                       "# significant PLSC dims",
                       outfile_base + "_compare_sig_dims.pdf")


def plot_comparison_plsc1_by_range(combined_df, outfile_base):
    _comparison_violin(combined_df, "top_corr",
                       "PLSC1 canonical correlation",
                       outfile_base + "_compare_plsc1_by_range.pdf")


# ── entry point ───────────────────────────────────────────────────────────────

def run_comparison(spec_dir, morm_range_cm=10.0, max_episodes=None, pca_dim=0):
    set_style()
    raw_dir     = os.path.join(spec_dir, "raw")
    derived_dir = os.path.join(spec_dir, "derived")
    out_dir     = os.path.join(spec_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "rnn_plsc")

    step_pkl = os.path.join(derived_dir, "per_env_ep_agent_step.pkl")
    if not os.path.exists(step_pkl):
        print(f"[rnn_plsc_compare] missing {step_pkl} -- skipping", flush=True)
        return
    print(f"[rnn_plsc_compare] loading {step_pkl}", flush=True)
    dff = pd.read_pickle(step_pkl)

    print(f"[rnn_plsc_compare] running both methods "
          f"(max_episodes={max_episodes}, pca_dim={pca_dim}) ...", flush=True)
    combined = compute_sig_dims_both(raw_dir, dff, morm_range_cm=morm_range_cm,
                                     max_episodes=max_episodes, pca_dim=pca_dim)

    if combined.empty:
        print("[rnn_plsc_compare] no valid triples -- skipping", flush=True)
        return

    combined.to_csv(base + "_compare.csv", index=False)
    plot_comparison_sig_dims(combined, base)
    plot_comparison_plsc1_by_range(combined, base)

    for method, grp in combined.groupby("method"):
        in_  = grp[grp.condition == "in_range" ]["num_sig"].mean()
        out_ = grp[grp.condition == "out_range"]["num_sig"].mean()
        print(f"  {method:10s}  sig dims: in={in_:.1f}  out={out_:.1f}", flush=True)

    print("[rnn_plsc_compare] done", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec_dir", required=True)
    ap.add_argument("--morm_range_cm", type=float, default=10.0)
    ap.add_argument("--max_episodes", type=int, default=None)
    ap.add_argument("--pca_dim", type=int, default=0,
                    help="PCA pre-projection dim before PLSC; 0 = disabled")
    args = ap.parse_args(argv)
    run_comparison(args.spec_dir, morm_range_cm=args.morm_range_cm,
                   max_episodes=args.max_episodes, pca_dim=args.pca_dim)


if __name__ == "__main__":
    main()
