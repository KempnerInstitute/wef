"""
Food spatial distribution vs. foraging and group behaviour.

Requires:
  derived/food_dispersion.pkl   (from preprocess_food_dispersion.py)
  derived/per_env_ep.pkl

Outputs to {spec_dir}/analyses/food_distribution/
  dispersion_trajectory.pdf      — Clark-Evans R at start / mid / end
  dispersion_vs_food_eaten.pdf   — R_start vs. total food eaten (Spearman ρ)
  dispersion_vs_food_theil.pdf   — R_start vs. Theil inequality index
  dispersion_vs_group_spacing.pdf — R_start vs. mean agent NN distance
  dispersion_vs_interactions.pdf  — R_start vs. number of interactions

Usage
-----
  python analysis_food_distribution.py --spec_dir path/to/evals/m1a1k1_patchy_square
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from analysis_style import set_style, panel, save

OUT_SUBDIR = "analyses/food_distribution"
_DATA_FILE  = "_food_distribution_data.pkl"


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------

def _load(spec_dir: str) -> pd.DataFrame:
    disp_path = os.path.join(spec_dir, "derived", "food_dispersion.pkl")
    ep_path   = os.path.join(spec_dir, "derived", "per_env_ep.pkl")

    if not os.path.exists(disp_path):
        sys.exit(f"Not found: {disp_path} — run preprocess_food_dispersion.py first")
    if not os.path.exists(ep_path):
        sys.exit(f"Not found: {ep_path}")

    disp = pd.read_pickle(disp_path)
    ep   = pd.read_pickle(ep_path)
    df   = disp.merge(ep, on=["env_id", "episode_index"], how="inner")
    print(f"  Merged {len(df)} episode rows  "
          f"({df['env_id'].nunique()} envs × {df['episode_index'].nunique()} episodes)")
    return df


def _save_data(df: pd.DataFrame, out_dir: str) -> None:
    df.to_pickle(os.path.join(out_dir, _DATA_FILE))


# ---------------------------------------------------------------------------
# scatter helper
# ---------------------------------------------------------------------------

def _spearman_scatter(ax, x: np.ndarray, y: np.ndarray,
                      xlabel: str, ylabel: str, color: str = "#4477AA"):
    """Returns (rho, p, n) if test was run, else (None, None, None)."""
    valid = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[valid], y[valid]
    ax.scatter(xv, yv, facecolors="none", edgecolors=color, s=18,
               linewidths=0.8, alpha=0.7)
    if len(xv) > 3:
        rho, p = stats.spearmanr(xv, yv)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.text(0.05, 0.95,
                f"$\\rho={rho:+.2f}$, $P={p:.3f}$ {sig}",
                transform=ax.transAxes, va="top", ha="left",
                style="italic", fontsize=6,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5})
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        sns.despine(ax=ax)
        return float(rho), float(p), int(len(xv))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    sns.despine(ax=ax)
    return None, None, None


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def plot_dispersion_trajectory(df: pd.DataFrame, out_dir: str) -> None:
    """Mean ± SEM of Clark-Evans R at start, mid, end across episodes."""
    set_style()
    fig, ax = panel(2.56, 2.24)

    checkpoints = ["start", "mid", "end"]
    means = [df[f"food_dispersion_{c}"].mean() for c in checkpoints]
    sems  = [df[f"food_dispersion_{c}"].sem()  for c in checkpoints]

    ax.errorbar([0, 1, 2], means, yerr=sems,
                fmt="o-", color="#4477AA", capsize=3, lw=1.5, ms=5)
    ax.axhline(1.0, ls="--", lw=0.8, color="grey", label="CSR (R=1)")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Start", "Mid", "End"])
    ax.set_ylabel("Clark-Evans R")
    ax.set_xlabel("Episode checkpoint")
    ax.legend(fontsize=6)
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "dispersion_trajectory.pdf"))


def plot_dispersion_vs_food_eaten(df: pd.DataFrame, out_dir: str):
    set_style()
    fig, ax = panel()
    rho, p, n = _spearman_scatter(
        ax,
        df["food_dispersion_start"].values,
        df["food_eaten"].values,
        xlabel="Clark-Evans R (start)",
        ylabel="Total food eaten (episode)",
        color="#4477AA",
    )
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "dispersion_vs_food_eaten.pdf"))
    return {"x_var": "food_dispersion_start", "y_var": "food_eaten", "rho": rho, "p": p, "n": n}


def plot_dispersion_vs_food_theil(df: pd.DataFrame, out_dir: str):
    set_style()
    fig, ax = panel()
    rho, p, n = _spearman_scatter(
        ax,
        df["food_dispersion_start"].values,
        df["food_eaten_theil"].values,
        xlabel="Clark-Evans R (start)",
        ylabel="Food Theil inequality index",
        color="#EE6677",
    )
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "dispersion_vs_food_theil.pdf"))
    return {"x_var": "food_dispersion_start", "y_var": "food_eaten_theil", "rho": rho, "p": p, "n": n}


def plot_dispersion_vs_group_spacing(df: pd.DataFrame, out_dir: str):
    set_style()
    fig, ax = panel()
    rho, p, n = _spearman_scatter(
        ax,
        df["food_dispersion_start"].values,
        df["mean_nn_distance_cm"].values,
        xlabel="Clark-Evans R (start)",
        ylabel="Mean agent NN distance (cm)",
        color="#228833",
    )
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "dispersion_vs_group_spacing.pdf"))
    return {"x_var": "food_dispersion_start", "y_var": "mean_nn_distance_cm", "rho": rho, "p": p, "n": n}


def plot_dispersion_vs_interactions(df: pd.DataFrame, out_dir: str):
    set_style()
    fig, ax = panel()
    rho, p, n = _spearman_scatter(
        ax,
        df["food_dispersion_start"].values,
        df["num_interactions"].values,
        xlabel="Clark-Evans R (start)",
        ylabel="Number of interactions (episode)",
        color="#CCBB44",
    )
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "dispersion_vs_interactions.pdf"))
    return {"x_var": "food_dispersion_start", "y_var": "num_interactions", "rho": rho, "p": p, "n": n}


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def load(spec_dir, force=False):
    out_dir = os.path.join(spec_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, _DATA_FILE)
    if not force and os.path.exists(cache):
        df = pd.read_pickle(cache)
        print(f"Loaded {len(df)} rows from cache")
    else:
        df = _load(spec_dir)
        _save_data(df, out_dir)
    return {"df": df, "out_dir": out_dir}


def run(spec_dir: str, plot_only: bool = False, force: bool = False) -> None:
    out_dir = os.path.join(spec_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    cache = os.path.join(out_dir, _DATA_FILE)
    if plot_only and os.path.exists(cache):
        df = pd.read_pickle(cache)
        print(f"[plot_only] loaded {len(df)} rows from cache")
    else:
        df = _load(spec_dir)
        _save_data(df, out_dir)

    plot_dispersion_trajectory(df, out_dir)
    spearman_rows = [
        plot_dispersion_vs_food_eaten(df, out_dir),
        plot_dispersion_vs_food_theil(df, out_dir),
        plot_dispersion_vs_group_spacing(df, out_dir),
        plot_dispersion_vs_interactions(df, out_dir),
    ]
    pd.DataFrame([r for r in spearman_rows if r["rho"] is not None]).to_csv(
        os.path.join(out_dir, "spearman_correlations.csv"), index=False)

    print(f"Saved → {out_dir}/")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--spec_dir", required=True)
    ap.add_argument("--plot_only", action="store_true",
                    help="Skip data load; use cached intermediate pkl")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    run(args.spec_dir, plot_only=args.plot_only, force=args.force)


if __name__ == "__main__":
    main()
