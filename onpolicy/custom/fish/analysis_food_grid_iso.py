"""
Iso-total-food patch sweep: does spatial structure matter beyond total food amount?

Two series across n_patches ∈ {1, 2, 3, 4, 5}:

  iso   — n × food_mult = 1.0  (total food ≈ 26.5 items regardless of n_patches)
          specs: iso_p1_m1, iso_p2_m05, iso_p3_m0333, iso_p4_m025, iso_p5_m02

  free  — food_mult = 1.0      (total food scales with n_patches)
          specs: iso_p1_m1 (anchor), free_p2_m1, free_p3_m1, free_p4_m1, free_p5_m1

If the iso curve is flat, spatial patch structure has no effect beyond changing
total food quantity. Any slope is a genuine spatial-distribution effect.

Outputs under out_dir/:
  comparison/  — line plots (iso vs free) per metric + ratio + competition
  summary.csv  — per-series × per-n_patches summary statistics
"""

import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from analysis_style import COND_PALETTE, panel, save, set_style, format_n_fish, add_right_count_title

warnings.filterwarnings("ignore")

N_LIST   = [1, 2, 3, 4, 5]
ISO_MAP  = {1: "iso_p1_m1",  2: "iso_p2_m05",  3: "iso_p3_m0333",
            4: "iso_p4_m025", 5: "iso_p5_m02"}
FREE_MAP = {1: "iso_p1_m1",  2: "free_p2_m1",  3: "free_p3_m1",
            4: "free_p4_m1", 5: "free_p5_m1"}

COL_ISO  = COND_PALETTE[0]
COL_FREE = COND_PALETTE[1]

_METRICS = [
    ("food_per_fish",      "Food per fish"),
    ("food_eaten_theil",   "Food inequality (Theil)"),
    ("num_biting_events",  "Biting events"),
    ("mean_nn_distance_cm","Mean NN dist (cm)"),
    ("p_emit_eod",         "EOD rate"),
]


def _sem(x):
    return x.std() / np.sqrt(len(x)) if len(x) > 1 else 0.0


def _load_ep(evals_dir, spec_key):
    p = os.path.join(evals_dir, spec_key, "derived", "per_env_ep.pkl")
    return pd.read_pickle(p) if os.path.exists(p) else None


def load_series(evals_dir):
    """Return (iso_df, free_df) from evals_dir/{spec_key}/derived/per_env_ep.pkl."""
    iso_rows, free_rows = [], []
    for n in N_LIST:
        for rows, key_map in [(iso_rows, ISO_MAP), (free_rows, FREE_MAP)]:
            key = key_map[n]
            ep = _load_ep(evals_dir, key)
            if ep is None:
                continue
            ep = ep.copy()
            ep["n_patches"] = n
            ep["food_per_fish"] = ep["food_eaten"] / ep["num_agents"]
            ep["spec_key"] = key
            rows.append(ep)
    iso_df  = pd.concat(iso_rows,  ignore_index=True) if iso_rows  else pd.DataFrame()
    free_df = pd.concat(free_rows, ignore_index=True) if free_rows else pd.DataFrame()
    return iso_df, free_df


def _line(ax, df, col, color, label):
    if df.empty or col not in df.columns:
        return
    agg = df.groupby("n_patches")[col].agg(["mean", _sem]).reset_index()
    agg.columns = ["n", "m", "s"]
    ax.fill_between(agg["n"], agg["m"] - agg["s"], agg["m"] + agg["s"],
                    color=color, alpha=0.2)
    ax.plot(agg["n"], agg["m"], color=color, lw=1.5, marker="o", ms=5, label=label)


def _n_summary(iso_df, free_df):
    """Return (n_fish_str, n_obs_str) for the right-aligned panel titles."""
    n_fish_vals, counts = set(), []
    for df in (iso_df, free_df):
        if df is None or df.empty:
            continue
        if "num_agents" in df.columns:
            n_fish_vals.update(df["num_agents"].dropna().unique())
        if "n_patches" in df.columns:
            counts.extend(df.groupby("n_patches").size().tolist())
    n_fish = format_n_fish(n_fish_vals) if n_fish_vals else None
    if counts:
        lo, hi = min(counts), max(counts)
        n_obs = f"N={lo}" if lo == hi else f"N={lo}–{hi}"
    else:
        n_obs = None
    return n_fish, n_obs


def plot_comparison(iso_df, free_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    n_fish, n_obs = _n_summary(iso_df, free_df)

    for col, ylabel in _METRICS:
        if col not in iso_df.columns and col not in free_df.columns:
            continue
        set_style()
        fig, ax = panel()
        _line(ax, iso_df,  col, COL_ISO,  "Iso-total-food")
        _line(ax, free_df, col, COL_FREE, "Unconstrained")
        ax.set_xlabel("# patches")
        ax.set_ylabel(ylabel)
        ax.set_xticks(N_LIST)
        ax.legend(frameon=False, fontsize=6)
        add_right_count_title(ax, n_obs, n_fish=n_fish)
        sns.despine(ax=ax)
        plt.tight_layout(pad=0.3)
        save(fig, os.path.join(out_dir, f"{col}.pdf"))

    # Ratio iso/free for food_per_fish — how much of any n_patches effect is food-quantity confound
    if not iso_df.empty and not free_df.empty and "food_per_fish" in iso_df.columns:
        iso_m  = iso_df.groupby("n_patches")["food_per_fish"].mean()
        free_m = free_df.groupby("n_patches")["food_per_fish"].mean()
        common = sorted(set(iso_m.index) & set(free_m.index))
        if len(common) >= 2:
            ratio = iso_m[common] / free_m[common]
            set_style()
            fig, ax = panel()
            ax.plot(common, ratio.values, color=COND_PALETTE[2], lw=1.5, marker="o", ms=5)
            ax.axhline(1.0, color="grey", lw=0.8, ls="--")
            ax.set_xlabel("# patches")
            ax.set_ylabel("Iso / unconstrained\n(food per fish ratio)")
            ax.set_xticks(common)
            ax.set_ylim(0, 1.2)
            add_right_count_title(ax, n_obs, n_fish=n_fish)
            sns.despine(ax=ax)
            plt.tight_layout(pad=0.3)
            save(fig, os.path.join(out_dir, "food_ratio_iso_vs_free.pdf"))

    # Competition: bites per food item
    set_style()
    fig, ax = panel()
    for df, label, color in [(iso_df, "Iso-total-food", COL_ISO),
                              (free_df, "Unconstrained", COL_FREE)]:
        if df.empty or "num_biting_events" not in df.columns:
            continue
        dfc = df.copy()
        dfc["competition"] = dfc["num_biting_events"] / (dfc["food_per_fish"] + 1e-9)
        _line(ax, dfc, "competition", color, label)
    ax.set_xlabel("# patches")
    ax.set_ylabel("Bites per food item")
    ax.set_xticks(N_LIST)
    ax.legend(frameon=False, fontsize=6)
    add_right_count_title(ax, n_obs, n_fish=n_fish)
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "competition.pdf"))


def save_summary(iso_df, free_df, out_dir):
    rows = []
    for df, series in [(iso_df, "iso"), (free_df, "free")]:
        if df.empty:
            continue
        for n in N_LIST:
            sub = df[df["n_patches"] == n]
            if sub.empty:
                continue
            rows.append({
                "series":            series,
                "n_patches":         n,
                "food_mult":         round(1.0 / n, 4) if series == "iso" else 1.0,
                "food_per_fish_mean": sub["food_per_fish"].mean(),
                "food_per_fish_sem":  _sem(sub["food_per_fish"]),
                "theil_mean":   sub["food_eaten_theil"].mean()    if "food_eaten_theil"    in sub.columns else np.nan,
                "biting_mean":  sub["num_biting_events"].mean()   if "num_biting_events"   in sub.columns else np.nan,
                "nn_dist_mean": sub["mean_nn_distance_cm"].mean() if "mean_nn_distance_cm" in sub.columns else np.nan,
                "eod_mean":     sub["p_emit_eod"].mean()          if "p_emit_eod"          in sub.columns else np.nan,
                "n_episodes":   len(sub),
            })
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "summary.csv"), index=False)


def run(evals_dir, out_dir, plot_only=False):
    iso_df, free_df = load_series(evals_dir)
    n_iso  = iso_df["n_patches"].nunique()  if not iso_df.empty  else 0
    n_free = free_df["n_patches"].nunique() if not free_df.empty else 0
    print(f"  food_grid_iso: iso={len(iso_df)} rows ({n_iso} n_patches), "
          f"free={len(free_df)} rows ({n_free} n_patches)", flush=True)
    if iso_df.empty and free_df.empty:
        print("  [SKIP] no iso/free patch specs found in evals_dir")
        return
    os.makedirs(out_dir, exist_ok=True)
    plot_comparison(iso_df, free_df, os.path.join(out_dir, "comparison"))
    save_summary(iso_df, free_df, out_dir)
    print("  food_grid_iso: done", flush=True)
