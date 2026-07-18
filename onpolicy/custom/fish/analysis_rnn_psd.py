"""
Per-timestep RNN wideband power and contextual analyses.

Power measure: mean_wideband_power_ts = mean_H(h_t²) — instantaneous mean
squared activation across hidden units at each timestep.  Analogous to
eod_rate_centered; enables dist-to-X, peri-event, and condition boxplots.

Usage
-----
    python analysis_rnn_psd.py --spec_dir path/to/evals/2fish_m1a1k1_uniform_square

Outputs to {spec_dir}/analyses/rnn_psd/
    rnn_psd_power_per_agent_timestep.csv
    rnn_psd_power_ts_vs_dist_agent.pdf
    rnn_psd_power_ts_vs_nearby.pdf
    rnn_psd_power_ts_vs_dominant.pdf
    rnn_psd_peri_{event}.pdf
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_style import (
    set_style, panel, save, COND_PALETTE, AGENT_COLORS,
    add_size_advantage, condition_plot, TIME_STEP_MS,
)
from rnn_loader import iter_rnn_episodes

OUT_SUBDIR = "analyses/rnn_psd"
PERI_WINDOW = 100  # ±steps around event (~1200 ms at 83 Hz), matches analysis_eod.py
POWER_TS_COL = "mean_wideband_power_ts"


# ── per-timestep power ────────────────────────────────────────────────────────

def compute_power_ts_df(raw_dir: str, dff: pd.DataFrame) -> pd.DataFrame:
    """
    Per-timestep wideband RNN power: mean_H(h_t²) for each timestep.

    This is the instantaneous wideband power at single-timestep resolution,
    analogous to eod_rate_centered.  One row per (episode_index, env_id,
    agent_id, time_step).
    """
    chunks = []
    for k, rnn_arr, dff_ep in iter_rnn_episodes(raw_dir, dff):
        T, E, A, H = rnn_arr.shape
        ts_power = (rnn_arr.astype(np.float32) ** 2).mean(axis=3)  # (T, E, A)
        for e in range(E):
            for a in range(A):
                ep_rows = dff_ep[
                    (dff_ep["env_id"] == e) & (dff_ep["agent_id"] == a)
                ]
                ts_arr = ep_rows["time_step"].values
                valid = ts_arr < T
                if not valid.all():
                    ts_arr = ts_arr[valid]
                chunks.append(pd.DataFrame({
                    "episode_index": k,
                    "env_id": e,
                    "agent_id": a,
                    "time_step": ts_arr,
                    POWER_TS_COL: ts_power[ts_arr, e, a],
                }))
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


# ── per-timestep analysis helpers ─────────────────────────────────────────────

def _binned_mean_sem(x, y, n_bins=15, x_range=None):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    if x_range is None:
        x_range = (np.percentile(x, 2), np.percentile(x, 98))
    bins = np.linspace(x_range[0], x_range[1], n_bins + 1)
    idx = np.clip(np.digitize(x, bins) - 1, 0, n_bins - 1)
    centres, means, sems, ns = [], [], [], []
    for b in range(n_bins):
        vals = y[idx == b]
        if len(vals) < 5:
            continue
        centres.append((bins[b] + bins[b + 1]) / 2)
        means.append(np.mean(vals))
        sems.append(np.std(vals) / np.sqrt(len(vals)))
        ns.append(len(vals))
    return np.array(centres), np.array(means), np.array(sems), np.array(ns)


def peri_event_power(df, event_col, window=PERI_WINDOW, min_events=3):
    """
    Extract ±window steps of per-timestep wideband power around each event.
    Returns (offsets, mean_power, sem_power, n_events).
    """
    if POWER_TS_COL not in df.columns:
        return None, None, None, 0
    offsets = np.arange(-window, window + 1)
    all_traces = []
    for (env_id, ep_idx, agent_id), grp in df.groupby(
            ["env_id", "episode_index", "agent_id"]):
        grp = grp.sort_values("time_step").reset_index(drop=True)
        power_arr = grp[POWER_TS_COL].values
        events = grp[grp[event_col].astype(bool)]["time_step"].values
        for t in events:
            row_mask = grp["time_step"] == t
            if not row_mask.any():
                continue
            i0 = int(row_mask.idxmax())
            i_start, i_end = i0 - window, i0 + window + 1
            if i_start < 0 or i_end > len(grp):
                continue
            trace = power_arr[i_start:i_end]
            if len(trace) == 2 * window + 1:
                all_traces.append(trace)
    if len(all_traces) < min_events:
        return None, None, None, len(all_traces)
    traces = np.array(all_traces)
    return offsets, traces.mean(axis=0), traces.std(axis=0) / np.sqrt(len(traces)), len(all_traces)


# ── per-timestep plot functions ───────────────────────────────────────────────

def plot_power_ts_vs_dist_agent(df, out_dir):
    """Binned mean per-timestep wideband power vs distance to nearest agent."""
    if POWER_TS_COL not in df.columns or "distance_to_nearest_agent" not in df.columns:
        return
    set_style()
    x = df["distance_to_nearest_agent"].values
    y = df[POWER_TS_COL].values
    centres, means, sems, ns = _binned_mean_sem(x, y, n_bins=15, x_range=(0, 150))
    if len(centres) == 0:
        return
    fig, ax = panel()
    ax.plot(centres, means, color=AGENT_COLORS[0], lw=1.5)
    ax.fill_between(centres, means - sems, means + sems, color=AGENT_COLORS[0], alpha=0.3)
    ax.axvline(10,  color="gray", lw=0.8, ls="--", alpha=0.7, label="10 cm")
    ax.axvline(100, color="gray", lw=0.8, ls=":",  alpha=0.7, label="100 cm")
    if len(ns) > 0:
        ax.plot([], [], color="none", label=f"N={int(ns.sum()):,} obs")
    ax.set_xlabel("Dist. to nearest agent (cm)")
    ax.set_ylabel("Mean wideband power")
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, 150)
    ax.legend(frameon=False, fontsize=6)
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "rnn_psd_power_ts_vs_dist_agent.pdf"))


def plot_peri_event_power(df, event_col, event_label, out_dir, window=PERI_WINDOW):
    """Peri-event wideband power: mean ± SEM trace aligned to event onset."""
    offsets, mean, sem, n = peri_event_power(df, event_col, window=window)
    if offsets is None or n == 0:
        print(f"  [rnn_psd] peri_{event_col}: only {n} events — skipping", flush=True)
        return
    set_style()
    t_ms = offsets * TIME_STEP_MS
    fig, ax = panel()
    ax.plot(t_ms, mean, color=AGENT_COLORS[0], lw=1.5)
    ax.fill_between(t_ms, mean - sem, mean + sem, color=AGENT_COLORS[0], alpha=0.3)
    ax.axvline(0, color="red", lw=0.8, ls="--", alpha=0.7)
    ax.set_xlabel(f"Time re. {event_label} (ms)")
    ax.set_ylabel("Mean wideband power")
    ax.text(0.98, 0.98, f"N={n}", transform=ax.transAxes,
            ha="right", va="top", fontsize=7)
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    fname = f"rnn_psd_peri_{event_col.replace('_event', '')}.pdf"
    save(fig, os.path.join(out_dir, fname))
    print(f"  [rnn_psd] peri_{event_col}: N={n} events", flush=True)


def _plot_power_ts_by_condition(df, cond_col, labels, out_fname, out_dir):
    """
    Boxplot of mean per-timestep power per (env, ep, agent) within each condition,
    mirroring the plot_eod_vs_nearby pattern in analysis_eod.py.
    """
    if POWER_TS_COL not in df.columns or cond_col not in df.columns:
        return
    grp_cols = ["env_id", "episode_index", "agent_id"]
    true_label, false_label = labels
    near = (df[df[cond_col].astype(bool)]
            .groupby(grp_cols)[POWER_TS_COL].mean().reset_index())
    near["condition"] = true_label
    far = (df[~df[cond_col].astype(bool)]
           .groupby(grp_cols)[POWER_TS_COL].mean().reset_index())
    far["condition"] = false_label
    if len(near) < 2 or len(far) < 2:
        return
    long = pd.concat([far, near], ignore_index=True)
    palette = {false_label: COND_PALETTE[0], true_label: COND_PALETTE[1]}
    order = [false_label, true_label]
    U, p = stats.mannwhitneyu(
        near[POWER_TS_COL].values, far[POWER_TS_COL].values, alternative="two-sided"
    )
    stars = ("****" if p < 1e-4 else "***" if p < 1e-3
             else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s.")
    set_style()
    fig, ax = panel()
    condition_plot(ax, long, "condition", POWER_TS_COL, palette, order=order,
                   ylabel="Mean wideband power")
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + (yhi - ylo) * 0.12)
    ax.text(0.5, 0.97, stars, transform=ax.transAxes,
            ha="center", va="top", fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5})
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, out_fname))
    print(f"  [rnn_psd] {out_fname}: p={p:.3g}", flush=True)


def plot_power_ts_vs_nearby(df, out_dir):
    _plot_power_ts_by_condition(
        df, "has_nearby",
        labels=("Agent\nnearby", "No agent\nnearby"),
        out_fname="rnn_psd_power_ts_vs_nearby.pdf",
        out_dir=out_dir,
    )


def plot_power_ts_vs_dominant(df, out_dir):
    _plot_power_ts_by_condition(
        df, "is_dominant",
        labels=("Dominant", "Subordinate"),
        out_fname="rnn_psd_power_ts_vs_dominant.pdf",
        out_dir=out_dir,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def run(spec_dir, force_recompute=False):
    set_style()
    raw_dir     = os.path.join(spec_dir, "raw")
    derived_dir = os.path.join(spec_dir, "derived")
    out_dir     = os.path.join(spec_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "rnn_psd")

    step_pkl = os.path.join(derived_dir, "per_env_ep_agent_step.pkl")
    if not os.path.exists(step_pkl):
        print(f"[rnn_psd] missing {step_pkl} — skipping", flush=True)
        return
    print(f"[rnn_psd] loading {step_pkl}", flush=True)
    dff = pd.read_pickle(step_pkl)

    # ── per-timestep power ────────────────────────────────────────────────────
    power_ts_csv = base + "_power_per_agent_timestep.csv"
    if not force_recompute and os.path.exists(power_ts_csv):
        print("[rnn_psd] loading cached per-timestep power CSV", flush=True)
        power_ts_df = pd.read_csv(power_ts_csv)
    else:
        print("[rnn_psd] computing per-timestep power ...", flush=True)
        power_ts_df = compute_power_ts_df(raw_dir, dff)
    if power_ts_df.empty:
        print("[rnn_psd] no per-timestep power — skipping contextual analyses", flush=True)
        print("[rnn_psd] done", flush=True)
        return
    if force_recompute or not os.path.exists(power_ts_csv):
        power_ts_df.to_csv(power_ts_csv, index=False)

    merged_ts = dff.merge(
        power_ts_df,
        on=["episode_index", "env_id", "agent_id", "time_step"],
        how="left",
    )
    merged_ts = add_size_advantage(merged_ts)
    if "size_advantage" in merged_ts.columns:
        merged_ts["is_dominant"] = merged_ts["size_advantage"] > 0

    print(f"[rnn_psd] per-timestep power merged: "
          f"{merged_ts[POWER_TS_COL].notna().sum()} rows", flush=True)

    plot_power_ts_vs_dist_agent(merged_ts, out_dir)
    plot_power_ts_vs_nearby(merged_ts, out_dir)
    if "is_dominant" in merged_ts.columns:
        if merged_ts["is_dominant"].any() and (~merged_ts["is_dominant"]).any():
            plot_power_ts_vs_dominant(merged_ts, out_dir)

    for event_col, event_label in [
        ("eating_event", "eating"),
        ("biting_event", "biting"),
        ("was_bitten",   "being bitten"),
    ]:
        if event_col in merged_ts.columns:
            plot_peri_event_power(merged_ts, event_col, event_label, out_dir)

    print("[rnn_psd] done", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec_dir", required=True)
    ap.add_argument("--force_recompute", action="store_true",
                    help="Ignore cached power CSV and recompute from raw RNN arrays")
    args = ap.parse_args(argv)
    run(args.spec_dir, force_recompute=args.force_recompute)


if __name__ == "__main__":
    main()
