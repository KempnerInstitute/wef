"""
EOD rate vs. context (distance to agent / food / wall) and peri-event analyses.

Usage:
    python analysis_eod.py --spec_dir path/to/evals/m1a1k1_patchy_square

Outputs to {spec_dir}/analyses/eod/
  eod_vs_dist_agent.pdf            — EOD rate vs distance to nearest agent (y-axis from 0)
  eod_vs_dist_agent_adaptive.pdf   — same, y-axis limits set by data range
  eod_vs_dist_food.pdf             — EOD rate vs distance to food
  eod_vs_dist_food_adaptive.pdf    — same, adaptive y-axis
  eod_vs_dist_wall.pdf             — EOD rate vs distance to wall
  eod_vs_dist_wall_adaptive.pdf    — same, adaptive y-axis
  peri_eating.pdf                  — peri-event EOD around eating events
  peri_eating_adaptive.pdf         — same, adaptive y-axis
  peri_biting.pdf                  — peri-event EOD around biting another fish
  peri_biting_adaptive.pdf         — same, adaptive y-axis
  peri_bitten.pdf                  — peri-event EOD around being bitten
  peri_bitten_adaptive.pdf         — same, adaptive y-axis
  eod_vs_nearby.pdf                — EOD rate: agent nearby vs not
  eod_vs_size_advantage.pdf        — scatter: mean EOD (when nearby) vs size advantage
  eod_by_size_dominance.pdf        — boxplot: mean EOD dominant vs subordinate (when nearby)
  eod_rate_corr_vs_dist.pdf        — Pearson r between fish-pair EOD rates, binned by inter-fish distance
  eod_rate_corr_vs_dist.csv        — underlying per-bin stats
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
from analysis_style import (
    set_style, panel, save, condition_plot, compute_wall_distance,
    add_size_advantage, AGENT_COLORS, SEMANTIC_COLORS, COND_PALETTE, TIME_STEP_MS, SIM_FPS,
)
from utils_features import add_eod_rolling_windows
from utils_figsaving import _save_data, _load_data

OUT_SUBDIR = "analyses/eod"
PERI_WINDOW = 100         # ± steps around event (~1200 ms at 83 Hz)

EOD_RATE_COL = "eod_rate_centered"  # symmetric filter: correct instantaneous rate at each distance/event
EOD_RATE_LABEL = "EOD rate (Hz)"


def _ensure_eod_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Add eod_rate_* columns if they are not already present."""
    if EOD_RATE_COL not in df.columns:
        df, _ = add_eod_rolling_windows(df)
    return df


# ── binned mean helper ────────────────────────────────────────────────────────

def binned_mean_sem(x, y, n_bins=15, x_range=None):
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


def plot_eod_vs_distance(ax, x, rate, x_label, color, n_bins=15, x_range=None, adaptive=False):
    centres, means, sems, ns = binned_mean_sem(x, rate, n_bins, x_range)
    ax.plot(centres, means, color=color, lw=1.5)
    ax.fill_between(centres, means - sems, means + sems, color=color, alpha=0.3)
    if len(ns) > 0:
        ax.plot([], [], color="none", label=f"N={int(ns.sum()):,}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(EOD_RATE_LABEL)
    if not adaptive:
        ax.set_ylim(bottom=0)
    if x_range is not None:
        ax.set_xlim(x_range)
    sns.despine(ax=ax)


# ── peri-event ────────────────────────────────────────────────────────────────

def peri_event_eod(df, event_col, window=PERI_WINDOW, min_events=3):
    """
    Extract ±window steps of EOD rate around each event.
    Returns (offsets, mean_rate_hz, sem_rate_hz, n_events).
    """
    offsets = np.arange(-window, window + 1)
    all_traces = []

    for (env_id, ep_idx, agent_id), grp in df.groupby(
            ["env_id", "episode_index", "agent_id"]):
        grp = grp.sort_values("time_step").reset_index(drop=True)
        rate_arr = grp[EOD_RATE_COL].values
        events = grp[grp[event_col].astype(bool)]["time_step"].values

        for t in events:
            row_mask = grp["time_step"] == t
            if not row_mask.any():
                continue
            i0 = row_mask.idxmax()
            i_start = i0 - window
            i_end   = i0 + window + 1
            if i_start < 0 or i_end > len(grp):
                continue
            trace = rate_arr[i_start:i_end]
            if len(trace) == 2 * window + 1:
                all_traces.append(trace)

    if len(all_traces) < min_events:
        return None, None, None, len(all_traces)

    traces = np.array(all_traces)
    mean = traces.mean(axis=0)
    sem  = traces.std(axis=0) / np.sqrt(len(traces))
    return offsets, mean, sem, len(all_traces)


# ── panels ────────────────────────────────────────────────────────────────────

def plot_eod_vs_dist_agent(df, out_dir, adaptive=False):
    set_style()
    x   = df["distance_to_nearest_agent"].values
    rate = df[EOD_RATE_COL].values
    fig, ax = panel()
    plot_eod_vs_distance(ax, x, rate, "Dist. to nearest agent (cm)", AGENT_COLORS[0],
                         x_range=(0, 150), adaptive=adaptive)
    ax.axvline(10,  color="gray", lw=0.8, ls="--", alpha=0.7, label="10 cm")
    ax.axvline(100, color="gray", lw=0.8, ls=":",  alpha=0.7, label="100 cm")
    ax.legend(frameon=False, fontsize=6)
    plt.tight_layout(pad=0.3)
    suffix = "_adaptive" if adaptive else ""
    save(fig, os.path.join(out_dir, f"eod_vs_dist_agent{suffix}.pdf"))


def plot_eod_vs_dist_food(df, out_dir, adaptive=False):
    set_style()
    x   = df["distance_to_closest_food"].values
    rate = df[EOD_RATE_COL].values
    fig, ax = panel()
    plot_eod_vs_distance(ax, x, rate, "Dist. to closest food (cm)", AGENT_COLORS[2],
                         x_range=(0, 10), adaptive=adaptive)
    ax.axvline(5, color="gray", lw=0.8, ls="--", alpha=0.7, label="~5 cm sensor range")
    ax.legend(frameon=False, fontsize=6)
    plt.tight_layout(pad=0.3)
    suffix = "_adaptive" if adaptive else ""
    save(fig, os.path.join(out_dir, f"eod_vs_dist_food{suffix}.pdf"))


def plot_eod_vs_dist_wall(df, out_dir, adaptive=False):
    set_style()
    wall_dist = compute_wall_distance(df).values
    rate      = df[EOD_RATE_COL].values
    valid     = np.isfinite(wall_dist) & (wall_dist >= 0)
    fig, ax   = panel()
    plot_eod_vs_distance(ax, wall_dist[valid], rate[valid],
                         "Dist. to wall (cm)", AGENT_COLORS[3],
                         x_range=(0, 10), adaptive=adaptive)
    ax.legend(frameon=False, fontsize=6)
    plt.tight_layout(pad=0.3)
    suffix = "_adaptive" if adaptive else ""
    save(fig, os.path.join(out_dir, f"eod_vs_dist_wall{suffix}.pdf"))


def plot_eod_vs_nearby(df, out_dir):
    """Boxplot: mean EOD rate per agent-episode when agent nearby vs not."""
    set_style()
    near = (df[df["has_nearby"]]
            .groupby(["env_id", "episode_index", "agent_id"])[EOD_RATE_COL]
            .mean().reset_index())
    near["proximity"] = "Agent\nnearby"

    far = (df[~df["has_nearby"]]
           .groupby(["env_id", "episode_index", "agent_id"])[EOD_RATE_COL]
           .mean().reset_index())
    far["proximity"] = "No agent\nnearby"

    long   = pd.concat([far, near], ignore_index=True)
    palette = {"No agent\nnearby": COND_PALETTE[0], "Agent\nnearby": COND_PALETTE[1]}
    order  = ["No agent\nnearby", "Agent\nnearby"]
    U, p   = stats.mannwhitneyu(near[EOD_RATE_COL].values, far[EOD_RATE_COL].values,
                                 alternative="two-sided")
    fig, ax = panel()
    condition_plot(ax, long, "proximity", EOD_RATE_COL, palette, order=order,
                   ylabel=EOD_RATE_LABEL)
    ax.set_title(f"p = {p:.3f}", fontsize=7)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "eod_vs_nearby.pdf"))
    pd.DataFrame([{
        "test": "mannwhitneyu",
        "group_a": "agent_nearby",
        "group_b": "no_agent_nearby",
        "n_a": len(near),
        "n_b": len(far),
        "mean_a": near[EOD_RATE_COL].mean(),
        "mean_b": far[EOD_RATE_COL].mean(),
        "U_stat": U,
        "p": p,
    }]).to_csv(os.path.join(out_dir, "eod_proximity_mwu.csv"), index=False)


def plot_eod_vs_size_advantage(df: pd.DataFrame, out_dir: str):
    """
    Scatter per (env, ep, agent): mean EOD rate (when nearby) vs size_advantage.
    Tests whether dominant fish emit at different rates during social encounters.
    """
    if "size_advantage" not in df.columns or "has_nearby" not in df.columns:
        return
    set_style()
    nearby = df[df["has_nearby"].astype(bool)].copy()
    if len(nearby) < 10:
        return
    agg = (
        nearby.groupby(["env_id", "episode_index", "agent_id"])
        .agg(mean_eod=(EOD_RATE_COL, "mean"), size_advantage=("size_advantage", "first"))
        .dropna()
        .reset_index()
    )
    if len(agg) < 5:
        return

    x = agg["size_advantage"].values
    y = agg["mean_eod"].values
    r, p_r = stats.pearsonr(x, y)

    from statsmodels.api import OLS, add_constant
    model = OLS(y, add_constant(x)).fit()
    xs_sorted = np.sort(x)
    pred = model.get_prediction(add_constant(xs_sorted))
    ci = pred.conf_int()
    p_ols = model.pvalues[1]

    def _sig(p): return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    fig, ax = panel()
    ax.scatter(x, y, facecolors="none", edgecolors=SEMANTIC_COLORS["eod"], s=18, linewidths=0.8, alpha=0.7)
    ax.fill_between(xs_sorted, ci[:, 0], ci[:, 1], color="black", alpha=0.12)
    ax.plot(xs_sorted, pred.predicted_mean, color="black", lw=1.2)
    ax.axvline(0, color="gray", lw=0.7, ls="--", alpha=0.5)
    ax.text(0.05, 1.02,
            f"$R^2={model.rsquared:.2f}$, $P={p_ols:.3f}$ {_sig(p_ols)}\n"
            f"$r={r:.2f}$, $N={len(agg)}$",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=6, style="italic",
            clip_on=False,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5))
    ax.set_xlabel("Size advantage (self − other)")
    ax.set_ylabel(f"Mean {EOD_RATE_LABEL} (when nearby)")
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "eod_vs_size_advantage.pdf"))
    agg.to_csv(os.path.join(out_dir, "eod_vs_size_advantage.csv"), index=False)


def plot_eod_by_size_dominance(df: pd.DataFrame, out_dir: str):
    """
    Per (env, ep, agent): mean EOD rate (when nearby) compared between
    dominant (size_advantage > 0) and subordinate (size_advantage < 0) agents.
    """
    if "size_advantage" not in df.columns or "has_nearby" not in df.columns:
        return
    set_style()
    nearby = df[df["has_nearby"].astype(bool)].copy()
    if len(nearby) < 10:
        return
    agg = (
        nearby.groupby(["env_id", "episode_index", "agent_id"])
        .agg(mean_eod=(EOD_RATE_COL, "mean"), size_advantage=("size_advantage", "first"))
        .dropna()
        .reset_index()
    )
    dom_label = "Dominant\n(self > other)"
    sub_label = "Subordinate\n(self < other)"
    agg["role"] = agg["size_advantage"].apply(
        lambda v: dom_label if v > 0 else (sub_label if v < 0 else None)
    )
    agg = agg.dropna(subset=["role"])
    order   = [sub_label, dom_label]
    palette = {dom_label: "#CC4433", sub_label: "#4477AA"}
    n_dom = int((agg["role"] == dom_label).sum())
    n_sub = int((agg["role"] == sub_label).sum())
    if min(n_dom, n_sub) < 3:
        return

    U, p = stats.mannwhitneyu(
        agg.loc[agg["role"] == dom_label, "mean_eod"].values,
        agg.loc[agg["role"] == sub_label, "mean_eod"].values,
        alternative="two-sided",
    )
    n_labels = [f"{o}\n(N={n})" for o, n in [(sub_label, n_sub), (dom_label, n_dom)]]

    fig, ax = panel()
    condition_plot(ax, agg, "role", "mean_eod", palette, order=order,
                   ylabel=f"Mean {EOD_RATE_LABEL} (when nearby)")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(n_labels, rotation=15, ha="right")
    ax.set_title(f"p = {p:.3f}", fontsize=7)
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "eod_by_size_dominance.pdf"))
    pd.DataFrame([{
        "comparison": "dominant_vs_subordinate_when_nearby",
        "n_dominant": n_dom, "n_subordinate": n_sub,
        "mean_eod_dominant":    float(agg.loc[agg["role"] == dom_label, "mean_eod"].mean()),
        "mean_eod_subordinate": float(agg.loc[agg["role"] == sub_label, "mean_eod"].mean()),
        "U_stat": float(U), "p": float(p),
    }]).to_csv(os.path.join(out_dir, "eod_by_size_dominance.csv"), index=False)


# ── inter-fish EOD rate correlation vs. distance ─────────────────────────────

def _build_pairwise_eod(df: pd.DataFrame) -> pd.DataFrame:
    """Return a row per (env, ep, t, pair) with columns dist, eod_i, eod_j.

    Uses each agent's nearest-neighbour pair; deduplicates by keeping only
    rows where agent_id < nearest_agent_id.  Works for any N ≥ 2.
    """
    if EOD_RATE_COL not in df.columns:
        df, _ = add_eod_rolling_windows(df)
    if "nearest_agent_id" not in df.columns:
        return pd.DataFrame()

    # Build lookup table: (env, ep, t, agent_id) → eod_rate
    eod_j = (
        df[["env_id", "episode_index", "time_step", "agent_id", EOD_RATE_COL]]
        .rename(columns={"agent_id": "nearest_agent_id", EOD_RATE_COL: "eod_j"})
    )

    pairs = df[df["agent_id"] < df["nearest_agent_id"]].copy()
    pairs = pairs.merge(
        eod_j,
        on=["env_id", "episode_index", "time_step", "nearest_agent_id"],
        how="left",
    )
    pairs = pairs.rename(columns={
        EOD_RATE_COL: "eod_i",
        "distance_to_nearest_agent": "dist",
    })
    return (
        pairs[["env_id", "episode_index", "time_step", "agent_id",
               "nearest_agent_id", "dist", "eod_i", "eod_j"]]
        .dropna()
        .reset_index(drop=True)
    )


def _compute_eod_corr_by_dist(pairs: pd.DataFrame, n_bins: int = 12,
                               dist_range=None) -> pd.DataFrame:
    """Pearson r between paired EOD rates in distance bins.

    Per-bin stats are computed two ways:
      r_pooled  — single r over all observations in the bin
      r_mean/r_sem — mean ± SEM of per-episode r values (Fisher Z-transformed)

    Returns a DataFrame with columns:
      dist_centre, r_pooled, r_mean, r_sem, n_obs, n_episodes
    """
    dist = pairs["dist"].values
    if dist_range is None:
        dist_range = (np.percentile(dist, 1), np.percentile(dist, 99))
    bins = np.linspace(dist_range[0], dist_range[1], n_bins + 1)

    records = []
    for b in range(n_bins):
        mask = (dist >= bins[b]) & (dist < bins[b + 1])
        sub = pairs[mask]
        if len(sub) < 20:
            continue
        r_pooled, _ = stats.pearsonr(sub["eod_i"].values, sub["eod_j"].values)

        ep_rs = []
        for (_, _), ep_sub in sub.groupby(["env_id", "episode_index"]):
            if len(ep_sub) < 10:
                continue
            r_ep, _ = stats.pearsonr(ep_sub["eod_i"].values, ep_sub["eod_j"].values)
            if np.isfinite(r_ep):
                ep_rs.append(r_ep)

        if len(ep_rs) >= 2:
            z = np.arctanh(np.clip(ep_rs, -0.9999, 0.9999))
            r_mean = float(np.tanh(np.mean(z)))
            r_sem  = float(np.std(z, ddof=1) / np.sqrt(len(z)))
        else:
            r_mean = r_sem = np.nan

        records.append({
            "dist_centre": (bins[b] + bins[b + 1]) / 2,
            "r_pooled":    float(r_pooled),
            "r_mean":      r_mean,
            "r_sem":       r_sem,
            "n_obs":       int(len(sub)),
            "n_episodes":  len(ep_rs),
        })
    return pd.DataFrame(records)


def plot_eod_rate_corr_vs_dist(corr_df: pd.DataFrame, out_dir: str):
    """Plot per-bin Pearson r (mean ± SEM across episodes) vs. inter-fish distance."""
    if corr_df.empty:
        return
    set_style()
    fig, ax = panel()
    x = corr_df["dist_centre"].values
    y = corr_df["r_mean"].values
    yerr = corr_df["r_sem"].values
    valid = np.isfinite(y)
    ax.plot(x[valid], y[valid], color=AGENT_COLORS[0], lw=1.5)
    ax.fill_between(x[valid], y[valid] - yerr[valid], y[valid] + yerr[valid],
                    color=AGENT_COLORS[0], alpha=0.3)
    ax.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel("Inter-fish distance (cm)")
    ax.set_ylabel("Pearson r  (EOD rate, fish i vs. j)")
    total_eps = int(corr_df["n_episodes"].median())
    ax.text(0.97, 0.97, f"~{total_eps} ep/bin",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5})
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "eod_rate_corr_vs_dist.pdf"))


def _plot_peri(offsets, mean, sem, n_events, label, color, out_path, adaptive=False):
    set_style()
    t_ms = offsets * TIME_STEP_MS
    fig, ax = panel()
    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.plot(t_ms, mean, color=color, lw=1.5, label=f"N={n_events}")
    ax.fill_between(t_ms, mean - sem, mean + sem, color=color, alpha=0.3)
    ax.set_xlabel(f"Time from {label} (ms)")
    ax.set_ylabel(EOD_RATE_LABEL)
    ax.legend(frameon=False, fontsize=6)
    if not adaptive:
        ax.set_ylim(bottom=0)
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, out_path)


def plot_peri_events(df, out_dir, adaptive=False):
    suffix = "_adaptive" if adaptive else ""
    events = [
        ("eating_event",    "eating",        AGENT_COLORS[2], f"peri_eating{suffix}.pdf"),
        ("bite_other_fish", "biting",        AGENT_COLORS[1], f"peri_biting{suffix}.pdf"),
        ("was_bitten",      "being bitten",  AGENT_COLORS[0], f"peri_bitten{suffix}.pdf"),
    ]
    for col, label, color, fname in events:
        offsets, mean, sem, n = peri_event_eod(df, col)
        if offsets is None:
            print(f"  too few events for {col} (n={n}) — skipped")
            continue
        _plot_peri(offsets, mean, sem, n, label, color,
                   os.path.join(out_dir, fname), adaptive=adaptive)


# ── main ─────────────────────────────────────────────────────────────────────

def load(spec_dir):
    step_pkl = os.path.join(spec_dir, "derived", "per_env_ep_agent_step.pkl")
    if not os.path.exists(step_pkl):
        sys.exit(f"Not found: {step_pkl}")
    out_dir = os.path.join(spec_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Loading {step_pkl} \u2026")
    df = pd.read_pickle(step_pkl)
    df = _ensure_eod_rate(df)
    print(f"  {len(df):,} rows")
    return {"df": df, "out_dir": out_dir}


def run(spec_dir, force_recompute=False):
    step_pkl = os.path.join(spec_dir, "derived", "per_env_ep_agent_step.pkl")
    out_dir  = os.path.join(spec_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    cache    = os.path.join(out_dir, "_eod_corr_data")

    cache_hit = not force_recompute
    if cache_hit:
        try:
            corr_df = _load_data(cache)
        except FileNotFoundError:
            cache_hit = False

    if not cache_hit:
        if not os.path.exists(step_pkl):
            sys.exit(f"Not found: {step_pkl}")
        print(f"Loading {step_pkl} …")
        df = pd.read_pickle(step_pkl)
        print(f"  {len(df):,} rows")
        df = _ensure_eod_rate(df)
        df = add_size_advantage(df)

        plot_eod_vs_dist_agent(df, out_dir)
        plot_eod_vs_dist_agent(df, out_dir, adaptive=True)
        plot_eod_vs_dist_food(df, out_dir)
        plot_eod_vs_dist_food(df, out_dir, adaptive=True)
        plot_eod_vs_dist_wall(df, out_dir)
        plot_eod_vs_dist_wall(df, out_dir, adaptive=True)
        plot_eod_vs_nearby(df, out_dir)
        plot_eod_vs_size_advantage(df, out_dir)
        plot_eod_by_size_dominance(df, out_dir)
        plot_peri_events(df, out_dir)
        plot_peri_events(df, out_dir, adaptive=True)

        # EOD rate correlation vs. inter-fish distance (multi-agent only)
        if df["agent_id"].nunique() >= 2:
            print("  computing inter-fish EOD rate correlation …")
            pairs   = _build_pairwise_eod(df)
            dist_max = float(np.percentile(pairs["dist"].values, 98)) if len(pairs) else 0
            corr_df = _compute_eod_corr_by_dist(pairs, n_bins=12,
                                                 dist_range=(0, dist_max)) if len(pairs) >= 20 else pd.DataFrame()
            _save_data(corr_df, cache)
        else:
            corr_df = pd.DataFrame()

    plot_eod_rate_corr_vs_dist(corr_df, out_dir)
    if not corr_df.empty:
        corr_df.to_csv(os.path.join(out_dir, "eod_rate_corr_vs_dist.csv"), index=False)
    print("Done.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec_dir", required=True)
    ap.add_argument("--force_recompute", action="store_true")
    args = ap.parse_args(argv)
    run(args.spec_dir, force_recompute=args.force_recompute)


if __name__ == "__main__":
    main()
