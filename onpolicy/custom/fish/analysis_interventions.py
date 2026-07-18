"""
Intervention / ablation comparison plots.

Each intervention group is a set of eval specs compared on key behavioural metrics,
plotted as strip + mean±SEM bars (violin-style but works with few episodes).

Usage:
    python analysis_interventions.py \\
        --evals_dir <run_dir>/evals \\
        --out_dir   <run_dir>/multi_eval/interventions

Outputs one subfolder per group:
  sensor/            — mormyromast / ampullary / knollen ablations
  muting/            — fish 0 muted vs intact
  collective_sensing/ — cs0 / cs1 / cs2 sweep
  food_abundance/    — normal vs half-food
  num_patches/       — 1 patch vs n patches
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
    set_style, save, panel, condition_plot, COND_PALETTE,
    load_consumption_times, load_biting_times,
    time_to_X_linear, format_n_fish,
)
from cfg import PLOT_POINTS_BELOW_N

# ── intervention group definitions ──────────────────────────────────────────
# Each group: list of (spec_key, display_label) pairs, plus metrics to plot.

GROUPS = {
    "sensor": {
        "title": "Sensor ablations (patchy arena)",
        "conditions": [
            ("m1a1k1_patchy_square",  "All on"),
            ("m0a1k1_patchy_square",  "−Morm"),
            ("m1a0k1_patchy_square",  "−Amp"),
            ("m1a1k0_patchy_square",  "−Knollen"),
            ("m0a0k1_patchy_square",  "K only"),
            ("m0a0k0_patchy_square",  "All off"),
        ],
        "ep_metrics": [
            ("food_eaten",        "Food eaten"),
            ("food_eaten_theil",  "Food inequality\n(Theil)"),
            ("mean_nn_distance_cm", "Mean NN dist\n(cm)"),
            ("p_emit_eod",        "P(emit EOD)"),
            ("num_biting_events", "Biting events"),
            ("polarization",      "Polarization"),
        ],
        "ablation_profiles": True,
    },
    "muting": {
        "title": "Muting ablation",
        "conditions": [
            ("2fish_m1a1k1_uniform_square",       "Intact"),
            ("2fish_1mute_m1a1k1_uniform_square", "1 muted"),
        ],
        "ep_metrics": [
            ("food_eaten",        "Food eaten"),
            ("food_eaten_theil",  "Food inequality\n(Theil)"),
            ("mean_nn_distance_cm", "Mean NN dist\n(cm)"),
            ("num_biting_events", "Biting events"),
        ],
    },
    "collective_sensing": {
        "title": "Collective-sensing sweep (small arena)",
        "conditions": [
            ("small_cs0", "Self-image\nonly"),
            ("small_cs1", "Self + cons.\nimages"),
            ("small_cs2", "Cons-image\nonly"),
        ],
        "stats_baseline_spec_key": "small_cs1",
        "ep_metrics": [
            ("p_emit_eod",          "P(emit EOD)"),
            ("mean_nn_distance_cm", "Mean NN dist\n(cm)"),
            ("food_eaten",          "Food eaten"),
            ("num_interactions",    "Interactions"),
        ],
    },
    "food_abundance": {
        "title": "Food density sweep (patchy arena)",
        "conditions": [
            ("m1a1k1_patchy_square",        "Full food"),
            ("food05_m1a1k1_patchy_square",  "Half food"),
            ("food025_m1a1k1_patchy_square", "Quarter food"),
        ],
        "ep_metrics": [
            ("food_eaten",       "Food eaten"),
            ("food_eaten_theil", "Food inequality\n(Theil)"),
            ("p_near_food",      "P(near food)"),
        ],
    },
    "sensor_1fish": {
        "title": "Sensor ablations (1-fish, patchy arena)",
        "conditions": [
            ("1fish_m1a1k1_patchy_square", "All on"),
            ("1fish_m0a1k1_patchy_square", "−Morm"),
            ("1fish_m1a0k1_patchy_square", "−Amp"),
            ("1fish_m1a1k0_patchy_square", "−Knollen"),
            ("1fish_m0a0k1_patchy_square", "K only"),
            ("1fish_m0a0k0_patchy_square", "All off"),
        ],
        "ep_metrics": [
            ("food_eaten",          "Food eaten"),
            ("food_eaten_theil",    "Food inequality\n(Theil)"),
            ("mean_nn_distance_cm", "Mean NN dist\n(cm)"),
            ("p_emit_eod",          "P(emit EOD)"),
            ("num_biting_events",   "Biting events"),
            ("polarization",        "Polarization"),
        ],
        "ablation_profiles": True,
    },
    "sensor_food05": {
        "title": "Sensor ablations (half food, patchy arena)",
        "conditions": [
            ("food05_m1a1k1_patchy_square", "All on"),
            ("food05_m0a1k1_patchy_square", "−Morm"),
            ("food05_m1a0k1_patchy_square", "−Amp"),
            ("food05_m1a1k0_patchy_square", "−Knollen"),
        ],
        "ep_metrics": [
            ("food_eaten",          "Food eaten"),
            ("food_eaten_theil",    "Food inequality\n(Theil)"),
            ("mean_nn_distance_cm", "Mean NN dist\n(cm)"),
            ("p_emit_eod",          "P(emit EOD)"),
            ("num_biting_events",   "Biting events"),
            ("polarization",        "Polarization"),
        ],
        "ablation_profiles": True,
    },
    "sensor_food025": {
        "title": "Sensor ablations (quarter food, patchy arena)",
        "conditions": [
            ("food025_m1a1k1_patchy_square", "All on"),
            ("food025_m0a1k1_patchy_square", "−Morm"),
            ("food025_m1a0k1_patchy_square", "−Amp"),
            ("food025_m1a1k0_patchy_square", "−Knollen"),
        ],
        "ep_metrics": [
            ("food_eaten",          "Food eaten"),
            ("food_eaten_theil",    "Food inequality\n(Theil)"),
            ("mean_nn_distance_cm", "Mean NN dist\n(cm)"),
            ("p_emit_eod",          "P(emit EOD)"),
            ("num_biting_events",   "Biting events"),
            ("polarization",        "Polarization"),
        ],
        "ablation_profiles": True,
    },
    "uniform_amp": {
        "title": "Amp ablation (uniform arena)",
        "conditions": [
            ("m1a1k1_uniform_square", "All on"),
            ("m1a0k1_uniform_square", "−Amp"),
        ],
        "ep_metrics": [
            ("food_eaten",          "Food eaten"),
            ("food_eaten_theil",    "Food inequality\n(Theil)"),
            ("mean_nn_distance_cm", "Mean NN dist\n(cm)"),
            ("p_emit_eod",          "P(emit EOD)"),
            ("num_biting_events",   "Biting events"),
            ("polarization",        "Polarization"),
        ],
    },
    "num_patches": {
        "title": "Number of patches",
        "conditions": [
            ("m1a1k1_1patch",        "1 patch"),
            ("m1a1k1_patchy_square", "N patches"),
        ],
        "ep_metrics": [
            ("food_eaten",        "Food eaten"),
            ("food_eaten_theil",  "Food inequality\n(Theil)"),
            ("mean_nn_distance_cm", "Mean NN dist\n(cm)"),
        ],
    },
}

_PALETTE = COND_PALETTE  # alias


# ── Dunnett significance helpers ─────────────────────────────────────────────

def _significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _add_dunnett_annotations(ax, long_df, x_col, y_col, order, control_label):
    """
    Run Dunnett's test (each non-control condition vs. control_label) and
    annotate significance stars above each non-control box.

    Stars are placed at a common y level (just above the global max).  "ns" is
    shown in grey; significant results in black.  y-limits are extended to fit.

    Returns a list of {"condition", "p_dunnett", "stars"} dicts, empty on failure.
    """
    try:
        from scipy.stats import dunnett as _dunnett
    except ImportError:
        return []

    ctrl_vals = long_df.loc[long_df[x_col] == control_label, y_col].dropna().values
    if len(ctrl_vals) < 3:
        return []

    non_ctrl = [lbl for lbl in order if lbl != control_label]
    if not non_ctrl:
        return []

    groups = [long_df.loc[long_df[x_col] == lbl, y_col].dropna().values
              for lbl in non_ctrl]
    if any(len(g) < 3 for g in groups):
        return []

    result = _dunnett(*groups, control=ctrl_vals)
    pvalues = result.pvalue

    # Base ylim on per-condition whisker extents (Q1±1.5*IQR per group, max across
    # groups), matching exactly what seaborn draws. Using the combined distribution's
    # whisker would clip conditions whose per-group IQR differs from the combined IQR.
    whisker_hi = -np.inf
    whisker_lo = np.inf
    for lbl in order:
        g = long_df.loc[long_df[x_col] == lbl, y_col].dropna()
        if len(g) == 0:
            continue
        q1g, q3g = g.quantile(0.25), g.quantile(0.75)
        iqrg = q3g - q1g
        if iqrg > 0:
            whisker_hi = max(whisker_hi, g[g <= q3g + 1.5 * iqrg].max())
            whisker_lo = min(whisker_lo, g[g >= q1g - 1.5 * iqrg].min())
        else:
            whisker_hi = max(whisker_hi, g.max())
            whisker_lo = min(whisker_lo, g.min())
    y_range = max(whisker_hi - whisker_lo, 1e-9)
    annot_y = whisker_hi + y_range * 0.08

    stat_rows = []
    for lbl, p in zip(non_ctrl, pvalues):
        cond_x = order.index(lbl)
        stars = _significance_stars(p)
        color = "black" if stars != "ns" else "#888888"
        ax.text(cond_x, annot_y, stars,
                ha="center", va="bottom", fontsize=7, color=color,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5})
        stat_rows.append({"condition": lbl, "p_dunnett": float(p), "stars": stars})

    # Label the control arm where its (blank) Dunnett annotation would otherwise be
    if control_label in order:
        ax.text(order.index(control_label), annot_y, "(control)",
                ha="center", va="bottom", fontsize=6, color="#555555", style="italic",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5})

    ax.set_ylim(bottom=whisker_lo - y_range * 0.05, top=annot_y + y_range * 0.18)
    return stat_rows


# ── ablation profile plots ────────────────────────────────────────────────────

_MATCH_KEYS = ["env_id", "episode_index"]
_ABLATION_PROFILE_METRICS = [
    # "food_eaten",  # Temporarily disabled; uncomment to restore.
    "food_eaten_theil",
    "mean_nn_distance_cm",
    "p_emit_eod",
    "num_biting_events",
]


def _run_ablation_profiles(group_cfg, loaded, gout):
    """One figure per non-baseline condition: per-trial Δ for all metrics in 1 row.

    Trials matched on (env_id, episode_index). Wilcoxon signed-rank test
    against zero annotated as stars on each panel.
    """
    if len(loaded) < 2:
        return

    baseline_label, baseline_df = loaded[0]
    ep_metrics = [
        metric for metric in group_cfg["ep_metrics"]
        if metric[0] in _ABLATION_PROFILE_METRICS
    ]
    if not ep_metrics:
        return
    n_metrics = len(ep_metrics)

    for cond_label, cond_df in loaded[1:]:
        metric_cols = [col for col, _ in ep_metrics
                       if col in baseline_df.columns and col in cond_df.columns]
        if not metric_cols:
            continue

        bl = baseline_df[_MATCH_KEYS + metric_cols]
        ab = cond_df[_MATCH_KEYS + metric_cols]
        merged = bl.merge(ab, on=_MATCH_KEYS, suffixes=("_bl", "_ab"))
        if merged.empty:
            continue

        set_style()
        # fig, axes = plt.subplots(1, n_metrics, figsize=(4.0, 2.0), squeeze=False)
        fig, axes = plt.subplots(1, n_metrics, figsize=(8/3, 2.0), squeeze=False)
        axes = axes.flatten()

        rng = np.random.default_rng(0)
        for ax_idx, (col, ylabel) in enumerate(ep_metrics):
            ax = axes[ax_idx]
            bl_col, ab_col = f"{col}_bl", f"{col}_ab"
            if bl_col not in merged.columns or ab_col not in merged.columns:
                ax.set_visible(False)
                continue

            delta = (merged[ab_col] - merged[bl_col]).dropna().values

            ax.boxplot(delta, widths=0.5, patch_artist=True,
                       boxprops=dict(facecolor=_PALETTE[1], alpha=0.7),
                       medianprops=dict(color="black", linewidth=1.0),
                       whiskerprops=dict(color="black", linewidth=0.7),
                       capprops=dict(color="black", linewidth=0.7),
                       flierprops=dict(marker="", markersize=0))
            if len(delta) < PLOT_POINTS_BELOW_N:
                jitter = rng.uniform(-0.15, 0.15, size=len(delta))
                ax.scatter(1 + jitter, delta, color="black", alpha=0.5, s=5, zorder=3)

            ax.axhline(0, color="crimson", linewidth=0.7, linestyle="--")
            ax.set_title(ylabel, fontsize=5.5, pad=2)
            ax.set_xticks([])
            ax.set_xlabel("")
            sns.despine(ax=ax, bottom=True)
            ax.tick_params(axis="y", labelsize=5)

            # Wilcoxon signed-rank against zero
            if len(delta) >= 5 and delta.std() > 0:
                try:
                    _, p = stats.wilcoxon(delta)
                except ValueError:
                    p = 1.0
            else:
                p = 1.0
            stars = _significance_stars(p)
            color = "black" if stars != "ns" else "#999999"
            ylo, yhi = ax.get_ylim()
            ax.set_ylim(top=yhi + (yhi - ylo) * 0.18)
            ax.text(0.5, 0.97, stars, transform=ax.transAxes,
                    ha="center", va="top", fontsize=6, color=color)

        axes[0].set_ylabel(f"Δ ({cond_label}−{baseline_label})", fontsize=5.5)
        for ax in axes[1:]:
            ax.set_ylabel("")

        plt.tight_layout(pad=0.3)
        safe = cond_label.replace("−", "no").replace(" ", "_").lower()
        save(fig, os.path.join(gout, f"ablation_profile_{safe}.pdf"))


def _run_muting_delta_profile(group_cfg, loaded, gout):
    """Combined paired Δ figure for muting: muted minus intact per metric."""
    if len(loaded) < 2:
        return

    baseline_label, baseline_df = loaded[0]
    muted_label, muted_df = loaded[1]
    ep_metrics = [
        (col, ylabel) for col, ylabel in group_cfg["ep_metrics"]
        if col in baseline_df.columns and col in muted_df.columns
    ]
    if not ep_metrics:
        return

    metric_cols = [col for col, _ in ep_metrics]
    bl = baseline_df[_MATCH_KEYS + metric_cols]
    muted = muted_df[_MATCH_KEYS + metric_cols]
    merged = bl.merge(muted, on=_MATCH_KEYS, suffixes=("_bl", "_muted"))
    if merged.empty:
        return

    set_style()
    n_metrics = len(ep_metrics)
    fig_w = max(8 / 3, n_metrics * 0.67)
    fig, axes = plt.subplots(1, n_metrics, figsize=(fig_w, 2.0), squeeze=False)
    axes = axes.flatten()

    rng = np.random.default_rng(0)
    stat_rows = []
    for ax, (col, ylabel) in zip(axes, ep_metrics):
        bl_col, muted_col = f"{col}_bl", f"{col}_muted"
        delta = (merged[muted_col] - merged[bl_col]).dropna().values

        ax.boxplot(delta, widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor=_PALETTE[1], alpha=0.7),
                   medianprops=dict(color="black", linewidth=1.0),
                   whiskerprops=dict(color="black", linewidth=0.7),
                   capprops=dict(color="black", linewidth=0.7),
                   flierprops=dict(marker="", markersize=0))
        if len(delta) < PLOT_POINTS_BELOW_N:
            jitter = rng.uniform(-0.15, 0.15, size=len(delta))
            ax.scatter(1 + jitter, delta, color="black", alpha=0.5, s=5, zorder=3)

        ax.axhline(0, color="crimson", linewidth=0.7, linestyle="--")
        ax.set_title(ylabel, fontsize=5.5, pad=2)
        ax.set_xticks([])
        ax.set_xlabel("")
        sns.despine(ax=ax, bottom=True)
        ax.tick_params(axis="y", labelsize=5)

        if len(delta) >= 5 and not np.allclose(delta, 0):
            try:
                _, p = stats.wilcoxon(delta)
            except ValueError:
                p = 1.0
        else:
            p = 1.0
        stars = _significance_stars(p)
        color = "black" if stars != "ns" else "#999999"
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(top=yhi + (yhi - ylo) * 0.18)
        ax.text(0.5, 0.97, stars, transform=ax.transAxes,
                ha="center", va="top", fontsize=6, color=color)
        stat_rows.append({
            "metric": col,
            "comparison": f"{muted_label} - {baseline_label}",
            "p_wilcoxon": float(p),
            "stars": stars,
            "mean_delta": float(np.mean(delta)) if len(delta) else np.nan,
        })

    axes[0].set_ylabel(f"Δ ({muted_label}−{baseline_label})", fontsize=5.5)
    for ax in axes[1:]:
        ax.set_ylabel("")

    plt.tight_layout(pad=0.3, w_pad=0.35)
    save(fig, os.path.join(gout, "muting_delta_profile.pdf"))
    pd.DataFrame(stat_rows).to_csv(
        os.path.join(gout, "muting_delta_profile_stats.csv"), index=False)


# ── per-group analysis ────────────────────────────────────────────────────────

def _load_group_episode_dfs(group_cfg, evals_dir):
    """Load per-episode summary DataFrames for all available group conditions."""
    loaded = []
    for spec_key, label in group_cfg["conditions"]:
        path = os.path.join(evals_dir, spec_key, "derived", "per_env_ep.pkl")
        if not os.path.exists(path):
            print(f"    [{spec_key}] not found — skipped")
            continue
        df = pd.read_pickle(path)
        df["_spec"] = spec_key
        df["_label"] = label
        loaded.append((label, df))

    return loaded


def _group_palette(loaded):
    return {
        label: _PALETTE[i % len(_PALETTE)]
        for i, (label, _) in enumerate(loaded)
    }


def _group_n_fish_str(loaded):
    """Formatted fish-count string ('4 fish') from num_agents across conditions."""
    vals = set()
    for _, df in loaded:
        if "num_agents" in df.columns:
            vals.update(df["num_agents"].dropna().unique())
    return format_n_fish(vals) if vals else None


def _metric_long_df(loaded, col):
    rows = []
    for label, df in loaded:
        if col not in df.columns:
            continue
        tmp = df[[col]].copy()
        tmp["condition"] = label
        rows.append(tmp)
    if not rows:
        return None, []
    long = pd.concat(rows, ignore_index=True)
    order = [label for label, df in loaded if col in df.columns]
    return long, order


def _metric_panel_layout(order):
    n_cond = len(order)
    return max(2.0, n_cond * 0.48), 30 if n_cond >= 3 else 0


def _resolve_baseline(loaded, group_cfg):
    baseline_label, baseline_df = loaded[0]
    stats_baseline_spec_key = group_cfg.get("stats_baseline_spec_key")
    if stats_baseline_spec_key is not None:
        baseline_label, baseline_df = next(
            ((label, df) for label, df in loaded
             if df["_spec"].iloc[0] == stats_baseline_spec_key),
            (baseline_label, baseline_df),
        )
    return baseline_label, baseline_df


def _plot_metric_panels(loaded, ep_metrics, gout, palette, count_label):
    """One figure per metric — seaborn boxplot + optional stripplot.

    The observation count is shown in a right-aligned title (with fish count)
    rather than inside the plot body (count_label="title").
    """
    n_fish = _group_n_fish_str(loaded)
    for col, ylabel in ep_metrics:
        long, order = _metric_long_df(loaded, col)
        if long is None:
            continue
        w, rot = _metric_panel_layout(order)
        set_style()
        # fig, ax = plt.subplots(figsize=(w, 2.0))
        fig, ax = plt.subplots(figsize=(8/3, 2.0))
        condition_plot(ax, long, "condition", col, palette, order=order,
                       ylabel=ylabel, rotation=rot, count_label="title", n_fish=n_fish)
        plt.tight_layout(pad=0.3)
        save(fig, os.path.join(gout, f"{col}.pdf"))


def _plot_metric_dunnett_panels(loaded, ep_metrics, gout, palette, count_label, control_label):
    """Metric plots with Dunnett annotations versus the configured baseline.

    Observation count shown in a right-aligned title with fish count
    (count_label="title"); the control arm is labelled '(control)'.
    """
    n_fish = _group_n_fish_str(loaded)
    dunnett_stat_rows = []
    for col, ylabel in ep_metrics:
        long, order = _metric_long_df(loaded, col)
        if long is None:
            continue
        w, rot = _metric_panel_layout(order)
        set_style()
        fig, ax = plt.subplots(figsize=(w, 2.24))
        # fig, ax = plt.subplots(figsize=(2, 2))
        condition_plot(ax, long, "condition", col, palette, order=order,
                       ylabel=ylabel, rotation=rot, count_label="title", n_fish=n_fish)
        new_rows = _add_dunnett_annotations(ax, long, "condition", col, order, control_label)
        if new_rows:
            for r in new_rows:
                dunnett_stat_rows.append({"metric": col, **r})
        plt.tight_layout(pad=0.3)
        save(fig, os.path.join(gout, f"{col}_dunnett.pdf"))
    return dunnett_stat_rows


def _plot_metric_dunnett_combined(loaded, ep_metrics, gout, palette, control_label):
    """Compact one-row Dunnett plot set, matching the ablation-profile layout."""
    available = [
        (col, ylabel, *_metric_long_df(loaded, col))
        for col, ylabel in ep_metrics
    ]
    available = [
        (col, ylabel, long, order)
        for col, ylabel, long, order in available
        if long is not None
    ]
    if not available:
        return

    set_style()
    n_metrics = len(available)
    fig_w = max(8 / 3, n_metrics * 0.67)
    fig, axes = plt.subplots(1, n_metrics, figsize=(fig_w, 2.0), squeeze=False)
    axes = axes.flatten()

    for ax, (col, ylabel, long, order) in zip(axes, available):
        rot = 30 if len(order) >= 3 else 0
        condition_plot(ax, long, "condition", col, palette, order=order,
                       ylabel=None, rotation=rot, count_label="none")
        _add_dunnett_annotations(ax, long, "condition", col, order, control_label)
        ax.set_title(ylabel, fontsize=5.5, pad=2)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=5, pad=1)
        ax.tick_params(axis="y", labelsize=5)
        sns.despine(ax=ax)

    plt.tight_layout(pad=0.3, w_pad=0.35)
    save(fig, os.path.join(gout, "metrics_dunnett_combined.pdf"))


def _plot_event_time_panels(group_cfg, evals_dir, gout, palette, count_label):
    consumption_ct, biting_ct = [], []
    for spec_key, label in group_cfg["conditions"]:
        spec_dir = os.path.join(evals_dir, spec_key)
        ct = load_consumption_times(spec_dir)
        if ct:
            consumption_ct.append((label, ct))
        bt = load_biting_times(spec_dir)
        if bt:
            biting_ct.append((label, bt))
    if len(consumption_ct) >= 2:
        time_to_X_linear(
            consumption_ct, palette,
            os.path.join(gout, "time_to_consumption_linear.pdf"),
            count_label=count_label)
    if len(biting_ct) >= 2:
        time_to_X_linear(
            biting_ct, palette,
            os.path.join(gout, "time_to_bitten_linear.pdf"),
            event_label="Biting events",
            time_label="Time to biting event (s)",
            count_label=count_label)


def _plot_metric_correlation(loaded, ep_metrics, gout):
    """Exploratory correlation grid over all available episode metrics."""
    set_style()
    all_dfs = []
    for label, df in loaded:
        sub = df[[c for c, _ in ep_metrics if c in df.columns]].copy()
        sub["condition"] = label
        all_dfs.append(sub)
    combined = pd.concat(all_dfs, ignore_index=True)
    num_cols = [c for c, _ in ep_metrics if c in combined.columns]
    if len(num_cols) < 3:
        return

    corr = combined[num_cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(2.8, 2.4))
    sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 6},
                linewidths=0.3, square=True,
                xticklabels=[c.replace("_", "\n") for c in num_cols],
                yticklabels=[c.replace("_", "\n") for c in num_cols])
    ax.tick_params(labelsize=6)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(gout, "metric_correlations.pdf"))


def _write_mwu_stats(loaded, ep_metrics, gout, baseline_label, baseline_df):
    stat_rows = []
    for label, df in loaded:
        if label == baseline_label:
            continue
        for col, ylabel in ep_metrics:
            if col not in df.columns or col not in baseline_df.columns:
                continue
            a = baseline_df[col].dropna().values
            b = df[col].dropna().values
            if len(a) < 3 or len(b) < 3:
                continue
            stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            stat_rows.append({"metric": col, "vs": label, "p": p,
                               "mean_baseline": a.mean(), "mean_cond": b.mean()})
    if stat_rows:
        pd.DataFrame(stat_rows).to_csv(os.path.join(gout, "stats.csv"), index=False)


def run_group(group_name, group_cfg, evals_dir, out_dir):
    loaded = _load_group_episode_dfs(group_cfg, evals_dir)
    if len(loaded) < 2:
        print(f"  {group_name}: need ≥2 conditions — skipped")
        return

    ep_metrics = group_cfg["ep_metrics"]
    count_label = "bottom_right" if group_name == "collective_sensing" else "corner"
    palette = _group_palette(loaded)
    baseline_label, baseline_df = _resolve_baseline(loaded, group_cfg)
    gout = os.path.join(out_dir, group_name)
    os.makedirs(gout, exist_ok=True)

    _plot_metric_panels(loaded, ep_metrics, gout, palette, count_label)

    dunnett_stat_rows = _plot_metric_dunnett_panels(
        loaded, ep_metrics, gout, palette, count_label, baseline_label,
    )
    if dunnett_stat_rows:
        pd.DataFrame(dunnett_stat_rows).to_csv(
            os.path.join(gout, "stats_dunnett.csv"), index=False)
    _plot_metric_dunnett_combined(loaded, ep_metrics, gout, palette, baseline_label)

    _plot_event_time_panels(group_cfg, evals_dir, gout, palette, count_label)
    _plot_metric_correlation(loaded, ep_metrics, gout)

    if group_cfg.get("ablation_profiles"):
        _run_ablation_profiles(group_cfg, loaded, gout)
    if group_name == "muting":
        _run_muting_delta_profile(group_cfg, loaded, gout)

    _write_mwu_stats(loaded, ep_metrics, gout, baseline_label, baseline_df)


# ── main ─────────────────────────────────────────────────────────────────────

def run(evals_dir, out_dir=None):
    if out_dir is None:
        out_dir = os.path.join(evals_dir, "..", "multi_eval", "interventions")
    out_dir = os.path.normpath(out_dir)

    for group_name, group_cfg in GROUPS.items():
        print(f"▶ {group_name}")
        run_group(group_name, group_cfg, evals_dir, out_dir)

    print("Done.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals_dir", required=True,
                    help="Directory containing {spec_key}/ eval folders")
    ap.add_argument("--out_dir", default=None,
                    help="Output directory (default: <evals_dir>/../multi_eval/interventions)")
    args = ap.parse_args(argv)
    run(args.evals_dir, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
