"""
Pairwise interaction analysis: extract contiguous proximity segments, classify
them by relative orientation, and test size-asymmetry in chaser/chased roles.

Usage:
    python analysis_pairwise.py --spec_dir path/to/evals/m1a1k1_patchy_square

Outputs to {spec_dir}/analyses/pairwise/
  interactions.pkl                          — per-interaction summary DataFrame
  timestep_interactions.pkl                 — per-timestep data for each interaction
  interactions_summary.csv                  — class-count table
  polarization_histogram.pdf                — distribution of pairwise polarization
  interaction_duration_histogram.pdf        — distribution of interaction duration (steps)
  interaction_distance_histogram.pdf        — distribution of max inter-agent distance (interaction reach)
  chaser_size_by_role_boxplot.pdf           — Wilcoxon: chaser vs chased size
  chaser_size_by_role_hist2d.pdf            — 2-D histogram chaser × chased size
  chaser_size_by_role_heatmap.pdf           — % heatmap chaser × chased size (bite_network style)
  chaser_size_by_role_summary.csv           — Wilcoxon test result
  chaser_eod_by_role_boxplot.pdf            — Wilcoxon: chaser vs chased EOD rate (Hz)
  chaser_eod_by_role_summary.csv            — Wilcoxon test result
  eod_by_interaction_class.pdf              — mean ± SEM EOD rate per interaction class
  eod_by_interaction_class.csv              — per-class EOD mean/sem/n
  confrontation_size_hist2d.pdf             — unordered agent sizes during confrontations
  confrontation_size_heatmap.pdf            — % heatmap smaller × larger size (bite_network style)
  interaction_classes_by_distance.pdf       — stacked bar: interaction class vs distance
  samples/interaction_{n}_env{e}_ep{p}.pdf  — trajectory + distance + polarisation traces
  biting_rate_by_interaction_class.pdf     — P(has_bitten) per interaction class
  biter_size_advantage.pdf                 — histogram: biter_size − victim_size
  biter_size_by_role_boxplot.pdf           — Wilcoxon: biter vs victim size
  biter_size_by_role_summary.csv           — Wilcoxon test result
  biting_timing_in_interaction.pdf         — normalised time of biting within interaction
"""

import argparse
import os
import sys
import warnings
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from scipy.stats import wilcoxon
import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_style import set_style, panel, save, size_pct_heatmap, AGENT_COLORS, TIME_STEP_MS
from cfg import PLOT_POINTS_BELOW_N, SIM_FPS

OUT_SUBDIR = "analyses/pairwise"
N_SAMPLE_INTERACTIONS = 5   # trajectory plots to produce


# ── geometry helpers ──────────────────────────────────────────────────────────

def _pairwise_polarization(ori_a, ori_b):
    """Mean cosine of angle difference for two orientations (scalar)."""
    return float(np.cos(ori_a - ori_b))


def _classify(pos_a, pos_b, ori_a, ori_b,
              theta_close=np.pi / 4, theta_opposite=3 * np.pi / 4):
    """
    Classify a single timestep of an A–B interaction.
    Returns one of: 'confronting', 'chasing', 'fleeing', 'dispersing', 'unaligned'.
    (chasing = A chases B; fleeing = B chases A)
    """
    d = pos_b - pos_a
    b12 = np.arctan2(d[1], d[0])
    b21 = np.arctan2(-d[1], -d[0])
    r1 = np.arctan2(np.sin(ori_a - b12), np.cos(ori_a - b12))
    r2 = np.arctan2(np.sin(ori_b - b21), np.cos(ori_b - b21))
    if abs(r1) < theta_close and abs(r2) < theta_close:
        return "confronting"
    if abs(r1) < theta_close and abs(r2) > theta_opposite:
        return "chasing"
    if abs(r2) < theta_close and abs(r1) > theta_opposite:
        return "fleeing"
    if abs(r1) > theta_opposite and abs(r2) > theta_opposite:
        return "dispersing"
    return "unaligned"


# ── interaction extraction ────────────────────────────────────────────────────

def extract_interactions(dff,
                         contact_distance=2,
                         theta_close=np.pi / 4,
                         theta_opposite=3 * np.pi / 4,
                         reduced_threshold_distance=5):
    """
    Extract contiguous proximity segments for every agent pair in every episode.

    Returns
    -------
    interactions_df : DataFrame  — one row per interaction segment
    timestep_df     : DataFrame  — one row per timestep within each segment
    """
    dff = dff.sort_values(["env_id", "episode_index", "time_step", "agent_id"])
    has_emit = "emit_eod" in dff.columns  # bout EOD rate = mean(emit_eod) * fps

    irecords, trecords = [], []
    for (env_id, ep_idx), grp in tqdm.tqdm(
            dff.groupby(["env_id", "episode_index"]),
            desc="Extracting interactions"):
        agents = sorted(grp["agent_id"].unique())
        for a1, a2 in combinations(agents, 2):
            df1 = grp[grp["agent_id"] == a1].reset_index(drop=True)
            df2 = grp[grp["agent_id"] == a2].reset_index(drop=True)

            if len(df1) != len(df2) or not np.array_equal(
                    df1["time_step"].values, df2["time_step"].values):
                warnings.warn(f"Unaligned data env={env_id} ep={ep_idx} "
                              f"agents=({a1},{a2}) — skipping")
                continue

            ts = df1["time_step"].values
            mutual = df1["has_nearby"].values & df2["has_nearby"].values

            # find contiguous True segments
            segments = []
            start = None
            for i, v in enumerate(mutual):
                if v and start is None:
                    start = i
                elif not v and start is not None:
                    segments.append((start, i - 1))
                    start = None
            if start is not None:
                segments.append((start, len(mutual) - 1))

            last_end = None
            for seg_idx, (si, ei) in enumerate(segments):
                s1, s2 = df1.iloc[si:ei + 1], df2.iloc[si:ei + 1]
                if len(s1) == 0:
                    continue

                dists = s1["distance_to_nearest_agent"].values
                pols, classes = [], []
                pos1 = np.stack(s1["position"].values)
                pos2 = np.stack(s2["position"].values)
                ori1 = s1["orientation"].values
                ori2 = s2["orientation"].values

                for j in range(len(s1)):
                    pols.append(_pairwise_polarization(ori1[j], ori2[j]))
                    classes.append(_classify(pos1[j], pos2[j], ori1[j], ori2[j],
                                             theta_close, theta_opposite))

                dominant = Counter(classes).most_common(1)[0][0]
                close_classes = [c for c, d in zip(classes, dists)
                                 if d <= reduced_threshold_distance]
                dominant_filtered = (Counter(close_classes).most_common(1)[0][0]
                                     if close_classes else "unaligned")

                irecords.append({
                    "env_id": env_id,
                    "episode_index": ep_idx,
                    "interaction_index": seg_idx,
                    "focal_agent": a1,
                    "other_agent": a2,
                    "agent_size_a1": s1["agent_size"].iloc[0],
                    "agent_size_a2": s2["agent_size"].iloc[0],
                    "eod_rate_a1": (float(s1["emit_eod"].values.mean() * SIM_FPS)
                                    if has_emit else np.nan),
                    "eod_rate_a2": (float(s2["emit_eod"].values.mean() * SIM_FPS)
                                    if has_emit else np.nan),
                    "start_time_step": int(ts[si]),
                    "end_time_step": int(ts[ei]),
                    "duration_steps": ei - si + 1,
                    "time_since_last": (float(ts[si] - last_end)
                                        if last_end is not None else np.nan),
                    "has_bitten": bool(
                        np.any(s1["was_bitten"].values | s2["was_bitten"].values)),
                    "has_contact": bool(np.any(dists <= contact_distance)),
                    "has_eating": bool(
                        s1["eating_event"].any() or s2["eating_event"].any()),
                    "mean_distance": float(np.mean(dists)),
                    "max_distance": float(np.max(dists)),
                    "min_distance": float(np.min(dists)),
                    "mean_polarization": float(np.mean(pols)),
                    "dominant_interaction_class": dominant,
                    "dominant_interaction_class_filtered": dominant_filtered,
                })
                last_end = int(ts[ei])

                trecords.append(pd.DataFrame({
                    "env_id": env_id,
                    "episode_index": ep_idx,
                    "interaction_index": seg_idx,
                    "focal_agent": a1,
                    "other_agent": a2,
                    "time_step": ts[si:ei + 1],
                    "distance": dists,
                    "interaction_class": classes,
                    "polarization_pairwise": pols,
                    "position_a1": list(pos1),
                    "position_a2": list(pos2),
                    "orientation_a1": ori1,
                    "orientation_a2": ori2,
                    "was_bitten_a1": s1["was_bitten"].values,
                    "was_bitten_a2": s2["was_bitten"].values,
                    "bite_other_fish_a1": s1["bite_other_fish"].values,
                    "bite_other_fish_a2": s2["bite_other_fish"].values,
                }))

    interactions_df = pd.DataFrame(irecords)
    timestep_df = (pd.concat(trecords, ignore_index=True)
                   if trecords else pd.DataFrame())
    return interactions_df, timestep_df


# ── plots ─────────────────────────────────────────────────────────────────────

def _p_stars(p):
    if p is None or not np.isfinite(p):
        return "n.s."
    return ("****" if p < 1e-4 else "***" if p < 1e-3
            else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s.")


def plot_polarization_histogram(timestep_df, out_dir):
    set_style()
    vals = timestep_df["polarization_pairwise"].dropna().values
    median_val = float(np.median(vals))
    fig, ax = panel()
    ax.hist(vals, bins=30, range=(-1, 1), density=True,
            color=AGENT_COLORS[0], edgecolor="white", linewidth=0.3)
    ax.axvline(median_val, color="black", lw=1.2, label=f"median = {median_val:.2f}")
    ax.set_xlabel("Pairwise polarization")
    ax.set_ylabel("Density")
    ax.set_xlim(-1, 1)
    ax.text(0.95, 0.93, f"N={len(vals)}", transform=ax.transAxes,
            ha="right", fontsize=6,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5})
    ax.legend(frameon=False, fontsize=6)
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "polarization_histogram.pdf"))


def plot_interaction_duration_histogram(interactions_df, out_dir):
    """Histogram of interaction duration in timesteps."""
    set_style()
    vals = interactions_df["duration_steps"].dropna().values
    median_val = float(np.median(vals))
    fig, ax = panel()
    ax.hist(vals, bins=40, color=AGENT_COLORS[1], edgecolor="white", linewidth=0.3)
    ax.axvline(median_val, color="black", lw=1.2, label=f"median = {median_val:.0f}")
    ax.set_xlabel("Interaction duration (steps)")
    ax.set_ylabel("Count")
    ax.text(0.95, 0.93, f"N={len(vals)}", transform=ax.transAxes,
            ha="right", fontsize=6,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5})
    ax.legend(frameon=False, fontsize=6)
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "interaction_duration_histogram.pdf"))


def plot_interaction_distance_histogram(interactions_df, out_dir):
    """Histogram of max inter-agent distance reached per interaction (interaction 'reach')."""
    set_style()
    vals = interactions_df["max_distance"].dropna().values
    median_val = float(np.median(vals))
    fig, ax = panel()
    ax.hist(vals, bins=40, color=AGENT_COLORS[2], edgecolor="white", linewidth=0.3)
    ax.axvline(median_val, color="black", lw=1.2, label=f"median = {median_val:.1f} cm")
    ax.set_xlabel("Max distance during interaction (cm)")
    ax.set_ylabel("Count")
    ax.text(0.95, 0.93, f"N={len(vals)}", transform=ax.transAxes,
            ha="right", fontsize=6,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5})
    ax.legend(frameon=False, fontsize=6)
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "interaction_distance_histogram.pdf"))


def plot_chaser_size_by_role(interactions_df, out_dir):
    """Boxplot + Wilcoxon: chaser size vs chased size (paired)."""
    set_style()
    fc = "dominant_interaction_class_filtered"
    chasing = interactions_df[interactions_df[fc] == "chasing"]
    fleeing  = interactions_df[interactions_df[fc] == "fleeing"]

    pairs = pd.concat([
        pd.DataFrame({"chaser": chasing["agent_size_a1"].values,
                      "chased": chasing["agent_size_a2"].values}),
        pd.DataFrame({"chaser": fleeing["agent_size_a2"].values,
                      "chased": fleeing["agent_size_a1"].values}),
    ], ignore_index=True).dropna()

    result = {"test": "Wilcoxon", "statistic": np.nan, "pvalue": np.nan,
              "n_pairs": len(pairs), "stars": "n.s.", "diff_means": np.nan}

    if len(pairs) >= 5:
        try:
            stat, pval = wilcoxon(pairs["chaser"], pairs["chased"],
                                  alternative="two-sided", zero_method="pratt",
                                  mode="auto")
            result.update(statistic=float(stat), pvalue=float(pval),
                          stars=_p_stars(pval),
                          diff_means=float(pairs["chaser"].mean()
                                           - pairs["chased"].mean()))
        except Exception as e:
            warnings.warn(f"Wilcoxon failed: {e}")
    else:
        warnings.warn(f"Too few chasing/fleeing pairs ({len(pairs)}) for Wilcoxon test")

    print(f"  [pairwise] Wilcoxon chaser vs chased: "
          f"W={result['statistic']:.3f}  p={result['pvalue']:.4g}  "
          f"{result['stars']}  n={result['n_pairs']}")

    # Boxplot
    long = pd.melt(pairs.reset_index(), id_vars=["index"],
                   value_vars=["chaser", "chased"],
                   var_name="role", value_name="agent_size")
    long["role"] = long["role"].map({"chaser": "Chaser", "chased": "Chased"})

    role_order = ["Chaser", "Chased"]
    n_per_role = long.groupby("role")["agent_size"].count()
    n_map = {r: int(n_per_role.get(r, 0)) for r in role_order}
    n_labels = [f"{r}\n(N={n_map[r]})" for r in role_order]

    # fig, ax = panel(w=1.6, h=2.0)  # previous narrower size
    fig, ax = panel(w=2.0, h=2.0)
    sns.boxplot(x="role", y="agent_size", data=long, order=role_order,
                palette=[AGENT_COLORS[0], AGENT_COLORS[2]], ax=ax,
                width=0.5, linewidth=0.8, fliersize=0,
                boxprops=dict(alpha=0.75))
    if max(n_map.values(), default=0) < PLOT_POINTS_BELOW_N:
        sns.stripplot(x="role", y="agent_size", data=long, order=role_order,
                      ax=ax, color="black", alpha=0.5, jitter=0.2, size=3)
    ax.set_xticks(range(len(role_order)))
    ax.set_xticklabels(n_labels)
    ax.set_xlabel("")
    ax.set_ylabel("Agent size")
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + (yhi - ylo) * 0.12)
    if np.isfinite(result["pvalue"]):
        ax.text(0.5, 0.97, result["stars"], transform=ax.transAxes,
                ha="center", va="top", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5})
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "chaser_size_by_role_boxplot.pdf"))

    # Add per-role N to the summary CSV (n_pairs is total; add per-role counts)
    result["n_chaser"] = n_map["Chaser"]
    result["n_chased"] = n_map["Chased"]

    # 2-D histogram
    if len(pairs) >= 2:
        fig, ax = panel()
        h = ax.hist2d(pairs["chaser"].values, pairs["chased"].values,
                      bins=5, cmap="Blues", cmin=1)
        plt.colorbar(h[3], ax=ax, label="Count", shrink=0.8)
        ax.set_xlabel("Chaser size")
        ax.set_ylabel("Chased size")
        sns.despine(ax=ax)
        plt.tight_layout(pad=0.3)
        save(fig, os.path.join(out_dir, "chaser_size_by_role_hist2d.pdf"))

    # Save CSV
    pd.DataFrame([result]).to_csv(
        os.path.join(out_dir, "chaser_size_by_role_summary.csv"),
        index=False, sep="\t")
    return result


def plot_chaser_eod_by_role(interactions_df, out_dir):
    """Boxplot + Wilcoxon: chaser EOD rate vs chased EOD rate (paired, per bout).

    EOD rate is the per-bout mean(emit_eod) * fps stored on each interaction.
    Same chaser/chased role convention as plot_chaser_size_by_role.
    """
    if not {"eod_rate_a1", "eod_rate_a2"}.issubset(interactions_df.columns):
        print("  [pairwise] no eod_rate columns — chaser_eod_by_role skipped "
              "(re-run extract; spec may lack emit_eod)")
        return

    set_style()
    fc = "dominant_interaction_class_filtered"
    chasing = interactions_df[interactions_df[fc] == "chasing"]
    fleeing  = interactions_df[interactions_df[fc] == "fleeing"]

    pairs = pd.concat([
        pd.DataFrame({"chaser": chasing["eod_rate_a1"].values,
                      "chased": chasing["eod_rate_a2"].values}),
        pd.DataFrame({"chaser": fleeing["eod_rate_a2"].values,
                      "chased": fleeing["eod_rate_a1"].values}),
    ], ignore_index=True).dropna()

    result = {"test": "Wilcoxon", "statistic": np.nan, "pvalue": np.nan,
              "n_pairs": len(pairs), "stars": "n.s.", "diff_means": np.nan}

    if len(pairs) >= 5:
        try:
            stat, pval = wilcoxon(pairs["chaser"], pairs["chased"],
                                  alternative="two-sided", zero_method="pratt",
                                  mode="auto")
            result.update(statistic=float(stat), pvalue=float(pval),
                          stars=_p_stars(pval),
                          diff_means=float(pairs["chaser"].mean()
                                           - pairs["chased"].mean()))
        except Exception as e:
            warnings.warn(f"Wilcoxon failed: {e}")
    else:
        warnings.warn(f"Too few chasing/fleeing pairs ({len(pairs)}) for EOD Wilcoxon test")

    print(f"  [pairwise] Wilcoxon chaser vs chased EOD rate: "
          f"W={result['statistic']:.3f}  p={result['pvalue']:.4g}  "
          f"{result['stars']}  n={result['n_pairs']}")

    long = pd.melt(pairs.reset_index(), id_vars=["index"],
                   value_vars=["chaser", "chased"],
                   var_name="role", value_name="eod_rate")
    long["role"] = long["role"].map({"chaser": "Chaser", "chased": "Chased"})

    role_order = ["Chaser", "Chased"]
    n_per_role = long.groupby("role")["eod_rate"].count()
    n_map = {r: int(n_per_role.get(r, 0)) for r in role_order}
    n_labels = [f"{r}\n(N={n_map[r]})" for r in role_order]

    fig, ax = panel(w=1.6, h=2.0)
    sns.boxplot(x="role", y="eod_rate", data=long, order=role_order,
                palette=[AGENT_COLORS[0], AGENT_COLORS[2]], ax=ax,
                width=0.5, linewidth=0.8, fliersize=0,
                boxprops=dict(alpha=0.75))
    if max(n_map.values(), default=0) < PLOT_POINTS_BELOW_N:
        sns.stripplot(x="role", y="eod_rate", data=long, order=role_order,
                      ax=ax, color="black", alpha=0.5, jitter=0.2, size=3)
    ax.set_xticks(range(len(role_order)))
    ax.set_xticklabels(n_labels)
    ax.set_xlabel("")
    ax.set_ylabel("EOD rate (Hz)")
    if np.isfinite(result["pvalue"]):
        ax.text(0.5, 0.97, result["stars"], transform=ax.transAxes,
                ha="center", va="top", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5})
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "chaser_eod_by_role_boxplot.pdf"))

    result["n_chaser"] = n_map["Chaser"]
    result["n_chased"] = n_map["Chased"]
    pd.DataFrame([result]).to_csv(
        os.path.join(out_dir, "chaser_eod_by_role_summary.csv"),
        index=False, sep="\t")
    return result


_CLASS_ORDER = ["confronting", "chasing", "fleeing", "dispersing", "unaligned"]


def plot_eod_by_interaction_class(interactions_df, out_dir):
    """Bar (mean ± SEM) of per-bout EOD rate, by interaction class.

    Pools both agents' bout EOD rates (eod_rate_a1, eod_rate_a2) within each
    bout, grouped by the filtered dominant interaction class.
    """
    if not {"eod_rate_a1", "eod_rate_a2"}.issubset(interactions_df.columns):
        print("  [pairwise] no eod_rate columns — eod_by_interaction_class skipped")
        return

    fc = "dominant_interaction_class_filtered"
    long = pd.concat([
        interactions_df[[fc, "eod_rate_a1"]].rename(columns={"eod_rate_a1": "eod_rate"}),
        interactions_df[[fc, "eod_rate_a2"]].rename(columns={"eod_rate_a2": "eod_rate"}),
    ], ignore_index=True).dropna(subset=["eod_rate"])
    long = long[long[fc].notna()]
    if len(long) == 0:
        print("  [pairwise] no EOD observations to plot by class — skipped")
        return

    present = [c for c in _CLASS_ORDER if c in set(long[fc])]
    present += [c for c in sorted(set(long[fc])) if c not in present]

    stats = (long.groupby(fc)["eod_rate"]
             .agg(mean="mean", sem=lambda x: x.std(ddof=1) / np.sqrt(len(x)), n="count")
             .reindex(present))

    set_style()
    fig, ax = panel(w=2.4, h=2.0)
    x = np.arange(len(present))
    ax.bar(x, stats["mean"].values, yerr=stats["sem"].values,
           color=AGENT_COLORS[0], alpha=0.8, capsize=3, linewidth=0.8,
           error_kw={"linewidth": 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(N={int(stats.loc[c, 'n'])})" for c in present],
                       rotation=30, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("EOD rate (Hz)")
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "eod_by_interaction_class.pdf"))

    stats.to_csv(os.path.join(out_dir, "eod_by_interaction_class.csv"), sep="\t")
    print("  [pairwise] EOD rate by class (Hz): "
          + ", ".join(f"{c}={stats.loc[c, 'mean']:.2f}" for c in present))
    return stats


def plot_confrontation_size_hist2d(interactions_df, out_dir):
    """Unordered (smaller, larger) agent size pairs during confrontations."""
    set_style()
    fc = "dominant_interaction_class_filtered"
    conf = interactions_df[interactions_df[fc] == "confronting"]
    if len(conf) < 2:
        print("  [pairwise] not enough confrontations for size hist2d — skipped")
        return
    ordered = np.sort(conf[["agent_size_a1", "agent_size_a2"]].values, axis=1)
    fig, ax = panel()
    h = ax.hist2d(ordered[:, 0], ordered[:, 1], bins=5, cmap="Blues", cmin=1)
    plt.colorbar(h[3], ax=ax, label="Count", shrink=0.8)
    ax.set_xlabel("Smaller agent size")
    ax.set_ylabel("Larger agent size")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "confrontation_size_hist2d.pdf"))


def plot_chaser_size_by_role_heatmap(interactions_df, out_dir):
    """% heatmap of chaser × chased size bins (analysis_bite_network style).

    A new version of chaser_size_by_role_hist2d: quantile size bins shared
    across both axes, rendered as a percentage-annotated seaborn heatmap with
    N observations in the title (matches plot_biting_heatmap)."""
    fc = "dominant_interaction_class_filtered"
    chasing = interactions_df[interactions_df[fc] == "chasing"]
    fleeing = interactions_df[interactions_df[fc] == "fleeing"]
    pairs = pd.concat([
        pd.DataFrame({"chaser": chasing["agent_size_a1"].values,
                      "chased": chasing["agent_size_a2"].values}),
        pd.DataFrame({"chaser": fleeing["agent_size_a2"].values,
                      "chased": fleeing["agent_size_a1"].values}),
    ], ignore_index=True).dropna()
    if size_pct_heatmap(
        pairs["chaser"], pairs["chased"],
        os.path.join(out_dir, "chaser_size_by_role_heatmap.pdf"),
        cmap="Blues", 
        xlabel="Chased size (bin)", 
        ylabel="Chaser size (bin)",
        title="Chase interactions %",
    ) is None:
        print("  [pairwise] chaser_size_by_role_heatmap — skipped (too few pairs)")


def plot_confrontation_size_heatmap(interactions_df, out_dir):
    """% heatmap of (smaller, larger) agent size during confrontations.

    A new version of confrontation_size_hist2d in analysis_bite_network style:
    quantile size bins, percentage annotations, N observations in the title."""
    fc = "dominant_interaction_class_filtered"
    conf = interactions_df[interactions_df[fc] == "confronting"]
    if len(conf) < 2:
        print("  [pairwise] not enough confrontations for size heatmap — skipped")
        return
    ordered = np.sort(conf[["agent_size_a1", "agent_size_a2"]].values, axis=1)
    size_pct_heatmap(
        ordered[:, 1], ordered[:, 0],
        os.path.join(out_dir, "confrontation_size_heatmap.pdf"),
        cmap="Blues", xlabel="Smaller agent size (bin)",
        ylabel="Larger agent size (bin)", title="Confrontation %",
    )


_CLASS_COLORS = {
    "Chasing":     "#377EB8",   # blue
    "Confronting": "#E41A1C",   # red
    "Dispersing":  "#FF7F00",   # orange
    "Fleeing":     "#4DAF4A",   # green
    "Unaligned":   "#984EA3",   # purple
}


def plot_interaction_classes_by_distance(timestep_df, out_dir):
    """Stacked bar: interaction class frequency binned by 1-cm distance bins."""
    set_style()
    df = timestep_df[["distance", "interaction_class"]].dropna().copy()
    df["interaction_class"] = (df["interaction_class"]
                               .str.replace("_", " ").str.title())
    df["distance_bin"] = (df["distance"] // 1) * 1
    counts = (df.groupby(["distance_bin", "interaction_class"])
              .size().unstack(fill_value=0).sort_index())

    col_colors = [_CLASS_COLORS.get(c, "#888888") for c in counts.columns]
    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    counts.plot(kind="bar", stacked=True, ax=ax, color=col_colors, width=1.0)
    ncol = int(np.ceil(counts.shape[1] / 2))
    ax.legend(title="Interaction class", ncol=ncol,
              loc="upper center", bbox_to_anchor=(0.5, 1.25),
              frameon=True, handlelength=1.0, fontsize=6)
    ax.set_xlabel(r"$d_{AB}$ (cm)")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "interaction_classes_by_distance.pdf"))


def plot_biting_rate_by_interaction_class(interactions_df, out_dir):
    """Bar: fraction of interactions that contain biting, per dominant interaction class."""
    if interactions_df.empty or "has_bitten" not in interactions_df.columns:
        return
    fc = "dominant_interaction_class_filtered"
    rate = (interactions_df.groupby(fc)["has_bitten"]
            .agg(["mean", "count"])
            .reset_index())
    rate.columns = ["interaction_class", "biting_rate", "n"]
    rate = rate.sort_values("biting_rate", ascending=False)
    if rate.empty:
        return

    set_style()
    n_bars = len(rate)
    # fig, ax = panel(w=max(1.6, n_bars * 0.6), h=2.0)  # previous width scaled with # bars
    fig, ax = panel(w=2.0, h=2.0)
    colors = [AGENT_COLORS[i % len(AGENT_COLORS)] for i in range(n_bars)]
    pct = rate["biting_rate"] * 100
    ax.bar(range(n_bars), pct, color=colors, width=0.6, alpha=0.85)
    ax.set_xticks(range(n_bars))
    ax.set_xticklabels(
        [f"{c.replace('_', ' ').title()}\n(N={n})"
         for c, n in zip(rate["interaction_class"], rate["n"])],
        rotation=20, ha="right", fontsize=6,
    )
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_ylabel("% interactions with biting")
    ax.set_ylim(0, min(115, pct.max() * 1.4 + 5))
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "biting_rate_by_interaction_class.pdf"))


def plot_biter_size_advantage(interactions_df, timestep_df, out_dir):
    """Histogram: biter_size − victim_size for within-interaction biting events.
    Positive values = larger fish bites smaller fish."""
    if timestep_df.empty or interactions_df.empty:
        return
    keys = ["env_id", "episode_index", "interaction_index", "focal_agent", "other_agent"]
    ts = timestep_df.merge(
        interactions_df[keys + ["agent_size_a1", "agent_size_a2"]],
        on=keys, how="left",
    )
    diffs = []
    # was_bitten_a1 → a2 bites a1: biter_size = a2, victim_size = a1
    ba1 = ts[ts["was_bitten_a1"].astype(bool)]
    if len(ba1):
        diffs.append(ba1["agent_size_a2"] - ba1["agent_size_a1"])
    # was_bitten_a2 → a1 bites a2: biter_size = a1, victim_size = a2
    ba2 = ts[ts["was_bitten_a2"].astype(bool)]
    if len(ba2):
        diffs.append(ba2["agent_size_a1"] - ba2["agent_size_a2"])
    if not diffs:
        print("  biter_size_advantage: no biting events — skipped")
        return
    size_diff = pd.concat(diffs).dropna().values
    if len(size_diff) < 3:
        print("  biter_size_advantage: too few events — skipped")
        return

    set_style()
    fig, ax = panel()
    ax.hist(size_diff, bins=20, density=True, color=AGENT_COLORS[0],
            edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(float(np.mean(size_diff)), color="black", lw=1.2,
               label=f"mean = {np.mean(size_diff):.3f}")
    ax.set_xlabel("Biter size − Victim size")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=6)
    ax.text(0.95, 0.93, f"N={len(size_diff)}", transform=ax.transAxes,
            ha="right", fontsize=6,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5})
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "biter_size_advantage.pdf"))


def plot_biter_size_by_role(interactions_df, timestep_df, out_dir):
    """Boxplot + Wilcoxon: biter size vs victim size (per biting-event timestep)."""
    if timestep_df.empty or interactions_df.empty:
        return
    keys = ["env_id", "episode_index", "interaction_index", "focal_agent", "other_agent"]
    ts = timestep_df.merge(
        interactions_df[keys + ["agent_size_a1", "agent_size_a2"]],
        on=keys, how="left",
    )
    biters, victims = [], []
    ba1 = ts[ts["was_bitten_a1"].astype(bool)]
    if len(ba1):
        biters.append(ba1["agent_size_a2"])
        victims.append(ba1["agent_size_a1"])
    ba2 = ts[ts["was_bitten_a2"].astype(bool)]
    if len(ba2):
        biters.append(ba2["agent_size_a1"])
        victims.append(ba2["agent_size_a2"])
    if not biters:
        print("  biter_size_by_role: no biting events — skipped")
        return
    pairs = pd.DataFrame({
        "biter":  pd.concat(biters).values,
        "victim": pd.concat(victims).values,
    }).dropna()
    if len(pairs) < 3:
        print("  biter_size_by_role: too few events — skipped")
        return

    result = {"test": "Wilcoxon", "statistic": np.nan, "pvalue": np.nan,
              "n_pairs": len(pairs), "stars": "n.s.", "diff_means": np.nan}
    if len(pairs) >= 5:
        try:
            stat, pval = wilcoxon(pairs["biter"], pairs["victim"],
                                  alternative="two-sided", zero_method="pratt",
                                  mode="auto")
            result.update(statistic=float(stat), pvalue=float(pval),
                          stars=_p_stars(pval),
                          diff_means=float(pairs["biter"].mean() - pairs["victim"].mean()))
        except Exception as e:
            warnings.warn(f"Wilcoxon failed: {e}")
    else:
        warnings.warn(f"Too few biting events ({len(pairs)}) for Wilcoxon test")

    print(f"  [pairwise] Wilcoxon biter vs victim: "
          f"W={result['statistic']:.3f}  p={result['pvalue']:.4g}  "
          f"{result['stars']}  n={result['n_pairs']}")

    long = pd.melt(pairs.reset_index(), id_vars=["index"],
                   value_vars=["biter", "victim"],
                   var_name="role", value_name="agent_size")
    long["role"] = long["role"].map({"biter": "Biter", "victim": "Victim"})

    role_order = ["Biter", "Victim"]
    n_per_role = long.groupby("role")["agent_size"].count()
    n_map = {r: int(n_per_role.get(r, 0)) for r in role_order}
    n_labels = [f"{r}\n(N={n_map[r]})" for r in role_order]

    set_style()
    fig, ax = panel(w=1.6, h=2.0)
    sns.boxplot(x="role", y="agent_size", data=long, order=role_order,
                palette=[AGENT_COLORS[0], AGENT_COLORS[2]], ax=ax,
                width=0.5, linewidth=0.8, fliersize=0,
                boxprops=dict(alpha=0.75))
    if max(n_map.values(), default=0) < PLOT_POINTS_BELOW_N:
        sns.stripplot(x="role", y="agent_size", data=long, order=role_order,
                      ax=ax, color="black", alpha=0.5, jitter=0.2, size=3)
    ax.set_xticks(range(len(role_order)))
    ax.set_xticklabels(n_labels)
    ax.set_xlabel("")
    ax.set_ylabel("Agent size")
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + (yhi - ylo) * 0.12)
    if np.isfinite(result["pvalue"]):
        ax.text(0.5, 0.97, result["stars"], transform=ax.transAxes,
                ha="center", va="top", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5})
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "biter_size_by_role_boxplot.pdf"))

    result["n_biter"] = n_map["Biter"]
    result["n_victim"] = n_map["Victim"]
    pd.DataFrame([result]).to_csv(
        os.path.join(out_dir, "biter_size_by_role_summary.csv"),
        index=False, sep="\t")
    return result


def plot_biting_timing_in_interaction(interactions_df, timestep_df, out_dir):
    """Histogram: when within an interaction does biting occur (normalised time 0→1)."""
    if timestep_df.empty or interactions_df.empty:
        return
    keys = ["env_id", "episode_index", "interaction_index", "focal_agent", "other_agent"]
    ts = timestep_df.merge(
        interactions_df[keys + ["start_time_step", "duration_steps"]],
        on=keys, how="left",
    )
    bitten = ts[ts["was_bitten_a1"].astype(bool) | ts["was_bitten_a2"].astype(bool)].copy()
    if len(bitten) < 5:
        print("  biting_timing: too few biting events — skipped")
        return
    bitten["norm_t"] = ((bitten["time_step"] - bitten["start_time_step"])
                        / bitten["duration_steps"].clip(lower=1))

    set_style()
    fig, ax = panel()
    ax.hist(bitten["norm_t"].clip(0, 1), bins=20, density=True,
            color=AGENT_COLORS[1], edgecolor="white", linewidth=0.3)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--", label=f"uniform \nN={len(bitten)}")
    ax.set_xlabel("Normalised time within interaction\n(0 = start, 1 = end)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=6)
    # ax.text(0.95, 0.93, f"N={len(bitten)}", transform=ax.transAxes,
    #         ha="right", fontsize=6,
    #         bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5})
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "biting_timing_in_interaction.pdf"))


def plot_sample_interactions(interactions_df, timestep_df, samples_dir,
                             n=N_SAMPLE_INTERACTIONS):
    """Trajectory + distance + polarisation plots for n sample interactions."""
    set_style()
    # Pick confronting first, then any
    fc = "dominant_interaction_class_filtered"
    conf = interactions_df[interactions_df[fc] == "confronting"]
    sample = (conf if len(conf) >= n else interactions_df).head(n)

    for _, rec in sample.iterrows():
        seg = timestep_df[
            (timestep_df["env_id"] == rec["env_id"]) &
            (timestep_df["episode_index"] == rec["episode_index"]) &
            (timestep_df["interaction_index"] == rec["interaction_index"]) &
            (timestep_df["focal_agent"] == rec["focal_agent"])
        ]
        if len(seg) == 0:
            continue

        pos1 = np.stack(seg["position_a1"].values)
        pos2 = np.stack(seg["position_a2"].values)
        ori1 = seg["orientation_a1"].values
        ori2 = seg["orientation_a2"].values
        dists = seg["distance"].values
        pols  = seg["polarization_pairwise"].values
        t     = np.arange(len(seg))
        a1, a2 = int(rec["focal_agent"]), int(rec["other_agent"])

        norm = mcolors.Normalize(0, max(1, len(seg) - 1))
        c1 = mcm.Blues(norm(t))
        c2 = mcm.Greens(norm(t))

        fig = plt.figure(figsize=(4.0, 4.8), constrained_layout=True)
        gs  = fig.add_gridspec(4, 1, height_ratios=[3, 0.4, 1, 1], hspace=0.05)
        ax_t = fig.add_subplot(gs[0])
        ax_sp = fig.add_subplot(gs[1]);  ax_sp.axis("off")
        ax_d = fig.add_subplot(gs[2])
        ax_p = fig.add_subplot(gs[3], sharex=ax_d)

        # quiver trajectories
        scale = max(np.linalg.norm(pos2 - pos1, axis=1).max() * 0.15, 0.3)
        for j in range(len(seg)):
            ax_t.quiver(pos1[j, 0], pos1[j, 1],
                        scale * np.cos(ori1[j]), scale * np.sin(ori1[j]),
                        angles="xy", scale_units="xy", scale=1,
                        color=c1[j], alpha=0.8, width=0.008)
            ax_t.quiver(pos2[j, 0], pos2[j, 1],
                        scale * np.cos(ori2[j]), scale * np.sin(ori2[j]),
                        angles="xy", scale_units="xy", scale=1,
                        color=c2[j], alpha=0.8, width=0.008)
        # bitten markers
        for idx, col, marker in [(seg["was_bitten_a1"].values, c1[-1], "o"),
                                  (seg["was_bitten_a2"].values, c2[-1], "x")]:
            bitten = np.where(idx)[0]
            if len(bitten):
                ax_t.scatter(pos1[bitten, 0] if marker == "o" else pos2[bitten, 0],
                             pos1[bitten, 1] if marker == "o" else pos2[bitten, 1],
                             color="red", marker=marker, s=40, zorder=5)
        ax_t.set_aspect("equal")
        ax_t.set_xlabel("x (cm)", fontsize=7)
        ax_t.set_ylabel("y (cm)", fontsize=7)
        ax_t.set_title(f"A{a1}–A{a2}  env={rec['env_id']}  ep={rec['episode_index']}  "
                       f"[{rec['dominant_interaction_class_filtered']}]", fontsize=7)
        legend = [Line2D([0], [0], color=c1[len(c1)//2], lw=2, label=f"A{a1}"),
                  Line2D([0], [0], color=c2[len(c2)//2], lw=2, label=f"A{a2}")]
        ax_t.legend(handles=legend, fontsize=6, frameon=False)
        sns.despine(ax=ax_t)

        ax_d.plot(t * TIME_STEP_MS / 1000, dists, color="purple", lw=1)
        ax_d.set_ylabel("Dist (cm)", fontsize=7)
        ax_d.set_ylim(bottom=0)
        ax_d.tick_params(labelbottom=False)
        sns.despine(ax=ax_d)

        ax_p.plot(t * TIME_STEP_MS / 1000, pols, color="teal", lw=1)
        ax_p.axhline(0, color="gray", lw=0.5, ls="--")
        ax_p.set_ylabel("Polarization", fontsize=7)
        ax_p.set_xlabel("Time (s)", fontsize=7)
        ax_p.set_ylim(-1.05, 1.05)
        sns.despine(ax=ax_p)

        fname = os.path.join(samples_dir,
                             f"interaction_{rec['interaction_index']}"
                             f"_env{int(rec['env_id'])}"
                             f"_ep{int(rec['episode_index'])}.pdf")
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved → {fname}")


# ── main ──────────────────────────────────────────────────────────────────────

def load(spec_dir):
    step_pkl    = os.path.join(spec_dir, "derived", "per_env_ep_agent_step.pkl")
    out_dir     = os.path.join(spec_dir, OUT_SUBDIR)
    samples_dir = os.path.join(out_dir, "samples")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    int_pkl = os.path.join(out_dir, "interactions.pkl")
    ts_pkl  = os.path.join(out_dir, "timestep_interactions.pkl")
    if os.path.exists(int_pkl) and os.path.exists(ts_pkl):
        print(f"Loading cached interactions from {out_dir} \u2026")
        interactions_df = pd.read_pickle(int_pkl)
        timestep_df     = pd.read_pickle(ts_pkl)
    else:
        if not os.path.exists(step_pkl):
            sys.exit(f"Not found: {step_pkl}")
        print(f"Loading {step_pkl} \u2026")
        dff = pd.read_pickle(step_pkl)
        print(f"  {len(dff):,} rows, {dff['agent_id'].nunique()} agents")
        interactions_df, timestep_df = extract_interactions(dff)
        interactions_df.to_pickle(int_pkl)
        timestep_df.to_pickle(ts_pkl)
        print(f"  {len(interactions_df):,} interactions")
    return {
        "interactions_df": interactions_df, "timestep_df": timestep_df,
        "out_dir": out_dir, "samples_dir": samples_dir,
    }


def run(spec_dir, force_recompute=False):
    step_pkl = os.path.join(spec_dir, "derived", "per_env_ep_agent_step.pkl")

    out_dir     = os.path.join(spec_dir, OUT_SUBDIR)
    samples_dir = os.path.join(out_dir, "samples")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    int_pkl  = os.path.join(out_dir, "interactions.pkl")
    ts_pkl   = os.path.join(out_dir, "timestep_interactions.pkl")
    csv_path = os.path.join(out_dir, "interactions_summary.csv")

    if not force_recompute and os.path.exists(int_pkl) and os.path.exists(ts_pkl):
        print(f"Loading cached interactions from {out_dir} …")
        interactions_df = pd.read_pickle(int_pkl)
        timestep_df     = pd.read_pickle(ts_pkl)
    else:
        if not os.path.exists(step_pkl):
            sys.exit(f"Not found: {step_pkl}")
        print(f"Loading {step_pkl} …")
        dff = pd.read_pickle(step_pkl)
        print(f"  {len(dff):,} rows, {dff['agent_id'].nunique()} agents")
        interactions_df, timestep_df = extract_interactions(dff)
        interactions_df.to_pickle(int_pkl)
        timestep_df.to_pickle(ts_pkl)
        print(f"  {len(interactions_df):,} interactions, "
              f"{len(timestep_df):,} interaction-timesteps")

    if len(interactions_df) == 0:
        print("[pairwise] no interactions found — nothing to plot")
        return

    # Summary table
    vc = interactions_df["dominant_interaction_class_filtered"].value_counts()
    vc.to_csv(csv_path, header=["count"])
    print("  Class distribution (filtered):")
    print(vc.to_string()); print()

    # Figures
    plot_polarization_histogram(timestep_df, out_dir)
    plot_interaction_duration_histogram(interactions_df, out_dir)
    plot_interaction_distance_histogram(interactions_df, out_dir)
    plot_chaser_size_by_role(interactions_df, out_dir)
    plot_chaser_size_by_role_heatmap(interactions_df, out_dir)
    plot_chaser_eod_by_role(interactions_df, out_dir)
    plot_eod_by_interaction_class(interactions_df, out_dir)
    plot_confrontation_size_hist2d(interactions_df, out_dir)
    plot_confrontation_size_heatmap(interactions_df, out_dir)
    plot_interaction_classes_by_distance(timestep_df, out_dir)
    # Biting-network-inspired additions
    plot_biting_rate_by_interaction_class(interactions_df, out_dir)
    plot_biter_size_advantage(interactions_df, timestep_df, out_dir)
    plot_biter_size_by_role(interactions_df, timestep_df, out_dir)
    plot_biting_timing_in_interaction(interactions_df, timestep_df, out_dir)
    plot_sample_interactions(interactions_df, timestep_df, samples_dir)
    print("Done.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec_dir", required=True,
                    help="Path to evals/<spec_key>")
    ap.add_argument("--force_recompute", action="store_true",
                    help="Ignore cached interactions.pkl and recompute from step pkl")
    args = ap.parse_args(argv)
    run(args.spec_dir, force_recompute=args.force_recompute)


if __name__ == "__main__":
    main()
