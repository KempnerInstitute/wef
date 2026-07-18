"""
N-fish scaling analysis: RNN dimensionality and inter-agent distances vs. group size.

Multi-spec analysis — run after per-spec preprocessing. Reads from:
  evals/nfish{N}_m1a1k1_patchy_square/raw/rnn/ep{k}_rnn.npy
  evals/nfish{N}_m1a1k1_patchy_square/derived/per_env_ep_agent_step.pkl

Always produces plots for the k1 (knollen-on) condition. If k0 (knollen-off)
specs are also present, additionally produces per-condition and comparison plots.

Outputs go to multi_eval/nfish/ (or out_dir).

Usage:
    python analysis_nfish.py --evals_dir <run_dir>/evals [--out_dir ...] [--plot_only]
"""

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from analysis_style import (
    set_style,
    consumption_times_from_step_df, biting_times_from_step_df,
    save_event_time_plot_set, save as save_fig,
)
from utils_rnn import svd_evr, SVD_BACKEND
from cfg import PLOT_POINTS_BELOW_N
from utils_figsaving import _load_data, _save_data
from rnn_loader import iter_rnn_episodes


_EQUALIZE_N_EPISODES = 10
_MAX_PCA_ROWS = 10_000
_K1_RE = re.compile(r"^nfish(\d+)_m1a1k1_patchy_square$")
_K0_RE = re.compile(r"^nfish(\d+)_m1a1k0_patchy_square$")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _compact_legend(ax):
    ax.legend(
        frameon=False,
        loc="upper left",
        handlelength=1.0,
        handletextpad=0.35,
        labelspacing=0.2,
        borderaxespad=0.2,
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _downsample_rows(X: np.ndarray, max_rows: int = _MAX_PCA_ROWS) -> np.ndarray:
    if X.shape[0] <= max_rows:
        return X
    idxs = np.random.choice(X.shape[0], size=max_rows, replace=False)
    return X[idxs]


def _compute_pca_metrics(X: np.ndarray) -> dict:
    X = _downsample_rows(np.asarray(X, dtype=np.float32))
    evr = svd_evr(X)
    cumvar = np.cumsum(evr)

    reached = np.where(cumvar >= 0.90)[0]
    idx_cumvar = int(reached[0]) + 1 if len(reached) > 0 else -1  # 1-based

    D_eff = float(1.0 / np.sum(evr ** 2))

    from kneed import KneeLocator
    pcs = np.arange(1, len(evr) + 1)
    k_idx = KneeLocator(pcs, cumvar, curve="concave", direction="increasing").knee

    return {"cumvar": cumvar, "idx_cumvar": idx_cumvar, "D_eff": D_eff, "k_idx": k_idx}


def _episode_interagent_distance(
    df_ep: pd.DataFrame,
    stat: str = "median",
    timestep_pairwise_metric: str = "max",
) -> float:
    values = []
    for _, g in df_ep.groupby("time_step"):
        pos = g[["position_x", "position_y"]].to_numpy()
        if pos.shape[0] < 2:
            continue
        diffs = pos[:, None, :] - pos[None, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=-1))
        iu = np.triu_indices(pos.shape[0], k=1)
        pairwise = dists[iu]
        if timestep_pairwise_metric == "max":
            values.append(float(np.max(pairwise)))
        else:
            values.append(float(np.mean(pairwise)))
    if not values:
        return np.nan
    if stat == "mean":
        return float(np.mean(values))
    if stat == "median":
        return float(np.median(values))
    return float(np.max(values))


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _discover_nfish_specs(evals_dir: str) -> tuple:
    """Return (k1_specs, k0_specs) as {n_agents: Path} dicts."""
    evals_path = Path(evals_dir)
    k1_specs, k0_specs = {}, {}
    for spec_dir in sorted(evals_path.iterdir()):
        if not spec_dir.is_dir():
            continue
        name = spec_dir.name
        m = _K1_RE.match(name)
        if m:
            k1_specs[int(m.group(1))] = spec_dir
            continue
        m = _K0_RE.match(name)
        if m:
            k0_specs[int(m.group(1))] = spec_dir
    return dict(sorted(k1_specs.items())), dict(sorted(k0_specs.items()))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_rnn_episodes(spec_dir: Path, n_agents: int) -> list:
    """Return list of (T*n_agents, H) float32 arrays, one per (episode_file, env_thread)."""
    rnn_dir = spec_dir / "raw"
    episodes = []
    for _k, rnn_arr, _ in iter_rnn_episodes(rnn_dir):
        T, E, _A, H = rnn_arr.shape
        for e in range(E):
            ep_mat = rnn_arr[:, e, :n_agents, :].reshape(T * n_agents, H).astype(np.float32)
            episodes.append(ep_mat)
    return episodes


def _equalize_episodes(episodes_by_n: dict, n_target: int = _EQUALIZE_N_EPISODES,
                        random_state: int = 0) -> dict:
    rng = np.random.RandomState(random_state)
    out = {}
    for n_agents, eps in episodes_by_n.items():
        n_have = len(eps)
        if n_have > n_target:
            idxs = sorted(rng.choice(n_have, size=n_target, replace=False))
            out[n_agents] = [eps[i] for i in idxs]
        else:
            if n_have < n_target:
                print(f"[nfish] N={n_agents}: {n_have} episodes (< target {n_target}); using all.")
            out[n_agents] = eps
    return out


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def _compute_group_pca_stats(episodes_by_n: dict) -> dict:
    group_stats = {}
    for n_agents, eps in episodes_by_n.items():
        X = np.vstack(eps)
        print(f"[nfish] N={n_agents}: group PCA on ({X.shape[0]}, {X.shape[1]}) matrix")
        group_stats[n_agents] = _compute_pca_metrics(X)
    return group_stats


def _compute_interagent_distances_by_n(
    specs_by_n: dict,
    stat: str = "median",
    timestep_pairwise_metric: str = "max",
) -> dict:
    dist_by_n = {}
    for n_agents, spec_dir in specs_by_n.items():
        if n_agents < 2:
            dist_by_n[n_agents] = [np.nan]
            continue
        step_pkl = spec_dir / "derived" / "per_env_ep_agent_step.pkl"
        if not step_pkl.exists():
            print(f"[nfish] N={n_agents}: missing {step_pkl}; skipping distances")
            dist_by_n[n_agents] = []
            continue
        try:
            dff = pd.read_pickle(step_pkl)
        except Exception as e:
            print(f"[nfish] N={n_agents}: failed to load {step_pkl} ({e}); skipping distances")
            dist_by_n[n_agents] = []
            continue
        episode_vals = []
        for _, df_ep in tqdm.tqdm(
            dff.groupby(["env_id", "episode_index"]),
            desc=f"N={n_agents} inter-agent dist",
        ):
            val = _episode_interagent_distance(
                df_ep[["time_step", "agent_id", "position_x", "position_y"]],
                stat=stat,
                timestep_pairwise_metric=timestep_pairwise_metric,
            )
            if not np.isnan(val):
                episode_vals.append(val)
        dist_by_n[n_agents] = episode_vals
        med = np.median(episode_vals) if episode_vals else np.nan
        print(f"[nfish] N={n_agents}: {len(episode_vals)} episodes, median dist={med:.3f} cm")
    return dist_by_n


def _bootstrap_by_episode(episodes_by_n: dict, n_repeats: int = 30) -> dict:
    boot_results = {}
    for n_agents, eps in episodes_by_n.items():
        n_ep = len(eps)
        deff_list = []
        for _ in tqdm.trange(n_repeats, desc=f"N={n_agents} episode bootstrap"):
            idxs = np.random.randint(0, n_ep, size=n_ep)
            X = np.vstack([eps[i] for i in idxs])
            deff_list.append(_compute_pca_metrics(X)["D_eff"])
        boot_results[n_agents] = deff_list
    return boot_results


def _bootstrap_unit_subsampling(
    episodes_by_n: dict,
    subsample_percents: list,
    n_repeats: int = 30,
) -> dict:
    results = {n: {p: [] for p in subsample_percents} for n in episodes_by_n}
    for n_agents, eps in episodes_by_n.items():
        X = np.vstack(eps)
        n_units = X.shape[1]
        print(f"[nfish] N={n_agents}: unit subsampling on ({X.shape[0]}, {n_units}) matrix")
        for pct in tqdm.tqdm(subsample_percents, desc=f"N={n_agents} unit subsample"):
            n_cols = max(1, int(n_units * pct))
            for _ in range(n_repeats):
                cols = np.random.choice(n_units, size=n_cols, replace=False)
                results[n_agents][pct].append(_compute_pca_metrics(X[:, cols])["D_eff"])
    return results


# ---------------------------------------------------------------------------
# Single-condition plots
# ---------------------------------------------------------------------------

def _plot_cumvar(group_stats: dict, n_agents_list: list, out_dir: str,
                 label: str, metric_to_plot: str) -> None:
    fig, ax = plt.subplots(figsize=(2.4, 2.0))
    for n_agents in n_agents_list:
        if n_agents not in group_stats:
            continue
        stats = group_stats[n_agents]
        cumvar = stats["cumvar"]
        ax.plot(range(1, len(cumvar) + 1), cumvar, label=f"N={n_agents}")
        if metric_to_plot == "D_eff":
            ax.axvline(x=stats["D_eff"], linestyle=":", alpha=0.5)
        elif metric_to_plot == "idx_cumvar":
            ax.axvline(x=stats["idx_cumvar"], linestyle="--", alpha=0.5)
        elif metric_to_plot == "k_idx":
            ax.axvline(x=stats["k_idx"], linestyle="-.", alpha=0.5)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Cumulative Variance Explained")
    _compact_legend(ax)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, f"{label}_cumvar_{metric_to_plot}"))
    plt.close(fig)


def _plot_deff_vs_nagents(group_stats: dict, n_agents_list: list, out_dir: str,
                          label: str, metric_to_plot: str) -> None:
    n_vals = [n for n in n_agents_list if n in group_stats]
    m_vals = [group_stats[n][metric_to_plot] for n in n_vals]
    fig, ax = plt.subplots(figsize=(2.4, 2.0))
    ax.plot(n_vals, m_vals, marker="o")
    ax.set_xlabel("Number of agents")
    ax.set_ylabel(metric_to_plot)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, f"{label}_{metric_to_plot}_vs_nagents"))
    plt.close(fig)


def _plot_deff_boxplot(boot_results_env: dict, n_agents_list: list,
                       out_dir: str, label: str) -> None:
    present = [n for n in n_agents_list if n in boot_results_env]
    data = [np.array(boot_results_env[n]) for n in present]
    n_boot = min(len(v) for v in data)
    data = [v[:n_boot] for v in data]
    x_pos = np.arange(1, len(present) + 1)
    xlabels = [f"{n}\n(n_boot={n_boot})" for n in present]

    fig, ax = plt.subplots(figsize=(2.25, 2.25))
    for i in range(n_boot):
        ax.plot(x_pos, [data[j][i] for j in range(len(present))],
                color="gray", alpha=0.3, linewidth=1, zorder=0)
    ax.boxplot(
        data, positions=x_pos, labels=xlabels,
        widths=0.5, patch_artist=True,
        boxprops=dict(facecolor="white", edgecolor="black", linewidth=1),
        medianprops=dict(color="black", linewidth=1),
        whiskerprops=dict(color="black", linewidth=1),
        capprops=dict(color="black", linewidth=1),
    )
    ax.set_xlabel("Number of agents")
    ax.set_ylabel("Effective rank")
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, f"{label}_deff_boxplot_paired"))
    plt.close(fig)


def _plot_interagent_dist_boxplot(dist_by_n: dict, n_agents_list: list,
                                  out_dir: str, label: str) -> None:
    data, xlabels = [], []
    for n in n_agents_list:
        vals = list(dist_by_n.get(n, []))
        if n == 1:
            vals = [0.0]
        data.append(vals)
        xlabels.append(f"{n}\n(N={len(vals)})")
    fig, ax = plt.subplots(figsize=(1.6, 1.6))
    ax.boxplot(data, labels=xlabels)
    # Conditional point overlay
    if all(len(v) < PLOT_POINTS_BELOW_N for v in data):
        rng = np.random.default_rng(0)
        for i, vals in enumerate(data):
            if vals:
                jitter = rng.uniform(-0.15, 0.15, len(vals))
                ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
                           color="black", alpha=0.5, s=9, zorder=3)
    ax.set_xlabel("Number of agents (N)")
    ax.set_ylabel("Inter-agent distance [cm]")
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, f"{label}_interagent_dist_boxplot"))
    plt.close(fig)


def _plot_deff_unit_subsampling(boot_results_units: dict, n_agents_list: list,
                                subsample_percents: list, out_dir: str, label: str) -> None:
    fig, ax = plt.subplots(figsize=(2.25, 2.25))
    for n_agents in n_agents_list:
        if n_agents not in boot_results_units:
            continue
        x, y_mean, y_std = [], [], []
        for pct in subsample_percents:
            vals = np.array(boot_results_units[n_agents][pct])
            x.append(pct * 100.0)
            y_mean.append(np.mean(vals))
            y_std.append(np.std(vals, ddof=1))
        ax.errorbar(x, y_mean, yerr=y_std, marker="o", capsize=3,
                    elinewidth=1, label=f"N={n_agents}")
    ax.set_xlabel("% of RNN units sampled")
    ax.set_ylabel("Effective rank (±1 SD)")
    _compact_legend(ax)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, f"{label}_deff_unit_subsampling"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Comparison plots (k0 vs k1)
# ---------------------------------------------------------------------------

def _plot_comparison_deff_vs_nagents(results_by_condition: dict, metric_to_plot: str,
                                     out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(2.25, 2.25))
    for cond_label, results in results_by_condition.items():
        gs = results["group_stats"]
        n_vals = sorted(gs.keys())
        ax.plot(n_vals, [gs[n][metric_to_plot] for n in n_vals], marker="o", label=cond_label)
    ax.set_xlabel("Number of agents")
    ax.set_ylabel("Effective rank (+/- SD)")
    _compact_legend(ax)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, f"comparison_{metric_to_plot}_vs_nagents"))
    plt.close(fig)


def _plot_comparison_deff_boxplot(results_by_condition: dict, out_dir: str) -> None:
    condition_labels = list(results_by_condition.keys())
    n_conditions = len(condition_labels)
    n_agents_vals = sorted(
        set().union(*[set(r["boot_results_env"].keys()) for r in results_by_condition.values()])
    )
    offsets = np.linspace(-0.25, 0.25, n_conditions)
    width = 0.5 / max(1, n_conditions)

    # Collect per-(n_agents, condition) data for N labels and conditional overlay
    all_data_flat: dict[int, list] = {}   # n_agents → all vals across conditions
    cond_data: dict[str, tuple[list, list]] = {}  # cond → (data_to_plot, positions)
    for cond_label in condition_labels:
        boot = results_by_condition[cond_label]["boot_results_env"]
        dt, pos = [], []
        for n_idx, n_agents in enumerate(n_agents_vals, start=1):
            if n_agents not in boot:
                continue
            v = np.asarray(boot[n_agents])
            dt.append(v)
            pos.append(n_idx + offsets[condition_labels.index(cond_label)])
            all_data_flat.setdefault(n_agents, []).extend(v.tolist())
        cond_data[cond_label] = (dt, pos)

    n_labels = [f"{n}\n(N={len(all_data_flat.get(n, []))})" for n in n_agents_vals]
    max_box_n = max((len(v) for v in all_data_flat.values()), default=0)

    fig, ax = plt.subplots(figsize=(2.25, 2.25))
    for cond_idx, cond_label in enumerate(condition_labels):
        dt, pos = cond_data[cond_label]
        if not dt:
            continue
        color = f"C{cond_idx}"
        box = ax.boxplot(dt, positions=pos, widths=width, patch_artist=True)
        for patch in box["boxes"]:
            patch.set(facecolor="white", edgecolor=color, linewidth=1)
        for el in ["whiskers", "caps", "medians"]:
            for artist in box[el]:
                artist.set(color=color, linewidth=1)
        if max_box_n < PLOT_POINTS_BELOW_N:
            rng = np.random.default_rng(cond_idx)
            for vals, p in zip(dt, pos):
                ax.scatter(np.full(len(vals), p) + rng.uniform(-width * 0.3, width * 0.3, len(vals)),
                           vals, color=color, alpha=0.4, s=9, zorder=3)
        ax.plot([], [], color=color, label=cond_label)
    ax.set_xticks(range(1, len(n_agents_vals) + 1))
    ax.set_xticklabels(n_labels)
    ax.set_xlabel("Number of agents")
    ax.set_ylabel("Effective rank")
    _compact_legend(ax)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "comparison_deff_boxplot"))
    plt.close(fig)


def _plot_comparison_dist_boxplot(results_by_condition: dict, out_dir: str) -> None:
    condition_labels = list(results_by_condition.keys())
    n_conditions = len(condition_labels)
    n_agents_vals = sorted(
        set().union(*[set(r["dist_by_n"].keys()) for r in results_by_condition.values()])
    )
    offsets = np.linspace(-0.25, 0.25, n_conditions)
    width = 0.5 / max(1, n_conditions)

    all_dist_flat: dict[int, list] = {}
    cond_dist: dict[str, tuple[list, list]] = {}
    for cond_label in condition_labels:
        dist_by_n = results_by_condition[cond_label]["dist_by_n"]
        dt, pos = [], []
        for n_idx, n_agents in enumerate(n_agents_vals, start=1):
            vals = list(dist_by_n.get(n_agents, []))
            if n_agents == 1:
                vals = [0.0]
            if not vals:
                continue
            dt.append(vals)
            pos.append(n_idx + offsets[condition_labels.index(cond_label)])
            all_dist_flat.setdefault(n_agents, []).extend(vals)
        cond_dist[cond_label] = (dt, pos)

    n_labels = [f"{n}\n(N={len(all_dist_flat.get(n, []))})" for n in n_agents_vals]
    max_box_n = max((len(v) for v in all_dist_flat.values()), default=0)

    fig, ax = plt.subplots(figsize=(2.25, 2.25))
    for cond_idx, cond_label in enumerate(condition_labels):
        dt, pos = cond_dist[cond_label]
        if not dt:
            continue
        color = f"C{cond_idx}"
        box = ax.boxplot(dt, positions=pos, widths=width, patch_artist=True)
        for patch in box["boxes"]:
            patch.set(facecolor="white", edgecolor=color, linewidth=1)
        for el in ["whiskers", "caps", "medians"]:
            for artist in box[el]:
                artist.set(color=color, linewidth=1)
        if max_box_n < PLOT_POINTS_BELOW_N:
            rng = np.random.default_rng(cond_idx)
            for vals, p in zip(dt, pos):
                ax.scatter(np.full(len(vals), p) + rng.uniform(-width * 0.3, width * 0.3, len(vals)),
                           vals, color=color, alpha=0.4, s=9, zorder=3)
        ax.plot([], [], color=color, label=cond_label)
    ax.set_xticks(range(1, len(n_agents_vals) + 1))
    ax.set_xticklabels(n_labels)
    ax.set_xlabel("Number of agents (N)")
    ax.set_ylabel("Inter-agent distance [cm]")
    _compact_legend(ax)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "comparison_interagent_dist_boxplot"))
    plt.close(fig)


def _comparison_series(results_by_condition: dict, source_key: str,
                       special_n1_zero: bool = False):
    """Shared extractor for the comparison plots.

    Returns (n_agents_vals, condition_labels, per_cond, all_flat) where
    per_cond[cond][n_agents] is a float array of per-env values and
    all_flat[n_agents] is the pooled list across conditions (for N labels).
    """
    condition_labels = list(results_by_condition.keys())
    n_agents_vals = sorted(
        set().union(*[set(r[source_key].keys()) for r in results_by_condition.values()])
    )
    per_cond: dict[str, dict[int, np.ndarray]] = {}
    all_flat: dict[int, list] = {}
    for cond_label in condition_labels:
        src = results_by_condition[cond_label][source_key]
        d: dict[int, np.ndarray] = {}
        for n_agents in n_agents_vals:
            vals = list(np.asarray(src.get(n_agents, []), dtype=float).ravel())
            if special_n1_zero and n_agents == 1:
                vals = [0.0]
            if not vals:
                continue
            d[n_agents] = np.asarray(vals, dtype=float)
            all_flat.setdefault(n_agents, []).extend(vals)
        per_cond[cond_label] = d
    return n_agents_vals, condition_labels, per_cond, all_flat


def _plot_comparison_meansem(results_by_condition: dict, source_key: str,
                             ylabel: str, xlabel: str, fname_base: str,
                             out_dir: str, special_n1_zero: bool = False) -> None:
    """Mean ± error point-plot alternative to the box-and-whisker comparisons.

    Emits two files — ``{fname_base}_mean_sem`` and ``{fname_base}_mean_sd`` —
    sharing the data backing ``{fname_base}_boxplot`` (per-env values keyed by
    number of agents).
    """
    n_agents_vals, condition_labels, per_cond, all_flat = _comparison_series(
        results_by_condition, source_key, special_n1_zero=special_n1_zero)
    if not any(per_cond.values()):
        return

    n_conditions = len(condition_labels)
    offsets = np.linspace(-0.18, 0.18, n_conditions) if n_conditions > 1 else [0.0]
    n_labels = [f"{n}\n(N={len(all_flat.get(n, []))})" for n in n_agents_vals]
    x_index = {n: i + 1 for i, n in enumerate(n_agents_vals)}

    for errtype in ("sem", "sd"):
        fig, ax = plt.subplots(figsize=(2.25, 2.25))
        for cond_idx, cond_label in enumerate(condition_labels):
            d = per_cond[cond_label]
            xs, means, errs = [], [], []
            for n_agents in n_agents_vals:
                if n_agents not in d:
                    continue
                v = d[n_agents]
                sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
                xs.append(x_index[n_agents] + offsets[cond_idx])
                means.append(float(np.mean(v)))
                errs.append(sd / np.sqrt(len(v)) if errtype == "sem" else sd)
            if not xs:
                continue
            ax.errorbar(xs, means, yerr=errs, color=f"C{cond_idx}", marker="o",
                        ms=4, lw=1, capsize=2, label=cond_label)
        ax.set_xticks(range(1, len(n_agents_vals) + 1))
        ax.set_xticklabels(n_labels)
        ax.set_xlabel(xlabel)
        err_lbl = "SEM" if errtype == "sem" else "SD"
        ax.set_ylabel(f"{ylabel} (±{err_lbl})")
        _compact_legend(ax)
        fig.tight_layout()
        save_fig(fig, os.path.join(out_dir, f"{fname_base}_mean_{errtype}"))
        plt.close(fig)


def _plot_comparison_cumvar(results_by_condition: dict, metric_to_plot: str,
                            out_dir: str) -> None:
    n_agents_vals = sorted(
        set().union(*[set(r["group_stats"].keys()) for r in results_by_condition.values()])
    )
    condition_labels = list(results_by_condition.keys())
    for n_agents in n_agents_vals:
        fig, ax = plt.subplots(figsize=(2.4, 2.0))
        for cond_idx, cond_label in enumerate(condition_labels):
            gs = results_by_condition[cond_label]["group_stats"]
            if n_agents not in gs:
                continue
            stats = gs[n_agents]
            cumvar = stats["cumvar"]
            ax.plot(range(1, len(cumvar) + 1), cumvar, label=cond_label)
            if metric_to_plot == "D_eff":
                ax.axvline(x=stats["D_eff"], linestyle=":", alpha=0.5, color=f"C{cond_idx}")
            elif metric_to_plot == "idx_cumvar":
                ax.axvline(x=stats["idx_cumvar"], linestyle="--", alpha=0.5, color=f"C{cond_idx}")
            elif metric_to_plot == "k_idx":
                ax.axvline(x=stats["k_idx"], linestyle="-.", alpha=0.5, color=f"C{cond_idx}")
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Cumulative Variance Explained")
        _compact_legend(ax)
        fig.tight_layout()
        save_fig(fig, os.path.join(out_dir, f"comparison_cumvar_nagents{n_agents}_{metric_to_plot}"))
        plt.close(fig)


# ---------------------------------------------------------------------------
# Per-condition pipeline
# ---------------------------------------------------------------------------

def _run_condition(
    specs_by_n: dict,
    condition_label: str,
    out_dir: str,
    force_recompute: bool,
    n_repeats: int,
    metric_to_plot: str,
    do_unit_subsampling: bool,
    subsample_percents: list,
) -> dict:
    n_agents_list = sorted(specs_by_n.keys())
    cache = lambda suffix: os.path.join(out_dir, f"{condition_label}_{suffix}")

    def _cache_exists(base):
        return any(os.path.exists(base + ext) for ext in [".csv", ".json", ".pkl"])

    cache_hit = (not force_recompute and
                 all(_cache_exists(cache(k)) for k in ["group_stats", "dist_by_n", "boot_results_env"]))
    if cache_hit:
        group_stats      = _load_data(cache("group_stats"))
        dist_by_n        = _load_data(cache("dist_by_n"))
        boot_results_env = _load_data(cache("boot_results_env"))
        boot_results_units = (
            _load_data(cache("boot_results_units")) if do_unit_subsampling and _cache_exists(cache("boot_results_units")) else None
        )
    else:
        episodes_raw = {n: _load_rnn_episodes(spec_dir, n)
                        for n, spec_dir in specs_by_n.items()}
        episodes = _equalize_episodes(episodes_raw)

        group_stats = _compute_group_pca_stats(episodes)
        _save_data(group_stats, cache("group_stats"))

        dist_by_n = _compute_interagent_distances_by_n(specs_by_n)
        _save_data(dist_by_n, cache("dist_by_n"))

        boot_results_env = _bootstrap_by_episode(episodes, n_repeats=n_repeats)
        _save_data(boot_results_env, cache("boot_results_env"))

        if do_unit_subsampling:
            boot_results_units = _bootstrap_unit_subsampling(
                episodes, subsample_percents, n_repeats=n_repeats,
            )
            _save_data(boot_results_units, cache("boot_results_units"))
        else:
            boot_results_units = None

    # Unused in manuscript/figseed.tex and copy_to_figs.py:
    # _plot_cumvar(group_stats, n_agents_list, out_dir, condition_label, metric_to_plot)
    # _plot_deff_vs_nagents(group_stats, n_agents_list, out_dir, condition_label, metric_to_plot)
    # _plot_deff_boxplot(boot_results_env, n_agents_list, out_dir, condition_label)
    # _plot_interagent_dist_boxplot(dist_by_n, n_agents_list, out_dir, condition_label)
    if boot_results_units is not None and condition_label == "k1":
        _plot_deff_unit_subsampling(
            boot_results_units, n_agents_list, subsample_percents, out_dir, condition_label,
        )

    return {
        "n_agents_list": n_agents_list,
        "group_stats": group_stats,
        "dist_by_n": dist_by_n,
        "boot_results_env": boot_results_env,
        "boot_results_units": boot_results_units,
    }


# ---------------------------------------------------------------------------
# Foraging / biting time curves vs N
# ---------------------------------------------------------------------------

def _plot_time_to_events_by_n(specs_by_n: dict, out_dir: str, condition_label: str) -> None:
    """Time-to-nth-food and time-to-nth-bite curves, one line per N-fish group."""
    n_vals = sorted(specs_by_n.keys())
    palette = {f"N={n}": f"C{i}" for i, n in enumerate(n_vals)}

    consumption_ct, biting_ct = [], []
    for n in n_vals:
        spec_dir = specs_by_n[n]
        step_pkl = Path(spec_dir) / "derived" / "per_env_ep_agent_step.pkl"
        if not step_pkl.exists():
            print(f"[nfish] N={n}: missing step pkl — skipping event curves")
            continue
        df = pd.read_pickle(step_pkl)
        label = f"N={n}"
        ctimes = consumption_times_from_step_df(df)
        if ctimes:
            consumption_ct.append((label, ctimes))
        btimes = biting_times_from_step_df(df)
        if btimes:
            biting_ct.append((label, btimes))

    save_event_time_plot_set(
        consumption_ct, palette, out_dir, f"{condition_label}_time_to_consumption")
    save_event_time_plot_set(
        biting_ct, palette, out_dir, f"{condition_label}_time_to_bitten",
        event_label="Biting events", time_label="Time to biting event (s)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    evals_dir: str,
    out_dir: str = None,
    force_recompute: bool = False,
    do_unit_subsampling: bool = True,
    subsample_percents: list = None,
    n_repeats: int = 30,
    metric_to_plot: str = "D_eff",
) -> None:
    if subsample_percents is None:
        subsample_percents = [0.05, 0.15, 0.25, 0.5, 0.75, 1.0]

    plt.switch_backend("Agg")
    set_style()
    from matplotlib import rcParams
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(evals_dir), "multi_eval", "nfish")
    os.makedirs(out_dir, exist_ok=True)

    k1_specs, k0_specs = _discover_nfish_specs(evals_dir)

    if len(k1_specs) < 2:
        print(f"[nfish] fewer than 2 k1 nfish specs found in {evals_dir}; skipping.")
        return

    print(f"[nfish] SVD backend: {SVD_BACKEND}", flush=True)
    print(f"[nfish] k1 specs: N={sorted(k1_specs.keys())}")
    if k0_specs:
        print(f"[nfish] k0 specs: N={sorted(k0_specs.keys())}")
    else:
        print("[nfish] no k0 specs found; running k1 only.")

    condition_kwargs = dict(
        out_dir=out_dir,
        force_recompute=force_recompute,
        n_repeats=n_repeats,
        metric_to_plot=metric_to_plot,
        do_unit_subsampling=do_unit_subsampling,
        subsample_percents=subsample_percents,
    )

    results_by_condition = {}

    k1_results = _run_condition(k1_specs, "k1", **condition_kwargs)
    results_by_condition["Knollen on"] = k1_results

    if len(k0_specs) >= 2:
        k0_results = _run_condition(k0_specs, "k0", **condition_kwargs)
        results_by_condition["Knollen off"] = k0_results

        # Unused in manuscript/figseed.tex and copy_to_figs.py:
        _plot_comparison_deff_vs_nagents(results_by_condition, metric_to_plot, out_dir)
        _plot_comparison_deff_boxplot(results_by_condition, out_dir)
        _plot_comparison_dist_boxplot(results_by_condition, out_dir)
        _plot_comparison_meansem(results_by_condition, "boot_results_env",
                                 "Effective rank", "Number of agents",
                                 "comparison_deff", out_dir)
        _plot_comparison_meansem(results_by_condition, "dist_by_n",
                                 "Inter-agent distance [cm]", "Number of agents (N)",
                                 "comparison_interagent_dist", out_dir,
                                 special_n1_zero=True)
        _plot_comparison_cumvar(results_by_condition, metric_to_plot, out_dir)

    # Unused in manuscript/figseed.tex and copy_to_figs.py:
    # _plot_time_to_events_by_n(k1_specs, out_dir, "k1")
    # if len(k0_specs) >= 2:
    #     _plot_time_to_events_by_n(k0_specs, out_dir, "k0")

    print(f"[nfish] done — outputs in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="N-fish scaling analysis")
    parser.add_argument("--evals_dir", required=True,
                        help="Path to <run_dir>/evals/")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: <run_dir>/multi_eval/nfish/)")
    parser.add_argument("--force_recompute", action="store_true",
                        help="Ignore cached intermediates and recompute from raw RNN arrays")
    parser.add_argument("--no_unit_subsampling", action="store_true",
                        help="Skip unit-subsampling bootstrap")
    parser.add_argument("--n_repeats", type=int, default=30)
    parser.add_argument("--metric_to_plot", default="D_eff",
                        choices=["D_eff", "idx_cumvar", "k_idx"])
    args = parser.parse_args()
    run(
        evals_dir=args.evals_dir,
        out_dir=args.out_dir,
        force_recompute=args.force_recompute,
        do_unit_subsampling=not args.no_unit_subsampling,
        n_repeats=args.n_repeats,
        metric_to_plot=args.metric_to_plot,
    )
