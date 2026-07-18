"""
Per-episode PCA of RNN hidden states.  Fig 6A.

Fits PCA independently on each (env_id, episode_index) pair using only the
rnn.npy files — no behavioral pkl required.  Aggregates cumulative-variance
curves, effective rank, and n_pcs_90 across episodes.

Usage
-----
    python analysis_rnn_dim.py --spec_dir path/to/evals/nfish4_m1a1k1_patchy_square

Outputs to {spec_dir}/analyses/rnn_dim/
    pca_per_episode_cumvar.pdf         mean ± SEM band
    pca_per_episode_cumvar_sd.pdf      mean ± SD band (shows episode-level spread)
    pca_per_episode_cumvar.csv    columns: pc_idx, mean_cumvar, sem_cumvar
    pca_summary.pdf               effective rank D_eff + n_pcs_90 boxplot
    pca_summary.csv               columns: episode_index, env_id, D_eff, n_pcs_90
    pca_per_pair_cumvar.csv       columns: episode_index, env_id, pc_1 … pc_H

Notes on SEM vs SD bands
------------------------
SEM (= SD / sqrt(n)) shrinks with sample size and shows uncertainty in the mean
estimate.  With n=30 episodes it is ~5× tighter than SD.  When all episodes are
consistent it becomes essentially invisible on a [0,1] axis, which is technically
correct but visually uninformative.

SD shows the actual spread of individual episode curves.  It is more honest about
episode-to-episode variability and stays constant as n grows, making it easier to
judge consistency by eye.  The downside is that it does not shrink with more data,
so it conflates genuine trajectory variability with estimation uncertainty.

For this figure the SD band is preferred for display: if the curves are genuinely
consistent (as observed), the SD band will also be tight — and that consistency is
visible rather than hidden.  The SEM variant is retained as a diagnostic.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu
from statannotations.Annotator import Annotator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_style import set_style, panel, save, add_size_advantage
from utils_rnn import svd_evr, SVD_BACKEND
from cfg import PLOT_POINTS_BELOW_N
from rnn_loader import iter_rnn_episodes

OUT_SUBDIR = "analyses/rnn_dim"
N_MAX_ROWS_DEFF = 5000   # rows pooled per condition for D_eff estimates

# Palettes
_RANGE_PALETTE   = ["#2C5AA0", "#6AAFD4", "#A6CEE3"]   # dark→light blue (close→far)
_BOOLEAN_PALETTE = {"False": "#4477AA", "True": "#EE6677"}


# ── computation ───────────────────────────────────────────────────────────────

def compute_pca_per_env_episode(raw_dir: str) -> tuple[np.ndarray, pd.DataFrame]:
    """
    For every (episode_index, env_id) pair, fit PCA on (T*A, H) and return:
      - cumvar_mat: (n_pairs, H) — cumulative variance per PC per pair, padded to H
      - summary_df: DataFrame with columns [episode_index, env_id, D_eff, n_pcs_90]
    """
    cumvar_rows = []
    summary_rows = []

    for k, rnn_arr, _ in iter_rnn_episodes(raw_dir):
        T, E, A, H = rnn_arr.shape
        for e in range(E):
            X = rnn_arr[:, e, :, :].reshape(T * A, H)   # (T*A, H)
            evr = svd_evr(X)                              # all rank-feasible components

            # cumulative variance, padded to H entries
            cumvar = np.cumsum(evr)
            if len(cumvar) < H:
                cumvar = np.concatenate([cumvar, np.full(H - len(cumvar), cumvar[-1])])
            cumvar_rows.append(cumvar)

            # effective rank
            D_eff = 1.0 / np.sum(evr ** 2)

            # n_pcs_90: fewest PCs explaining ≥ 90 % variance
            idx90 = int(np.searchsorted(cumvar, 0.90)) + 1   # 1-based PC index
            idx90 = min(idx90, H)

            summary_rows.append({
                "episode_index": k,
                "env_id": e,
                "D_eff": D_eff,
                "n_pcs_90": idx90,
            })

    if not cumvar_rows:
        return np.empty((0, 0)), pd.DataFrame()

    cumvar_mat = np.stack(cumvar_rows)   # (n_pairs, H)
    summary_df = pd.DataFrame(summary_rows)
    return cumvar_mat, summary_df


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_cumvar(cumvar_mat: np.ndarray, summary_df: pd.DataFrame, outfile_base: str,
                band: str = "sem"):
    """Mean ± band cumulative variance curve with D_eff and n_pcs_90 markers.

    band: "sem" (uncertainty in the mean) or "sd" (episode-level spread).
    """
    import seaborn as sns

    mean   = cumvar_mat.mean(axis=0)
    sd     = cumvar_mat.std(axis=0, ddof=1)
    half   = stats.sem(cumvar_mat, axis=0) if band == "sem" else sd
    label  = "Mean ± SEM" if band == "sem" else "Mean ± SD"
    suffix = "_pca_per_episode_cumvar.pdf" if band == "sem" else "_pca_per_episode_cumvar_sd.pdf"

    H   = len(mean)
    pcs = np.arange(1, H + 1)

    n_pairs = cumvar_mat.shape[0]
    fig, ax = panel()
    ax.fill_between(pcs, mean - half, mean + half, alpha=0.25, color="#4477AA")
    ax.plot(pcs, mean, color="#4477AA", lw=1.2, label=f"{label} (N={n_pairs})")

    # 90 % threshold marker
    mean_n90 = float(summary_df["n_pcs_90"].mean())
    idx90_mean = int(np.searchsorted(mean, 0.90))
    if idx90_mean < H:
        ax.axhline(0.90, color="#888888", lw=0.8, ls="--", zorder=0,
                   label=f"90% var. threshold")
        ax.axvline(idx90_mean + 1, color="#888888", lw=0.8, ls="--", zorder=0,
                   label=f"90% var. PC = {mean_n90:.0f}")

    # D_eff marker
    mean_Deff = float(summary_df["D_eff"].mean())
    if mean_Deff <= H:
        ax.axvline(mean_Deff, color="#CC6677", lw=0.8, ls=":", zorder=0,
                   label=rf"$D_{{\mathrm{{eff}}}}$ = {mean_Deff:.1f}")

    ax.set_xlabel("PC index")
    ax.set_ylabel("Cumulative variance explained")
    ax.set_xlim(1, H)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6)
    sns.despine(ax=ax)
    save(fig, outfile_base + suffix)


def plot_summary(summary_df: pd.DataFrame, outfile_base: str):
    """Side-by-side boxplots of D_eff and n_pcs_90 across all (env, episode) pairs."""
    import seaborn as sns

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2.0, 2.0))
    colors = ["#4477AA", "#EE6677"]

    for ax, col, label, color in [
        (ax1, "D_eff",    r"Effective rank $D_{\mathrm{eff}}$", colors[0]),
        (ax2, "n_pcs_90", "PCs for 90 % var.",                  colors[1]),
    ]:
        vals = summary_df[col].dropna().values
        ax.boxplot(vals, widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.7),
                   medianprops=dict(color="black", lw=1.2),
                   whiskerprops=dict(lw=0.8),
                   flierprops=dict(marker=""))
        if len(vals) < PLOT_POINTS_BELOW_N:
            rng = np.random.default_rng(0)
            ax.scatter(np.ones(len(vals)) + rng.uniform(-0.15, 0.15, len(vals)),
                       vals, color="black", alpha=0.5, s=9, zorder=3)
        ax.set_xticks([1])
        ax.set_xticklabels([f"N={len(vals)}"])
        ax.set_ylabel(label)
        sns.despine(ax=ax)

    plt.tight_layout(pad=0.4)
    save(fig, outfile_base + "_pca_summary.pdf")


# ── two-fish: D_eff by condition (pooled) ────────────────────────────────────

def _deff_from_X(X: np.ndarray) -> float:
    """D_eff via participation ratio of explained variance of (n, H) matrix."""
    X = X - X.mean(axis=0)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    evr = (s ** 2) / (s ** 2).sum()
    return float(1.0 / np.sum(evr ** 2))


def compute_deff_conditions(raw_dir: str, dff: pd.DataFrame,
                             cond_list: list,
                             max_rows: int = N_MAX_ROWS_DEFF,
                             seed: int = 42) -> tuple[dict, dict]:
    """Pool RNN states (both agents) across all (ep, env) pairs and compute D_eff per condition.

    cond_list: [(label, fn)] where fn(agent0_rows_for_env_e) → bool array of length T.
    Pools across episodes/envs; all conditions are subsampled to the same n =
    min(smallest_pool, max_rows) so D_eff estimates are comparable.
    Returns (deff_dict, n_used_dict) where n_used is the common subsample size.
    """
    rng = np.random.default_rng(seed)
    pools: dict[str, list] = {lbl: [] for lbl, _ in cond_list}

    for k, rnn_arr, dff_ep in iter_rnn_episodes(raw_dir, dff):
        T, E, A, H = rnn_arr.shape
        if A < 2:
            continue
        for e in range(E):
            a0 = dff_ep[(dff_ep["env_id"] == e) & (dff_ep["agent_id"] == 0)]
            if len(a0) != T:
                continue
            for lbl, fn in cond_list:
                idx = np.where(np.asarray(fn(a0)))[0]
                if len(idx) == 0:
                    continue
                pools[lbl].append(
                    np.concatenate([rnn_arr[idx, e, 0, :], rnn_arr[idx, e, 1, :]])
                )

    # Assemble per-condition arrays
    X_all: dict[str, np.ndarray] = {}
    for lbl, chunks in pools.items():
        if chunks:
            X_all[lbl] = np.concatenate(chunks, axis=0)

    if not X_all:
        return {}, {}

    # Equal subsampling: all conditions use the same n rows
    n_common = min(len(X) for X in X_all.values())
    n_common = min(n_common, max_rows)

    H_dim = next(iter(X_all.values())).shape[1]
    deff_dict: dict[str, float] = {}
    n_dict: dict[str, int] = {}
    for lbl, X in X_all.items():
        n_dict[lbl] = len(X)   # store original pool size for reporting
        if n_common < H_dim:
            print(f"[rnn_dim] {lbl}: n_common={n_common} < H={H_dim} — skipping", flush=True)
            continue
        X_sub = X[rng.choice(len(X), size=n_common, replace=False)]
        deff_dict[lbl] = _deff_from_X(X_sub)
    return deff_dict, n_dict


def plot_deff_conditions(deff_dict: dict, n_dict: dict, order: list,
                          labels: dict, palette: list, ylabel: str, outfile: str,
                          xlabel: str = ""):
    """Bar chart of pooled D_eff per condition with value labels."""
    present = [o for o in order if o in deff_dict]
    if not present:
        return
    n_common = min(min(n_dict.get(o, 0) for o in present), N_MAX_ROWS_DEFF)
    fig, ax = panel()
    vals = [deff_dict[o] for o in present]
    colors = palette[:len(present)]
    ax.bar(range(len(present)), vals, color=colors, alpha=0.75, width=0.5, edgecolor="white")
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.01, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([labels[o] for o in present])
    ax.set_ylabel("Effective rank")
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.text(1.0, 0.03, f"N={n_common} each", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6, color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5))
    sns.despine(ax=ax)
    plt.tight_layout()
    save(fig, outfile)


def _load_deff_csv(csv_path: str):
    """Load deff_dict and n_dict from a cached CSV, or return None if missing/malformed."""
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return None
    if not {"condition", "D_eff", "n_pooled"}.issubset(df.columns):
        return None
    return (dict(zip(df["condition"], df["D_eff"])),
            dict(zip(df["condition"], df["n_pooled"].astype(int))))


def run_deff_by_range(raw_dir: str, dff: pd.DataFrame, base: str,
                       morm_range_cm: float = 10.0, knollen_range_cm: float = None,
                       force_recompute: bool = False):
    """Compute and plot pooled D_eff by conspecific distance range."""
    if knollen_range_cm is not None:
        cond_list = [
            ("within_morm",     lambda df: df["distance_to_nearest_agent"].values <= morm_range_cm),
            ("morm_to_knollen", lambda df: (df["distance_to_nearest_agent"].values > morm_range_cm) &
                                            (df["distance_to_nearest_agent"].values <= knollen_range_cm)),
            ("beyond_knollen",  lambda df: df["distance_to_nearest_agent"].values > knollen_range_cm),
        ]
        order  = ["within_morm", "morm_to_knollen", "beyond_knollen"]
        labels = {"within_morm":     f"≤{morm_range_cm:.0f}",
                  "morm_to_knollen": f"{morm_range_cm:.0f}–{knollen_range_cm:.0f}",
                  "beyond_knollen":  f">{knollen_range_cm:.0f}"}
    else:
        cond_list = [
            ("in_range",  lambda df: df["distance_to_nearest_agent"].values <= morm_range_cm),
            ("out_range", lambda df: df["distance_to_nearest_agent"].values > morm_range_cm),
        ]
        order  = ["in_range", "out_range"]
        labels = {"in_range":  f"≤{morm_range_cm:.0f}",
                  "out_range": f">{morm_range_cm:.0f}"}

    cache_csv = base + "_deff_by_range.csv"
    cached = None if force_recompute else _load_deff_csv(cache_csv)
    if cached is not None:
        print("[rnn_dim] loading cached D_eff by range", flush=True)
        deff_dict, n_dict = cached
    else:
        if "distance_to_nearest_agent" not in dff.columns:
            print("[rnn_dim] distance_to_nearest_agent not found — skipping range D_eff", flush=True)
            return
        print("[rnn_dim] computing D_eff by range (pooled, equal n) ...", flush=True)
        deff_dict, n_dict = compute_deff_conditions(raw_dir, dff, cond_list)
        n_common = min((n_dict[l] for l in order if l in n_dict), default=0)
        n_common = min(n_common, N_MAX_ROWS_DEFF)
        for lbl in order:
            if lbl in deff_dict:
                print(f"[rnn_dim]   {lbl}: D_eff={deff_dict[lbl]:.1f}  "
                      f"(pool={n_dict[lbl]:,}, used n={n_common:,})", flush=True)
        pd.DataFrame([{"condition": k, "D_eff": v, "n_pooled": n_dict.get(k, 0)}
                      for k, v in deff_dict.items()]).to_csv(cache_csv, index=False)
    plot_deff_conditions(deff_dict, n_dict, order, labels, _RANGE_PALETTE,
                          rf"RNN effective rank $D_{{\mathrm{{eff}}}}$",
                          base + "_deff_by_range.pdf",
                          xlabel="Distance (cm)")


_BOOLEAN_LABELS = {
    "has_nearby":   {"False": "No consp.", "True": "Consp. nearby"},
    "was_bitten":   {"False": "Not bitten", "True": "Bitten"},
    "is_dominant":  {"False": "Subordinate", "True": "Dominant"},
}


def run_deff_by_boolean(raw_dir: str, dff: pd.DataFrame, base: str, feature_col: str,
                         force_recompute: bool = False):
    """Compute and plot pooled D_eff split by a boolean column (e.g. has_nearby, was_bitten)."""
    if feature_col not in dff.columns:
        return
    lbl_map = _BOOLEAN_LABELS.get(feature_col,
                                   {"False": "False", "True": "True"})
    cache_csv = base + f"_deff_by_{feature_col}.csv"
    cached = None if force_recompute else _load_deff_csv(cache_csv)
    if cached is not None:
        print(f"[rnn_dim] loading cached D_eff by {feature_col}", flush=True)
        deff_dict, n_dict = cached
    else:
        cond_list = [
            ("False", lambda df, fc=feature_col: ~df[fc].values.astype(bool)),
            ("True",  lambda df, fc=feature_col:  df[fc].values.astype(bool)),
        ]
        print(f"[rnn_dim] computing D_eff by {feature_col} (pooled, equal n) ...", flush=True)
        deff_dict, n_dict = compute_deff_conditions(raw_dir, dff, cond_list)
        n_common = min((n_dict[l] for l in ["False", "True"] if l in n_dict), default=0)
        n_common = min(n_common, N_MAX_ROWS_DEFF)
        for lbl in ["False", "True"]:
            if lbl in deff_dict:
                print(f"[rnn_dim]   {feature_col}={lbl}: D_eff={deff_dict[lbl]:.1f}  "
                      f"(pool={n_dict[lbl]:,}, used n={n_common:,})", flush=True)
        pd.DataFrame([{"condition": k, "D_eff": v, "n_pooled": n_dict.get(k, 0)}
                      for k, v in deff_dict.items()]).to_csv(cache_csv, index=False)
    plot_deff_conditions(deff_dict, n_dict, ["False", "True"],
                          lbl_map,
                          list(_BOOLEAN_PALETTE.values()),
                          rf"RNN effective rank $D_{{\mathrm{{eff}}}}$",
                          base + f"_deff_by_{feature_col}.pdf")


# ── main ──────────────────────────────────────────────────────────────────────

def load(spec_dir):
    set_style()
    raw_dir = os.path.join(spec_dir, "raw")
    out_dir = os.path.join(spec_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "rnn_dim")
    print(f"[rnn_dim] computing PCA \u2026", flush=True)
    cumvar_mat, summary_df = compute_pca_per_env_episode(raw_dir)
    if cumvar_mat.shape[0] == 0:
        print("[rnn_dim] no rnn episodes found")
        return None
    H = cumvar_mat.shape[1]
    mean = cumvar_mat.mean(axis=0)
    sem_arr = stats.sem(cumvar_mat, axis=0)
    sd = cumvar_mat.std(axis=0, ddof=1)
    pd.DataFrame({
        "pc_idx": np.arange(1, H + 1),
        "mean_cumvar": mean, "sem_cumvar": sem_arr, "sd_cumvar": sd,
    }).to_csv(base + "_pca_per_episode_cumvar.csv", index=False)
    summary_df.to_csv(base + "_pca_summary.csv", index=False)
    pc_cols = {f"pc_{i+1}": cumvar_mat[:, i] for i in range(H)}
    pd.concat([
        summary_df[["episode_index", "env_id"]].reset_index(drop=True),
        pd.DataFrame(pc_cols),
    ], axis=1).to_csv(base + "_pca_per_pair_cumvar.csv", index=False)
    print(f"[rnn_dim] {cumvar_mat.shape[0]} (ep, env) pairs, H={H}", flush=True)
    return {"cumvar_mat": cumvar_mat, "summary_df": summary_df, "base": base, "out_dir": out_dir}


def _load_cached(base):
    per_pair_csv = base + "_pca_per_pair_cumvar.csv"
    summary_csv = base + "_pca_summary.csv"
    if not (os.path.exists(per_pair_csv) and os.path.exists(summary_csv)):
        return None
    per_pair_df = pd.read_csv(per_pair_csv)
    summary_df = pd.read_csv(summary_csv)
    pc_cols = [c for c in per_pair_df.columns if c.startswith("pc_")]
    pc_cols.sort(key=lambda c: int(c.split("_", 1)[1]))
    if not pc_cols or summary_df.empty:
        return None
    return per_pair_df[pc_cols].to_numpy(), summary_df


def run(spec_dir, force_recompute=False,
        two_fish_mode=False, morm_range_cm=10.0, knollen_range_cm=None):
    set_style()
    raw_dir = os.path.join(spec_dir, "raw")
    derived_dir = os.path.join(spec_dir, "derived")
    out_dir = os.path.join(spec_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "rnn_dim")

    print(f"[rnn_dim] raw_dir: {raw_dir}", flush=True)
    print(f"[rnn_dim] out_dir: {out_dir}", flush=True)
    print(f"[rnn_dim] SVD backend: {SVD_BACKEND}", flush=True)

    cached = None if force_recompute else _load_cached(base)
    if cached is not None:
        print("[rnn_dim] loading cached PCA results", flush=True)
        cumvar_mat, summary_df = cached
    else:
        print("[rnn_dim] computing PCA ...", flush=True)
        cumvar_mat, summary_df = compute_pca_per_env_episode(raw_dir)
    if cumvar_mat.shape[0] == 0:
        print("[rnn_dim] no rnn episodes found — skipping", flush=True)
        return

    H    = cumvar_mat.shape[1]
    mean = cumvar_mat.mean(axis=0)
    sem  = stats.sem(cumvar_mat, axis=0)
    sd   = cumvar_mat.std(axis=0, ddof=1)

    if cached is None:
        cumvar_csv = pd.DataFrame({
            "pc_idx":      np.arange(1, H + 1),
            "mean_cumvar": mean,
            "sem_cumvar":  sem,
            "sd_cumvar":   sd,
        })
        cumvar_csv.to_csv(base + "_pca_per_episode_cumvar.csv", index=False)
        summary_df.to_csv(base + "_pca_summary.csv", index=False)

        pc_cols = {f"pc_{i+1}": cumvar_mat[:, i] for i in range(H)}
        per_pair_df = pd.concat([
            summary_df[["episode_index", "env_id"]].reset_index(drop=True),
            pd.DataFrame(pc_cols),
        ], axis=1)
        per_pair_df.to_csv(base + "_pca_per_pair_cumvar.csv", index=False)
    print(f"[rnn_dim] {cumvar_mat.shape[0]} (ep, env) pairs processed, H={H}", flush=True)

    # Figures
    plot_cumvar(cumvar_mat, summary_df, base, band="sem")
    plot_cumvar(cumvar_mat, summary_df, base, band="sd")
    plot_summary(summary_df, base)

    # Two-fish-only: D_eff conditioned on inter-agent distance range and behavioral events
    if two_fish_mode:
        step_pkl = os.path.join(derived_dir, "per_env_ep_agent_step.pkl")
        if not os.path.exists(step_pkl):
            print("[rnn_dim] two_fish_mode: missing step pkl — skipping D_eff by condition",
                  flush=True)
        else:
            dff = pd.read_pickle(step_pkl)
            run_deff_by_range(raw_dir, dff, base,
                              morm_range_cm=morm_range_cm,
                              knollen_range_cm=knollen_range_cm,
                              force_recompute=force_recompute)
            dff = add_size_advantage(dff)
            if "size_advantage" in dff.columns:
                dff["is_dominant"] = dff["size_advantage"] > 0
            for feature_col in ["has_nearby", "was_bitten", "is_dominant"]:
                run_deff_by_boolean(raw_dir, dff, base, feature_col,
                                    force_recompute=force_recompute)

    print("[rnn_dim] done", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec_dir", required=True,
                    help="Path to evals/<spec_key>")
    ap.add_argument("--force_recompute", action="store_true",
                    help="Ignore cached CSVs and recompute from raw RNN arrays")
    ap.add_argument("--two_fish_mode", action="store_true",
                    help="Also run D_eff by distance range and behavioral events")
    ap.add_argument("--morm_range_cm", type=float, default=10.0)
    ap.add_argument("--knollen_range_cm", type=float, default=None)
    args = ap.parse_args(argv)
    run(args.spec_dir, force_recompute=args.force_recompute,
        two_fish_mode=args.two_fish_mode,
        morm_range_cm=args.morm_range_cm,
        knollen_range_cm=args.knollen_range_cm)


if __name__ == "__main__":
    main()
