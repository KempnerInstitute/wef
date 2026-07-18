"""
Partial Least Squares Correlation of agent RNN states.

Fig 6G: Number of significant shared dimensions by distance range.
        PLSC1 canonical correlation by distance range.

Requires 2-agent data (agents 0 & 1).

Usage
-----
    python analysis_rnn_plsc.py --spec_dir path/to/evals/2fish_m1a1k1_uniform_square

    # Three-range mode (default): within_morm / morm_to_knollen / beyond_knollen
    python analysis_rnn_plsc.py --spec_dir ... --knollen_range_cm 100

    # Two-range mode: in_range / out_range
    python analysis_rnn_plsc.py --spec_dir ... --no_knollen_range

Outputs to {spec_dir}/analyses/rnn_plsc/
    rnn_plsc_plsc_sig_dims.csv / .pdf
    rnn_plsc_plsc1_by_range.pdf
"""

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSCanonical
from sklearn.decomposition import PCA
from statannotations.Annotator import Annotator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_style import set_style, panel, save, COND_PALETTE
from cfg import PLOT_POINTS_BELOW_N
from rnn_loader import iter_rnn_episodes

# Palette: dark blue (within_morm) / mid blue (morm_to_knollen) / light blue (beyond_knollen)
_PLSC_PALETTE = ["#2C5AA0", "#6AAFD4", "#A6CEE3"]
# Comparison palette: ours (blue) / zhang-phi (orange)
_CMP_PALETTE  = {"ours": "#2C5AA0", "zhang-phi": "#E36B2E"}

OUT_SUBDIR   = "analyses/rnn_plsc"
MIN_SAMPLES  = 40
N_COMPONENTS = 100
N_SHUFFLES   = 30


# ── shared GPU / math helpers ─────────────────────────────────────────────────

def _pearson_batch_torch(A, B):
    """Pearson r along the sample dim; A, B: (..., n, k) → (..., k)."""
    import torch
    A = A - A.mean(dim=-2, keepdim=True)
    B = B - B.mean(dim=-2, keepdim=True)
    num   = (A * B).sum(dim=-2)
    denom = (A.norm(dim=-2) * B.norm(dim=-2)).clamp(min=1e-12)
    return num / denom


def _zscore_cols(X):
    """Z-score columns (ddof=1, matching MATLAB); constant columns → 0."""
    X   = np.asarray(X, dtype=np.float32)
    mu  = X.mean(axis=0)
    std = X.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    return (X - mu) / std


# ── core PLSC ─────────────────────────────────────────────────────────────────

def compute_plsc_old(X, Y, n_components=N_COMPONENTS, n_shuffles=N_SHUFFLES, seed=42):
    """
    Original sklearn NIPALS PLSC.  Kept for reference; superseded by
    compute_plsc_gpu_zhangphi (circular-shift null, dual criterion).
    """
    rng = np.random.RandomState(seed)
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    n_components = min(n_components, X.shape[1], Y.shape[1], X.shape[0] - 1)
    plsc = PLSCanonical(n_components=n_components)
    plsc.fit(X, Y)
    Xs, Ys = plsc.transform(X, Y)
    correlations = [pearsonr(Xs[:, i], Ys[:, i])[0] for i in range(n_components)]

    if n_shuffles <= 0:
        return {"correlations": correlations, "null_corrs": np.zeros((0, n_components)),
                "num_sig": 0, "top_corr": correlations[0] if correlations else 0.0}

    null_corrs = np.zeros((n_shuffles, n_components))
    for s in range(n_shuffles):
        idx = rng.permutation(X.shape[0])
        plsc_null = PLSCanonical(n_components=n_components).fit(X[idx], Y)
        Xn, Yn = plsc_null.transform(X[idx], Y)
        for i in range(n_components):
            null_corrs[s, i] = pearsonr(Xn[:, i], Yn[:, i])[0]

    sig = [i for i, r in enumerate(correlations)
           if r > np.percentile(null_corrs[:, i], 97.5)]
    return {
        "correlations": correlations,
        "null_corrs": null_corrs,
        "num_sig": len(sig),
        "top_corr": correlations[0] if correlations else 0.0,
    }


def compute_plsc_gpu(X, Y, n_components=N_COMPONENTS, n_shuffles=N_SHUFFLES, seed=42,
                     device=None):
    """
    SVD-based PLSC with GPU-batched row-permutation test.  Drop-in for compute_plsc.

    Uses SVD of the cross-covariance matrix (McIntosh & Lobaugh 2004) instead of
    sklearn's NIPALS PLSCanonical.  Avoids non-convergence when H >> n.
    All n_shuffles permutations are batched into a single GPU SVD call.

    Null: random row permutation of X (independent of temporal structure).
    See compute_plsc_gpu_zhangphi for circular-shift null + dual criterion.

    Timings at H=512, ns=30:  n=79 → 0.04s GPU;  n=1521 → 0.6s GPU.
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rng = np.random.RandomState(seed)
    X = (X - X.mean(axis=0)).astype(np.float32)
    Y = (Y - Y.mean(axis=0)).astype(np.float32)
    n_components = min(n_components, X.shape[1], Y.shape[1], X.shape[0] - 1)

    def _svd_corrs(Xb, Yt):
        """Xb: (S, n, H) or (n, H); Yt: (n, H). Returns (S, k) or (k,)."""
        squeeze = Xb.dim() == 2
        if squeeze:
            Xb = Xb.unsqueeze(0)
        C  = Xb.mT @ Yt.unsqueeze(0)
        U, _, Vh = torch.linalg.svd(C, full_matrices=False)
        Xs = Xb @ U[..., :n_components]
        Ys = Yt.unsqueeze(0) @ Vh.mT[..., :n_components]
        r  = _pearson_batch_torch(Xs, Ys)
        return r.squeeze(0) if squeeze else r

    Xt = torch.from_numpy(X).to(device)
    Yt = torch.from_numpy(Y).to(device)

    correlations = _svd_corrs(Xt, Yt).cpu().tolist()

    if n_shuffles <= 0:
        return {"correlations": correlations, "null_corrs": np.zeros((0, n_components)),
                "num_sig": 0, "top_corr": correlations[0] if correlations else 0.0}

    perms      = np.stack([rng.permutation(X.shape[0]) for _ in range(n_shuffles)])
    Xshuf      = torch.from_numpy(X[perms]).to(device)
    null_corrs = _svd_corrs(Xshuf, Yt).cpu().numpy()

    sig = [i for i, r in enumerate(correlations)
           if r > np.percentile(null_corrs[:, i], 97.5)]
    return {
        "correlations": correlations,
        "null_corrs":   null_corrs,
        "num_sig":      len(sig),
        "top_corr":     correlations[0] if correlations else 0.0,
    }


def compute_plsc_gpu_zhangphi(X, Y, n_components=N_COMPONENTS, n_shuffles=N_SHUFFLES,
                               seed=42, lag=60, sig_thr=0.975, device=None):
    """
    GPU-accelerated PLSC following hongw-lab/code_for_2024_zhang-phi.
    This is the default PLSC method used by the pipeline.

    Differences from compute_plsc_gpu:
      null      — circular temporal shift (preserves autocorrelation) not row permutation
      inputs    — z-scored column-wise (not just centered)
      criterion — dual: singular-value covariance AND projected correlation must both pass
      counting  — contiguous: stops at first non-significant dimension
      C         — normalised by (n-1) matching MATLAB convention

    Ref: hongw-lab/code_for_2024_zhang-phi, ported in plsc/plsc_analysis.py
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    X = _zscore_cols(X)
    Y = _zscore_cols(Y)
    n = X.shape[0]
    n_components = min(n_components, X.shape[1], Y.shape[1], n - 1)

    rng     = np.random.RandomState(seed)
    half    = max(1, lag // 2)
    lo, hi  = half, max(half + 1, n - half)
    shifts  = rng.randint(lo, hi, size=n_shuffles)
    row_idx = (np.arange(n)[None, :] - shifts[:, None]) % n

    Xt   = torch.from_numpy(X).to(device)
    Yt   = torch.from_numpy(Y).to(device)
    Ys_b = Yt[torch.from_numpy(row_idx).long().to(device)]

    def _svd_cov_corr(Xa, Ya, k):
        squeeze = Xa.dim() == 2
        if squeeze:
            Xa, Ya = Xa.unsqueeze(0), Ya.unsqueeze(0)
        C  = Xa.mT @ Ya / (n - 1)
        U, sv, Vh = torch.linalg.svd(C, full_matrices=False)
        Xs = Xa @ U[..., :k]
        Ys = Ya @ Vh.mT[..., :k]
        r  = _pearson_batch_torch(Xs, Ys)
        if squeeze:
            sv, r = sv.squeeze(0), r.squeeze(0)
        return sv[..., :k], r

    real_sv, real_corr = _svd_cov_corr(Xt, Yt, n_components)
    real_sv, real_corr = real_sv.cpu().numpy(), real_corr.cpu().numpy()

    Xt_b               = Xt.unsqueeze(0).expand(n_shuffles, -1, -1)
    null_sv, null_corr = _svd_cov_corr(Xt_b, Ys_b, n_components)
    null_sv, null_corr = null_sv.cpu().numpy(), null_corr.cpu().numpy()

    sv_thr   = np.percentile(null_sv,   sig_thr * 100, axis=0)
    corr_thr = np.percentile(null_corr, sig_thr * 100, axis=0)
    passes   = (real_sv > sv_thr) & (real_corr > corr_thr)

    first_fail = np.flatnonzero(~passes)
    num_sig    = int(first_fail[0]) if first_fail.size else n_components

    return {
        "correlations": real_corr.tolist(),
        "null_corrs":   null_corr,
        "num_sig":      num_sig,
        "top_corr":     float(real_corr[0]) if len(real_corr) else 0.0,
    }


def compute_plsc_sig_dims(raw_dir: str, dff: pd.DataFrame,
                          morm_range_cm: float = 10.0,
                          knollen_range_cm: float = 100.0,
                          max_episodes: int = None,
                          pca_dim: int = 0,
                          rowperm_null: bool = False) -> pd.DataFrame:
    """
    Compute number of significant PLSC components by distance range.

    knollen_range_cm: if not None, splits into three ranges —
        within_morm (d <= morm_range_cm), morm_to_knollen (morm < d <= knollen),
        beyond_knollen (d > knollen_range_cm).
        If None, uses two ranges: in_range / out_range (legacy behaviour).
    max_episodes:  cap on number of episode files to process (None = all).
    pca_dim:       if > 0, pre-project RNN states via PCA before PLSC.
    rowperm_null:  if True, use compute_plsc_gpu (row-permutation null) instead
                   of default compute_plsc_gpu_zhangphi (circular-shift null).
    """
    if "distance_to_nearest_agent" not in dff.columns:
        return pd.DataFrame()

    plsc_fn = compute_plsc_gpu if rowperm_null else compute_plsc_gpu_zhangphi

    records = []
    n_eps = 0
    for k, rnn_arr, dff_ep in iter_rnn_episodes(raw_dir, dff):
        if max_episodes is not None and n_eps >= max_episodes:
            break
        T, E, A, H = rnn_arr.shape
        if A < 2:
            continue

        for e in range(E):
            mask_a0 = (dff_ep["env_id"] == e) & (dff_ep["agent_id"] == 0)
            dist_vals = dff_ep.loc[mask_a0, "distance_to_nearest_agent"].values
            if len(dist_vals) != T:
                continue

            rnn_a0 = rnn_arr[:, e, 0, :]   # (T, H)
            rnn_a1 = rnn_arr[:, e, 1, :]

            if pca_dim > 0 and pca_dim < H:
                rnn_a0 = PCA(n_components=pca_dim).fit_transform(rnn_a0)
                rnn_a1 = PCA(n_components=pca_dim).fit_transform(rnn_a1)

            if knollen_range_cm is not None:
                cond_list = [
                    ("within_morm",     dist_vals <= morm_range_cm),
                    ("morm_to_knollen", (dist_vals > morm_range_cm) & (dist_vals <= knollen_range_cm)),
                    ("beyond_knollen",  dist_vals > knollen_range_cm),
                ]
            else:
                cond_list = [
                    ("in_range",  dist_vals <= morm_range_cm),
                    ("out_range", dist_vals >  morm_range_cm),
                ]

            for label, idx_mask in cond_list:
                idx = np.where(idx_mask)[0]
                if len(idx) < MIN_SAMPLES * 2:
                    continue
                result = plsc_fn(rnn_a0[idx], rnn_a1[idx],
                                 n_components=N_COMPONENTS,
                                 n_shuffles=N_SHUFFLES)
                records.append({
                    "episode_index": k,
                    "env_id": e,
                    "condition": label,
                    "num_sig": result["num_sig"],
                    "top_corr": result["top_corr"],
                })
        n_eps += 1
    return pd.DataFrame(records)


# ── plotting ──────────────────────────────────────────────────────────────────

def _plsc_violin(plot_df, x_col, y_col, label_order, ylabel, outfile):
    """Shared violin helper with statannotations and per-violin N annotation."""
    palette = {lbl: _PLSC_PALETTE[i] for i, lbl in enumerate(label_order)}
    pairs = list(itertools.combinations(label_order, 2))

    fig, ax = panel()
    sns.violinplot(data=plot_df, x=x_col, y=y_col, hue=x_col, order=label_order,
                   palette=palette, inner="quart", cut=0, linewidth=0.8,
                   legend=False, ax=ax)

    ann = Annotator(ax, pairs, data=plot_df, x=x_col, y=y_col, order=label_order)
    ann.configure(test="Mann-Whitney", text_format="star", loc="inside", verbose=0)
    ann.apply_and_annotate()

    # Put per-violin sample sizes in the tick labels so they occupy reserved
    # layout space instead of overlapping data near the lower y-limit.
    n_by_label = plot_df.groupby(x_col)[y_col].count()
    tick_labels = [
        f"{lbl}\nN={int(n_by_label.get(lbl, 0))}"
        for lbl in label_order
    ]
    ax.set_xticks(range(len(label_order)))
    ax.set_xticklabels(tick_labels)

    ax.set_xlabel("Conspecific distance (cm)")
    ax.set_ylabel(ylabel)
    sns.despine(ax=ax)
    plt.tight_layout()
    save(fig, outfile)


def _range_order_labels(knollen_range_cm, morm_range_cm=10.0):
    """Return (raw_order, label_map) for 3-range or 2-range mode."""
    if knollen_range_cm is not None:
        order = ["within_morm", "morm_to_knollen", "beyond_knollen"]
        labels = {
            "within_morm":     f"≤{morm_range_cm:.0f}",
            "morm_to_knollen": f"{morm_range_cm:.0f}–{knollen_range_cm:.0f}",
            "beyond_knollen":  f">{knollen_range_cm:.0f}",
        }
    else:
        order = ["in_range", "out_range"]
        labels = {
            "in_range":  f"≤{morm_range_cm:.0f}",
            "out_range": f">{morm_range_cm:.0f}",
        }
    return order, labels


def plot_sig_dims(sig_df: pd.DataFrame, outfile_base: str, knollen_range_cm=100.0):
    """Violin of significant shared PLSC dimensions by distance range."""
    if sig_df.empty:
        return
    order, labels = _range_order_labels(knollen_range_cm)
    plot_df = sig_df.copy()
    plot_df["label"] = plot_df["condition"].map(labels)
    plot_df = plot_df.dropna(subset=["label", "num_sig"])
    label_order = [labels[o] for o in order if o in sig_df["condition"].values]

    _plsc_violin(plot_df, "label", "num_sig", label_order,
                 "# sig. PLSC dims", outfile_base + "_plsc_sig_dims.pdf")
    sig_df.to_csv(outfile_base + "_plsc_sig_dims.csv", index=False)


def plot_plsc1_by_range(sig_df: pd.DataFrame, outfile_base: str, knollen_range_cm=100.0):
    """Violin of PLSC1 correlation by distance range."""
    if sig_df.empty or "top_corr" not in sig_df.columns:
        return
    order, labels = _range_order_labels(knollen_range_cm)
    plot_df = sig_df.copy()
    plot_df["label"] = plot_df["condition"].map(labels)
    plot_df = plot_df.dropna(subset=["label", "top_corr"])
    label_order = [labels[o] for o in order if o in sig_df["condition"].values]

    _plsc_violin(plot_df, "label", "top_corr", label_order,
                 "PLSC1 correlation", outfile_base + "_plsc1_by_range.pdf")


# ── main ──────────────────────────────────────────────────────────────────────

def load(spec_dir, morm_range_cm=10.0, knollen_range_cm=100.0,
         max_episodes=None, pca_dim=0, rowperm_null=False):
    set_style()
    raw_dir     = os.path.join(spec_dir, "raw")
    derived_dir = os.path.join(spec_dir, "derived")
    out_dir     = os.path.join(spec_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "rnn_plsc")
    step_pkl = os.path.join(derived_dir, "per_env_ep_agent_step.pkl")
    if not os.path.exists(step_pkl):
        print(f"[rnn_plsc] missing {step_pkl}")
        return None
    dff = pd.read_pickle(step_pkl)
    print(f"[rnn_plsc] computing sig dims (morm={morm_range_cm}, knollen={knollen_range_cm}, "
          f"max_episodes={max_episodes}, pca_dim={pca_dim}, rowperm_null={rowperm_null}) ...", flush=True)
    sig_df = compute_plsc_sig_dims(raw_dir, dff, morm_range_cm=morm_range_cm,
                                   knollen_range_cm=knollen_range_cm,
                                   max_episodes=max_episodes, pca_dim=pca_dim,
                                   rowperm_null=rowperm_null)
    return {"sig_df": sig_df, "base": base, "out_dir": out_dir}


def run(spec_dir, morm_range_cm=10.0, knollen_range_cm=100.0,
        max_episodes=None, pca_dim=0, rowperm_null=False, force_recompute=False):
    set_style()
    raw_dir     = os.path.join(spec_dir, "raw")
    derived_dir = os.path.join(spec_dir, "derived")
    out_dir     = os.path.join(spec_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "rnn_plsc")

    step_pkl = os.path.join(derived_dir, "per_env_ep_agent_step.pkl")
    if not os.path.exists(step_pkl):
        print(f"[rnn_plsc] missing {step_pkl} -- skipping", flush=True)
        return
    print(f"[rnn_plsc] loading {step_pkl}", flush=True)
    dff = pd.read_pickle(step_pkl)

    cache_csv = base + "_plsc_sig_dims.csv"
    if not force_recompute and os.path.exists(cache_csv):
        print("[rnn_plsc] loading cached PLSC CSV", flush=True)
        sig_df = pd.read_csv(cache_csv)
    else:
        n_ranges = 3 if knollen_range_cm is not None else 2
        print(f"[rnn_plsc] computing sig dims ({n_ranges} ranges, {N_SHUFFLES} shuffles, "
              f"morm={morm_range_cm}, knollen={knollen_range_cm}, "
              f"max_episodes={max_episodes}, pca_dim={pca_dim}, rowperm_null={rowperm_null}) ...", flush=True)
        sig_df = compute_plsc_sig_dims(raw_dir, dff, morm_range_cm=morm_range_cm,
                                       knollen_range_cm=knollen_range_cm,
                                       max_episodes=max_episodes, pca_dim=pca_dim,
                                       rowperm_null=rowperm_null)
    if not sig_df.empty:
        plot_sig_dims(sig_df, base, knollen_range_cm=knollen_range_cm)
        plot_plsc1_by_range(sig_df, base, knollen_range_cm=knollen_range_cm)
        for cond, grp in sig_df.groupby("condition"):
            print(f"[rnn_plsc]   {cond}: num_sig={grp['num_sig'].mean():.1f}", flush=True)

    print("[rnn_plsc] done", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec_dir", required=True)
    ap.add_argument("--morm_range_cm", type=float, default=10.0,
                    help="Mormyromast detection range in cm (default: 10)")
    ap.add_argument("--knollen_range_cm", type=float, default=100.0,
                    help="Knollen detection range in cm — upper edge of middle bin "
                         "(default: 100). Use --no_knollen_range for 2-range mode.")
    ap.add_argument("--no_knollen_range", action="store_true",
                    help="Disable third range; revert to 2-range in_range/out_range split.")
    ap.add_argument("--max_episodes", type=int, default=None,
                    help="Cap number of episode files processed (default: all)")
    ap.add_argument("--pca_dim", type=int, default=0,
                    help="PCA pre-projection dim before PLSC; 0 = disabled (default)")
    ap.add_argument("--rowperm_null", action="store_true",
                    help="Use row-permutation null (compute_plsc_gpu) instead of "
                         "default circular-shift null (compute_plsc_gpu_zhangphi); both run on GPU")
    ap.add_argument("--force_recompute", action="store_true",
                    help="Ignore cached PLSC CSV and recompute from raw RNN arrays")
    args = ap.parse_args(argv)
    knollen = None if args.no_knollen_range else args.knollen_range_cm
    run(args.spec_dir, morm_range_cm=args.morm_range_cm, knollen_range_cm=knollen,
        max_episodes=args.max_episodes, pca_dim=args.pca_dim,
        rowperm_null=args.rowperm_null, force_recompute=args.force_recompute)


if __name__ == "__main__":
    main()
