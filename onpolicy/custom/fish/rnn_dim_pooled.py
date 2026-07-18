"""
Pooled PCA of RNN hidden states using IncrementalPCA.

Streams ep{k}_rnn.npy files one episode at a time via partial_fit — never
holds more than one episode in memory.  Standalone script, NOT wired into
the pipeline (run manually when needed).

Usage
-----
    python rnn_dim_pooled.py \\
        --raw_dir results/.../evals/nfish4_m1a1k1_patchy_square/raw \\
        --outfile_base results/.../evals/nfish4_m1a1k1_patchy_square/analyses/rnn_dim/rnn_dim_pooled \\
        [--n_components 256]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_style import set_style, panel, save
from rnn_loader import iter_rnn_episodes


def run_rnn_dim_pooled(raw_dir: str, outfile_base: str, n_components: int = 256):
    set_style()
    os.makedirs(os.path.dirname(os.path.abspath(outfile_base)), exist_ok=True)

    ipca = IncrementalPCA(n_components=n_components)
    n_chunks = 0

    for k, rnn_arr, _ in iter_rnn_episodes(raw_dir):
        T, E, A, H = rnn_arr.shape
        chunk = rnn_arr.reshape(T * E * A, H)
        ipca.partial_fit(chunk)
        n_chunks += 1
        print(f"  partial_fit ep{k}: {chunk.shape}", flush=True)

    if n_chunks == 0:
        print("[rnn_dim_pooled] no episodes found", flush=True)
        return

    evr = ipca.explained_variance_ratio_
    cumvar = np.cumsum(evr)
    D_eff = 1.0 / np.sum(evr ** 2)
    pcs = np.arange(1, len(cumvar) + 1)

    # Save CSV
    df = pd.DataFrame({"pc_idx": pcs, "cumvar": cumvar})
    df.to_csv(outfile_base + "_pca_pooled_cumvar.csv", index=False)
    pd.DataFrame({"D_eff": [D_eff]}).to_csv(
        outfile_base + "_pca_pooled_effrank.csv", index=False)

    # Plot
    fig, ax = panel(2.5, 2.5)
    ax.plot(pcs, cumvar, color="#4477AA", lw=1.2)
    idx90 = np.searchsorted(cumvar, 0.90)
    if idx90 < len(cumvar):
        ax.axhline(0.90, color="gray", lw=0.8, ls="--", zorder=0)
        ax.axvline(idx90 + 1, color="gray", lw=0.8, ls="--", zorder=0)
        ax.text(idx90 + 2, 0.02, f"PC{idx90+1}", fontsize=6, color="gray", ha="left")
    ax.axvline(D_eff, color="#EE6677", lw=0.8, ls=":", label=f"$D_{{\\rm eff}}$={D_eff:.1f}")
    ax.set_xlabel("PC index")
    ax.set_ylabel("Cumulative variance explained")
    ax.set_xlim(1, min(len(pcs), 100))
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6)
    sns.despine(ax=ax)
    save(fig, outfile_base + "_pca_pooled_cumvar.png")
    print(f"[rnn_dim_pooled] D_eff={D_eff:.1f}, 90% at PC{idx90+1}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--outfile_base", required=True)
    ap.add_argument("--n_components", type=int, default=256)
    args = ap.parse_args(argv)
    run_rnn_dim_pooled(args.raw_dir, args.outfile_base, args.n_components)


if __name__ == "__main__":
    main()
