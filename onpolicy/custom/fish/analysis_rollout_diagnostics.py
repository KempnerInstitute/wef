"""
Sensor diagnostics and observation sanity checks.

Usage:
    python analysis_rollout_diagnostics.py --spec_dir path/to/evals/m1a1k1_patchy_square

Outputs to {spec_dir}/analyses/rollout_diagnostics/
  *_initial_positions_agent_*.pdf      — start-position scatter per agent
  *_sensor_correlation_matrix.pdf      — obs correlation heatmap
  *_boxplot_vertical_sparse*.pdf       — obs distribution boxplots
  *_histogram_self_cons_morm.pdf       — self vs conspecific morm histograms
  *_attention_entropy_*.pdf            — entropy distributions (if attn_mask present)
  *_attention_weights_*.pdf            — weight distributions (if attn_mask present)
  *_sensing_*_mormyromast_*.pdf        — sensing-by-target-location plots
  *_sensing_*_ampullary_*.pdf
  *_sensing_hexbin_*.pdf               — hexbin conditional expectation plots
  *_hist_*.pdf                         — per-column histograms (up to 80)
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import utils_rollout_diagnostics as _rd

OUT_SUBDIR = "analyses/rollout_diagnostics"


def _load_obs_into_df(dff: pd.DataFrame, raw_dir: str) -> pd.DataFrame:
    """
    Merge ep{k}_obs.npy arrays into dff as an 'observations' column.

    obs.npy shape: (T, E, A, D) — time_steps × envs × agents × obs_dim.
    Aligned by: episode_index=k, time_step=t, env_id=e, agent_id=a.
    """
    obs_files = sorted(glob.glob(os.path.join(raw_dir, "ep*_obs.npy")))
    if not obs_files:
        raise FileNotFoundError(f"No ep*_obs.npy files found in {raw_dir}")

    obs_arrays = [np.load(f) for f in obs_files]
    n_ep = len(obs_arrays)
    T, E, A, D = obs_arrays[0].shape

    all_obs = np.stack(obs_arrays, axis=0)   # (n_ep, T, E, A, D)
    all_obs_flat = all_obs.reshape(-1, D)    # (n_ep*T*E*A, D)

    # Flat index = k*T*E*A + t*E*A + e*A + a
    idx = (dff["episode_index"].values * (T * E * A)
           + dff["time_step"].values * (E * A)
           + dff["env_id"].values * A
           + dff["agent_id"].values)

    dff = dff.copy()
    dff["observations"] = list(all_obs_flat[idx])
    return dff


def _pdf_save_figure(fig, output_dir, pkl_str, suffix):
    """PDF override for utils_rollout_diagnostics.save_figure."""
    fname = os.path.join(output_dir, f"{pkl_str}_{suffix}.pdf")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {fname}")
    return fname


def run(spec_dir, plot_only=False):
    raw_dir     = os.path.join(spec_dir, "raw")
    derived_pkl = os.path.join(spec_dir, "derived", "per_env_ep_agent_step.pkl")

    if not os.path.exists(derived_pkl):
        sys.exit(f"Not found: {derived_pkl}")

    out_dir = os.path.join(spec_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {derived_pkl} …")
    dff = pd.read_pickle(derived_pkl)
    print(f"  {len(dff):,} rows")

    print("Merging observations from ep*_obs.npy …")
    dff = _load_obs_into_df(dff, raw_dir)
    print(f"  obs_dim = {dff['observations'].iloc[0].shape[0]}")

    _orig_save = _rd.save_figure
    _rd.save_figure = _pdf_save_figure
    try:
        _rd.run_rollout_diagnostics_report(
            dff,
            out_dir,
            analyses_to_run=None,   # DEFAULT_ANALYSES
            sensor_model="dynamic",
            use_signed_max=True,
            rotate_agent_up=True,
        )
    finally:
        _rd.save_figure = _orig_save

    print("Done.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec_dir", required=True)
    ap.add_argument("--plot_only", action="store_true")
    args = ap.parse_args(argv)
    run(args.spec_dir, plot_only=args.plot_only)


if __name__ == "__main__":
    main()
