"""
Compute food spatial dispersion at episode start, mid, and end.

Metric: Clark-Evans nearest-neighbour index
    R = r_obs / r_exp,  r_exp = 0.5 / sqrt(rho),  rho = n_food / arena_area

  R < 1  → clustered   (patchy arena at episode start)
  R ≈ 1  → random (CSR)
  R > 1  → overdispersed / regular

Checkpoints
-----------
  start  first recorded timestep  (initial food layout)
  mid    timestep nearest to 50% of episode length
  end    last recorded timestep   (remaining food at episode end)

Output
------
  derived/food_dispersion.pkl  — one row per (env_id, episode_index)

Columns
  env_id, episode_index
  food_count_{start,mid,end}         int    number of food items
  food_dispersion_{start,mid,end}    float  Clark-Evans R (NaN if < 2 items)
  mean_nn_food_{start,mid,end}       float  mean nearest-neighbour distance (cm)

Usage
-----
  python preprocess_food_dispersion.py --spec_dir path/to/evals/m1a1k1_patchy_square
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# core metric
# ---------------------------------------------------------------------------

def clark_evans_r(positions: np.ndarray, arena_area: float):
    """
    Clark-Evans nearest-neighbour index for a set of 2-D points.
    Returns (R, mean_nn_dist).  Both are NaN when n < 2.
    """
    positions = np.asarray(positions, dtype=float)
    n = len(positions)
    if n < 2:
        return np.nan, np.nan

    # pairwise distances; set diagonal to inf so min gives NN distance
    diff = positions[:, None, :] - positions[None, :, :]        # (n, n, 2)
    dists = np.linalg.norm(diff, axis=2)                         # (n, n)
    np.fill_diagonal(dists, np.inf)
    nn_dists = dists.min(axis=1)                                 # (n,)

    r_obs = nn_dists.mean()
    rho   = n / arena_area
    r_exp = 0.5 / np.sqrt(rho)
    return r_obs / r_exp, r_obs


# ---------------------------------------------------------------------------
# main logic
# ---------------------------------------------------------------------------

def _get_checkpoint_steps(time_steps: np.ndarray):
    """Return (t_start, t_mid, t_end) for a sorted array of time steps."""
    t_start = time_steps[0]
    t_end   = time_steps[-1]
    mid_val = (t_start + t_end) / 2.0
    # nearest recorded step to the midpoint
    t_mid   = time_steps[np.argmin(np.abs(time_steps - mid_val))]
    return t_start, t_mid, t_end


def run(spec_dir: str, force: bool = False) -> None:
    raw_dir     = os.path.join(spec_dir, "raw")
    derived_dir = os.path.join(spec_dir, "derived")
    out_path    = os.path.join(derived_dir, "food_dispersion.pkl")

    if os.path.exists(out_path) and not force:
        print(f"[SKIP] already exists: {out_path}")
        return

    # ── load arena pkls ──────────────────────────────────────────────────────
    arena_files = sorted(glob.glob(os.path.join(raw_dir, "*_arena.pkl")))
    if not arena_files:
        sys.exit(f"No *_arena.pkl files found in {raw_dir}")

    print(f"Loading {len(arena_files)} arena pkl(s) …")
    arena_df = pd.concat(
        [pd.read_pickle(f) for f in arena_files], ignore_index=True
    )
    print(f"  {len(arena_df):,} arena rows  "
          f"({arena_df['episode_index'].nunique()} episodes × "
          f"{arena_df['env_id'].nunique()} envs)")

    # ── get arena_size from agg_flat.pkl ─────────────────────────────────────
    flat_pkl = os.path.join(raw_dir, "agg_flat.pkl")
    if not os.path.exists(flat_pkl):
        sys.exit(f"Not found: {flat_pkl}")

    flat = pd.read_pickle(flat_pkl)
    arena_size_lookup = (
        flat.groupby(["env_id", "episode_index"])["arena_size"]
        .first()
        .reset_index()
    )
    arena_df = arena_df.merge(arena_size_lookup, on=["env_id", "episode_index"],
                              how="left")

    # ── compute per-episode checkpoints ──────────────────────────────────────
    rows = []
    for (env_id, ep), grp in arena_df.groupby(["env_id", "episode_index"]):
        grp = grp.sort_values("time_step")
        t_steps = grp["time_step"].values

        arena_size = grp["arena_size"].iloc[0]
        area = float(arena_size[0]) * float(arena_size[1])

        t_start, t_mid, t_end = _get_checkpoint_steps(t_steps)

        row = {"env_id": env_id, "episode_index": ep}
        lookup = grp.set_index("time_step")["food_positions"]

        for label, t in [("start", t_start), ("mid", t_mid), ("end", t_end)]:
            fp = np.asarray(lookup.loc[t])
            row[f"food_count_{label}"]      = len(fp)
            R, mean_nn = clark_evans_r(fp, area)
            row[f"food_dispersion_{label}"] = R
            row[f"mean_nn_food_{label}"]    = mean_nn

        rows.append(row)

    result = pd.DataFrame(rows)
    os.makedirs(derived_dir, exist_ok=True)
    result.to_pickle(out_path)
    print(f"Saved → {out_path}  ({len(result):,} episode rows)")
    _print_summary(result)


def _print_summary(df: pd.DataFrame) -> None:
    for label in ("start", "mid", "end"):
        r_col = f"food_dispersion_{label}"
        c_col = f"food_count_{label}"
        r_vals = df[r_col].dropna()
        print(
            f"  {label:5s}  n_food={df[c_col].mean():.1f} ± {df[c_col].std():.1f}  "
            f"R={r_vals.mean():.3f} ± {r_vals.std():.3f}  "
            f"(range {r_vals.min():.3f}–{r_vals.max():.3f})"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--spec_dir", required=True,
                    help="Path to evals/{spec_key}/ directory")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing output")
    args = ap.parse_args(argv)
    run(args.spec_dir, force=args.force)


if __name__ == "__main__":
    main()
