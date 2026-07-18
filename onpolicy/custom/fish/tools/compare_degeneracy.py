#!/usr/bin/env python3
"""
compare_uniform_patchy_summary.py

Finds *_uniform_run_summary.csv and *_patchy_run_summary.csv files under a parent directory,
computes the mean of numeric metrics across runs in each file (dropping ID-like columns),
and plots side-by-side comparisons (Patchy vs Uniform) for each metric.

Behavior:
- Recursively searches parent_dir for files matching:
    '*uniform_run_summary.csv' and '*patchy_run_summary.csv'
- For each found CSV, computes the mean across rows for numeric columns, excluding ID-like columns.
- Creates a wrapped figure (ncols configurable) where each metric gets a subplot.
  For each metric: the x positions are 'Patchy' and 'Uniform' (0 and 1).
  Each experiment (identified by the top-level folder relative to parent_dir) is assigned a
  consistent color and horizontal offset (jitter), and plotted as a point for any condition it has.
- If a condition (patchy/uniform) is missing for an experiment, that point is skipped.
- Saves figure 'uniform_patchy_summary_comparison.png' in parent_dir.

Usage:
    python compare_uniform_patchy_summary.py /path/to/parent_dir

"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# --------------------------
# Configuration / defaults
# --------------------------
EXCLUDE_COLS = {
    "arena_type", "num_agents", "num_time_steps",
    "run_id", "env_id", "episode_index"  # common id-like columns
}
FILENAME_PATTERNS = {
    "uniform": "*uniform_run_summary.csv",
    "patchy": "*patchy_run_summary.csv"
}
DEFAULT_NCOLS = 3
JITTER_SCALE = 0.06
SEED = 2026

# --------------------------
# Helper functions
# --------------------------
def find_summary_files(parent_dir: Path):
    """Return dict mapping condition -> list of Path objects."""
    out = {"uniform": [], "patchy": []}
    for cond, patt in FILENAME_PATTERNS.items():
        out[cond] = list(parent_dir.rglob(patt))
    return out

def safe_read_csv(path: Path):
    """Try to read CSV, return DataFrame or None on failure."""
    try:
        df = pd.read_csv(path, sep='\t')
        return df
    except Exception as e:
        print(f"Warning: failed to read {path}: {e}")
        return None

def read_last_r_components(train_metrics_csv: Path):
    """
    Reads <run_dir>/logs/train_metrics.csv and returns a dict of r_components from the final row.
    Returns None if file missing/unreadable/field missing/unparseable.
    """
    try:
        df = pd.read_csv(train_metrics_csv)  # train_metrics.csv is comma-separated in your example
        if df.empty or "r_components_json" not in df.columns:
            return None
        s = df["r_components_json"].iloc[-1]
        if pd.isna(s):
            return None
        # s looks like: "{""r_bite"": -1e-6, ...}" so json.loads works after pandas reads it as a normal string
        return json.loads(s)
    except Exception as e:
        print(f"Warning: couldn't parse r_components from {train_metrics_csv}: {e}")
        return None


def is_relative_to(path: Path, base: Path) -> bool:
    """
    Python <3.9 compatibility for Path.is_relative_to().
    Returns True iff `path` is within `base` (or equal to it).
    """
    try:
        path.relative_to(base)
        return True
    except Exception:
        return False


def compute_file_means(df: pd.DataFrame, exclude_cols=EXCLUDE_COLS):
    """
    Convert columns to numeric where possible, drop explicitly excluded cols,
    then return a Series of means for numeric columns.
    """
    # Try to coerce all columns to numeric where possible (non-convertible -> NaN)
    coerced = df.apply(pd.to_numeric, errors="coerce")
    # Drop excluded columns if present
    drop_cols = [c for c in coerced.columns if c in exclude_cols]
    coerced = coerced.drop(columns=drop_cols, errors="ignore")
    # Select numeric columns only
    numeric = coerced.select_dtypes(include=[np.number])
    # Drop columns that are constant NaN or zero-length
    if numeric.shape[1] == 0:
        return pd.Series(dtype=float)
    means = numeric.mean(axis=0, skipna=True)
    raw_numeric = coerced
    return means, coerced

def experiment_key_from_path(parent_dir: Path, file_path: Path):
    """
    Derive an experiment key (string) from the file path that is consistent for files
    within the same experiment. Strategy:
      - Use the first path part under parent_dir (relative path's first component).
      - If that fails, use the parent directory name of the file.
    """
    try:
        rel = file_path.relative_to(parent_dir)
        parts = rel.parts
        if len(parts) >= 2:
            # first component likely the experiment folder
            return parts[0]
        else:
            return file_path.parent.name
    except Exception:
        return file_path.parent.name

# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Compare uniform vs patchy run summary CSVs across experiments.")
    parser.add_argument(
        "parent_dirs",
        nargs="+",
        type=str,
        help="One or more parent directories containing experiment subdirectories."
    )
    parser.add_argument("--ncols", type=int, default=DEFAULT_NCOLS, help="Number of columns in the wrapped figure.")
    parser.add_argument("--out", type=str, default="degeneracy_uniform_patchy_summary_comparison_returns.png", help="Output filename.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory to save output. Defaults to the first parent_dir."
    )
    parser.add_argument("--reward_tail_pct", type=float, default=10.0,
        help="Percent of the end of train_metrics.csv to average for reward smoothing (default 10).")

    args = parser.parse_args()

    parents = [Path(p).expanduser().resolve() for p in args.parent_dirs]
    missing = [p for p in parents if not p.exists()]
    if missing:
        print("[WARN] Parent directory(s) do not exist:\n  "  "\n  ".join(str(p) for p in missing))

    # Default output directory to the first parent dir
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else parents[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect summary files across all parents
    files_by_cond = {"uniform": [], "patchy": []}
    for parent in parents:
        found = find_summary_files(parent)
        for cond in files_by_cond:
            files_by_cond[cond].extend(found.get(cond, []))

    # Read files and compute per-file means
    # We'll store results in a dict: results[(experiment_key, condition)] = series_of_means
    results = {}
    experiments = set()
    all_metrics = set()

    for cond, file_list in files_by_cond.items():
        for f in file_list:
            df = safe_read_csv(f)
            if df is None:
                continue
            means, raw = compute_file_means(df)
            if means.empty:
                print(f"Skipping {f} because no numeric metrics after exclusions.")
                continue
            # Determine which parent this file belongs to, then build a unique key across parents
            parent_for_file = next((p for p in parents if is_relative_to(f, p)), None)
            if parent_for_file is None:
                # Shouldn't happen, but keep things robust
                key = f.parent.name
            else:
                exp_key = experiment_key_from_path(parent_for_file, f)
                # Prefix with parent directory name to avoid collisions across different roots
                key = f"{parent_for_file.name}/{exp_key}"
            experiments.add(key)
            results[(key, cond)] = {
                "mean": means,
                "raw": raw
            }
            all_metrics.update(means.index.tolist())

    if not results:
        raise SystemExit("No valid uniform/patchy run summary CSVs found. Exiting.")

    # Sort metrics for stable ordering
    all_metrics = sorted(all_metrics)

    # Build a pivot DataFrame: index=experiment, columns=(metric,condition) OR easier: build two DataFrames
    # cond_df[cond] rows=experiments, cols=metrics (NaN when missing)
    cond_df = { "uniform": pd.DataFrame(index=sorted(experiments), columns=all_metrics, dtype=float),
                "patchy" : pd.DataFrame(index=sorted(experiments), columns=all_metrics, dtype=float) }

    # NEW: compute IQR (or std) per experiment / condition / metric
    iqr = {
        "uniform": {},
        "patchy": {}
    }

    for (exp, cond), payload in results.items():
        raw = payload["raw"]
        for metric in raw.columns:
            vals = raw[metric].dropna()
            if len(vals) == 0:
                continue
            q25 = np.percentile(vals, 25)
            q75 = np.percentile(vals, 75)
            iqr.setdefault(cond, {}).setdefault(exp, {})[metric] = (q25, q75)


    for (exp, cond), payload in results.items():
        # Place series values into corresponding row
        if isinstance(payload, dict):
            means = payload.get("mean", pd.Series(dtype=float))
        else:
            means = payload
        for metric, val in means.items():
            cond_df[cond].at[exp, metric] = val

    # Optionally drop metrics that do not vary at all (across both conditions & all experiments)
    combined_for_variation = pd.concat([cond_df["uniform"], cond_df["patchy"]], keys=["uniform","patchy"])
    # Keep metrics that have at least two unique values across the combined DataFrame (ignoring NaN)
    varying_metrics = []
    for m in all_metrics:
        vals = combined_for_variation.xs(m, axis=1, level=None, drop_level=False, errors='ignore') if False else combined_for_variation[m]
        # Flatten and drop NaN
        unique_vals = pd.Series(vals.values.ravel()).dropna().unique()
        if unique_vals.size > 1:
            varying_metrics.append(m)
        else:
            # If user specifically wants to keep some metrics, you could add exceptions here.
            pass

    if not varying_metrics:
        print("No varying numeric metrics found after exclusions — plotting whatever numeric metrics remain.")
        varying_metrics = all_metrics

    # We'll plot varying_metrics
    metrics = varying_metrics

    # Prepare consistent colors and jitters for experiments
    exp_list = sorted(experiments)
    num_exps = len(exp_list)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(num_exps)]
    color_map = {exp_list[i]: colors[i] for i in range(num_exps)}

    rng = np.random.RandomState(SEED)
    jitter_vals = rng.normal(loc=0.0, scale=JITTER_SCALE, size=num_exps)
    jitter_map = {exp_list[i]: jitter_vals[i] for i in range(num_exps)}

    # -----------------------------
    # Collect train r_components (one value per run_dir)
    # -----------------------------
    train_csvs = []
    for parent in parents:
        train_csvs.extend(list(parent.rglob("logs/train_metrics.csv")))

    train_components_by_exp = {}  # exp -> dict(component -> value)

    def compute_tail_averages(df: pd.DataFrame, pct: float):
        """
        Return dict mapping column_name -> mean over the last pct% of the non-NaN rows.
        Only considers numeric columns whose name contains 'reward' (case-insensitive).
        Ensures at least one row is used.
        """
        out = {}
        if df is None or df.empty:
            return out
        pct = float(pct)
        # candidates: column names containing 'reward'
        cand_cols = [c for c in df.columns if 'reward' in c.lower()]
        # keep only numeric candidates
        cand_cols = [c for c in cand_cols if pd.api.types.is_numeric_dtype(df[c])]
        # fallback: common reward column names if none found
        fallback_names = ['eval_reward', 'train_reward', 'episode_reward', 'reward_mean', 'mean_reward', 'reward']
        if not cand_cols:
            for name in fallback_names:
                if name in df.columns and pd.api.types.is_numeric_dtype(df[name]):
                    cand_cols.append(name)
        if not cand_cols:
            return out

        nrows = len(df)
        tail_n = max(1, int(np.ceil(nrows * (pct / 100.0))))
        for c in cand_cols:
            non_na = df[c].dropna()
            if non_na.empty:
                continue
            # take the last tail_n non-na values (if less than tail_n available, take what exists)
            tail_vals = non_na.tail(tail_n)
            out[c] = float(tail_vals.mean())
        return out

    def compute_tail_avg_r_components(df: pd.DataFrame, pct: float, col="r_components_json"):
        """
        Parse df[col] as JSON dict per row and return mean per r_component over last pct% rows.
        """
        if df is None or df.empty or col not in df.columns:
            return {}

        pct = float(pct)
        nrows = len(df)
        tail_n = max(1, int(np.ceil(nrows * (pct / 100.0))))
        tail = df[col].dropna().tail(tail_n)
        if tail.empty:
            return {}

        rows = []
        for s in tail:
            try:
                rows.append(json.loads(s))
            except Exception:
                continue
        if not rows:
            return {}

        # union of keys
        keys = sorted({k for d in rows for k in d.keys()})
        out = {}
        for k in keys:
            vals = [d.get(k, np.nan) for d in rows]
            vals = pd.to_numeric(pd.Series(vals), errors="coerce").dropna()
            if not vals.empty:
                out[k] = float(vals.mean())
        return out

    # Read and combine r_components and reward averages
    for tcsv in train_csvs:
        parent_for_file = next((p for p in parents if is_relative_to(tcsv, p)), None)
        if parent_for_file is None:
            continue
        exp_key = experiment_key_from_path(parent_for_file, tcsv)
        exp = f"{parent_for_file.name}/{exp_key}"

        # also try to compute reward averages from the CSV
        try:
            df_metrics = pd.read_csv(tcsv)  # train_metrics.csv in most cases is comma-separated
        except Exception:
            df_metrics = None

        reward_avgs = compute_tail_averages(df_metrics, args.reward_tail_pct)
        comps = compute_tail_avg_r_components(df_metrics, args.reward_tail_pct)
        reward_avgs.update(comps)
        original_keys = list(comps.keys())
        for k, v in reward_avgs.items():
            comps[f"avg_last{int(args.reward_tail_pct)}%: {k}"] = v
        # remove original keys if we added them with new names to avoid confusion
        for k in original_keys:
            if f"avg_last{int(args.reward_tail_pct)}%: {k}" in comps:
                del comps[k]

        if comps:
            train_components_by_exp[exp] = comps

    all_r_keys = sorted({k for d in train_components_by_exp.values() for k in d.keys()})
    # If none found, we'll just make eval section as before
    has_r = len(all_r_keys) > 0

    # Build r_df aligned to exp_list
    if has_r:
        r_df = pd.DataFrame(index=exp_list, columns=all_r_keys, dtype=float)
        for exp in exp_list:
            d = train_components_by_exp.get(exp, {})
            for k in all_r_keys:
                if k in d:
                    r_df.at[exp, k] = d[k]

    # -----------------------------
    # Layout: one figure with two grids (eval + train)
    # -----------------------------
    ncols = max(1, args.ncols)

    # Eval grid geometry
    nmetrics = len(metrics)
    nrows_eval = int(np.ceil(nmetrics / ncols))

    # Train grid geometry
    n_r = len(all_r_keys)
    nrows_r = int(np.ceil(n_r / ncols)) if has_r else 0

    # Add a spacer row between sections if train exists
    spacer = 1 if has_r else 0
    nrows_total = nrows_eval + spacer + nrows_r

    fig_w = ncols * 4.0
    # slightly taller because we add another block
    fig_h = (nrows_eval * 2.6) + (nrows_r * 2.6) + (spacer * 0.8) + 1.0
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Height ratios: make spacer short
    height_ratios = ([1.0] * nrows_eval) + ([0.25] * spacer) + ([1.0] * nrows_r)
    gs = fig.add_gridspec(nrows=nrows_total, ncols=ncols, height_ratios=height_ratios)

    # Build axes arrays
    axes_eval = np.empty((nrows_eval, ncols), dtype=object)
    for r in range(nrows_eval):
        for c in range(ncols):
            axes_eval[r, c] = fig.add_subplot(gs[r, c])

    axes_r = None
    if has_r:
        axes_r = np.empty((nrows_r, ncols), dtype=object)
        base = nrows_eval + spacer
        for r in range(nrows_r):
            for c in range(ncols):
                axes_r[r, c] = fig.add_subplot(gs[base + r, c])

    # If spacer row exists, turn it off
    if has_r:
        for c in range(ncols):
            ax_sp = fig.add_subplot(gs[nrows_eval, c])
            ax_sp.axis("off")

    # -----------------------------
    # Plot eval metrics into axes_eval (your existing logic)
    # -----------------------------
    x_positions = {"patchy": 0.0, "uniform": 1.0}
    x_labels = ["Patchy", "Uniform"]

    for idx, metric in enumerate(metrics):
        r = idx // ncols
        c = idx % ncols
        ax = axes_eval[r, c]

        # Boxplots (patchy/uniform)
        group_patchy = cond_df["patchy"][metric].dropna()
        group_uniform = cond_df["uniform"][metric].dropna()
        box_data, box_pos = [], []
        if len(group_patchy) > 0:
            box_data.append(group_patchy.values); box_pos.append(x_positions["patchy"])
        if len(group_uniform) > 0:
            box_data.append(group_uniform.values); box_pos.append(x_positions["uniform"])
        if box_data:
            bp = ax.boxplot(box_data, positions=box_pos, widths=0.25,
                            patch_artist=True,
                            boxprops=dict(facecolor='white', alpha=0.6),
                            medianprops=dict(color='black'),
                            showcaps=True,
                            whiskerprops=dict(alpha=0.8))
            for patch in bp.get('boxes', []):
                patch.set_alpha(0.5)

        # Dots (+ your IQR bars if you added them earlier)
        for exp in exp_list:
            # patchy
            val_p = cond_df["patchy"].at[exp, metric] if metric in cond_df["patchy"].columns else np.nan
            if not pd.isna(val_p):
                x = x_positions["patchy"] + jitter_map[exp]
                ax.scatter(x, val_p, color=color_map[exp], s=36, edgecolor='k', linewidth=0.4, zorder=3)
                # If you have iqr bars:
                if "iqr" in locals() and exp in iqr.get("patchy", {}) and metric in iqr["patchy"][exp]:
                    q25, q75 = iqr["patchy"][exp][metric]
                    ax.vlines(x, q25, q75, color=color_map[exp], alpha=0.6, linewidth=1.2, zorder=2)

            # uniform
            val_u = cond_df["uniform"].at[exp, metric] if metric in cond_df["uniform"].columns else np.nan
            if not pd.isna(val_u):
                x = x_positions["uniform"] + jitter_map[exp]
                ax.scatter(x, val_u, color=color_map[exp], s=36, edgecolor='k', linewidth=0.4, zorder=3)
                if "iqr" in locals() and exp in iqr.get("uniform", {}) and metric in iqr["uniform"][exp]:
                    q25, q75 = iqr["uniform"][exp][metric]
                    ax.vlines(x, q25, q75, color=color_map[exp], alpha=0.6, linewidth=1.2, zorder=2)

        # group means diamonds
        if not group_patchy.empty:
            ax.scatter(x_positions["patchy"], group_patchy.mean(), marker='D', s=80, color='white', edgecolor='k', zorder=4)
        if not group_uniform.empty:
            ax.scatter(x_positions["uniform"], group_uniform.mean(), marker='D', s=80, color='white', edgecolor='k', zorder=4)

        ax.set_xticks([x_positions["patchy"], x_positions["uniform"]])
        ax.set_xticklabels(x_labels)
        ax.set_title(metric, fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xlim(-0.6, 1.6)

    # Hide unused eval axes
    for idx in range(nmetrics, nrows_eval * ncols):
        r = idx // ncols
        c = idx % ncols
        axes_eval[r, c].axis("off")

    # -----------------------------
    # Plot train r_components into axes_r (if present)
    # -----------------------------
    if has_r:
        x0 = 1.0
        for idx, rk in enumerate(all_r_keys):
            rr = idx // ncols
            cc = idx % ncols
            ax = axes_r[rr, cc]

            vals = r_df[rk].dropna()
            if len(vals) > 0:
                bp = ax.boxplot([vals.values], positions=[x0], widths=0.3,
                                patch_artist=True,
                                boxprops=dict(facecolor='white', alpha=0.6),
                                medianprops=dict(color='black'),
                                showcaps=True,
                                whiskerprops=dict(alpha=0.8))
                for patch in bp.get('boxes', []):
                    patch.set_alpha(0.5)

            for exp in exp_list:
                v = r_df.at[exp, rk]
                if pd.isna(v):
                    continue
                x = x0 + jitter_map[exp]
                ax.scatter(x, v, color=color_map[exp], s=36, edgecolor='k', linewidth=0.4, zorder=3)

            ax.set_title(rk, fontsize=9)
            ax.set_xlim(0.5, 1.5)
            ax.set_xticks([x0])
            ax.set_xticklabels(["final"])
            ax.grid(True, linestyle='--', alpha=0.4)

        # Hide unused r axes
        for idx in range(n_r, nrows_r * ncols):
            rr = idx // ncols
            cc = idx % ncols
            axes_r[rr, cc].axis("off")

    # -----------------------------
    # One shared legend + titles + save
    # -----------------------------
    def tail_label(s, max_chars=40):
        s = str(s)
        if len(s) <= max_chars:
            return s
        return "…" + s[-max_chars:]

    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                markerfacecolor=color_map[e], markeredgecolor='k',
                markersize=7, label=tail_label(e))
        for e in exp_list
    ]
    fig.legend(handles=handles, loc='lower right', bbox_to_anchor=(0.98, 0.02),
            title='Experiment (top-level folder)', frameon=True)

    fig.suptitle("Eval metrics (top) and final train reward components (bottom)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.93, 0.95])

    out_path = out_dir / args.out  # keep your existing output name
    plt.savefig(out_path, dpi=300)
    print(f"Saved combined figure to: {out_path}")


if __name__ == "__main__":
    main()

