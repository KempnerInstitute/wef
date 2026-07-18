"""
utils_seeds_homing.py — data loading and plotting for seed_ranking_homing.ipynb.

Mirrors the structure of utils_seeds.py but uses homing-specific metrics:
  - homing_episode_outcomes.csv (from analyses/homing/)
  - r_homing reward component from train_metrics.csv
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_homing_ranking_df(run_dirs, train_final_window=5):
    """
    Per-seed DataFrame for seed_ranking_homing.ipynb.
    Reads: all_args.json, train_metrics.csv, homing_episode_outcomes.csv.
    """
    records = []
    for run_dir in run_dirs:
        rec = {"run_dir": str(run_dir)}

        with open(run_dir / "logs" / "all_args.json") as f:
            args = json.load(f)
        rec["seed"] = args.get("seed")
        rec["num_env_steps"] = args.get("num_env_steps")

        # ── Training metrics ──────────────────────────────────────────────────
        tm_path = run_dir / "logs" / "train_metrics.csv"
        if tm_path.exists() and tm_path.stat().st_size > 0:
            tm = pd.read_csv(tm_path)
            if not tm.empty:
                tail = tm.iloc[-train_final_window:]
                rec["train_final_reward"] = tail["average_episode_reward"].mean()
                rec["train_n_steps"] = int(tm["step"].iloc[-1])
                for comp in ("r_homing", "r_homing_shaping", "r_homing_time_penalty"):
                    vals = tail["r_components_json"].apply(
                        lambda v, c=comp: json.loads(v).get(c, np.nan)
                        if isinstance(v, str) else np.nan
                    )
                    rec[f"train_final_{comp}"] = vals.mean()

        # ── Homing episode outcomes ───────────────────────────────────────────
        outcomes_path = (
            run_dir / "evals" / "homing" / "analyses" / "homing"
            / "homing_episode_outcomes.csv"
        )
        if outcomes_path.exists():
            eo = pd.read_csv(outcomes_path)
            d_start = eo["distance_to_target_start"]
            d_end   = eo["distance_to_target_end"]
            closure = (d_start - d_end) / d_start.replace(0, np.nan)

            successful = eo[eo["success"]]
            reached    = eo[eo["ever_reached_target"]]

            rec["num_episodes"]                = len(eo)
            rec["success_percent"]             = 100.0 * eo["success"].mean()
            rec["ever_reached_percent"]        = 100.0 * eo["ever_reached_target"].mean()
            rec["mean_initial_distance"]       = d_start.mean()
            rec["mean_end_distance"]           = d_end.mean()
            rec["median_end_distance"]         = d_end.median()
            rec["mean_distance_closure"]       = closure.mean()
            rec["mean_steps_to_target_reached"] = reached["steps_to_target"].mean()
            rec["mean_steps_to_target_success"] = successful["steps_to_target"].mean()
            rec["median_steps_to_target_success"] = successful["steps_to_target"].median()
        else:
            print(f"  [MISSING outcomes] {run_dir}")

        records.append(rec)

    return pd.DataFrame(records).sort_values("seed").reset_index(drop=True)


# ── Ranking ───────────────────────────────────────────────────────────────────

def rank_homing_seeds(df, rank_targets=None):
    """Add rank_{col} and combined_rank columns. Returns df copy."""
    if rank_targets is None:
        rank_targets = {
            "ever_reached_percent":  False,  # higher is better
            "mean_distance_closure": False,  # higher is better
            "mean_end_distance":     True,   # lower is better
        }
    df = df.copy()
    for col, ascending in rank_targets.items():
        if col in df.columns:
            df[f"rank_{col}"] = df[col].rank(method="min", ascending=ascending)
    rank_cols = [c for c in df.columns if c.startswith("rank_")]
    df["combined_rank"] = df[rank_cols].sum(axis=1)
    return df


# ── Save helper ───────────────────────────────────────────────────────────────

def _maybe_save(fig, save_path):
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_homing_ranking_landscape(df, save_path=None):
    """2-panel: ever_reached vs closure scatter + combined-rank bar."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sc = axes[0].scatter(
        df["ever_reached_percent"],
        df["mean_distance_closure"],
        c=df["mean_end_distance"],
        cmap="RdYlGn_r",
        s=90, edgecolors="k", linewidths=0.5,
    )
    plt.colorbar(sc, ax=axes[0], label="Mean end distance (cm)")
    for _, row in df.iterrows():
        axes[0].annotate(
            str(int(row["seed"])),
            (row["ever_reached_percent"], row["mean_distance_closure"]),
            textcoords="offset points", xytext=(4, 3), fontsize=8,
        )
    axes[0].axhline(0, color="gray", lw=0.8, ls="--")
    axes[0].set_xlabel("Ever reached target (%)")
    axes[0].set_ylabel("Mean distance closure fraction")
    axes[0].set_title("Homing ranking landscape")

    rdf = df.sort_values("combined_rank")
    axes[1].barh(
        [f"S{int(s)}" for s in rdf["seed"]],
        rdf["combined_rank"],
        color="steelblue", edgecolor="k", linewidth=0.4,
    )
    axes[1].set_xlabel("Combined rank (lower = better)")
    axes[1].set_title("Seed ranking")

    plt.tight_layout()
    _maybe_save(fig, save_path)
    plt.show()


def plot_homing_training_curves(run_dirs, save_path=None):
    """2-panel: total reward + r_homing component (5-pt EMA), one line per seed."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for run_dir in run_dirs:
        tm_path = run_dir / "logs" / "train_metrics.csv"
        if not tm_path.exists() or tm_path.stat().st_size == 0:
            continue
        tm = pd.read_csv(tm_path)
        if tm.empty:
            continue
        with open(run_dir / "logs" / "all_args.json") as f:
            seed = json.load(f).get("seed", "?")
        x = tm["step"] / 1e6

        axes[0].plot(
            x,
            tm["average_episode_reward"].rolling(5, min_periods=1).mean(),
            label=f"S{seed}", alpha=0.8, lw=1,
        )
        r_homing = tm["r_components_json"].apply(
            lambda v: json.loads(v).get("r_homing", np.nan) if isinstance(v, str) else np.nan
        ).rolling(5, min_periods=1).mean()
        axes[1].plot(x, r_homing, label=f"S{seed}", alpha=0.8, lw=1)

    for ax, title, ylabel in zip(
        axes,
        ["Training curves — total reward", "Training curves — r_homing"],
        ["Avg episode reward (5-pt EMA)", "r_homing (5-pt EMA)"],
    ):
        ax.set_xlabel("Steps (M)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=6, ncol=4)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _maybe_save(fig, save_path)
    plt.show()


# ── Summary ───────────────────────────────────────────────────────────────────

def print_homing_summary(df):
    """Print ranked seed recommendation to stdout."""
    print("=== Homing Seed Recommendation ===")
    for _, row in df.sort_values("combined_rank").iterrows():
        marker = "  <- BEST" if row["combined_rank"] == df["combined_rank"].min() else ""
        print(
            f"Seed {int(row['seed']):2d}: "
            f"ever_reached={row['ever_reached_percent']:.1f}%  "
            f"closure={row['mean_distance_closure']:.3f}  "
            f"end_dist={row['mean_end_distance']:.1f}cm  "
            f"rank={int(row['combined_rank'])}"
            f"{marker}"
        )
