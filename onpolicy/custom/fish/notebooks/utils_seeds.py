"""
utils_seeds.py — shared data loading and plotting for seed_ranking.ipynb / seed_analysis.ipynb.

All functions accept a list of run_dirs (Path objects returned by discover_run_dirs).
Plotting functions accept pre-built axes so callers control layout.
"""

import json
import warnings
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")


# ── Constants ─────────────────────────────────────────────────────────────────

SENSOR_MULTI_KEYS = {
    "m1a1k1": "m1a1k1_patchy_square",  # all sensors  (= main eval)
    "m0a0k0": "m0a0k0_patchy_square",  # no sensors
    "m0a0k1": "m0a0k1_patchy_square",  # knollen only
    "m0a1k1": "m0a1k1_patchy_square",  # amp + knollen (no morm)
    "m1a0k1": "m1a0k1_patchy_square",  # morm + knollen (no amp)
    "m1a1k0": "m1a1k0_patchy_square",  # morm + amp (no knollen)
}

SENSOR_1FISH_KEYS = {
    "full":       "1fish_m1a1k1_patchy_square",
    "no_morm":    "1fish_m0a1k1_patchy_square",
    "no_amp":     "1fish_m1a0k1_patchy_square",
    "no_knollen": "1fish_m1a1k0_patchy_square",
}

NFISH_KEYS    = [f"nfish{n}_m1a1k1_patchy_square" for n in [1, 2, 3, 4]]
TWO_F1P_CONDS = ["2f1p_AltB", "2f1p_AeqB", "2f1p_AgtB"]
TWO_F1P_CTRLS = {"ctrl_a": "2f1p_control_a", "ctrl_b": "2f1p_control_b"}


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover_run_dirs(results_dir, group_folder_name):
    """Return sorted list of run-dir Paths (one per seed, most-recent timestamp)."""
    group_dir = Path(results_dir) / group_folder_name
    assert group_dir.is_dir(), f"Group folder not found: {group_dir}"
    run_dirs = []
    for seed_dir in sorted(group_dir.iterdir()):
        if not seed_dir.is_dir():
            continue
        timestamps = sorted(
            d for d in seed_dir.iterdir()
            if d.is_dir() and (d / "logs" / "all_args.json").exists()
        )
        if timestamps:
            run_dirs.append(timestamps[-1])
    return run_dirs


# ── Internal helpers ──────────────────────────────────────────────────────────

def _seed(run_dir):
    with open(run_dir / "logs" / "all_args.json") as f:
        return json.load(f).get("seed")


def _ols_row(ols_path, x_pat, y_pat):
    p = Path(ols_path)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    mask = (df["x_var"].str.contains(x_pat, case=False) &
            df["y_var"].str.contains(y_pat, case=False))
    return df[mask].iloc[0].to_dict() if mask.any() else {}


def _ep_stats(run_dir, spec_key):
    """Return dict {food, biting, theil, p_eod, interactions} from per_env_ep.pkl."""
    p = run_dir / "evals" / spec_key / "derived" / "per_env_ep.pkl"
    if not p.exists():
        return {}
    df = pd.read_pickle(p)
    return {
        "food":         df["food_eaten"].mean(),
        "biting":       df["num_biting_events"].mean(),
        "theil":        df["food_eaten_theil"].mean() if "food_eaten_theil" in df.columns else np.nan,
        "p_eod":        df["p_emit_eod"].mean(),
        "interactions": df["num_interactions"].mean(),
    }


def _agent_stats(run_dir, spec_key, agent_id):
    """Return (food_mean, biting_mean) for a single agent_id from per_env_ep_agent.pkl."""
    p = run_dir / "evals" / spec_key / "derived" / "per_env_ep_agent.pkl"
    if not p.exists():
        return np.nan, np.nan
    df = pd.read_pickle(p)
    sub = df[df["agent_id"] == agent_id]
    if sub.empty:
        return np.nan, np.nan
    return sub["food_eaten"].mean(), sub["num_biting_events"].mean()


def _r_components(tm_row_series):
    """Parse r_components_json column into a dict of component means."""
    out = {}
    for v in tm_row_series:
        try:
            for k, val in json.loads(v).items():
                out.setdefault(k, []).append(val)
        except Exception:
            pass
    return {k: float(np.nanmean(vals)) for k, vals in out.items()}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_ranking_df(run_dirs, main_eval="m1a1k1_patchy_square", last_n=5):
    """
    Lightweight per-seed DataFrame for seed_ranking.ipynb.
    Reads: all_args.json, train_metrics.csv, per_env_ep.pkl, ols_regression_summary.csv.
    """
    records = []
    for run_dir in run_dirs:
        rec = {"run_dir": str(run_dir), "seed": _seed(run_dir)}

        with open(run_dir / "logs" / "all_args.json") as f:
            args = json.load(f)
        rec["num_env_steps"] = args.get("num_env_steps")

        tm_path = run_dir / "logs" / "train_metrics.csv"
        if tm_path.exists() and tm_path.stat().st_size > 0:
            tm = pd.read_csv(tm_path)
            if not tm.empty:
                tail = tm.iloc[-last_n:]
                rec["train_final_reward"] = tail["average_episode_reward"].mean()
                comps = _r_components(tail["r_components_json"])
                rec["train_final_r_food"]   = comps.get("r_food", np.nan)
                rec["train_final_r_bitten"] = comps.get("r_bitten", np.nan)
                rec["train_n_steps"] = int(tm["step"].iloc[-1])

        ep_path = run_dir / "evals" / main_eval / "derived" / "per_env_ep.pkl"
        if ep_path.exists():
            ep = pd.read_pickle(ep_path)
            rec["food_mean"]   = ep["food_eaten"].mean()
            rec["food_std"]    = ep["food_eaten"].std()
            rec["theil_mean"]  = ep["food_eaten_theil"].mean()
            rec["biting_mean"] = ep["num_biting_events"].mean()
            rec["p_emit_eod"]  = ep["p_emit_eod"].mean()

        ols_path = run_dir / "evals" / main_eval / "analyses" / "general" / "ols_regression_summary.csv"
        for x_pat, y_pat, prefix in [
            ("Agent size", "Food eaten", "size_food"),
            ("Agent size", "EOD rate",   "size_eod"),
        ]:
            row = _ols_row(ols_path, x_pat, y_pat)
            rec[f"{prefix}_slope"] = row.get("slope", np.nan)
            rec[f"{prefix}_r2"]    = row.get("r2", np.nan)
            rec[f"{prefix}_p"]     = row.get("p", np.nan)

        records.append(rec)

    return pd.DataFrame(records).sort_values("seed").reset_index(drop=True)


def load_analysis_df(run_dirs, main_eval="m1a1k1_patchy_square", last_n=5):
    """
    Comprehensive per-seed DataFrame for seed_analysis.ipynb.
    Covers: training, main eval, multi-fish sensor ablation, 1-fish sensor ablation,
    n-fish scaling, 2f1p competitive outcomes.
    """
    records = []
    for run_dir in run_dirs:
        rec = {"run_dir": str(run_dir), "seed": _seed(run_dir)}

        # ── Training ──────────────────────────────────────────────────────────
        tm = pd.read_csv(run_dir / "logs" / "train_metrics.csv")
        tail = tm.iloc[-last_n:]
        rec["train_final_reward"]    = tail["average_episode_reward"].mean()
        rec["train_reward_std_tail"] = tm["average_episode_reward"].iloc[-10:].std()
        comps = _r_components(tail["r_components_json"])
        rec["train_final_r_food"]   = comps.get("r_food", np.nan)
        rec["train_final_r_bitten"] = comps.get("r_bitten", np.nan)

        # ── Main eval ─────────────────────────────────────────────────────────
        ep = pd.read_pickle(run_dir / "evals" / main_eval / "derived" / "per_env_ep.pkl")
        rec["food_mean"]         = ep["food_eaten"].mean()
        rec["food_std"]          = ep["food_eaten"].std()
        rec["theil_mean"]        = ep["food_eaten_theil"].mean()
        rec["biting_mean"]       = ep["num_biting_events"].mean()
        rec["p_emit_eod"]        = ep["p_emit_eod"].mean()
        rec["interactions_mean"] = ep["num_interactions"].mean()
        rec["polarization_mean"] = ep["polarization"].mean() if "polarization" in ep.columns else np.nan
        rec["mean_nn_dist"]      = ep["mean_nn_distance_cm"].mean() if "mean_nn_distance_cm" in ep.columns else np.nan

        ols_path = run_dir / "evals" / main_eval / "analyses" / "general" / "ols_regression_summary.csv"
        for x_pat, y_pat, prefix in [
            ("Agent size", "Food eaten",   "size_food"),
            ("Agent size", "EOD rate",     "size_eod"),
            ("Agent size", "displacement", "size_disp"),
        ]:
            row = _ols_row(ols_path, x_pat, y_pat)
            rec[f"{prefix}_slope"] = row.get("slope", np.nan)
            rec[f"{prefix}_r2"]    = row.get("r2", np.nan)
            rec[f"{prefix}_p"]     = row.get("p", np.nan)

        # ── Multi-fish sensor ablation ────────────────────────────────────────
        for key, spec in SENSOR_MULTI_KEYS.items():
            st = _ep_stats(run_dir, spec)
            rec[f"multi_{key}_food"]   = st.get("food",   np.nan)
            rec[f"multi_{key}_biting"] = st.get("biting", np.nan)
            rec[f"multi_{key}_theil"]  = st.get("theil",  np.nan)
            rec[f"multi_{key}_p_eod"]  = st.get("p_eod",  np.nan)

        base = rec.get("multi_m1a1k1_food", np.nan)
        for label, key in [("morm", "m0a1k1"), ("amp", "m1a0k1"), ("knollen", "m1a1k0")]:
            ablated = rec.get(f"multi_{key}_food", np.nan)
            rec[f"multi_{label}_contrib"] = (base - ablated) / base if base else np.nan
        rec["multi_sensing_benefit"] = (base - rec.get("multi_m0a0k0_food", np.nan)) / base if base else np.nan

        # ── Single-fish sensor ablation ───────────────────────────────────────
        for label, spec in SENSOR_1FISH_KEYS.items():
            p = run_dir / "evals" / spec / "derived" / "per_env_ep_agent.pkl"
            rec[f"1fish_{label}_food"] = pd.read_pickle(p)["food_eaten"].mean() if p.exists() else np.nan

        full1 = rec.get("1fish_full_food", np.nan)
        for label, key in [("morm", "no_morm"), ("amp", "no_amp"), ("knollen", "no_knollen")]:
            ablated = rec.get(f"1fish_{key}_food", np.nan)
            rec[f"1fish_{label}_contrib"] = (full1 - ablated) / full1 if full1 else np.nan

        # ── N-fish scaling ────────────────────────────────────────────────────
        nfish_food = {}
        for spec in NFISH_KEYS:
            n = int(spec.split("nfish")[1].split("_")[0])
            st = _ep_stats(run_dir, spec)
            nfish_food[n] = st.get("food", np.nan)
            rec[f"food_nfish{n}"] = nfish_food[n]
        ns  = np.array([k for k, v in nfish_food.items() if not np.isnan(v)])
        fds = np.array([nfish_food[k] for k in ns])
        rec["nfish_slope"] = float(np.polyfit(ns, fds, 1)[0]) if len(ns) >= 2 else np.nan

        # ── 2f1p ─────────────────────────────────────────────────────────────
        for cond in TWO_F1P_CONDS:
            short = cond.replace("2f1p_", "")
            a_food, a_bites = _agent_stats(run_dir, cond, 0)
            b_food, b_bites = _agent_stats(run_dir, cond, 1)
            rec[f"a_food_{short}"]  = a_food;  rec[f"b_food_{short}"]  = b_food
            rec[f"a_bites_{short}"] = a_bites; rec[f"b_bites_{short}"] = b_bites

        a_ctrl, _ = _agent_stats(run_dir, TWO_F1P_CTRLS["ctrl_a"], 0)
        b_ctrl, _ = _agent_stats(run_dir, TWO_F1P_CTRLS["ctrl_b"], 1)
        rec["a_food_ctrl"] = a_ctrl
        rec["b_food_ctrl"] = b_ctrl
        rec["b_food_size_effect"]  = rec.get("b_food_AgtB",  np.nan) - rec.get("b_food_AltB",  np.nan)
        rec["b_bites_size_effect"] = rec.get("b_bites_AgtB", np.nan) - rec.get("b_bites_AltB", np.nan)

        ols2 = run_dir / "multi_eval" / "2f1p" / "food_vs_size_ols.csv"
        if ols2.exists():
            df2 = pd.read_csv(ols2)
            br = df2[df2["role"] == "B"]
            if not br.empty:
                rec["b_food_slope_2f1p"] = br["slope"].iloc[0]
                rec["b_food_r2_2f1p"]    = br["r2"].iloc[0]

        records.append(rec)

    return pd.DataFrame(records).sort_values("seed").reset_index(drop=True)


def rank_seeds(df,
               rank_targets=None):
    """Add rank_{col} and combined_rank columns. Returns df copy."""
    if rank_targets is None:
        rank_targets = {
            "size_food_slope": False,   # False = higher is better
            "food_mean":       False,
            "size_food_r2":    False,
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


# ── Ranking plots (single-panel, self-contained) ──────────────────────────────

def plot_ranking_landscape(df, save_path=None):
    """food_mean vs size_food_slope scatter, coloured by R²."""
    r2 = df.get("size_food_r2", pd.Series([0.5] * len(df)))
    fig, ax = plt.subplots(figsize=(5, 4))
    sc = ax.scatter(df["food_mean"], df["size_food_slope"],
        c=r2, cmap="Blues", s=90, edgecolors="k", linewidths=0.5, vmin=0, vmax=0.2)
    plt.colorbar(sc, ax=ax, label="R² (food ~ size)")
    for _, row in df.iterrows():
        ax.annotate(str(int(row["seed"])), (row["food_mean"], row["size_food_slope"]),
            textcoords="offset points", xytext=(4, 3), fontsize=8)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Mean food eaten / episode")
    ax.set_ylabel("food ~ size slope")
    ax.set_title("Ranking landscape")
    plt.tight_layout()
    _maybe_save(fig, save_path)
    plt.show()
    return fig


def plot_ranking_bar(df, save_path=None):
    """Combined rank horizontal bar chart."""
    rdf = df.sort_values("combined_rank")
    fig, ax = plt.subplots(figsize=(4, max(3, len(df) * 0.3)))
    ax.barh([f"S{int(s)}" for s in rdf["seed"]], rdf["combined_rank"],
        color="steelblue", edgecolor="k", linewidth=0.4)
    ax.set_xlabel("Combined rank (lower = better)")
    ax.set_title("Seed ranking")
    plt.tight_layout()
    _maybe_save(fig, save_path)
    plt.show()
    return fig


def plot_slope_vs_r2(df, save_path=None):
    """food ~ size slope vs R², coloured by food_mean."""
    r2 = df.get("size_food_r2", pd.Series([0.0] * len(df)))
    fig, ax = plt.subplots(figsize=(5, 4))
    sc = ax.scatter(df["size_food_slope"], r2,
        c=df["food_mean"], cmap="plasma", s=90, edgecolors="k", linewidths=0.5)
    plt.colorbar(sc, ax=ax, label="food_mean")
    for _, row in df.iterrows():
        ax.annotate(str(int(row["seed"])), (row["size_food_slope"], row.get("size_food_r2", 0)),
            textcoords="offset points", xytext=(4, 3), fontsize=8)
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("food ~ size slope")
    ax.set_ylabel("R² (food ~ size)")
    ax.set_title("Slope vs. fit quality")
    plt.tight_layout()
    _maybe_save(fig, save_path)
    plt.show()
    return fig


def plot_reward_curve(run_dirs, save_path=None):
    """Total episode reward training curve (5-pt EMA), one line per seed."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for run_dir in run_dirs:
        tm_path = run_dir / "logs" / "train_metrics.csv"
        if not tm_path.exists() or tm_path.stat().st_size == 0:
            continue
        tm = pd.read_csv(tm_path)
        with open(run_dir / "logs" / "all_args.json") as f:
            seed = json.load(f).get("seed", "?")
        ax.plot(tm["step"] / 1e6,
                tm["average_episode_reward"].rolling(5, min_periods=1).mean(),
                label=f"S{seed}", alpha=0.8, lw=1)
    ax.set_xlabel("Steps (M)")
    ax.set_ylabel("Avg episode reward (5-pt EMA)")
    ax.set_title("Training curves — total reward")
    ax.legend(fontsize=6, ncol=4)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _maybe_save(fig, save_path)
    plt.show()
    return fig


def plot_r_food_curve(run_dirs, save_path=None):
    """r_food reward component training curve (5-pt EMA), one line per seed."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for run_dir in run_dirs:
        tm_path = run_dir / "logs" / "train_metrics.csv"
        if not tm_path.exists() or tm_path.stat().st_size == 0:
            continue
        tm = pd.read_csv(tm_path)
        with open(run_dir / "logs" / "all_args.json") as f:
            seed = json.load(f).get("seed", "?")
        r_food = tm["r_components_json"].apply(
            lambda v: json.loads(v).get("r_food", np.nan) if isinstance(v, str) else np.nan
        ).rolling(5, min_periods=1).mean()
        ax.plot(tm["step"] / 1e6, r_food, label=f"S{seed}", alpha=0.8, lw=1)
    ax.set_xlabel("Steps (M)")
    ax.set_ylabel("r_food (5-pt EMA)")
    ax.set_title("Training curves — food reward component")
    ax.legend(fontsize=6, ncol=4)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _maybe_save(fig, save_path)
    plt.show()
    return fig


def plot_r_bitten_curve(run_dirs, save_path=None):
    """r_bitten reward component training curve (5-pt EMA), one line per seed."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for run_dir in run_dirs:
        tm_path = run_dir / "logs" / "train_metrics.csv"
        if not tm_path.exists() or tm_path.stat().st_size == 0:
            continue
        tm = pd.read_csv(tm_path)
        with open(run_dir / "logs" / "all_args.json") as f:
            seed = json.load(f).get("seed", "?")
        r_bitten = tm["r_components_json"].apply(
            lambda v: json.loads(v).get("r_bitten", np.nan) if isinstance(v, str) else np.nan
        ).rolling(5, min_periods=1).mean()
        ax.plot(tm["step"] / 1e6, r_bitten, label=f"S{seed}", alpha=0.8, lw=1)
    ax.set_xlabel("Steps (M)")
    ax.set_ylabel("r_bitten (5-pt EMA)")
    ax.set_title("Training curves — bitten reward component")
    ax.legend(fontsize=6, ncol=4)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _maybe_save(fig, save_path)
    plt.show()
    return fig


# ── Ranking plots (multi-panel, for composing into analysis figures) ──────────

def plot_ranking_scatter(df, axes, save_path=None):
    """3-panel ranking landscape. axes must have length 3."""
    n = len(df)
    r2 = df.get("size_food_r2", pd.Series([0.5] * n))

    sc = axes[0].scatter(df["food_mean"], df["size_food_slope"],
        c=r2, cmap="viridis", s=90, edgecolors="k", linewidths=0.5, vmin=0, vmax=0.2)
    plt.colorbar(sc, ax=axes[0], label="R² (food ~ size)")
    for _, row in df.iterrows():
        axes[0].annotate(str(int(row["seed"])), (row["food_mean"], row["size_food_slope"]),
            textcoords="offset points", xytext=(4, 3), fontsize=8)
    axes[0].axhline(0, color="gray", lw=0.8, ls="--")
    axes[0].set_xlabel("Mean food eaten / episode")
    axes[0].set_ylabel("food ~ size slope")
    axes[0].set_title("Ranking landscape")

    rdf = df.sort_values("combined_rank")
    axes[1].barh([f"S{int(s)}" for s in rdf["seed"]], rdf["combined_rank"],
        color="steelblue", edgecolor="k", linewidth=0.4)
    axes[1].set_xlabel("Combined rank (lower = better)")
    axes[1].set_title("Seed ranking")

    sc2 = axes[2].scatter(df["size_food_slope"], r2,
        c=df["food_mean"], cmap="plasma", s=90, edgecolors="k", linewidths=0.5)
    plt.colorbar(sc2, ax=axes[2], label="food_mean")
    for _, row in df.iterrows():
        axes[2].annotate(str(int(row["seed"])), (row["size_food_slope"], row.get("size_food_r2", 0)),
            textcoords="offset points", xytext=(4, 3), fontsize=8)
    axes[2].axvline(0, color="gray", lw=0.8, ls="--")
    axes[2].set_xlabel("food ~ size slope")
    axes[2].set_ylabel("R² (food ~ size)")
    axes[2].set_title("Slope vs. fit quality")
    _maybe_save(axes[0].get_figure(), save_path)


def plot_training_curves(run_dirs, axes, save_path=None):
    """2-panel: total reward + r_food component, EMA-smoothed. axes must have length 2."""
    for run_dir in run_dirs:
        tm_path = run_dir / "logs" / "train_metrics.csv"
        if not tm_path.exists() or tm_path.stat().st_size == 0:
            continue
        tm = pd.read_csv(tm_path)
        with open(run_dir / "logs" / "all_args.json") as f:
            seed = json.load(f).get("seed", "?")
        x = tm["step"] / 1e6
        axes[0].plot(x, tm["average_episode_reward"].rolling(5, min_periods=1).mean(),
            label=f"S{seed}", alpha=0.8, lw=1)
        r_food = tm["r_components_json"].apply(
            lambda v: json.loads(v).get("r_food", np.nan) if isinstance(v, str) else np.nan
        ).rolling(5, min_periods=1).mean()
        axes[1].plot(x, r_food, label=f"S{seed}", alpha=0.8, lw=1)

    for ax, title, ylabel in zip(axes, ["Total reward", "Food reward component"],
                                  ["Avg reward (5-pt EMA)", "r_food (5-pt EMA)"]):
        ax.set_xlabel("Steps (M)"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=6, ncol=4); ax.grid(True, alpha=0.3)
    _maybe_save(axes[0].get_figure(), save_path)


# ── Analysis plots ────────────────────────────────────────────────────────────

def _seed_cmap(n):
    return cm.get_cmap("tab20", n)


def plot_main_eval_bars(df, axes, save_path=None):
    """3 panels: food_mean bars, size_food_slope bars, food vs slope scatter. axes length 3."""
    x = np.arange(len(df)); labels = [f"S{int(s)}" for s in df["seed"]]

    axes[0].bar(x, df["food_mean"], yerr=df["food_std"], capsize=2,
        color="steelblue", edgecolor="k", linewidth=0.4)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=7)
    axes[0].set_ylabel("Mean food / episode"); axes[0].set_title("Foraging performance")

    colors = ["#2ca02c" if v > 0 else "#d62728" for v in df["size_food_slope"]]
    axes[1].bar(x, df["size_food_slope"], color=colors, edgecolor="k", linewidth=0.4)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=7)
    axes[1].set_ylabel("food ~ size slope"); axes[1].set_title("Size-dependent foraging")

    sc = axes[2].scatter(df["food_mean"], df["size_food_slope"],
        c=df.get("size_food_r2", 0.1), cmap="viridis", s=70, edgecolors="k", linewidths=0.5, vmin=0, vmax=0.2)
    plt.colorbar(sc, ax=axes[2], label="R²"); axes[2].axhline(0, color="gray", lw=0.8, ls="--")
    for _, row in df.iterrows():
        axes[2].annotate(str(int(row["seed"])), (row["food_mean"], row["size_food_slope"]),
            textcoords="offset points", xytext=(3, 2), fontsize=6)
    axes[2].set_xlabel("food_mean"); axes[2].set_ylabel("size_food_slope")
    axes[2].set_title("Foraging vs. size effect")
    _maybe_save(axes[0].get_figure(), save_path)


def plot_main_eval_violins(run_dirs, main_eval, axes, save_path=None):
    """3 violin panels: food, theil, biting. axes length 3."""
    dfs = []
    for run_dir in run_dirs:
        ep_path = run_dir / "evals" / main_eval / "derived" / "per_env_ep.pkl"
        if not ep_path.exists(): continue
        with open(run_dir / "logs" / "all_args.json") as f:
            seed = json.load(f).get("seed")
        ep = pd.read_pickle(ep_path); ep["seed"] = seed
        dfs.append(ep)
    if not dfs:
        return
    ep_all = pd.concat(dfs, ignore_index=True)
    for ax, col, title in zip(axes,
        ["food_eaten", "food_eaten_theil", "num_biting_events"],
        ["Food eaten / episode", "Theil inequality", "Biting events / episode"]):
        sns.violinplot(data=ep_all, x="seed", y=col, ax=ax, inner="box", linewidth=0.8, color="steelblue")
        ax.set_title(title); ax.set_xlabel("Seed")
    _maybe_save(axes[0].get_figure(), save_path)


def plot_sensor_multi_profiles(df, axes, save_path=None):
    """
    4 panels: food, biting, theil, p_eod across all sensor conditions.
    Each line is one seed; thick dashed = population mean. axes length 4.
    """
    multi_order = ["m1a1k1", "m0a0k0", "m0a0k1", "m0a1k1", "m1a0k1", "m1a1k0"]
    metrics = [("food", "Mean food eaten"), ("biting", "Biting events"),
               ("theil", "Theil inequality"), ("p_eod", "P(emit EOD)")]
    cmap = _seed_cmap(len(df))

    for ax, (metric, label) in zip(axes, metrics):
        for i, (_, row) in enumerate(df.iterrows()):
            vals = [row.get(f"multi_{k}_{metric}", np.nan) for k in multi_order]
            ax.plot(range(len(multi_order)), vals, marker="o", lw=1.2, ms=4,
                color=cmap(i), alpha=0.7, label=f"S{int(row['seed'])}")
        means = [df[f"multi_{k}_{metric}"].mean() if f"multi_{k}_{metric}" in df.columns else np.nan
                 for k in multi_order]
        ax.plot(range(len(multi_order)), means, "k--", lw=2, marker="D", ms=5, zorder=5)
        ax.set_xticks(range(len(multi_order)))
        ax.set_xticklabels(multi_order, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(label); ax.set_title(label); ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=5, ncol=4)
    _maybe_save(axes[0].get_figure(), save_path)


def plot_sensor_multi_contrib(df, axes, save_path=None):
    """
    2 panels: fractional contribution bars, and multi vs 1-fish contribution scatter.
    axes length 2.
    """
    x = np.arange(len(df)); w = 0.18
    contrib_pairs = [
        ("Mormyromast", "multi_morm_contrib",    "#4C72B0"),
        ("Ampullary",   "multi_amp_contrib",      "#55A868"),
        ("Knollen",     "multi_knollen_contrib",  "#C44E52"),
        ("All sensors", "multi_sensing_benefit",  "#8172B2"),
    ]
    for i, (name, col, color) in enumerate(contrib_pairs):
        vals = df[col] if col in df.columns else [np.nan] * len(df)
        axes[0].bar(x + (i - 1.5) * w, vals, w, label=name, color=color,
            edgecolor="k", linewidth=0.4)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"S{int(s)}" for s in df["seed"]], fontsize=7)
    axes[0].set_ylabel("Fractional food contribution")
    axes[0].set_title("Multi-fish per-sensor food contribution")
    axes[0].legend(fontsize=8)

    for name, multi_col, fish_col, mk in [
        ("Mormyromast", "multi_morm_contrib", "1fish_morm_contrib", "o"),
        ("Ampullary",   "multi_amp_contrib",  "1fish_amp_contrib",  "s"),
    ]:
        axes[1].scatter(df.get(multi_col, np.nan), df.get(fish_col, np.nan),
            s=60, marker=mk, edgecolors="k", linewidths=0.5, label=name, alpha=0.85)
    for _, row in df.iterrows():
        axes[1].annotate(str(int(row["seed"])),
            (row.get("multi_morm_contrib", np.nan), row.get("1fish_morm_contrib", np.nan)),
            textcoords="offset points", xytext=(3, 2), fontsize=6)
    lims = [0, 1]
    axes[1].plot(lims, lims, "k--", lw=0.8, alpha=0.5)
    axes[1].set_xlabel("Multi-fish contribution"); axes[1].set_ylabel("Single-fish contribution")
    axes[1].set_title("Multi- vs. single-fish sensor contribution\n(above diagonal = social amplification)")
    axes[1].legend(fontsize=8)
    _maybe_save(axes[0].get_figure(), save_path)


def plot_sensor_multi_biting_theil(df, axes, save_path=None):
    """Biting and Theil boxplots across sensor conditions. axes length 2."""
    multi_order = ["m1a1k1", "m0a0k0", "m0a0k1", "m0a1k1", "m1a0k1", "m1a1k0"]
    for ax, metric, label in zip(axes, ["biting", "theil"], ["Biting events", "Theil inequality"]):
        data = [df[f"multi_{k}_{metric}"].values
                for k in multi_order if f"multi_{k}_{metric}" in df.columns]
        bp = ax.boxplot(data, positions=range(len(data)), widths=0.5,
            patch_artist=True, medianprops=dict(color="k", lw=1.5))
        for patch in bp["boxes"]:
            patch.set(facecolor="steelblue", alpha=0.6)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(multi_order, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(f"{label} by sensor condition\n(boxes = inter-seed variability)")
        ax.grid(True, alpha=0.3, axis="y")
    _maybe_save(axes[0].get_figure(), save_path)


def plot_sensor_1fish(df, axes, save_path=None):
    """2 panels: absolute food and fractional contribution for 1-fish ablations. axes length 2."""
    cond_keys  = list(SENSOR_1FISH_KEYS.keys())
    cond_names = {"full": "All sensors", "no_morm": "No morm", "no_amp": "No amp", "no_knollen": "No knollen"}
    cmap = _seed_cmap(len(df))
    x = np.arange(len(cond_keys)); w = 0.7 / max(len(df), 1)

    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row.get(f"1fish_{k}_food", np.nan) for k in cond_keys]
        offset = (i - len(df) / 2) * w
        axes[0].bar(x + offset, vals, w, color=cmap(i), edgecolor="k", linewidth=0.2, alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels([cond_names[k] for k in cond_keys], fontsize=9)
    axes[0].set_ylabel("Mean food (1-fish)"); axes[0].set_title("1-fish foraging by sensor condition")

    contrib_keys = [("Mormyromast", "1fish_morm_contrib"), ("Ampullary", "1fish_amp_contrib"),
                    ("Knollen", "1fish_knollen_contrib")]
    xc = np.arange(len(contrib_keys)); wc = 0.7 / max(len(df), 1)
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row.get(col, np.nan) for _, col in contrib_keys]
        offset = (i - len(df) / 2) * wc
        axes[1].bar(xc + offset, vals, wc, color=cmap(i), edgecolor="k", linewidth=0.2, alpha=0.85)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xticks(xc); axes[1].set_xticklabels([n for n, _ in contrib_keys], fontsize=9)
    axes[1].set_ylabel("Fractional contribution"); axes[1].set_title("1-fish per-sensor contribution")
    _maybe_save(axes[0].get_figure(), save_path)


def plot_nfish(df, axes, save_path=None):
    """2 panels: absolute and normalised food vs. N fish. axes length 2."""
    ns = [1, 2, 3, 4]
    cmap = _seed_cmap(len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        fns = [row.get(f"food_nfish{n}", np.nan) for n in ns]
        valid = [(n, f) for n, f in zip(ns, fns) if not np.isnan(f)]
        if not valid: continue
        vn, vf = zip(*valid)
        axes[0].plot(vn, vf, marker="o", label=f"S{int(row['seed'])}", alpha=0.75, lw=1.2, ms=4, color=cmap(i))
        f1 = row.get("food_nfish1", np.nan)
        if not np.isnan(f1) and f1 > 0:
            axes[1].plot(vn, [f / f1 for f in vf], marker="o", alpha=0.75, lw=1.2, ms=4, color=cmap(i))
    axes[1].plot(ns, ns, "k--", lw=1.2, label="linear")
    for ax, title, ylabel in zip(axes,
        ["Absolute food vs. N fish", "Normalised food (÷ nfish1)"],
        ["Mean food eaten", "Food / nfish1"]):
        ax.set_xlabel("N fish"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.set_xticks(ns); ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=5, ncol=4)
    _maybe_save(axes[0].get_figure(), save_path)


def plot_2f1p(df, axes, save_path=None):
    """3 panels: B food by condition, B biting by condition, alone vs social. axes length 3."""
    cond_labels = ["AltB", "AeqB", "AgtB"]
    cond_names  = ["A<B (B big)", "A=B", "A>B (B small)"]
    xc = np.arange(len(cond_labels)); ww = 0.7 / max(len(df), 1)
    cmap = _seed_cmap(len(df))

    for i, (_, row) in enumerate(df.iterrows()):
        offset = (i - len(df) / 2) * ww
        axes[0].bar(xc + offset, [row.get(f"b_food_{c}",  np.nan) for c in cond_labels], ww,
            color=cmap(i), edgecolor="k", linewidth=0.15, alpha=0.85)
        axes[1].bar(xc + offset, [row.get(f"b_bites_{c}", np.nan) for c in cond_labels], ww,
            color=cmap(i), edgecolor="k", linewidth=0.15, alpha=0.85)
    for ax in axes[:2]:
        ax.set_xticks(xc); ax.set_xticklabels(cond_names, fontsize=8)
    axes[0].set_ylabel("B food eaten"); axes[0].set_title("B foraging by size condition")
    axes[1].set_ylabel("B biting events"); axes[1].set_title("B aggression by size condition")

    xs2 = np.arange(len(df))
    a_social = df[[f"a_food_{c}" for c in cond_labels]].mean(axis=1)
    b_social = df[[f"b_food_{c}" for c in cond_labels]].mean(axis=1)
    for col, social, label, color in [
        ("a_food_ctrl", a_social, ("A alone", "A social"), "#4C72B0"),
        ("b_food_ctrl", b_social, ("B alone", "B social"), "#DD8452"),
    ]:
        axes[2].scatter(xs2, df[col],    s=60, marker="^", color=color, label=label[0], zorder=4)
        axes[2].scatter(xs2, social,     s=60, marker="o", color=color, alpha=0.5, label=label[1], zorder=4)
        for j in range(len(df)):
            axes[2].plot([j, j], [df[col].iloc[j], social.iloc[j]], color=color, lw=0.7, alpha=0.5)
    axes[2].set_xticks(xs2); axes[2].set_xticklabels([f"S{int(s)}" for s in df["seed"]], fontsize=6)
    axes[2].set_ylabel("Mean food"); axes[2].set_title("Alone vs. social foraging")
    axes[2].legend(fontsize=7)
    _maybe_save(axes[0].get_figure(), save_path)


def plot_corr_clustermap(df, save_path=None):
    """Pearson correlation clustermap across seeds. Returns (g, corr) or (None, None) if n < 3."""
    n = len(df)
    if n < 3:
        print(f"Only {n} seed(s) — need ≥ 3 for meaningful correlation. Skipping.")
        return None, None
    num_df = df.select_dtypes("float").drop(columns=["seed"], errors="ignore")
    num_df = num_df.loc[:, num_df.std() > 0]
    num_df = num_df.dropna(axis=1, thresh=max(2, n // 2))
    corr = num_df.corr(method="pearson")
    sz = max(12, len(corr) * 0.38)
    g = sns.clustermap(corr, cmap="coolwarm", vmin=-1, vmax=1,
        figsize=(sz, sz), linewidths=0.3, annot=False)
    g.fig.suptitle(f"Cross-metric Pearson correlation (n={n} seeds)", y=1.01, fontsize=12)
    _maybe_save(g.fig, save_path)
    return g, corr


def top_correlations(corr, n=25):
    """Return top-n pairwise correlations as a styled DataFrame."""
    long = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack().reset_index()
    )
    long.columns = ["metric_a", "metric_b", "pearson_r"]
    long["abs_r"] = long["pearson_r"].abs()
    top = long.sort_values("abs_r", ascending=False).head(n)
    return (top[["metric_a", "metric_b", "pearson_r"]]
            .style
            .background_gradient(cmap="RdBu_r", subset=["pearson_r"], vmin=-1, vmax=1)
            .format({"pearson_r": "{:.3f}"}))


def plot_scatter_matrix(df, focus_cols=None, save_path=None):
    """Pairplot for key metrics. Returns the PairGrid."""
    if focus_cols is None:
        focus_cols = [
            "food_mean", "size_food_slope", "size_food_r2",
            "theil_mean", "biting_mean",
            "multi_morm_contrib", "multi_amp_contrib", "multi_sensing_benefit",
            "nfish_slope",
            "b_food_size_effect", "b_bites_size_effect",
            "train_final_reward",
        ]
    focus_cols = [c for c in focus_cols if c in df.columns and df[c].notna().sum() >= 3]
    g = sns.PairGrid(df[focus_cols], diag_sharey=False, height=1.6)
    g.map_diag(sns.histplot, bins=8, color="steelblue")
    g.map_offdiag(sns.scatterplot, s=40, edgecolor="k", linewidth=0.4, alpha=0.8)
    for i, ax_row in enumerate(g.axes):
        for j, ax in enumerate(ax_row):
            if i != j:
                col_x, col_y = focus_cols[j], focus_cols[i]
                for _, row in df.iterrows():
                    ax.annotate(str(int(row["seed"])),
                        (row.get(col_x, np.nan), row.get(col_y, np.nan)),
                        textcoords="offset points", xytext=(2, 2), fontsize=4, alpha=0.6)
    g.fig.suptitle("Key metrics pairwise scatter", y=1.01, fontsize=10)
    _maybe_save(g.fig, save_path)
    return g
