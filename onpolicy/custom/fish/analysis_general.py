"""
Group-level and agent-level behavioural metrics.

Usage:
    python analysis_general.py --spec_dir path/to/evals/m1a1k1_patchy_square

Outputs to {spec_dir}/analyses/general/
  size_vs_food.pdf               — agent size vs food eaten per agent-episode
  size_vs_eod_rate.pdf           — agent size vs mean EOD rate per agent-episode
  size_vs_displacement.pdf       — agent size vs mean displacement per step
  size_vs_biting.pdf             — agent size vs biting events per episode
  food_rank_vs_size_rank.pdf     — Spearman ρ of size rank vs food rank, per episode
  food_ineq_vs_size_ineq_*.pdf   — per-episode food inequality vs size inequality
  food_theil_vs_size_advantage.pdf — food Theil index vs mean |size advantage| per episode
  exploratory_correlations.pdf   — Spearman correlation matrix
  eod_rate_vs_food.pdf           — mean EOD rate vs food eaten
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from analysis_style import (
    set_style, panel, save, AGENT_COLORS, SEMANTIC_COLORS, SIM_FPS, COND_PALETTE,
    load_consumption_times, time_to_X_log, time_to_X_linear,
    add_size_advantage,
)

OUT_SUBDIR = "analyses/general"


def _add_eod_rate_hz(df_agent: pd.DataFrame) -> pd.DataFrame:
    """Add mean_eod_rate_hz = p_emit_eod * fps_sim if not already present."""
    df_agent = df_agent.copy()
    if "mean_eod_rate_hz" not in df_agent.columns:
        if "p_emit_eod" in df_agent.columns:
            df_agent["mean_eod_rate_hz"] = df_agent["p_emit_eod"] * SIM_FPS
        else:
            df_agent["mean_eod_rate_hz"] = np.nan
    return df_agent


# ── regression helper ─────────────────────────────────────────────────────────

def scatter_regression(ax, x, y, color, xlabel, ylabel, s=18, text_pos=(0.05, 1.02)):
    """
    Scatter (open circles) + OLS regression line + shaded CI.
    Annotation shows R², r, p-values, and N on separate lines.
    Returns a stats dict (or None if regression was not fit).
    """
    from statsmodels.api import OLS, add_constant
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    ax.scatter(x, y, facecolors="none", edgecolors=color, s=s, linewidths=0.8, alpha=0.7)
    stats_out = None
    if len(x) > 3 and np.unique(x).size > 1:
        X = add_constant(x)
        model = OLS(y, X).fit()
        xs_sorted = np.sort(x)
        X_sorted = add_constant(xs_sorted)
        pred_sorted = model.get_prediction(X_sorted)
        ci = pred_sorted.conf_int()
        ax.fill_between(xs_sorted, ci[:, 0], ci[:, 1], color="black", alpha=0.12)
        ax.plot(xs_sorted, pred_sorted.predicted_mean, color="black", lw=1.2)
        if len(model.pvalues) > 1:
            p_ols = model.pvalues[1]
            r, p_r = stats.pearsonr(x, y)
            def _sig(p): return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            ax.text(*text_pos,
                    f"$R^2={model.rsquared:.2f}$, $P={p_ols:.3f}$ {_sig(p_ols)}\n"
                    f"$r={r:.2f}$, $N={len(x)}$",
                    transform=ax.transAxes, va="bottom", ha="left",
                    style="italic", fontsize=6, clip_on=False,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5})
            stats_out = {
                "x_var": xlabel, "y_var": ylabel,
                "slope": float(model.params[1]),
                "intercept": float(model.params[0]),
                "r2": float(model.rsquared),
                "r": float(r),
                "p": float(p_ols),
                "p_r": float(p_r),
                "n": int(len(x)),
            }
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    sns.despine(ax=ax)
    return stats_out


# ── panels ────────────────────────────────────────────────────────────────────

def plot_size_vs_food(df_agent, out_dir):
    set_style()
    fig, ax = panel()
    st = scatter_regression(ax, df_agent["agent_size"].values,
                            df_agent["food_eaten"].values,
                            SEMANTIC_COLORS["food"],
                            "Agent size",
                            "Food eaten per agent-episode")
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "size_vs_food.pdf"))
    return st


def plot_size_vs_eod_rate(df_agent, out_dir):
    set_style()
    fig, ax = panel()
    st = scatter_regression(ax, df_agent["agent_size"].values,
                            df_agent["mean_eod_rate_hz"].values,
                            SEMANTIC_COLORS["eod"],
                            "Agent size",
                            "Mean EOD rate per episode (Hz)")
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "size_vs_eod_rate.pdf"))
    return st


def plot_size_vs_displacement(df_agent, out_dir):
    set_style()
    fig, ax = panel()
    st = scatter_regression(ax, df_agent["agent_size"].values,
                            df_agent["mean_displacement"].values,
                            AGENT_COLORS[2],
                            "Agent size",
                            "Mean displacement per step (cm)")
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "size_vs_displacement.pdf"))
    return st


def plot_size_vs_pbite(df_agent, out_dir):
    set_style()
    fig, ax = panel()
    st = scatter_regression(ax, df_agent["agent_size"].values,
                            df_agent["num_biting_events"].values,
                            AGENT_COLORS[3],
                            "Agent size",
                            "Biting events per episode")
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "size_vs_biting.pdf"))
    return st


def plot_food_rank_vs_size_rank(df_agent, out_dir):
    """Spearman ρ of size rank vs food rank, one point per episode."""
    required = {"env_id", "episode_index", "agent_size", "food_eaten"}
    if not required.issubset(df_agent.columns):
        missing = sorted(required - set(df_agent.columns))
        print(f"  food_rank_vs_size_rank: missing columns {missing} — skipped")
        return None

    rhos = []
    for _, grp in df_agent.groupby(["env_id", "episode_index"]):
        if len(grp) < 3:
            continue
        r, _ = stats.spearmanr(grp["agent_size"], grp["food_eaten"])
        if np.isfinite(r):
            rhos.append(r)

    if not rhos:
        print("  food_rank_vs_size_rank: too few episodes — skipped")
        return None

    rhos = np.array(rhos)
    median_rho = float(np.median(rhos))
    set_style()
    fig, ax = panel()
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.hist(rhos, bins=20, color=AGENT_COLORS[1], edgecolor="white", linewidth=0.4)
    # ax.axvline(median_rho, color="black", lw=1.2, label=f"median \nρ = {median_rho:.2f}\nN = {len(rhos)} episodes")
    ax.axvline(median_rho, color="black", lw=1.2, label=f"median ρ = {median_rho:.2f}")
    ax.set_xlabel("Spearman ρ\n(size rank vs food rank)")
    ax.set_ylabel("Episode count")
    ax.text(0.95, 0.93, f"N={len(rhos)} episodes", transform=ax.transAxes,
            ha="right", fontsize=6,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5})
    ax.legend(frameon=False, fontsize=6)
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "food_rank_vs_size_rank.pdf"))
    pd.DataFrame([{
        "test": "spearmanr_per_episode",
        "x_var": "agent_size", "y_var": "food_eaten",
        "n_episodes": len(rhos),
        "median_rho": median_rho,
        "mean_rho": float(np.mean(rhos)),
        "rhos_all": rhos.tolist(),
    }]).to_csv(os.path.join(out_dir, "food_rank_vs_size_rank_spearman.csv"), index=False)
    return None


def _theil_index(values):
    values = _finite_values(values)
    total = values.sum()
    if total == 0 or len(values) == 0:
        return np.nan
    n  = len(values)
    mu = total / n
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(values > 0, (values / mu) * np.log(values / mu), 0.0)
    return t.sum() / n


def _finite_values(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _gini_index(values):
    values = _finite_values(values)
    if len(values) == 0:
        return np.nan
    total = values.sum()
    if total == 0:
        return 0.0
    sorted_values = np.sort(values)
    n = len(sorted_values)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_values)) / (n * total) - (n + 1) / n)


def _std(values):
    values = _finite_values(values)
    if len(values) < 2:
        return np.nan
    return float(np.std(values, ddof=1))


def _coefficient_of_variation(values):
    values = _finite_values(values)
    if len(values) < 2:
        return np.nan
    mean = values.mean()
    if mean == 0:
        return np.nan
    return float(np.std(values, ddof=1) / mean)


def _range(values):
    values = _finite_values(values)
    if len(values) == 0:
        return np.nan
    return float(values.max() - values.min())


def _iqr(values):
    values = _finite_values(values)
    if len(values) == 0:
        return np.nan
    return float(np.percentile(values, 75) - np.percentile(values, 25))


def _mean_abs_deviation(values):
    values = _finite_values(values)
    if len(values) == 0:
        return np.nan
    return float(np.mean(np.abs(values - values.mean())))


INEQUALITY_METRICS = {
    "theil": ("Theil index", _theil_index),
    "gini": ("Gini index", _gini_index),
    "std": ("Std", _std),
    "cv": ("Coefficient of variation", _coefficient_of_variation),
    "range": ("Range", _range),
    "iqr": ("IQR", _iqr),
    "mad": ("Mean absolute deviation", _mean_abs_deviation),
}


def _episode_food_size_ineq(df_agent):
    records = []
    for (env_id, ep_idx), grp in df_agent.groupby(["env_id", "episode_index"]):
        row = {"env_id": env_id, "episode_index": ep_idx}
        for metric, (_, fn) in INEQUALITY_METRICS.items():
            row[f"food_{metric}"] = fn(grp["food_eaten"].values)
            row[f"size_{metric}"] = fn(grp["agent_size"].values)
        records.append(row)
    return pd.DataFrame(records)


def plot_food_ineq_vs_size_ineq(df_agent, out_dir):
    set_style()
    ep_df = _episode_food_size_ineq(df_agent)
    ep_df.to_csv(os.path.join(out_dir, "food_ineq_vs_size_ineq_metrics.csv"), index=False)

    stats_rows = []
    for metric, (label, _) in INEQUALITY_METRICS.items():
        x_col = f"size_{metric}"
        y_col = f"food_{metric}"
        plot_df = ep_df.dropna(subset=[x_col, y_col])
        if len(plot_df) < 4:
            print(f"  {y_col}_vs_{x_col}: too few data points — skipped")
            continue

        fig, ax = panel()
        st = scatter_regression(
            ax,
            plot_df[x_col].values,
            plot_df[y_col].values,
            SEMANTIC_COLORS["food"],
            f"Agent-size {label} per episode",
            f"Food {label} per episode",
        )
        plt.tight_layout(pad=0.3)
        save(
            fig,
            os.path.join(out_dir, f"food_ineq_vs_size_ineq_{metric}.pdf"),
        )
        if st is not None:
            st.update({
                "x_var": x_col,
                "y_var": y_col,
                "size_metric": metric,
                "food_metric": metric,
            })
            stats_rows.append(st)
    return stats_rows


def plot_food_theil_vs_size_advantage(df_agent, out_dir):
    """
    Per-episode scatter: food Theil index vs mean |size_advantage| (absolute size
    difference between the two fish).  Tests whether larger size disparities
    produce more unequal foraging outcomes at the group level.
    """
    if "size_advantage" not in df_agent.columns:
        return None
    set_style()
    records = []
    for (env_id, ep_idx), grp in df_agent.groupby(["env_id", "episode_index"]):
        food_theil = _theil_index(grp["food_eaten"].values)
        mean_abs_adv = grp["size_advantage"].abs().mean()
        records.append({"food_theil": food_theil, "mean_abs_size_adv": mean_abs_adv})
    ep_df = pd.DataFrame(records).dropna()
    if len(ep_df) < 5:
        return None

    fig, ax = panel()
    st = scatter_regression(ax, ep_df["mean_abs_size_adv"].values,
                            ep_df["food_theil"].values,
                            AGENT_COLORS[0],
                            "Mean |size advantage| per episode",
                            "Food Theil index per episode")
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "food_theil_vs_size_advantage.pdf"))
    ep_df.to_csv(os.path.join(out_dir, "food_theil_vs_size_advantage.csv"), index=False)
    return st


def plot_exploratory_correlations(df_agent, out_dir):
    set_style()
    cols = [
        "agent_size", "food_eaten", "mean_eod_rate_hz", "p_nearby",
        "mean_displacement", "num_interactions", "num_biting_events",
        "distance_to_nearest_agent",
    ]
    available = [c for c in cols if c in df_agent.columns]
    corr = df_agent[available].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 5},
                linewidths=0.3, square=True,
                xticklabels=[c.replace("_", "\n") for c in available],
                yticklabels=[c.replace("_", "\n") for c in available])
    ax.tick_params(axis="both", labelsize=6)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "exploratory_correlations.pdf"))


def plot_eod_rate_vs_food_eaten(df_agent, out_dir):
    set_style()
    fig, ax = panel()
    st = scatter_regression(ax, df_agent["mean_eod_rate_hz"].values,
                            df_agent["food_eaten"].values,
                            "#888888",
                            "Mean EOD rate per agent-episode (Hz)",
                            "Food eaten per agent-episode")
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "eod_rate_vs_food.pdf"))
    return st


# ── main ─────────────────────────────────────────────────────────────────────

def load(spec_dir):
    agent_pkl = os.path.join(spec_dir, "derived", "per_env_ep_agent.pkl")
    if not os.path.exists(agent_pkl):
        sys.exit(f"Not found: {agent_pkl}")
    out_dir = os.path.join(spec_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Loading {agent_pkl} \u2026")
    df_agent = pd.read_pickle(agent_pkl)
    df_agent = _add_eod_rate_hz(df_agent)
    print(f"  {len(df_agent):,} agent-episodes")
    times = load_consumption_times(spec_dir)
    return {"df_agent": df_agent, "out_dir": out_dir, "times": times}


def plot_time_to_consumption_single(times, out_dir):
    if not times:
        print("No consumption time data for this spec.")
        return
    palette = {"All agents": COND_PALETTE[0]}
    time_to_X_log(
        [("All agents", times)], palette,
        os.path.join(out_dir, "time_to_consumption_log.pdf"),
        event_label="Food consumed", time_label="Time to consumption (s)",
    )
    time_to_X_linear(
        [("All agents", times)], palette,
        os.path.join(out_dir, "time_to_consumption_linear.pdf"),
        event_label="Food consumed", time_label="Time to consumption (s)",
    )


def run(spec_dir):
    agent_pkl = os.path.join(spec_dir, "derived", "per_env_ep_agent.pkl")
    if not os.path.exists(agent_pkl):
        sys.exit(f"Not found: {agent_pkl}")

    out_dir = os.path.join(spec_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {agent_pkl} …")
    df_agent = pd.read_pickle(agent_pkl)
    print(f"  {len(df_agent):,} agent-episodes")

    df_agent = _add_eod_rate_hz(df_agent)
    df_agent = add_size_advantage(df_agent)

    # time-to-consumption (single-condition)
    times = load_consumption_times(spec_dir)
    if times:
        palette = {"All agents": COND_PALETTE[0]}
        time_to_X_log(
            [("All agents", times)], palette,
            os.path.join(out_dir, "time_to_consumption.pdf"),
        )
        time_to_X_linear(
            [("All agents", times)], palette,
            os.path.join(out_dir, "time_to_consumption_linear.pdf"),
        )

    ols_rows = []
    for fn in (plot_size_vs_food, plot_size_vs_eod_rate, plot_size_vs_displacement,
               plot_size_vs_pbite, plot_food_rank_vs_size_rank, plot_food_ineq_vs_size_ineq,
               plot_food_theil_vs_size_advantage, plot_eod_rate_vs_food_eaten):
        st = fn(df_agent, out_dir)
        if isinstance(st, list):
            ols_rows.extend(st)
        elif st is not None:
            ols_rows.append(st)
    plot_exploratory_correlations(df_agent, out_dir)
    if ols_rows:
        pd.DataFrame(ols_rows).to_csv(
            os.path.join(out_dir, "ols_regression_summary.csv"), index=False)
    print("Done.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec_dir", required=True)
    args = ap.parse_args(argv)
    run(args.spec_dir)


if __name__ == "__main__":
    main()
