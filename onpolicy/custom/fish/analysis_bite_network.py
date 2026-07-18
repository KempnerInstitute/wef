"""
Bite network: directed bite graph and size-based dominance hierarchy.

Merges the former biter-centric (`analysis_biting_network`) and victim-centric
(`analysis_bitten_network`) scripts into one module. A single `run()` builds the
bite-event table and per-agent stats once, then emits both plot sets at once.

Plots → analyses/biting_network/   (biter-centric)
----------------------------------
biting_heatmap.pdf          biter-size × bitten-size heatmap (% of bites)
win_ratio_vs_size.pdf       per-agent win-ratio vs size (scatter + regression)

Plots → analyses/bitten_network/   (victim-centric)
----------------------------------
bitten_heatmap.pdf          victim-size × biter-size heatmap (% of bites)
loss_ratio_vs_size.pdf      per-agent loss-ratio vs size (scatter + regression)
bitten_vs_food.pdf          bites received vs food eaten per agent-episode
bitten_rank_stability.pdf   Spearman ρ of size rank vs not-bitten rank, per episode

Data
----
derived/per_env_ep_agent_step.pkl  bite events (was_bitten_by_agent_id)
derived/per_env_ep_agent.pkl       food_eaten per agent-episode (bitten plots only)
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress, spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from analysis_style import set_style, panel, save, size_pct_heatmap, AGENT_COLORS, SEMANTIC_COLORS

BITING_OUT_SUBDIR = "analyses/biting_network"
BITTEN_OUT_SUBDIR = "analyses/bitten_network"
N_SIZE_BINS = 4


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _size_bins(sizes, n=N_SIZE_BINS):
    """Return pd.Categorical labels using quantile-based bins."""
    edges = np.quantile(sizes.dropna(), np.linspace(0, 1, n + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        return pd.cut(sizes, bins=2, labels=False)
    labels = [f"{e:.2f}" for e in edges[:-1]]
    return pd.cut(sizes, bins=edges, labels=labels, include_lowest=True)


def _build_bite_events(df):
    """
    Return a DataFrame with one row per bite event:
      env_id, episode_index, time_step, biter_id, victim_id,
      biter_size, victim_size
    """
    victims = df.loc[df["was_bitten"].astype(bool)].copy()
    if victims.empty:
        return pd.DataFrame()

    victims = victims.rename(columns={
        "agent_id": "victim_id",
        "agent_size": "victim_size",
        "was_bitten_by_agent_id": "biter_id",
    })[["env_id", "episode_index", "time_step", "victim_id", "victim_size", "biter_id"]]
    victims = victims.dropna(subset=["biter_id"])
    victims["biter_id"] = victims["biter_id"].astype(int)

    # look up biter size (constant per agent per episode)
    size_lookup = (
        df.groupby(["env_id", "episode_index", "agent_id"])["agent_size"]
        .first()
        .rename("biter_size")
    )
    victims = victims.join(
        size_lookup,
        on=["env_id", "episode_index", "biter_id"],
    )
    return victims


def _build_agent_bite_stats(df):
    """
    Per (env_id, episode_index, agent_id): wins, losses, agent_size.
    """
    wins = (
        df.loc[df["bite_other_fish"].astype(bool)]
        .groupby(["env_id", "episode_index", "agent_id"])
        .size()
        .rename("wins")
    )
    losses = (
        df.loc[df["was_bitten"].astype(bool)]
        .groupby(["env_id", "episode_index", "agent_id"])
        .size()
        .rename("losses")
    )
    sizes = df.groupby(["env_id", "episode_index", "agent_id"])["agent_size"].first()

    stats = pd.concat([sizes, wins, losses], axis=1).fillna(0)
    stats["total"] = stats["wins"] + stats["losses"]
    stats["win_ratio"] = np.where(
        stats["total"] > 0,
        stats["wins"] / stats["total"],
        np.nan,
    )
    return stats.reset_index()


# ---------------------------------------------------------------------------
# biter-centric plots  →  analyses/biting_network/
# ---------------------------------------------------------------------------

def plot_biting_heatmap(bite_events, out_dir):
    if bite_events.empty or bite_events["biter_size"].isna().all():
        print("  biting_heatmap: no bite events — skipped")
        return
    counts = size_pct_heatmap(
        bite_events["biter_size"], bite_events["victim_size"],
        os.path.join(out_dir, "biting_heatmap.pdf"),
        cmap="Blues", xlabel="Bitten size (bin)", ylabel="Biter size (bin)",
        title="Biting %", n_bins=N_SIZE_BINS,
    )
    if counts is None:
        print("  biting_heatmap: insufficient size variance — skipped")


def plot_win_ratio_vs_size(stats, out_dir):
    d = stats.dropna(subset=["win_ratio", "agent_size"])
    if len(d) < 4:
        print("  win_ratio_vs_size: too few data points — skipped")
        return

    from statsmodels.api import OLS, add_constant
    set_style()
    fig, ax = panel()
    x = d["agent_size"].values
    y = d["win_ratio"].values
    ax.scatter(x, y, facecolors="none", edgecolors=SEMANTIC_COLORS["dominance"], s=18, linewidths=0.8, alpha=0.7)

    X = add_constant(x)
    model = OLS(y, X).fit()
    xs = np.linspace(x.min(), x.max(), 100)
    pred = model.get_prediction(add_constant(xs))
    ci = pred.conf_int()
    ax.fill_between(xs, ci[:, 0], ci[:, 1], color="black", alpha=0.12)
    ax.plot(xs, pred.predicted_mean, color="black", lw=1.2)

    slope, intercept, r, p_r, se = linregress(x, y)
    p_ols = model.pvalues[1]
    def _sig(p): return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    ax.text(
        0.05, 1.02,
        f"$R^2={model.rsquared:.2f}$, $P={p_ols:.3f}$ {_sig(p_ols)}\n"
        f"$r={r:.2f}$, $N={len(x)}$",
        transform=ax.transAxes, va="bottom", ha="left", fontsize=6, style="italic",
        clip_on=False,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5},
    )
    ax.set_xlabel("Agent size")
    ax.set_ylabel("Win ratio (bites given / total)")
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "win_ratio_vs_size.pdf"))
    pd.DataFrame([{
        "x_var": "agent_size", "y_var": "win_ratio",
        "slope": slope, "intercept": intercept,
        "r": r, "r2": model.rsquared, "p": p_r, "p_ols": p_ols, "stderr": se, "n": len(x),
    }]).to_csv(os.path.join(out_dir, "size_vs_winratio_regression.csv"), index=False)


# ---------------------------------------------------------------------------
# victim-centric plots  →  analyses/bitten_network/
# ---------------------------------------------------------------------------

def plot_bitten_heatmap(bite_events, out_dir):
    """Heatmap from victim perspective: rows=victim size bin, cols=biter size bin."""
    if bite_events.empty or bite_events["biter_size"].isna().all():
        print("  bitten_heatmap: no bite events — skipped")
        return
    counts = size_pct_heatmap(
        bite_events["victim_size"], bite_events["biter_size"],
        os.path.join(out_dir, "bitten_heatmap.pdf"),
        cmap="Reds", xlabel="Biter size (cm bin)", ylabel="Victim size (cm bin)",
        title="Biting % (victim view)", n_bins=N_SIZE_BINS,
    )
    if counts is None:
        print("  bitten_heatmap: insufficient size variance — skipped")


def plot_loss_ratio_vs_size(stats, out_dir):
    """Scatter + regression: agent_size vs loss_ratio (times bitten / total)."""
    d = stats.copy()
    d["loss_ratio"] = np.where(d["total"] > 0, d["losses"] / d["total"], np.nan)
    d = d.dropna(subset=["loss_ratio", "agent_size"])
    if len(d) < 4:
        print("  loss_ratio_vs_size: too few data points — skipped")
        return

    from statsmodels.api import OLS, add_constant
    x = d["agent_size"].values
    y = d["loss_ratio"].values
    model = OLS(y, add_constant(x)).fit()
    xs = np.linspace(x.min(), x.max(), 100)
    pred = model.get_prediction(add_constant(xs))
    ci = pred.conf_int()
    p_ols = model.pvalues[1]
    slope, intercept, r, p_r, se = linregress(x, y)
    def _sig(p): return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    set_style()
    fig, ax = panel()
    ax.scatter(x, y, facecolors="none", edgecolors=AGENT_COLORS[2], s=18, linewidths=0.8, alpha=0.7)
    ax.fill_between(xs, ci[:, 0], ci[:, 1], color="black", alpha=0.12)
    ax.plot(xs, pred.predicted_mean, color="black", lw=1.2)
    ax.text(0.05, 0.95,
            f"$R^2={model.rsquared:.2f}$, $P={p_ols:.3f}$ {_sig(p_ols)}\n"
            f"$r={r:.2f}$, $N={len(x)}$",
            transform=ax.transAxes, va="top", ha="left", fontsize=6, style="italic",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5})
    ax.set_xlabel("Agent size")
    ax.set_ylabel("Loss ratio (times bitten / total)")
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "loss_ratio_vs_size.pdf"))
    pd.DataFrame([{
        "x_var": "agent_size", "y_var": "loss_ratio",
        "slope": slope, "intercept": intercept,
        "r": r, "r2": model.rsquared, "p": p_r, "p_ols": p_ols, "stderr": se, "n": len(x),
    }]).to_csv(os.path.join(out_dir, "loss_ratio_vs_size_regression.csv"), index=False)


def plot_bitten_vs_food(stats, agent_df, out_dir):
    """Scatter: bites received vs food eaten per agent-episode (does biting suppress foraging?)."""
    key = ["env_id", "episode_index", "agent_id"]
    merged = stats[key + ["losses", "agent_size"]].merge(
        agent_df[key + ["food_eaten"]], on=key, how="inner"
    )
    d = merged[merged["losses"] > 0].copy()
    if len(d) < 4:
        print("  bitten_vs_food: too few data points — skipped")
        return

    from statsmodels.api import OLS, add_constant
    x = d["losses"].values
    y = d["food_eaten"].values
    model = OLS(y, add_constant(x)).fit()
    xs = np.linspace(x.min(), x.max(), 100)
    pred = model.get_prediction(add_constant(xs))
    ci = pred.conf_int()
    p_ols = model.pvalues[1]
    slope, intercept, r, p_r, se = linregress(x, y)
    def _sig(p): return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    set_style()
    fig, ax = panel()
    ax.scatter(x, y, facecolors="none", edgecolors=AGENT_COLORS[3], s=18, linewidths=0.8, alpha=0.7)
    ax.fill_between(xs, ci[:, 0], ci[:, 1], color="black", alpha=0.12)
    ax.plot(xs, pred.predicted_mean, color="black", lw=1.2)
    ax.text(0.05, 0.95,
            f"$R^2={model.rsquared:.2f}$, $P={p_ols:.3f}$ {_sig(p_ols)}\n"
            f"$r={r:.2f}$, $N={len(x)}$",
            transform=ax.transAxes, va="top", ha="left", fontsize=6, style="italic",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5})
    ax.set_xlabel("Bites received per episode")
    ax.set_ylabel("Food eaten per episode")
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "bitten_vs_food.pdf"))
    pd.DataFrame([{
        "x_var": "losses", "y_var": "food_eaten",
        "slope": slope, "intercept": intercept,
        "r": r, "r2": model.rsquared, "p": p_r, "p_ols": p_ols, "stderr": se, "n": len(x),
    }]).to_csv(os.path.join(out_dir, "bitten_vs_food_regression.csv"), index=False)


def plot_bitten_rank_stability(stats, agent_df, out_dir):
    """Histogram of Spearman ρ (size rank vs not-bitten rank) per episode."""
    key = ["env_id", "episode_index", "agent_id"]
    merged = stats[key + ["agent_size", "losses"]].merge(
        agent_df[key + ["food_eaten"]], on=key, how="inner"
    )
    if merged.empty:
        print("  bitten_rank_stability: no merged data — skipped")
        return

    rhos = []
    for (eid, ep), grp in merged.groupby(["env_id", "episode_index"]):
        if len(grp) < 3:
            continue
        if grp["losses"].nunique() < 2:
            continue
        r, _ = spearmanr(grp["agent_size"], -grp["losses"])
        rhos.append(r)

    if not rhos:
        print("  bitten_rank_stability: too few episodes — skipped")
        return

    rhos = np.array(rhos)
    set_style()
    fig, ax = panel()
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    median_rho = float(np.median(rhos))
    ax.hist(rhos, bins=20, color=AGENT_COLORS[2], edgecolor="white", linewidth=0.4)
    ax.axvline(median_rho, color="black", lw=1.2, label=f"median ρ = {median_rho:.2f}")
    ax.set_xlabel("Spearman ρ  (size rank vs not-bitten rank)")
    ax.set_ylabel("Episode count")
    ax.text(0.95, 0.93, f"N={len(rhos)} episodes", transform=ax.transAxes,
            ha="right", fontsize=6,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5})
    ax.legend(frameon=False, fontsize=6)
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "bitten_rank_stability.pdf"))
    pd.DataFrame([{
        "test": "spearmanr_per_episode",
        "x_var": "agent_size", "y_var": "neg_losses",
        "n_episodes": len(rhos),
        "median_rho": float(np.median(rhos)),
        "mean_rho": float(np.mean(rhos)),
        "rhos_all": rhos.tolist(),
    }]).to_csv(os.path.join(out_dir, "bitten_rank_stability_spearman.csv"), index=False)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def load(spec_dir):
    step_pkl  = os.path.join(spec_dir, "derived", "per_env_ep_agent_step.pkl")
    agent_pkl = os.path.join(spec_dir, "derived", "per_env_ep_agent.pkl")
    biting_dir = os.path.join(spec_dir, BITING_OUT_SUBDIR)
    bitten_dir = os.path.join(spec_dir, BITTEN_OUT_SUBDIR)
    os.makedirs(biting_dir, exist_ok=True)
    os.makedirs(bitten_dir, exist_ok=True)
    for path in (step_pkl, agent_pkl):
        if not os.path.exists(path):
            sys.exit(f"Not found: {path}")
    print(f"Loading {step_pkl} …")
    df = pd.read_pickle(step_pkl)
    print(f"Loading {agent_pkl} …")
    agent_df = pd.read_pickle(agent_pkl)
    bite_events = _build_bite_events(df)
    print(f"  {len(bite_events):,} bite events")
    stats = _build_agent_bite_stats(df)
    return {
        "bite_events": bite_events, "stats": stats, "agent_df": agent_df,
        "biting_dir": biting_dir, "bitten_dir": bitten_dir,
    }


def run(spec_dir, force_recompute=False):
    step_pkl  = os.path.join(spec_dir, "derived", "per_env_ep_agent_step.pkl")
    agent_pkl = os.path.join(spec_dir, "derived", "per_env_ep_agent.pkl")
    biting_dir = os.path.join(spec_dir, BITING_OUT_SUBDIR)
    bitten_dir = os.path.join(spec_dir, BITTEN_OUT_SUBDIR)
    os.makedirs(biting_dir, exist_ok=True)
    os.makedirs(bitten_dir, exist_ok=True)

    bite_cache = os.path.join(biting_dir, "_bite_events_cache.pkl")
    stats_cache = os.path.join(biting_dir, "_stats_cache.pkl")

    cache_hit = (not force_recompute and
                 os.path.exists(bite_cache) and os.path.exists(stats_cache))
    if cache_hit:
        print(f"[bite_network] loading cached bite events and stats …")
        bite_events = pd.read_pickle(bite_cache)
        stats       = pd.read_pickle(stats_cache)
        agent_df    = pd.read_pickle(agent_pkl) if os.path.exists(agent_pkl) else None
    else:
        if not os.path.exists(step_pkl):
            sys.exit(f"Not found: {step_pkl}")

        print(f"Loading {step_pkl} …")
        df = pd.read_pickle(step_pkl)
        print(f"  {len(df):,} agent-steps")

        agent_df = None
        if os.path.exists(agent_pkl):
            print(f"Loading {agent_pkl} …")
            agent_df = pd.read_pickle(agent_pkl)
            print(f"  {len(agent_df):,} agent-episodes")
        else:
            print(f"  [WARN] {agent_pkl} not found — bitten food-coupling plots skipped")

        bite_events = _build_bite_events(df)
        print(f"  {len(bite_events):,} bite events")
        stats = _build_agent_bite_stats(df)
        bite_events.to_pickle(bite_cache)
        stats.to_pickle(stats_cache)

    # --- biter-centric ---
    plot_biting_heatmap(bite_events, biting_dir)
    plot_win_ratio_vs_size(stats, biting_dir)

    # --- victim-centric ---
    plot_bitten_heatmap(bite_events, bitten_dir)
    plot_loss_ratio_vs_size(stats, bitten_dir)
    if agent_df is not None:
        plot_bitten_vs_food(stats, agent_df, bitten_dir)
        plot_bitten_rank_stability(stats, agent_df, bitten_dir)
    print("Done.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec_dir", required=True)
    ap.add_argument("--force_recompute", action="store_true")
    args = ap.parse_args(argv)
    run(args.spec_dir, force_recompute=args.force_recompute)


if __name__ == "__main__":
    main()
