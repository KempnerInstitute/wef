"""
Per-episode behavior sample plots.

Usage:
    python analysis_behavior.py --spec_dir path/to/evals/m1a1k1_patchy_square --n_samples 8

Outputs to {spec_dir}/analyses/behavior/
  ethogram_env{env}_ep{ep}.pdf         — 5-10 per-episode ethogram rasters
  trajectory_env{env}_ep{ep}.pdf       — 5-10 per-episode trajectories
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(__file__))
from analysis_style import (
    set_style,
    AGENT_COLORS,
)
from cfg import SIM_FPS

OUT_SUBDIR = "analyses/behavior"
N_SAMPLES = 8   # default number of sample episodes to produce

# Ethogram state priority (highest wins) and palette — matches utils_behavior.py
_PRIORITY = ["was_bitten", "bite_other_fish", "eating_event", "food_observed", "has_nearby"]
_ETHOGRAM_COLORS = {
    "was_bitten":     "#E53935",  # red
    "bite_other_fish":"#7B3294",  # purple
    "eating_event":   "#4CAF50",  # green
    "food_observed":  "#C8E6C9",  # very light green
    "has_nearby":     "#FFF59D",  # yellow (was food_observed color)
    "none":           "#EEEEEE",  # light gray
}
_ETHOGRAM_NAMES = {
    "was_bitten":     "Bitten",
    "bite_other_fish":"Biting",
    "eating_event":   "Eating",
    "food_observed":  "Food in range",
    "has_nearby":     "Agent in range",
}
_EMA_WINDOW = 10


# ── episode selection ─────────────────────────────────────────────────────────

def _select_sample_episodes(df, n=N_SAMPLES, rank_by="eating+biting"):
    """
    Return up to n (env_id, episode_index) pairs.

    rank_by:
      "eating"         — top-n by eating events only (original behaviour)
      "biting"         — top-n by biting events only
      "eating+biting"  — top-n by sum of eating + biting events (default)
    """
    g = df.groupby(["env_id", "episode_index"])
    if rank_by == "eating":
        score = g["eating_event"].sum()
    elif rank_by == "biting":
        score = g["bite_other_fish"].sum()
    else:  # "eating+biting"
        score = g["eating_event"].sum() + g["bite_other_fish"].sum()
    return list(score.sort_values(ascending=False).head(n).index)


# ── per-episode ethogram raster ───────────────────────────────────────────────

def _build_ethogram_raster(rollout_df, agents):
    """
    Build the (n_agents, T) integer raster and EMA EOD arrays.
    Priority: was_bitten > bite_other_fish > eating_event > food_observed > has_nearby > none
    """
    time_steps = np.arange(rollout_df["time_step"].min(),
                           rollout_df["time_step"].max() + 1)
    color_to_idx = {k: i + 1 for i, k in enumerate(_PRIORITY)}
    ethogram   = np.zeros((len(agents), len(time_steps)), dtype=int)
    ema_arrays = []

    for ai, agent_id in enumerate(agents):
        agent_df = (rollout_df[rollout_df["agent_id"] == agent_id]
                    .sort_values("time_step")
                    .set_index("time_step"))
        emit_eod_series = []

        for ti, ts in enumerate(time_steps):
            val = 0
            if ts in agent_df.index:
                row = agent_df.loc[ts]
                for s in _PRIORITY:
                    if bool(row[s]):
                        val = color_to_idx[s]
                        break
                emit_eod_series.append(float(row["emit_eod"]))
            else:
                emit_eod_series.append(np.nan)
            ethogram[ai, ti] = val

        ema = (pd.Series(emit_eod_series)
               .ffill().fillna(0)
               .ewm(span=_EMA_WINDOW, adjust=False)
               .mean()
               .values)
        ema_arrays.append(ema)

    return ethogram, ema_arrays, time_steps


def _save_ethogram_raster(ethogram, ema_arrays, time_steps, agents, out_path):
    """Render and save one ethogram raster figure."""
    set_style()
    num_agents  = len(agents)
    color_to_idx = {k: i + 1 for i, k in enumerate(_PRIORITY)}
    idx_to_color = {0: _ETHOGRAM_COLORS["none"]}
    for k, v in color_to_idx.items():
        idx_to_color[v] = _ETHOGRAM_COLORS[k]

    cmap  = mcolors.ListedColormap([idx_to_color[i] for i in range(len(idx_to_color))])
    blues = plt.cm.Blues
    norm  = plt.Normalize(0, 1)
    time_seconds = time_steps / SIM_FPS
    dt_seconds = 1 / SIM_FPS

    # fig_h = max(1.5, 0.5 * num_agents + 0.8)
    # fig, ax = plt.subplots(figsize=(7, fig_h))
    fig_h = 2.0
    fig, ax = plt.subplots(figsize=(4.0, fig_h))

    ax.imshow(ethogram, aspect="auto", cmap=cmap,
              interpolation="nearest",
              extent=[time_seconds[0], time_seconds[-1] + dt_seconds,
                      -0.5, num_agents - 0.5])

    # EMA EOD overlay — one imshow per agent instead of per-timestep patches
    for ai, ema_row in enumerate(ema_arrays):
        rgba = blues(norm(ema_row[np.newaxis, :]))  # (1, T, 4)
        ax.imshow(rgba, aspect="auto", interpolation="nearest",
                  extent=[time_seconds[0], time_seconds[-1] + dt_seconds,
                          ai + 0.4, ai + 0.5])

    ax.set_yticks(range(num_agents))
    ax.set_yticklabels([f"A{a+1}" for a in agents], fontsize=6)
    ax.set_xlabel("Time [s]")
    ax.set_xlim(time_seconds[0], time_seconds[-1] + dt_seconds)
    # Lock y limits after all imshow calls so each agent row is exactly 1 unit tall.
    # set_ylim(big, small) produces an inverted axis (A0 at top) without relying on
    # invert_yaxis(), which could be toggled incorrectly by the EMA overlay imshows.
    ax.set_ylim(num_agents - 0.5, -0.5)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:g}"))
    ax.tick_params(axis="x", labelsize=6, length=2)
    for i in range(num_agents):
        ax.axhline(i + 0.5, color="white", linewidth=0.5)

    # Legend — use a gradient patch proxy for the EOD EMA entry
    patches = [mpatches.Patch(color=_ETHOGRAM_COLORS[k], label=_ETHOGRAM_NAMES[k])
               for k in _PRIORITY]
    eod_proxy = mpatches.Patch(color=blues(0.6), label="EOD")
    ax.legend(handles=patches + [eod_proxy],
              loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=len(patches) + 1, fontsize=8, frameon=False,
              columnspacing=0.8, handlelength=1.0)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.28)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    for ext in (".pdf",):
        base = out_path if not out_path.endswith(ext) else out_path[:-len(ext)]
        fig.savefig(f"{base}{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


def plot_ethogram_samples(df, out_dir, episodes):
    """Produce one raster per sample episode. The top-ranked episode (index 0)
    is also saved as ethogram_sample.pdf for deterministic use by copy_to_figs."""
    agents = sorted(df["agent_id"].unique())
    for i, (env_id, ep_idx) in enumerate(episodes):
        rollout_df = df[(df["env_id"] == env_id) & (df["episode_index"] == ep_idx)]
        if len(rollout_df) == 0:
            continue
        ethogram, ema_arrays, time_steps = _build_ethogram_raster(rollout_df, agents)
        out_path = os.path.join(out_dir, f"ethogram_env{env_id}_ep{ep_idx}.pdf")
        _save_ethogram_raster(ethogram, ema_arrays, time_steps, agents, out_path)
        if i == 0:
            canonical = os.path.join(out_dir, "ethogram_sample.pdf")
            import shutil; shutil.copy2(out_path, canonical)
            print(f"  saved → {canonical} (top-ranked)")


# ── per-episode trajectory ───────────────────────────────────────────────────

def _save_trajectory(rollout_df, env_id, ep_idx, out_path):
    """
    Plot all agents' trajectories.
    Eating events shown as open circles.
    """
    set_style()
    agents     = sorted(rollout_df["agent_id"].unique())
    arena_size = rollout_df["arena_size"].iloc[0]
    w, h       = arena_size[0], arena_size[1]

    fig, ax = plt.subplots(figsize=(2.0, 2.0))

    from matplotlib.lines import Line2D
    for i, agent_id in enumerate(agents):
        traj = (rollout_df[rollout_df["agent_id"] == agent_id]
                .sort_values("time_step"))
        xs   = traj["position_x"].values
        ys   = traj["position_y"].values

        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        ax.plot(xs, ys, color=color, lw=1.0)

        eat = traj[traj["eating_event"].astype(bool)]
        if len(eat):
            ax.scatter(eat["position_x"], eat["position_y"],
                       marker="o", facecolors="none",
                       edgecolors="black", s=40, zorder=5, linewidths=0.9)

        bit = traj[traj["was_bitten"].astype(bool)] if "was_bitten" in traj.columns else pd.DataFrame()
        if len(bit):
            ax.scatter(bit["position_x"], bit["position_y"],
                       marker="x", c="red", s=40, zorder=5, linewidths=0.9)

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    legend_handles = (
        [Line2D([0], [0], color=AGENT_COLORS[i % len(AGENT_COLORS)],
                lw=1.2, label=f"A{aid + 1}")
         for i, aid in enumerate(agents[:4])]
        + [plt.scatter([], [], marker="o", facecolors="none",
                       edgecolors="black", s=40, lw=0.9, label="Eating")]
        + [plt.scatter([], [], marker="x", c="red", s=40, lw=0.9, label="Bitten")]
    )
    ax.legend(handles=legend_handles,
              loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=3, fontsize=9, frameon=True, handlelength=1.2)

    plt.tight_layout(pad=0.2)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


def plot_trajectory_samples(df, out_dir, episodes):
    """Produce one trajectory plot per sample episode. The top-ranked episode
    (index 0) is also saved as trajectory_sample.pdf for deterministic use by
    copy_to_figs."""
    for i, (env_id, ep_idx) in enumerate(episodes):
        rollout_df = df[(df["env_id"] == env_id) & (df["episode_index"] == ep_idx)]
        if len(rollout_df) == 0:
            continue
        out_path = os.path.join(out_dir, f"trajectory_env{env_id}_ep{ep_idx}.pdf")
        _save_trajectory(rollout_df, env_id, ep_idx, out_path)
        if i == 0:
            canonical = os.path.join(out_dir, "trajectory_sample.pdf")
            import shutil; shutil.copy2(out_path, canonical)
            print(f"  saved → {canonical} (top-ranked)")


# ── main ─────────────────────────────────────────────────────────────────────

def load(spec_dir, n_samples=N_SAMPLES, rank_by="eating+biting"):
    step_pkl = os.path.join(spec_dir, "derived", "per_env_ep_agent_step.pkl")
    if not os.path.exists(step_pkl):
        sys.exit(f"Not found: {step_pkl}")
    out_dir = os.path.join(spec_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Loading {step_pkl} \u2026")
    df = pd.read_pickle(step_pkl)
    print(f"  {len(df):,} rows")
    episodes = _select_sample_episodes(df, n=n_samples, rank_by=rank_by)
    print(f"  {len(episodes)} sample episode(s) selected")
    return {"df": df, "out_dir": out_dir, "episodes": episodes}


def run(spec_dir, n_samples=N_SAMPLES, rank_by="eating+biting"):
    step_pkl = os.path.join(spec_dir, "derived", "per_env_ep_agent_step.pkl")
    if not os.path.exists(step_pkl):
        sys.exit(f"Not found: {step_pkl}")

    out_dir = os.path.join(spec_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {step_pkl} …")
    df = pd.read_pickle(step_pkl)
    print(f"  {len(df):,} rows")

    episodes = _select_sample_episodes(df, n=n_samples, rank_by=rank_by)
    print(f"  {len(episodes)} sample episode(s) selected ({rank_by})")

    plot_ethogram_samples(df, out_dir, episodes)
    plot_trajectory_samples(df, out_dir, episodes)
    print("Done.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec_dir", required=True)
    ap.add_argument("--n_samples", type=int, default=N_SAMPLES,
                    help=f"number of sample episodes to plot (default: {N_SAMPLES})")
    ap.add_argument("--rank_by", default="eating+biting",
                    choices=["eating", "biting", "eating+biting"],
                    help="how to rank episodes for sample selection (default: eating+biting)")
    args = ap.parse_args(argv)
    run(args.spec_dir, n_samples=args.n_samples, rank_by=args.rank_by)


if __name__ == "__main__":
    main()
