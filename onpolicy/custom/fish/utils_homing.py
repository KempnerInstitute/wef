""" Example:
HOMING_PKL="results/Homing1000000Seed1NoFood/20251017_191127/outputs/MAFish_neural_20251017_191127_GiUEdQat_agg_flattened.pkl"
python utils_homing.py --flat_pkl_file $HOMING_PKL
"""

import os
import re
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from pathlib import Path
import time
import argparse
import tqdm
from typing import Dict, List, Optional
import seaborn as sns
from cfg import FEATURE_METADATA, FEATURE_TYPE_COLORMAP, EXCLUDE_FROM_DECODING, AGENT_PARAMS, m_to_cm
import utils_features as uf
import utils_report as ru
from analysis_style import save as _save_pdf, panel as _panel
from utils_behavior import plot_agent_trajectories_together


HOMING_REGRESSION_FEATURES = [
    # "mormyromast_field_magnitude_log",
    # "ampullary_field_magnitude_log",
    # "knollen_field_magnitude_log", # Not calculated, Knollen is binary
    # "mormyromast_field_direction",
    # "ampullary_field_direction",
    # "knollen_field_direction_nearest",
    "knollen_error_angle_nearest",
    "mormyromast_error_angle",
    "ampullary_error_angle",
    "angle_to_closest_agent",
    "distance_to_wall",
    "distance_to_nearest_agent",
]




def compute_homing_episode_outcomes(dff, homing_radius=5):
    # Expects dff pre-filtered to agent_id=1 (homing agent) only.
    # Caller must extract agent 0's positions as target_x/target_y and drop agent 0 rows
    # before calling — see analysis_homing.py for the canonical filtering sequence.
    num_unique_agents = dff["agent_id"].nunique()
    assert num_unique_agents == 1, f"Expected only one unique agent_id in flattened data, found {num_unique_agents}"

    episode_keys = ["env_id", "episode_index", "agent_id"]
    episode_outcomes = dff.groupby(episode_keys).size().reset_index(name="length")

    # add first and last timesteps/positions of each episode
    start_rows = dff.loc[
        dff.groupby(episode_keys)["time_step"].idxmin(),
        episode_keys + ["time_step", "position_x", "position_y"],
    ].rename(
        columns={
            "time_step": "start_time_step",
            "position_x": "start_pos_x",
            "position_y": "start_pos_y",
        }
    )
    episode_outcomes = episode_outcomes.merge(start_rows, on=episode_keys, how="left")

    end_rows = dff.loc[
        dff.groupby(episode_keys)["time_step"].idxmax(),
        episode_keys + ["time_step", "position_x", "position_y", "target_x", "target_y"],
    ].rename(
        columns={
            "time_step": "end_time_step",
            "position_x": "end_pos_x",
            "position_y": "end_pos_y",
            "target_x": "target_pos_x",
            "target_y": "target_pos_y",
        }
    )
    episode_outcomes = episode_outcomes.merge(end_rows, on=episode_keys, how="left")

    episode_outcomes["total_distance"] = np.sqrt(
        (episode_outcomes["end_pos_x"] - episode_outcomes["start_pos_x"]) ** 2
        + (episode_outcomes["end_pos_y"] - episode_outcomes["start_pos_y"]) ** 2
    )
    episode_outcomes["distance_to_target_start"] = np.sqrt(
        (episode_outcomes["start_pos_x"] - episode_outcomes["target_pos_x"]) ** 2
        + (episode_outcomes["start_pos_y"] - episode_outcomes["target_pos_y"]) ** 2
    )
    episode_outcomes["distance_to_target_end"] = np.sqrt(
        (episode_outcomes["end_pos_x"] - episode_outcomes["target_pos_x"]) ** 2
        + (episode_outcomes["end_pos_y"] - episode_outcomes["target_pos_y"]) ** 2
    )

    row_distance_to_target = np.sqrt(
        (dff["position_x"] - dff["target_x"]) ** 2 + (dff["position_y"] - dff["target_y"]) ** 2
    )
    first_reach = (
        dff.loc[row_distance_to_target < homing_radius]
        .groupby(episode_keys)["time_step"]
        .min()
        .reset_index(name="time_to_target_step")
    )
    episode_outcomes = episode_outcomes.merge(first_reach, on=episode_keys, how="left")
    episode_outcomes["ever_reached_target"] = episode_outcomes["time_to_target_step"].notna()
    episode_outcomes["steps_to_target"] = (
        episode_outcomes["time_to_target_step"] - episode_outcomes["start_time_step"]
    )
    episode_outcomes["success"] = episode_outcomes["distance_to_target_end"] < homing_radius
    episode_outcomes["outcome"] = np.where(episode_outcomes["success"], "success", "fail")

    return episode_outcomes


def get_eligible_homing_episode_ids(dff, homing_radius=5):
    episode_outcomes = compute_homing_episode_outcomes(dff, homing_radius=homing_radius)
    num_episodes_before = episode_outcomes.shape[0]
    eligible_episodes = episode_outcomes[episode_outcomes["success"]]
    num_episodes_after = eligible_episodes.shape[0]
    success_percent = 100.0 * num_episodes_after / num_episodes_before
    print("Percent successful homing episodes: %.2f%% (%d / %d)" % (
        success_percent,
        num_episodes_after,
        num_episodes_before
    ))
    return eligible_episodes, success_percent


def plot_homing_trajectory(dff, env_id=0, episode_idx=0, legend_prefix=None, nolegend_prefix=None):
    subset = dff[
        (dff["env_id"] == env_id) & (dff["episode_index"] == episode_idx)
    ]
    target_location = dff.loc[(dff["env_id"] == env_id) & (dff["episode_index"] == episode_idx), ["target_x", "target_y"]].values[0]


    agent_id = 1  # agent_id=1 is always the homing agent; agent_id=0 is the stationary target
    agent_data = subset[subset["agent_id"] == agent_id]
    positions = np.array(agent_data["position"].tolist())

    fig, ax = plt.subplots(figsize=(2, 2))
    trajectory_blue = "#0619d8"
    trajectory_light = "#d9d8ff"
    target_red = "#a51e32"

    if len(positions) > 1:
        points = positions[:, :2].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "homing_trajectory", [trajectory_light, trajectory_blue]
        )
        colors = cmap(np.linspace(0, 1, len(segments)))
        colors[:, 3] = np.linspace(0.35, 1.0, len(segments))
        ax.add_collection(LineCollection(segments, colors=colors, linewidths=1.1, zorder=2))
    elif len(positions) == 1:
        ax.scatter(positions[0, 0], positions[0, 1], c=trajectory_blue, s=20, zorder=2)

    # Plot target
    ax.scatter(
        target_location[0],
        target_location[1],
        c=target_red,
        marker="s",
        s=38,
        label="Target",
        zorder=4,
    )
    ax.scatter(
        positions[0, 0],
        positions[0, 1],
        c=trajectory_blue,
        marker="o",
        s=38,
        label="Start",
        zorder=5,
    )
    ax.scatter(
        positions[-1, 0],
        positions[-1, 1],
        c=trajectory_blue,
        marker="s",
        s=38,
        label="End",
        zorder=5,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.08)

    ax.set_facecolor("white")

    # plt.title(f"Env {env_id}, Episode {episode_idx}")
    # plt.xlabel("X Position")
    # plt.ylabel("Y Position")
    # plt.axis("scaled")
    # plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    legend = ax.legend(loc="best", frameon=True, facecolor="white", edgecolor="#dddddd")
    # plt.grid(True)
    fig.tight_layout()

    if legend_prefix is not None:
        _save_pdf(fig, legend_prefix)
    if nolegend_prefix is not None:
        legend.set_visible(False)
        _save_pdf(fig, nolegend_prefix)
    if legend_prefix is not None or nolegend_prefix is not None:
        print(f"Saved trajectory: {legend_prefix or nolegend_prefix}")
    else:
        plt.show()
    plt.close(fig)


def plot_error_histogram(error_angles, title_str=None, n_bins=30, figsize=(3, 3), outfile_prefix=None):
    # Wrap angles to [-π, π] for consistency
    error_angles = (error_angles + np.pi) % (2 * np.pi) - np.pi

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=figsize)
    ax.set_theta_zero_location("N")      # 0 radians at the top
    ax.set_theta_direction(-1)           # Clockwise
    # ax.set_thetalim(-np.pi, np.pi)       # Explicitly set domain
    #     
    n, bins, patches = ax.hist(
        error_angles,
        bins=n_bins,
        density=True,
        alpha=0.75,
        color="steelblue",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.grid(alpha=0.3)
    if title_str is not None:
        ax.set_title(title_str, va="bottom", fontsize=10)
    ax.set_yticklabels([])  # Hide radial ticks
    ax.set_xticks([np.pi/2, 0, 3*np.pi/2, np.pi])
    ax.set_xticklabels(["90°", "0°", "-90°", "180°"])
    # ax.legend(loc="upper right", fontsize=8, frameon=False)
    plt.tight_layout()
    if outfile_prefix is not None:
        _save_pdf(fig, f"{outfile_prefix}_polar_histogram_{title_str}")
    plt.show()

def plot_error_angle_raster(dff, col_name, title_str=None, figsize=(4, 3), outfile_prefix=None):
    fig, ax = plt.subplots(figsize=figsize)
    # Wrap angles to [-π, π] for consistency
    error_angles = dff[col_name].values
    error_angles = (error_angles + np.pi) % (2 * np.pi) - np.pi

    ax.scatter(
        dff["timestep_normalized"].values,
        error_angles,
        s=0.5,
        color="black",
        alpha=0.25,
        edgecolors="none",
    )

    # Add horizontal lines at ±π
    plt.axhline(y=np.pi, color='k', linestyle='--', alpha=1)
    plt.axhline(y=-np.pi, color='k', linestyle='--', alpha=1)
    plt.axhline(y=0, color='k', linestyle='-', alpha=1)
    
    ax.set_ylabel("Error Angle (radians)")
    ax.set_xlabel("Normalized Time")
    ax.set_xlim(0, 1)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels(["-180°", "0°", "180°"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0%", "100%"])

    if title_str is not None:
        ax.set_title(title_str)
    plt.tight_layout()
    if outfile_prefix is not None:
        _save_pdf(fig, f"{outfile_prefix}_raster_{title_str}")
    plt.show()

def add_timestep_normalized_column(dff):
    # Normalizes per-episode so rasters align across variable-length episodes.
    # Homing episodes are variable-length: successful ones terminate early
    # (MAEFish homing_success_counter >= required_homing_steps), failed ones run
    # to max_episode_length.
    dff["timestep_normalized"] = dff.groupby(["env_id", "episode_index"])["time_step"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )
    return dff

# def add_nearest_agent_id_column(dff, task):
#     if "nearest_agent_id" not in dff.columns and task in ["2ff", "2f1p", "homing"]:
#         print("Computing nearest_agent_id column for 2f tasks...")
#         dff["nearest_agent_id"] = np.abs(1 - dff["agent_id"]).astype(int)
#     return dff


def _scatter_jitter(ax, dist, success):
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.03, 0.03, len(dist))
    colors = ["#2ca02c" if s else "#d62728" for s in success]
    ax.scatter(dist, success.astype(float) + jitter, c=colors, s=10, alpha=0.45, zorder=2)
    ax.axhline(0, color="0.75", lw=0.5)
    ax.axhline(1, color="0.75", lw=0.5)
    ax.set_ylim(-0.18, 1.22)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("Initial distance to target (cm)")


def plot_success_vs_initial_distance(episode_outcomes, outfile_prefix, figsize=(4, 3)):
    """Logistic regression of P(success) vs initial distance, with 95% CI band."""
    import statsmodels.api as sm

    df = episode_outcomes.copy()
    dist = df["distance_to_target_start"].values.astype(float)
    y = df["success"].astype(float).values
    n_total = len(df)
    n_success = int(y.sum())

    X = sm.add_constant(dist)
    # The Logit Hessian is singular under perfect separation or tiny n (e.g. a
    # re-eval with very few episodes, or an all-success/all-fail policy). Catch
    # that and fall back to a flat success-rate line rather than aborting the
    # whole homing report. Cases that currently fit successfully are unchanged.
    result = None
    try:
        result = sm.Logit(y, X).fit(disp=0)
    except Exception as e:
        print(f"[homing] success_vs_initial_distance: logistic fit skipped "
              f"({type(e).__name__}: {e}).")

    fig, ax = _panel(2, 2)
    _scatter_jitter(ax, dist, y)
    if result is not None:
        x_grid = np.linspace(dist.min(), dist.max(), 300)
        X_grid = sm.add_constant(x_grid)
        pred = result.get_prediction(X_grid)
        pred_df = pred.summary_frame(alpha=0.05)
        ax.fill_between(x_grid, pred_df["ci_lower"], pred_df["ci_upper"],
                        color="k", alpha=0.15, zorder=3)
        ax.plot(x_grid, pred_df["predicted"], "k-", lw=2, zorder=4)

        b1 = result.params[1]
        se1 = result.bse[1]
        p1 = result.pvalues[1]
        r2 = result.prsquared
        p_str = r"$p < 0.001$" if p1 < 0.001 else fr"$p = {p1:.3f}$"
        annot = (
            r"$\mathrm{logit}\,P = \beta_0 + \beta_1 d$" + "\n"
            + fr"$\beta_1 = {b1:.3f} \pm {se1:.3f}$  ({p_str})" + "\n"
            + fr"McFadden $R^2 = {r2:.3f}$" + "\n"
            + fr"$n = {n_total}$ ({n_success} success)"
        )
    else:
        rate = float(y.mean()) if n_total else 0.0
        ax.axhline(rate, color="k", lw=2, ls="--", zorder=4)
        annot = (
            "logistic fit skipped (singular / perfect separation)" + "\n"
            + fr"$P = {rate:.2f}$,  $n = {n_total}$ ({n_success} success)"
        )
    ax.set_ylabel("P(success)")
    ax.set_xlabel("Initial distance to target (cm)")
    fig.tight_layout()
    # Place annotation below axes so it never overlaps data
    ax.text(0.5, -0.38, annot, transform=ax.transAxes, ha="center", va="top",
            fontsize=6, color="0.15", clip_on=False,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8", alpha=0.9))
    _save_pdf(fig, f"{outfile_prefix}success_vs_initial_distance")
    plt.close(fig)


def plot_success_vs_initial_distance_compare(episode_outcomes, outfile_prefix, figsize=(4, 3.5)):
    """Logistic regression fit of P(success) vs initial distance to target."""
    from sklearn.linear_model import LogisticRegression

    df = episode_outcomes.copy()
    dist = df["distance_to_target_start"].values
    y = df["success"].astype(float).values
    x_grid = np.linspace(dist.min(), dist.max(), 300)

    fig, ax = plt.subplots(figsize=figsize)
    _scatter_jitter(ax, dist, df["success"])
    if len(np.unique(y)) < 2:
        # Degenerate case (e.g. a strong policy that succeeds on every episode):
        # LogisticRegression cannot fit with a single class. Draw the constant rate.
        rate = float(y.mean())
        ax.axhline(rate, color="k", lw=2, ls="--", zorder=3)
        ax.text(0.5, 0.05, f"all-{'success' if rate >= 0.5 else 'fail'} "
                           f"(P={rate:.2f}); logistic fit skipped",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=6, color="0.3")
    else:
        lr = LogisticRegression().fit(dist.reshape(-1, 1), y)
        lr_proba = lr.predict_proba(x_grid.reshape(-1, 1))[:, 1]
        ax.plot(x_grid, lr_proba, "k-", lw=2, zorder=3)
    ax.set_ylabel("P(success)")
    ax.set_xlabel("Initial distance to target (cm)")

    fig.tight_layout()
    _save_pdf(fig, f"{outfile_prefix}success_vs_initial_distance_logistic")
    plt.close(fig)


def plot_starting_positions(episode_outcomes, outfile_prefix=None):
    """Scatter plot of homing agent and target starting positions across all episodes."""
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(5, 5))

    success = episode_outcomes["success"]
    ax.scatter(
        episode_outcomes.loc[success, "start_pos_x"],
        episode_outcomes.loc[success, "start_pos_y"],
        c="steelblue", s=20, alpha=0.7, label="homer (success)", zorder=3,
    )
    ax.scatter(
        episode_outcomes.loc[~success, "start_pos_x"],
        episode_outcomes.loc[~success, "start_pos_y"],
        c="tomato", s=20, alpha=0.7, label="homer (fail)", zorder=3,
    )
    ax.scatter(
        episode_outcomes["target_pos_x"],
        episode_outcomes["target_pos_y"],
        c="black", marker="x", s=30, alpha=0.6, label="target", zorder=4,
    )

    # Draw lines from homer start to target
    for _, row in episode_outcomes.iterrows():
        col = "steelblue" if row["success"] else "tomato"
        ax.plot(
            [row["start_pos_x"], row["target_pos_x"]],
            [row["start_pos_y"], row["target_pos_y"]],
            color=col, alpha=0.15, lw=0.6,
        )

    arena_w = episode_outcomes[["start_pos_x", "target_pos_x"]].max().max()
    arena_h = episode_outcomes[["start_pos_y", "target_pos_y"]].max().max()
    ax.set_xlim(0, max(arena_w * 1.05, 10))
    ax.set_ylim(0, max(arena_h * 1.05, 10))
    ax.set_aspect("equal")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_title(f"Starting positions (N={len(episode_outcomes)} episodes)")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()

    if outfile_prefix:
        _save_pdf(fig, f"{outfile_prefix}starting_positions")
    plt.close(fig)


def run_homing_report(
    dff,
    outfile_prefix,
    raw_dir=None,
    num_samples=50,
    run_episode_outcomes=True,
    run_trajectories=True,
    run_error_angles=True,
    run_decoding=True,
    dff_full=None,
):
    needs_features = run_error_angles or run_decoding
    if needs_features:
        columns_before = dff.columns.tolist()
        dff, decoding_feature_names = uf.prepare_features_for_decoding(
            dff,
            normalize=False,
            behavior_mode="homing",
        )
        print("New columns:", set(dff.columns) - set(columns_before))
        print("Existing columns:", set(dff.columns) & set(columns_before))

    episode_outcomes = None
    if run_episode_outcomes or run_trajectories:
        episode_outcomes = compute_homing_episode_outcomes(dff, homing_radius=5)

    if run_episode_outcomes:
        outcomes_csv = f"{outfile_prefix}homing_episode_outcomes.csv"
        episode_outcomes.to_csv(outcomes_csv, index=False)
        print(f"Saved homing episode outcomes CSV to {outcomes_csv}")
        plot_success_vs_initial_distance(episode_outcomes, outfile_prefix=outfile_prefix)
        plot_success_vs_initial_distance_compare(episode_outcomes, outfile_prefix=outfile_prefix)
        plot_starting_positions(episode_outcomes, outfile_prefix=outfile_prefix)

    if run_trajectories:
        num_episodes_before = episode_outcomes.shape[0]
        successful_eps = episode_outcomes[episode_outcomes["success"]].sort_values("length", ascending=False)
        num_episodes_after = len(successful_eps)
        success_percent = 100.0 * num_episodes_after / num_episodes_before
        print("Percent successful homing episodes: %.2f%% (%d / %d)" % (
            success_percent, num_episodes_after, num_episodes_before
        ))

        # Fill up to num_samples: longest successes first, then failures to top up
        n_success = min(num_samples, len(successful_eps))
        sampled = successful_eps.head(n_success)
        n_fail = num_samples - n_success
        if n_fail > 0:
            failed_eps = episode_outcomes[~episode_outcomes["success"]].head(n_fail)
            if len(failed_eps) > 0:
                sampled = pd.concat([sampled, failed_eps])
                print(f"Only {n_success} successful episodes; adding {len(failed_eps)} failed episodes "
                      f"to reach {len(sampled)} total trajectories.")

        # Trajectory outputs go into subfolders to keep analyses/homing/ tidy
        if outfile_prefix.endswith("/"):
            _traj_base = outfile_prefix.rstrip("/")
        else:
            _traj_base = os.path.dirname(outfile_prefix)
        legend_dir = os.path.join(_traj_base, "trajectories_legend")
        nolegend_dir = os.path.join(_traj_base, "trajectories_nolegend")
        os.makedirs(legend_dir, exist_ok=True)
        os.makedirs(nolegend_dir, exist_ok=True)

        for row in sampled.itertuples(index=False):
            env_id, episode_index = row.env_id, row.episode_index
            outcome_label = "success" if row.success else "fail"
            print(f"Plotting env_id={env_id}, episode_index={episode_index} ({outcome_label})")
            basename = f"trajectory_{outcome_label}_env{env_id}_ep{episode_index}"
            plot_homing_trajectory(
                dff,
                env_id=env_id,
                episode_idx=episode_index,
                legend_prefix=os.path.join(legend_dir, basename),
                nolegend_prefix=os.path.join(nolegend_dir, basename),
            )

    if run_error_angles:
        dff = add_timestep_normalized_column(dff)
        for col in [
            "knollen_error_angle_nearest",
            "mormyromast_error_angle",
            "ampullary_error_angle",
            ]:
            if col not in dff.columns:
                print(f"[homing] Skipping error-angle column '{col}': not in data.")
                continue
            print(f"Statistics for {col}:")
            # plot_error_histogram(error_angles=dff[col].dropna().values, title_str=f"{col}")
            # plot_error_angle_raster(dff, col, title_str=f"{col}")
            plot_error_histogram(error_angles=dff[col].dropna().values, figsize=(2, 2), title_str=None, outfile_prefix=outfile_prefix+f"{col}")
            plot_error_angle_raster(dff, col, figsize=(4, 3), title_str=None, outfile_prefix=outfile_prefix+f"{col}")

    if run_decoding and raw_dir is None:
        print("[homing] Skipping decoding: raw_dir not provided (need ep*_rnn.npy).")
        run_decoding = False

    if run_decoding:
        # Decode HOMING_REGRESSION_FEATURES from the RNN state using the shared
        # foraging decoder (analysis_rnn_decoding): GroupKFold OLS with a per-fold
        # mean baseline, identical to the old modular pipeline.  No sensor-range
        # split — homing decodes the pooled ("combined") condition only.
        #
        # Use dff_full (untrimmed agent-1 rows) for accumulate: ep*_rnn.npy files
        # contain T steps for the entire episode, but dff is trimmed to the first
        # successful homing trial.  For high-success-rate seeds, trimmed episodes
        # have fewer than T rows and accumulate rejects them all.
        import analysis_rnn_decoding as ard

        if dff_full is not None:
            dff_dec, _ = uf.prepare_features_for_decoding(
                dff_full, normalize=False, behavior_mode="homing"
            )
        else:
            dff_dec = dff

        feats = [f for f in HOMING_REGRESSION_FEATURES
                 if f in dff_dec.columns and f not in EXCLUDE_FROM_DECODING]
        print("[homing] decoding features:", feats)

        X, feat_df, groups, _ = ard.accumulate(raw_dir, dff_dec, feats)
        if X is None:
            print("[homing] No RNN states matched — skipping decoding.")
        else:
            print(f"[homing] decoding on {len(X)} timesteps")
            # Use ALL rows (no subsample cap): the old utils_decoding pipeline did
            # not downsample, and with H=512 RNN units a 5000-row cap overfits OLS.
            res = ard.run_decoding_condition(X, feat_df, groups, feats, "combined",
                                             max_rows=len(X))
            res.to_csv(f"{outfile_prefix}decoding_homing.csv", index=False)
            # Compact square panel (old utils_decoding aesthetic), not the wide
            # multi-condition foraging bar chart.
            ard.plot_decoding_homing(res, f"{outfile_prefix}decoding_homing",
                                     keep_negative=True)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run homing report with flat pkl data file")
    parser.add_argument("--flat_pkl_file", type=str, default=None, help="Path to flattened pickle file")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of trajectories to plot (successes first, then failures)")
    parser.add_argument("--episode_outcomes", action="store_true", help="Run homing episode outcomes CSV analysis")
    parser.add_argument("--trajectories", action="store_true", help="Plot sample successful homing trajectories")
    parser.add_argument("--error_angles", action="store_true", help="Plot error-angle histograms and rasters")
    parser.add_argument("--decoding", action="store_true", help="Run decoding analysis")
    args = parser.parse_args()
    flat_pkl_file = args.flat_pkl_file
    task = "homing"

    selected_analyses = [
        args.episode_outcomes,
        args.trajectories,
        args.error_angles,
        args.decoding,
    ]
    run_all = not any(selected_analyses)

    # LOAD
    start_time = time.time()
    dff, OUTPUTS_FOLDER, pkl_str = ru.load_flat_pkl_file(flat_pkl_file, task=task)
    end_time = time.time()
    print(f"Loaded in {end_time - start_time:.2f} seconds.")
    print(f"Columns in DataFrame: {dff.columns.tolist()}")

    # RUN REPORT
    outfile_prefix = OUTPUTS_FOLDER + "/" + pkl_str + "_"
    run_homing_report(
        dff,
        outfile_prefix=outfile_prefix,
        num_samples=args.num_samples,
        run_episode_outcomes=run_all or args.episode_outcomes,
        run_trajectories=run_all or args.trajectories,
        run_error_angles=run_all or args.error_angles,
        run_decoding=run_all or args.decoding,
    )
