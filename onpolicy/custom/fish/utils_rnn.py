import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from moviepy.editor import VideoFileClip, ColorClip, clips_array
from scipy.spatial.distance import pdist
from scipy.stats import linregress
import math
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.cluster.hierarchy as sch
import os
import umap
import time
import matplotlib
import matplotlib.cm as cm
from matplotlib.colors import Normalize, to_rgba
import argparse
from utils_figsaving import _save_data, _load_data



def get_frame_counts(dff):
    # Group by env_id, episode_index, and agent_id to count the number of frames
    frame_counts = (
        dff.groupby(["env_id", "episode_index", "agent_id"])
        .size()
        .reset_index(name="frame_count")
    )
    # print(frame_counts.head())

    # Pivot the table so that each agent_id is a column
    pivot_frame_counts = frame_counts.pivot(
        index=["env_id", "episode_index"], columns="agent_id", values="frame_count"
    )
    pivot_frame_counts.columns.name = None
    pivot_frame_counts = pivot_frame_counts.reset_index()
    # print(pivot_frame_counts.head())

    agent_columns = pivot_frame_counts.columns[2:]  # Get only the agent columns
    pivot_frame_counts["frame_discrepancy"] = pivot_frame_counts[agent_columns].max(
        axis=1
    ) - pivot_frame_counts[agent_columns].min(axis=1)
    return pivot_frame_counts


def create_pca_basis(dff, common_pca, anim_dim):
    """Creates a PCA basis, either common across all agents or per agent."""
    if common_pca:
        # Calculate PCA based on all agents' rnn_states combined
        all_rnn_states = np.vstack(dff["rnn_states"].tolist())
        pca = PCA(n_components=anim_dim)
        pca.fit(all_rnn_states)
        return pca  # Return the common PCA object
    else:
        # Calculate PCA for each agent individually
        pca_dict = {}
        for agent_id in dff["agent_id"].unique():
            agent_rnn_states = np.vstack(
                dff[dff["agent_id"] == agent_id]["rnn_states"].tolist()
            )
            pca = PCA(n_components=anim_dim)
            pca.fit(agent_rnn_states)
            pca_dict[agent_id] = pca  # Store the PCA object for each agent
        return pca_dict  # Return a dictionary of PCA objects keyed by agent_id


def plot_cumulative_variance(
    pca,
    title,
    threshold=0.90,
    markers=("threshold", "effrank"),
    knee_curve="concave",
    knee_direction="increasing",
    figsize=(2.5, 2.5),
    outfile_base=None,
    do_plot=True,
):
    """
    Plot cumulative variance and optionally mark:
      'threshold' = first PC reaching `threshold`
      'effrank'   = effective rank 1/sum(rho_i^2)
      knee point is computed via kneed.KneeLocator (returned as k_idx only)

    Returns
    -------
    cumvar : np.ndarray
    idx_cumvar : int
        1-based index of first PC reaching threshold, or -1 if not reached
    D_eff : float
    k_idx : int or None
    """
    evr = pca.explained_variance_ratio_
    cumvar = np.cumsum(evr)
    pcs = np.arange(1, len(evr) + 1)

    # compute all stats first (pure)
    if np.any(cumvar >= threshold):
        idx_cumvar = np.where(cumvar >= threshold)[0][0] + 1  # 1-based
    else:
        idx_cumvar = -1

    D_eff = 1.0 / np.sum(evr**2)

    # knee is optional but inexpensive
    from kneed import KneeLocator
    kl = KneeLocator(pcs, cumvar, curve=knee_curve, direction=knee_direction)
    k_idx = kl.knee

    # If no plotting requested AND no saving, we can return now
    if not do_plot and outfile_base is None:
        return cumvar, idx_cumvar, D_eff, k_idx

    # otherwise, build the figure
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(pcs, cumvar, marker="o", label="Cumu. variance", markersize=1)

    # 1) threshold marker
    if "threshold" in markers:
        if idx_cumvar != -1:
            ax.axhline(
                y=threshold,
                color="r",
                linestyle=":",
                label=f"{int(threshold*100)}% cumu. var.",
            )
            ax.axvline(
                x=idx_cumvar,
                color="r",
                linestyle="--",
                label=f"{int(threshold*100)}% at PC{idx_cumvar}",
            )
        else:
            ax.axhline(
                y=threshold,
                color="r",
                linestyle="--",
                label=f"{int(threshold*100)}% (not reached)",
            )

    # 2) effective-rank marker
    if "effrank" in markers:
        ax.axvline(
            x=D_eff, color="g", linestyle="--", label=f"Eff. rank: {D_eff:.1f}"
        )

    # 3) knee marker (disabled by default via `markers`)
    if "knee" in markers:
        if k_idx is not None:
            ax.axvline(
                x=k_idx, color="b", linestyle="--", label=f"Knee at PC:{k_idx}"
            )
        else:
            ax.text(
                0.5,
                0.1,
                "KneeLocator found no knee",
                transform=ax.transAxes,
                color="m",
                alpha=0.7,
            )

    ax.set_title(title)
    ax.set_xlabel("PC index")
    ax.set_ylabel("Variance explained")
    ax.legend(loc="best")
    fig.tight_layout()

    if outfile_base is not None:
        outfile_png = f"{outfile_base}_cumulative_variance.png"
        outfile_pdf = f"{outfile_base}_cumulative_variance.pdf"
        fig.savefig(outfile_png, dpi=300)
        fig.savefig(outfile_pdf)
        print(f"Saved cumulative variance plot to {outfile_png} and {outfile_pdf}")

    if do_plot:
        plt.show()
    else:
        plt.close(fig)

    return cumvar, idx_cumvar, D_eff, k_idx


def plot_pca_2d(rnn_states_2d, state_colors, pca, title_string=None):
    plt.figure(figsize=(5, 4))
    plt.scatter(
        rnn_states_2d[:, 0],
        rnn_states_2d[:, 1],
        c=state_colors,
        cmap="viridis",
        alpha=0.5,
        s=4,
    )
    if title_string is not None:
        plt.title(title_string)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% var)")
    plt.colorbar()
    plt.show()


def plot_pca_3d(rnn_states_3d, state_colors, pca, title_string=None):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        rnn_states_3d[:, 0],
        rnn_states_3d[:, 1],
        rnn_states_3d[:, 2],
        c=state_colors,
        cmap="viridis",
        alpha=0.5,
        s=4,
    )
    if title_string is not None:
        plt.title(title_string)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% var)")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.2f}% var)")
    fig.colorbar(scatter, pad=0.1)
    plt.show()


def plot_all_agents_projected_2d(dff, pca, edge=2, outfile_base=None):
    agent_ids = dff["agent_id"].unique()
    num_agents = len(agent_ids)

    fig, axs = plt.subplots(1, num_agents, figsize=(edge * num_agents, edge))
    if num_agents == 1:
        axs = [axs]

    for i, agent_id in enumerate(agent_ids):
        dff_agent = dff[dff["agent_id"] == agent_id]
        rnn_states = np.vstack(dff_agent["rnn_states"].tolist())
        proj_2d = pca.transform(rnn_states)[:, :2]

        ax = axs[i]
        ax.scatter(
            proj_2d[:, 0],
            proj_2d[:, 1],
            c=dff_agent["agent_id"],
            cmap="tab20",
            alpha=0.5,
            s=2,
        )
        ax.set_title(f"Agent: {agent_id}")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")

    fig.tight_layout()
    if outfile_base is not None:
        outfile = f"{outfile_base}_all_agents_pca_2d.png"
        plt.savefig(outfile, dpi=300)
        # plt.savefig(outfile.replace(".png", ".pdf"))
        print(f"Saved all agents PCA 2D plot to {outfile}")
    plt.show()


def plot_all_agents_projected_3d(dff, pca):
    agent_ids = dff["agent_id"].unique()
    num_agents = len(agent_ids)

    fig = plt.figure(figsize=(4 * num_agents, 4))

    for i, agent_id in enumerate(agent_ids):
        dff_agent = dff[dff["agent_id"] == agent_id]
        rnn_states = np.vstack(dff_agent["rnn_states"].tolist())
        proj_3d = pca.transform(rnn_states)[:, :3]

        ax = fig.add_subplot(1, num_agents, i + 1, projection="3d")
        sc = ax.scatter(
            proj_3d[:, 0],
            proj_3d[:, 1],
            proj_3d[:, 2],
            c=dff_agent["agent_id"],
            cmap="tab20",
            alpha=0.5,
            s=2,
        )
        ax.set_title(f"Agent: {agent_id}")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
        ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.2f}%)")

    fig.tight_layout()
    plt.show()


def perform_pca_analysis_rnn_obs_mask(data_matrix, title_prefix, sensor_types=None):
    """
    Perform PCA and plot variance + 2D projections on the given data matrix.

    Args:
        data_matrix (np.ndarray): Full matrix of shape (num_samples, num_features).
        title_prefix (str): Prefix for plot titles.
        sensor_types (list of tuples): Optional list of (sensor_name, indices) to run per-sensor PCA.
    """
    # Full data PCA
    pca = PCA()
    pcs_transformed = pca.fit_transform(data_matrix)

    plot_cumulative_variance(
        pca,
        f"{title_prefix} - Cumulative Variance Explained",
        knee_curve="concave",
        knee_direction="increasing",
    )

    plot_pca_2d(
        rnn_states_2d=pcs_transformed,
        state_colors=[1] * len(pcs_transformed),  # Dummy color for all points
        pca=pca,
        title_string=f"{title_prefix} - PCA 2D Projection",
    )

    # Optional: per-sensor PCA
    if sensor_types is not None:
        for sensor_name, indices in sensor_types:
            sensor_data = data_matrix[:, indices]

            # histogram of sensor data
            plt.figure(figsize=(6, 3))
            plt.hist(sensor_data.flatten(), bins=51, alpha=0.7)
            plt.title(f"{title_prefix} - {sensor_name} Sensors Data Histogram")
            plt.xlabel("Sensor Value")
            plt.ylabel("Frequency")
            plt.yscale("log")  # Log scale for better visibility
            plt.tight_layout()
            plt.show()
            plt.close()

            pca_sensor = PCA()
            sensor_pca_transformed = pca_sensor.fit_transform(sensor_data)

            plot_cumulative_variance(
                pca_sensor,
                f"{title_prefix} - {sensor_name} Sensors Cumulative Variance",
                knee_curve="concave",
                knee_direction="increasing",
            )

            plot_pca_2d(
                rnn_states_2d=sensor_pca_transformed,
                state_colors=[1] * len(sensor_pca_transformed),
                pca=pca_sensor,
                title_string=f"{title_prefix} - {sensor_name} Sensors PCA 2D",
            )


def other_pca_analyses(dff):
    all_observations = np.vstack(dff["observations"].tolist())

    # All observations PCA analysis
    pca_obs = PCA()
    obs_pcs_transformed = pca_obs.fit_transform(all_observations)
    plot_cumulative_variance(
        pca_obs,
        f"Cumulative Variance Explained for all observations",
        knee_curve="concave",
        knee_direction="increasing",
    )

    plot_pca_2d(
        rnn_states_2d=obs_pcs_transformed,
        state_colors=[1] * len(obs_pcs_transformed),  # Dummy color for all points
        pca=pca_obs,
        title_string="All Observations PCA 2D Projection",
    )

    # Sensor specific PCA analysis
    for sensor_name, indices in sensor_types:
        sensor_data = all_observations[:, indices]

        pca_sensor = PCA()
        sensor_pca_transformed = pca_sensor.fit_transform(sensor_data)

        plot_cumulative_variance(
            pca_sensor,
            f"Cumulative Variance Explained for {sensor_name} Sensors",
            knee_curve="concave",
            knee_direction="increasing",
        )

        plot_pca_2d(
            rnn_states_2d=sensor_pca_transformed,
            state_colors=[1] * len(sensor_pca_transformed),  # Dummy color for all points
            pca=pca_sensor,
            title_string=sensor_name,
        )

###### ANIMATION ######
def animate_2d(
    pca_result, pca, agent_id, episode_index, env_id, output_dir, file_prefix, fps
):
    """Creates a 2D animation."""
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.set_title(f"Agent {agent_id}", fontsize=10)
    ax.scatter(pca_result[:, 0], pca_result[:, 1], color="grey", alpha=0.5, s=2)
    (line,) = ax.plot([], [], f"C{agent_id}o-", markersize=4)
    (trace,) = ax.plot([], [], f"C{agent_id}-", linewidth=1)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% var)")
    # ax.set_title(f"Agent {agent_id} | Ep {episode_index} | Env {env_id}", fontsize=10)
    num_frames = len(pca_result)

    def init():
        ax.set_xlim(np.min(pca_result[:, 0] * 1.1), np.max(pca_result[:, 0]) * 1.1)
        ax.set_ylim(np.min(pca_result[:, 1] * 1.1), np.max(pca_result[:, 1]) * 1.1)
        trace.set_data([], [])
        line.set_data([], [])
        return trace, line

    def update(frame):
        trace.set_data(
            pca_result[max(0, frame - 9) : frame + 1, 0],
            pca_result[max(0, frame - 9) : frame + 1, 1],
        )
        line.set_data([pca_result[frame, 0]], [pca_result[frame, 1]])
        print(
            f"\rAnimating env {env_id}, episode {episode_index}, agent {agent_id}: {frame + 1}/{num_frames} frames",
            end="",
        )
        return trace, line

    ani = FuncAnimation(
        fig, update, frames=np.arange(len(pca_result)), init_func=init, blit=True
    )
    outfile = (
        f"{output_dir}/{file_prefix}{agent_id}_ep{episode_index}_env{env_id}_2D.mp4"
    )
    fig.tight_layout()
    ani.save(outfile, writer="ffmpeg", fps=fps)
    plt.close(fig)
    print(
        f"\nSaved 2D animation for agent {agent_id}, episode {episode_index}, env {env_id} to {outfile}"
    )
    return outfile  # Return the path to the generated video


def animate_3d(
    pca_result, pca, agent_id, episode_index, env_id, output_dir, file_prefix, fps
):
    """Creates a 3D animation."""
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"Agent {agent_id}", fontsize=10)
    ax.scatter(
        pca_result[:, 0],
        pca_result[:, 1],
        pca_result[:, 2],
        color="grey",
        alpha=0.5,
        s=4,
    )
    (line,) = ax.plot([], [], [], f"C{agent_id}o-", markersize=8)
    (trace,) = ax.plot([], [], [], f"C{agent_id}-", linewidth=3)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% var)")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.2f}% var)")
    num_frames = len(pca_result)

    def init():
        ax.set_xlim(np.min(pca_result[:, 0] * 1.1), np.max(pca_result[:, 0]) * 1.1)
        ax.set_ylim(np.min(pca_result[:, 1] * 1.1), np.max(pca_result[:, 1]) * 1.1)
        ax.set_zlim(np.min(pca_result[:, 2] * 1.1), np.max(pca_result[:, 2]) * 1.1)
        trace.set_data(np.empty(0), np.empty(0))
        trace.set_3d_properties(np.empty(0))
        line.set_data(np.empty(0), np.empty(0))
        line.set_3d_properties(np.empty(0))
        return trace, line

    def update(frame):
        trace.set_data(
            pca_result[max(0, frame - 9) : frame + 1, 0],
            pca_result[max(0, frame - 9) : frame + 1, 1],
        )
        trace.set_3d_properties(pca_result[max(0, frame - 9) : frame + 1, 2])
        line.set_data(
            [pca_result[frame, 0]], [pca_result[frame, 1]]
        )  # Ensure sequence is passed
        line.set_3d_properties([pca_result[frame, 2]])  # Ensure sequence is passed
        # Progress update
        print(
            f"\rAnimating agent {agent_id} in env {env_id}: {frame + 1}/{num_frames} frames",
            end="",
        )
        return trace, line

    ani = FuncAnimation(
        fig, update, frames=np.arange(len(pca_result)), init_func=init, blit=True
    )
    outfile = (
        f"{output_dir}/{file_prefix}{agent_id}_ep{episode_index}_env{env_id}_3D.mp4"
    )
    fig.tight_layout()
    ani.save(outfile, writer="ffmpeg", fps=fps)
    plt.close(fig)
    print(
        f"\nSaved 3D animation for agent {agent_id}, episode {episode_index}, env {env_id} to {outfile}"
    )
    return outfile  # Return the path to the generated video


def create_collage(env_id, episode_index, agent_videos, output_dir, file_prefix):
    """Creates a collage of agent videos, filling in with blank spaces if necessary."""
    # Determine the number of columns and rows for the collage
    num_videos = len(agent_videos)
    num_columns = (num_videos + 1) // 2
    num_rows = 2

    # Load video clips
    clips = [VideoFileClip(video) for video in agent_videos]

    # Add blank clips if there are fewer than 4 videos
    while len(clips) < 4:
        blank_clip = ColorClip(
            size=(clips[0].w, clips[0].h), color=(0, 0, 0), duration=clips[0].duration
        )
        clips.append(blank_clip)

    # Arrange clips in the specified grid
    clips_grid = [clips[:num_columns], clips[num_columns:]]
    final_clip = clips_array(clips_grid)

    # Save the collage
    collage_filename = (
        f"{output_dir}/{file_prefix}_collage_ep{episode_index}_env{env_id}.mp4"
    )
    final_clip.write_videofile(collage_filename, fps=24)
    print(f"Saved collage video to {collage_filename}")


def animate_per_agent_pca(
    dff,
    pca,
    anim_dim,
    output_dir,
    file_prefix,
    fps,
    max_envs=1,
    max_episodes=1,
):
    """Loops through agents to animate their RNN states and creates a collage."""
    env_ids = np.sort(dff["env_id"].unique()).tolist()[:max_envs]
    print(
        f"Animating up to {max_episodes} episodes of {max_envs} environments: {env_ids}"
    )
    for env_id in env_ids:
        env_dff = dff[dff["env_id"] == env_id].copy()
        episode_ids = env_dff["episode_index"].unique()
        # print(f"Animating environment {env_id} with episodes: {episode_ids}")
        for episode_index in episode_ids[:max_episodes]:
            episode_dff = env_dff[env_dff["episode_index"] == episode_index]
            agent_videos = []
            for agent_id in episode_dff["agent_id"].unique():
                agent_dff = episode_dff[episode_dff["agent_id"] == agent_id]
                rnn_states_agent = np.array(agent_dff["rnn_states"].tolist())

                # Use the common PCA or the per-agent PCA
                if isinstance(pca, PCA):
                    pca_to_use = pca  # Common PCA for all agents
                else:
                    pca_to_use = pca[agent_id]  # Agent-specific PCA

                pca_result = pca_to_use.transform(rnn_states_agent)

                if anim_dim == 3:
                    video_file = animate_3d(
                        pca_result,
                        pca_to_use,
                        agent_id,
                        episode_index,
                        env_id,
                        output_dir,
                        file_prefix,
                        fps,
                    )
                else:
                    video_file = animate_2d(
                        pca_result,
                        pca_to_use,
                        agent_id,
                        episode_index,
                        env_id,
                        output_dir,
                        file_prefix,
                        fps,
                    )
                agent_videos.append(video_file)

            # Create a collage of the videos, accommodating up to 4 videos
            if len(agent_videos) > 0:  # Ensure there is at least one video
                create_collage(
                    env_id, episode_index, agent_videos, output_dir, file_prefix
                )


def animate_per_agent_umap(
    dff,
    umap_embedding,
    output_dir,
    file_prefix,
    fps,
    max_envs=1,
    max_episodes=1,
):
    dff["umap_1"] = umap_embedding[:, 0]
    dff["umap_2"] = umap_embedding[:, 1]

    """Loops through agents to animate their RNN states and creates a collage."""
    env_ids = np.sort(dff["env_id"].unique()).tolist()[:max_envs]
    print(
        f"Animating up to {max_episodes} episodes of {max_envs} environments: {env_ids}"
    )
    for env_id in env_ids:
        env_dff = dff[dff["env_id"] == env_id]
        episode_ids = env_dff["episode_index"].unique()
        # print(f"Animating environment {env_id} with episodes: {episode_ids}")
        for episode_index in episode_ids[:max_episodes]:
            episode_dff = env_dff[env_dff["episode_index"] == episode_index]
            agent_videos = []
            for agent_id in episode_dff["agent_id"].unique():
                agent_dff = episode_dff[episode_dff["agent_id"] == agent_id]

                video_file = animate_2d_umap(
                    agent_dff,
                    agent_id,
                    episode_index,
                    env_id,
                    output_dir,
                    file_prefix,
                    fps,
                )
                agent_videos.append(video_file)

            # Create a collage of the videos, accommodating up to 4 videos
            if len(agent_videos) > 0:  # Ensure there is at least one video
                create_collage(
                    env_id, episode_index, agent_videos, output_dir, file_prefix
                )


def animate_2d_umap(
    agent_dff, agent_id, episode_index, env_id, output_dir, file_prefix, fps
):
    """Creates a 2D animation from UMAP projections."""
    umap_result = agent_dff[["umap_1", "umap_2"]].values
    num_frames = len(umap_result)

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.set_title(f"Agent {agent_id}", fontsize=10)
    ax.scatter(umap_result[:, 0], umap_result[:, 1], color="grey", alpha=0.5, s=2)

    (line,) = ax.plot([], [], f"C{agent_id}o-", markersize=4)
    (trace,) = ax.plot([], [], f"C{agent_id}-", linewidth=1)

    def init():
        ax.set_xlim(np.min(umap_result[:, 0]) * 1.1, np.max(umap_result[:, 0]) * 1.1)
        ax.set_ylim(np.min(umap_result[:, 1]) * 1.1, np.max(umap_result[:, 1]) * 1.1)
        trace.set_data([], [])
        line.set_data([], [])
        return trace, line

    def update(frame):
        trace.set_data(
            umap_result[max(0, frame - 9) : frame + 1, 0],
            umap_result[max(0, frame - 9) : frame + 1, 1],
        )
        line.set_data([umap_result[frame, 0]], [umap_result[frame, 1]])
        print(
            f"\rAnimating env {env_id}, episode {episode_index}, agent {agent_id}: {frame + 1}/{num_frames} frames",
            end="",
        )
        return trace, line

    ani = FuncAnimation(
        fig, update, frames=np.arange(num_frames), init_func=init, blit=True
    )

    outfile = (
        f"{output_dir}/{file_prefix}{agent_id}_ep{episode_index}_env{env_id}_UMAP2D.mp4"
    )
    fig.tight_layout()
    ani.save(outfile, writer="ffmpeg", fps=fps)
    plt.close(fig)
    print(
        f"\nSaved 2D UMAP animation for agent {agent_id}, episode {episode_index}, env {env_id} to {outfile}"
    )
    return outfile


def _filter_all_ate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a subset of df containing only (env_id, episode_index) groups
    where *every* agent_id has at least one eating_event == True.
    """
    # Identify good (env_id, episode_index) pairs
    grp_keys = ["env_id", "episode_index"]
    good_pairs = []

    # Iterate groups
    for (env, ep), grp in df.groupby(grp_keys):
        total_agents = grp["agent_id"].nunique()
        ate_agents = grp.loc[grp["eating_event"], "agent_id"].nunique()
        if ate_agents == total_agents:
            good_pairs.append((env, ep))

    # Build mask to filter df
    mask = df.set_index(grp_keys).index.isin(good_pairs)
    return df[mask].reset_index(drop=True)


def plot_rnn_vs_arena_distances(dff):
    # Prepare lists to collect distances
    space_dists = []
    rnn_dists = []

    dff2 = _filter_all_ate(dff)
    print(
        "dff filtered to only episodes where all agents eat; shape:",
        dff2.shape,
        "dff.shape:",
        dff.shape,
    )

    # Group by env_id, episode_index, time_step
    group_cols = ["env_id", "episode_index", "time_step"]
    for _, grp in dff2.groupby(group_cols):
        # Stack positions (n_agents × 2)
        pos = np.stack(grp["position"].values)
        # Stack RNN states (n_agents × state_dim)
        rnn = np.stack(grp["rnn_states"].values)

        # Skip groups with fewer than 2 agents
        if pos.shape[0] < 2:
            continue

        # Compute pairwise Euclidean distances
        sd = pdist(pos, metric="euclidean")
        rd = pdist(rnn, metric="euclidean")

        space_dists.append(sd)
        rnn_dists.append(rd)

    # Concatenate all distances
    space_all = np.concatenate(space_dists)
    rnn_all = np.concatenate(rnn_dists)

    # Plain
    # Plot scatter of space vs. RNN distances
    # plt.figure(figsize=(6, 6))
    # plt.scatter(space_all, rnn_all, alpha=0.3)
    # plt.xlabel('Euclidean distance in space', fontsize=12)
    # plt.ylabel('Euclidean distance in RNN activity', fontsize=12)
    # plt.title('Space vs. RNN distance for each agent pair', fontsize=14)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    reg = linregress(space_all, rnn_all)
    slope, intercept, r_val, p_val, std_err = (
        reg.slope,
        reg.intercept,
        reg.rvalue,
        reg.pvalue,
        reg.stderr,
    )

    # Make the scatter + regression line
    plt.figure(figsize=(6, 6))
    plt.scatter(
        space_all,
        rnn_all,
        s=10,  # smaller points
        alpha=0.1,  # more transparent
        color="tab:blue",
        label="agent‐pair distances",
    )

    x_fit = np.linspace(space_all.min(), space_all.max(), 200)
    y_fit = intercept + slope * x_fit
    plt.plot(
        x_fit,
        y_fit,
        color="tab:red",
        linewidth=2,
        label=f"Fit: y={slope:.2f}x+{intercept:.2f}",
    )

    plt.text(
        0.05,
        0.05,
        f"p-value (slope) = {p_val:.2e}",
        transform=plt.gca().transAxes,
        # verticalalignment='top',
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5},
    )

    plt.xlabel("Euclidean distance in space", fontsize=12)
    plt.ylabel("Euclidean distance in RNN activity", fontsize=12)
    plt.title("Space vs. RNN distance for each agent pair", fontsize=14)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_behavior_and_rnn_state_trajectories(
    dff, episode_keys, colormap=None, arena_size_predefined=None
):
    for env_id, episode_idx in episode_keys:
        subset = dff[(dff["env_id"] == env_id) & (dff["episode_index"] == episode_idx)]
        if subset.empty:
            print(
                f"[Warning] Skipping missing episode: Env {env_id}, Episode {episode_idx}"
            )
            continue

        agent_ids = sorted(subset["agent_id"].unique())
        num_agents = len(agent_ids)
        # num_rows = 1 + math.ceil(num_agents / 2)
        num_rows = 1 if num_agents == 1 else 1 + math.ceil(num_agents / 2)

        fig, axes = plt.subplots(num_rows, 2, figsize=(10, 4 * num_rows))
        axes = np.array(axes).reshape(num_rows, 2)  # ensure 2D even for 1 row

        ax_pos = axes[0, 0]
        ax_pca = axes[0, 1]

        def get_agent_color(agent_id):
            if colormap:
                return colormap(agent_id % colormap.N)
            else:
                return f"C{agent_id % 10}"

        # Arena Trajectories
        for agent_id in agent_ids:
            agent_data = subset[subset["agent_id"] == agent_id].sort_values("time_step")
            positions = np.vstack(agent_data["position"].tolist())
            norm_times = agent_data["time_step"].values / agent_data["time_step"].max()
            color = get_agent_color(agent_id)

            for i in range(len(positions) - 1):
                ax_pos.plot(
                    positions[i : i + 2, 0],
                    positions[i : i + 2, 1],
                    color=color,
                    alpha=norm_times[i],
                    linewidth=1,
                )
            ax_pos.plot([], [], color=color, label=f"Agent {agent_id}")
        ax_pos.set_title(f"Arena Trajectory (Env {env_id}, Ep {episode_idx})")
        ax_pos.set_xlabel("X")
        ax_pos.set_ylabel("Y")
        if arena_size_predefined:
            ax_pos.set_xlim(0, arena_size_predefined[0])
            ax_pos.set_ylim(0, arena_size_predefined[1])
        ax_pos.grid(True)

        # Shared PCA of all RNN states
        episode_rnn_states = np.vstack(subset["rnn_states"].tolist())
        pca = PCA(n_components=2)
        rnn_pca = pca.fit_transform(episode_rnn_states)
        subset = subset.copy()
        subset["rnn_pca_x"] = rnn_pca[:, 0]
        subset["rnn_pca_y"] = rnn_pca[:, 1]
        var_exp = pca.explained_variance_ratio_
        pc1_label = f"PC1 ({var_exp[0]*100:.1f}%)"
        pc2_label = f"PC2 ({var_exp[1]*100:.1f}%)"

        for agent_id in agent_ids:
            agent_data = subset[subset["agent_id"] == agent_id].sort_values("time_step")
            x = agent_data["rnn_pca_x"].values
            y = agent_data["rnn_pca_y"].values
            norm_times = agent_data["time_step"].values / agent_data["time_step"].max()
            color = get_agent_color(agent_id)

            for i in range(len(x) - 1):
                ax_pca.plot(
                    x[i : i + 2],
                    y[i : i + 2],
                    color=color,
                    alpha=norm_times[i],
                    linewidth=1,
                )
            ax_pca.plot([], [], color=color, label=f"Agent {agent_id}")
        ax_pca.set_title("Shared RNN PCA")
        ax_pca.set_xlabel(pc1_label)
        ax_pca.set_ylabel(pc2_label)
        ax_pca.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax_pca.grid(True)

        # Individual agent PCA plots
        # if only one agent, no need for second row
        if num_agents > 1:
            for i, agent_id in enumerate(agent_ids):
                row = 1 + i // 2
                col = i % 2
                ax = axes[row, col]

                agent_data = subset[subset["agent_id"] == agent_id].sort_values(
                    "time_step"
                )
                rnn = np.vstack(agent_data["rnn_states"].tolist())
                pca_ind = PCA(n_components=2)
                ind_pca = pca_ind.fit_transform(rnn)
                var_exp = pca_ind.explained_variance_ratio_
                norm_times = (
                    agent_data["time_step"].values / agent_data["time_step"].max()
                )
                color = get_agent_color(agent_id)

                for j in range(len(ind_pca) - 1):
                    ax.plot(
                        ind_pca[j : j + 2, 0],
                        ind_pca[j : j + 2, 1],
                        color=color,
                        alpha=norm_times[j],
                        linewidth=1,
                    )
                ax.set_title(f"Agent {agent_id} Individual PCA")
                ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
                ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
                ax.grid(True)

            # Hide unused subplots
            for i in range(num_agents, (num_rows - 1) * 2):
                row = 1 + i // 2
                col = i % 2
                fig.delaxes(axes[row, col])

        plt.suptitle(f"Env {env_id}, Episode {episode_idx}")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

###### CLUSTERING ######
def plot_rnn_clusters_spectral(dff_agent, dff, agent_id, pca):
    # Cluster embeddings + plot PCA -- Spectral Clustering
    from sklearn.cluster import SpectralClustering

    # All agents
    rnn_states_all_agents = dff_agent["rnn_states"].tolist()
    rnn_states_3d = pca.transform(rnn_states_all_agents)[:, :3]

    # Single agent
    dff_agent = dff[dff["agent_id"] == agent_id]
    rnn_states_agent = dff_agent["rnn_states"].tolist()
    rnn_states_3d = pca.transform(rnn_states_agent)[:, :3]

    n_clusters = 3  # Define the number of clusters
    spectral = SpectralClustering(
        n_clusters=n_clusters, affinity="nearest_neighbors", random_state=42
    )
    labels = spectral.fit_predict(rnn_states_3d)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for cluster in range(n_clusters):
        cluster_points = rnn_states_3d[labels == cluster]
        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            cluster_points[:, 2],
            label=f"Cluster {cluster}",
            s=1,
        )

    ax.set_title("3D PCA Embeddings with Spectral Clustering")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_zlabel("PCA Component 3")
    ax.legend()
    plt.show()


###### UMAP ######
def get_umap_embedding(dff, force_rerun = True, umap_embedding_file = "./umap_embedding.npy", do_plot=True):
    rnn_states_all_agents = np.squeeze(np.array(dff["rnn_states"].tolist()))
    # rnn_states_scaled = StandardScaler().fit_transform(rnn_states_all_agents)
    rnn_states_scaled = rnn_states_all_agents

    # Check if UMAP embedding already exists and handle force_rerun option
    # if True re-run UMAP fitting    
    if (
        umap_embedding_file is not None
        and os.path.exists(umap_embedding_file)
        and not force_rerun
    ):
        print(f"Loading UMAP embedding from {umap_embedding_file}")
        umap_embedding = np.load(umap_embedding_file)
    else:
        print("Fitting UMAP model to RNN-states matrix of shape:", rnn_states_scaled.shape)
        start_time = time.time()
        umap_model = umap.UMAP(
            # n_neighbors=200,
            # min_dist=0.1,
            n_components=2,
            random_state=42,
        )
        umap_embedding = umap_model.fit_transform(rnn_states_scaled)
        end_time = time.time()
        print(
            f"UMAP fitting completed in {end_time - start_time:.2f} seconds. "
            f"UMAP embedding shape: {umap_embedding.shape}"
        )

        # Save UMAP embedding to file
        np.save(umap_embedding_file, umap_embedding)
        print(f"UMAP embedding saved to {umap_embedding_file}")

    # Plot the UMAP projection
    if do_plot:
        # plt.figure(figsize=(5, 4))
        plt.figure(figsize=(10, 7))
        plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], s=1, alpha=0.3)
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.title("UMAP Projection of RNN States")
        plt.show()
    
    return umap_embedding

def plot_bool_features_umap(dff, umap_embedding, out_dir, bool_features):
    plt.figure(figsize=(10, 7))
    base_color = matplotlib.colors.to_rgba("lightgrey", 0.3)

    # Plot all as grey first
    plt.scatter(
        umap_embedding[:, 0],
        umap_embedding[:, 1],
        s=1,
        c=[base_color] * len(dff),
        label="_nolegend_",
    )

    # Overlay each bool feature where it's True
    for i, feature in enumerate(bool_features):
        mask = dff[feature] == True
        if mask.sum() == 0:
            print(f"Skipping feature '{feature}' as it has no True values.")
            continue  # skip if feature has no True values
        color = matplotlib.cm.tab10(i % 10)  # cycle through 10 distinct colors
        plt.scatter(
            umap_embedding[mask, 0],
            umap_embedding[mask, 1],
            s=2,
            c=[color],
            label=feature,
        )

    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.title("UMAP Colored by Boolean Features")
    plt.legend(markerscale=4)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "bool_features.pdf"), dpi=300)
    plt.show()
    # plt.close()


def plot_feature_umap(dff, umap_embedding, feature, out_dir, config, show=False):
    os.makedirs(out_dir, exist_ok=True)

    values = dff[feature]
    valid = values.notna()
    if valid.sum() == 0:
        print(f"Skipping {feature}: no valid values")
        return

    # Get color configs
    values = values.copy()
    mode = config.get("color_mode", "default")
    cmap = config.get("cmap", "viridis")
    if mode == "inverted":
        pass
        # values[valid] = 1.0 / (1.0 + values[valid])
    elif mode == "circular":
        cmap = "twilight_shifted" if "cmap" not in config else config["cmap"]
        maxval = np.nanmax(np.abs(values))
        vmin = -maxval
        vmax = maxval
    elif mode == "probability":
        cmap = "Blues"
    elif mode == "divergent":
        maxval = np.nanmax(np.abs(values))
        vmin = -maxval
        vmax = maxval
        cmap = "coolwarm"
    # Overrides
    vmin = config.get("vmin", None)
    vmax = config.get("vmax", None)
    if "time_since_" in feature:  # Override for time_since features
        vmin = 0
        vmax = 30
    if "distance_to_" in feature:  # Override for time_since features
        vmin = 0
        vmax = 20

    plt.figure(figsize=(10, 7))
    plt.scatter(
        umap_embedding[~valid, 0],
        umap_embedding[~valid, 1],
        s=1,
        c="lightgrey",
        alpha=0.3,
    )

    sc = plt.scatter(
        umap_embedding[valid, 0],
        umap_embedding[valid, 1],
        s=2,
        c=values[valid],
        cmap=cmap,
        alpha=0.8,
        norm=(
            Normalize(vmin=vmin, vmax=vmax)
            if vmin is not None and vmax is not None
            else None
        ),
    )
    plt.colorbar(sc, label=feature)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.title(f"UMAP Colored by {feature}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{feature}.pdf"))
    if show:
        plt.show()
    plt.close()

###### RNN VELOCITY RELATED ######
# TODO: move more here



def run_rnn_other_report(dff, outfile_base, to_run=["pca_pooled_agent"], plot_only=False):
    if not plot_only and "rnn_states" not in dff.columns:
        print("[rnn_other] skipped: no rnn_states column")
        return

    if "pca_pooled_agent" in to_run:
        pca_path = outfile_base + "_rnn_pca_common"
        if plot_only:
            pca_common = _load_data(pca_path)
        else:
            all_rnn_states = np.vstack(dff["rnn_states"].tolist())
            pca_common = PCA()
            pca_common.fit_transform(all_rnn_states)
            _save_data(pca_common, pca_path)

        plot_cumulative_variance(pca_common, title="", outfile_base=outfile_base)
        if not plot_only:
            plot_all_agents_projected_2d(dff, pca_common, outfile_base=outfile_base)
        # plot_all_agents_projected_3d(dff, pca_common)    

 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run report with flat pkl data file")
    parser.add_argument("--flat_pkl_file", type=str, default=None, help="Path to flattened pickle file")
    args = parser.parse_args()
    flat_pkl_file = args.flat_pkl_file

    if flat_pkl_file is not None and os.path.isfile(flat_pkl_file):
        print(f"Loading: {flat_pkl_file} ...")
    else:
        print("No matching file found.")
        flat_pkl_file = None

    if flat_pkl_file is not None:
        start_time = time.time()
        dff = pd.read_pickle(flat_pkl_file)
        end_time = time.time()
        print(f"Loaded in {end_time - start_time:.2f} seconds.")
        print(f"Columns in DataFrame: {dff.columns.tolist()}")

        output_folder = flat_pkl_file.replace("_flattened.pkl", "_rnn_psd_report/")
        print(f"Output folder: {output_folder}")    
        os.makedirs(output_folder, exist_ok=True)

        run_rnn_other_report(dff, output_folder)


# ── GPU / SVD backend ────────────────────────────────────────────────────────
try:
    import torch as _torch
    _DEVICE = _torch.device("cuda") if _torch.cuda.is_available() else None
except ImportError:
    _DEVICE = None

SVD_BACKEND = "GPU (torch)" if _DEVICE is not None else "CPU (numpy)"


def svd_evr(X: np.ndarray) -> np.ndarray:
    """
    Explained-variance ratios for all rank-feasible singular values of X.

    Centers X, then calls torch.linalg.svdvals on CUDA if available,
    otherwise falls back to numpy.linalg.svd.  Returns a float32 array
    of length min(n_samples, n_features).
    """
    X_c = X - X.mean(axis=0, keepdims=True)
    if _DEVICE is not None:
        X_t = _torch.from_numpy(X_c.astype(np.float32)).to(_DEVICE)
        S = _torch.linalg.svdvals(X_t).cpu().numpy()
    else:
        S = np.linalg.svd(X_c.astype(np.float32), compute_uv=False)
    ev = S ** 2
    total = ev.sum()
    return (ev / total).astype(np.float32) if total > 0 else ev.astype(np.float32)
