from ast import Not
import os
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import numpy as np
from itertools import combinations
from scipy.optimize import curve_fit

from utils_figstyle import set_nature_style
from utils_figsaving import _make_fig, _save_data, _load_data, save_fig
set_nature_style()

# Zero safe Theil index 
def theil_index(consumptions):
    consumptions = np.asarray(consumptions, dtype=float)
    if consumptions.size == 0:
        return 0.0

    if not np.all(np.isfinite(consumptions)):
        raise ValueError("consumptions must be finite")
    if np.any(consumptions < 0):
        raise ValueError("consumptions must be nonnegative")

    mean = consumptions.mean()
    if mean == 0:
        return 0.0

    ratio = consumptions / mean
    return float(np.mean(np.where(ratio > 0, ratio * np.log(ratio), 0.0)))


def calculate_theil_index_last(dff):
    """
    Calculate the Theil index for the last time step in each (env_id, episode_index).
    """
    # Filter eating events
    eating_events = dff[dff["eating_event"]].copy()

    # Calculate cumulative food consumed
    eating_events["cumulative_food"] = (
        eating_events.groupby(["env_id", "episode_index", "agent_id"]).cumcount() + 1
    )

    # Calculate Theil index at the last time step for each (env_id, episode_idx)
    results = []

    for (env_id, episode_index), subset in eating_events.groupby(
        ["env_id", "episode_index"]
    ):
        # Get the last time step
        last_time_step = subset["time_step"].max()
        # print(f"Env {env_id}, Episode {episode_index}, Last Time Step: {last_time_step}")
        subset_at_last_time = subset[subset["time_step"] <= last_time_step]
        consumptions = (
            subset_at_last_time.groupby("agent_id")["cumulative_food"].max().values
        )
        # print(consumptions)
        theil = theil_index(consumptions)  # Custom function to calculate Theil index

        results.append(
            {
                "env_id": env_id,
                "episode_index": episode_index,
                "time_step": last_time_step,
                "theil_index": theil,
            }
        )

    theil_df = pd.DataFrame(results)
    theil_df.sort_values(by=["env_id", "episode_index", "time_step"], inplace=True)

    # Calculate aggregated Theil index over all data
    aggregated_consumptions = (
        eating_events.groupby("agent_id")["cumulative_food"].max().values
    )
    aggregated_theil = theil_index(aggregated_consumptions)

    # Return the Theil index DataFrame and the aggregated Theil index
    return theil_df, aggregated_theil

def calculate_theil_indexes(dff):
    # Filter eating events
    eating_events = dff[dff["eating_event"]].copy()

    # Calculate cumulative food consumed
    eating_events["cumulative_food"] = (
        eating_events.groupby(["env_id", "episode_index", "agent_id"]).cumcount() + 1
    )

    # Calculate Theil index over time for each (env_id, episode_idx)
    results = []

    for (env_id, episode_index), subset in eating_events.groupby(
        ["env_id", "episode_index"]
    ):
        time_steps = subset["time_step"].unique()
        for time_step in time_steps:
            subset_at_time = subset[subset["time_step"] <= time_step]
            consumptions = (
                subset_at_time.groupby("agent_id")["cumulative_food"].max().values
            )
            theil = theil_index(consumptions)
            results.append(
                {
                    "env_id": env_id,
                    "episode_index": episode_index,
                    "time_step": time_step,
                    "theil_index": theil,
                }
            )

    theil_df = pd.DataFrame(results)
    theil_df.sort_values(by=["env_id", "episode_index", "time_step"], inplace=True)

    # Calculate aggregated Theil index over all data
    aggregated_consumptions = (
        eating_events.groupby("agent_id")["cumulative_food"].max().values
    )
    aggregated_theil = theil_index(aggregated_consumptions)

    # Plot Theil index over time for each (env_id, episode_idx)
    plt.figure(figsize=(6, 3))
    for (env_id, episode_index), subset in theil_df.groupby(
        ["env_id", "episode_index"]
    ):
        plt.plot(
            subset["time_step"],
            subset["theil_index"],
            label=f"Env {env_id} Ep {episode_index}",
        )

    plt.axhline(
        y=aggregated_theil,
        color="black",
        linestyle="--",
        label="Aggregated Theil Index",
    )
    plt.xlabel("Time Step")
    plt.ylabel("Theil Index")
    plt.xlim(0, theil_df["time_step"].max())
    plt.title("Theil Index Over Time for Each (env_id, episode_index)")
    # plt.legend()
    plt.show()

    print(f"Aggregated Theil Index: {aggregated_theil}")

    return theil_df, aggregated_theil


def calculate_gini(x: np.ndarray, eps: float = 1e-12) -> float:
    """
    Gini coefficient for a 1D array of nonnegative values.
    Returns np.nan if undefined (e.g., empty or all zeros after cleaning).
    Notes:
      - If negatives exist, we shift the distribution up so min is 0.
      - Works fine with zeros.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan

    # Shift if any negatives (shouldn't happen for food eaten, but size could be weird)
    xmin = x.min()
    if xmin < 0:
        x = x - xmin

    s = x.sum()
    if s <= eps:
        return 0.0  # perfectly equal at zero mass (all zeros)

    # Standard efficient Gini using sorted values
    x_sorted = np.sort(x)
    n = x_sorted.size
    idx = np.arange(1, n + 1)
    # G = (2*sum(i*x_i)/(n*sum(x))) - (n+1)/n
    return (2.0 * np.sum(idx * x_sorted) / (n * s)) - (n + 1.0) / n


def calculate_polarization(orientations):
    """
    Calculate the polarization of a group of orientations.

    Polarization is a measure of how aligned the orientations are within a group.
    It is calculated as the mean cosine of the difference between each orientation
    and the mean orientation of the group.

    Parameters:
    orientations (array-like): A list or array of orientation angles in radians.

    Returns:
    float: The polarization value, ranging from -1 (completely anti-aligned)
           to 1 (completely aligned).
    """
    orientations = np.array(orientations)
    if orientations.size <= 2:
        raise ValueError("Polarization works best with more than 2 agents...")

    mean_orientation = np.arctan2(
        np.mean(np.sin(orientations)), np.mean(np.cos(orientations))
    )
    return np.mean(np.cos(orientations - mean_orientation))




def calculate_cohesion(positions):
    """
    Calculate the cohesion of a group of positions.

    Cohesion is measured as the average nearest neighbor distance within a group of positions.
    It indicates how close the individuals are to each other.

    Parameters:
    positions (array-like): A list or array of positions, where each position is
                            a list or array of coordinates [x, y].

    Returns:
    float: The average nearest neighbor distance. Returns NaN if there are
           fewer than 2 positions.
    """
    positions = np.array([pos for pos in positions])
    num_positions = len(positions)
    if num_positions < 2:
        return np.nan
    distances = np.linalg.norm(positions[:, np.newaxis] - positions, axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest_distances = np.min(distances, axis=1)
    return np.mean(nearest_distances)


def calculate_cohesion_metrics(positions):
    """
    Calculate the cohesion, minimum, and maximum nearest-neighbor distances for a group of positions.

    Cohesion is measured as the average nearest neighbor distance within a group of positions.
    It indicates how close the individuals are to each other.

    Parameters:
    positions (array-like): A list or array of positions, where each position is
                            a list or array of coordinates [x, y].

    Returns:
    tuple: A tuple containing the average, minimum, and maximum nearest neighbor distances.
           Returns (NaN, NaN, NaN) if there are fewer than 2 positions.
    """
    positions = np.array([pos for pos in positions])
    num_positions = len(positions)
    if num_positions < 2:
        return np.nan, np.nan, np.nan

    # Calculate pairwise distances
    distances = np.linalg.norm(positions[:, np.newaxis] - positions, axis=2)
    np.fill_diagonal(
        distances, np.inf
    )  # Ignore self-distances by setting diagonal to infinity

    # Nearest neighbor distances
    nearest_distances = np.min(distances, axis=1)

    # Return average (cohesion), minimum, and maximum of nearest neighbor distances
    return (
        np.mean(nearest_distances),
        np.min(nearest_distances),
        np.max(nearest_distances),
    )


def calculate_fluctuation_ratio(displacements):
    """
    Calculate the fluctuation ratio r = <x^2> / <x>^2.

    Parameters:
    displacements (array-like): List of x displacements (or y, depending on choice).

    Returns:
    float: Fluctuation ratio
    """
    displacements = np.array(displacements)
    mean_x = np.mean(displacements)
    mean_x2 = np.mean(displacements**2)
    if mean_x == 0:
        return np.nan
    return mean_x2 / (mean_x**2)


def calculate_hurst_exponent(ts):
    """
    Estimate the Hurst exponent of a time series.

    Parameters:
    ts (array-like): The time series (e.g., angle positions)

    Returns:
    float: Hurst exponent estimate

    """
    ts = np.array(ts)
    N = len(ts)
    if N < 20:
        return np.nan  # Not enough data

    T = np.arange(1, N + 1)
    Y = np.cumsum(ts - np.mean(ts))
    R = np.max(Y) - np.min(Y)
    S = np.std(ts)
    if S == 0:
        return 0.0
    return np.log(R / S) / np.log(N)


############ TRAJECTORY ANALYSIS ############


def plot_agent_trajectories_for_specific_episode(dff, env_id, episode_idx):
    subset = dff[(dff["env_id"] == env_id) & (dff["episode_index"] == episode_idx)]
    agent_ids = subset["agent_id"].unique()

    fig, axes = plt.subplots(1, len(agent_ids), figsize=(12, 3))
    for i, agent_id in enumerate(agent_ids):
        agent_data = subset[subset["agent_id"] == agent_id]
        positions = np.array(agent_data["position"].tolist())

        # Plotting trajectories
        ax = axes[i]
        ax.plot(positions[:, 0], positions[:, 1], label=f"Agent {agent_id}")

        ax.set_title(f"Env {env_id}, Episode {episode_idx}, Agent {agent_id}")
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # remove ticks and ticklabels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    #         ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


def plot_agent_trajectories_with_eating_events(dff, env_id=0, episode_idx=0):
    subset = dff[
        (dff["env_id"] == env_id) & (dff["episode_index"] == episode_idx)
    ]
    plt.figure(figsize=(4, 4))
    agent_ids = subset["agent_id"].unique()
    for agent_id in agent_ids:
        agent_data = subset[subset["agent_id"] == agent_id]
        positions = np.array(agent_data["position"].tolist())
        plt.plot(positions[:, 0], positions[:, 1], label=f"Agent {agent_id}")
        eating_positions = np.array(
            agent_data[agent_data["eating_event"]]["position"].tolist()
        )
        if len(eating_positions) > 0:
            plt.scatter(
                eating_positions[:, 0],
                eating_positions[:, 1],
                c="red",
                marker="x",
                label=f"Eating" if agent_id == agent_ids[-1] else None,
            )
    plt.title(f"Env {env_id}, Episode {episode_idx}")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.show()




def bin_agent_size(dff, num_bins=3, bin_column='size_bin'):    
    edges = np.linspace(dff['agent_size'].min(),
                        dff['agent_size'].max(),
                        num=4)
    bin_labels = [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(num_bins)]
    dff[bin_column] = pd.cut(dff['agent_size'],
                            bins=edges,
                            labels=bin_labels,
                            include_lowest=True)
    return dff, bin_labels


# ——— Bin agent_size into 3 equal‐width intervals ———
def plot_behavior_densities_1d(dff, bin_column='size_bin'):
    fig, axes = plt.subplots(1, 2, figsize=(5, 3), sharey=False)
    bin_labels = dff[bin_column].cat.categories
    for ax, col in zip(axes, ['move_forward', 'turn_angle']):
        for label in bin_labels:
            subset = dff.loc[dff[bin_column] == label, col]
            sns.kdeplot(
                subset,
                ax=ax,
                bw_adjust=1.0,
                fill=False,
                clip=(subset.min(), subset.max()),
                label=label
            )
        ax.set_xlabel(col.replace('_', ' ').title(), fontsize=12)
        ax.set_yticks([])  # no y‐axis ticks
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.8)
        if ax is axes[1]:
            ax.legend(title='agent_size bin', loc='upper right')

    plt.tight_layout()
    plt.show()


####### 

def plot_ethogram(
    dff, env_id, episode_index, agents,
    ema_window=10, plot_eods=True, figsize=(6, 2.5),
    output_dir="./", output_str="", outfile_base=None, max_time_steps=None,
    plot_only=False,
):
    # --- Priority and colors (always needed for plotting) ---
    priority = ['was_bitten', 'bite_other_fish', 'eating_event', 'food_observed', 'has_nearby']
    colors = {
        'was_bitten': '#E53935',       # red
        'bite_other_fish': '#7B3294',  # purple
        'eating_event': '#4CAF50',     # green
        'food_observed': '#FFF59D',    # pale yellow
        'has_nearby': '#FFB74D',       # soft orange
        'none': '#EEEEEE'              # light gray background
    }

    names_dict = {
        'was_bitten': 'Bitten',
        'bite_other_fish': 'Biting',
        'eating_event': 'Eating',
        'food_observed': 'Food Observed',
        'has_nearby': 'Agent Nearby',
    }

    data_path = outfile_base + f"_ethogram_env{env_id}_ep{episode_index}_data" if outfile_base else None

    if plot_only and data_path:
        saved      = _load_data(data_path)
        ethogram   = saved["ethogram"]
        ema_arrays = saved["ema_arrays"]
        time_steps = np.array(saved["time_steps"])
        agents     = saved["agents"]
        num_agents = len(agents)
    else:
        rollout_df = dff[
            (dff['env_id'] == env_id) &
            (dff['episode_index'] == episode_index)
        ]

        min_ts = rollout_df['time_step'].min()
        max_ts_actual = rollout_df['time_step'].max()
        if max_time_steps is not None:
            max_ts = min(max_ts_actual, min_ts + max_time_steps - 1)
        else:
            max_ts = max_ts_actual
        time_steps = np.arange(min_ts, max_ts + 1)

        num_agents = len(agents)
        color_to_idx = {k: i + 1 for i, k in enumerate(priority)}
        ethogram = np.zeros((num_agents, len(time_steps)), dtype=int)
        ema_arrays = []

        for ai, agent_id in enumerate(agents):
            agent_df = rollout_df[rollout_df['agent_id'] == agent_id].sort_values("time_step").set_index("time_step")
            emit_eod_series = []
            for ti, ts in enumerate(time_steps):
                row = agent_df.loc[ts] if ts in agent_df.index else None
                val = 0
                if row is not None:
                    for s in priority:
                        if row[s]:
                            val = color_to_idx[s]
                            break
                    emit_eod_series.append(float(row['emit_eod']))
                else:
                    emit_eod_series.append(np.nan)
                ethogram[ai, ti] = val
            if plot_eods:
                ema_series = pd.Series(emit_eod_series).fillna(method='ffill').fillna(0).ewm(span=ema_window, adjust=False).mean()
                ema_arrays.append(ema_series.values)

        if data_path:
            _save_data(
                {"ethogram": ethogram, "ema_arrays": ema_arrays, "time_steps": time_steps, "agents": list(agents)},
                data_path,
            )

    color_to_idx = {k: i + 1 for i, k in enumerate(priority)}
    idx_to_color = {0: colors['none']}
    for k, v in color_to_idx.items():
        idx_to_color[v] = colors[k]

    fig, ax = plt.subplots(figsize=figsize)

    # Plot ethogram
    cmap = mcolors.ListedColormap([idx_to_color[i] for i in range(len(idx_to_color))])
    ax.imshow(
        ethogram,
        aspect='auto',
        cmap=cmap,
        interpolation='nearest',
        extent=[time_steps[0], time_steps[-1] + 1, -0.5, num_agents - 0.5]
    )

    # Plot EMA overlays (optional)
    if plot_eods:
        norm = plt.Normalize(0, 1)
        blues = plt.cm.Blues
        for ai, ema_row in enumerate(ema_arrays):
            for ti, ts in enumerate(time_steps):
                c = blues(norm(ema_row[ti]))
                ax.add_patch(plt.Rectangle(
                    (ts, ai + 0.4), 1, 0.1,  # Small thin stripe below main row
                    color=c, linewidth=0
                ))

    # Formatting
    ax.set_yticks(range(num_agents))
    ax.set_yticklabels([f'A{a + 1}' for a in agents], fontsize=6)
    ax.set_xticks([])
    ax.tick_params(axis='y', labelsize=6)
    ax.invert_yaxis()

    for i in range(num_agents):
        ax.axhline(i + 0.5, color='white', linewidth=0.5)

    # # Legend (excluding emit_eod, which is now a heatmap, not discrete)
    # patches = [mpatches.Patch(color=colors[k], label=k.replace('_', ' ')) for k in priority]
    # ax.legend(
    #     handles=patches,
    #     loc='upper center',
    #     bbox_to_anchor=(0.5, -0.15),
    #     ncol=3,
    #     fontsize=8,
    #     frameon=False
    # )

    # Legend categorical patches
    patches = [mpatches.Patch(color=colors[k], label=k.replace('_', ' ')) for k in priority]

    # Create a proxy ScalarMappable for the emit_eod colormap
    sm = cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])  # required for ScalarMappable in legend

    # Plot legend including emit_eod heatmap proxy
    handles = patches + [sm]
    # labels = [k.replace('_', ' ') for k in priority] + ['emit eod (EMA)']
    labels = [names_dict[label] for label in priority] + ['emit eod (EMA)']

    ax.legend(
        handles=handles,
        labels=labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0),
        ncol=5,             # adjust if needed
        fontsize=8,
        frameon=False
    )

    # Add colorbar for emit_eod (EMA)
    # if plot_eods:
    #     cbar_ax = fig.add_axes([0.4, -0.25, 0.2, 0.03])  # [left, bottom, width, height]
    #     sm = cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=1))
    #     cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    #     cbar.set_label('emit eod (EMA)', fontsize=8)
    #     cbar.ax.tick_params(labelsize=6)

    plt.tight_layout()
    if outfile_base is not None:
        base = outfile_base + f"_ethogram_env{env_id}_ep{episode_index}"
    elif output_dir and output_str:
        base = f"{output_dir}/ethogram_{output_str}"
    else:
        print("Warning: No output path provided, showing but not saving ethogram.")
        plt.show()

    save_fig(plt.gcf(), base)
    print(f"Saved ethogram to {base}.png")

# def plot_ethogram_old(dff, desired_env_id = 0, desired_episode_index = 0, agents = [0, 1, 2, 3]):
#     # --- Priority and colors ---
#     priority = ['was_bitten', 'bite_other_fish', 'eating_event', 'food_observed', 'has_nearby']
#     colors = {
#         'was_bitten': '#E53935',       # red
#         'bite_other_fish': '#7B3294',  # purple
#         'eating_event': '#4CAF50',     # green
#         'food_observed': '#FFF59D',    # pale yellow
#         'has_nearby': '#FFB74D',       # soft orange
#         'emit_eod': '#2196F3',         # blue
#         'none': '#EEEEEE'              # light gray background
#     }

#     rollout_df = dff[
#         (dff['env_id'] == desired_env_id) &
#         (dff['episode_index'] == desired_episode_index)
#     ]

#     time_steps = np.arange(rollout_df['time_step'].min(), rollout_df['time_step'].max()+1)
#     num_agents = len(agents)
#     ethogram = np.zeros((num_agents, len(time_steps)), dtype=int)

#     color_to_idx = {k: i+1 for i, k in enumerate(priority)}
#     idx_to_color = {0: colors['none']}
#     for k, v in color_to_idx.items():
#         idx_to_color[v] = colors[k]

#     for ai, agent_id in enumerate(agents):
#         agent_df = rollout_df[rollout_df['agent_id'] == agent_id].sort_values("time_step").set_index("time_step")
#         for ti, ts in enumerate(time_steps):
#             row = agent_df.loc[ts] if ts in agent_df.index else None
#             if row is not None:
#                 val = 0
#                 for s in priority:
#                     if row[s]:
#                         val = color_to_idx[s]
#                         break
#                 ethogram[ai, ti] = val

#     cmap = plt.cm.colors.ListedColormap([idx_to_color[i] for i in range(len(idx_to_color))])
#     fig, ax = plt.subplots(figsize=(6, 2.5))  # WIDTH = 3 inches!

#     # Ethogram base
#     ax.imshow(
#         ethogram,
#         aspect='auto',
#         cmap=cmap,
#         interpolation='nearest',
#         extent=[time_steps[0], time_steps[-1]+1, -0.5, num_agents-0.5]
#     )

#     # Plot emit_eod thin rows
#     for ai, agent_id in enumerate(agents):
#         agent_df = rollout_df[rollout_df['agent_id'] == agent_id].sort_values("time_step").set_index("time_step")
#         emit_eod_times = []
#         for ts in time_steps:
#             row = agent_df.loc[ts] if ts in agent_df.index else None
#             if row is not None and row['emit_eod']:
#                 emit_eod_times.append(ts)
        
#         # Plot very thin rectangle or markers at y=ai + small offset
#         ax.fill_between(
#             emit_eod_times,
#             ai + 0.5 - 0.05,  # bottom
#             ai + 0.5 + 0.05,  # top
#             color=colors['emit_eod'],
#             linewidth=0
#         )

#     # Same axis formatting
#     ax.set_yticks(range(num_agents))
#     ax.set_yticklabels([f'Agent {a}' for a in agents], fontsize=6)
#     ax.set_xticks([])
#     ax.tick_params(axis='y', labelsize=6)
#     ax.invert_yaxis()

#     # Optional subtle horizontal lines
#     for i in range(num_agents):
#         ax.axhline(i+0.5, color='white', linewidth=0.5)

#     # Legend including emit_eod patch
#     patches = [mpatches.Patch(color=colors[k], label=k.replace('_', ' ')) for k in priority]
#     patches.append(mpatches.Patch(color=colors['emit_eod'], label='emit eod'))

#     ax.legend(
#         handles=patches,
#         loc='upper center',
#         bbox_to_anchor=(0.5, -0.15),
#         ncol=4,
#         fontsize=8,
#         frameon=False
#     )

#     plt.tight_layout()
#     plt.show()


def plot_agent_trajectories_side_by_side(dff, env_id, episode_idx, output_dir="./", output_str="", outfile_base=None):
    subset = dff[(dff["env_id"] == env_id) & (dff["episode_index"] == episode_idx)]
    agent_ids = subset["agent_id"].unique()

    fig, axes = plt.subplots(1, len(agent_ids), figsize=(12, 3))
    for i, agent_id in enumerate(agent_ids):
        agent_data = subset[subset["agent_id"] == agent_id]
        positions = np.array(agent_data["position"].tolist())

        # Plotting trajectories
        ax = axes[i]
        ax.plot(positions[:, 0], positions[:, 1], label=f"Agent {agent_id}")

        ax.set_title(f"Env {env_id}, Episode {episode_idx}, Agent {agent_id}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
    #         ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    if outfile_base is not None:
        base = outfile_base + f"_agent_trajectories_wide_env{env_id}_episode{episode_idx}"
    elif output_dir and output_str:
        base = f"{output_dir}/agent_trajectories_wide_env{env_id}_episode{episode_idx}_{output_str}"
    else:
        print("Warning: No output path provided, showing but not saving agent trajectories.")
        plt.show()
    save_fig(fig, base)
    print(f"Saved agent trajectories (wide) to {base}.png")


def plot_agent_trajectories_together(dff, env_id=0, episode_idx=0,
                                     mark_eating=True,
                                     mark_bitten=True,
                                     output_dir="./", output_str="", outfile_base=None, target_location=None,
                                     plot_only=False):
    data_path = outfile_base + f"_agent_exploration_trajectories_env{env_id}_ep{episode_idx}_data" if outfile_base else None

    if plot_only and data_path:
        saved = _load_data(data_path)
        arena_size = tuple(saved["arena_size"])
        agent_ids = saved["agent_ids"]
        traj_data = saved
    else:
        subset = dff[
            (dff["env_id"] == env_id) & (dff["episode_index"] == episode_idx)
        ]
        arena_size = subset[subset['arena_size'].notna()]['arena_size'].iloc[0]
        if len(subset[subset['arena_size'].notna()]) > dff['agent_id'].nunique():
            print("Warning: Multiple arena_size entries found; using the first one and trimming subset.")
            cutoff_time = subset[subset['arena_size'].notna()]['time_step'].iloc[1]
            subset = subset[subset['time_step'] < cutoff_time]
        agent_ids = sorted(subset["agent_id"].unique())
        bitten_col = "bitten" if "bitten" in subset.columns else "was_bitten"
        traj_data = {"arena_size": list(arena_size), "agent_ids": agent_ids}
        for aid in agent_ids:
            ad = subset[subset["agent_id"] == aid]
            traj_data[f"pos_{aid}"] = np.array(ad["position"].tolist())
            traj_data[f"eat_{aid}"] = np.array(ad[ad["eating_event"]]["position"].tolist()) if ad["eating_event"].any() else np.empty((0, 2))
            traj_data[f"bit_{aid}"] = np.array(ad[ad[bitten_col]]["position"].tolist()) if bitten_col in ad.columns and ad[bitten_col].any() else np.empty((0, 2))
        if data_path:
            _save_data(traj_data, data_path)

    fig, ax = _make_fig(1, 1)
    for agent_id in agent_ids:
        positions = np.array(traj_data[f"pos_{agent_id}"])
        plt.plot(positions[:, 0], positions[:, 1], label=f"A{agent_id + 1}")

        if mark_eating:
            eating_positions = np.array(traj_data[f"eat_{agent_id}"])
            if len(eating_positions) > 0:
                plt.scatter(
                eating_positions[:, 0],
                eating_positions[:, 1],
                marker="o",
                facecolors="none",
                edgecolors="black",
                label=f"Eating" if agent_id == agent_ids[-1] else None,
            )

        if mark_bitten:
            bitten_positions = np.array(traj_data[f"bit_{agent_id}"])
            if len(bitten_positions) > 0:
                    plt.scatter(
                        bitten_positions[:, 0],
                        bitten_positions[:, 1],
                        c="red",
                        marker="x",
                        label="Bitten" if agent_id == agent_ids[-1] else None,
                    )
    # Helper for homing trials
    if target_location is not None:
        plt.scatter(
            target_location[0],
            target_location[1],
            c="green",
            marker="*",
            s=50,
            label="Target"
        )
    # plt.title(f"Env {env_id}, Episode {episode_idx}")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.xlim(0, arena_size[0])
    plt.ylim(0, arena_size[1])
    # plt.axis("scaled")

    ax.legend(
        loc='upper center',
        # bbox_to_anchor=(0.5, -0.2),  # used if you have "X" axis label
        bbox_to_anchor=(0.5, -0.0),  # used if you have "X" axis label
        ncol=3,             # adjust if needed
        fontsize=8,
        frameon=True
    )
    # plt.grid(True)
    plt.tight_layout()
    if outfile_base is not None:
        base = outfile_base + f"_agent_exploration_trajectories_env{env_id}_ep{episode_idx}"
    elif output_dir and output_str:
        base = f"{output_dir}/agent_exploration_trajectories_env{env_id}_ep{episode_idx}_{output_str}"
    else:
        print("Warning: No output path provided, showing but not saving agent trajectories.")
        plt.show()

    save_fig(fig, base)
    print(f"Saved agent trajectories (together) to {base}.png")
    # plt.savefig (os.path.join(output_dir, f"agent_trajectories_env{env_id}_ep{episode_idx}.pdf"))
    # plt.savefig(os.path.join(output_dir, f"agent_trajectories_env{env_id}_ep{episode_idx}.png"))
    # plt.show()


def compute_histograms_for_episode(agent_positions, bin_width, bin_height):
    # Step 2: Define a function to compute bin edges based on the desired bin width/height
    def compute_bin_edges(
        x_positions, y_positions, bin_width, bin_height, x_range, y_range
    ):
        # Create bin edges based on the desired bin width and height
        x_bin_edges = np.arange(x_range[0], x_range[1] + bin_width, bin_width)
        y_bin_edges = np.arange(y_range[0], y_range[1] + bin_height, bin_height)
        return x_bin_edges, y_bin_edges


    # Step 3: Compute the 2D histogram using precomputed bin edges
    def compute_2d_histogram_with_edges(positions, x_bin_edges, y_bin_edges):
        x_positions = [pos[0] for pos in positions]
        y_positions = [pos[1] for pos in positions]

        # Compute the 2D histogram for the agent's positions
        hist, _, _ = np.histogram2d(
            x_positions, y_positions, bins=[x_bin_edges, y_bin_edges]
        )
        return hist

    # Flatten all positions across all agents in the episode
    all_positions = np.vstack(agent_positions)

    # Compute the arena size (x_range, y_range) from the positions
    x_positions = all_positions[:, 0]
    y_positions = all_positions[:, 1]
    x_range = (x_positions.min(), x_positions.max())
    y_range = (y_positions.min(), y_positions.max())

    # Precompute bin edges
    x_bin_edges, y_bin_edges = compute_bin_edges(
        x_positions, y_positions, bin_width, bin_height, x_range, y_range
    )

    # Compute histograms for each agent using the precomputed bin edges
    histograms = agent_positions.apply(
        lambda positions: compute_2d_histogram_with_edges(
            positions, x_bin_edges, y_bin_edges
        )
    )

    return histograms, x_range, y_range

def plot_2d_histograms_of_agent_positions(dff, env_id=0, episode_index=0, bin_width = 5, bin_height = 5):
    # TODO: Clean up function signature and docstring 
    #  
    # Step 1: Group data by episode, env_id, and agent_id, and collect positions
    agent_positions = dff.groupby(["env_id", "episode_index", "agent_id"])[
        "position"
    ].apply(list)

    # Compute histograms for a specific episode/env (e.g., env_id=0, episode_index=0)
    episode_positions = agent_positions.xs(
        (0, 0), level=["env_id", "episode_index"]
    )  # Selecting one episode/env
    episode_histograms, x_range, y_range = compute_histograms_for_episode(
        episode_positions, bin_width, bin_height
    )

    # Step 6: Plot the histograms for each agent
    hist_min = np.inf
    hist_max = -np.inf
    for hist in episode_histograms:
        hist_min = min(hist_min, hist.min())
        hist_max = max(hist_max, hist.max())

    fig, axes = plt.subplots(
        1, len(episode_histograms), figsize=(12, 3), sharex=True, sharey=True
    )
    for i, (agent_id, hist) in enumerate(episode_histograms.items()):
        ax = axes[i]
        cax = ax.imshow(
            hist.T,
            origin="lower",
            cmap="viridis",
            vmin=hist_min,
            vmax=hist_max,
            extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        )
        ax.set_title(f"A{agent_id + 1}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        if i != 0:  # Keep the labels only on the first subplot
            #         ax.set_xlabel('')
            ax.set_ylabel("")
    plt.subplots_adjust(wspace=0.05, hspace=0.0)

    fig.colorbar(cax, ax=axes.ravel().tolist(), shrink=0.70)
    plt.suptitle("2D Histograms of Agent Positions")
    plt.show()

# Define a function to compute IoU for all environments and episodes
def compute_iou_for_all_episodes(agent_positions, bin_width, bin_height):
    ious = {}

    # Iterate over each (env_id, episode_index) combination
    for (env_id, episode_index), episode_positions in agent_positions.groupby(
        level=["env_id", "episode_index"]
    ):
        # Compute histograms for the current episode
        episode_histograms, x_range, y_range = compute_histograms_for_episode(
            episode_positions, bin_width, bin_height
        )

        # Get list of agents
        #         agents = list(episode_histograms.index)
        agents = [agent_id[-1] for agent_id in episode_histograms.index]
        #         print(agents)

        # Compute pairwise IoU between all agent combinations
        for agent1, agent2 in combinations(agents, 2):
            #             hist1 = episode_histograms[agent1]
            #             hist2 = episode_histograms[agent2]
            hist1 = episode_histograms[(env_id, episode_index, agent1)]
            hist2 = episode_histograms[(env_id, episode_index, agent2)]

            # Compute intersection as the sum of the minimum values in the histograms
            intersection = np.sum(np.minimum(hist1, hist2))

            # Compute union as the sum of the maximum values in the histograms
            union = np.sum(np.maximum(hist1, hist2))

            if union > 0:  # Avoid division by zero
                iou = intersection / union
            else:
                iou = 0

            ious[(env_id, episode_index, agent1, agent2)] = iou

    # Convert the IoUs dictionary to a DataFrame for better visualization
    iou_df = pd.DataFrame.from_dict(ious, orient="index", columns=["IoU"])
    return iou_df

def deprecated_iou_analysis():
    # Manual merge
    # episode, env, arena_type
    # 0,0,random
    # 0,1,random
    # 0,2,patchy
    # 0,3,random
    # 0,4,patchy
    # 0,5,random
    # 0,6,feeder
    # 0,7,feeder
    arena_data = {
        "episode_index": [0, 0, 0, 0, 0, 0, 0, 0],
        "env_id": [0, 1, 2, 3, 4, 5, 6, 7],
        "arena_type": [
            "Random",
            "Random",
            "Patchy",
            "Random",
            "Patchy",
            "Random",
            "Feeder",
            "Feeder",
        ],
    }

    arena_df = pd.DataFrame(arena_data)
    # iou_df_merged = iou_df.merge(arena_df, on=["env_id", "episode_index"], how="left")

    # # sns.set(style="whitegrid")
    # plt.figure(figsize=(3, 2.5))
    # sns.boxplot(x='arena_type', y='IoU', data=iou_df_merged)
    # # plt.title('IoU by Arena Type')
    # plt.xlabel('Arena Type')
    # plt.ylabel('IoU')
    # plt.show()

    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # from scipy.stats import mannwhitneyu
    # import statsmodels
    # from statannotations.Annotator import Annotator

    # # group_sizes = iou_df_merged.groupby('arena_type').size().reset_index(name='N')
    # # Calculate the number of points per group
    # group_sizes = iou_df_merged.groupby("arena_type").size().reset_index(name="N")
    # # Sort group_sizes to preserve the original order of arena_type in the x-axis
    # group_sizes = (
    #     group_sizes.set_index("arena_type")
    #     .reindex(iou_df_merged["arena_type"].unique())
    #     .reset_index()
    # )
    # print(group_sizes)

    # # Create the plot
    # plt.figure(figsize=(3, 3))
    # ax = sns.boxplot(x="arena_type", y="IoU", data=iou_df_merged)
    # plt.xlabel("Arena Type")
    # plt.ylabel("Agent pairwise IoU")

    # # Add N per group to plot
    # arena_labels = [
    #     f"{row['arena_type']}\n(N={row['N']})" for _, row in group_sizes.iterrows()
    # ]
    # ax.set_xticklabels(arena_labels)

    # # Define the pairs for comparison (all pairwise combinations of arena types)
    # arena_types = iou_df_merged["arena_type"].unique()
    # pairs = [
    #     (arena1, arena2)
    #     for i, arena1 in enumerate(arena_types)
    #     for arena2 in arena_types[i + 1 :]
    # ]

    # # Perform the pairwise statistical tests
    # annotator = Annotator(ax, pairs, data=iou_df_merged, x="arena_type", y="IoU")

    # # Use the Mann-Whitney U test (non-parametric) and apply Bonferroni correction for multiple comparisons
    # annotator.configure(
    #     test="Mann-Whitney",
    #     text_format="star",
    #     loc="outside",
    #     comparisons_correction="bonferroni",
    # )
    # annotator.apply_and_annotate()

    # # Make the plot manuscript-ready
    # plt.tight_layout()
    # plt.show()

def plot_behavioral_maneuvers_around_eating(dff, agents=None, window_size = 5):
    if agents is None:
        agents = dff["agent_id"].unique()
    
    for agent_id in agents:
        agent_data = dff[dff["agent_id"] == agent_id]

        speed_changes = []
        trajectory_curvatures = []

        for (env_id, episode_index), subset_data in agent_data.groupby(
            ["env_id", "episode_index"]
        ):
            eating_indices = subset_data[subset_data["eating_event"]].index

            for idx in eating_indices:
                pos = subset_data.index.get_loc(idx)
                if pos - window_size >= 0 and pos + window_size < len(subset_data):
                    segment = subset_data.iloc[
                        pos - window_size : pos + window_size + 1
                    ]

                    # Calculate speed change
                    speed_change = segment["displacement"].values
                    speed_changes.append(speed_change)

                    # Calculate trajectory curvature
                    positions = np.stack(segment["position"].values)
                    diffs = np.diff(positions, axis=0)
                    directions = np.arctan2(diffs[:, 1], diffs[:, 0])
                    curvature = np.abs(np.diff(directions))
                    curvature = np.concatenate(
                        ([0], curvature, [0])
                    )  # to match the length of the window
                    trajectory_curvatures.append(curvature)

        if (
            speed_changes and trajectory_curvatures
        ):  # Check if there are any valid maneuvers
            speed_changes = np.array(speed_changes)
            trajectory_curvatures = np.array(trajectory_curvatures)

            # Calculate mean and standard error for speed changes
            mean_speed_change = np.mean(speed_changes, axis=0)
            std_speed_change = np.std(speed_changes, axis=0) / np.sqrt(
                speed_changes.shape[0]
            )

            # Calculate mean and standard error for trajectory curvatures
            mean_curvature = np.mean(trajectory_curvatures, axis=0)
            std_curvature = np.std(trajectory_curvatures, axis=0) / np.sqrt(
                trajectory_curvatures.shape[0]
            )

            print(f"Agent {agent_id} speed_changes.shape", speed_changes.shape)
            print(
                f"Agent {agent_id} trajectory_curvatures.shape",
                trajectory_curvatures.shape,
            )

            time_steps = np.arange(-window_size, window_size + 1)

            # Plot speed changes
            plt.figure(figsize=(5, 2))
            plt.plot(time_steps, mean_speed_change, label="Speed Change")
            plt.fill_between(
                time_steps,
                mean_speed_change - std_speed_change,
                mean_speed_change + std_speed_change,
                alpha=0.2,
            )
            plt.axvline(x=0, color="red", linestyle="--", label="Eating Event")
            plt.title(f"Agent {agent_id} Speed Changes Around Eating Events")
            plt.xlabel("Time Steps")
            plt.ylabel("Speed Change")
            plt.legend()
            plt.show()

            # Plot trajectory curvatures
            plt.figure(figsize=(5, 2))
            plt.plot(time_steps, mean_curvature, label="Trajectory Curvature")
            plt.fill_between(
                time_steps,
                mean_curvature - std_curvature,
                mean_curvature + std_curvature,
                alpha=0.2,
            )
            plt.axvline(x=0, color="red", linestyle="--", label="Eating Event")
            plt.title(f"Agent {agent_id} Trajectory Curvature Around Eating Events")
            plt.xlabel("Time Steps")
            plt.ylabel("Curvature")
            plt.legend()
            plt.show()

def plot_cumulative_food_consumption(dff):
    # TODO: Fix nonmonotonic cumulative plot bug
    # Filter eating events
    eating_events = dff[dff["eating_event"]].copy()

    # Create cumulative food consumed
    eating_events["cumulative_food"] = (
        eating_events.groupby(["env_id", "episode_index", "agent_id"]).cumcount() + 1
    )

    plt.figure(figsize=(5, 5))
    colors = sns.color_palette("husl", len(dff["agent_id"].unique()))

    # Plot individual trajectories
    for (env_id, episode_index, agent_id), group in eating_events.groupby(
        ["env_id", "episode_index", "agent_id"]
    ):
        print(f"Env {env_id}, Episode {episode_index}, Agent {agent_id}: {len(group)}")
        plt.plot(
            group["time_step"],
            group["cumulative_food"],
            color=colors[agent_id],
            # alpha=0.5,
            ls="--",
        )

    # Plot average trajectories per agent
    for agent_id in dff["agent_id"].unique():
        agent_data = eating_events[eating_events["agent_id"] == agent_id]
        avg_cumulative_food = agent_data.groupby("time_step")["cumulative_food"].mean()
        plt.plot(
            avg_cumulative_food.index,
            avg_cumulative_food.values,
            color=colors[agent_id],
            linewidth=1,
            ls="-.",
            alpha=0.75,
            label=f"Agent {agent_id} Average",
        )

    # Plot overall average trajectory
    overall_avg_cumulative_food = (
        eating_events.groupby("time_step")["cumulative_food"]
        .mean()
        .rolling(window=10)  # smoothing
        .mean()
    )
    plt.plot(
        overall_avg_cumulative_food.index,
        overall_avg_cumulative_food.values,
        color="black",
        linewidth=1,
        label="Overall Average",
    )

    # Adding labels and legend
    plt.xlabel("Time Steps")
    plt.ylabel("Cumulative Food Consumed")
    plt.title("Cumulative Food Consumed Over Time")
    plt.legend()
    plt.show()

def fit_consumption_time_model(dff):
    # TODO: better documentation; inspired by Harpaz et al 

    def exponential_model(n, a, b):
        """Exponential model for consumption time prediction."""
        return np.exp(a + b * n)

    # Filter eating events
    eating_events = dff[dff["eating_event"]].copy()

    # Calculate cumulative food consumed
    eating_events["cumulative_food"] = (
        eating_events.groupby(["env_id", "episode_index", "agent_id"]).cumcount() + 1
    )

    # Calculate the total time to consume all food items in each episode for each group size
    consumption_times = (
        eating_events.groupby(["env_id", "episode_index", "agent_id"])
        .agg({"time_step": "max", "cumulative_food": "max"})
        .reset_index()
    )

    # Calculate group size for each episode
    consumption_times["group_size"] = consumption_times.groupby(
        ["env_id", "episode_index"]
    )["agent_id"].transform("nunique")

    # Fit the exponential model for each group size
    group_sizes = consumption_times["group_size"].unique()
    model_params = {}

    plt.figure(figsize=(5, 2))

    for group_size in group_sizes:
        group_data = consumption_times[consumption_times["group_size"] == group_size]
        n = group_data["cumulative_food"]
        T = group_data["time_step"]

        # # Fit the model
        # popt, _ = curve_fit(exponential_model, n, np.log(T), maxfev=10000)
        # model_params[group_size] = popt

        # Plot the data and the fitted model
        plt.scatter(n, T, label=f"Group Size {group_size} Data")
        # plt.plot(n, exponential_model(n, *popt), label=f"Group Size {group_size} Fit")

    plt.xlabel("Number of Food Items (n)")
    plt.ylabel("Consumption Time (T)")
    plt.title("Consumption Time Prediction")
    plt.legend()
    plt.show()

    return model_params

def plot_histograms_split_by_column(dff, variables, split_column):

    for var in variables:
        plt.figure(figsize=(5, 2))
        sns.histplot(
            data=dff,
            x=var,
            hue=split_column,
            element="step",
            stat="density",
            common_norm=False,
        )
        plt.title(f"Distribution of {var} with has_nearby States")
        plt.xlabel(var)
        plt.ylabel("Density")
        plt.show()

def calculate_group_statistics(dff):
    # Function to calculate group polarization
    results = []
    for (env_id, episode_index, time_step), group in dff.groupby(
        ["env_id", "episode_index", "time_step"]
    ):
        group_positions = group["position"].apply(np.array).values
        group_orientations = group["orientation"].values
        polarization = calculate_polarization(group_orientations)
        cohesion = calculate_cohesion(group_positions)
        results.append(
            {
                "env_id": env_id,
                "episode_index": episode_index,
                "time_step": time_step,
                "polarization": polarization,
                "mean_nn_distance_cm": cohesion,
            }
        )

    stats_df = pd.DataFrame(results).sort_values(
        by=["env_id", "episode_index", "time_step"]
    )
    return stats_df

def plot_group_statistics_timecourses(dff, env_id, episode_index):
    """
    Plot timecourses of group polarization and cohesion for a specific environment and episode.
    """
    subset = dff[(dff["env_id"] == env_id) & (dff["episode_index"] == episode_index)]
    if subset.empty:
        print(f"No data found for env_id={env_id}, episode_index={episode_index}")
        return  
    subset = subset.sort_values(by="time_step").reset_index(drop=True)
    plt.figure(figsize=(6, 4))

    plt.subplot(2, 1, 1)
    plt.plot(subset["time_step"], subset["polarization"], marker="o", markersize=2)
    plt.title(f"Env {env_id} Episode {episode_index} - Polarization")
    plt.xlabel("Time Step")
    plt.ylabel("Polarization")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(subset["time_step"], subset["mean_nn_distance_cm"], marker="o", markersize=2)
    plt.title(f"Env {env_id} Episode {episode_index} - Cohesion")
    plt.xlabel("Time Step")
    plt.ylabel("Cohesion")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


########### TESTS ############
class TestSwimmingStatistics(unittest.TestCase):

    def test_calculate_polarization(self):
        # Test with aligned orientations
        orientations = [0, 0, 0, 0]
        self.assertAlmostEqual(calculate_polarization(orientations), 1.0, places=5)

        # Test with anti-aligned orientations
        orientations = [0, np.pi, 0, np.pi]
        self.assertAlmostEqual(calculate_polarization(orientations), -1.0, places=5)

        # Test with random orientations
        orientations = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        self.assertAlmostEqual(calculate_polarization(orientations), 0.0, places=5)

    def test_calculate_cohesion(self):
        # Test with two positions
        positions = [[0, 0], [3, 4]]  # Distance is 5
        self.assertAlmostEqual(calculate_cohesion(positions), 5.0, places=5)

        # Test with multiple positions
        positions = [[0, 0], [1, 1], [2, 2], [3, 3]]
        expected_cohesion = np.mean(
            [np.sqrt(2)] * 4
        )  # Each point has sqrt(2) as the nearest distance
        self.assertAlmostEqual(
            calculate_cohesion(positions), expected_cohesion, places=5
        )

        # Test with less than two positions
        positions = [[0, 0]]
        self.assertTrue(np.isnan(calculate_cohesion(positions)))

def test_interaction_classification():
    def plot_scenario(pos_a1, pos_a2, ori_a1, ori_a2, label):
        fig, ax = plt.subplots(figsize=(3,3))
        ax.plot(pos_a1[0], pos_a1[1], 'bo', label='A1')
        ax.plot(pos_a2[0], pos_a2[1], 'ro', label='A2')

        ax.arrow(pos_a1[0], pos_a1[1], 0.5*np.cos(ori_a1), 0.5*np.sin(ori_a1),
                head_width=0.08, fc='blue', ec='blue')
        ax.arrow(pos_a2[0], pos_a2[1], 0.5*np.cos(ori_a2), 0.5*np.sin(ori_a2),
                head_width=0.08, fc='red', ec='red')

        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title(f"{label}")
        ax.legend()
        plt.show()

    # Define unit test scenarios for all categories
    scenarios = [
        # Confronting
        {"pos_a1": np.array([0, 0]), "pos_a2": np.array([1, 0]), "ori_a1": 0, "ori_a2": np.pi, "label": "Confronting"},
        # Chasing
        {"pos_a1": np.array([0, 0]), "pos_a2": np.array([1, 0]), "ori_a1": 0, "ori_a2": 0, "label": "Chasing"},
        # Being chased
        {"pos_a1": np.array([0, 0]), "pos_a2": np.array([1, 0]), "ori_a1": np.pi, "ori_a2": np.pi, "label": "Being Chased"},
        # Dispersing
        {"pos_a1": np.array([0, 0]), "pos_a2": np.array([1, 0]), "ori_a1": np.pi, "ori_a2": 0, "label": "Dispersing"},
        # Unaligned
        {"pos_a1": np.array([0, 0]), "pos_a2": np.array([1, 0]), "ori_a1": np.pi/2, "ori_a2": 0, "label": "Unaligned"},
    ]

    for s in scenarios:
        result = classify_interaction(s["pos_a1"], s["pos_a2"], s["ori_a1"], s["ori_a2"])
        plot_scenario(s["pos_a1"], s["pos_a2"], s["ori_a1"], s["ori_a2"], f"{s['label']} → {result}")




def run_behavior_report(dff, outfile_base, plot_only=False):
    MAX_ENVS_TO_PLOT = 1
    MAX_EPISODES_PER_ENV_TO_PLOT = 3

    env_ids = dff['env_id'].unique()
    episode_indices = dff['episode_index'].unique()
    print(f"Found {len(episode_indices)} episodes: {episode_indices}")

    for env_id in env_ids[:MAX_ENVS_TO_PLOT]:
        for episode_index in episode_indices[:MAX_EPISODES_PER_ENV_TO_PLOT]:
            print(f"Plotting data for env {env_id}, episode {episode_index}")

            plot_ethogram(
                dff,
                env_id=env_id,
                episode_index=episode_index,
                agents=dff["agent_id"].unique().tolist(),
                ema_window=10,
                plot_eods=True,
                figsize=(5.5, 2.5),
                outfile_base=outfile_base,
                output_dir=None,
                output_str=None,
                plot_only=plot_only,
            )

            plot_agent_trajectories_side_by_side(
                dff,
                env_id=env_id,
                episode_idx=episode_index,
                outfile_base=outfile_base,
                output_dir=None,
                output_str=None,
            )

            plot_agent_trajectories_together(
                dff,
                env_id=env_id,
                episode_idx=episode_index,
                outfile_base=outfile_base,
                output_dir=None,
                output_str=None,
                plot_only=plot_only,
            )



if __name__ == "__main__":
    unittest.main()
