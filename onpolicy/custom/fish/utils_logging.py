import os
import csv
import ast
from datetime import datetime

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

def log_on_done(env):
    # Grab the folder where Runner is writing its logs:
    log_dir = env.all_args["run_dir"] + "/logs"
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "env_params.csv")
    write_header = not os.path.exists(path)

    # Collect exactly the fields you care about
    row = {
        "timestamp": datetime.now().isoformat(),
        "arena_type": env.arena.__class__.__name__,
        "food_scaling_factor": env.curr_food_scaling_factor,
        "arena_size_x": env.arena_size[0],
        "arena_size_y": env.arena_size[1],
        # "knollen_mode": str([agent.knollen_mode for agent in env.agent_objects]),
        # "ampullary_mode": str([agent.ampullary_mode for agent in env.agent_objects]),
        # "mormyromast_mode": str([agent.mormyromast_mode for agent in env.agent_objects]),
        "knollen_mode": env.sensor_modes_episode.knollen_mode,
        "ampullary_mode": env.sensor_modes_episode.ampullary_mode,
        "mormyromast_mode": env.sensor_modes_episode.mormyromast_mode,
        "cumulative_rewards": str(
            [agent.cumulative_reward for agent in env.agent_objects]
        ),
    }
    # Per‐agent sizes:
    for i, agent in enumerate(env.agent_objects):
        row[f"agent_{i}_size"] = round(agent.agent_size, 3)

    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)


def plot_mean_cumulative_rewards(
    csv_path,
    smoothing_window: int = None,
    smoothing_type: str = 'moving'
):
    """
    Reads a CSV file with a 'cumulative_rewards' column containing list-like strings,
    attempts to parse and compute the mean cumulative reward per row,
    printing and skipping any malformed lines,
    then plots time (row index) vs. mean reward with separate lines for each arena type.
    """
    # Load the data
    df = pd.read_csv(csv_path)
    df = df[df['cumulative_rewards'] != 'cumulative_rewards']  # sometimes header row is duplicated
    
    # Parse cumulative_rewards safely, collecting mean rewards
    df['cumulative_rewards'] = df['cumulative_rewards'].apply(ast.literal_eval)
    
    df['mean_reward'] = df['cumulative_rewards'].apply(lambda rewards: sum(rewards) / len(rewards))

    # Create the plot
    fig, ax = plt.subplots()
    for arena_type, group in df.groupby('arena_type'):
        series = group['mean_reward']
        if smoothing_window:
            if smoothing_type == 'moving':
                series = series.rolling(window=smoothing_window, min_periods=1).mean()
            elif smoothing_type == 'ewm':
                series = series.ewm(span=smoothing_window, adjust=False).mean()
            else:
                raise ValueError("smoothing_type must be 'moving' or 'ewm'")
        ax.plot(group.index, series, label=arena_type)
    
    # Labeling
    ax.set_xlabel('Time (row index)')
    ax.set_ylabel('Mean Cumulative Reward')
    # ax.set_title('Mean Cumulative Rewards Over Time by Arena Type')
    title = 'Mean Cumulative Rewards Over Time'
    if smoothing_window:
        title += f' (Window: {smoothing_window})'
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    # Save the plot
    filename = 'mean_cumulative_rewards_by_arena'
    if smoothing_window:
        filename += f'_{smoothing_type}_{smoothing_window}'
    plt.savefig(os.path.join(os.path.dirname(csv_path), f'{filename}.pdf'))


def plot_train_metrics(log_dir):
    try:
        csv_path = os.path.join(log_dir, "train_metrics.csv")
        df = pd.read_csv(csv_path)

        df["rew_smooth"] = df["average_episode_reward"].rolling(20, min_periods=1).mean()

        _, axes = plt.subplots(1, 3, sharex=True, figsize=(12, 4))
        axes[0].plot(df["step"], df["rew_smooth"])
        axes[0].set_ylabel("Avg Return (Smoothed)")
        axes[0].set_xlabel("Environment Steps")

        axes[1].plot(df["step"], df["policy_loss"], label="policy")
        axes[1].plot(df["step"], df["value_loss"], label="value")
        axes[1].legend()
        axes[1].set_ylabel("Loss")
        axes[1].set_xlabel("Environment Steps")

        axes[2].plot(df["step"], df["dist_entropy"])
        axes[2].set_ylabel("Entropy")
        axes[2].set_xlabel("Environment Steps")

        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "training_curve.pdf"))
    except Exception as e:
        print("Error plotting training metrics:", e)


def plot_train_metrics_subrewards(log_dir):
    try:
        csv_path = os.path.join(log_dir, "train_metrics.csv")
        df = pd.read_csv(csv_path)

        if "r_components_json" in df.columns:
            r_components = df["r_components_json"].apply(
                lambda x: {} if pd.isna(x) else ast.literal_eval(x)
            )
            reward_df = pd.json_normalize(r_components)
            reward_cols = list(reward_df.columns)
            reward_df = reward_df.rolling(20, min_periods=1).mean()
            for col in reward_cols:
                df[col] = reward_df[col]
        else:
            reward_cols = []

        total_rows = len(reward_cols)
        if total_rows == 0:
            print("No subreward components found to plot.")
            return
        fig_height = max(1.2 * total_rows, 6.0)
        fig, axes = plt.subplots(total_rows, 1, sharex=True, figsize=(6, fig_height))
        if total_rows == 1:
            axes = [axes]

        row = 0
        for col in reward_cols:
            axes[row].plot(df["step"], df[col], label=col)
            axes[row].legend(fontsize=9, loc="upper right")
            row += 1

        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)

        axes[-1].set_xlabel("Environment Steps")
        fig.text(
            0.02,
            0.5,
            "Reward Components (Smoothed)",
            va="center",
            rotation="vertical",
        )
        fig.subplots_adjust(hspace=0.0, top=0.98, bottom=0.06, left=0.2)
        plt.savefig(os.path.join(log_dir, "training_curve_subrewards.pdf"))
    except Exception as e:
        print("Error plotting training subrewards:", e)
