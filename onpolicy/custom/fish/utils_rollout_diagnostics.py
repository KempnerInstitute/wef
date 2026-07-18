# Analyze MARL rollouts after applying preprocess_flatten.py



import argparse
import os
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import utils_report as ru
from utils_features import add_attention_entropy_columns
from utils_sensors import (
    get_obs_partitions, get_obs_partitions_from_df,
    get_sensor_indices_from_cfg,
    compute_sensor_boundaries,
)
from cfg import AGENT_PARAMS, m_to_cm


AVAILABLE_ANALYSES = [
    "initial_positions",
    "sensor_heatmaps",
    "sensor_timecourses",
    "sensor_magnitudes",
    "attention_entropy",
    "attention_weights",
    "sensing_by_target_location",
    "column_histograms",
]
DEFAULT_ANALYSES = [
    "sensor_heatmaps",
    "attention_entropy",
    "attention_weights",
    "sensing_by_target_location",
    "column_histograms",
]




def resolve_analyses(requested):
    if not requested:
        return list(DEFAULT_ANALYSES)
    if "all" in requested:
        return list(AVAILABLE_ANALYSES)
    return requested


def run_rollout_diagnostics_report(
    dff,
    outfile_base,
    analyses_to_run=None,
    sensor_model="fracrand",
    num_agents=None,
    use_signed_max=False,
    rotate_agent_up=True,
    plot_only=False,
):
    outputs_folder = outfile_base
    ensure_output_dir(outputs_folder)
    pkl_str = os.path.basename(os.path.normpath(outfile_base)) or "rollout_diagnostics"

    if num_agents is None:
        if "agent_id" in dff.columns:
            num_agents = int(dff["agent_id"].nunique())
        else:
            raise ValueError("agent_id column missing; provide num_agents.")

    if "metadata" in dff.columns:
        p = get_obs_partitions_from_df(dff)
        mormyromast_indices = p["mormyromast"]
        ampullary_indices   = p["ampullary"]
        knollen_indices     = p["knollen"]
    else:
        mormyromast_indices, ampullary_indices, knollen_indices = get_sensor_indices_from_cfg(
            num_agents=num_agents,
            model=sensor_model,
        )
    sensor_boundaries = compute_sensor_boundaries(
        mormyromast_indices, ampullary_indices, knollen_indices
    )

    observation_width = dff["observations"].values[0].shape[0]
    knollen_end = int(knollen_indices[-1]) if len(knollen_indices) > 0 else -1
    sensor_types = [
        ("Mormyromast", mormyromast_indices),
        ("Ampullary", ampullary_indices),
        ("Knollen", knollen_indices),
        ("Other", slice(knollen_end + 1, observation_width)),
    ]

    analyses = resolve_analyses(analyses_to_run)
    print(f"[rollout_diagnostics] analyses: {analyses}")
    has_attn_mask = ("attn_mask" in dff.columns) and (not dff["attn_mask"].isna().all())

    if "initial_positions" in analyses:
        run_initial_positions(dff, outputs_folder, pkl_str)
    if "sensor_heatmaps" in analyses:
        run_sensor_heatmaps(dff, outputs_folder, pkl_str, mormyromast_indices, sensor_boundaries)
    if "sensor_timecourses" in analyses:
        run_sensor_timecourses(dff, outputs_folder, pkl_str)
    if "sensor_magnitudes" in analyses:
        run_sensor_magnitudes(dff, outputs_folder, pkl_str, mormyromast_indices)
    if "attention_entropy" in analyses:
        if has_attn_mask:
            run_attention_entropy(dff, outputs_folder, pkl_str, sensor_types)
        else:
            print("[rollout_diagnostics] skipping attention_entropy: no attn_mask column")
    if "attention_weights" in analyses:
        if has_attn_mask:
            run_attention_weights(dff, outputs_folder, pkl_str, sensor_types)
        else:
            print("[rollout_diagnostics] skipping attention_weights: no attn_mask column")
    if "sensing_by_target_location" in analyses:
        run_sensing_by_target_location(
            dff,
            outputs_folder,
            pkl_str,
            mormyromast_indices,
            ampullary_indices,
            use_signed_max=use_signed_max,
            sensor_model=sensor_model,
            rotate_agent_up=rotate_agent_up,
        )
    if "column_histograms" in analyses:
        run_column_histograms(dff, outputs_folder, pkl_str)


def ensure_output_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created output directory: {path}")


def save_figure(fig, output_dir, pkl_str, suffix):
    fname = os.path.join(output_dir, f"{pkl_str}_{suffix}.png")
    fig.savefig(fname)
    plt.close(fig)
    print(f"Saved: {fname}")
    return fname


def sample_dff(dff, max_rows, seed=42):
    if dff.shape[0] > max_rows:
        return dff.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return dff.copy()


def _sanitize_suffix(text, max_len=80):
    safe = []
    for ch in str(text):
        if ch.isalnum() or ch in ("_", "-"):
            safe.append(ch)
        else:
            safe.append("_")
    out = "".join(safe)
    if len(out) > max_len:
        out = out[:max_len].rstrip("_")
    return out or "col"


def get_one_episode_df(dff, env_id=0, episode_index=0, agent_id=0):
    subset = dff[
        (dff["env_id"] == env_id)
        & (dff["episode_index"] == episode_index)
        & (dff["agent_id"] == agent_id)
    ]
    return subset.reset_index(drop=True)


def visualize_sensor_corr_matrix(obs, sensor_boundaries=None, title_suffix=""):
    corr_mat = np.corrcoef(obs, rowvar=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    img = ax.imshow(corr_mat, cmap="coolwarm", vmin=-1, vmax=1)
    if sensor_boundaries:
        for boundary in sensor_boundaries:
            ax.axvline(x=boundary, color="black", linestyle="--", linewidth=0.8)
            ax.axhline(y=boundary, color="black", linestyle="--", linewidth=0.8)
    ax.set_title(f"Sensor Corr {title_suffix}")
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    return fig


def viz_boxplot_vertical_sparse(
    obs,
    title="",
    sensor_boundaries=None,
    vmin=None,
    vmax=None,
    tick_step=25,
):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.boxplot(
        obs,
        vert=True,
        showfliers=False,
        boxprops=dict(linewidth=0.3),
        whiskerprops=dict(linewidth=0.3),
        capprops=dict(linewidth=0.3),
        medianprops=dict(linewidth=0.6, color="red"),
    )
    if vmin is not None and vmax is not None:
        ax.set_ylim(vmin, vmax)
    if sensor_boundaries:
        for boundary in sensor_boundaries:
            ax.axvline(x=boundary, color="blue", linestyle="--", linewidth=0.8)
    ticks = np.arange(1, obs.shape[1] + 1, tick_step)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks - 1, rotation=0)
    ax.set_ylabel("Value")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def viz_histogram_self_cons_morm(obs, morm_indices):
    fig, ax = plt.subplots(figsize=(6, 3))
    if len(morm_indices) > 36:
        self_morm_indices = morm_indices[:36]
        cons_morm_indices = morm_indices[36:]
        self_morm_obs = obs[:, self_morm_indices]
        cons_morm_obs = obs[:, cons_morm_indices]
        ax.hist(
            self_morm_obs.flatten(),
            bins=50,
            alpha=0.5,
            label="Self",
            color="blue",
            density=True,
        )
        ax.hist(
            cons_morm_obs.flatten(),
            bins=50,
            alpha=0.5,
            label="Cons",
            color="orange",
            density=True,
        )
        ax.set_title("Histogram of Self vs Conspecific Mormyromast Sensor Values")
        ax.legend()
    else:
        morm_obs = obs[:, morm_indices]
        ax.hist(morm_obs.flatten(), bins=50, alpha=0.7, color="green", density=True)
        ax.set_title("Histogram of Mormyromast Sensor Values")
    ax.set_xlabel("Mormyromast Sensor Value")
    ax.set_ylabel("Density")
    fig.tight_layout()
    return fig


def run_initial_positions(dff, output_dir, pkl_str):
    agent_start_position_df = (
        dff.groupby(["env_id", "episode_index", "agent_id"])
        .first()
        .reset_index()[["env_id", "episode_index", "agent_id", "position_x", "position_y"]]
    )
    agent_ids = agent_start_position_df["agent_id"].unique()
    for agent_id in agent_ids:
        subset = agent_start_position_df[agent_start_position_df["agent_id"] == agent_id]
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(subset["position_x"], subset["position_y"], alpha=0.5)
        ax.set_title(f"Agent {agent_id} Start Positions")
        ax.set_xlabel("Position X")
        ax.set_ylabel("Position Y")
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        save_figure(fig, output_dir, pkl_str, f"initial_positions_agent_{agent_id}")


def run_sensor_heatmaps(dff, output_dir, pkl_str, morm_indices, sensor_boundaries):
    dff_subset = sample_dff(dff, max_rows=100000)
    obs = np.stack(dff_subset["observations"].values, axis=0)

    fig = visualize_sensor_corr_matrix(obs, sensor_boundaries=sensor_boundaries)
    save_figure(fig, output_dir, pkl_str, "sensor_correlation_matrix")

    nan_cols = np.any(np.isnan(obs), axis=0)
    cols_with_nan = np.where(nan_cols)[0]
    print("Number of columns with NaNs:", len(cols_with_nan))
    print("Columns with NaNs:", cols_with_nan)

    fig = viz_boxplot_vertical_sparse(
        obs,
        title="Sensor Boxplot (Sparse)",
        sensor_boundaries=sensor_boundaries,
    )
    save_figure(fig, output_dir, pkl_str, "boxplot_vertical_sparse")

    fig = viz_boxplot_vertical_sparse(
        obs,
        title="Sensor Boxplot (Sparse, Clipped)",
        sensor_boundaries=sensor_boundaries,
        vmin=-1.1,
        vmax=1.1,
    )
    save_figure(fig, output_dir, pkl_str, "boxplot_vertical_sparse_clipped")

    fig = viz_histogram_self_cons_morm(obs, morm_indices)
    save_figure(fig, output_dir, pkl_str, "histogram_self_cons_morm")


def run_sensor_timecourses(dff, output_dir, pkl_str):
    dff_subset = sample_dff(dff, max_rows=100000)
    dff_subset = dff_subset.sort_values(
        by=["env_id", "episode_index", "agent_id", "time_step"]
    )
    obs = np.stack(dff_subset["observations"].values, axis=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    img = ax.imshow(obs, aspect="auto", cmap="PiYG", vmin=-1.1, vmax=1.1)
    fig.colorbar(img, ax=ax, label="Sensor Value")
    ax.set_xlabel("Sensor Index")
    ax.set_ylabel("Timestep (with breaks)")
    fig.tight_layout()
    save_figure(fig, output_dir, pkl_str, "sensor_timecourses_heatmap")

    one_episode_df = get_one_episode_df(dff)
    if one_episode_df.empty:
        print("Skipping per-sensor timecourses (no episode 0/0/0).")
        return
    ep_obs_df = one_episode_df["observations"].apply(pd.Series)
    ep_obs_df.columns = [f"obs_{i}" for i in ep_obs_df.columns]
    sample_size = min(10, len(ep_obs_df.columns))
    random_columns = np.random.choice(ep_obs_df.columns, size=sample_size, replace=False)
    print("Randomly selected columns for timecourse visualization:", random_columns)
    for col in random_columns:
        fig, ax = plt.subplots(figsize=(5, 1.5))
        ax.plot(ep_obs_df[col], label=col)
        ax.set_xlabel("Timestep")
        ax.set_ylabel(col)
        ax.set_ylim(-1.1, 1.1)
        fig.tight_layout()
        save_figure(fig, output_dir, pkl_str, f"sensor_timecourse_{col}")


def run_sensor_magnitudes(dff, output_dir, pkl_str, morm_indices):
    required_cols = {"center_field.mormyromast", "distance_to_closest_food"}
    if not required_cols.issubset(dff.columns):
        print(f"Skipping sensor_magnitudes (missing columns: {required_cols}).")
        return

    dff_subset = sample_dff(dff, max_rows=100000)
    if len(morm_indices) == 0:
        print("Skipping sensor_magnitudes (no mormyromast indices).")
        return
    obs_morm = np.stack(dff_subset["observations"].values, axis=0)[:, morm_indices]
    center = np.stack(dff_subset["center_field.mormyromast"].values, axis=0)
    center_mag = np.log(np.linalg.norm(center, axis=1))
    dist_food = np.stack(dff_subset["distance_to_closest_food"].values, axis=0)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(obs_morm[:, 0], center_mag, s=2, alpha=0.2)
    ax.set_ylabel("|center_field.mormyromast|")
    ax.set_xlabel("max |obs_morm| per timestep")
    fig.tight_layout()
    save_figure(fig, output_dir, pkl_str, "sensor_magnitudes_center_vs_obs")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(dist_food, center_mag, s=2, alpha=0.2)
    ax.set_xlabel("distance_to_closest_food")
    ax.set_ylabel("center_field.mormyromast")
    fig.tight_layout()
    save_figure(fig, output_dir, pkl_str, "sensor_magnitudes_center_vs_distance")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(dist_food, obs_morm[:, 0], s=2, alpha=0.2)
    ax.set_xlabel("distance_to_closest_food")
    ax.set_ylabel("max |obs_morm| per timestep")
    fig.tight_layout()
    save_figure(fig, output_dir, pkl_str, "sensor_magnitudes_obs_vs_distance")


def run_attention_entropy(dff, output_dir, pkl_str, sensor_types):
    if "attn_mask" not in dff.columns:
        print("[rollout_diagnostics] attention_entropy skipped: no attn_mask column")
        return
    dff_subset = sample_dff(dff, max_rows=10000)
    dff_subset = add_attention_entropy_columns(dff_subset, sensor_types)
    entropy_columns = [col for col in dff_subset.columns if "entropy" in col]

    fig, ax = plt.subplots(figsize=(7, 4))
    for col in entropy_columns:
        ax.hist(dff_subset[col], bins=50, alpha=0.5, label=col)
    ax.set_title("Entropy Distribution for Attention (Sub)Masks")
    ax.set_xlabel("Entropy [bits]")
    ax.set_ylabel("Frequency")
    ax.set_xlim(0, 15)
    ax.legend(entropy_columns)
    fig.tight_layout()
    save_figure(fig, output_dir, pkl_str, "attention_entropy_distribution")

    fig, ax = plt.subplots(figsize=(7, 4))
    for col in entropy_columns:
        entropy_deltas = (
            dff_subset.groupby(["env_id", "episode_index", "agent_id"], group_keys=False)
            .apply(lambda group: group[col].diff().abs())
            .dropna()
        )
        ax.hist(entropy_deltas, bins=50, alpha=0.7, label=f"{col} delta")
    ax.set_title("Attention Entropy Delta Distributions")
    ax.set_xlabel("Absolute Entropy Change [bits]")
    ax.set_ylabel("Frequency")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir, pkl_str, "attention_entropy_delta_distribution")


def add_attention_weight_columns(dff, sensor_types, colname="attn_mask", attn_type="softmax"):
    data_matrix = np.stack(dff[colname].values)
    print("data_matrix.shape", data_matrix.shape)
    for sensor_name, indices in sensor_types:
        sensor_data = data_matrix[:, indices]

        col_name = f"attn_{sensor_name.lower()}_weight_actual"
        dff[col_name] = np.sum(sensor_data, axis=1)

        col_name = f"attn_{sensor_name.lower()}_weight_expected"
        dff[col_name] = sensor_data.shape[1] / data_matrix.shape[1]
        if attn_type == "sigmoid":
            dff[col_name] = sensor_data.shape[1]
    return dff


def run_attention_weights(dff, output_dir, pkl_str, sensor_types, attn_type="softmax"):
    if "attn_mask" not in dff.columns:
        print("[rollout_diagnostics] attention_weights skipped: no attn_mask column")
        return
    dff_subset = sample_dff(dff, max_rows=10000)
    dff_subset = add_attention_weight_columns(dff_subset, sensor_types, attn_type=attn_type)
    weight_columns = [col for col in dff_subset.columns if "_weight_actual" in col]

    sensor_names = [col.split("_")[1] for col in weight_columns]
    cmap = cm.get_cmap("Dark2", len(sensor_names))
    color_map = {name: cmap(i) for i, name in enumerate(sensor_names)}

    fig, ax = plt.subplots(figsize=(7, 4))
    for col in weight_columns:
        sensor_name = col.split("_")[1]
        color = color_map[sensor_name]
        ax.hist(dff_subset[col], bins=50, alpha=0.4, label=f"{sensor_name} actual", color=color)

        expected_weight_col = col.replace("_actual", "_expected")
        expected_weight = dff_subset[expected_weight_col].iloc[0]
        ax.axvline(expected_weight, color=color, linestyle="--", label=f"{sensor_name} expected")

    ax.set_title("Weight Distribution for Attention (Sub)Masks")
    ax.set_xlabel("Weight")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir, pkl_str, "attention_weights_distribution")


def _extract_numeric_values(series, max_values=200000):
    if series.empty:
        return np.array([])
    if pd.api.types.is_numeric_dtype(series):
        values = series.to_numpy()
        values = values[np.isfinite(values)]
        if values.size > max_values:
            idx = np.random.choice(values.size, size=max_values, replace=False)
            values = values[idx]
        return values

    first_valid = None
    for item in series:
        if item is not None and not (isinstance(item, float) and np.isnan(item)):
            first_valid = item
            break
    if first_valid is None:
        return np.array([])
    if isinstance(first_valid, (list, tuple, np.ndarray)):
        collected = []
        count = 0
        for item in series:
            if item is None:
                continue
            arr = np.asarray(item).ravel()
            if arr.size == 0:
                continue
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            if count + arr.size > max_values:
                remaining = max_values - count
                if remaining <= 0:
                    break
                arr = arr[:remaining]
            collected.append(arr)
            count += arr.size
            if count >= max_values:
                break
        if not collected:
            return np.array([])
        return np.concatenate(collected, axis=0)
    return np.array([])


def run_column_histograms(
    dff,
    output_dir,
    pkl_str,
    *,
    max_rows=200000,
    max_columns=80,
    bins=60,
    max_values_per_col=200000,
):
    dff_subset = sample_dff(dff, max_rows=max_rows)
    plotted = 0

    for col in dff_subset.columns:
        if plotted >= max_columns:
            print(f"Reached max_columns={max_columns}.")
            break
        try:
            series = dff_subset[col].dropna()
            if series.empty:
                continue
            values = _extract_numeric_values(series, max_values=max_values_per_col)
            if values.size == 0:
                continue
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(values, bins=bins, color="steelblue", alpha=0.8)
            ax.set_title(f"Histogram: {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            fig.tight_layout()
            suffix = f"hist_{_sanitize_suffix(col)}"
            save_figure(fig, output_dir, pkl_str, suffix)
            plotted += 1
        except Exception as exc:
            print(f"Column histogram failed for {col}: {exc}")


# NOTE: Kind of a hack -- the variable names should have been informative in the first place
def _angle_to_radians(angle_series):
    if angle_series.abs().max() > 2 * np.pi:
        return np.radians(angle_series)
    return angle_series


def run_sensing_by_target_location(
    dff,
    output_dir,
    pkl_str,
    morm_indices,
    amp_indices,
    use_signed_max=False,
    target_types=("food", "agent"),
    sensor_model=None,
    rotate_agent_up=True,
):
    plot_inputs = prepare_sensing_by_target_location(
        dff,
        morm_indices,
        amp_indices,
        use_signed_max=use_signed_max,
        target_types=target_types,
        sensor_model=sensor_model,
        rotate_agent_up=rotate_agent_up,
    )
    plot_sensing_by_target_location_spatial_conditional_expectation(plot_inputs, output_dir, pkl_str)
    plot_sensing_by_target_location_scatter(plot_inputs, output_dir, pkl_str)


def prepare_sensing_by_target_location(
    dff,
    morm_indices,
    amp_indices,
    use_signed_max=False,
    target_types=("food", "agent"),
    sensor_model=None,
    rotate_agent_up=True,
):
    if len(morm_indices) == 0 or len(amp_indices) == 0:
        print("Skipping sensing_by_target_location (missing sensor indices).")
        return []

    dff_subset = sample_dff(dff, max_rows=200000)
    dff_subset = dff_subset.dropna(subset=["observations", "emit_eod"])
    plot_inputs = []

    for target in target_types:
        if target == "food":
            distance_col = "distance_to_closest_food"
            angle_col = "angle_to_closest_food"
            target_label = "food"
            default_threshold_cm = AGENT_PARAMS["morm_food_detection_range_m"] * m_to_cm + 2.0
        elif target == "agent":
            distance_col = "distance_to_nearest_agent"
            angle_col = "angle_to_closest_agent"
            target_label = "agent"
            default_threshold_cm = AGENT_PARAMS["morm_agent_detection_range_m"] * m_to_cm + 2.0
        else:
            print(f"Skipping unknown target type: {target}")
            continue

        required_cols = {distance_col, angle_col, "observations", "emit_eod"}
        if not required_cols.issubset(dff_subset.columns):
            print(f"Skipping {target_label} (missing columns: {required_cols}).")
            continue

        dff_target = dff_subset.dropna(subset=[distance_col, angle_col, "emit_eod"]).copy()
        close_mask = dff_target[distance_col] <= default_threshold_cm
        dff_close = dff_target[close_mask].copy()
        if dff_close.empty:
            print(f"No rows with {target_label} within distance threshold.")
            continue

        angle_rad = _angle_to_radians(dff_close[angle_col])
        if rotate_agent_up:
            angle_rad = angle_rad + (np.pi / 2.0)
        rel_x = dff_close[distance_col] * np.cos(angle_rad)
        rel_y = dff_close[distance_col] * np.sin(angle_rad)

        obs = np.stack(dff_close["observations"].values, axis=0)
        morm_indices_use = morm_indices
        if sensor_model == "fracrand":
            num_morm_real = AGENT_PARAMS["num_rays"]
            morm_indices_use = morm_indices[:num_morm_real]
        morm_obs = obs[:, morm_indices_use]
        amp_obs = obs[:, amp_indices]
        if use_signed_max:
            morm_argmax = np.argmax(np.abs(morm_obs), axis=1)
            amp_argmax = np.argmax(np.abs(amp_obs), axis=1)
            max_morm = morm_obs[np.arange(morm_obs.shape[0]), morm_argmax]
            max_amp = amp_obs[np.arange(amp_obs.shape[0]), amp_argmax]
        else:
            max_morm = np.max(np.abs(morm_obs), axis=1)
            max_amp = np.max(np.abs(amp_obs), axis=1)

        for is_eod in [True, False]:
            label = "self_eod" if is_eod else "no_self_eod"
            mask = dff_close["emit_eod"] == is_eod
            if not np.any(mask):
                print(f"No rows for condition: {label} ({target_label}).")
                continue
            plot_inputs.append(
                {
                    "target_label": target_label,
                    "label": label,
                    "rel_x": rel_x[mask],
                    "rel_y": rel_y[mask],
                    "use_signed_max": use_signed_max,
                    "morm_values": max_morm[mask],
                    "amp_values": max_amp[mask],
                }
            )

    return plot_inputs


def plot_sensing_by_target_location_scatter(plot_inputs, output_dir, pkl_str):
    for item in plot_inputs:
        target_label = item["target_label"]
        label = item["label"]
        rel_x = item["rel_x"]
        rel_y = item["rel_y"]
        use_signed_max = item["use_signed_max"]

        for sensor_name, magnitudes in [
            ("mormyromast", item["morm_values"]),
            ("ampullary", item["amp_values"]),
        ]:
            fig, ax = plt.subplots(figsize=(4.5, 4))
            vals = magnitudes
            vlim = float(np.nanmax(np.abs(vals))) if len(vals) else 1.0
            cmap = "bwr" if use_signed_max else "Greens"
            vmin = -vlim if use_signed_max else 0.0
            vmax = vlim
            sc = ax.scatter(
                rel_x,
                rel_y,
                c=vals,
                s=8,
                alpha=0.25,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(f"Relative X to {target_label} (cm)")
            ax.set_ylabel(f"Relative Y to {target_label} (cm)")
            title_metric = "signed max |obs|" if use_signed_max else "max |obs|"
            ax.set_title(f"{sensor_name} {title_metric} ({label})")
            cbar_label = "Sensor value" if use_signed_max else "Max |sensor|"
            fig.colorbar(sc, ax=ax, label=cbar_label)
            fig.tight_layout()
            save_figure(
                fig,
                output_dir,
                pkl_str,
                f"sensing_{target_label}_{sensor_name}_{label}",
            )

def plot_sensing_by_target_location_spatial_conditional_expectation(
    plot_inputs,
    output_dir,
    pkl_str,
    *,
    gridsize=20,           # hexbin grid size (larger = finer grid)   
    reduce_fn="mean",            # "mean" or "median"
    mincnt=10,                   # minimum samples per hexbin
    robust_clim=True,            # percentile-based color limits
    clim_pct=(1, 99),            # (low, high) percentiles for unsigned
    clim_pct_signed=99,          # symmetric percentile for signed
    make_count_plot=False,       # optionally also save counts per bin
):
    """
    Spatial conditional expectation plots:
      color(x, y) = E[ sensor_summary | (x, y) in hexbin ].

    Assumes each item in plot_inputs contains:
      - target_label, label, use_signed_max
      - rel_x, rel_y (array-like, same length)
      - morm_values, amp_values (np.ndarray, same length)
    """

    import numpy as np
    import matplotlib.pyplot as plt

    reduce_map = {
        "mean": np.nanmean,
        "median": np.nanmedian,
    }
    if reduce_fn not in reduce_map:
        raise ValueError(f"reduce_fn must be one of {list(reduce_map)}, got {reduce_fn}")
    reduce_C_function = reduce_map[reduce_fn]

    def _as_numpy(a):
        # Supports pandas Series / numpy arrays
        return a.to_numpy() if hasattr(a, "to_numpy") else np.asarray(a)

    def _robust_limits(vals, use_signed):
        vals = np.asarray(vals)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return (0.0, 1.0) if not use_signed else (-1.0, 1.0)

        if use_signed:
            # symmetric about 0
            v = np.nanpercentile(np.abs(vals), clim_pct_signed) if robust_clim else np.nanmax(np.abs(vals))
            if not np.isfinite(v) or v <= 0:
                v = 1.0
            return (-float(v), float(v))
        else:
            if robust_clim:
                lo, hi = np.nanpercentile(vals, clim_pct[0]), np.nanpercentile(vals, clim_pct[1])
                # ensure sensible limits
                if not np.isfinite(lo):
                    lo = 0.0
                if not np.isfinite(hi) or hi <= lo:
                    hi = float(np.nanmax(vals)) if np.isfinite(np.nanmax(vals)) else (lo + 1.0)
                return (float(lo), float(hi))
            else:
                hi = float(np.nanmax(vals))
                if not np.isfinite(hi) or hi <= 0:
                    hi = 1.0
                return (0.0, hi)

    for item in plot_inputs:
        target_label = item["target_label"]
        label = item["label"]
        use_signed_max = bool(item["use_signed_max"])

        x = _as_numpy(item["rel_x"])
        y = _as_numpy(item["rel_y"])

        for sensor_name, vals_raw in [
            ("mormyromast", item["morm_values"]),
            ("ampullary", item["amp_values"]),
        ]:
            vals = _as_numpy(vals_raw)

            # Ensure aligned lengths and finite coords
            n = min(len(x), len(y), len(vals))
            if n == 0:
                continue
            x_n, y_n, v_n = x[:n], y[:n], vals[:n]
            good = np.isfinite(x_n) & np.isfinite(y_n) & np.isfinite(v_n)
            if not np.any(good):
                continue
            xg, yg, vg = x_n[good], y_n[good], v_n[good]

            cmap = "bwr" if use_signed_max else "Greens"
            vmin, vmax = _robust_limits(vg, use_signed_max)

            fig, ax = plt.subplots(figsize=(4.5, 4))

            hb = ax.hexbin(
                xg,
                yg,
                C=vg,
                reduce_C_function=reduce_C_function,
                gridsize=gridsize,
                mincnt=mincnt,
                linewidths=0.0,
                cmap=cmap,
            )
            hb.set_clim(vmin, vmax)

            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(f"Relative X to {target_label} (cm)")
            ax.set_ylabel(f"Relative Y to {target_label} (cm)")

            metric = (
                f"{reduce_fn} signed-max"
                if use_signed_max
                else f"{reduce_fn} max|obs|"
            )
            ax.set_title(f"{sensor_name} E[.|(x,y)] ({label})")

            cbar_label = "E[sensor summary]" if not use_signed_max else "E[signed sensor summary]"
            fig.colorbar(hb, ax=ax, label=cbar_label)

            fig.tight_layout()
            save_figure(
                fig,
                output_dir,
                pkl_str,
                f"sensing_hexbin_{target_label}_{sensor_name}_{label}_{reduce_fn}",
            )

            if make_count_plot:
                fig2, ax2 = plt.subplots(figsize=(4.5, 4))
                hb2 = ax2.hexbin(
                    xg,
                    yg,
                    gridsize=gridsize,
                    mincnt=1,
                    linewidths=0.0,
                )
                ax2.set_aspect("equal", adjustable="box")
                ax2.set_xlabel(f"Relative X to {target_label} (cm)")
                ax2.set_ylabel(f"Relative Y to {target_label} (cm)")
                ax2.set_title(f"{sensor_name} samples per hexbin ({label})")
                fig2.colorbar(hb2, ax=ax2, label="Count")
                fig2.tight_layout()
                save_figure(
                    fig2,
                    output_dir,
                    pkl_str,
                    f"sensing_hexbin_count_{target_label}_{sensor_name}_{label}",
                )


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flat_pkl_file",
        "--fpkl",
        dest="flat_pkl_file",
        type=str,
        required=True,
        help="Path to input pickle file",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="foraging",
        help="Task used by load_flat_pkl_file (e.g., foraging, 2f1p, homing).",
    )
    parser.add_argument(
        "--analyses",
        nargs="+",
        default=None,
        choices=AVAILABLE_ANALYSES + ["all"],
        help="Analyses to run (space-separated). Use 'all' for everything.",
    )
    parser.add_argument(
        "--sensor_model",
        type=str,
        default="fracrand",
        choices=["fracrand", "old"],
        help="Sensor model used for index lookup.",
    )
    parser.add_argument(
        "--num_agents",
        type=int,
        default=None,
        help="Override number of agents (default: inferred from data).",
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default="rollout_diagnostics",
        help="Subfolder under outputs folder for plots.",
    )
    return parser.parse_args(argv)



if __name__ == "__main__":
    argv = sys.argv[1:]
    args = parse_args(argv)
    analyses_to_run = resolve_analyses(args.analyses)

    dff, outputs_folder, pkl_str = ru.load_flat_pkl_file(
        args.flat_pkl_file, task=args.task
    )
    outputs_folder = os.path.join(outputs_folder, args.output_subdir)
    run_rollout_diagnostics_report(
        dff,
        outputs_folder,
        analyses_to_run=analyses_to_run,
        sensor_model=args.sensor_model,
        num_agents=args.num_agents,
        use_signed_max=True,
        rotate_agent_up=True,
    )

"""
@ Old model example usage:
fpkl="/srv/marl/$USER/marl_fish/SJY_20250728_193228/outputs/MAFish_neural_20250728_193228_m1a1k1cs1sd0patchydiverse_agg_flattened.pkl"
python utils_rollout_diagnostics.py --flat_pkl_file $fpkl --sensor_model old --task foraging --analyses all


fpkl=results/20260121Eat20NoCurriculum1MOrder1LinearX2.0AngularX2.0Gamma0.995DCL10TD0.0PR1UR0NP0A1K1M1GRUSeed1/20260121_122841/outputs/MAFish_neural_20260121_122841_75H9uWJZ_0_raw.pkl
python utils_rollout_diagnostics.py --flat_pkl_file $fpkl --sensor_model fracrand --task foraging --analyses all

"""
