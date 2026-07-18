import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon
from scipy.stats import gaussian_kde

from statannotations.Annotator import Annotator

import cfg
from utils_figstyle import set_nature_style
from utils_figsaving import _load_data, _save_data, _make_fig
from utils_features import add_morm_amp_field_features
from utils_behavior import calculate_gini, calculate_polarization, calculate_cohesion, calculate_theil_index_last

FISH_CONSTANTS = cfg.FISH_CONSTANTS
OBJECT_TYPES = cfg.OBJECT_TYPES
ENV_PARAMS = cfg.ENV_PARAMS
AGENT_PARAMS = cfg.AGENT_PARAMS
REWARDS = cfg.REWARDS
COLORS = cfg.COLORS
m_to_cm = cfg.m_to_cm
_NEAR_FOOD_MARGIN_CM = AGENT_PARAMS['morm_food_detection_range_m'] * m_to_cm

set_nature_style()

def get_food_and_size_stats(dff):
    grouped = dff.groupby(['env_id', 'episode_index', 'agent_id']).agg(
        food_eaten=('eating_event', 'sum'),
        agent_size=('agent_size', 'first')  # assuming size is constant per agent in a given episode
    ).reset_index()

    food_eaten_per_agent = grouped['food_eaten'].tolist()
    agent_sizes = grouped['agent_size'].tolist()

    food_eaten_agent_size_correlation = np.corrcoef(food_eaten_per_agent, agent_sizes)[0, 1]

    return food_eaten_per_agent, agent_sizes, food_eaten_agent_size_correlation


def compute_eod_rates(dff, timesteps_per_second=ENV_PARAMS['fps_sim'], distance_threshold=AGENT_PARAMS['morm_agent_detection_range_m'] * m_to_cm):
    """
    For each (env_id, episode_index, agent_id):
      - Overall EOD rate in Hz
      - EOD rate when near another agent (distance < distance_threshold)
      - EOD rate when far from other agents (distance >= distance_threshold)

    Returns a DataFrame with columns:
      [env_id, episode_index, agent_id, agent_size,
       eod_count, total_timesteps, eod_rate_hz,
       eod_rate_hz_close, eod_rate_hz_far]
    """
    dff["is_close"] = dff["distance_to_nearest_agent"] < distance_threshold

    # Group at the (env_id, episode_index, agent_id) level
    # Summarize how many EODs, how many timesteps, and how many close steps, etc.
    grouped = dff.groupby(["env_id", "episode_index", "agent_id"], as_index=False)

    # We build a summary DataFrame
    summary = grouped.agg(
        agent_size=("agent_size", "last"),  # size is typically constant across an episode
        eod_count=("emit_eod", "sum"),
        total_timesteps=("time_step", "count"),
        close_timesteps=("is_close", "sum"),
        close_eod_count=("emit_eod", lambda x: np.sum(x & dff.loc[x.index, "is_close"]))
    )

    # Overall EOD rate
    summary["eod_rate_hz"] = (summary["eod_count"] / summary["total_timesteps"]) * timesteps_per_second

    # EOD rate for close range (only among the "close" steps)
    # If no close steps occur, we set it to NaN
    summary["eod_rate_hz_close"] = np.where(
        summary["close_timesteps"] > 0,
        (summary["close_eod_count"] / summary["close_timesteps"]) * timesteps_per_second,
        np.nan
    )

    # EOD rate for far range
    # The number of far steps = total_timesteps - close_timesteps
    summary["far_timesteps"] = summary["total_timesteps"] - summary["close_timesteps"]
    summary["far_eod_count"] = summary["eod_count"] - summary["close_eod_count"]
    summary["eod_rate_hz_far"] = np.where(
        summary["far_timesteps"] > 0,
        (summary["far_eod_count"] / summary["far_timesteps"]) * timesteps_per_second,
        np.nan
    )

    return summary


def get_arena_stats(dff):
    stats_df = {}
    stats_df['num_agents'] = dff["agent_id"].nunique()
    stats_df['num_envs'] = dff["env_id"].nunique()
    stats_df['num_episodes'] = dff["episode_index"].nunique()
    stats_df['num_time_steps'] = dff["time_step"].nunique()
    stats_df['num_total_time_steps'] = stats_df['num_envs'] * stats_df['num_episodes'] * stats_df['num_time_steps']

    stats_df['food_eaten'] = dff['eating_event'].sum()
    stats_df['food_eaten_per_agent_timestep'] = stats_df['food_eaten'] / (stats_df['num_agents'] * stats_df['num_total_time_steps'])
    _, _, food_size_coeff = get_food_and_size_stats(dff)
    stats_df['food_eaten_agent_size_correlation'] = food_size_coeff

    stats_df['percent_emit_eod'] = dff["emit_eod"].mean()
    stats_df['percent_has_nearby_emit_eod'] = dff["nearby_emitting"].mean()
    stats_df['percent_freeloading'] = (~dff["emit_eod"] & dff["nearby_emitting"]).mean()
    stats_df['percent_emit_eod_despite_collective_sensing'] = (dff["emit_eod"] & dff["nearby_emitting"]).mean()

    stats_df['percent_has_nearby'] = dff['has_nearby'].mean()
    stats_df['num_interactions'] = dff['has_nearby'].sum()  # NOTE: not unique; same meeting event can be counted multiple times
    # need to look at meeting events and see if they contain bites or not

    stats_df['num_biting_events'] = dff['was_bitten'].sum()
    stats_df['num_biting_attempts'] = dff['bite_action'].sum()
    stats_df['bite_success_rate'] = stats_df['num_biting_events'] / stats_df['num_biting_attempts']

    bitten_events = dff[dff['was_bitten']].copy()
    biter_sizes = dff[['agent_id', 'agent_size', 'episode_index', 'env_id', 'time_step']].rename(
        columns={'agent_id': 'was_bitten_by_agent_id', 'agent_size': 'agent_size_biter'}
    )
    bitten_events = bitten_events.merge(
        biter_sizes,
        on=['was_bitten_by_agent_id', 'episode_index', 'env_id', 'time_step'],
        how='left'
    )
    bitten_events.rename(columns={'agent_size': 'agent_size_bitee'}, inplace=True)

    stats_df['num_smaller_bite_larger'] = (bitten_events['agent_size_biter'] < bitten_events['agent_size_bitee']).sum()
    stats_df['num_larger_bite_smaller'] = (bitten_events['agent_size_biter'] > bitten_events['agent_size_bitee']).sum()
    stats_df['num_bite_same_size_within_0.1'] = (bitten_events['agent_size_biter'] - bitten_events['agent_size_bitee']).abs().lt(0.1).sum()
    stats_df['average_biter_to_bitee_size_diff'] = (bitten_events['agent_size_biter'] - bitten_events['agent_size_bitee']).mean()
    stats_df['average_num_agents_in_knollen_range'] = dff['num_agents_in_knollen_range'].mean()

    stats_df['mean_displacement'] = dff['displacement'].mean()
    stats_df['std_displacement'] = dff['displacement'].std()
    stats_df['25th_percentile_displacement'] = dff['displacement'].quantile(0.25)
    stats_df['75th_percentile_displacement'] = dff['displacement'].quantile(0.75)

    if 'collided' in dff.columns:
        stats_df['percent_collided'] = dff['collided'].mean()

    eod_rate_summary = compute_eod_rates(dff, timesteps_per_second=ENV_PARAMS['fps_sim'], distance_threshold=AGENT_PARAMS['morm_agent_detection_range_m'] * m_to_cm)
    corr, pval = stats.pearsonr(eod_rate_summary["agent_size"], eod_rate_summary["eod_rate_hz"])
    stats_df['eod_rate_agent_size_correlation'] = corr
    stats_df['eod_rate_agent_size_correlation_pval'] = pval

    # TODO: add back in
    # corr_close, pval_close = stats.pearsonr(eod_rate_summary["agent_size"], eod_rate_summary["eod_rate_hz_close"])
    # corr_far, pval_far = stats.pearsonr(eod_rate_summary["agent_size"], eod_rate_summary["eod_rate_hz_far"])
    # stats_df['eod_rate_agent_size_correlation_close'] = corr_close
    # stats_df['eod_rate_agent_size_correlation_pval_close'] = pval_close
    # stats_df['eod_rate_agent_size_correlation_far'] = corr_far
    # stats_df['eod_rate_agent_size_correlation_pval_far'] = pval_far

    return stats_df


def get_stats_df(dff):
    stats_data = {}
    stats_data['All'] = get_arena_stats(dff)

    dff['arena_type'] = dff.groupby(['env_id', 'episode_index'])['arena_type'].ffill()
    for arena_type in dff['arena_type'].dropna().unique():
        arena_df = dff[dff['arena_type'] == arena_type]
        stats_data[arena_type] = get_arena_stats(arena_df)

    stats_df = pd.DataFrame(stats_data)
    return stats_df



####### Summary DF Creation ########

# ----------------------------
# Core utilities
# ----------------------------
def add_run_id(dff: pd.DataFrame) -> pd.DataFrame:
    """
    Adds integer run_id with env as the outside loop:
        run_id = env_rank * num_episodes + episode_rank
    """
    out = dff.copy()

    envs = sorted(out["env_id"].dropna().unique().tolist())
    eps = sorted(out["episode_index"].dropna().unique().tolist())

    env_map = {e: i for i, e in enumerate(envs)}
    ep_map = {ep: i for i, ep in enumerate(eps)}

    num_episodes = len(eps)
    out["run_id"] = (
        out["env_id"].map(env_map).astype("int64") * num_episodes
        + out["episode_index"].map(ep_map).astype("int64")
    )
    return out


def first_non_null(s: pd.Series):
    s2 = s.dropna()
    return s2.iloc[0] if len(s2) else np.nan


def mode_or_first(s: pd.Series):
    s2 = s.dropna()
    if len(s2) == 0:
        return np.nan
    m = s2.mode()
    return m.iloc[0] if len(m) else s2.iloc[0]


def initial_final_by_time(
    group_df: pd.DataFrame,
    value_col: str,
    time_col: str = "time_step",
):
    try:
        if value_col not in group_df.columns or time_col not in group_df.columns:
            return np.nan, np.nan

        tmp = group_df[[time_col, value_col]].dropna(subset=[value_col])
        if tmp.empty:
            return np.nan, np.nan

        t0 = tmp[time_col].min()
        t1 = tmp[time_col].max()
        v0 = tmp.loc[tmp[time_col] == t0, value_col].iloc[0]
        v1 = tmp.loc[tmp[time_col] == t1, value_col].iloc[-1]
        return float(v0), float(v1)
    except Exception:
        return np.nan, np.nan


def round_df_smart(df: pd.DataFrame, decimals: int = 3, tiny_threshold: float = 1e-3) -> pd.DataFrame:
    """
    Round numeric columns:
    - standard numbers rounded to `decimals`
    - very small magnitudes keep more precision so they don't all become 0.000
    """
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns

    for c in num_cols:
        x = out[c].to_numpy(dtype=float)
        # keep ints as ints when possible
        if np.all(np.isfinite(x)) and np.all(np.equal(np.mod(x, 1), 0)):
            # all integer-valued
            out[c] = out[c].astype("Int64")
            continue

        # smart rounding for floats
        def _round_val(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return v
            av = abs(float(v))
            if av != 0 and av < tiny_threshold:
                # keep more digits for tiny values
                return float(np.round(v, max(decimals, 6)))
            return float(np.round(v, decimals))

        out[c] = out[c].apply(_round_val)

    return out


# ----------------------------
# Metrics helpers (run-level)
# ----------------------------
def run_average_pairwise_distances(run_df: pd.DataFrame) -> float:
    """
    Run-level: mean over timesteps of average pairwise distance between agents.
    """
    if "position" not in run_df.columns or "time_step" not in run_df.columns:
        return np.nan

    g = run_df.groupby("time_step")["position"].apply(list)
    per_t = []
    for pos_list in g:
        if len(pos_list) < 2:
            continue
        P = np.asarray(pos_list, dtype=float)
        D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        np.fill_diagonal(D, np.nan)
        per_t.append(np.nanmean(D))
    return float(np.mean(per_t)) if len(per_t) else np.nan


def run_wall_metrics(run_df: pd.DataFrame, close_to_wall_margin=5.0):
    """
    Run-level: p_near_wall and mean distance-to-wall (over all agent-timesteps).
    """
    needed = {"position_x", "position_y", "arena_size"}
    if not needed.issubset(set(run_df.columns)):
        return np.nan, np.nan

    arena_size = first_non_null(run_df["arena_size"])
    if isinstance(arena_size, float) and np.isnan(arena_size):
        return np.nan, np.nan

    w, h = arena_size
    x = run_df["position_x"].to_numpy(dtype=float)
    y = run_df["position_y"].to_numpy(dtype=float)
    dist = np.minimum.reduce([x, w - x, y, h - y]).astype(float)

    p_near_wall = float(np.mean(dist < close_to_wall_margin)) if len(dist) else np.nan
    mean_dist = float(np.mean(dist)) if len(dist) else np.nan
    return p_near_wall, mean_dist


def run_food_metrics(run_df: pd.DataFrame, near_food_margin=_NEAR_FOOD_MARGIN_CM):
    """
    Run-level: p_near_food and mean dist to nearest food (over all agent-timesteps).
    """
    if "distance_to_closest_food" not in run_df.columns:
        return np.nan, np.nan
    d = run_df["distance_to_closest_food"].to_numpy(dtype=float)
    p = float(np.mean(d < near_food_margin)) if len(d) else np.nan
    m = float(np.mean(d)) if len(d) else np.nan
    return p, m


def run_n_step_displacement_mean(run_df: pd.DataFrame, n=20) -> float:
    """
    Run-level: mean over agents of n-step displacement mean (matches comparison spirit).
    """
    if "position" not in run_df.columns:
        return np.nan
    tmp = run_df.copy()
    tmp["position"] = tmp["position"].apply(lambda x: np.array(x, dtype=float))
    tmp["disp_vec"] = tmp.groupby("agent_id")["position"].diff(n)
    tmp["disp"] = tmp["disp_vec"].apply(
        lambda v: float(np.linalg.norm(v)) if isinstance(v, np.ndarray) else np.nan
    )
    per_agent = tmp.groupby("agent_id")["disp"].mean()
    return float(per_agent.mean()) if len(per_agent) else np.nan

def _round_arena_size(x, ndigits=2):
    if x is None:
        return np.nan
    if isinstance(x, float) and np.isnan(x):
        return np.nan
    try:
        w, h = x
        return (round(float(w), ndigits), round(float(h), ndigits))
    except Exception:
        return x

# ----------------------------
# DF builders
# ----------------------------
def build_agent_run_df(
    dff: pd.DataFrame,
    displacement_n: int = 20,
    close_to_wall_margin: float = 5.0,
    near_food_margin: float = _NEAR_FOOD_MARGIN_CM,
) -> pd.DataFrame:
    """
    Aggregated over timesteps, agents separate.
    One row per (run_id, env_id, episode_index, agent_id).
    """
    out = add_run_id(dff)
    run_keys = ["run_id", "env_id", "episode_index"]
    keys = run_keys + ["agent_id"]

    # base agent aggregation
    agg = {}
    if "displacement" in out.columns:
        agg["mean_displacement"] = ("displacement", "mean")
        agg["std_displacement"] = ("displacement", "std")
    if "emit_eod" in out.columns:
        agg["p_emit_eod"] = ("emit_eod", "mean")
    if "has_nearby" in out.columns:
        agg["p_nearby"] = ("has_nearby", "mean")
    if "eating_event" in out.columns:
        agg["food_eaten"] = ("eating_event", "sum")
    if "meeting_event" in out.columns:
        agg["num_interactions"] = ("meeting_event", "sum")
    elif "has_nearby" in out.columns:
        agg["num_interactions"] = ("has_nearby", "sum")
    if "was_bitten" in out.columns:
        agg["num_biting_events"] = ("was_bitten", "sum")
    if "distance_to_nearest_agent" in out.columns:
        agg["distance_to_nearest_agent"] = ("distance_to_nearest_agent", "mean")
        agg["distance_to_nearest_agent_min"] = ("distance_to_nearest_agent", "min")
        agg["distance_to_nearest_agent_max"] = ("distance_to_nearest_agent", "max")
    if "distance_to_closest_food" in out.columns:
        agg["distance_to_closest_food"] = ("distance_to_closest_food", "mean")

    df_agent = out.groupby(keys, sort=False).agg(**agg).reset_index()

    if "distance_to_nearest_agent" in out.columns and "time_step" in out.columns:
        try:
            dist_initial_final = (
                out.groupby(keys, sort=False)
                .apply(
                    lambda g: pd.Series(
                        initial_final_by_time(g, "distance_to_nearest_agent"),
                        index=[
                            "distance_to_nearest_agent_initial",
                            "distance_to_nearest_agent_final",
                        ],
                    )
                )
                .reset_index()
            )
            df_agent = df_agent.merge(dist_initial_final, on=keys, how="left")
        except Exception:
            df_agent["distance_to_nearest_agent_initial"] = np.nan
            df_agent["distance_to_nearest_agent_final"] = np.nan
    else:
        df_agent["distance_to_nearest_agent_initial"] = np.nan
        df_agent["distance_to_nearest_agent_final"] = np.nan

    # agent_size (constant per agent per run): first non-null
    if "agent_size" in out.columns:
        sizes = out.groupby(keys, sort=False)["agent_size"].apply(first_non_null).rename("agent_size").reset_index()
        df_agent = df_agent.merge(sizes, on=keys, how="left")
    else:
        df_agent["agent_size"] = np.nan

    # arena_size (constant per run; duplicate across agents, rounded)
    if "arena_size" in out.columns:
        arena = (
            out.groupby(keys, sort=False)["arena_size"]
            .apply(lambda s: _round_arena_size(first_non_null(s)))
            .rename("arena_size")
            .reset_index()
        )
        df_agent = df_agent.merge(arena, on=keys, how="left")
    else:
        df_agent["arena_size"] = np.nan

    # p_eod_interactions per agent
    if "meeting_event" in out.columns and "emit_eod" in out.columns:
        sub = out[out["meeting_event"].astype(bool)]
        pei = sub.groupby(keys, sort=False)["emit_eod"].mean().rename("p_eod_interactions").reset_index()
        df_agent = df_agent.merge(pei, on=keys, how="left")
    else:
        df_agent["p_eod_interactions"] = np.nan

    # p_near_food per agent
    if "distance_to_closest_food" in out.columns:
        p_nf = (
            out.groupby(keys, sort=False)["distance_to_closest_food"]
            .apply(lambda s: float(np.mean(s.to_numpy(dtype=float) < near_food_margin)))
            .rename("p_near_food")
            .reset_index()
        )
        df_agent = df_agent.merge(p_nf, on=keys, how="left")
    else:
        df_agent["p_near_food"] = np.nan

    # wall metrics per agent
    if {"position_x", "position_y", "arena_size"}.issubset(set(out.columns)):
        wall_rows = []
        for (run_id, env_id, ep), run_df in out.groupby(run_keys, sort=False):
            arena_size = first_non_null(run_df["arena_size"])
            if isinstance(arena_size, float) and np.isnan(arena_size):
                continue
            w, h = arena_size
            x = run_df["position_x"].to_numpy(dtype=float)
            y = run_df["position_y"].to_numpy(dtype=float)
            dist = np.minimum.reduce([x, w - x, y, h - y]).astype(float)

            tmp = run_df[["agent_id"]].copy()
            tmp["dist_to_wall"] = dist
            pw = tmp.groupby("agent_id")["dist_to_wall"].apply(lambda s: float(np.mean(s < close_to_wall_margin)))
            dw = tmp.groupby("agent_id")["dist_to_wall"].mean()

            wall_rows.append(
                pd.DataFrame({
                    "run_id": int(run_id),
                    "env_id": env_id,
                    "episode_index": ep,
                    "agent_id": pw.index.values,
                    "p_near_wall": pw.values,
                    "distance_to_nearest_wall": dw.reindex(pw.index).values,
                })
            )
        wall_df = pd.concat(wall_rows, ignore_index=True) if wall_rows else pd.DataFrame(
            columns=keys + ["p_near_wall", "distance_to_nearest_wall"]
        )
        df_agent = df_agent.merge(wall_df, on=keys, how="left")
    else:
        df_agent["p_near_wall"] = np.nan
        df_agent["distance_to_nearest_wall"] = np.nan

    # n-step displacement per agent (mean)
    if "position" in out.columns:
        disp_rows = []
        for (run_id, env_id, ep), run_df in out.groupby(run_keys, sort=False):
            tmp = run_df.copy()
            tmp["position"] = tmp["position"].apply(lambda x: np.array(x, dtype=float))
            tmp["disp_vec"] = tmp.groupby("agent_id")["position"].diff(displacement_n)
            tmp["disp"] = tmp["disp_vec"].apply(
                lambda v: float(np.linalg.norm(v)) if isinstance(v, np.ndarray) else np.nan
            )
            per_agent = tmp.groupby("agent_id")["disp"].mean()
            disp_rows.append(
                pd.DataFrame({
                    "run_id": int(run_id),
                    "env_id": env_id,
                    "episode_index": ep,
                    "agent_id": per_agent.index.values,
                    f"displacement_{displacement_n}_step": per_agent.values,
                })
            )
        disp_df = pd.concat(disp_rows, ignore_index=True) if disp_rows else pd.DataFrame(
            columns=keys + [f"displacement_{displacement_n}_step"]
        )
        df_agent = df_agent.merge(disp_df, on=keys, how="left")
    else:
        df_agent[f"displacement_{displacement_n}_step"] = np.nan

    # convenience alias from comparison script
    # p_near_other_fish in your comparisons is basically p(has_nearby)
    if "p_nearby" in df_agent.columns:
        df_agent["p_near_other_fish"] = df_agent["p_nearby"]
    else:
        df_agent["p_near_other_fish"] = np.nan

    # passthrough: any rw_* columns (constant per episode, take first value per group)
    rw_passthrough = [c for c in out.columns if c.startswith("rw_")]
    if rw_passthrough:
        rw_df = (
            out.groupby(keys, sort=False)[rw_passthrough]
            .first()
            .reset_index()
        )
        df_agent = df_agent.merge(rw_df, on=keys, how="left")

    return df_agent


def build_run_df(
    dff: pd.DataFrame,
    displacement_n: int = 20,
    close_to_wall_margin: float = 5.0,
    near_food_margin: float = _NEAR_FOOD_MARGIN_CM,
) -> pd.DataFrame:
    """
    Aggregated over timesteps AND agents.
    One row per (run_id, env_id, episode_index).
    This is where the “comparison script” metrics live (run-level distributions collapsed).
    """
    out = add_run_id(dff)
    run_keys = ["run_id", "env_id", "episode_index"]

    rows = []
    for (run_id, env_id, ep), run_df in out.groupby(run_keys, sort=False):
        r = {
            "run_id": int(run_id),
            "env_id": env_id,
            "episode_index": ep,
            "num_time_steps": int(run_df["time_step"].nunique()),
            "num_agents": int(run_df["agent_id"].nunique()),
        }

        # arena_type without ffill
        r["arena_type"] = mode_or_first(run_df["arena_type"]) if "arena_type" in run_df.columns else np.nan
        r["arena_size"] = mode_or_first(run_df["arena_size"]) if "arena_size" in run_df.columns else np.nan
        r["arena_size"] = _round_arena_size(r["arena_size"], ndigits=2)

        # core comparison metrics
        r["food_eaten"] = float(run_df["eating_event"].sum()) if "eating_event" in run_df.columns else np.nan
        r["p_emit_eod"] = float(run_df["emit_eod"].mean()) if "emit_eod" in run_df.columns else np.nan

        if "meeting_event" in run_df.columns:
            r["num_interactions"] = float(run_df["meeting_event"].sum())
            if "emit_eod" in run_df.columns:
                sub = run_df[run_df["meeting_event"].astype(bool)]
                r["p_eod_interactions"] = float(sub["emit_eod"].mean()) if len(sub) else np.nan
            else:
                r["p_eod_interactions"] = np.nan
        elif "has_nearby" in run_df.columns:
            r["num_interactions"] = float(run_df["has_nearby"].sum())
            r["p_eod_interactions"] = np.nan
        else:
            r["num_interactions"] = np.nan
            r["p_eod_interactions"] = np.nan

        if "has_nearby" in run_df.columns:
            r["p_near_other_fish"] = float(run_df["has_nearby"].mean())
        else:
            r["p_near_other_fish"] = np.nan

        # arena-normalized interaction (run-level)
        if "has_nearby" in run_df.columns and "arena_size" in run_df.columns:
            arena_size = first_non_null(run_df["arena_size"])
            if not (isinstance(arena_size, float) and np.isnan(arena_size)):
                w, h = arena_size
                area = float(w) * float(h)
                r["p_near_other_fish_arena_norm"] = float(run_df["has_nearby"].mean()) / area if area else np.nan
            else:
                r["p_near_other_fish_arena_norm"] = np.nan
        else:
            r["p_near_other_fish_arena_norm"] = np.nan

        # wall + food metrics (run-level)
        pw, dw = run_wall_metrics(run_df, close_to_wall_margin=close_to_wall_margin)
        r["p_near_wall"] = pw
        r["distance_to_nearest_wall"] = dw

        pf, df_ = run_food_metrics(run_df, near_food_margin=near_food_margin)
        r["p_near_food"] = pf
        r["distance_to_closest_food"] = df_

        # nearest-agent distance metrics (run-level)
        if "distance_to_nearest_agent" in run_df.columns:
            try:
                dist_vals = pd.to_numeric(run_df["distance_to_nearest_agent"], errors="coerce")
                r["distance_to_nearest_agent"] = float(dist_vals.mean())
                r["distance_to_nearest_agent_min"] = float(dist_vals.min())
                r["distance_to_nearest_agent_max"] = float(dist_vals.max())
            except Exception:
                r["distance_to_nearest_agent"] = np.nan
                r["distance_to_nearest_agent_min"] = np.nan
                r["distance_to_nearest_agent_max"] = np.nan
        else:
            r["distance_to_nearest_agent"] = np.nan
            r["distance_to_nearest_agent_min"] = np.nan
            r["distance_to_nearest_agent_max"] = np.nan

        if "distance_to_nearest_agent" in run_df.columns and "time_step" in run_df.columns:
            try:
                d0, d1 = initial_final_by_time(run_df, "distance_to_nearest_agent")
                r["distance_to_nearest_agent_initial"] = d0
                r["distance_to_nearest_agent_final"] = d1
            except Exception:
                r["distance_to_nearest_agent_initial"] = np.nan
                r["distance_to_nearest_agent_final"] = np.nan
        else:
            r["distance_to_nearest_agent_initial"] = np.nan
            r["distance_to_nearest_agent_final"] = np.nan

        # average pairwise distances (run-level)
        r["average_pairwise_distances"] = run_average_pairwise_distances(run_df)

        # n-step displacement (run-level)
        r[f"displacement_{displacement_n}_step"] = run_n_step_displacement_mean(run_df, n=displacement_n)

        # biting events (run-level)
        r["num_biting_events"] = float(run_df["was_bitten"].sum()) if "was_bitten" in run_df.columns else np.nan

        # Theil index (run-level; inequality across agents, so belongs here)
        r["food_eaten_theil"] = np.nan
        try:
            if "eating_event" in run_df.columns:
                theil_df, _ = calculate_theil_index_last(run_df)
                r["food_eaten_theil"] = float(theil_df["theil_index"].iloc[-1]) if len(theil_df) else np.nan
        except Exception:
            pass

        # mean_nn_distance_cm/polarization (run-level)
        r["mean_nn_distance_cm"] = np.nan
        r["polarization"] = np.nan
        try:
            if "position" in run_df.columns:
                pos_groups = run_df.groupby("time_step")["position"].apply(list)
                coh_vals = pos_groups.apply(calculate_cohesion)
                r["mean_nn_distance_cm"] = float(coh_vals.mean()) if len(coh_vals) else np.nan
            if "orientation" in run_df.columns:
                ori_groups = run_df.groupby("time_step")["orientation"].apply(list)
                pol_vals = ori_groups.apply(calculate_polarization)
                # calculate_polarization raises ValueError for ≤2 agents; swallowed by outer except
                r["polarization"] = float(pol_vals.mean()) if len(pol_vals) else np.nan
        except Exception:
            pass

        # passthrough: any rw_* columns (constant per episode, take first value)
        for col in run_df.columns:
            if col.startswith("rw_"):
                r[col] = run_df[col].iloc[0]

        rows.append(r)

    return pd.DataFrame(rows)


def build_step_df(dff: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate over agents, keep time_step.
    One row per (env_id, episode_index, time_step) — agents collapsed.
    """
    out = add_run_id(dff)
    keys = ["run_id", "env_id", "episode_index", "time_step"]

    agg = {}
    if "displacement" in out.columns:
        agg["mean_displacement"] = ("displacement", "mean")
    if "emit_eod" in out.columns:
        agg["p_emit_eod"] = ("emit_eod", "mean")
        agg["n_emit_eod"] = ("emit_eod", "sum")
    if "eating_event" in out.columns:
        agg["food_eaten"] = ("eating_event", "sum")
    if "meeting_event" in out.columns:
        agg["n_interactions"] = ("meeting_event", "sum")
    elif "has_nearby" in out.columns:
        agg["n_nearby"] = ("has_nearby", "sum")
    if "distance_to_nearest_agent" in out.columns:
        agg["mean_distance_to_nearest_agent"] = ("distance_to_nearest_agent", "mean")
    if "distance_to_closest_food" in out.columns:
        agg["mean_distance_to_closest_food"] = ("distance_to_closest_food", "mean")
    if "agent_id" in out.columns:
        agg["num_agents"] = ("agent_id", "nunique")

    df_step = out.groupby(keys, sort=False).agg(**agg).reset_index()
    return df_step


def get_summary_dfs(
    dff: pd.DataFrame,
    displacement_n: int = 20,
    close_to_wall_margin: float = 5.0,
    near_food_margin: float = _NEAR_FOOD_MARGIN_CM,
    round_decimals: int = 3,
):
    """
    Returns:
      df_agent: one row per run×agent (timesteps aggregated)
      df_run:   one row per run (timesteps+agents aggregated; comparison metrics live here)
    Also applies smart rounding.
    """
    df_agent = build_agent_run_df(
        dff,
        displacement_n=displacement_n,
        close_to_wall_margin=close_to_wall_margin,
        near_food_margin=near_food_margin,
    )
    df_run = build_run_df(
        dff,
        displacement_n=displacement_n,
        close_to_wall_margin=close_to_wall_margin,
        near_food_margin=near_food_margin,
    )

    df_agent = round_df_smart(df_agent, decimals=round_decimals)
    df_run = round_df_smart(df_run, decimals=round_decimals)

    # nice column ordering
    agent_front = ["run_id", "env_id", "episode_index", "agent_id", "agent_size"]
    agent_rest = [c for c in df_agent.columns if c not in agent_front]
    df_agent = df_agent[[c for c in agent_front if c in df_agent.columns] + agent_rest]

    run_front = ["run_id", "env_id", "episode_index", "arena_type", "num_agents", "num_time_steps"]
    run_rest = [c for c in df_run.columns if c not in run_front]
    df_run = df_run[[c for c in run_front if c in df_run.columns] + run_rest]

    return df_agent, df_run


######### Plot-related functions #########
# def plot_aligned_movement_2d_histogram(dff, save_path=None):
#     """
#     Plot 2D histograms of movement patterns aligned with food/agent positions.
#     Cases:
#     1. Food & No Agent
#     2. Agent & No Food
#     3. Food & Agent
#     4. No Food & No Agent
#     """
#     # Create figure with a special layout for the colorbar
#     fig = plt.figure(figsize=(12, 10))

#     # Create a GridSpec layout with 2 rows, 3 columns
#     # The last column will be narrower and used for the colorbar
#     gs = plt.GridSpec(2, 3, width_ratios=[1, 1, 0.1])

#     # Create the four main subplot axes
#     axs = [[plt.subplot(gs[i, j]) for j in range(2)] for i in range(2)]

#     # Create the colorbar axis
#     cbar_ax = plt.subplot(gs[:, 2])

#     titles = ['Food & No Agent', 'Agent & No Food',
#              'Food & Agent', 'No Food & No Agent']
#     all_hist_values = []

#     agent_args = dff['metadata'].iloc[0]['agent_args']
#     dff['food_observed'] = dff['distance_to_closest_food'] < agent_args['morm_food_detection_range'] * cfg.m_to_cm
#     dff['agent_observed'] = dff['distance_to_nearest_agent'] < agent_args['morm_agent_detection_range'] * cfg.m_to_cm

#     # Create masks for each case
#     case_1_mask = dff['food_observed'] & ~dff['agent_observed']
#     case_2_mask = ~dff['food_observed'] & dff['agent_observed']
#     case_3_mask = dff['food_observed'] & dff['agent_observed']
#     case_4_mask = ~dff['food_observed'] & ~dff['agent_observed']

#     cases = [(case_1_mask, 0, 0), (case_2_mask, 0, 1),
#              (case_3_mask, 1, 0), (case_4_mask, 1, 1)]

#     # First pass to gather histogram values for normalization
#     for mask, i, j in cases:
#         case_data = dff[mask].copy()

#         # Calculate rotated movements
#         movements = np.zeros((len(case_data), 2))
#         for i_row, (_, row) in enumerate(case_data.iterrows()):
#             if row['food_observed']:
#                 angle = -(row['orientation']%(2*np.pi)+np.radians(row['angle_to_closest_food']))+np.pi/2
#             elif row['agent_observed']:
#                 angle = -(row['orientation']%(2*np.pi)+np.radians(row['angle_to_closest_agent_observed']))+np.pi/2
#             else:
#                 angle = 0

#             dx = row['displacement_x']
#             dy = row['displacement_y']
#             rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
#                                  [np.sin(angle), np.cos(angle)]])
#             movements[i_row] = np.dot(rot_matrix, np.array([dx, dy]))

#         valid_mask = (~np.isnan(movements).any(axis=1)) & (np.abs(movements) < 2).all(axis=1)
#         valid_movements = movements[valid_mask]

#         if len(valid_movements) > 0:
#             hist, _, _ = np.histogram2d(valid_movements[:, 0], valid_movements[:, 1],
#                                       bins=50, range=[[-2, 2], [-2, 2]], density=True)
#             all_hist_values.extend(hist.flatten())

#     # Calculate percentiles for consistent color scaling
#     vmin = np.percentile(all_hist_values, 1)
#     vmax = np.percentile(all_hist_values, 99.5)

#     # Second pass to create plots
#     for mask, i, j in cases:
#         ax = axs[i][j]
#         case_data = dff[mask].copy()

#         movements = np.zeros((len(case_data), 2))
#         for i_row, (_, row) in enumerate(case_data.iterrows()):
#             if row['food_observed']:
#                 angle = -(row['orientation']%(2*np.pi)+np.radians(row['angle_to_closest_food']))+np.pi/2
#             elif row['agent_observed']:
#                 angle = -(row['orientation']%(2*np.pi)+np.radians(row['angle_to_closest_agent_observed']))+np.pi/2
#             else:
#                 angle = 0

#             dx = row['displacement_x']
#             dy = row['displacement_y']
#             rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
#                                  [np.sin(angle), np.cos(angle)]])
#             movements[i_row] = np.dot(rot_matrix, np.array([dx, dy]))

#         valid_mask = (~np.isnan(movements).any(axis=1)) & (np.abs(movements) < 2).all(axis=1)
#         valid_movements = movements[valid_mask]

#         if len(valid_movements) > 0:
#             hist, xedges, yedges = np.histogram2d(valid_movements[:, 0], valid_movements[:, 1],
#                                                 bins=50, range=[[-2, 2], [-2, 2]], density=True)

#             im = ax.imshow(hist.T, origin='lower', extent=[-2, 2, -2, 2],
#                           cmap='Reds', vmin=vmin, vmax=vmax)

#         ax.set_title(f'{titles[i*2+j]}\nn={len(valid_movements)}')
#         ax.set_xlabel('Δx')
#         ax.set_ylabel('Δy')
#         # ax.grid(True, alpha=0.3)

#     # Add colorbar in its own subplot
#     plt.colorbar(im, cax=cbar_ax, label='Probability Density')

#     plt.tight_layout()
#     if save_path is not None:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         plt.close()

#     return fig, axs


def plot_aligned_movement_2d_histogram(
    dff,
    save_path=None,
    load_data=False,
    save_data=True
):
    """
    Plot 2D histograms of movement patterns aligned with food/agent positions.
    Cases:
      1. Food & No Agent
      2. Agent & No Food
      3. Food & Agent
      4. No Food & No Agent
    """
    base = os.path.splitext(save_path)[0] if save_path else None

    # --- Prepare the four masks & titles exactly as before ---
    if 'metadata' in dff.columns:
        agent_args = dff['metadata'].iloc[0]['agent_args']
        food_range_cm  = agent_args['morm_food_detection_range_m'] * cfg.m_to_cm
        agent_range_cm = agent_args['morm_agent_detection_range_m'] * cfg.m_to_cm
    else:
        # New Recorder format: no metadata column; fall back to cfg defaults
        food_range_cm  = AGENT_PARAMS['morm_food_detection_range_m'] * m_to_cm
        agent_range_cm = AGENT_PARAMS['morm_agent_detection_range_m'] * m_to_cm
    dff['food_observed']  = dff['distance_to_closest_food'] < food_range_cm
    dff['agent_observed'] = dff['distance_to_nearest_agent'] < agent_range_cm

    masks = [
      dff['food_observed'] & ~dff['agent_observed'],
      ~dff['food_observed'] & dff['agent_observed'],
      dff['food_observed'] & dff['agent_observed'],
      ~dff['food_observed'] & ~dff['agent_observed']
    ]
    locs   = [(0,0),(0,1),(1,0),(1,1)]
    titles = ['Food & No Agent','Agent & No Food','Food & Agent','No Food & No Agent']

    cases = []
    for idx, (mask, (i,j), t) in enumerate(zip(masks, locs, titles)):
        cases.append((mask, i, j, t))

    # --- Load or compute raw movements per case ---
    case_moves = None
    if load_data and base:
        try:
            case_moves = _load_data(base)
            # ← convert each list to np.array so .size works
            for k, v in case_moves.items():
                case_moves[k] = np.array(v)
        except Exception:
            print(f"Warning: no data to load for '{base}', recalculating.")
            case_moves = None

    if case_moves is None:
        case_moves = {}
        for idx, (mask, _, _, _) in enumerate(cases):
            sub = dff[mask]
            moves = []
            for _, row in sub.iterrows():
                ori = row['orientation'] % (2*np.pi)
                if row['food_observed']:
                    ang = -(ori + row['angle_to_closest_food']) + np.pi/2
                elif row['agent_observed']:
                    ang = -(ori + row['angle_to_closest_agent_observed']) + np.pi/2
                else:
                    ang = 0
                dx, dy = row['displacement_x'], row['displacement_y']
                R = np.array([
                    [np.cos(ang), -np.sin(ang)],
                    [np.sin(ang),  np.cos(ang)]
                ])
                mv = R.dot([dx, dy])
                if not np.any(np.isnan(mv)) and np.all(np.abs(mv) < 2):
                    moves.append(mv.tolist())
            case_moves[f'case{idx}'] = np.array(moves)
        if save_data and base:
            _save_data({k: v.tolist() for k,v in case_moves.items()}, base)

    # --- Gather vmin/vmax across all cases for consistent color scale ---
    all_h = []
    for idx in range(len(cases)):
        mv = case_moves[f'case{idx}']
        if mv.size:
            h, _, _ = np.histogram2d(
                mv[:,0], mv[:,1],
                bins=50, range=[[-1,1],[-1,1]],
                density=True
            )
            all_h.extend(h.flatten())
    vmin = np.percentile(all_h, 1) if all_h else None
    vmax = np.percentile(all_h, 99.5) if all_h else None

    fig = plt.figure(figsize=(6, 5))
    gs = plt.GridSpec(2, 3, width_ratios=[1, 1, 0.1])
    axs = [[plt.subplot(gs[i, j]) for j in range(2)] for i in range(2)]
    cbar_ax = plt.subplot(gs[:, 2])

    for idx, (mask, i, j, title) in enumerate(cases):
        ax = axs[i][j]
        mv = case_moves[f'case{idx}']
        if mv.size:
            hist = np.histogram2d(
                mv[:,0], mv[:,1],
                bins=50, range=[[-1,1],[-1,1]],
                density=True
            )[0]
            im = ax.imshow(
                hist.T, origin='lower',
                extent=[-1,1,-1,1],
                cmap='Reds', vmin=vmin, vmax=vmax
            )
        ax.set_title(f"{title}\nn={len(mv):,}")
        ax.set_xlabel('Δx')
        ax.set_ylabel('Δy')
        # no grid, remove top/right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.colorbar(im, cax=cbar_ax, label='Probability Density')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig, axs


def plot_displacement_histogram(
    dff,
    save_path=None,
    load_data=False,
    save_data=True
):
    """
    Histogram of displacement (< max_linear_velocity, no collisions if present).
    """
    base = os.path.splitext(save_path)[0] if save_path else None

    # filter
    max_linear_velocity = AGENT_PARAMS['max_linear_velocity_baseline']
    df2 = dff[dff['displacement'] < max_linear_velocity]  # NOTE: might have issues as this interacts with "turbo move"
    if 'collided' in df2.columns:
        df2 = df2[df2['collided']==False]

    # load or compute
    disp = None
    if load_data and base:
        try:
            disp = np.array(_load_data(base))
        except Exception:
            print(f"Warning: no data to load for '{base}', recalculating.")
            disp = None
    if disp is None:
        disp = df2['displacement'].values
        if save_data and base:
            _save_data(disp.tolist(), base)

    # plot
    fig, ax = _make_fig(1,1)
    ax.hist(disp, bins=30)
    ax.set_xlabel('Displacement')
    ax.set_ylabel('Count')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_conditional_displacement_histograms(
    dff,
    condition_col='has_nearby',
    save_path=None,
    load_data=False,
    save_data=True
):
    """
    Split displacement histograms by a boolean column.
    """
    base = os.path.splitext(save_path)[0] if save_path else None

    # filter
    max_linear_velocity = AGENT_PARAMS['max_linear_velocity_baseline']
    df2 = dff[dff['displacement'] < max_linear_velocity].copy()
    if 'collided' in df2.columns:
        df2 = df2[df2['collided']==False]

    # prepare data dict
    data = None
    if load_data and base:
        try:
            data = _load_data(base)
        except Exception:
            print(f"Warning: no data to load for '{base}', recalculating.")
            data = None

    if data is None:
        data = {}
        for label, mask in [('true', df2[df2[condition_col]]),
                            ('false',df2[~df2[condition_col]])]:
            data[label] = mask['displacement'].tolist()
        if save_data and base:
            _save_data(data, base)

    # convert back
    disp_true  = np.array(data['true'])
    disp_false = np.array(data['false'])

    # plot
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(2.5*2,2.5*1))
    for ax, disp, label in [(ax1,disp_true, f"{condition_col}=True"),
                             (ax2,disp_false,f"{condition_col}=False")]:
        ax.hist(disp, bins=30, alpha=0.7)
        ax.set_title(f"{label}\nn={len(disp):,}")
        ax.set_xlabel('Displacement')
        ax.set_ylabel('Count')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig, (ax1, ax2)


def plot_turn_angle_histogram(
    dff,
    save_path=None,
    load_data=False,
    save_data=True,
    ax=None
):
    """
    Histogram of 'actual_turn'.
    """
    base = os.path.splitext(save_path)[0] if save_path else None

    # load or compute
    ta = None
    if load_data and base:
        try:
            ta = np.array(_load_data(base))
        except Exception:
            print(f"Warning: no data to load for '{base}', recalculating.")
            ta = None
    if ta is None:
        ta = dff['actual_turn'].values
        if save_data and base:
            _save_data(ta.tolist(), base)

    # plot
    if ax is None:
        fig, ax = _make_fig(1, 1, height_multiplier=2/3, width_multiplier=2/3)
    else:
        fig = ax.figure
    ax.hist(ta, bins=30)
    ax.set_xlabel('Turn Angle (radians)')
    ax.set_ylabel('Count')
    # remove x ticks
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_conditional_turn_angle_histograms(
    dff,
    condition_col='food_observed',
    save_path=None,
    load_data=False,
    save_data=True
):
    """
    Split 'actual_turn' histograms by condition_col.
    """
    base = os.path.splitext(save_path)[0] if save_path else None

    # load or compute
    data = None
    if load_data and base:
        try:
            data = _load_data(base)
        except Exception:
            print(f"Warning: no data to load for '{base}', recalculating.")
            data = None

    if data is None:
        data = {}
        for lab, mask in [('true',  dff[dff[condition_col]]),
                          ('false', dff[~dff[condition_col]])]:
            data[lab] = mask['actual_turn'].tolist()
        if save_data and base:
            _save_data(data, base)

    ta_true  = np.array(data['true'])
    ta_false = np.array(data['false'])

    # plot
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(2.5*2,2.5*1))
    for ax, ta, label in [(ax1,ta_true,  f"{condition_col}=True"),
                          (ax2,ta_false, f"{condition_col}=False")]:
        ax.hist(ta, bins=30, alpha=0.7)
        ax.set_title(f"Turn Angle\n{label} (n={len(ta):,})")
        ax.set_xlabel('Turn Angle (radians)')
        ax.set_ylabel('Count')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig, (ax1, ax2)



def plot_food_metrics_before_eating(
    dff,
    steps_before: int = 5,
    save_path: str = None,
    load_data: bool = False,
    save_data: bool = True
):
    """
    Plot distance and angle to closest food before eating events.
    """
    base = os.path.splitext(save_path)[0] if save_path else None

    # Load or compute
    if load_data and base:
        all_distances = _load_data(f"{base}_distances")
        all_angles    = _load_data(f"{base}_angles")
    else:
        eating_events = dff[dff['eating_event'] == True].reset_index()
        total_steps   = steps_before + 1
        all_distances = [[] for _ in range(total_steps)]
        all_angles    = [[] for _ in range(total_steps)]

        for _, ev in eating_events.iterrows():
            idxs = range(ev['index'] - steps_before, ev['index'] + 1)
            valid = []
            for i in idxs:
                if 0 <= i < len(dff):
                    r = dff.iloc[i]
                    if (r['episode_index']==ev['episode_index']
                        and r['agent_id']==ev['agent_id']
                        and r['env_id']==ev['env_id']):
                        valid.append(i)
            if len(valid)==total_steps:
                block = dff.iloc[valid]
                for s in range(total_steps):
                    all_distances[s].append(block.iloc[s]['distance_to_closest_food'])
                    all_angles   [s].append(block.iloc[s]['angle_to_closest_food'])

        if save_data and base:
            _save_data(all_distances, f"{base}_distances")
            _save_data(all_angles,    f"{base}_angles")

    # Make figure
    fig, (ax1, ax2) = _make_fig(2, 1)
    time_steps = range(-steps_before, 1)

    # Distance
    means = [np.nanmean(x) for x in all_distances]
    sems  = [
        np.nanstd(x)/np.sqrt(np.count_nonzero(~np.isnan(x)))
        if np.count_nonzero(~np.isnan(x))>0 else np.nan
        for x in all_distances
    ]
    for s, data in enumerate(all_distances):
        pts = [v for v in data if not np.isnan(v)]
        if pts:
            ax1.scatter([time_steps[s]]*len(pts), pts, alpha=0.02, color='blue')
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.plot(time_steps, means, 'b-')
    ax1.fill_between(
        time_steps,
        [m-se if not np.isnan(m) else np.nan for m, se in zip(means, sems)],
        [m+se if not np.isnan(m) else np.nan for m, se in zip(means, sems)],
        alpha=0.3
    )
    ax1.set_xlabel('Time Steps (0 = Eating Event)')
    ax1.set_ylabel('Distance to Closest Food')
    ax1.set_title('Distance to Closest Food Before Eating')

    # Angle
    means = [np.nanmean(x) for x in all_angles]
    sems  = [
        np.nanstd(x)/np.sqrt(np.count_nonzero(~np.isnan(x)))
        if np.count_nonzero(~np.isnan(x))>0 else np.nan
        for x in all_angles
    ]
    for s, data in enumerate(all_angles):
        pts = [v for v in data if not np.isnan(v)]
        if pts:
            ax2.scatter([time_steps[s]]*len(pts), pts, alpha=0.02, color='red')
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.plot(time_steps, means, 'r-')
    ax2.fill_between(
        time_steps,
        [m-se if not np.isnan(m) else np.nan for m, se in zip(means, sems)],
        [m+se if not np.isnan(m) else np.nan for m, se in zip(means, sems)],
        alpha=0.3
    )
    ax2.set_xlabel('Time Steps (0 = Eating Event)')
    ax2.set_ylabel('Angle to Closest Food (degrees)')
    ax2.set_title('Angle to Closest Food Before Eating')
    ax2.set_ylim(-180, 180)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig, (ax1, ax2)


def plot_angle_to_closest_food_polar_histogram(
    dff,
    save_path=None,
    load_data=False,
    save_data=True
):
    """
    Single-panel polar histogram of angles to closest food.
    """
    base = os.path.splitext(save_path)[0] if save_path else None
    if load_data and base:
        try:
            arr = _load_data(base)
            angles = np.array(arr)
        except FileNotFoundError:
            print(f"Warning: no data to load for '{base}', recalculating.")
            load_data = False
    if not load_data:
        angles = dff['angle_to_closest_food'].dropna().values
        if save_data and base:
            _save_data(angles.tolist(), base)

    # Polar subplot
    fig, ax = plt.subplots(
        1, 1,
        figsize=(2.5, 2.5),
        subplot_kw={'projection': 'polar'}
    )
    width = 2 * np.pi / 90
    counts, bins = np.histogram(angles, bins=90, range=(-np.pi, np.pi))
    centers = (bins[:-1] + bins[1:]) / 2

    ax.bar(centers, counts, width=width, bottom=0,
           edgecolor='black', alpha=0.7)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetamin(-180)
    ax.set_thetamax(180)
    ax.set_title('Angles to Closest Food')

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_conditional_angle_to_closest_food_polar_histograms(
    dff,
    save_path=None,
    load_data=False,
    save_data=True
):
    """
    Two-panel polar histograms split by `food_observed`.
    """
    base = os.path.splitext(save_path)[0] if save_path else None
    if load_data and base:
        try:
            records = _load_data(base)
            df_data = pd.DataFrame(records)
        except FileNotFoundError:
            print(f"Warning: no data to load for '{base}', recalculating.")
            load_data = False
    if not load_data:
        angles_true = dff.loc[dff['food_observed'], 'angle_to_closest_food'].dropna().values
        angles_false = dff.loc[~dff['food_observed'], 'angle_to_closest_food'].dropna().values
        df_data = pd.DataFrame({
            'angle': np.concatenate([angles_true, angles_false]),
            'food_observed': [True]*len(angles_true) + [False]*len(angles_false)
        })
        if save_data and base:
            _save_data(df_data.to_dict(orient='list'), base)

    # Two polar subplots
    fig, axes = plt.subplots(
        1, 2,
        figsize=(2.5*2, 2.5*1),
        subplot_kw={'projection': 'polar'}
    )
    width = 2 * np.pi / 90
    for ax, cond in zip(axes.flatten(), [True, False]):
        arr = df_data.loc[df_data['food_observed']==cond, 'angle'].values
        counts, bins = np.histogram(arr, bins=90, range=(-np.pi, np.pi))
        centers = (bins[:-1] + bins[1:]) / 2
        ax.bar(centers, counts, width=width, bottom=0,
               edgecolor='black', alpha=0.7)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_thetamin(-180)
        ax.set_thetamax(180)
        ax.set_title(f'Observed={cond}')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, axes


def plot_combined_angle_to_closest_food_polar_histogram(
    dff,
    save_path=None,
    load_data=False,
    save_data=True,
    method='hist',  # 'hist' or 'kde'
):
    """
    Combined polar histogram of angles to closest food, normalized.
    """
    base = os.path.splitext(save_path)[0] if save_path else None
    if load_data and base:
        try:
            data = _load_data(base)
            counts_false = np.array(data['false'])
            counts_true = np.array(data['true'])
        except Exception:
            print(f"Warning: no data to load for '{base}', recalculating.")
            load_data = False
    if not load_data:
        if method == 'hist':
            angles_true = dff.loc[dff['food_observed'], 'angle_to_closest_food'].dropna().values
            angles_false = dff.loc[~dff['food_observed'], 'angle_to_closest_food'].dropna().values
            edges = np.linspace(-np.pi, np.pi, 91)
            counts_true, _ = np.histogram(angles_true, bins=edges)
            counts_false, _ = np.histogram(angles_false, bins=edges)
            counts_true = counts_true / counts_true.sum()
            counts_false = counts_false / counts_false.sum()
            centers = (edges[:-1] + edges[1:]) / 2
            width = 2 * np.pi / 90
            if save_data and base:
                _save_data({'true': counts_true.tolist(), 'false': counts_false.tolist()}, base)
        elif method == 'kde':
            theta_grid = np.linspace(-np.pi, np.pi, 1000)
            kde_true = gaussian_kde(angles_true)
            kde_false = gaussian_kde(angles_false)
            density_true = kde_true(theta_grid)
            density_false = kde_false(theta_grid)
            # normalize
            density_true = density_true / density_true.sum()
            density_false = density_false / density_false.sum()


    fig, ax = plt.subplots(
        1, 1,
        figsize=(2.5, 2.5),
        subplot_kw={'projection': 'polar'}
    )
    if method == 'kde':
        ax.plot(theta_grid, density_false, color='C0', alpha=0.5,
            label=fr"$d_f>5$ cm")
            # label=fr"$d_f>5$ cm (n={len(angles_false):,})")
            # label=fr"$d_f>5$ cm (n={len(angles_false):,})")
        ax.plot(theta_grid, density_true, color='C1', alpha=0.8,
            label=fr"$d_f\leq 5$ cm")
            # label=fr"$d_f\leq 5$ cm (n={len(angles_true):,})")
    elif method == 'hist':
        ax.bar(centers, counts_false, width=width, bottom=0,
            edgecolor='C0',
            alpha=0.4,
            label=fr"$d_f >$5 cm")
            # label=fr"$d_{{food}} >$5 cm (n={len(angles_false) if not load_data else ''})")
        ax.bar(centers, counts_true, width=width, bottom=0,
            edgecolor='C1',
            alpha=0.4,
            label=fr"$d_f \leq$5 cm")
            # label=fr"$d_{{food}} \leq$5 cm (n={len(angles_true) if not load_data else ''})")
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    # ax.set_title('Normalized Angles to Closest Food')
    ax.set_yticklabels([])
    # ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.08))
    ax.legend(loc='upper right', bbox_to_anchor=(0.35, 1.23))
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_conditional_angle_polar_histogram(
    dff,
    save_path=None,
    load_data=False,
    save_data=True,
    angle_col='angle_to_closest_food',
    condition_col='food_observed'
):
    """
    Combined polar histogram of angles to closest food, normalized.
    """
    base = os.path.splitext(save_path)[0] if save_path else None
    if load_data and base:
        try:
            data = _load_data(base)
            counts_false = np.array(data['false'])
            counts_true = np.array(data['true'])
        except Exception:
            print(f"Warning: no data to load for '{base}', recalculating.")
            load_data = False
    if not load_data:
        angles_true = dff.loc[dff[condition_col], angle_col].dropna().values
        angles_false = dff.loc[~dff[condition_col], angle_col].dropna().values
        edges = np.linspace(-np.pi, np.pi, 91)
        counts_true, _ = np.histogram(angles_true, bins=edges)
        counts_false, _ = np.histogram(angles_false, bins=edges)
        counts_true = counts_true / counts_true.sum()
        counts_false = counts_false / counts_false.sum()
        if save_data and base:
            _save_data({'true': counts_true.tolist(), 'false': counts_false.tolist()}, base)

    centers = (edges[:-1] + edges[1:]) / 2
    width = 2 * np.pi / 90
    fig, ax = plt.subplots(
        1, 1,
        figsize=(2.5, 2.5),
        subplot_kw={'projection': 'polar'}
    )
    ax.bar(centers, counts_false, width=width, bottom=0,
           edgecolor='C0',
           alpha=0.4,
           label=f"obs=False (n={len(angles_false) if not load_data else ''})")
    ax.bar(centers, counts_true, width=width, bottom=0,
           edgecolor='C1',
           alpha=0.4,
           label=f"obs=True  (n={len(angles_true) if not load_data else ''})")
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    # ax.set_title('Normalized Angles to Closest Food')
    ax.set_yticklabels([])
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.08))
    ax.legend(loc='upper right', bbox_to_anchor=(0.5, 1.18))
    # ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.2))
    # ax.legend(loc='upper right')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_distance_to_closest_food_histogram(
    dff,
    save_path=None,
    load_data=False,
    save_data=True
):
    """
    Histogram of distances to closest food.
    """
    base = os.path.splitext(save_path)[0] if save_path else None
    if load_data and base:
        try:
            distances = np.array(_load_data(base))
        except Exception:
            print(f"Warning: no data to load for '{base}', recalculating.")
            load_data = False
    if not load_data:
        distances = dff['distance_to_closest_food'].dropna().values
        if save_data and base:
            _save_data(distances.tolist(), base)

    fig, ax = _make_fig(1, 1)
    ax.hist(distances, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Distance to Closest Food (cm)')
    ax.set_ylabel('Count')
    # ax.set_title('Distances to Closest Food')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    # remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_conditional_distance_to_closest_food_histograms(
    dff,
    save_path=None,
    load_data=False,
    save_data=True
):
    """
    Two-panel histograms of distances split by `food_observed`.
    """
    base = os.path.splitext(save_path)[0] if save_path else None
    if load_data and base:
        try:
            data = _load_data(base)
            dist_true = np.array(data['true'])
            dist_false = np.array(data['false'])
        except Exception:
            print(f"Warning: no data to load for '{base}', recalculating.")
            load_data = False
    if not load_data:
        dist_true = dff.loc[dff['food_observed'], 'distance_to_closest_food'].dropna().values
        dist_false = dff.loc[~dff['food_observed'], 'distance_to_closest_food'].dropna().values
        if save_data and base:
            _save_data({'true': dist_true.tolist(), 'false': dist_false.tolist()}, base)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2.5*2, 2.5*1))
    ax1.hist(dist_true, bins=50, alpha=0.7, edgecolor='black', color='forestgreen')
    ax1.set_title(f'True (n={len(dist_true)})')
    ax1.set_xlabel('Distance')
    ax1.set_ylabel('Count')
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)


    ax2.hist(dist_false, bins=50, alpha=0.7, edgecolor='black', color='crimson')
    ax2.set_title(f'False (n={len(dist_false)})')
    ax2.set_xlabel('Distance')
    ax2.set_ylabel('Count')
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)


    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, (ax1, ax2)


def plot_time_between_eating_histogram(
    dff,
    save_path=None,
    load_data=False,
    save_data=True
):
    """
    Histogram of time intervals between eating events, full vs ≤95th percentile.
    """
    base = os.path.splitext(save_path)[0] if save_path else None
    if load_data and base:
        try:
            time_intervals = _load_data(base)
        except Exception:
            print(f"Warning: no data to load for '{base}', recalculating.")
            load_data = False
    if not load_data:
        time_intervals = []
        for (env, agent), grp in dff.groupby(['env_id', 'agent_id']):
            grp = grp.sort_values(['episode_index', 'time_step'])
            ev = grp[grp['eating_event']]
            if len(ev)>1:
                ev['next_episode']=ev['episode_index'].shift(-1)
                ev['next_time']=ev['time_step'].shift(-1)
                same = ev['episode_index']==ev['next_episode']
                diffs = (ev.loc[same,'next_time'] - ev.loc[same,'time_step']).tolist()
                time_intervals.extend(diffs)
        if save_data and base:
            _save_data(time_intervals, base)

    if not time_intervals:
        print("No valid time intervals between eating events found.")
        return None, None

    cutoff = np.percentile(time_intervals, 90)
    filtered = [t for t in time_intervals if t<=cutoff]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2.5*2, 2.5*1))
    ax1.hist(time_intervals, bins=100, edgecolor='black')
    ax1.axvline(cutoff, color='r', linestyle='--', label=f'95th pct: {cutoff:.1f}')
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Full (n={len(time_intervals)})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    mn, mx = int(np.floor(min(filtered))), int(np.ceil(max(filtered)))
    bins = np.arange(mn, mx+2)-0.5
    ax2.hist(filtered, bins=bins, edgecolor='black')
    ax2.set_xlabel('Timesteps')
    ax2.set_ylabel('Count')
    ax2.set_title(f'≤95th pct (n={len(filtered)})')
    ax2.set_xticks(np.arange(mn, mx+1, max(1,(mx-mn)//5)))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, (ax1, ax2)


def plot_movement_around_eating(
    dff,
    steps_around=4,
    save_path=None,
    load_data=False,
    save_data=True
):
    """
    Plot movement patterns before and after eating events.
    """
    base = os.path.splitext(save_path)[0] if save_path else None
    # No caching for movement data
    fig, (ax1, ax2, ax3) = _make_fig(3, 1)
    eating_events = dff[dff['eating_event'] == True]
    total_steps = 2 * steps_around + 1
    all_move_forward = [[] for _ in range(total_steps)]
    all_move_backward = [[] for _ in range(total_steps)]
    all_turn_angle = [[] for _ in range(total_steps)]

    for idx, event in eating_events.iterrows():
        surrounding_indices = range(idx - steps_around, idx + steps_around + 1)
        valid_indices = []
        for s_idx in surrounding_indices:
            if 0 <= s_idx < len(dff):
                row = dff.iloc[s_idx]
                if (row['episode_index'] == event['episode_index'] and
                    row['agent_id'] == event['agent_id'] and
                    row['env_id'] == event['env_id']):
                    valid_indices.append(s_idx)
        if len(valid_indices) == total_steps:
            surr_rows = dff.iloc[valid_indices]
            for step in range(total_steps):
                mv = surr_rows.iloc[step]['move_forward']
                if mv >= 0:
                    all_move_forward[step].append(mv)
                    all_move_backward[step].append(np.nan)
                else:
                    all_move_forward[step].append(np.nan)
                    all_move_backward[step].append(abs(mv))
                all_turn_angle[step].append(abs(surr_rows.iloc[step]['turn_angle']))

    time_steps = range(-steps_around, steps_around + 1)

    # Plot forward movement
    move_forward_means = [np.nanmean(x) for x in all_move_forward]
    move_forward_sems = [
        np.nanstd(x) / np.sqrt(np.count_nonzero(~np.isnan(x)))
        if np.count_nonzero(~np.isnan(x)) > 0 else np.nan
        for x in all_move_forward
    ]
    for step in range(total_steps):
        data = [v for v in all_move_forward[step] if not np.isnan(v)]
        if data:
            ax1.scatter(
                [time_steps[step]] * len(data),
                data,
                alpha=0.02,
                color='blue'
            )
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.plot(time_steps, move_forward_means, 'b-')
    ax1.fill_between(
        time_steps,
        [m - s if not np.isnan(m) else np.nan for m, s in zip(move_forward_means, move_forward_sems)],
        [m + s if not np.isnan(m) else np.nan for m, s in zip(move_forward_means, move_forward_sems)],
        alpha=0.3
    )
    ax1.set_xlabel('Time Steps (0 = Eating Event)')
    ax1.set_ylabel('Move Forward Action')
    ax1.set_title('Forward Movement Pattern Around Eating')

    # Plot backward movement
    move_backward_means = [np.nanmean(x) for x in all_move_backward]
    move_backward_sems = [
        np.nanstd(x) / np.sqrt(np.count_nonzero(~np.isnan(x)))
        if np.count_nonzero(~np.isnan(x)) > 0 else np.nan
        for x in all_move_backward
    ]
    for step in range(total_steps):
        data = [v for v in all_move_backward[step] if not np.isnan(v)]
        if data:
            ax2.scatter(
                [time_steps[step]] * len(data),
                data,
                alpha=0.02,
                color='green'
            )
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.plot(time_steps, move_backward_means, 'g-')
    ax2.fill_between(
        time_steps,
        [m - s if not np.isnan(m) else np.nan for m, s in zip(move_backward_means, move_backward_sems)],
        [m + s if not np.isnan(m) else np.nan for m, s in zip(move_backward_means, move_backward_sems)],
        alpha=0.3,
        color='green'
    )
    ax2.set_xlabel('Time Steps (0 = Eating Event)')
    ax2.set_ylabel('Move Backward Action')
    ax2.set_title('Backward Movement Pattern Around Eating')

    # Plot turn angle
    turn_angle_means = [np.nanmean(x) for x in all_turn_angle]
    turn_angle_sems = [
        np.nanstd(x) / np.sqrt(len(x))
        if len(x) > 0 else np.nan
        for x in all_turn_angle
    ]
    for step in range(total_steps):
        data = all_turn_angle[step]
        if data:
            ax3.scatter(
                [time_steps[step]] * len(data),
                data,
                alpha=0.02,
                color='red'
            )
    ax3.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.plot(time_steps, turn_angle_means, 'r-')
    ax3.fill_between(
        time_steps,
        [m - s if not np.isnan(m) else np.nan for m, s in zip(turn_angle_means, turn_angle_sems)],
        [m + s if not np.isnan(m) else np.nan for m, s in zip(turn_angle_means, turn_angle_sems)],
        alpha=0.3
    )
    ax3.set_xlabel('Time Steps (0 = Eating Event)')
    ax3.set_ylabel('Absolute Turn Angle Action')
    ax3.set_title('Turn Angle Pattern Around Eating')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig, (ax1, ax2, ax3)


def plot_movement_between_eating(
    dff,
    max_steps_between=10,
    save_path=None,
    load_data=False,
    save_data=True
):
    """
    Plot movement patterns between closely spaced eating events.
    """
    fig, (ax1, ax2, ax3) = _make_fig(3, 1)
    eating_events = dff[dff['eating_event'] == True].reset_index()
    all_move_forward = [[] for _ in range(max_steps_between)]
    all_move_backward = [[] for _ in range(max_steps_between)]
    all_turn_angle = [[] for _ in range(max_steps_between)]

    for i in range(len(eating_events) - 1):
        current = eating_events.iloc[i]
        nxt = eating_events.iloc[i + 1]
        if (current['episode_index'] == nxt['episode_index'] and
            current['agent_id'] == nxt['agent_id'] and
            current['env_id'] == nxt['env_id']):
            steps_between = nxt['index'] - current['index']
            if 0 < steps_between <= max_steps_between:
                between_rows = dff.iloc[current['index'] + 1 : nxt['index']]
                for idx, row in between_rows.iterrows():
                    sb = nxt['index'] - idx - 1
                    mv = row['move_forward']
                    if mv >= 0:
                        all_move_forward[sb].append(mv)
                        all_move_backward[sb].append(np.nan)
                    else:
                        all_move_forward[sb].append(np.nan)
                        all_move_backward[sb].append(abs(mv))
                    all_turn_angle[sb].append(abs(row['turn_angle']))

    time_steps = range(-1, -max_steps_between - 1, -1)

    # forward movement
    move_forward_means = [np.nanmean(x) if x else np.nan for x in all_move_forward]
    move_forward_sems = [
        np.nanstd(x) / np.sqrt(np.count_nonzero(~np.isnan(x))) if np.count_nonzero(~np.isnan(x)) > 0 else np.nan
        for x in all_move_forward
    ]
    for step in range(max_steps_between):
        data = [v for v in all_move_forward[step] if not np.isnan(v)]
        if data:
            ax1.scatter(
                [time_steps[step]] * len(data),
                data,
                alpha=0.1,
                color='blue'
            )
    ax1.plot(time_steps, move_forward_means, 'b-')
    ax1.fill_between(
        time_steps,
        [m - s if not np.isnan(m) else np.nan for m, s in zip(move_forward_means, move_forward_sems)],
        [m + s if not np.isnan(m) else np.nan for m, s in zip(move_forward_means, move_forward_sems)],
        alpha=0.3
    )
    ax1.set_xlabel('Time Steps Before Next Eating')
    ax1.set_ylabel('Move Forward Action')
    ax1.set_title('Forward Movement Pattern Between Eating Events')

    # backward movement
    move_backward_means = [np.nanmean(x) if x else np.nan for x in all_move_backward]
    move_backward_sems = [
        np.nanstd(x) / np.sqrt(np.count_nonzero(~np.isnan(x))) if np.count_nonzero(~np.isnan(x)) > 0 else np.nan
        for x in all_move_backward
    ]
    for step in range(max_steps_between):
        data = [v for v in all_move_backward[step] if not np.isnan(v)]
        if data:
            ax2.scatter(
                [time_steps[step]] * len(data),
                data,
                alpha=0.1,
                color='green'
            )
    ax2.plot(time_steps, move_backward_means, 'g-')
    ax2.fill_between(
        time_steps,
        [m - s if not np.isnan(m) else np.nan for m, s in zip(move_backward_means, move_backward_sems)],
        [m + s if not np.isnan(m) else np.nan for m, s in zip(move_backward_means, move_backward_sems)],
        alpha=0.3
    )
    ax2.set_xlabel('Time Steps Before Next Eating')
    ax2.set_ylabel('Move Backward Action')
    ax2.set_title('Backward Movement Pattern Between Eating Events')

    # turn angle
    turn_angle_means = [np.nanmean(x) if x else np.nan for x in all_turn_angle]
    turn_angle_sems = [
        np.nanstd(x) / np.sqrt(len(x)) if x else np.nan
        for x in all_turn_angle
    ]
    for step in range(max_steps_between):
        data = all_turn_angle[step]
        if data:
            ax3.scatter(
                [time_steps[step]] * len(data),
                data,
                alpha=0.1,
                color='red'
            )
    ax3.plot(time_steps, turn_angle_means, 'r-')
    ax3.fill_between(
        time_steps,
        [m - s if not np.isnan(m) else np.nan for m, s in zip(turn_angle_means, turn_angle_sems)],
        [m + s if not np.isnan(m) else np.nan for m, s in zip(turn_angle_means, turn_angle_sems)],
        alpha=0.3
    )
    ax3.set_xlabel('Time Steps Before Next Eating')
    ax3.set_ylabel('Absolute Turn Angle Action')
    ax3.set_title('Turn Angle Pattern Between Eating Events')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig, (ax1, ax2, ax3)

def plot_food_eaten_vs_agent_size(
    dff,
    save_path: str = None,
    load_data: bool = False,
    save_data: bool = True
):
    """
    Scatter + trend: total food eaten vs. agent size.
    """
    base = os.path.splitext(save_path)[0] if save_path else None

    if load_data and base:
        grouped = _load_data(f"{base}_grouped")
    else:
        grouped = (
            dff.groupby(['env_id','episode_index','agent_id'])
               .agg(arena_size=('patch_kwargs.arena_size','first'),
                    food_eaten=('eating_event','sum'),
                    agent_size=('agent_size','first'))
               .reset_index()
        )
        if save_data and base:
            _save_data(grouped, f"{base}_grouped")

    fig, ax = _make_fig(1, 1)
    sns.regplot(
        data=grouped,
        x='agent_size', y='food_eaten',
        scatter_kws={'alpha':0.6},
        line_kws   ={'color':'red'},
        ax=ax
    )
    try:
        r, p = stats.pearsonr(grouped['agent_size'], grouped['food_eaten'])
        # ax.text(
        #     0.58, 0.95,
        #     f"r={r:.3f}, p={p:.3e}",
        #     ha='left', va='top',
        #     transform=ax.transAxes,
        #     fontsize=9,
        #     bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')
        # )

        txt = f"r = {r:.3f}\np = {p:.1e}"
        at = AnchoredText(
            txt, loc="upper right", prop=dict(size=9),
            frameon=True, borderpad=0.4, pad=0.3
        )
        at.patch.set_alpha(0.7)
        at.patch.set_edgecolor("black")
        ax.add_artist(at)
    except:
        pass
    ax.set_xlabel("Agent Size")
    ax.set_ylabel("Food Eaten")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig, ax


def plot_food_eaten_normalized_vs_agent_size(
    dff,
    save_path: str = None,
    load_data: bool = False,
    save_data: bool = True,
    ax=None
):
    """
    Scatter + trend: normalized food eaten vs. agent size.
    """
    base = os.path.splitext(save_path)[0] if save_path else None

    if load_data and base:
        grouped = _load_data(f"{base}_grouped_norm")
    else:
        # arena area
        av = (
            dff[dff['agent_id']==0]
            .groupby(['env_id','episode_index'])['patch_kwargs.arena_size']
            .first()
            .apply(lambda x: x[0]*x[1] if len(x)==2 else np.nan)
            .reset_index(name='arena_area')
        )
        grp = (
            dff.groupby(['env_id','episode_index','agent_id'])
               .agg(food_eaten=('eating_event','sum'),
                    agent_size=('agent_size','first'))
               .reset_index()
               .merge(av, on=['env_id','episode_index'])
               .dropna(subset=['arena_area'])
        )
        grp['normalized_food_eaten'] = grp['food_eaten']/grp['arena_area']
        grouped = grp
        if save_data and base:
            _save_data(grouped, f"{base}_grouped_norm")

    if ax is None:
        fig, ax = _make_fig(1, 1, width_multiplier=1)
    else:
        fig = ax.figure
    sns.regplot(
        data=grouped,
        x='agent_size', y='normalized_food_eaten',
        scatter_kws={'alpha':0.6},
        line_kws   ={'color':'red'},
        ax=ax
    )
    try:
        r, p = stats.pearsonr(grouped['agent_size'], grouped['normalized_food_eaten'])
        # add text box
        txt = f"r = {r:.3f}\np = {p:.1e}"
        at = AnchoredText(
            txt, loc="upper right", prop=dict(size=9),
            frameon=True, borderpad=0.4, pad=0.3
        )
        at.patch.set_alpha(0.7)
        at.patch.set_edgecolor("black")
        ax.add_artist(at)
        # ax.set_title(f"Normalized Food Eaten vs Agent Size\nr={r:.3f}, p={p:.3e}")
    except:
        pass
    ax.set_xlabel("Agent Size")
    ax.set_ylabel(f"Food Eaten Per cm$^2$")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig, ax


def plot_gini_agent_size_vs_food_inequality(
    dff: pd.DataFrame,
    save_path: str = None,
    load_data: bool = False,
    save_data: bool = True,
    ax=None,
    arena_area_col: str = "arena_area",
    normalize_food_by_area: bool = True,
):
    """
    Scatter + trend: Gini(agent size) vs Gini(food eaten), computed per (env_id, episode_index).

    x-axis: Gini(agent size)
    y-axis: Gini(food eaten)

    Returns:
      fig, ax, episode_df
    """
    base = os.path.splitext(save_path)[0] if save_path else None

    if load_data and base:
        episode_df = _load_data(f"{base}_gini_episode")
    else:
        df = dff.copy()

        # Arena area per episode (only if normalization requested)
        if normalize_food_by_area:
            av = (
                df[df["agent_id"] == 0]
                .groupby(["env_id", "episode_index"])["patch_kwargs.arena_size"]
                .first()
                .apply(lambda x: x[0] * x[1] if hasattr(x, "__len__") and len(x) == 2 else np.nan)
                .reset_index(name=arena_area_col)
            )

        # Per-agent within episode aggregation
        grp = (
            df.groupby(["env_id", "episode_index", "agent_id"])
              .agg(
                  food_eaten=("eating_event", "sum"),
                  agent_size=("agent_size", "first"),
              )
              .reset_index()
        )

        if normalize_food_by_area:
            grp = grp.merge(av, on=["env_id", "episode_index"], how="left").dropna(subset=[arena_area_col])
            grp["food_eaten_norm"] = grp["food_eaten"] / grp[arena_area_col]
            food_col = "food_eaten_norm"
        else:
            food_col = "food_eaten"

        # Per-episode Gini across agents
        def _episode_ginis(g: pd.DataFrame) -> pd.Series:
            return pd.Series(
                {
                    "gini_size": calculate_gini(g["agent_size"].to_numpy()),
                    "gini_food": calculate_gini(g[food_col].to_numpy()),
                    "n_agents": g["agent_id"].nunique(),
                    "total_food": np.nansum(g[food_col].to_numpy()),
                }
            )

        episode_df = (
            grp.groupby(["env_id", "episode_index"], sort=False)
               .apply(_episode_ginis)
               .reset_index()
        )

        # Keep valid episodes
        episode_df = episode_df[
            (episode_df["n_agents"] >= 2)
            & np.isfinite(episode_df["gini_food"])
            & np.isfinite(episode_df["gini_size"])
        ].reset_index(drop=True)

        if save_data and base:
            _save_data(episode_df, f"{base}_gini_episode")

    # Plot
    if ax is None:
        fig, ax = _make_fig(1, 1, width_multiplier=1)
    else:
        fig = ax.figure

    sns.regplot(
        data=episode_df,
        x="gini_size",
        y="gini_food",
        scatter_kws={"alpha": 0.6},
        line_kws={"color": "red"},
        ax=ax,
    )

    # Correlation annotation
    try:
        r, p = stats.pearsonr(
            episode_df["gini_size"].to_numpy(),
            episode_df["gini_food"].to_numpy(),
        )
        txt = f"r = {r:.3f}\np = {p:.1e}\nN = {len(episode_df)}"
        at = AnchoredText(
            txt, loc="upper right", prop=dict(size=9),
            frameon=True, borderpad=0.4, pad=0.3
        )
        at.patch.set_alpha(0.7)
        at.patch.set_edgecolor("black")
        ax.add_artist(at)
    except Exception:
        pass

    ax.set_xlabel("Inequality in Agent Size")
    if normalize_food_by_area:
        ax.set_ylabel("Inequality in Food Eaten per cm$^2$")
    else:
        ax.set_ylabel("Inequality in Food Eaten")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig, ax, episode_df


def plot_gini_agent_size_vs_eod_inequality(
    dff: pd.DataFrame,
    save_path: str = None,
    load_data: bool = False,
    save_data: bool = True,
    ax=None,
    emit_col: str = "emit_eod",
):
    """
    Scatter + trend: Gini(agent size) vs Gini(EOD rate), computed per (env_id, episode_index).

    x-axis: Gini(agent size)
    y-axis: Gini(EOD rate), where per-agent EOD rate = mean(emit_eod) within episode.

    Returns:
      fig, ax, episode_df
    """
    base = os.path.splitext(save_path)[0] if save_path else None

    if load_data and base:
        episode_df = _load_data(f"{base}_gini_episode")
    else:
        df = dff.copy()

        # Per-agent within-episode aggregation
        # EOD rate = average over emit_eod (assumes emit_eod is 0/1 or numeric rate-like)
        grp = (
            df.groupby(["env_id", "episode_index", "agent_id"])
              .agg(
                  agent_size=("agent_size", "first"),
                  eod_rate=(emit_col, "mean"),
                  n_steps=(emit_col, "count"),
              )
              .reset_index()
        )

        # Per-episode Gini across agents
        def _episode_ginis(g: pd.DataFrame) -> pd.Series:
            eod = g["eod_rate"].to_numpy()
            size = g["agent_size"].to_numpy()
            return pd.Series(
                {
                    "gini_size": calculate_gini(size),
                    "gini_eod": calculate_gini(eod),
                    "n_agents": g["agent_id"].nunique(),
                    "mean_eod": np.nanmean(eod),
                    "sum_eod_rate": np.nansum(eod),  # not a "count", but sometimes useful
                    "min_steps_per_agent": int(np.nanmin(g["n_steps"].to_numpy())) if len(g) else np.nan,
                }
            )

        episode_df = (
            grp.groupby(["env_id", "episode_index"], sort=False)
               .apply(_episode_ginis)
               .reset_index()
        )

        # Keep valid episodes
        episode_df = episode_df[
            (episode_df["n_agents"] >= 2)
            & np.isfinite(episode_df["gini_eod"])
            & np.isfinite(episode_df["gini_size"])
        ].reset_index(drop=True)

        if save_data and base:
            _save_data(episode_df, f"{base}_gini_episode")

    # Plot
    if ax is None:
        fig, ax = _make_fig(1, 1, width_multiplier=1)
    else:
        fig = ax.figure

    sns.regplot(
        data=episode_df,
        x="gini_size",
        y="gini_eod",
        scatter_kws={"alpha": 0.6},
        line_kws={"color": "red"},
        ax=ax,
    )

    # Correlation annotation
    try:
        r, p = stats.pearsonr(
            episode_df["gini_size"].to_numpy(),
            episode_df["gini_eod"].to_numpy(),
        )
        txt = f"r = {r:.3f}\np = {p:.1e}\nN = {len(episode_df)}"
        at = AnchoredText(
            txt, loc="upper right", prop=dict(size=9),
            frameon=True, borderpad=0.4, pad=0.3
        )
        at.patch.set_alpha(0.7)
        at.patch.set_edgecolor("black")
        ax.add_artist(at)
    except Exception:
        pass

    ax.set_xlabel("Inequality in Agent Size")
    ax.set_ylabel("Inequality in EOD Rate")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig, ax, episode_df


def _compute_episode_ginis_with_arena_area(
    dff: pd.DataFrame,
    arena_size_col: str = "arena_size",
    normalize_food_by_area: bool = True,
) -> pd.DataFrame:
    """
    Helper: returns one row per (env_id, episode_index) with:
      arena_area, gini_size, gini_food, n_agents, total_food
    """
    df = dff.copy()

    # Prefer arena_size_col (requested), but if missing, fall back to patch_kwargs.arena_size
    a_col = arena_size_col if arena_size_col in df.columns else "patch_kwargs.arena_size"
    if a_col not in df.columns:
        raise KeyError(f"Could not find '{arena_size_col}' or 'patch_kwargs.arena_size' in dff.columns.")

    av = (
        df[df["agent_id"] == 0]
        .groupby(["env_id", "episode_index"])[a_col]
        .first()
        .apply(lambda x: x[0] * x[1] if hasattr(x, "__len__") and len(x) == 2 else np.nan)
        .reset_index(name="arena_area")
        .dropna(subset=["arena_area"])
    )

    grp = (
        df.groupby(["env_id", "episode_index", "agent_id"])
          .agg(
              food_eaten=("eating_event", "sum"),
              agent_size=("agent_size", "first"),
          )
          .reset_index()
          .merge(av, on=["env_id", "episode_index"], how="inner")
    )

    if normalize_food_by_area:
        grp["food_used_for_gini"] = grp["food_eaten"] / grp["arena_area"]
    else:
        grp["food_used_for_gini"] = grp["food_eaten"]

    def _episode_stats(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "arena_area": g["arena_area"].iloc[0],
                "gini_size": calculate_gini(g["agent_size"].to_numpy()),
                "gini_food": calculate_gini(g["food_used_for_gini"].to_numpy()),
                "n_agents": g["agent_id"].nunique(),
                "total_food": float(np.nansum(g["food_used_for_gini"].to_numpy())),
            }
        )

    episode_df = (
        grp.groupby(["env_id", "episode_index"], sort=False)
           .apply(_episode_stats)
           .reset_index()
    )

    episode_df = episode_df[
        (episode_df["n_agents"] >= 2)
        & np.isfinite(episode_df["gini_size"])
        & np.isfinite(episode_df["gini_food"])
        & np.isfinite(episode_df["arena_area"])
    ].reset_index(drop=True)

    return episode_df


def _plot_gini_scatter_with_overall_fit(
    episode_df: pd.DataFrame,
    ax,
    title: str = None,
    normalize_food_by_area: bool = True,
):
    """
    Helper: scatter + ONE overall reg line + Pearson box. No legend title.
    """
    sns.scatterplot(
        data=episode_df,
        x="gini_size",
        y="gini_food",
        alpha=0.65,
        ax=ax,
    )

    if len(episode_df) >= 3:
        sns.regplot(
            data=episode_df,
            x="gini_size",
            y="gini_food",
            scatter=False,
            ax=ax,
        )

    try:
        r, p = stats.pearsonr(episode_df["gini_size"].to_numpy(), episode_df["gini_food"].to_numpy())
        txt = f"r = {r:.3f}\np = {p:.1e}\nN = {len(episode_df)}"
        at = AnchoredText(
            txt, loc="upper right", prop=dict(size=9),
            frameon=True, borderpad=0.4, pad=0.3
        )
        at.patch.set_alpha(0.7)
        at.patch.set_edgecolor("black")
        ax.add_artist(at)
    except Exception:
        pass

    ax.set_xlabel("Inequality in Agent Size")
    ax.set_ylabel("Inequality in Food Eaten per cm$^2$" if normalize_food_by_area else "Inequality in Food Eaten")

    if title:
        ax.set_title(title)


def plot_gini_size_vs_food_bottom_and_top_quartiles(
    dff: pd.DataFrame,
    save_path_bottom: str = None,
    save_path_top: str = None,
    load_data: bool = False,
    save_data: bool = True,
    axes=None,  # can pass (ax_bottom, ax_top)
    arena_size_col: str = "arena_size",
    normalize_food_by_area: bool = True,
    q: float = 0.25,
):
    """
    Makes TWO plots:
      1) Bottom q arenas (smallest by area)
      2) Top q arenas (largest by area)

    Each plot:
      - Scatter: x = Gini(agent_size), y = Gini(food_eaten)
      - ONE overall regression line
      - Pearson r/p/N box: "r = ...", "p = ...", "N = ..."

    Returns:
      (fig_bottom, ax_bottom, df_bottom), (fig_top, ax_top, df_top)
    """
    # --- caching (optional) ---
    base_b = os.path.splitext(save_path_bottom)[0] if save_path_bottom else None
    base_t = os.path.splitext(save_path_top)[0] if save_path_top else None
    cache_b = f"{base_b}_gini_episode_bottom_q{int(100*q)}" if base_b else None
    cache_t = f"{base_t}_gini_episode_top_q{int(100*q)}" if base_t else None

    if load_data and cache_b and cache_t:
        df_bottom = _load_data(cache_b)
        df_top = _load_data(cache_t)
    else:
        episode_df = _compute_episode_ginis_with_arena_area(
            dff=dff,
            arena_size_col=arena_size_col,
            normalize_food_by_area=normalize_food_by_area,
        )
        lo = episode_df["arena_area"].quantile(q)
        hi = episode_df["arena_area"].quantile(1.0 - q)

        df_bottom = episode_df[episode_df["arena_area"] <= lo].copy().reset_index(drop=True)
        df_top = episode_df[episode_df["arena_area"] >= hi].copy().reset_index(drop=True)

        if save_data:
            if cache_b:
                _save_data(df_bottom, cache_b)
            if cache_t:
                _save_data(df_top, cache_t)

    # --- axes / figures ---
    if axes is None:
        fig_b, ax_b = _make_fig(1, 1, width_multiplier=1)
        fig_t, ax_t = _make_fig(1, 1, width_multiplier=1)
    else:
        ax_b, ax_t = axes
        fig_b, fig_t = ax_b.figure, ax_t.figure

    # --- plot bottom ---
    _plot_gini_scatter_with_overall_fit(
        df_bottom,
        ax=ax_b,
        title=f"Bottom {int(100*q)}% arenas (by area)",
        normalize_food_by_area=normalize_food_by_area,
    )
    if save_path_bottom:
        fig_b.savefig(save_path_bottom, dpi=300, bbox_inches="tight")
        plt.close(fig_b)
    else:
        plt.close(fig_b)

    # --- plot top ---
    _plot_gini_scatter_with_overall_fit(
        df_top,
        ax=ax_t,
        title=f"Top {int(100*q)}% arenas (by area)",
        normalize_food_by_area=normalize_food_by_area,
    )
    if save_path_top:
        fig_t.savefig(save_path_top, dpi=300, bbox_inches="tight")
        plt.close(fig_t)
    else:
        plt.close(fig_t)

    return (fig_b, ax_b, df_bottom), (fig_t, ax_t, df_top)


def plot_prob_bite_action_prob_vs_agent_size(
    dff,
    save_path: str = None,
    load_data: bool = False,
    save_data: bool = True
):
    """
    P(bite_action | has_nearby) vs. agent_size.
    """
    # prepare base
    base = os.path.splitext(save_path)[0] if save_path else None

    # ensure column exists
    d = dff.copy()
    d['bite_and_nearby'] = d['bite_action'] & d['has_nearby']

    # load or compute
    if load_data and base:
        grouped = _load_data(f"{base}_grouped_bite_act")
    else:
        grouped = (
            d.groupby(['env_id', 'episode_index', 'agent_id'])
             .agg(
                prob_bite_given_nearby=('bite_and_nearby', 'sum'),
                has_nearby              =('has_nearby', 'sum'),
                agent_size              =('agent_size', 'first'),
             )
             .reset_index()
        )
        grouped['P(bite action | has_nearby)'] = (
            grouped['prob_bite_given_nearby'] / grouped['has_nearby']
        )
        if save_data and base:
            _save_data(grouped, f"{base}_grouped_bite_act")

    # make figure
    fig, ax = _make_fig(1, 1)
    sns.regplot(
        data=grouped,
        x='agent_size',
        y='P(bite action | has_nearby)',
        scatter_kws={'alpha':0.6},
        line_kws={'color':'red'},
        ax=ax
    )
    try:
        clean = grouped[['agent_size', 'P(Bite | Agent Nearby)']].dropna()
        r, p = stats.pearsonr(
            clean['agent_size'],
            clean['P(bite action | has_nearby)']
        )
        txt = f"r = {r:.3f}\np = {p:.1e}"
        at = AnchoredText(
            txt, loc="upper right", prop=dict(size=9),
            frameon=True, borderpad=0.4, pad=0.3
        )
        at.patch.set_alpha(0.7)
        at.patch.set_edgecolor("black")
        ax.add_artist(at)
        # ax.set_title(f"P(bite action | has_nearby) vs Agent Size\nr={r:.3f}, p={p:.3e}")
    except:
        print("Warning: Pearson correlation failed, skipping title update for plot_prob_bite_action_prob_vs_agent_size.")
    ax.set_xlabel("Agent Size")
    ax.set_ylabel("P(bite action | has_nearby)")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig, ax


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from matplotlib.offsetbox import AnchoredText


# -----------------------------
# Helpers
# -----------------------------

def _safe_entropy(p: np.ndarray) -> float:
    """Shannon entropy of a probability vector."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if len(p) == 0:
        return np.nan
    return -np.sum(p * np.log(p))


def _normalized_entropy_from_counts(counts: np.ndarray) -> float:
    """
    Normalize entropy to [0, 1].
    0 -> all mass on one agent
    1 -> perfectly even across all agents represented in `counts`
    """
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return np.nan
    k = len(counts)
    if k <= 1:
        return np.nan
    p = counts / total
    h = _safe_entropy(p)
    return h / np.log(k)


def _dominance_ratio(values: np.ndarray) -> float:
    """
    max share = max(values) / sum(values)
    Interpretable:
      - for 4 agents, perfectly equal -> 0.25
      - complete monopoly -> 1.0
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    total = values.sum()
    if len(values) == 0 or total <= 0:
        return np.nan
    return np.max(values) / total


def _rank_vector_desc(values: np.ndarray) -> np.ndarray:
    """
    Return descending ranks: largest value gets rank 1.
    Ties get average rank.
    """
    values = np.asarray(values, dtype=float)
    return stats.rankdata(-values, method="average")


def _rank_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Similarity of two rank vectors in [0, 1], based on Pearson correlation
    of descending-rank vectors.

    1.0 -> identical order
    0.0 -> perfectly reversed order
    0.5 -> no linear relation
    """
    r1 = _rank_vector_desc(v1)
    r2 = _rank_vector_desc(v2)

    s1 = np.std(r1)
    s2 = np.std(r2)

    if s1 == 0 and s2 == 0:
        return 1.0
    if s1 == 0 or s2 == 0:
        return np.nan

    rho = np.corrcoef(r1, r2)[0, 1]
    if not np.isfinite(rho):
        return np.nan

    return (rho + 1.0) / 2.0


def _build_episode_matrix(
    g: pd.DataFrame,
    time_col: str,
    agent_col: str,
    value_col: str,
):
    """
    Build a T x A matrix for a single episode.
    Rows = timesteps
    Cols = agents
    Entries = summed value at that timestep for that agent
    """
    wide = (
        g.groupby([time_col, agent_col])[value_col]
         .sum()
         .unstack(agent_col, fill_value=0.0)
         .sort_index(axis=0)
         .sort_index(axis=1)
    )

    if wide.shape[0] == 0 or wide.shape[1] == 0:
        return None, None, None

    M = wide.to_numpy(dtype=float)  # T x A
    time_index = wide.index.to_numpy()
    agent_ids = wide.columns.to_numpy()

    return M, time_index, agent_ids


def _compute_rank_stability(
    M: np.ndarray,
    use_cumulative: bool = True,
    skip_all_zero_steps: bool = True,
) -> float:
    """
    Average adjacent rank similarity over time.

    If use_cumulative=True, ranks are based on cumulative totals up to each timestep.
    This captures hierarchy stability / lock-in.
    """
    if M is None or M.shape[0] < 2:
        return np.nan

    X = np.cumsum(M, axis=0) if use_cumulative else M.copy()

    sims = []
    for t in range(1, X.shape[0]):
        prev_v = X[t - 1]
        curr_v = X[t]

        if skip_all_zero_steps and prev_v.sum() == 0 and curr_v.sum() == 0:
            continue

        sim = _rank_similarity(prev_v, curr_v)
        if np.isfinite(sim):
            sims.append(sim)

    return float(np.mean(sims)) if len(sims) else np.nan


def _compute_early_late_advantage(
    M: np.ndarray,
    early_frac: float = 0.25,
) -> dict:
    """
    Compare early-vs-late allocation across agents.

    Returns:
      - early_late_share_corr: correlation between early share and late share
      - early_leader_kept: whether early leader matches late leader
      - early_leader_final: whether early leader matches final leader
    """
    if M is None or M.shape[0] < 2:
        return {
            "early_late_share_corr": np.nan,
            "early_leader_kept": np.nan,
            "early_leader_final": np.nan,
        }

    T = M.shape[0]
    cut = int(np.floor(T * early_frac))
    cut = max(1, min(cut, T - 1))

    early = M[:cut].sum(axis=0)
    late = M[cut:].sum(axis=0)
    final = M.sum(axis=0)

    early_total = early.sum()
    late_total = late.sum()
    final_total = final.sum()

    if early_total <= 0 or late_total <= 0 or final_total <= 0:
        return {
            "early_late_share_corr": np.nan,
            "early_leader_kept": np.nan,
            "early_leader_final": np.nan,
        }

    early_share = early / early_total
    late_share = late / late_total
    final_share = final / final_total

    if np.std(early_share) == 0 or np.std(late_share) == 0:
        corr = np.nan
    else:
        corr = np.corrcoef(early_share, late_share)[0, 1]

    early_leader = int(np.argmax(early_share))
    late_leader = int(np.argmax(late_share))
    final_leader = int(np.argmax(final_share))

    return {
        "early_late_share_corr": float(corr) if np.isfinite(corr) else np.nan,
        "early_leader_kept": float(early_leader == late_leader),
        "early_leader_final": float(early_leader == final_leader),
    }


def _compute_turn_taking_fairness(
    M: np.ndarray,
    leader_window: int = 25,
) -> dict:
    """
    Bin time into windows, assign each active window to the top agent in that window,
    and quantify how evenly leadership rotates.

    Returns:
      - turn_taking_fairness: normalized entropy of window leaders in [0, 1]
      - leader_switch_rate: fraction of adjacent active windows whose leaders differ
      - n_active_leader_windows: number of windows that contained any activity
    """
    if M is None or M.shape[0] == 0:
        return {
            "turn_taking_fairness": np.nan,
            "leader_switch_rate": np.nan,
            "n_active_leader_windows": 0,
        }

    T, A = M.shape
    leader_ids = []

    for start in range(0, T, leader_window):
        stop = min(start + leader_window, T)
        window_totals = M[start:stop].sum(axis=0)

        if window_totals.sum() <= 0:
            continue

        leader_ids.append(int(np.argmax(window_totals)))

    if len(leader_ids) == 0:
        return {
            "turn_taking_fairness": np.nan,
            "leader_switch_rate": np.nan,
            "n_active_leader_windows": 0,
        }

    counts = np.bincount(leader_ids, minlength=A)
    fairness = _normalized_entropy_from_counts(counts)

    if len(leader_ids) < 2:
        switch_rate = np.nan
    else:
        leader_ids = np.asarray(leader_ids)
        switch_rate = np.mean(leader_ids[1:] != leader_ids[:-1])

    return {
        "turn_taking_fairness": float(fairness) if np.isfinite(fairness) else np.nan,
        "leader_switch_rate": float(switch_rate) if np.isfinite(switch_rate) else np.nan,
        "n_active_leader_windows": int(len(leader_ids)),
    }


# -----------------------------
# Main episode-level computation
# -----------------------------

def compute_episode_dynamic_inequality_metrics(
    dff: pd.DataFrame,
    value_col: str = "eating_event",
    time_col: str = "time_step",
    env_col: str = "env_id",
    episode_col: str = "episode_index",
    agent_col: str = "agent_id",
    size_col: str = "agent_size",
    early_frac: float = 0.25,
    leader_window: int = 25,
    min_agents: int = 2,
    min_total_value: float = 0.0,
):
    """
    Compute per-episode dynamic inequality metrics across agents.

    Metrics:
      - dominance_ratio
      - rank_stability
      - early_late_share_corr
      - early_leader_kept
      - early_leader_final
      - turn_taking_fairness
      - leader_switch_rate

    Notes:
      - `value_col` is assumed additive over timesteps within agent
        (e.g. food events, rewards, bites, etc.)
      - rank stability uses cumulative totals over time
      - turn-taking uses windowed leaders over `leader_window` timesteps

    Returns:
      episode_df
    """
    df = dff.copy()

    keep_cols = [env_col, episode_col, agent_col, time_col, value_col, size_col]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    results = []

    grouped = df.groupby([env_col, episode_col], sort=False)

    for (env_id, episode_index), g in grouped:
        n_agents = g[agent_col].nunique()
        if n_agents < min_agents:
            continue

        # agent totals
        per_agent = (
            g.groupby(agent_col)
             .agg(
                 total_value=(value_col, "sum"),
                 agent_size=(size_col, "first"),
             )
             .reset_index()
             .sort_values(agent_col)
        )

        total_value = per_agent["total_value"].sum()
        if total_value < min_total_value:
            continue

        M, _, agent_ids = _build_episode_matrix(
            g=g,
            time_col=time_col,
            agent_col=agent_col,
            value_col=value_col,
        )
        if M is None:
            continue

        dominance = _dominance_ratio(per_agent["total_value"].to_numpy())
        rank_stability = _compute_rank_stability(M, use_cumulative=True)
        early_late = _compute_early_late_advantage(M, early_frac=early_frac)
        turn_taking = _compute_turn_taking_fairness(M, leader_window=leader_window)

        row = {
            env_col: env_id,
            episode_col: episode_index,
            "n_agents": n_agents,
            "total_value": float(total_value),
            "mean_value_per_agent": float(per_agent["total_value"].mean()),
            "dominance_ratio": dominance,
            "rank_stability": rank_stability,
            **early_late,
            **turn_taking,
        }

        # Optional: include Gini on size if your helper exists
        if "calculate_gini" in globals():
            try:
                row["gini_size"] = calculate_gini(per_agent["agent_size"].to_numpy())
            except Exception:
                row["gini_size"] = np.nan

        results.append(row)

    episode_df = pd.DataFrame(results)

    if len(episode_df) == 0:
        return episode_df

    return episode_df.reset_index(drop=True)


# -----------------------------
# Optional plotting helper
# -----------------------------

def plot_gini_agent_size_vs_dynamic_metric(
    dff: pd.DataFrame,
    metric_col: str,
    value_col: str = "eating_event",
    time_col: str = "time_step",
    save_path: str = None,
    load_data: bool = False,
    save_data: bool = True,
    ax=None,
    early_frac: float = 0.25,
    leader_window: int = 25,
):
    """
    Scatter + regression:
      x = Gini(agent size)
      y = one dynamic inequality metric

    Requires `calculate_gini(...)` and optionally `_save_data/_load_data`.
    """
    base = os.path.splitext(save_path)[0] if save_path else None

    if load_data and base and "_load_data" in globals():
        episode_df = _load_data(f"{base}_dynamic_episode")
    else:
        episode_df = compute_episode_dynamic_inequality_metrics(
            dff=dff,
            value_col=value_col,
            time_col=time_col,
            early_frac=early_frac,
            leader_window=leader_window,
        )

        if save_data and base and "_save_data" in globals():
            _save_data(episode_df, f"{base}_dynamic_episode")

    if len(episode_df) == 0:
        raise ValueError("No valid episodes found for plotting.")

    if "gini_size" not in episode_df.columns:
        raise ValueError("gini_size not found. Make sure calculate_gini(...) is defined.")

    plot_df = episode_df[
        np.isfinite(episode_df["gini_size"]) & np.isfinite(episode_df[metric_col])
    ].copy()

    if len(plot_df) == 0:
        raise ValueError(f"No finite values available for metric '{metric_col}'.")

    if ax is None:
        if "_make_fig" in globals():
            fig, ax = _make_fig(1, 1, width_multiplier=1)
        else:
            fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    sns.regplot(
        data=plot_df,
        x="gini_size",
        y=metric_col,
        scatter_kws={"alpha": 0.6},
        line_kws={"color": "red"},
        ax=ax,
    )

    try:
        r, p = stats.pearsonr(
            plot_df["gini_size"].to_numpy(),
            plot_df[metric_col].to_numpy(),
        )
        txt = f"r = {r:.3f}\np = {p:.1e}\nN = {len(plot_df)}"
        at = AnchoredText(
            txt, loc="upper right", prop=dict(size=9),
            frameon=True, borderpad=0.4, pad=0.3
        )
        at.patch.set_alpha(0.7)
        at.patch.set_edgecolor("black")
        ax.add_artist(at)
    except Exception:
        pass

    ylabel_map = {
        "dominance_ratio": "Dominance Ratio (Max Share)",
        "rank_stability": "Rank Stability Over Time",
        "early_late_share_corr": "Early vs Late Share Correlation",
        "early_leader_kept": "Early Leader == Late Leader",
        "early_leader_final": "Early Leader == Final Leader",
        "turn_taking_fairness": "Turn-Taking / Temporal Fairness",
        "leader_switch_rate": "Leader Switch Rate",
    }

    ax.set_xlabel("Inequality in Agent Size")
    ax.set_ylabel(ylabel_map.get(metric_col, metric_col))

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig, ax, episode_df


def plot_gini_size_vs_episode_metric_df(
    episode_df: pd.DataFrame,
    metric_col: str,
    save_path: str = None,
    ax=None,
):
    """
    Plot x = gini_size vs y = metric_col from a precomputed episode_df.

    Returns:
      fig, ax, plot_df
    """
    if len(episode_df) == 0:
        raise ValueError("episode_df is empty.")

    required_cols = ["gini_size", metric_col]
    missing = [c for c in required_cols if c not in episode_df.columns]
    if missing:
        raise ValueError(f"episode_df is missing required columns: {missing}")

    plot_df = episode_df[
        np.isfinite(episode_df["gini_size"]) &
        np.isfinite(episode_df[metric_col])
    ].copy()

    if len(plot_df) == 0:
        raise ValueError(f"No finite values available for metric '{metric_col}'.")

    if ax is None:
        fig, ax = _make_fig(1, 1, width_multiplier=1)
    else:
        fig = ax.figure

    sns.regplot(
        data=plot_df,
        x="gini_size",
        y=metric_col,
        scatter_kws={"alpha": 0.6},
        line_kws={"color": "red"},
        ax=ax,
    )

    try:
        r, p = stats.pearsonr(
            plot_df["gini_size"].to_numpy(),
            plot_df[metric_col].to_numpy(),
        )
        txt = f"r = {r:.3f}\np = {p:.1e}\nN = {len(plot_df)}"
        at = AnchoredText(
            txt,
            loc="upper right",
            prop=dict(size=9),
            frameon=True,
            borderpad=0.4,
            pad=0.3,
        )
        at.patch.set_alpha(0.7)
        at.patch.set_edgecolor("black")
        ax.add_artist(at)
    except Exception:
        pass

    ylabel_map = {
        "dominance_ratio": "Dominance Ratio (Max Share)",
        "rank_stability": "Rank Stability Over Time",
        "early_late_share_corr": "Early vs Late Share Correlation",
        "early_leader_kept": "Early Leader == Late Leader",
        "early_leader_final": "Early Leader == Final Leader",
        "turn_taking_fairness": "Turn-Taking / Temporal Fairness",
        "leader_switch_rate": "Leader Switch Rate",
    }

    ax.set_xlabel("Inequality in Agent Size")
    ax.set_ylabel(ylabel_map.get(metric_col, metric_col))

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig, ax, plot_df


def plot_prob_bite_vs_agent_size(
    dff,
    save_path: str = None,
    load_data: bool = False,
    save_data: bool = True,
    ax=None
):
    """
    P(bite_other_fish | has_nearby) vs agent size.
    """
    base = os.path.splitext(save_path)[0] if save_path else None
    dff = dff.copy()
    dff['bite_and_nearby'] = dff['bite_other_fish'] & dff['has_nearby']

    if load_data and base:
        grouped = _load_data(f"{base}_grouped_bite")
    else:
        grouped = (
            dff.groupby(['env_id','episode_index','agent_id'])
               .agg(prob_bite_given_nearby=('bite_and_nearby','sum'),
                    has_nearby=('has_nearby','sum'),
                    agent_size=('agent_size','first'))
               .reset_index()
        )
        grouped['P(Bite | Agent Nearby)'] = grouped['prob_bite_given_nearby']/grouped['has_nearby']
        if save_data and base:
            _save_data(grouped, f"{base}_grouped_bite")

    if ax is None:
        fig, ax = _make_fig(1, 1, width_multiplier=1)
    else:
        fig = ax.figure
    sns.regplot(
        data=grouped,
        x='agent_size', y='P(Bite | Agent Nearby)',
        scatter_kws={'alpha':0.6},
        line_kws   ={'color':'red'},
        ax=ax
    )
    try:
        # remove NaNs for correlation
        clean = grouped[['agent_size', 'P(Bite | Agent Nearby)']].dropna()
        r, p = stats.pearsonr(clean['agent_size'], clean['P(Bite | Agent Nearby)'])
        txt = f"r = {r:.3f}\np = {p:.1e}"
        at = AnchoredText(
            txt, loc="upper right", prop=dict(size=9),
            frameon=True, borderpad=0.4, pad=0.3
        )
        at.patch.set_alpha(0.7)
        at.patch.set_edgecolor("black")
        ax.add_artist(at)
        # ax.set_title(f"P(Bite | Agent Nearby) vs Agent Size\nr={r:.3f}, p={p:.3e}")
    except:
        print("Warning: Pearson correlation failed, skipping title update for plot_prob_bite_vs_agent_size.")
        # print traceback
        import traceback
        traceback.print_exc()
    ax.set_xlabel("Agent Size")
    ax.set_ylabel("P(Bite | Agent Nearby)")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig, ax


def plot_size_diff_vs_prob_bite(
    dff,
    save_path: str = None,
    load_data: bool = False,
    save_data: bool = True
):
    """
    P(Bite | Agent Nearby) vs (biter_size - bitten_size).
    """
    dff = dff.copy()
    dff['bite_and_nearby'] = dff['bite_other_fish'] & dff['has_nearby']
    base = os.path.splitext(save_path)[0] if save_path else None
    evt = dff[dff['was_bitten']].copy()
    sizes = (
        dff[['agent_id', 'agent_size', 'episode_index', 'env_id', 'time_step']]
        .rename(columns={'agent_id':'was_bitten_by_agent_id','agent_size':'agent_size_biter'})
    )
    merged = evt.merge(sizes, on=['was_bitten_by_agent_id', 'episode_index', 'env_id', 'time_step'])
    merged['size_diff'] = merged['agent_size_biter'] - merged['agent_size']

    if load_data and base:
        grp = _load_data(f"{base}_grouped_size_diff")
    else:
        grp = (
            merged.groupby(['env_id','episode_index','agent_id'])
                  .agg(prob_bite_given_nearby=('bite_and_nearby','sum'),
                       has_nearby=('has_nearby','sum'),
                       size_diff=('size_diff','first'))
                  .reset_index()
        )
        grp['P(Bite | Agent Nearby)'] = grp['prob_bite_given_nearby']/grp['has_nearby']
        if save_data and base:
            _save_data(grp, f"{base}_grouped_size_diff")

    fig, ax = _make_fig(1,1)
    sns.regplot(
        data=grp,
        x='size_diff', y='P(Bite | Agent Nearby)',
        scatter_kws={'alpha':0.6},
        line_kws   ={'color':'red'},
        ax=ax
    )
    try:
        r, p = stats.pearsonr(grp['size_diff'], grp['P(Bite | Agent Nearby)'])
        ax.set_title(f"P(Bite | Agent Nearby) vs Size Diff\nr={r:.3f}, p={p:.3e}")
    except:
        pass
    ax.set_xlabel("Biter Size − Bitten Size")
    ax.set_ylabel("P(Bite | Agent Nearby)")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig, ax


def _p_to_stars(p):
    if p is None or np.isnan(p): return "n.s."
    return "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def plot_agent_size_by_role_biting(
        dff,
        outputs_folder=None,
        outfile_base=None,
):
    """
    Plot agent size by role (biter vs bitten) using boxplot.
    Upgraded: uses paired Wilcoxon test, robust annotation, and graceful handling
    """
    biter_sizes = []
    bitten_sizes = []
    skipped_counter = 0

    biting_events = dff[dff['was_bitten']]

    for _, event in biting_events.iterrows():
        env_id = event['env_id']
        episode_index = event['episode_index']
        time_step = event['time_step']
        bitten_agent_id = event['agent_id']
        biting_agent_id = event['was_bitten_by_agent_id']

        mask_bitten = (
            (dff['env_id'] == env_id) &
            (dff['episode_index'] == episode_index) &
            (dff['agent_id'] == bitten_agent_id) &
            (dff['time_step'] == time_step)
        )
        mask_biter = (
            (dff['env_id'] == env_id) &
            (dff['episode_index'] == episode_index) &
            (dff['agent_id'] == biting_agent_id) &
            (dff['time_step'] == time_step)
        )

        if not mask_bitten.any() or not mask_biter.any():
            skipped_counter += 1
            continue

        bitten_size = dff.loc[mask_bitten, 'agent_size'].values[0]
        biter_size = dff.loc[mask_biter, 'agent_size'].values[0]

        biter_sizes.append(biter_size)
        bitten_sizes.append(bitten_size)

    if skipped_counter > 0:
        print(f"[INFO] Skipped {skipped_counter} biting events due to missing agent data")

    if len(biter_sizes) == 0:
        print("[WARN] No valid biting events found")
        return {'test': 'Wilcoxon signed-rank', 'statistic': np.nan, 'pvalue': np.nan, 'n_pairs': 0, 'stars': 'n.s.'}

    biter_sizes = pd.Series(biter_sizes)
    bitten_sizes = pd.Series(bitten_sizes)

    stat, pval = np.nan, np.nan
    try:
        stat, pval = wilcoxon(
            biter_sizes, bitten_sizes,
            alternative='two-sided', zero_method='pratt', mode='auto')
    except Exception as e:
        print(f"[WARN] Wilcoxon(pratt) failed ({e}); retrying with zero_method='wilcox'")
        try:
            stat, pval = wilcoxon(
                biter_sizes, bitten_sizes,
                alternative='two-sided', zero_method='wilcox', mode='auto')
        except Exception as e2:
            print(f"[ERROR] Wilcoxon test failed: {e2}")
            stat, pval = np.nan, np.nan

    stars = _p_to_stars(pval)
    n_pairs = len(biter_sizes)
    print(f"[INFO] Wilcoxon signed-rank: W={stat:.3f}  p={pval:.6g}  stars={stars}")

    # Data for plotting (long form)
    data_q1 = pd.DataFrame({
        'Agent Size': pd.concat([biter_sizes, bitten_sizes]),
        'role': ['Biter'] * len(biter_sizes) + ['Bitten'] * len(bitten_sizes)
    })

    fig, ax = _make_fig(1, 1, edge=2)
    sns.boxplot(x='role', y='Agent Size', data=data_q1, ax=ax)
    pairs = [("Biter", "Bitten")]
    ax.set_xlabel("")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Star annotation using our mapping ---
    if np.isfinite(pval):
        try:
            annotator = Annotator(ax, pairs, data=data_q1, x="role", y="Agent Size", order=["Biter", "Bitten"])
            annotator.configure(test=None, text_format='simple', loc='inside', verbose=0)
            annotator.set_custom_annotations([stars]).annotate()
        except Exception as e:
            # fallback: simple text
            print(f"[WARN] Skipping statannotation stars: {e}")
            ax.text(0.5, 0.95, stars, transform=ax.transAxes, ha='center', va='top',
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5})

    if outfile_base is not None:
        fname = f"{outfile_base}_agent_size_by_biting_role_boxplot.pdf"
    elif outputs_folder:
        fname = f"{outputs_folder}/agent_size_by_biting_role_boxplot.pdf"
    else:
        print("[WARN] No outputs_folder or outfile_base provided, not saving agent_size_by_biting_role boxplot.")
        plt.show()
        return {'test': 'Wilcoxon signed-rank', 'statistic': float(stat), 'pvalue': float(pval), 'n_pairs': int(n_pairs), 'stars': stars}

    plt.tight_layout()
    fig.savefig(fname, bbox_inches='tight', dpi=300)
    print(f"Saved agent size by biting role plot to {fname}")

    # 2D Histogram
    fig, ax = _make_fig(1, 1, edge=2)
    h = ax.hist2d(biter_sizes, bitten_sizes, bins=5, cmap='Blues', cmin=1)
    ax.set_xlabel("Biter Size")
    ax.set_ylabel("Bitten Size")
    fig.colorbar(h[3], ax=ax, label='Counts')

    if outfile_base is not None:
        fname = f"{outfile_base}_agent_size_by_biting_role_hist2d.pdf"
    elif outputs_folder:
        fname = f"{outputs_folder}/agent_size_by_biting_role_hist2d.pdf"
    else:
        print("[WARN] No outputs_folder or outfile_base provided, not saving agent_size_by_biting_role histogram.")
        plt.show()
        return {'test': 'Wilcoxon signed-rank', 'statistic': float(stat), 'pvalue': float(pval), 'n_pairs': int(n_pairs), 'stars': stars}

    plt.tight_layout()
    fig.savefig(fname, bbox_inches='tight', dpi=300)
    print(f"Saved agent size by biting role histogram to {fname}")

    difference_of_means = biter_sizes.mean() - bitten_sizes.mean()
    return {'test': 'Wilcoxon signed-rank', 'statistic': float(stat), 'pvalue': float(pval), 'n_pairs': int(n_pairs), 'stars': stars, 'difference_of_means': float(difference_of_means)}


def plot_move_forward_histogram(
    dff,
    save_path: str = None,
    load_data: bool = False,
    save_data: bool = True
):
    """
    Histogram of forward moves (move_forward >= 0).
    """
    base = os.path.splitext(save_path)[0] if save_path else None

    # load or compute
    if load_data and base:
        moves = _load_data(f"{base}_forward")
    else:
        moves = dff.loc[dff['move_forward'] >= 0, 'move_forward'].tolist()
        if save_data and base:
            _save_data(moves, f"{base}_forward")

    # plot
    fig, ax = _make_fig(1, 1)
    ax.hist(moves, bins=30)
    ax.set_xlabel('Move Forward')
    ax.set_ylabel('Count')
    ax.set_title('Move Forward Distribution (Forward Only)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig, ax


def plot_move_backward_histogram(
    dff,
    save_path: str = None,
    load_data: bool = False,
    save_data: bool = True
):
    """
    Histogram of backward moves (|move_forward| where move_forward < 0).
    """
    base = os.path.splitext(save_path)[0] if save_path else None

    # load or compute
    if load_data and base:
        moves = _load_data(f"{base}_backward")
    else:
        moves = dff.loc[dff['move_forward'] < 0, 'move_forward'].abs().tolist()
        if save_data and base:
            _save_data(moves, f"{base}_backward")

    # plot
    fig, ax = _make_fig(1, 1)
    ax.hist(moves, bins=30)
    ax.set_xlabel('Move Backward')
    ax.set_ylabel('Count')
    ax.set_title('Move Backward Distribution')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig, ax


def plot_consecutive_backward_windows_histogram(
    dff,
    save_path: str = None,
    load_data: bool = False,
    save_data: bool = True
):
    """
    Histogram of lengths of consecutive backward-move windows.
    """
    base = os.path.splitext(save_path)[0] if save_path else None

    # load or compute
    if load_data and base:
        windows = _load_data(f"{base}_backward_windows")
    else:
        windows = []
        for (_, aid, epi), grp in dff.groupby(['env_id','agent_id','episode_index']):
            grp = grp.sort_values('time_step')
            is_b = grp['move_forward'] < 0
            t = is_b.astype(int).diff()
            starts = grp.index[t==1].tolist()
            ends   = grp.index[t==-1].tolist()
            if is_b.iloc[0]: starts.insert(0, grp.index[0])
            if is_b.iloc[-1]: ends.append(grp.index[-1])
            for s,e in zip(starts, ends):
                windows.append(len(grp.loc[s:e]))
        if save_data and base:
            _save_data(windows, f"{base}_backward_windows")

    cutoff = np.percentile(windows, 98)
    filt   = [w for w in windows if w <= cutoff]

    # plot
    fig, (ax1, ax2) = _make_fig(1, 2)
    ax1.hist(windows, bins=50, edgecolor='black')
    ax1.axvline(cutoff, color='r', linestyle='--',
                label=f'98th percentile: {cutoff:.1f}')
    ax1.set_xlabel('Window Length')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Full Distribution (n={len(windows):,})')
    ax1.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    mf = int(np.ceil(max(filt)))
    mi = int(np.floor(min(filt)))
    bins = np.arange(mi, mf+2) - 0.5
    ax2.hist(filt, bins=bins, edgecolor='black')
    ax2.set_xlabel('Window Length')
    ax2.set_ylabel('Count')
    ax2.set_title(f'≤98th percentile (n={len(filt):,})')
    ax2.set_xticks(np.arange(mi, mf+1, 1))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig, (ax1, ax2)


def plot_forward_move_histogram_overlay(
    dff,
    condition_col: str = 'has_nearby',
    true_label: str = r'$d_{a} \leq$10 cm',
    false_label: str = r'$d_{a} >$10 cm',
    save_path: str = None,
    load_data: bool = False,
    save_data: bool = True
):
    """
    Plot an overlaid histogram of forward movement magnitudes,
    split by whether `condition_col` is True or False.
    """
    import matplotlib.pyplot as plt
    import os

    base = os.path.splitext(save_path)[0] if save_path else None

    if load_data and base:
        data = _load_data(f"{base}_forward_by_{condition_col}")
        forward_true = data['true']
        forward_false = data['false']
    else:
        df_forward = dff[dff['move_forward'] >= 0]
        forward_true = df_forward[df_forward[condition_col]]['move_forward'].tolist()
        forward_false = df_forward[~df_forward[condition_col]]['move_forward'].tolist()
        if save_data and base:
            _save_data({'true': forward_true, 'false': forward_false},
                       f"{base}_forward_by_{condition_col}")

    fig, ax = _make_fig(1, 1, height_multiplier=2/3, width_multiplier=3/4)
    # sns.kdeplot(forward_false, ax=ax, label=f"{false_label} (n={len(forward_false):,})", color='C0', alpha=0.4)
    # sns.kdeplot(forward_true, ax=ax, label=f"{true_label} (n={len(forward_true):,})", color='C1', alpha=0.4)
    sns.kdeplot(forward_false, ax=ax, label=f"{false_label}", color='C0', alpha=0.4)
    sns.kdeplot(forward_true, ax=ax, label=f"{true_label}", color='C1', alpha=0.4)

    # ax.hist(forward_true, bins=30, alpha=0.6, label=f"{condition_col}=True (n={len(forward_true):,})", color='C0', density=True)
    # ax.hist(forward_false, bins=30, alpha=0.6, label=f"{condition_col}=False (n={len(forward_false):,})", color='C1', density=True)

    ax.set_xlabel("Forward Action Magnitude")
    ax.set_ylabel("Density")
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig, ax


def plot_backward_move_histogram_overlay(
    dff,
    condition_col: str = 'has_nearby',
    save_path: str = None,
    load_data: bool = False,
    save_data: bool = True
):
    """
    Plot an overlaid histogram of backward movement magnitudes,
    split by whether `condition_col` is True or False.
    """
    base = os.path.splitext(save_path)[0] if save_path else None

    if load_data and base:
        data = _load_data(f"{base}_backward_by_{condition_col}")
        backward_true = data['true']
        backward_false = data['false']
    else:
        df_backward = dff[dff['move_forward'] < 0].copy()
        df_backward['move_backward'] = df_backward['move_forward'].abs()
        backward_true = df_backward[df_backward[condition_col]]['move_backward'].tolist()
        backward_false = df_backward[~df_backward[condition_col]]['move_backward'].tolist()
        if save_data and base:
            _save_data({'true': backward_true, 'false': backward_false},
                       f"{base}_backward_by_{condition_col}")

    fig, ax = _make_fig(1, 1)
    ax.hist(backward_true, bins=30, alpha=0.6, label=f"{condition_col}=True (n={len(backward_true):,})", color='green', density=True)
    ax.hist(backward_false, bins=30, alpha=0.6, label=f"{condition_col}=False (n={len(backward_false):,})", color='red', density=True)

    ax.set_xlabel("Backward Action Magnitude")
    ax.set_ylabel("Density")
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig, ax


def plot_displacement_histogram_overlay(
    dff,
    condition_col: str = 'has_nearby',
    true_label: str = r'$d_{a} \leq$10 cm',
    false_label: str = r'$d_{a} >$10 cm',
    save_path: str = None,
    load_data: bool = False,
    save_data: bool = True
):
    """
    Plot an overlaid histogram of forward movement magnitudes,
    split by whether `condition_col` is True or False.
    """
    import matplotlib.pyplot as plt
    import os

    base = os.path.splitext(save_path)[0] if save_path else None

    if load_data and base:
        data = _load_data(f"{base}_displacement_by_{condition_col}")
        displacement_true = data['true']
        displacement_false = data['false']
    else:
        displacement_true = dff[dff[condition_col]]['displacement'].tolist()
        displacement_false = dff[~dff[condition_col]]['displacement'].tolist()
        if save_data and base:
            _save_data({'true': displacement_true, 'false': displacement_false},
                       f"{base}_displacement_by_{condition_col}")

    fig, ax = _make_fig(1, 1, height_multiplier=2/3, width_multiplier=3/4)

    sns.kdeplot(displacement_false, ax=ax, label=f"{false_label}", color='C0', alpha=0.4, clip=(0, np.inf))
    sns.kdeplot(displacement_true, ax=ax, label=f"{true_label}", color='C1', alpha=0.4, clip=(0, np.inf))

    ax.set_xlabel("Displacement (cm)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig, ax


def plot_summary_figs(dff, outfile_base):
    # plot displacement and turn angle histograms
    plot_displacement_histogram(dff, save_path=outfile_base + '_displacement_histogram_all.pdf')
    plot_conditional_displacement_histograms(dff, condition_col='has_nearby', save_path=outfile_base + '_displacement_histogram_has_nearby.pdf')
    plot_conditional_displacement_histograms(dff, condition_col='food_observed', save_path=outfile_base + '_displacement_histogram_food_observed.pdf')
    plot_aligned_movement_2d_histogram(dff, save_path=outfile_base + '_aligned_movement_2d_histogram.pdf')
    plot_displacement_histogram_overlay(dff, condition_col='has_nearby', true_label=r'$d_{a} \leq$10 cm', false_label=r'$d_{a} >$10 cm', save_path=outfile_base + '_displacement_histogram_overlay_has_nearby.pdf')
    plot_displacement_histogram_overlay(dff, condition_col='food_observed', true_label=r'$d_{f} \leq$5 cm', false_label=r'$d_{f} >$5 cm', save_path=outfile_base + '_displacement_histogram_overlay_food_observed.pdf')

    # turn angle histograms
    plot_turn_angle_histogram(dff, save_path=outfile_base + '_turn_angle_histogram_all.pdf')
    plot_conditional_turn_angle_histograms(dff, condition_col='has_nearby', save_path=outfile_base + '_turn_angle_histogram_has_nearby.pdf')
    plot_conditional_turn_angle_histograms(dff, condition_col='food_observed', save_path=outfile_base + '_turn_angle_histogram_food_observed.pdf')

    # # Move forward/backward histograms
    plot_move_forward_histogram(dff, save_path=outfile_base + '_move_forward_histogram_all.pdf')
    plot_forward_move_histogram_overlay(dff, condition_col='has_nearby', save_path=outfile_base + '_move_forward_histogram_has_nearby.pdf')
    plot_forward_move_histogram_overlay(dff, condition_col='food_observed', true_label=r'$d_{f} \leq$5 cm', false_label=r'$d_{f} >$5 cm', save_path=outfile_base + '_move_forward_histogram_food_observed.pdf')

    if dff['move_forward'].min() < 0:
        plot_move_backward_histogram(dff, save_path=outfile_base + '_move_backward_histogram_all.pdf')
        plot_backward_move_histogram_overlay(dff, condition_col='has_nearby', save_path=outfile_base + '_move_backward_histogram_has_nearby.pdf')
        plot_backward_move_histogram_overlay(dff, condition_col='food_observed', save_path=outfile_base + '_move_backward_histogram_food_observed.pdf')
        plot_consecutive_backward_windows_histogram(dff, save_path=outfile_base + '_consecutive_backward_windows_histogram.pdf')
    else:
        print("No backward movement detected, skipping backward movement plots.")

    # eating related
    try:
        plot_angle_to_closest_food_polar_histogram(dff, save_path=outfile_base + '_angle_to_closest_food_polar_histogram.pdf')
        plot_conditional_angle_to_closest_food_polar_histograms(dff, save_path=outfile_base + '_angle_to_closest_food_polar_histogram_food_observed.pdf')
        plot_combined_angle_to_closest_food_polar_histogram(dff, save_path=outfile_base + '_angle_to_closest_food_polar_histogram_food_observed_combined.pdf')
        plot_movement_between_eating(dff, max_steps_between=10, save_path=outfile_base + '_movement_between_eating.pdf')
        plot_movement_around_eating(dff, steps_around=4, save_path=outfile_base + '_movement_around_eating.pdf')
        plot_distance_to_closest_food_histogram(dff, save_path=outfile_base + '_distance_to_food_histogram.pdf')
        plot_conditional_distance_to_closest_food_histograms(dff, save_path=outfile_base + '_conditional_distance_to_food_histograms.pdf')
        plot_food_metrics_before_eating(dff, save_path=outfile_base + '_food_dist_and_angle_before_eating.pdf')
        plot_time_between_eating_histogram(dff, save_path=outfile_base + '_time_between_eating_histogram.pdf')
        morm_amp_df, _ = add_morm_amp_field_features(dff, [])
        plot_conditional_angle_polar_histogram(morm_amp_df, condition_col='food_observed', angle_col="mormyromast_error_angle", save_path=outfile_base + '_mormyromast_error_angle_food_observed.pdf')
        plot_conditional_angle_polar_histogram(morm_amp_df, condition_col='food_observed', angle_col="ampullary_error_angle", save_path=outfile_base + '_ampullary_error_angle_food_observed.pdf')
    except Exception as e:
        print(f"Error in eating related plots: {e}")

    # aggression and size plots
    try:
        plot_food_eaten_vs_agent_size(dff, save_path=outfile_base + '_food_eaten_vs_agent_size.pdf')
        plot_food_eaten_normalized_vs_agent_size(dff, save_path=outfile_base + '_food_eaten_normalized_vs_agent_size.pdf')
        plot_gini_agent_size_vs_food_inequality(dff, save_path=outfile_base + '_gini_size_vs_food_normalized.pdf', normalize_food_by_area=True)
        plot_gini_agent_size_vs_food_inequality(dff, save_path=outfile_base + '_gini_size_vs_food.pdf', normalize_food_by_area=False)
        plot_gini_agent_size_vs_eod_inequality(dff, save_path=outfile_base + '_gini_size_vs_eod.pdf')
        plot_gini_size_vs_food_bottom_and_top_quartiles(
            dff,
            save_path_bottom=outfile_base + '_gini_size_vs_food_bottom_25_normalized.pdf',
            save_path_top=outfile_base + '_gini_size_vs_food_top_25_normalized.pdf',
            normalize_food_by_area=True
        )
        plot_gini_size_vs_food_bottom_and_top_quartiles(
            dff,
            save_path_bottom=outfile_base + '_gini_size_vs_food_bottom_25.pdf',
            save_path_top=outfile_base + '_gini_size_vs_food_top_25.pdf',
            normalize_food_by_area=False
        )

        metrics_to_plot = [
            "dominance_ratio",
            "rank_stability",
            "early_late_share_corr",
            "early_leader_kept",
            "early_leader_final",
            "turn_taking_fairness",
            "leader_switch_rate",
        ]

        for label, col in [("food", "eating_event"), ("eod", "emit_eod")]:
            episode_df = compute_episode_dynamic_inequality_metrics(
                dff,
                value_col=col,
                time_col="time_step",
                early_frac=0.25,
                leader_window=25,
            )

            for metric in metrics_to_plot:
                fig, ax, plot_df = plot_gini_size_vs_episode_metric_df(
                    episode_df,
                    metric_col=metric,
                    save_path=f"{outfile_base}_gini_size_vs_{label}_{metric}.pdf",
                )


        if dff['was_bitten'].sum() > 0:
            plot_prob_bite_vs_agent_size(dff, save_path=outfile_base + '_size_vs_prob_bite.pdf')
            plot_prob_bite_action_prob_vs_agent_size(dff, save_path=outfile_base + '_size_vs_prob_bite_action.pdf')
            plot_size_diff_vs_prob_bite(dff, save_path=outfile_base + '_size_diff_vs_prob_bite.pdf')
            biting_summary = plot_agent_size_by_role_biting(dff, outfile_base=outfile_base)
            biting_summary_df = pd.DataFrame([biting_summary], index=[0])
            biting_summary_df.to_csv(outfile_base + '_biting_size_summary.csv', index=False, sep='\t')

    except Exception as e:
        print(f"Error in aggression and size plots: {e}")
