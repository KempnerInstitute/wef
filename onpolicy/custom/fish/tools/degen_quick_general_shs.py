import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import linregress


DEFAULT_BASE_PATH = "./results/degen10/"
M1A1K1_PATTERN = "**/outputs/reports/**/general/*_m1a1k1cs1sd0*_run_summary.csv"
M1A1K1_AGENT_PATTERN = "**/outputs/reports/**/general/*_m1a1k1cs1sd0*_run_agent_summary.csv"
CSV_PATH_COL = "csv_path"
REPORT_DIR_COL = "report_dir_path"
REPORT_LABEL_COL = "report_label"
M1_FEATURE_COLS = [
    "food_eaten",
    "num_biting_events",
    "p_emit_eod",
    "p_near_food",
    "p_near_other_fish",
    "mean_nn_distance_cm",
    "polarization",
    "food_eaten_theil",
    "displacement_20_step",
]
RUN_KEY_COLS = [REPORT_DIR_COL, "seed", "run_timestamp_id", "folder_name", REPORT_LABEL_COL]
JOIN_KEY_COLS = [REPORT_DIR_COL, "run_timestamp_id"]
JOIN_METADATA_COLS = ["seed", "run_timestamp_id", "folder_name", REPORT_LABEL_COL, CSV_PATH_COL]
EPISODE_KEY_COLS = ["run_id", "env_id", "episode_index"]
FOOD_EATEN_SIZE_SLOPE_COL = "slope_food_eaten_vs_agent_size"
FOOD_EATEN_SIZE_INTERCEPT_COL = "intercept_food_eaten_vs_agent_size"
FOOD_EATEN_SIZE_R2_COL = "r2_food_eaten_vs_agent_size"
REPRESENTATIVE_PATTERNS = [
    "**/outputs/reports/**/general/*_run_summary.csv",
    "**/outputs/reports/**/general/*_run_agent_summary.csv",
]
DEFAULT_TWO_FISH_PATTERN = "**/outputs/reports/*TwoFishSquare*/general/*_run_summary.csv"
DEFAULT_TWO_FISH_AGENT_PATTERN = "**/outputs/reports/*TwoFishSquare*/general/*_run_agent_summary.csv"


@dataclass(frozen=True)
class AnalysisContext:
    base_path: str
    base_path_abs: str
    out_dir: str
    out_dir_abs: str
    run_folders: List[str]
    run_paths: List[str]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "base_path",
        nargs="?",
        default=DEFAULT_BASE_PATH,
        help="Root folder containing run sub-folders.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Where to write output CSVs. Defaults to base_path.",
    )
    return parser.parse_args()


def build_context(base_path=DEFAULT_BASE_PATH, out_dir=None, run_folders=None, create_out_dir=True):
    base_path_abs = os.path.abspath(base_path)
    if run_folders is None:
        run_folders = discover_run_folders(base_path_abs)
    out_dir = out_dir or base_path
    out_dir_abs = os.path.abspath(out_dir)
    if create_out_dir:
        os.makedirs(out_dir_abs, exist_ok=True)
    run_paths = [os.path.join(base_path_abs, run) for run in run_folders]
    return AnalysisContext(
        base_path=base_path,
        base_path_abs=base_path_abs,
        out_dir=out_dir,
        out_dir_abs=out_dir_abs,
        run_folders=run_folders,
        run_paths=run_paths,
    )


def discover_run_folders(base_path):
    return sorted(
        name for name in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, name))
    )


def list_all_csvs(root_path):
    return glob.glob(os.path.join(root_path, "**", "*.csv"), recursive=True)


def read_csv_auto(path):
    df = pd.read_csv(path)
    if df.shape[1] == 1 and "\t" in df.columns[0]:
        df = pd.read_csv(path, sep="\t")
    return df


def extract_path_metadata(file_path, base_path):
    rel = os.path.relpath(file_path, base_path)
    parts = rel.split(os.sep)
    folder_name = parts[0] if parts else None
    seed_match = None
    for part in parts:
        seed_match = re.search(r"Seed(\d+)", part)
        if seed_match:
            break
    seed = int(seed_match.group(1)) if seed_match else None
    timestamp = None
    for part in parts:
        timestamp_match = re.search(r"(\d{8}_\d{6})", part)
        if timestamp_match:
            timestamp = timestamp_match.group(1)
            break
    return seed, timestamp, folder_name


def attach_run_metadata(df, csv_file, base_path_abs):
    df = df.copy()
    seed, timestamp, folder_name = extract_path_metadata(csv_file, base_path_abs)
    report_dir = os.path.dirname(os.path.dirname(csv_file))
    df["seed"] = seed
    df["run_timestamp_id"] = timestamp
    df["folder_name"] = folder_name
    df[CSV_PATH_COL] = csv_file
    df[REPORT_DIR_COL] = report_dir
    df[REPORT_LABEL_COL] = os.path.basename(report_dir)
    return df


def filter_m1_arena(df, selection="both"):
    if df is None or df.empty:
        return df
    selection = (selection or "both").lower()
    if selection == "both":
        return df
    if selection not in {"patchy", "uniform"}:
        raise ValueError("m1_arena_selection must be one of: 'patchy', 'uniform', 'both'")

    needle = selection.lower()
    mask = pd.Series(False, index=df.index)
    if "arena_type" in df.columns:
        mask = mask | df["arena_type"].astype(str).str.lower().str.contains(needle, na=False)
    for col in [REPORT_LABEL_COL, CSV_PATH_COL]:
        if col in df.columns:
            mask = mask | df[col].astype(str).str.lower().str.contains(needle, na=False)
    return df[mask].copy()


def load_matching_csvs(all_csvs, base_path_abs, matcher, label):
    matched_paths = [path for path in all_csvs if matcher(path)]
    print(f"[{label}] Matched {len(matched_paths)} CSV(s)")
    frames = []
    for csv_file in matched_paths:
        df = attach_run_metadata(read_csv_auto(csv_file), csv_file, base_path_abs)
        frames.append(df)
        print(f"[{label}] Found: {os.path.basename(csv_file)}  shape={df.shape}")
    combined = pd.concat(frames, ignore_index=True) if frames else None
    return matched_paths, frames, combined


def flatten_multiindex_columns(df):
    flat = df.copy()
    flat.columns = [
        f"{left}_{right}" if right else str(left)
        for left, right in flat.columns.to_list()
    ]
    return flat


def available_key_cols(df, key_cols):
    if df is None:
        return []
    return [col for col in key_cols if col in df.columns]


def available_run_key_cols(df):
    return available_key_cols(df, RUN_KEY_COLS)


def available_join_key_cols(df):
    if df is None:
        return []
    if REPORT_DIR_COL in df.columns:
        return [REPORT_DIR_COL]
    return available_key_cols(df, [col for col in JOIN_KEY_COLS if col != REPORT_DIR_COL])


def available_episode_key_cols(df):
    return available_key_cols(df, EPISODE_KEY_COLS)


def available_analysis_key_cols(df):
    return available_run_key_cols(df) + available_episode_key_cols(df)


def aggregate_metric_frame(metrics_df, group_cols, agg="mean"):
    if metrics_df is None or metrics_df.empty or not group_cols:
        return metrics_df

    available_group_cols = [col for col in group_cols if col in metrics_df.columns]
    if not available_group_cols:
        return None

    numeric_metric_cols = [
        col
        for col in metrics_df.select_dtypes(include="number").columns
        if col not in available_group_cols and col not in RUN_KEY_COLS and col not in EPISODE_KEY_COLS
    ]
    metadata_cols = [
        col
        for col in metrics_df.columns
        if (
            col not in available_group_cols
            and col not in numeric_metric_cols
            and col not in EPISODE_KEY_COLS
        )
    ]
    if not numeric_metric_cols and not metadata_cols:
        return metrics_df[available_group_cols].drop_duplicates()

    agg_fn = "median" if agg == "median" else "mean"
    agg_map = {col: agg_fn for col in numeric_metric_cols}
    agg_map.update({col: "first" for col in metadata_cols})
    return metrics_df.groupby(available_group_cols, dropna=False).agg(agg_map).reset_index()


def fit_food_eaten_vs_agent_size(df_agent):
    if df_agent is None or df_agent.empty:
        return None

    required_cols = {"food_eaten", "agent_size"}
    if not required_cols.issubset(df_agent.columns):
        return None

    group_cols = available_analysis_key_cols(df_agent)
    if not group_cols:
        return None

    def _fit(group):
        valid = group.dropna(subset=["food_eaten", "agent_size"])
        if len(valid) < 2 or valid["agent_size"].nunique() < 2:
            return pd.Series(
                {
                    FOOD_EATEN_SIZE_SLOPE_COL: np.nan,
                    FOOD_EATEN_SIZE_INTERCEPT_COL: np.nan,
                    FOOD_EATEN_SIZE_R2_COL: np.nan,
                }
            )
        fit = linregress(valid["agent_size"], valid["food_eaten"])
        return pd.Series(
            {
                FOOD_EATEN_SIZE_SLOPE_COL: fit.slope,
                FOOD_EATEN_SIZE_INTERCEPT_COL: fit.intercept,
                FOOD_EATEN_SIZE_R2_COL: fit.rvalue ** 2,
            }
        )

    return (
        df_agent.groupby(group_cols, dropna=False)[["food_eaten", "agent_size"]]
        .apply(_fit)
        .reset_index()
    )


def merge_run_level_metrics(left_df, metrics_df, validate):
    if left_df is None or left_df.empty or metrics_df is None or metrics_df.empty:
        return left_df

    merge_cols = [col for col in available_analysis_key_cols(left_df) if col in metrics_df.columns]
    if not merge_cols:
        return left_df

    metric_cols = [col for col in metrics_df.columns if col not in merge_cols]
    existing_metric_cols = [col for col in metric_cols if col in left_df.columns]
    merged_left = left_df.drop(columns=existing_metric_cols) if existing_metric_cols else left_df
    return merged_left.merge(metrics_df, on=merge_cols, how="left", validate=validate)


def describe_dataframe(df, max_cols=12):
    missingness = df.isna().mean().sort_values(ascending=False)
    top_missing = missingness.head(5)
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "preview_columns": list(df.columns)[:max_cols],
        "top_missingness": top_missing[top_missing > 0].to_dict(),
        "head": df.head(),
    }


def summarize_df(df, name, max_cols=12, display_fn=None):
    summary = describe_dataframe(df, max_cols=max_cols)
    print(f"\n=== {name} ===")
    print("shape:", summary["shape"])
    preview_cols = summary["preview_columns"]
    if len(summary["columns"]) > max_cols:
        preview_cols = preview_cols + ["..."]
    print("columns:", preview_cols)
    if summary["top_missingness"]:
        print("missingness (top 5):", summary["top_missingness"])
    print("head:")
    if display_fn is not None:
        display_fn(summary["head"])
    else:
        print(summary["head"].to_string(index=False))
    return summary


def summarize_m1a1k1(df_m1a1k1):
    if df_m1a1k1 is None or df_m1a1k1.empty:
        return None
    if "arena_type" not in df_m1a1k1.columns:
        raise ValueError("arena_type column not found; cannot group by arena_type.")
    numeric_cols = df_m1a1k1.select_dtypes(include="number").columns
    group_cols = [
        col
        for col in [REPORT_DIR_COL, "seed", "run_timestamp_id", "folder_name", REPORT_LABEL_COL, "arena_type"]
        if col in df_m1a1k1.columns
    ]
    df_summary = (
        df_m1a1k1.groupby(group_cols)[numeric_cols]
        .agg(["mean", "std", "sum", "median"])
        .sort_index()
    )
    patchy_only = (
        df_summary[df_summary.index.get_level_values("arena_type") == "PatchyArena"]
        if "arena_type" in group_cols
        else df_summary.iloc[0:0]
    )
    return {
        "summary": df_summary,
        "flat_summary": flatten_multiindex_columns(df_summary.reset_index()),
        "patchy_only": patchy_only,
    }


def summarize_agent_runs(df_agent):
    if df_agent is None or df_agent.empty:
        return None
    agent_numeric = df_agent.select_dtypes(include="number").columns
    group_cols = [
        col
        for col in [REPORT_DIR_COL, "seed", "run_timestamp_id", "folder_name", REPORT_LABEL_COL]
        if col in df_agent.columns
    ]
    df_agent_summary = (
        df_agent.groupby(group_cols)[agent_numeric]
        .agg(["mean", "std", "sum"])
        .sort_index()
    )
    return {
        "summary": df_agent_summary,
        "flat_summary": flatten_multiindex_columns(df_agent_summary.reset_index()),
    }


def summarize_2f1p(df_2f1p):
    if df_2f1p is None or df_2f1p.empty:
        return None
    group_cols = [
        col
        for col in [REPORT_DIR_COL, "seed", "run_timestamp_id", "folder_name", REPORT_LABEL_COL, "condition"]
        if col in df_2f1p.columns
    ]
    pivot_index_cols = [col for col in group_cols if col != "condition"]
    summary_2f1p = (
        df_2f1p.groupby(group_cols)
        .agg(
            success_rate_B=("success_bool", "mean"),
            n_episodes=("success_bool", "count"),
            start_dist_B_center_mean=("start_dist_B_center", "mean"),
            ab_dist_mean_mean=("ab_dist_mean", "mean"),
            ab_dist_min_mean=("ab_dist_min", "mean"),
        )
        .reset_index()
        .sort_values(group_cols)
    )
    pivot_2f1p = summary_2f1p.pivot_table(
        index=pivot_index_cols,
        columns="condition",
        values="success_rate_B",
    )
    return {
        "summary": summary_2f1p,
        "pivot": pivot_2f1p,
    }


def summarize_2f1p_grid(df_grid):
    if df_grid is None or df_grid.empty:
        return None
    df_grid = df_grid.copy()
    df_grid["size_diff_B_minus_A"] = df_grid["size_diff_B_minus_A"].round(1)
    group_cols = [
        col
        for col in [
            REPORT_DIR_COL,
            "seed",
            "run_timestamp_id",
            "folder_name",
            REPORT_LABEL_COL,
            "size_diff_B_minus_A",
            "init_radius_cm_B",
        ]
        if col in df_grid.columns
    ]
    pivot_index_cols = [col for col in group_cols if col != "init_radius_cm_B"]
    summary_grid = (
        df_grid.groupby(group_cols)
        .agg(
            success_rate_B=("success_B", "mean"),
            n_episodes=("success_B", "count"),
            total_eaten_B_mean=("total_eaten_B", "mean"),
            total_eaten_A_mean=("total_eaten_A", "mean"),
        )
        .reset_index()
        .sort_values(group_cols)
    )
    pivot_grid = summary_grid.pivot_table(
        index=pivot_index_cols,
        columns="init_radius_cm_B",
        values="success_rate_B",
    )
    pivot_grid.columns = [f"r={int(col)}cm" for col in pivot_grid.columns]
    return {
        "summary": summary_grid,
        "pivot": pivot_grid,
    }


def compute_joined_analysis(
    df_m1a1k1,
    summary_2f1p=None,
    summary_grid=None,
    feature_cols=None,
    run_level_metrics=None,
    m1_arena_selection="both",
):
    if df_m1a1k1 is None or df_m1a1k1.empty:
        return None

    feature_cols = feature_cols or M1_FEATURE_COLS
    present_feat_cols = [col for col in feature_cols if col in df_m1a1k1.columns]
    if not present_feat_cols:
        return None

    m1_group_cols = available_join_key_cols(df_m1a1k1)
    if not m1_group_cols:
        return None

    m1_rows = filter_m1_arena(df_m1a1k1, m1_arena_selection)
    if m1_rows is None or m1_rows.empty:
        return None
    m1_source = m1_arena_selection or "both"
    outcome_join_cols = m1_group_cols

    m1_feats = m1_rows.groupby(m1_group_cols)[present_feat_cols].mean().reset_index()
    run_level_metrics_agg = aggregate_metric_frame(run_level_metrics, m1_group_cols, agg="median")
    m1_feats = merge_run_level_metrics(m1_feats, run_level_metrics_agg, validate="one_to_one")
    extra_run_metadata_cols = [
        col
        for col in RUN_KEY_COLS
        if col not in m1_group_cols and col not in JOIN_METADATA_COLS and col in m1_feats.columns
    ]
    if extra_run_metadata_cols:
        m1_feats = m1_feats.drop(columns=extra_run_metadata_cols)
    outcome_blocks = []
    outcome_columns = []
    joined_sources = []

    if summary_2f1p is not None and not summary_2f1p.empty:
        summary_2f1p_group_cols = available_join_key_cols(summary_2f1p)
        cond_code = {"B only": 0, "A < B": 1, "A = B": 2, "A > B": 3}
        summary_2f1p = summary_2f1p.copy()
        summary_2f1p["cond_code"] = summary_2f1p["condition"].map(cond_code)

        def slope_2f1p(group):
            valid = group.dropna(subset=["cond_code", "success_rate_B"])
            if len(valid) < 2:
                return pd.Series({"slope_succB_vs_Adominance": np.nan})
            return pd.Series(
                {"slope_succB_vs_Adominance": linregress(valid["cond_code"], valid["success_rate_B"]).slope}
            )

        slopes_2f1p = (
            summary_2f1p.groupby(summary_2f1p_group_cols)[["cond_code", "success_rate_B"]]
            .apply(slope_2f1p)
            .reset_index()
        )
        pivot_succB_cond = (
            summary_2f1p.pivot_table(
                index=summary_2f1p_group_cols,
                columns="condition",
                values="success_rate_B",
            )
            .rename(
                columns=lambda col: (
                    f"succB_{col.replace(' ', '_').replace('<', 'lt').replace('>', 'gt').replace('=', 'eq')}"
                )
            )
            .reset_index()
        )
        if {"succB_B_only", "succB_A_lt_B"}.issubset(pivot_succB_cond.columns):
            pivot_succB_cond["delta_AltB_vs_Bonly"] = (
                pivot_succB_cond["succB_A_lt_B"] - pivot_succB_cond["succB_B_only"]
            )
        if {"succB_B_only", "succB_A_eq_B"}.issubset(pivot_succB_cond.columns):
            pivot_succB_cond["delta_AeqB_vs_Bonly"] = (
                pivot_succB_cond["succB_A_eq_B"] - pivot_succB_cond["succB_B_only"]
            )
        if {"succB_B_only", "succB_A_gt_B"}.issubset(pivot_succB_cond.columns):
            pivot_succB_cond["delta_AgtB_vs_Bonly"] = (
                pivot_succB_cond["succB_A_gt_B"] - pivot_succB_cond["succB_B_only"]
            )
        outcome_blocks.extend([slopes_2f1p, pivot_succB_cond])
        outcome_columns.extend(
            col for col in slopes_2f1p.columns
            if col not in summary_2f1p_group_cols
        )
        outcome_columns.extend(
            col for col in pivot_succB_cond.columns
            if col not in summary_2f1p_group_cols
        )
        joined_sources.append("2f1p")

    if summary_grid is not None and not summary_grid.empty:
        summary_grid_group_cols = available_join_key_cols(summary_grid)

        def slope_grid(group):
            valid = group.dropna(subset=["size_diff_B_minus_A", "success_rate_B"])
            if len(valid) < 2:
                return pd.Series({"slope_succB_vs_sizediff": np.nan})
            return pd.Series(
                {
                    "slope_succB_vs_sizediff": linregress(
                        valid["size_diff_B_minus_A"],
                        valid["success_rate_B"],
                    ).slope
                }
            )

        slopes_grid_pooled = (
            summary_grid.groupby(summary_grid_group_cols)[["size_diff_B_minus_A", "success_rate_B"]]
            .apply(slope_grid)
            .reset_index()
            .rename(columns={"slope_succB_vs_sizediff": "slope_succB_sizediff_pooled"})
        )
        slopes_grid_by_radius = (
            summary_grid.groupby(summary_grid_group_cols + ["init_radius_cm_B"])[
                ["size_diff_B_minus_A", "success_rate_B"]
            ]
            .apply(slope_grid)
            .reset_index()
        )
        pivot_slopes_by_radius = (
            slopes_grid_by_radius.pivot_table(
                index=summary_grid_group_cols,
                columns="init_radius_cm_B",
                values="slope_succB_vs_sizediff",
            )
            .rename(columns=lambda radius: f"slope_succB_sizediff_r{int(radius)}cm")
            .reset_index()
        )
        mean_success_by_radius = (
            summary_grid.groupby(summary_grid_group_cols + ["init_radius_cm_B"])["success_rate_B"]
            .mean()
            .unstack("init_radius_cm_B")
            .rename(columns=lambda radius: f"mean_succB_r{int(radius)}cm")
            .reset_index()
        )
        outcome_blocks.extend([slopes_grid_pooled, pivot_slopes_by_radius, mean_success_by_radius])
        outcome_columns.extend(
            col for col in slopes_grid_pooled.columns
            if col not in summary_grid_group_cols
        )
        outcome_columns.extend(
            col for col in pivot_slopes_by_radius.columns
            if col not in summary_grid_group_cols
        )
        outcome_columns.extend(
            col for col in mean_success_by_radius.columns
            if col not in summary_grid_group_cols
        )
        joined_sources.append("2f1p_grid")

    joined = m1_feats.copy()
    for right_df in outcome_blocks:
        merge_cols = [col for col in outcome_join_cols if col in right_df.columns]
        joined = joined.merge(right_df, on=merge_cols, how="left")
    joined = joined.sort_values(m1_group_cols).reset_index(drop=True)

    numeric_joined = joined.select_dtypes(include="number")
    m1_cols_present = [col for col in present_feat_cols if col in numeric_joined.columns]
    outcome_cols_all = [
        col for col in outcome_columns
        if col in numeric_joined.columns
    ]
    if outcome_cols_all:
        corr_block = numeric_joined[m1_cols_present + outcome_cols_all].corr()
        corr_m1_vs_outcomes = corr_block.loc[m1_cols_present, outcome_cols_all]
        strong = []
        for feat in m1_cols_present:
            for outcome in outcome_cols_all:
                corr = corr_m1_vs_outcomes.loc[feat, outcome]
                if not np.isnan(corr) and abs(corr) > 0.5:
                    strong.append((feat, outcome, corr))
        strong.sort(key=lambda item: -abs(item[2]))
    else:
        corr_m1_vs_outcomes = pd.DataFrame(index=m1_cols_present)
        strong = []

    return {
        "joined": joined,
        "correlation": corr_m1_vs_outcomes,
        "strong_associations": strong,
        "present_feature_columns": present_feat_cols,
        "joined_sources": joined_sources,
        "m1_source": m1_source,
        "outcome_columns": outcome_cols_all,
        "join_level": m1_group_cols,
    }


def inspect_run_csvs(run_path):
    csv_paths = list_all_csvs(run_path)
    rows = []
    for csv_file in csv_paths:
        df = read_csv_auto(csv_file)
        rows.append(
            {
                "csv_path": csv_file,
                "csv_name": os.path.basename(csv_file),
                "num_rows": df.shape[0],
                "num_cols": df.shape[1],
            }
        )
    return pd.DataFrame(rows).sort_values("csv_path").reset_index(drop=True)


def first_match(run_path, pattern):
    matches = glob.glob(os.path.join(run_path, pattern), recursive=True)
    return matches[0] if matches else None


def find_representative_files(run_path, patterns=None):
    patterns = patterns or REPRESENTATIVE_PATTERNS
    return {pattern: first_match(run_path, pattern) for pattern in patterns}


def load_representative_tables(run_path, patterns=None):
    files = find_representative_files(run_path, patterns=patterns)
    return {
        pattern: read_csv_auto(path)
        for pattern, path in files.items()
        if path is not None
    }


def load_single_run_table(run_path, pattern):
    match = first_match(run_path, pattern)
    if match is None:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    return match, read_csv_auto(match)


def export_joined_outputs(joined_analysis, out_dir):
    joined_path = os.path.join(out_dir, "joined_m1a1k1_outcomes.csv")
    corr_path = os.path.join(out_dir, "corr_m1a1k1_vs_outcomes.csv")
    joined_analysis["joined"].to_csv(joined_path, index=False)
    joined_analysis["correlation"].to_csv(corr_path)
    return {
        "joined_path": joined_path,
        "correlation_path": corr_path,
    }


def print_strong_associations(strong_associations):
    if not strong_associations:
        print("\nNo associations with |r| > 0.5 found.")
        return
    print("\n=== Strong associations (|r| > 0.5) ===")
    print(f"{'m1a1k1 feature':<28} {'outcome':<42} {'r':>7}")
    print("-" * 80)
    for feat, outcome, corr in strong_associations:
        print(f"{feat:<28} {outcome:<42} {corr:+.3f}")


def run_multi_run_analysis(context, m1_arena_selection="both"):
    all_csvs = list_all_csvs(context.base_path_abs)
    _, _, df_m1a1k1 = load_matching_csvs(
        all_csvs,
        context.base_path_abs,
        matcher=lambda path: glob.fnmatch.fnmatch(path, M1A1K1_PATTERN),
        label="m1a1k1",
    )
    _, _, df_agent = load_matching_csvs(
        all_csvs,
        context.base_path_abs,
        matcher=lambda path: glob.fnmatch.fnmatch(path, M1A1K1_AGENT_PATTERN),
        label="m1a1k1_agent",
    )
    df_m1a1k1 = filter_m1_arena(df_m1a1k1, m1_arena_selection)
    df_agent = filter_m1_arena(df_agent, m1_arena_selection)
    _, _, df_2f1p = load_matching_csvs(
        all_csvs,
        context.base_path_abs,
        matcher=lambda path: "2f1p_comparisons" in path and path.endswith("_summary_2f1p.csv"),
        label="2f1p",
    )
    _, _, df_grid = load_matching_csvs(
        all_csvs,
        context.base_path_abs,
        matcher=lambda path: "2f1p_grid" in path and path.endswith("_augmented.csv"),
        label="2f1p_grid",
    )

    food_eaten_size_regression = fit_food_eaten_vs_agent_size(df_agent)
    df_m1a1k1 = merge_run_level_metrics(df_m1a1k1, food_eaten_size_regression, validate="many_to_one")

    m1 = summarize_m1a1k1(df_m1a1k1) if df_m1a1k1 is not None else None
    agent = summarize_agent_runs(df_agent) if df_agent is not None else None
    two_fish = summarize_2f1p(df_2f1p) if df_2f1p is not None else None
    grid = summarize_2f1p_grid(df_grid) if df_grid is not None else None
    joined = None
    if df_m1a1k1 is not None:
        joined = compute_joined_analysis(
            df_m1a1k1=df_m1a1k1,
            summary_2f1p=two_fish["summary"] if two_fish is not None else None,
            summary_grid=grid["summary"] if grid is not None else None,
            run_level_metrics=food_eaten_size_regression,
            m1_arena_selection=m1_arena_selection,
        )
    return {
        "all_csvs": all_csvs,
        "m1_arena_selection": m1_arena_selection,
        "df_m1a1k1": df_m1a1k1,
        "df_agent": df_agent,
        "food_eaten_vs_agent_size": food_eaten_size_regression,
        "df_2f1p": df_2f1p,
        "df_grid": df_grid,
        "m1": m1,
        "agent": agent,
        "two_fish": two_fish,
        "grid": grid,
        "joined": joined,
    }


def print_multi_run_analysis(results, out_dir):
    m1 = results["m1"]
    if m1 is None:
        print("No m1a1k1 run_summary CSVs found.")
    else:
        cols_to_show = [
            ("food_eaten", "sum"),
            ("food_eaten", "mean"),
            ("food_eaten", "std"),
            ("num_biting_events", "sum"),
            ("num_biting_events", "mean"),
            ("num_biting_events", "std"),
            ("food_eaten_theil", "mean"),
            ("food_eaten_theil", "std"),
            ("num_interactions", "sum"),
            ("num_interactions", "mean"),
            ("num_interactions", "std"),
        ]
        present_cols = [col for col in cols_to_show if col in m1["summary"].columns]
        print("\n=== m1a1k1 run_summary (all arena types) ===")
        print(m1["summary"][present_cols].sort_values(by=[("food_eaten", "sum")], ascending=False))
        print("\n=== m1a1k1 run_summary (PatchyArena only) ===")
        print(m1["patchy_only"][present_cols].sort_values(by=[("food_eaten", "sum")], ascending=False))

    agent = results["agent"]
    if agent is not None:
        agent_cols = [
            ("food_eaten", "sum"),
            ("food_eaten", "mean"),
            ("food_eaten", "std"),
            ("num_biting_events", "sum"),
            ("num_biting_events", "mean"),
            ("num_biting_events", "std"),
        ]
        present_cols = [col for col in agent_cols if col in agent["summary"].columns]
        print("\n=== m1a1k1 run_agent_summary ===")
        print(agent["summary"][present_cols].sort_values(by=[("food_eaten", "sum")], ascending=False))

    two_fish = results["two_fish"]
    if two_fish is None:
        print("No 2f1p CSVs found.")
    else:
        print("\n=== 2f1p: B success rate by condition (A presence/size relative to B) ===")
        print(two_fish["summary"].to_string(index=False))
        print("\n=== 2f1p: success_rate_B pivot (rows=run, cols=condition) ===")
        print(two_fish["pivot"].to_string())

    grid = results["grid"]
    if grid is None:
        print("No 2f1p_grid CSVs found.")
    else:
        print("\n=== 2f1p_grid: B success rate by (size_diff_B_minus_A, init_radius_cm_B) ===")
        print(grid["summary"].to_string(index=False))
        print("\n=== 2f1p_grid: success_rate_B pivot (rows=size_diff, cols=init_radius) ===")
        print(grid["pivot"].to_string())

    joined = results["joined"]
    if joined is None:
        print("\n[join] Skipping — no usable m1a1k1 feature columns were found.")
        return

    pd.set_option("display.max_columns", 60)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.3f}".format)
    join_level = joined.get("join_level", RUN_KEY_COLS)
    print(f"\n=== JOINED TABLE (one row per {', '.join(join_level)}) ===")
    print(f"[join] m1 base: {joined['m1_source']}")
    if joined["joined_sources"]:
        print(f"[join] merged outcomes: {', '.join(joined['joined_sources'])}")
    else:
        print("[join] merged outcomes: none available; showing m1a1k1 features only")
    print(joined["joined"].to_string(index=False))
    exports = export_joined_outputs(joined, out_dir)
    print(f"\n[export] Joined table  -> {exports['joined_path']}")
    if joined["outcome_columns"]:
        print("\n=== CORRELATION: m1a1k1 features × available outcome datasets ===")
        print("(Pearson r; n=runs)\n")
        print(joined["correlation"].to_string())
        print(f"[export] Correlation matrix -> {exports['correlation_path']}")
        print_strong_associations(joined["strong_associations"])
    else:
        print("[join] No outcome datasets were available, so no correlation matrix was computed.")


def main():
    args = parse_args()
    context = build_context(base_path=args.base_path, out_dir=args.out_dir)
    print(f"Using BASE_PATH: {context.base_path_abs}")
    print(f"OUT_DIR: {context.out_dir_abs}")
    print(f"Discovered {len(context.run_folders)} run folders")
    results = run_multi_run_analysis(context)
    print_multi_run_analysis(results, context.out_dir_abs)


if __name__ == "__main__":
    main()
