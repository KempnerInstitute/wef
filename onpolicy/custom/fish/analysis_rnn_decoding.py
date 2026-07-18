"""
Linear decoding of behavioral/spatial features from RNN hidden states.  Fig 6E.

Uses ordinary least squares (LinearRegression, matching old_code/utils_decoding)
+ GroupKFold (groups = episode×env) so whole episodes are held out as test sets.
Splits data by distance to closest food:
  food_near : distance_to_closest_food <= morm_food_range_cm
  food_far  : distance_to_closest_food >  morm_food_range_cm

The plotted/reported value is a baseline-normalized R²: within each fold we
compute (model R² − baseline R²) and average across folds (matching old_code's
perf_test_normalized = perf_test − perf_test_baseline).  The baseline is the R²
of predicting the training mean on the held-out test fold (DummyRegressor mean).
Reported error bars (``r2_sem``) are the standard error of the per-fold
normalized R² across the GroupKFold splits (std / sqrt(n_folds)), NOT a std.

Usage
-----
    python analysis_rnn_decoding.py --spec_dir path/to/evals/2fish_m1a1k1_uniform_square
    # tune cross-validation / downsampling:
    python analysis_rnn_decoding.py --spec_dir ... --n-cv-splits 5 --max-rows 5000

Outputs to {spec_dir}/analyses/rnn_decoding/
    decoding_food_near.csv   columns: feature, r2_mean (= normalized: model − baseline),
                                      r2_sem, r2_raw (raw model R²), r2_dummy (baseline R²),
                                      n_obs (finite rows used for the fit)
    decoding_food_far.csv
    decoding_combined.png/.pdf
    decoding_feature_dists.png/.pdf   (sanity check)
"""

import argparse
import glob
import json
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_style import set_style, panel, save, COND_PALETTE
from rnn_loader import iter_rnn_episodes
from features import EXCLUDE_FROM_DECODING, FEATURE_METADATA
from utils_features import add_distance_to_wall

OUT_SUBDIR = "analyses/rnn_decoding"
DEFAULT_MORM_FOOD_RANGE_CM = 5.0   # cfg.py default
MAX_ROWS_PER_CONDITION = 5000      # per-condition downsample cap (override: --max-rows)
N_CV_SPLITS = 5                    # GroupKFold splits (override: --n-cv-splits)
TOP_N = 12   # features to show in bar chart
# Conspecific-distance split thresholds (cm): <lo, [lo, hi], >hi
AGENT_RANGE_BINS_CM = (10.0, 100.0)

# Per-fold mean-predictor baseline subtracted from each model R² before averaging
# (matches old_code's perf_test_normalized = perf_test - perf_test_baseline).

# B2: features decoded (kept in CSVs) but hidden from the bar chart because they
# are trivial/uninformative and otherwise dominate it — agent_size is a per-agent
# constant, actual_turn is the agent's own motor output, position_{x,y} is raw
# arena location.  (old_code excluded agent_size via DEFAULT_PERF_PLOT_EXCLUSION_LIST.)
# The directional food counts (food_{front,back,left,right}_5cm) are hidden for now
# at the user's request; they are still decoded and written to the CSVs.
PLOT_EXCLUDE_FEATURES = {
    "agent_size", "actual_turn", "position_x", "position_y",
    "food_front_5cm", "food_back_5cm", "food_left_5cm", "food_right_5cm",
}


# ── sensor-range helper ────────────────────────────────────────────────────────

def get_morm_food_range_cm(spec_dir: str) -> float:
    """
    Walk up from spec_dir to find logs/*_env_args.json and read
    agent_env_args.morm_food_detection_range_m, converted to cm.
    Falls back to cfg default if not found.
    """
    p = spec_dir
    for _ in range(6):
        candidates = glob.glob(os.path.join(p, "logs", "env_args.json"))
        if candidates:
            with open(candidates[0]) as f:
                env_args = json.load(f)
            v = env_args.get("agent_env_args", {}).get("morm_food_detection_range_m")
            if v is not None:
                return float(v) * 100.0
            break
        p = os.path.dirname(p)
    return DEFAULT_MORM_FOOD_RANGE_CM


# ── feature engineering ────────────────────────────────────────────────────────

def _add_field_features(dff: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Derive scalar features from center_field.* columns."""
    feats = []
    for sensor in ("mormyromast", "ampullary"):
        col = f"center_field.{sensor}"
        if col not in dff.columns:
            continue
        try:
            x = dff[col].apply(lambda v: float(v[0]) if hasattr(v, "__len__") else np.nan)
            y = dff[col].apply(lambda v: float(v[1]) if hasattr(v, "__len__") else np.nan)
            mag_col = f"{sensor}_field_mag"
            dff[mag_col] = np.hypot(x, y)
            feats.append(mag_col)   # raw linear magnitude — excluded from decoding (see features.py)
            # log10 magnitude (decoded instead of the raw linear one); matches
            # old_code add_morm_amp_field_features: log10(m) if m>0 else 0.
            m = dff[mag_col].values.astype(float)
            log = np.zeros_like(m)
            np.log10(m, where=m > 0, out=log)
            log_col = f"{sensor}_field_mag_log"
            dff[log_col] = log
            feats.append(log_col)
        except Exception:
            pass
    # Knollen error angle (nearest agent direction error)
    if "center_field.knollen" in dff.columns and "orientation" in dff.columns:
        try:
            kx = dff["center_field.knollen"].apply(
                lambda v: float(v[0]) if hasattr(v, "__len__") else np.nan)
            ky = dff["center_field.knollen"].apply(
                lambda v: float(v[1]) if hasattr(v, "__len__") else np.nan)
            dirs = np.arctan2(ky.values, kx.values)
            ori  = dff["orientation"].values
            err  = np.mod(ori - dirs + np.pi, 2 * np.pi) - np.pi
            dff["knollen_error_angle"] = err
            feats.append("knollen_error_angle")
        except Exception:
            pass
    # Distance to nearest wall (position + arena_size); the modular homing decoder
    # adds this unconditionally via the same helper, so mirror it here.
    try:
        dff, feats = add_distance_to_wall(dff, feats)
    except Exception as e:
        print(f"[rnn_decoding] could not add distance_to_wall: {e}", flush=True)
    return dff, feats


SCALAR_FEATURES = [
    "position_x", "position_y",
    "distance_to_nearest_agent",
    "distance_to_closest_food",
    "angle_to_closest_agent",
    "angle_to_closest_food",
    "displacement",
    "move_forward",
    "actual_turn",
    "agent_size",
    "energy",
    # ── A1: sensory/social features already present in per_env_ep_agent_step.pkl ──
    "num_agents_in_knollen_range",
    "size_of_nearest_agent",
    "distance_to_second_nearest_agent",
    "size_of_second_nearest_agent",
    "food_count_5cm",
    "food_front_5cm", "food_back_5cm", "food_left_5cm", "food_right_5cm",
]
BOOL_FEATURES = [
    "has_nearby", "emit_eod", "eating_event", "was_bitten",
    # ── A1 ──
    "has_agent_in_knollen_range",
]
# A2: the derived center_field magnitudes (mormyromast_field_mag,
# ampullary_field_mag) are intentionally NOT excluded here so they are decoded.
_EXCLUDE_FROM_DECODING = set(EXCLUDE_FROM_DECODING)
# Circular features get cos/sin treatment.  The homing error angles
# (mormyromast/ampullary/knollen_error_angle_nearest) are included so the homing
# decoder — which now reuses decode_feature — treats them correctly; these columns
# never appear in foraging data so listing them here is harmless there.
CIRCULAR_FEATURES = {
    "angle_to_closest_agent", "angle_to_closest_food", "knollen_error_angle",
    "mormyromast_error_angle", "ampullary_error_angle", "knollen_error_angle_nearest",
}

OPTIONAL_DERIVED_FEATURES = [
    "mormyromast_field_mag_log",
    "ampullary_field_mag_log",
    "knollen_error_angle",
    "distance_to_wall",
    "mormyromast_error_angle",
    "ampullary_error_angle",
    "knollen_error_angle_nearest",
]

FEATURE_TABLE_OVERRIDES = {
    "mormyromast_field_mag_log": {
        "name": "Mormyromast Field Magnitude",
        "feature_type": "scalar",
        "description": "Log-transformed Mormyromast field magnitude, when center-field columns are available",
    },
    "ampullary_field_mag_log": {
        "name": "Ampullary Field Magnitude",
        "feature_type": "scalar",
        "description": "Log-transformed Ampullary field magnitude, when center-field columns are available",
    },
    "knollen_error_angle": {
        "name": "Knollen Error Angle",
        "feature_type": "circular",
        "description": "Orientation error relative to the Knollenorgan field direction",
    },
    "distance_to_wall": {
        "name": "Distance to Wall",
        "feature_type": "scalar",
        "description": "Distance from the focal agent to the nearest arena wall",
    },
    "mormyromast_error_angle": {
        "name": "Mormyromast Error Angle",
        "feature_type": "circular",
        "description": "Homing-task error angle for the Mormyromast field",
    },
    "ampullary_error_angle": {
        "name": "Ampullary Error Angle",
        "feature_type": "circular",
        "description": "Homing-task error angle for the Ampullary field",
    },
    "knollen_error_angle_nearest": {
        "name": "Knollen Error Angle to Nearest Agent",
        "feature_type": "circular",
        "description": "Homing-task error angle for the nearest-agent Knollen field",
    },
}

FEATURE_DESCRIPTIONS = {
    "position_x": "Allocentric x position",
    "position_y": "Allocentric y position",
    "distance_to_nearest_agent": "Distance to closest conspecific",
    "distance_to_closest_food": "Distance to nearest food item",
    "angle_to_closest_agent": "Bearing to closest conspecific",
    "angle_to_closest_food": "Bearing to nearest food item",
    "actual_turn": "Realized angular change per timestep",
    "agent_size": "Scalar size context of the focal agent",
    "energy": "Internal energy state, when present in the rollout table",
    "num_agents_in_knollen_range": "Number of conspecifics within Knollenorgan range",
    "size_of_nearest_agent": "Size metadata for the nearest conspecific",
    "distance_to_second_nearest_agent": "Distance to the second-nearest conspecific",
    "size_of_second_nearest_agent": "Size metadata for the second-nearest conspecific",
    "food_count_5cm": "Number of food items within the centered 5 cm local window",
    "food_front_5cm": "Number of food items in the 5 cm front sector",
    "food_back_5cm": "Number of food items in the 5 cm rear sector",
    "food_left_5cm": "Number of food items in the 5 cm left sector",
    "food_right_5cm": "Number of food items in the 5 cm right sector",
    "has_nearby": "Whether a conspecific is within the nearby-agent threshold",
    "was_bitten": "Whether the agent was bitten on that timestep",
    "has_agent_in_knollen_range": "Whether any conspecific is within Knollenorgan range",
}


def _latex_escape(text: object) -> str:
    """Escape plain text for use in a LaTeX table cell."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def _feature_table_rows() -> list[dict[str, str]]:
    """Return decoder target metadata generated from the active decoder constants."""
    ordered = []
    for feature in SCALAR_FEATURES + BOOL_FEATURES + OPTIONAL_DERIVED_FEATURES:
        if feature in _EXCLUDE_FROM_DECODING or feature in ordered:
            continue
        ordered.append(feature)

    rows = []
    for feature in ordered:
        metadata = FEATURE_METADATA.get(feature, {}).copy()
        metadata.update(FEATURE_TABLE_OVERRIDES.get(feature, {}))
        feature_type = metadata.get("feature_type", "scalar")
        if feature in CIRCULAR_FEATURES:
            feature_type = "circular"
        rows.append({
            "target": metadata.get("name", feature.replace("_", " ").title()),
            "field": feature,
            "type": str(feature_type).title(),
            "description": FEATURE_DESCRIPTIONS.get(
                feature,
                metadata.get("description", "Decoder target, when present in the rollout table"),
            ),
        })
    return rows


def write_feature_table_tex(path: str) -> None:
    """Write the supplemental recurrent-state decoder target table as TeX."""
    rows = _feature_table_rows()
    lines = [
        r"\subsection{List of recurrent-state decoding targets}",
        r"\begin{table}[ht]",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{p{0.34\linewidth}p{0.16\linewidth}p{0.42\linewidth}}",
        r"\toprule",
        r"\textbf{Target} & \textbf{Type} & \textbf{Description} \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_latex_escape(row['target'])} & "
            f"{_latex_escape(row['type'])} & "
            f"{_latex_escape(row['description'])} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Recurrent-state decoding targets used in the analyses. Targets absent from a given assay were omitted from that assay-specific decoding fit.}",
        r"\label{tab:features}",
        r"\end{table}",
        "",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def prepare_features(dff: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return dff with derived columns + list of all target feature names."""
    dff = dff.copy()
    dff, derived = _add_field_features(dff)
    all_feats = [f for f in SCALAR_FEATURES + BOOL_FEATURES + derived
                 if f in dff.columns and f not in _EXCLUDE_FROM_DECODING]
    return dff, all_feats


# ── data accumulation ─────────────────────────────────────────────────────────

def accumulate(raw_dir: str, dff: pd.DataFrame, features: list[str]):
    """
    Stream through episodes and build arrays:
      X_all      : (N, H) rnn hidden states
      feat_df    : (N, len(features)) behavioral features
      groups     : (N,)  int — unique per (episode, env)
      dist_food  : (N,)  float — distance_to_closest_food
    """
    X_chunks, feat_chunks, group_chunks, dist_food_chunks = [], [], [], []
    group_counter = 0

    for k, rnn_arr, dff_ep in iter_rnn_episodes(raw_dir, dff):
        T, E, A, H = rnn_arr.shape
        for e in range(E):
            for a in range(A):
                mask = (dff_ep["env_id"] == e) & (dff_ep["agent_id"] == a)
                rows = dff_ep[mask].sort_values("time_step")
                if len(rows) != T:
                    continue    # length mismatch
                X_chunks.append(rnn_arr[:, e, a, :])
                feat_chunks.append(rows[features].values.astype(float))
                group_chunks.extend([group_counter] * T)
                dist_col = "distance_to_closest_food"
                if dist_col in rows.columns:
                    dist_food_chunks.append(rows[dist_col].values.astype(float))
                else:
                    dist_food_chunks.append(np.full(T, np.nan))
                group_counter += 1

    if not X_chunks:
        return None, None, None, None

    X_all       = np.concatenate(X_chunks, axis=0).astype(np.float32)
    feat_all    = np.concatenate(feat_chunks, axis=0)
    groups_all  = np.array(group_chunks, dtype=int)
    dist_food   = np.concatenate(dist_food_chunks, axis=0)
    return X_all, pd.DataFrame(feat_all, columns=features), groups_all, dist_food


# ── cross-validated decoding ──────────────────────────────────────────────────

def decode_feature(X, y, groups, feature_name: str, n_cv_splits: int = N_CV_SPLITS):
    """
    GroupKFold ordinary-least-squares regression with a per-fold mean-predictor
    baseline subtracted.  Returns (r2_mean, r2_sem, r2_raw, r2_dummy), where
    within each fold we compute (model R² − baseline R²) and report the mean/SEM
    of that across folds (matches old_code's perf_test_normalized).  The baseline
    is the R² of predicting the training mean on the held-out test fold
    (DummyRegressor(strategy="mean")), which cancels the GroupKFold mean-shift
    penalty.

    Matches old_code/utils_decoding: plain LinearRegression, no feature
    standardization (OLS R² is invariant to invertible rescaling of X anyway).
    Handles circular features by predicting cos/sin jointly.
    Handles boolean features as float regression.
    """
    valid = np.isfinite(y)
    n_obs = int(valid.sum())   # finite observations used for the fit (post-downsample)
    if n_obs < n_cv_splits * 2:
        return np.nan, np.nan, np.nan, np.nan, n_obs

    X_v, y_v, g_v = X[valid], y[valid], groups[valid]
    n_splits = min(n_cv_splits, len(np.unique(g_v)))
    if n_splits < 2:
        return np.nan, np.nan, np.nan, np.nan, n_obs

    gkf = GroupKFold(n_splits=n_splits)
    r2_raw, r2_base = [], []

    if feature_name in CIRCULAR_FEATURES:
        Y_v = np.stack([np.cos(y_v), np.sin(y_v)], axis=1)
        for train_idx, test_idx in gkf.split(X_v, y_v, g_v):
            mdl = LinearRegression().fit(X_v[train_idx], Y_v[train_idx])
            Y_pred = mdl.predict(X_v[test_idx])
            r2_model = (r2_score(Y_v[test_idx, 0], Y_pred[:, 0]) +
                        r2_score(Y_v[test_idx, 1], Y_pred[:, 1])) / 2
            dummy_pred = np.tile(Y_v[train_idx].mean(axis=0), (len(test_idx), 1))
            r2_b = (r2_score(Y_v[test_idx, 0], dummy_pred[:, 0]) +
                    r2_score(Y_v[test_idx, 1], dummy_pred[:, 1])) / 2
            r2_raw.append(r2_model)
            r2_base.append(r2_b)
    else:
        for train_idx, test_idx in gkf.split(X_v, y_v, g_v):
            mdl = LinearRegression().fit(X_v[train_idx], y_v[train_idx])
            r2_model = r2_score(y_v[test_idx], mdl.predict(X_v[test_idx]))
            dummy = DummyRegressor(strategy="mean").fit(X_v[train_idx], y_v[train_idx])
            r2_b = r2_score(y_v[test_idx], dummy.predict(X_v[test_idx]))
            r2_raw.append(r2_model)
            r2_base.append(r2_b)

    r2_raw  = np.array(r2_raw)
    r2_base = np.array(r2_base)
    r2_norm = r2_raw - r2_base          # per-fold, no clamp
    sem = float(np.std(r2_norm) / math.sqrt(len(r2_norm)))
    return float(np.mean(r2_norm)), sem, float(np.mean(r2_raw)), float(np.mean(r2_base)), n_obs


def run_decoding_condition(X, feat_df, groups, features, label,
                           max_rows=MAX_ROWS_PER_CONDITION, n_cv_splits=N_CV_SPLITS):
    """Run decoding for one subset.  Returns a DataFrame of results."""
    # downsample
    N = len(X)
    if N > max_rows:
        rng = np.random.RandomState(42)
        idx = rng.choice(N, max_rows, replace=False)
        idx.sort()
        X, feat_df, groups = X[idx], feat_df.iloc[idx], groups[idx]

    records = []
    for feat in features:
        y = feat_df[feat].values.astype(float)
        r2_mean, r2_sem, r2_raw, r2_dummy, n_obs = decode_feature(
            X, y, groups, feat, n_cv_splits=n_cv_splits)
        records.append({
            "feature": feat, "condition": label,
            "r2_mean": r2_mean, "r2_sem": r2_sem,   # r2_mean = normalized (model − baseline)
            "r2_raw": r2_raw, "r2_dummy": r2_dummy,  # raw model R² and baseline R² (unclamped)
            "n_obs": n_obs,                          # finite rows used for the fit
        })
    return pd.DataFrame(records)


# ── plotting ──────────────────────────────────────────────────────────────────

# Condition display labels (LaTeX)
_COND_LABELS = {
    "food_far":   "Food out of range",
    "food_near":  "Food in range",
    "combined":   "All timesteps",
    "agent_near": "<10 cm",
    "agent_mid":  "10–100 cm",
    "agent_far":  ">100 cm",
}

# LaTeX labels for derived features not in features.py FEATURE_METADATA
_EXTRA_FEAT_LABELS = {
    "mormyromast_field_mag":     r"$|\vec{E}_\mathrm{M}|$",
    "ampullary_field_mag":       r"$|\vec{E}_\mathrm{A}|$",
    "mormyromast_field_mag_log": r"$\log|\vec{E}_\mathrm{M}|$",
    "ampullary_field_mag_log":   r"$\log|\vec{E}_\mathrm{A}|$",
    "knollen_error_angle":       r"$\alpha^K_\mathrm{agent}$",
}

# Colors: seaborn default palette — blue then orange, matching reference figures
_BAR_COLORS = sns.color_palette()   # 0 = steel blue, 1 = orange


def _feat_label(feature: str) -> str:
    """Return LaTeX short name for a feature, falling back to EXTRA_FEAT_LABELS or raw name."""
    label = FEATURE_METADATA.get(feature, {}).get("short_name")
    if label is None:
        label = _EXTRA_FEAT_LABELS.get(feature, feature.replace("_", " "))
    return label


def plot_decoding(results_list: list[pd.DataFrame], outfile_base: str, show_n_obs: bool = False,
                  keep_negative: bool = False):
    """Vertical grouped bar chart: features (x) vs R² normalized (y), sorted descending.

    keep_negative=True retains features whose best R² is ≤ 0 (the old homing chart
    kept e.g. ampullary_error_angle at −0.3); the foraging plots drop them.
    """
    combined = pd.concat(results_list, ignore_index=True)
    combined = combined.dropna(subset=["r2_mean"])

    # B2: drop trivial/proprioceptive features from the chart (still in the CSVs)
    combined = combined[~combined["feature"].isin(PLOT_EXCLUDE_FEATURES)]

    # Filter: only features where at least one condition has positive R²
    # (matches old code's perf_threshold=0.0 filter applied before sorting)
    feat_max = combined.groupby("feature")["r2_mean"].max()
    positive_feats = feat_max.index if keep_negative else feat_max[feat_max > 0].index

    # Sort by MAX R² across conditions, descending (matches plot_grouped_agent_decoding)
    rank = (feat_max.loc[positive_feats]
            .sort_values(ascending=False)
            .head(TOP_N).index.tolist())
    combined = combined[combined["feature"].isin(rank)]

    conditions = list(combined["condition"].unique())
    n_conds = len(conditions)
    n_feats = len(rank)

    bar_w = min(0.35, 0.8 / n_conds)   # shrink bars so 3+ conditions fit per group
    offsets = (np.arange(n_conds) - (n_conds - 1) / 2) * bar_w
    fig_w = max(2.8, n_feats * (n_conds * bar_w + 0.16) + 0.64)

    set_style()
    fig, ax = panel(4.0, 2.0)

    x = np.arange(n_feats)
    for ci, cond in enumerate(conditions):
        sub = combined[combined["condition"] == cond].set_index("feature").reindex(rank)
        r2  = sub["r2_mean"].fillna(0).values
        err = sub["r2_sem"].fillna(0).values
        label = _COND_LABELS.get(cond, cond)
        ax.bar(x + offsets[ci], r2, width=bar_w * 1.0,
               yerr=err, color=_BAR_COLORS[ci], alpha=0.85, label=label,
               error_kw=dict(elinewidth=1.0, capsize=3, capthick=0.8, ecolor="black"))

    # Optionally annotate each tick label with the per-feature observation count
    # (max across plotted conditions).  Off by default; needs the n_obs column.
    if show_n_obs and "n_obs" in combined.columns:
        n_obs_by_feat = combined.groupby("feature")["n_obs"].max()

        def _fmt_n(v):
            return f"{int(v):,}" if pd.notna(v) else "?"
        tick_labels = [f"{_feat_label(f)}\n$n$={_fmt_n(n_obs_by_feat.get(f))}" for f in rank]
    else:
        tick_labels = [_feat_label(f) for f in rank]

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=7, rotation=30, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel(r"$R^2$ (normalized)")
    ax.set_xlabel("Feature")
    ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=0)

    if n_conds > 1:
        # Title the legend once for the conspecific-distance split rather than
        # repeating "Conspecific" on every entry.
        leg_title = "Conspecific dist." if all(c.startswith("agent_") for c in conditions) else None
        ax.legend(frameon=True, fontsize=6, loc="upper right", title=leg_title,
                  title_fontsize=6)

    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, outfile_base + ".pdf")


def plot_decoding_homing(results_list, outfile_base: str, keep_negative: bool = True):
    """Compact square horizontal-bar panel of normalized R² per feature.

    Matches the old utils_decoding homing aesthetic: a single (2.5×2.5)
    panel with features on the y-axis sorted descending (best at top), R² on
    the x-axis, one condition, single colour, vertical reference line at 0.
    Used by the homing report instead of the wide multi-condition
    ``plot_decoding`` (homing decodes only the pooled "combined" condition).
    """
    if isinstance(results_list, pd.DataFrame):
        results_list = [results_list]
    combined = pd.concat(results_list, ignore_index=True)
    combined = combined.dropna(subset=["r2_mean"])
    combined = combined[~combined["feature"].isin(PLOT_EXCLUDE_FEATURES)]

    feat_max = combined.groupby("feature")["r2_mean"].max()
    keep = feat_max.index if keep_negative else feat_max[feat_max > 0].index
    rank = feat_max.loc[keep].sort_values(ascending=False).head(TOP_N).index.tolist()
    if not rank:
        print("[homing] no features to plot for decoding")
        return

    sub = (combined[combined["feature"].isin(rank)]
           .groupby("feature").agg(r2_mean=("r2_mean", "mean"),
                                   r2_sem=("r2_sem", "mean"))
           .reindex(rank))

    set_style()
    fig, ax = panel(2.5, 2.5)
    y = np.arange(len(rank))[::-1]   # rank[0] (best) at the top
    ax.barh(y, sub["r2_mean"].values, xerr=sub["r2_sem"].fillna(0).values,
            color=_BAR_COLORS[0], alpha=0.85,
            error_kw=dict(elinewidth=1.0, capsize=2.5, capthick=0.8, ecolor="black"))
    ax.set_yticks(y)
    ax.set_yticklabels([_feat_label(f) for f in rank], fontsize=7)
    ax.set_xlabel(r"$R^2$ (normalized)")
    ax.set_ylabel("Feature")
    ax.axvline(0, color="black", lw=0.8, zorder=0)
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, outfile_base + ".pdf")


def plot_feature_distributions(feat_df: pd.DataFrame, features: list[str],
                               outfile_base: str):
    """Quick sanity-check histograms."""
    n = len(features)
    ncols = 4
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.44))
    axes = np.array(axes).flatten()
    for i, feat in enumerate(features):
        vals = feat_df[feat].dropna().values.astype(float)
        axes[i].hist(vals, bins=30, color=COND_PALETTE[0], alpha=0.7)
        axes[i].set_title(feat.replace("_", " "), fontsize=5)
        axes[i].tick_params(labelsize=5)
    for ax in axes[n:]:
        ax.set_visible(False)
    plt.tight_layout()
    save(fig, outfile_base + "_decoding_feature_dists.pdf")


# ── main ──────────────────────────────────────────────────────────────────────

def load(spec_dir):
    set_style()
    raw_dir     = os.path.join(spec_dir, "raw")
    derived_dir = os.path.join(spec_dir, "derived")
    out_dir     = os.path.join(spec_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "rnn_decoding")
    step_pkl = os.path.join(derived_dir, "per_env_ep_agent_step.pkl")
    if not os.path.exists(step_pkl):
        print(f"[rnn_decoding] missing {step_pkl}")
        return None
    dff = pd.read_pickle(step_pkl)
    morm_food_range_cm = get_morm_food_range_cm(spec_dir)
    dff, features = prepare_features(dff)
    print("[rnn_decoding] accumulating rnn states \u2026", flush=True)
    X_all, feat_df, groups_all, dist_food = accumulate(raw_dir, dff, features)
    if X_all is None:
        print("[rnn_decoding] no rnn data")
        return None
    results = []
    for label, mask in [("food_far",  dist_food > morm_food_range_cm),
                        ("food_near", dist_food <= morm_food_range_cm)]:
        if int(mask.sum()) < N_CV_SPLITS * 10:
            continue
        res = run_decoding_condition(X_all[mask], feat_df[mask].reset_index(drop=True),
                                    groups_all[mask], features, label)
        res.to_csv(f"{base}_decoding_{label}.csv", index=False)
        results.append(res)
    res_comb = run_decoding_condition(X_all, feat_df, groups_all, features, "combined")
    res_comb.to_csv(f"{base}_decoding_combined.csv", index=False)
    print("[rnn_decoding] decoding done", flush=True)
    return {
        "results": results, "res_comb": res_comb,
        "feat_df": feat_df, "features": features,
        "base": base, "out_dir": out_dir,
    }


def _load_cached_results(base):
    results = []
    for label in ["food_far", "food_near"]:
        path = f"{base}_decoding_{label}.csv"
        if os.path.exists(path):
            results.append(pd.read_csv(path))
    agent_results = []
    for label in ["agent_near", "agent_mid", "agent_far"]:
        path = f"{base}_decoding_{label}.csv"
        if os.path.exists(path):
            agent_results.append(pd.read_csv(path))
    combined_path = f"{base}_decoding_combined.csv"
    res_comb = pd.read_csv(combined_path) if os.path.exists(combined_path) else None
    if res_comb is None and not results and not agent_results:
        return None
    return results, agent_results, res_comb


def run(spec_dir, force_recompute=False, n_cv_splits=N_CV_SPLITS, max_rows=MAX_ROWS_PER_CONDITION,
        show_n_obs=False):
    set_style()
    raw_dir     = os.path.join(spec_dir, "raw")
    derived_dir = os.path.join(spec_dir, "derived")
    out_dir     = os.path.join(spec_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "rnn_decoding")

    cached = None if force_recompute else _load_cached_results(base)
    if cached is not None:
        print("[rnn_decoding] loading cached decoding CSVs", flush=True)
        results, agent_results, res_comb = cached
        if res_comb is not None:
            plot_decoding([res_comb], base + "_combined", show_n_obs=show_n_obs)
        if results:
            plot_decoding(results, base + "_split", show_n_obs=show_n_obs)
        if agent_results:
            plot_decoding(agent_results, base + "_split_agent", show_n_obs=show_n_obs)
        print("[rnn_decoding] done", flush=True)
        return

    step_pkl = os.path.join(derived_dir, "per_env_ep_agent_step.pkl")
    if not os.path.exists(step_pkl):
        print(f"[rnn_decoding] missing {step_pkl} — skipping", flush=True)
        return
    print(f"[rnn_decoding] loading {step_pkl}", flush=True)
    dff = pd.read_pickle(step_pkl)

    morm_food_range_cm = get_morm_food_range_cm(spec_dir)
    print(f"[rnn_decoding] morm_food_range_cm = {morm_food_range_cm}", flush=True)

    dff, features = prepare_features(dff)
    print(f"[rnn_decoding] {len(features)} target features: {features}", flush=True)

    print("[rnn_decoding] accumulating rnn states ...", flush=True)
    X_all, feat_df, groups_all, dist_food = accumulate(raw_dir, dff, features)
    if X_all is None:
        print("[rnn_decoding] no rnn data — skipping", flush=True)
        return
    print(f"[rnn_decoding] total rows: {len(X_all)}", flush=True)

    # Feature distribution sanity check
    plot_feature_distributions(feat_df, features, base)

    results = []
    for label, condition_mask in [
        ("food_far",  dist_food >  morm_food_range_cm),
        ("food_near", dist_food <= morm_food_range_cm),
    ]:
        n_cond = int(condition_mask.sum())
        print(f"[rnn_decoding] {label}: {n_cond} rows", flush=True)
        if n_cond < n_cv_splits * 10:
            print(f"  → too few rows for CV — skipping {label}", flush=True)
            continue
        X_sub      = X_all[condition_mask]
        feat_sub   = feat_df[condition_mask].reset_index(drop=True)
        groups_sub = groups_all[condition_mask]
        res = run_decoding_condition(X_sub, feat_sub, groups_sub, features, label,
                                     max_rows=max_rows, n_cv_splits=n_cv_splits)
        res.to_csv(f"{base}_decoding_{label}.csv", index=False)
        results.append(res)

    # Split by distance to nearest conspecific (<10 cm, 10–100 cm, >100 cm)
    agent_results = []
    if "distance_to_nearest_agent" in feat_df.columns:
        dist_agent = feat_df["distance_to_nearest_agent"].values.astype(float)
        lo, hi = AGENT_RANGE_BINS_CM
        for label, condition_mask in [
            ("agent_near", dist_agent < lo),
            ("agent_mid",  (dist_agent >= lo) & (dist_agent <= hi)),
            ("agent_far",  dist_agent > hi),
        ]:
            n_cond = int(condition_mask.sum())
            print(f"[rnn_decoding] {label}: {n_cond} rows", flush=True)
            if n_cond < n_cv_splits * 10:
                print(f"  → too few rows for CV — skipping {label}", flush=True)
                continue
            res = run_decoding_condition(
                X_all[condition_mask], feat_df[condition_mask].reset_index(drop=True),
                groups_all[condition_mask], features, label,
                max_rows=max_rows, n_cv_splits=n_cv_splits)
            res.to_csv(f"{base}_decoding_{label}.csv", index=False)
            agent_results.append(res)
    else:
        print("[rnn_decoding] distance_to_nearest_agent absent — skipping conspecific split",
              flush=True)

    # Combined (all timesteps pooled) — produces the single-bar Panel E-style plot
    print("[rnn_decoding] combined: running pooled decoding …", flush=True)
    res_comb = run_decoding_condition(X_all, feat_df, groups_all, features, "combined",
                                      max_rows=max_rows, n_cv_splits=n_cv_splits)
    res_comb.to_csv(f"{base}_decoding_combined.csv", index=False)

    if res_comb is not None:
        plot_decoding([res_comb], base + "_combined", show_n_obs=show_n_obs)

    if results:
        plot_decoding(results, base + "_split", show_n_obs=show_n_obs)
    else:
        print("[rnn_decoding] no conditions had enough data for food split plot", flush=True)

    if agent_results:
        plot_decoding(agent_results, base + "_split_agent", show_n_obs=show_n_obs)
    else:
        print("[rnn_decoding] no conditions had enough data for conspecific split plot",
              flush=True)
    print("[rnn_decoding] done", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec_dir")
    ap.add_argument("--force_recompute", action="store_true",
                    help="Ignore cached decoding CSVs and recompute from raw RNN arrays")
    ap.add_argument("--n-cv-splits", type=int, default=N_CV_SPLITS,
                    help=f"GroupKFold splits (default {N_CV_SPLITS}); r2_sem is SEM across these folds")
    ap.add_argument("--max-rows", type=int, default=MAX_ROWS_PER_CONDITION,
                    help=f"Per-condition downsample cap (default {MAX_ROWS_PER_CONDITION})")
    ap.add_argument("--show-n-obs", action="store_true",
                    help="Annotate each feature tick label with its fit observation count (default off)")
    ap.add_argument("--write-feature-table",
                    help="Write the supplemental LaTeX feature table to this path and exit")
    args = ap.parse_args(argv)
    if args.write_feature_table:
        write_feature_table_tex(args.write_feature_table)
        print(f"[rnn_decoding] wrote feature table: {args.write_feature_table}", flush=True)
        return
    if not args.spec_dir:
        ap.error("--spec_dir is required unless --write-feature-table is used")
    run(args.spec_dir, force_recompute=args.force_recompute,
        n_cv_splits=args.n_cv_splits, max_rows=args.max_rows,
        show_n_obs=args.show_n_obs)


if __name__ == "__main__":
    main()
