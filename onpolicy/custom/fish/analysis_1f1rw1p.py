"""
One-fish / one-random-walker / one-patch (1f1rw1p) robotic-waggle sweep.

36 configs: size_mode ∈ {AltB, AeqB, AgtB}  ×  eod_rate ∈ {0, .2, .4, .6, .8, 1}
            ×  freeze ∈ {moving, frozen}

Agent convention:
  agent_id=0 → "bot"  (BoundedRandomWalkerEFishAgent — scripted)
  agent_id=1 → "RL"   (EFishAgent — policy-controlled)

Usage:
    python analysis_1f1rw1p.py --spec_dir <run_dir>/evals/1rw1f1p_grid

Outputs to {spec_dir}/analyses/1f1rw1p/
"""

import argparse
import glob
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

sys.path.insert(0, os.path.dirname(__file__))
import pickle

from analysis_style import (
    set_style, panel, save, AGENT_COLORS,
    save_event_time_plot_set,
)

OUT_SUBDIR = "analyses/1f1rw1p"

SIZE_MODES = ["AltB", "AeqB", "AgtB"]
EOD_RATES  = [0.0, 0.25, 0.5, 0.75, 1.0]
SIZE_LABELS = {"AltB": "B>A", "AeqB": "A=B", "AgtB": "A>B"}
SIZE_COLORS = {"AltB": "#EE6677", "AeqB": "#4477AA", "AgtB": "#228833"}


def build_grid(spec_dir):
    """Load 1rw1f1p_grid derived data, joining episode-level bot params from JSON."""
    derived_path = os.path.join(spec_dir, "derived", "per_env_ep_agent.pkl")
    raw_dir = os.path.join(spec_dir, "raw")

    if not os.path.exists(derived_path):
        print(f"  Missing: {derived_path}")
        return None

    df = pd.read_pickle(derived_path)

    # Load episodes JSON and join episode-level bot params
    ep_rows = []
    for ep_file in sorted(glob.glob(os.path.join(raw_dir, "*_episodes.json"))):
        with open(ep_file) as f:
            ep_rows.extend(json.load(f))

    _rw_want = ["env_id", "episode_index", "rw_eod_A", "rw_freeze_A", "rw_size_A", "rw_size_B",
                "rw_trial_id", "rw_B_x", "rw_B_y"]
    _rw_already = [c for c in _rw_want if c not in ("env_id", "episode_index") and c in df.columns]
    if ep_rows and len(_rw_already) < len(_rw_want) - 2:
        # Only merge from episodes.json if rw_* columns are not already in df
        ep_df = pd.DataFrame(ep_rows)
        rw_cols = [c for c in _rw_want if c in ep_df.columns]
        ep_df = ep_df[rw_cols].copy()
        ep_df["episode_index"] = pd.to_numeric(ep_df["episode_index"])
        ep_df["env_id"] = pd.to_numeric(ep_df["env_id"])
        # Drop any rw_* already in df to avoid column-suffix collision on merge
        df = df.drop(columns=[c for c in rw_cols if c in df.columns and c not in ("env_id", "episode_index")])
        df = df.merge(ep_df, on=["env_id", "episode_index"], how="left")
    elif not ep_rows and not _rw_already:
        print("  Warning: no episodes JSON found — size_mode/eod_rate/freeze will be NaN")

    # Derive categorical size_mode from bot vs RL fish size (bot = agent_id=0)
    df["size_mode"] = np.select(
        [df["rw_size_A"] < df["rw_size_B"], df["rw_size_A"] == df["rw_size_B"]],
        ["AltB", "AeqB"],
        default="AgtB",
    )
    df["eod_rate"] = df["rw_eod_A"]
    df["freeze"] = df["rw_freeze_A"].astype(bool)
    df["role"] = df["agent_id"].map({0: "bot", 1: "RL"})
    return df


def b_eats_pct(df_a):
    """Fraction of episodes where the RL fish ate ≥1 food."""
    a = df_a[df_a["role"] == "RL"]
    ep_food = a.groupby(["size_mode", "eod_rate", "freeze",
                          "env_id", "episode_index"])["food_eaten"].sum()
    pct = (ep_food > 0).groupby(["size_mode", "eod_rate", "freeze"]).mean()
    return pct.reset_index().rename(columns={"food_eaten": "pct"})


def _ep_counts(df, groupby_cols):
    """Count unique RL-fish (env_id, episode_index) pairs per group."""
    a = df[df["role"] == "RL"]
    return (a.groupby(groupby_cols)
             .apply(lambda g: g[["env_id", "episode_index"]].drop_duplicates().shape[0],
                    include_groups=False)
             .rename("n")
             .reset_index())


# ── panels ────────────────────────────────────────────────────────────────────

def plot_heatmap(df_all, out_dir):
    """Heatmap: B-eats% vs (size_mode row × eod_rate col), moving bots only."""
    set_style()
    df = df_all[~df_all["freeze"]]
    if len(df) == 0:
        print("  heatmap: no moving-bot data — skipped")
        return

    stats = b_eats_pct(df)
    stats = stats[~stats["freeze"]]  # moving only

    pivot = stats.pivot(index="size_mode", columns="eod_rate", values="pct")
    pivot = pivot.reindex(index=SIZE_MODES, columns=EOD_RATES) * 100

    nc = _ep_counts(df, ["size_mode", "eod_rate"])
    n_pivot = nc.pivot(index="size_mode", columns="eod_rate", values="n")
    n_pivot = n_pivot.reindex(index=SIZE_MODES, columns=EOD_RATES)

    annot = np.empty(pivot.shape, dtype=object)
    for i, sm in enumerate(pivot.index):
        for j, er in enumerate(pivot.columns):
            v = pivot.iloc[i, j]
            n_val = n_pivot.iloc[i, j]
            n = int(n_val) if not pd.isna(n_val) else "?"
            annot[i, j] = f"{v:.0f}%\nn={n}" if not pd.isna(v) else ""

    fig, ax = plt.subplots(figsize=(4.0, 2.2))
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", vmin=0, vmax=100,
                annot=annot, fmt="", annot_kws={"size": 6},
                linewidths=0.3, cbar_kws={"label": "B eats (%)", "shrink": 0.8})
    ax.set_yticklabels([SIZE_LABELS[s] for s in pivot.index], rotation=0)
    ax.set_xlabel("Bot EOD rate")
    ax.set_ylabel("Size condition")
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "b_eats_pct_heatmap.pdf"))


def plot_lines(df_all, out_dir):
    """B-eats% vs bot EOD rate, line per size_mode (moving bots only)."""
    set_style()
    df = df_all[~df_all["freeze"]]
    if len(df) == 0:
        return

    stats = b_eats_pct(df)
    stats = stats[~stats["freeze"]]
    nc = _ep_counts(df, ["size_mode", "eod_rate"])

    fig, ax = panel()
    for size_mode in SIZE_MODES:
        sub = stats[stats["size_mode"] == size_mode].sort_values("eod_rate")
        if len(sub) == 0:
            continue
        n_vals = nc[nc["size_mode"] == size_mode]["n"]
        n_str = f"N={int(n_vals.iloc[0])}/pt" if len(n_vals) > 0 else ""
        ax.plot(sub["eod_rate"], sub["pct"] * 100,
                color=SIZE_COLORS[size_mode], lw=1.5, marker="o", ms=4,
                label=f"{SIZE_LABELS[size_mode]} ({n_str})")
    ax.set_xlabel("Bot EOD rate")
    ax.set_ylabel("B eats (% of episodes)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=7, handlelength=1.2)
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "b_eats_pct_lines.pdf"))


def plot_frozen_vs_moving(df_all, out_dir):
    """B-eats%: frozen bot vs moving bot, per size_mode (eod_rate=1.0)."""
    set_style()
    stats = b_eats_pct(df_all)
    # use eod_rate=1.0 as the most informative comparison
    sub = stats[stats["eod_rate"] == 1.0]
    if len(sub) == 0:
        # fallback: use whatever eod_rate has both freeze values
        sub = stats

    nc_src = df_all[df_all["eod_rate"] == 1.0] if (df_all["eod_rate"] == 1.0).any() else df_all
    nc = _ep_counts(nc_src, ["size_mode", "freeze"])

    n_sm = len(SIZE_MODES)
    fig, ax = panel(w=2.0 * max(1, n_sm / 2), h=2.0)
    x = np.arange(n_sm)
    width = 0.35
    for j, (freeze, label, alpha) in enumerate([(False, "Moving", 0.9), (True, "Frozen", 0.5)]):
        pcts = []
        ns = []
        for sm in SIZE_MODES:
            v = sub[(sub["size_mode"] == sm) & (sub["freeze"] == freeze)]["pct"].values
            pcts.append(v[0] * 100 if len(v) else np.nan)
            n_row = nc[(nc["size_mode"] == sm) & (nc["freeze"] == freeze)]["n"].values
            ns.append(int(n_row[0]) if len(n_row) else "")
        bars = ax.bar(x + (j - 0.5) * width, pcts,
                      width=width, label=label, alpha=alpha,
                      color=[SIZE_COLORS[sm] for sm in SIZE_MODES])
        for bar, n in zip(bars, ns):
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                        f"N={n}", ha="center", va="bottom", fontsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels([SIZE_LABELS[sm] for sm in SIZE_MODES])
    ax.set_ylabel("B eats (% of episodes)")
    ax.set_ylim(0, 110)
    ax.legend(frameon=False, fontsize=7)
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "frozen_vs_moving.pdf"))


def plot_metric_vs_eod_rate(df_all, col, ylabel, fname, out_dir):
    """RL-fish metric vs bot EOD rate, line per size_mode (moving bots)."""
    set_style()
    df = df_all[~df_all["freeze"] & (df_all["role"] == "RL")]
    if len(df) == 0:
        return

    grp = df.groupby(["size_mode", "eod_rate"])[col].mean().reset_index()
    nc = _ep_counts(df_all[~df_all["freeze"]], ["size_mode", "eod_rate"])
    fig, ax = panel()
    for size_mode in SIZE_MODES:
        sub = grp[grp["size_mode"] == size_mode].sort_values("eod_rate")
        if len(sub) == 0:
            continue
        n_vals = nc[nc["size_mode"] == size_mode]["n"]
        n_str = f"N={int(n_vals.iloc[0])}/pt" if len(n_vals) > 0 else ""
        ax.plot(sub["eod_rate"], sub[col],
                color=SIZE_COLORS[size_mode], lw=1.5, marker="o", ms=4,
                label=f"{SIZE_LABELS[size_mode]} ({n_str})")
    ax.set_xlabel("Bot EOD rate")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=7, handlelength=1.2)
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, fname))


def _load_event_times_by_size_mode(spec_dir, freeze_filter=0):
    """Load event times grouped by size_mode from event_times.pkl + per_env_ep.pkl.

    Returns (ct_grouped, bt_grouped) where each is (condition_times, palette),
    or (None, None) if data is missing or event_times.pkl lacks episode keys.
    freeze_filter=0 keeps moving-bot episodes; 1 keeps frozen-bot episodes.
    """
    et_path = os.path.join(spec_dir, "derived", "event_times.pkl")
    ep_path = os.path.join(spec_dir, "derived", "per_env_ep.pkl")
    if not os.path.exists(et_path) or not os.path.exists(ep_path):
        return None, None

    with open(et_path, "rb") as f:
        et = pickle.load(f)
    if "consumption_ep_keys" not in et:
        return None, None  # old format without episode keys

    ep_df = pd.read_pickle(ep_path)
    if not {"rw_size_A", "rw_size_B", "rw_freeze_A"}.issubset(ep_df.columns):
        return None, None

    ep_df["size_mode"] = np.select(
        [ep_df["rw_size_A"] < ep_df["rw_size_B"],
         ep_df["rw_size_A"] == ep_df["rw_size_B"]],
        ["AltB", "AeqB"], default="AgtB",
    )
    ep_idx = ep_df.set_index(["env_id", "episode_index"])

    def _group(arrays, keys):
        by_sm = {sm: [] for sm in SIZE_MODES}
        for arr, key in zip(arrays, keys):
            try:
                row = ep_idx.loc[key]
            except KeyError:
                continue
            if row["rw_freeze_A"] != freeze_filter:
                continue
            sm = row["size_mode"]
            if sm in by_sm:
                by_sm[sm].append(arr)
        condition_times, palette = [], {}
        for sm in SIZE_MODES:
            if by_sm[sm]:
                label = SIZE_LABELS[sm]
                condition_times.append((label, by_sm[sm]))
                palette[label] = SIZE_COLORS[sm]
        return condition_times, palette

    return (
        _group(et["consumption_times"], et["consumption_ep_keys"]),
        _group(et["biting_times"],      et["biting_ep_keys"]),
    )


def plot_grouped_by_size_all_rates(df_all, out_dir):
    """Grouped bar: x=size_mode, bars=Moving/Frozen (B-eats%), averaged over all EOD rates.
    Matches old generate_1rw_grouped_comparison_plots.py plot_grouped_by_size(aggregated=True)."""
    set_style()
    stats = b_eats_pct(df_all)

    pivot = (
        stats
        .groupby(["size_mode", "freeze"])["pct"]
        .mean()
        .unstack()
    )
    pivot = pivot.reindex(index=SIZE_MODES) * 100
    pivot.index = [SIZE_LABELS[s] for s in pivot.index]
    pivot.rename(columns={False: "Moving", True: "Frozen"}, inplace=True)
    if "Moving" in pivot.columns and "Frozen" in pivot.columns:
        pivot = pivot[["Moving", "Frozen"]]

    nc = _ep_counts(df_all, ["size_mode", "freeze"])
    n_each = int(nc["n"].iloc[0]) if len(nc) > 0 else "?"

    fig, ax = panel()
    pivot.plot(kind="bar", ax=ax, rot=0, color=["#4477AA", "#AACCEE"])
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="none"))
    labels.append(f"N={n_each} each")
    leg = ax.legend(handles=handles, labels=labels, loc="lower right", frameon=True,
                    facecolor="white", framealpha=0.92, edgecolor="none", fontsize=7)
    leg.get_frame().set_linewidth(0)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_ylabel("B eats (% episodes)")
    ax.set_xlabel("Size condition")
    ax.set_ylim(0, 100)
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "grouped_by_size_all_rates.pdf"))


def plot_grouped_by_rate(df_all, out_dir, freeze_filter=None, suffix="aggregated"):
    """Grouped bar: x=eod_rate, bars=size_mode (B-eats%), optionally filtered by freeze state.
    Matches old generate_1rw_grouped_comparison_plots.py plot_grouped_by_rate."""
    set_style()
    stats = b_eats_pct(df_all)

    nc_cols = ["eod_rate", "size_mode", "freeze"] if freeze_filter is not None else ["eod_rate", "size_mode"]
    if freeze_filter is not None:
        stats = stats[stats["freeze"] == freeze_filter]

    nc_full = _ep_counts(df_all, ["eod_rate", "size_mode", "freeze"])
    if freeze_filter is not None:
        nc = nc_full[nc_full["freeze"] == freeze_filter].groupby(["eod_rate", "size_mode"])["n"].sum().reset_index()
    else:
        nc = nc_full.groupby(["eod_rate", "size_mode"])["n"].sum().reset_index()

    pivot = (
        stats
        .groupby(["eod_rate", "size_mode"])["pct"]
        .mean()
        .unstack()
        .reindex(columns=SIZE_MODES) * 100
    )
    pivot.rename(columns=SIZE_LABELS, inplace=True)

    fig, ax = panel()
    pivot.plot(kind="bar", ax=ax, rot=45,
               color=[SIZE_COLORS[s] for s in SIZE_MODES])
    if freeze_filter is None:
        # uniform N across all bars — show once in legend
        n_each = int(nc["n"].iloc[0]) if len(nc) > 0 else "?"
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([], [], color="none"))
        labels.append(f"N={n_each} each")
        leg = ax.legend(handles=handles, labels=labels, loc="lower right",
                        frameon=True, facecolor="white", framealpha=0.92,
                        edgecolor="none", fontsize=6, handlelength=1.1,
                        handletextpad=0.35, borderpad=0.25,
                        labelspacing=0.25)
        leg.get_frame().set_linewidth(0)
    else:
        ax.get_legend().set_title(None)
        for container, sm in zip(ax.containers, SIZE_MODES):
            for bar, eod_rate in zip(container, pivot.index):
                n_row = nc[(nc["eod_rate"] == eod_rate) & (nc["size_mode"] == sm)]["n"].values
                n = int(n_row[0]) if len(n_row) else "?"
                h = bar.get_height()
                if not np.isnan(h):
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                            f"N={n}", ha="center", va="bottom", fontsize=4, rotation=90)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_ylabel("B eats (% episodes)")
    ax.set_xlabel("Bot EOD rate")
    ax.set_ylim(0, 100 if freeze_filter is None else 115)
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, f"grouped_by_rate_{suffix}.pdf"))


def plot_time_to_consumption_by_size_mode(spec_dir, out_dir):
    """Time-to-nth-food curves, one line per size_mode (moving bots only)."""
    (ct, palette), _ = _load_event_times_by_size_mode(spec_dir)
    if ct is None:
        return
    save_event_time_plot_set(ct, palette, out_dir, "time_to_consumption")


def plot_time_to_bitten_by_size_mode(spec_dir, out_dir):
    """Time-to-nth-biting-event curves, one line per size_mode (moving bots only)."""
    _, (ct, palette) = _load_event_times_by_size_mode(spec_dir)
    if ct is None:
        return
    save_event_time_plot_set(ct, palette, out_dir, "time_to_bitten",
                              event_label="Biting events",
                              time_label="Time to biting event (s)")


def _load_initial_positions_from_ep(spec_dir):
    """Load RL fish B initial positions from per_env_ep.pkl using rw_B_x/rw_B_y/rw_B_orientation.

    Orientation is taken from rw_B_orientation when present; otherwise computed
    from position (agent always faces center on reset).

    Returns DataFrame with columns:
        env_id, episode_index, position_x, position_y, orientation,
        dist_from_center, angle_from_center, size_mode, eod_rate, freeze
    or None if data is missing or rw_B_x/rw_B_y are absent.
    """
    ep_path = os.path.join(spec_dir, "derived", "per_env_ep.pkl")
    if not os.path.exists(ep_path):
        return None

    df = pd.read_pickle(ep_path)
    needed = {"rw_B_x", "rw_B_y", "rw_B_orientation", "rw_size_A", "rw_size_B", "rw_freeze_A", "rw_eod_A"}
    if not needed.issubset(df.columns):
        return None

    df = df.copy()
    df["size_mode"] = np.select(
        [df["rw_size_A"] < df["rw_size_B"],
         df["rw_size_A"] == df["rw_size_B"]],
        ["AltB", "AeqB"], default="AgtB",
    )
    df["eod_rate"]   = df["rw_eod_A"]
    df["freeze"]     = df["rw_freeze_A"].astype(bool)
    df["position_x"] = df["rw_B_x"]
    df["position_y"] = df["rw_B_y"]

    center = 75.0  # arena is 150×150 cm
    df["orientation"] = df["rw_B_orientation"]

    dx = df["position_x"] - center
    dy = df["position_y"] - center
    df["dist_from_center"]  = np.sqrt(dx**2 + dy**2)
    df["angle_from_center"] = np.arctan2(dy, dx)
    return df


def save_initial_positions_csv(spec_dir, out_dir):
    """Save RL fish initial positions to CSV and return the DataFrame."""
    df = _load_initial_positions_from_ep(spec_dir)
    if df is None:
        print("  initial_positions: per_env_ep.pkl missing or lacks rw_B_* columns — skipped")
        return None
    out_path = os.path.join(out_dir, "initial_positions_B.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows → {out_path}")
    return df


def plot_initial_positions(spec_dir, out_dir):
    """Scatter of RL fish B start positions in arena coords, coloured by size_mode."""
    df = _load_initial_positions_from_ep(spec_dir)
    if df is None:
        print("  initial_positions plot: per_env_ep.pkl missing or lacks rw_B_* columns — skipped")
        return
    _render_initial_positions(df, out_dir)


def _render_initial_positions(df, out_dir):
    """Render the RL fish B start-position scatter from a prepared DataFrame."""
    set_style()
    fig, ax = panel(2.0, 2.0)

    center = 75.0
    arena  = 150.0

    # Arena boundary
    rect = plt.Rectangle((0, 0), arena, arena, fill=False,
                          edgecolor="#aaaaaa", lw=0.8, zorder=0)
    ax.add_patch(rect)

    # Polar-grid radii (25 cm inner, 50 cm outer from center)
    for r, ls in [(25, ":"), (50, "--")]:
        circ = plt.Circle((center, center), r, fill=False,
                           edgecolor="#bbbbbb", lw=0.6, linestyle=ls, zorder=0)
        ax.add_patch(circ)

    # Bot at center, with heading arrow (+y / up; grid 0° is along this)
    bot_ori = np.pi / 2
    ax.plot(center, center, marker="x", color="black", ms=5, mew=1.2,
            zorder=3, label="Bot (center)")
    ax.annotate("", xy=(center + 8.0 * np.cos(bot_ori),
                        center + 8.0 * np.sin(bot_ori)),
                xytext=(center, center),
                arrowprops=dict(arrowstyle="-|>", color="black",
                                lw=1.0, mutation_scale=7),
                zorder=3)

    # RL fish scatter + orientation arrows
    arrow_len = 6.0
    for sm in SIZE_MODES:
        sub = df[df["size_mode"] == sm] if "size_mode" in df.columns else df
        label = SIZE_LABELS.get(sm, sm)
        color = SIZE_COLORS[sm]
        ax.scatter(sub["position_x"], sub["position_y"],
                   color=color, s=12, zorder=2, label=label, alpha=0.85)
        for _, row in sub.iterrows():
            ax.annotate("", xy=(row["position_x"] + arrow_len * np.cos(row["orientation"]),
                                row["position_y"] + arrow_len * np.sin(row["orientation"])),
                        xytext=(row["position_x"], row["position_y"]),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=0.6, mutation_scale=5),
                        zorder=2)

    ax.set_xlim(-2, arena + 2)
    ax.set_ylim(-2, arena + 2)
    ax.set_aspect("equal")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.legend(frameon=False, fontsize=6, handlelength=1.0, loc="upper left")
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "initial_positions_B.pdf"))


# ── main ─────────────────────────────────────────────────────────────────────

def load(spec_dir):
    out_dir = os.path.join(spec_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    print("Loading 1rw1f1p_grid \u2026")
    df_all = build_grid(spec_dir)
    if df_all is None:
        print("No 1rw1f1p_grid data found.")
        return None
    n_cond = df_all[["size_mode", "eod_rate", "freeze"]].drop_duplicates().shape[0]
    print(f"  {len(df_all):,} rows from {n_cond} conditions")
    return {"df_all": df_all, "out_dir": out_dir, "spec_dir": spec_dir}


def run(spec_dir, plot_only=False):
    out_dir = os.path.join(spec_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    print("Loading 1rw1f1p_grid …")
    df_all = build_grid(spec_dir)
    if df_all is None:
        print("No 1rw1f1p_grid data found — nothing to plot.")
        return
    n_cond = df_all[["size_mode", "eod_rate", "freeze"]].drop_duplicates().shape[0]
    print(f"  Loaded {len(df_all):,} agent-episode rows from {n_cond} unique conditions")

    plot_heatmap(df_all, out_dir)
    plot_lines(df_all, out_dir)
    plot_frozen_vs_moving(df_all, out_dir)
    plot_grouped_by_size_all_rates(df_all, out_dir)
    plot_grouped_by_rate(df_all, out_dir, freeze_filter=None, suffix="aggregated")
    plot_grouped_by_rate(df_all, out_dir, freeze_filter=True,  suffix="frozen")
    plot_metric_vs_eod_rate(df_all, "food_eaten",  "Food eaten (A)",       "food_vs_eod_rate.pdf",  out_dir)
    plot_metric_vs_eod_rate(df_all, "p_emit_eod",  "P(emit EOD) — A",      "peod_vs_bot_rate.pdf",  out_dir)
    plot_metric_vs_eod_rate(df_all, "distance_to_nearest_agent", "Mean NN dist (cm)", "nn_dist_vs_bot_rate.pdf", out_dir)
    plot_time_to_consumption_by_size_mode(spec_dir, out_dir)
    plot_time_to_bitten_by_size_mode(spec_dir, out_dir)
    save_initial_positions_csv(spec_dir, out_dir)
    plot_initial_positions(spec_dir, out_dir)
    print("Done.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec_dir", required=True)
    ap.add_argument("--plot_only", action="store_true")
    args = ap.parse_args(argv)
    run(args.spec_dir, plot_only=args.plot_only)


if __name__ == "__main__":
    main()
