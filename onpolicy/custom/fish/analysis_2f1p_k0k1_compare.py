"""
Side-by-side comparison of 2f1p results with knollen on (k1) vs off (k0).

For each major plot type the k1 panel is shown on the left and the k0 panel
on the right within the same figure, sharing a y-axis where possible.

Outputs (under multi_eval/2f1p_k0k1/):
  b_eats_pct_compare.pdf
  b_food_by_condition_compare.pdf
  survival_b_compare.pdf
  survival_b_main_compare.pdf
  time_to_first_food_b_compare.pdf
  nn_distance_by_condition_compare.pdf
  peod_by_role_condition_compare.pdf

Usage:
    python analysis_2f1p_k0k1_compare.py --evals_dir <run_dir>/evals [--out_dir ...]
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from analysis_style import set_style, save, condition_plot, AGENT_COLORS, COND_PALETTE
from analysis_2f1p_multispec import (
    _make_cfg, _km_curve, _logrank_p, _bh_correct,
    load_agent, load_step, load_ep,
)
from cfg import PLOT_POINTS_BELOW_N

_K1_CFG = _make_cfg("2f1p")
_K0_CFG = _make_cfg("2f1p_k0")

_KNOLLEN_LABELS = {True: "k1 (knollen on)", False: "k0 (knollen off)"}
_KNOLLEN_ALPHA  = {True: 1.0,               False: 0.55}
_KNOLLEN_LS     = {True: "-",               False: "--"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_all(evals_dir, cfg):
    all_specs = cfg.main_conditions + [cfg.control_a, cfg.control_b]
    return (
        {s: load_agent(evals_dir, s, cfg=cfg) for s in all_specs},
        {s: load_step(evals_dir, s)           for s in all_specs},
        {s: load_ep(evals_dir, s, cfg=cfg)    for s in all_specs},
    )


def _panel_pair(sharey=True, w=0.84, h=2.0):
    """Return (fig, ax_k1, ax_k0) for a two-column comparison figure."""
    fig, axes = plt.subplots(1, 2, figsize=(w * 2 + 0.32, h),
                              sharey=sharey, gridspec_kw={"wspace": 0.08})
    return fig, axes[0], axes[1]


def _label(ax, text, fontsize=7):
    ax.set_title(text, fontsize=fontsize, pad=3)


# ── comparison plots ──────────────────────────────────────────────────────────

def plot_b_eats_pct_compare(k1_agent, k0_agent, out_dir):
    """% episodes B ate ≥1 food, k1 left / k0 right."""
    set_style()
    fig, ax1, ax0 = _panel_pair(sharey=True)
    for ax, agent_dfs, cfg, k1 in [(ax1, k1_agent, _K1_CFG, True),
                                    (ax0, k0_agent, _K0_CFG, False)]:
        conditions = cfg.main_conditions + [cfg.control_b]
        results = []
        for spec in conditions:
            df = agent_dfs.get(spec)
            if df is None:
                continue
            b = df[df["role"] == "B"]
            if len(b) == 0:
                continue
            pct = (b.groupby(["env_id", "episode_index"])["food_eaten"].sum() > 0).mean()
            n_ep = b[["env_id", "episode_index"]].drop_duplicates().shape[0]
            results.append({"spec": spec, "label": cfg.cond_labels[spec],
                             "pct": pct * 100, "n": n_ep})
        if not results:
            continue
        df_r = pd.DataFrame(results)
        colors = [cfg.cond_colors[r["spec"]] for r in results]
        ax.bar(range(len(df_r)), df_r["pct"], color=colors,
               alpha=_KNOLLEN_ALPHA[k1], width=0.6, edgecolor="white", linewidth=0.5)
        for i, row in df_r.iterrows():
            ax.text(i, row["pct"] + 1.5, f"N={int(row['n'])}", ha="center", fontsize=5)
        ax.set_xticks(range(len(df_r)))
        ax.set_xticklabels(df_r["label"], rotation=20, ha="right", fontsize=6)
        ax.set_ylim(0, 115)
        _label(ax, _KNOLLEN_LABELS[k1])
        sns.despine(ax=ax)
    ax1.set_ylabel("B eats (% of episodes)")
    ax0.set_ylabel("")
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "b_eats_pct_compare.pdf"))


def plot_b_food_compare(k1_agent, k0_agent, out_dir):
    """Food eaten by B, k1 left / k0 right."""
    set_style()
    fig, ax1, ax0 = _panel_pair(sharey=True)
    for ax, agent_dfs, cfg, k1 in [(ax1, k1_agent, _K1_CFG, True),
                                    (ax0, k0_agent, _K0_CFG, False)]:
        conditions = cfg.main_conditions + [cfg.control_b]
        rows = []
        for spec in conditions:
            df = agent_dfs.get(spec)
            if df is None:
                continue
            sub = df[df["role"] == "B"]
            ep = sub.groupby(["env_id", "episode_index"])["food_eaten"].sum().reset_index()
            ep["cond_label"] = cfg.cond_labels[spec]
            rows.append(ep)
        if not rows:
            continue
        long = pd.concat(rows, ignore_index=True)
        order = [cfg.cond_labels[s] for s in conditions if agent_dfs.get(s) is not None]
        palette = {cfg.cond_labels[s]: cfg.cond_colors[s] for s in conditions}
        condition_plot(ax, long, "cond_label", "food_eaten", palette,
                       order=order, rotation=20)
        _label(ax, _KNOLLEN_LABELS[k1])
    ax1.set_ylabel("Food eaten by B")
    ax0.set_ylabel("")
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "b_food_by_condition_compare.pdf"))


def plot_time_to_first_food_b_compare(k1_step, k0_step, out_dir):
    """Censored time-to-first-food for B, k1 left / k0 right."""
    set_style()
    fig, ax1, ax0 = _panel_pair(sharey=True)
    for ax, step_dfs, cfg, k1 in [(ax1, k1_step, _K1_CFG, True),
                                   (ax0, k0_step, _K0_CFG, False)]:
        conditions = cfg.main_conditions + [cfg.control_b]
        rows, max_steps = [], None
        for spec in conditions:
            df = step_dfs.get(spec)
            if df is None:
                continue
            if max_steps is None:
                max_steps = df["time_step"].max()
            b_eat = df[(df["role"] == "B") & (df["eating_event"].astype(bool))]
            first = b_eat.groupby(["env_id", "episode_index"])["time_step"].min()
            all_eps = df[df["role"] == "B"][["env_id", "episode_index"]].drop_duplicates()
            merged = all_eps.merge(first.rename("t_first").reset_index(),
                                   on=["env_id", "episode_index"], how="left")
            from analysis_style import TIME_STEP_MS
            merged["t_ms"] = merged["t_first"].fillna(max_steps) * TIME_STEP_MS
            merged["cond_label"] = cfg.cond_labels[spec]
            rows.append(merged[["cond_label", "t_ms"]])
        if not rows:
            continue
        long = pd.concat(rows, ignore_index=True)
        order = [cfg.cond_labels[s] for s in conditions if step_dfs.get(s) is not None]
        palette = {cfg.cond_labels[s]: cfg.cond_colors[s] for s in conditions}
        condition_plot(ax, long, "cond_label", "t_ms", palette, order=order, rotation=20)
        if max_steps is not None:
            from analysis_style import TIME_STEP_MS as _TSM
            ax.axhline(max_steps * _TSM, color="gray", lw=0.8, ls="--", alpha=0.6)
        _label(ax, _KNOLLEN_LABELS[k1])
        sns.despine(ax=ax)
    ax1.set_ylabel("Time to first food — B (ms)")
    ax0.set_ylabel("")
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "time_to_first_food_b_compare.pdf"))


def plot_survival_b_compare(k1_step, k0_step, out_dir, main_only=False):
    """KM survival curves for B, k1 and k0 overlaid on the same axes."""
    set_style()
    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    for step_dfs, cfg, k1 in [(k1_step, _K1_CFG, True), (k0_step, _K0_CFG, False)]:
        conditions = cfg.main_conditions if main_only else cfg.main_conditions + [cfg.control_b]
        avail = [s for s in conditions if step_dfs.get(s) is not None]
        for spec in avail:
            df = step_dfs[spec]
            max_steps = df["time_step"].max()
            b_eat = df[(df["role"] == "B") & (df["eating_event"].astype(bool))]
            first = b_eat.groupby(["env_id", "episode_index"])["time_step"].min()
            all_eps = df[df["role"] == "B"][["env_id", "episode_index"]].drop_duplicates()
            merged = all_eps.merge(first.rename("t_first").reset_index(),
                                   on=["env_id", "episode_index"], how="left")
            from analysis_style import TIME_STEP_MS
            times  = merged["t_first"].fillna(max_steps).values * TIME_STEP_MS
            events = merged["t_first"].notna().astype(int).values
            t_km, s_km, lo_km, hi_km = _km_curve(times, events)
            color = cfg.cond_colors[spec]
            ls = _KNOLLEN_LS[k1]
            alpha = _KNOLLEN_ALPHA[k1]
            label = f"{cfg.cond_labels[spec]} ({'k1' if k1 else 'k0'})"
            ax.plot(t_km, s_km, color=color, lw=1.3, ls=ls, alpha=alpha, label=label)
            ax.fill_between(t_km, lo_km, hi_km, color=color, alpha=0.1 * alpha)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("P(B has not eaten)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=5, frameon=False, loc="upper right", ncol=2)
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    suffix = "_main" if main_only else ""
    save(fig, os.path.join(out_dir, f"survival_b{suffix}_compare.pdf"))


def plot_nn_distance_compare(k1_ep, k0_ep, out_dir):
    """Mean NN distance, k1 left / k0 right."""
    set_style()
    fig, ax1, ax0 = _panel_pair(sharey=True)
    for ax, ep_dfs, cfg, k1 in [(ax1, k1_ep, _K1_CFG, True), (ax0, k0_ep, _K0_CFG, False)]:
        rows = []
        for spec in cfg.main_conditions:
            df = ep_dfs.get(spec)
            if df is None or "mean_nn_distance_cm" not in df.columns:
                continue
            tmp = df[["mean_nn_distance_cm"]].copy()
            tmp["cond_label"] = cfg.cond_labels[spec]
            rows.append(tmp)
        if not rows:
            continue
        long = pd.concat(rows, ignore_index=True)
        order = [cfg.cond_labels[s] for s in cfg.main_conditions if ep_dfs.get(s) is not None]
        palette = {cfg.cond_labels[s]: cfg.cond_colors[s] for s in cfg.main_conditions}
        condition_plot(ax, long, "cond_label", "mean_nn_distance_cm", palette, order=order)
        _label(ax, _KNOLLEN_LABELS[k1])
        sns.despine(ax=ax)
    ax1.set_ylabel("Mean NN distance (cm)")
    ax0.set_ylabel("")
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "nn_distance_by_condition_compare.pdf"))


def plot_peod_compare(k1_agent, k0_agent, out_dir):
    """P(emit EOD) per role, k1 left / k0 right — one subplot per condition."""
    set_style()
    _ROLE_COLORS = {"A": AGENT_COLORS[0], "B": AGENT_COLORS[1]}
    role_order = ["A", "B"]
    for cfg, agent_dfs, k1 in [(_K1_CFG, k1_agent, True), (_K0_CFG, k0_agent, False)]:
        conditions = [s for s in cfg.main_conditions if agent_dfs.get(s) is not None]
        n = len(conditions)
        if n == 0:
            continue
        fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 2.0), sharey=True)
        if n == 1:
            axes = [axes]
        for ax, spec in zip(axes, conditions):
            df = agent_dfs[spec]
            df_long = df[["role", "p_emit_eod"]].copy()
            n_per_role = df_long.groupby("role")["p_emit_eod"].count()
            n_map = {r: int(n_per_role.get(r, 0)) for r in role_order}
            sns.boxplot(data=df_long, x="role", y="p_emit_eod", order=role_order,
                        palette={"A": _ROLE_COLORS["A"], "B": _ROLE_COLORS["B"]},
                        ax=ax, width=0.55, linewidth=0.8, fliersize=0,
                        boxprops=dict(alpha=0.75))
            if max(n_map.values(), default=0) < PLOT_POINTS_BELOW_N:
                sns.stripplot(data=df_long, x="role", y="p_emit_eod", order=role_order,
                              color="black", alpha=0.5, jitter=0.2, size=3, ax=ax)
            ax.set_xticks(range(len(role_order)))
            ax.set_xticklabels([f"{r}\n(N={n_map[r]})" for r in role_order])
            ax.set_title(cfg.cond_labels[spec], fontsize=7)
            ax.set_xlabel("")
            sns.despine(ax=ax)
        axes[0].set_ylabel("P(emit EOD)")
        fig.suptitle(_KNOLLEN_LABELS[k1], fontsize=8, y=1.02)
        plt.tight_layout(pad=0.3)
        tag = "k1" if k1 else "k0"
        save(fig, os.path.join(out_dir, f"peod_by_role_condition_{tag}_compare.pdf"))


# ── main ──────────────────────────────────────────────────────────────────────

def run(evals_dir, out_dir=None):
    if out_dir is None:
        out_dir = os.path.join(evals_dir, "..", "multi_eval", "2f1p_k0k1")
    out_dir = os.path.normpath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    k1_agent, k1_step, k1_ep = _load_all(evals_dir, _K1_CFG)
    k0_agent, k0_step, k0_ep = _load_all(evals_dir, _K0_CFG)

    k1_found = [s for s in _K1_CFG.main_conditions if k1_agent.get(s) is not None]
    k0_found = [s for s in _K0_CFG.main_conditions if k0_agent.get(s) is not None]
    print(f"k1 specs found: {k1_found}")
    print(f"k0 specs found: {k0_found}")
    if not k1_found and not k0_found:
        sys.exit("No 2f1p k0/k1 data found.")

    plot_b_eats_pct_compare(k1_agent, k0_agent, out_dir)
    plot_b_food_compare(k1_agent, k0_agent, out_dir)
    plot_time_to_first_food_b_compare(k1_step, k0_step, out_dir)
    plot_survival_b_compare(k1_step, k0_step, out_dir, main_only=False)
    plot_survival_b_compare(k1_step, k0_step, out_dir, main_only=True)
    plot_nn_distance_compare(k1_ep, k0_ep, out_dir)
    plot_peod_compare(k1_agent, k0_agent, out_dir)
    print("Done.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args(argv)
    run(args.evals_dir, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
