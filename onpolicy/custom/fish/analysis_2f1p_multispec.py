"""
Two-fish one-patch (2f1p) dyadic assay analysis.

Convention: agent_id=0 → "A" (resident, starts at patch)
            agent_id=1 → "B" (challenger, starts away)
Conditions: 2f1p_AltB (B large > A), 2f1p_AeqB (equal), 2f1p_AgtB (A large > B)
Controls:   2f1p_control_a (A alone), 2f1p_control_b (B alone)

Usage:
    python analysis_2f1p_multispec.py \\
        --evals_dir <run_dir>/evals \\
        --out_dir   <run_dir>/multi_eval/2f1p

Outputs:
  b_eats_pct.png             — fraction of episodes B ate ≥1 food, by condition
  b_food_by_condition.png    — food eaten by B across conditions (boxplot+strip)
  a_food_social_vs_alone.png — A: AeqB vs control_a
  b_food_social_vs_alone.png — B: AeqB vs control_b
  time_to_first_food_b.pdf          — steps until B first eats (censored at max_steps if never)
  time_to_first_food_b_stats.pdf    — same + BH-corrected pairwise MWU brackets
  time_to_first_food_b_eaters.pdf   — same, restricted to episodes where B ate ≥1 food
  time_to_first_food_b_eaters_stats.pdf — same + BH-corrected pairwise MWU brackets
  survival_b.pdf                    — Kaplan-Meier curves + pairwise log-rank BH p-values
  nn_distance_by_condition.png — mean nearest-neighbour distance
  peod_by_role_condition.png  — P(emit EOD) for A and B, by condition
  dist_to_patch_center_by_condition.pdf — mean dist to food-patch centroid (roles A & B, by condition)
  food_vs_size.png            — food eaten vs agent size (open-circle regression)
  trajectories/{spec_key}/trajectory_env{e}_ep{i}.pdf — N sample trajectories per condition (no legend)
  trajectories/{spec_key}/trajectory_env{e}_ep{i}_legend.pdf — same with legend
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.legend_handler import HandlerBase
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from analysis_style import (
    set_style, panel, save, condition_plot,
    AGENT_COLORS, COND_PALETTE, TIME_STEP_MS,
    consumption_times_from_step_df, biting_times_from_step_df,
    save_event_time_plot_set,
)
from cfg import PLOT_POINTS_BELOW_N

MAIN_CONDITIONS = ["2f1p_AltB", "2f1p_AeqB", "2f1p_AgtB"]
CONTROL_A = "2f1p_control_a"
CONTROL_B = "2f1p_control_b"

COND_LABELS = {
    "2f1p_AltB":      "B>A",
    "2f1p_AeqB":      "A=B",
    "2f1p_AgtB":      "A>B",
    "2f1p_control_a": "A alone",
    "2f1p_control_b": "B alone",
}
# Per-condition colours (for bar charts and palette lists)
COND_COLORS = {
    "2f1p_AltB":      COND_PALETTE[1],   # red
    "2f1p_AeqB":      COND_PALETTE[0],   # blue
    "2f1p_AgtB":      COND_PALETTE[2],   # green
    "2f1p_control_a": COND_PALETTE[3],   # yellow
    "2f1p_control_b": COND_PALETTE[4],   # purple
}
_ROLE_COLORS = {"A": AGENT_COLORS[0], "B": AGENT_COLORS[1]}


@dataclass
class _Cfg:
    main_conditions: List[str]
    control_a: str
    control_b: str
    cond_labels: Dict[str, str]
    cond_colors: Dict[str, str]


def _make_cfg(spec_prefix: str = "2f1p") -> "_Cfg":
    p = spec_prefix
    main = [f"{p}_AltB", f"{p}_AeqB", f"{p}_AgtB"]
    ctrl_a, ctrl_b = f"{p}_control_a", f"{p}_control_b"
    labels = {
        f"{p}_AltB":      "B>A",
        f"{p}_AeqB":      "A=B",
        f"{p}_AgtB":      "A>B",
        f"{p}_control_a": "A alone",
        f"{p}_control_b": "B alone",
    }
    colors = {
        f"{p}_AltB":      COND_PALETTE[1],
        f"{p}_AeqB":      COND_PALETTE[0],
        f"{p}_AgtB":      COND_PALETTE[2],
        f"{p}_control_a": COND_PALETTE[3],
        f"{p}_control_b": COND_PALETTE[4],
    }
    return _Cfg(main, ctrl_a, ctrl_b, labels, colors)


_DEFAULT_CFG = _make_cfg("2f1p")

N_TRAJ_SAMPLES = 30


# ── loading ──────────────────────────────────────────────────────────────────

def load_agent(evals_dir, spec_key, cfg=None):
    p = os.path.join(evals_dir, spec_key, "derived", "per_env_ep_agent.pkl")
    if not os.path.exists(p):
        return None
    if cfg is None:
        cfg = _DEFAULT_CFG
    df = pd.read_pickle(p)
    df["condition"] = spec_key
    df["cond_label"] = cfg.cond_labels[spec_key]
    df["role"] = df["agent_id"].map({0: "A", 1: "B"})
    return df


def load_step(evals_dir, spec_key):
    p = os.path.join(evals_dir, spec_key, "derived", "per_env_ep_agent_step.pkl")
    if not os.path.exists(p):
        return None
    df = pd.read_pickle(p)
    df["condition"] = spec_key
    df["role"] = df["agent_id"].map({0: "A", 1: "B"})
    return df


def load_ep(evals_dir, spec_key, cfg=None):
    p = os.path.join(evals_dir, spec_key, "derived", "per_env_ep.pkl")
    if not os.path.exists(p):
        return None
    if cfg is None:
        cfg = _DEFAULT_CFG
    df = pd.read_pickle(p)
    df["condition"] = spec_key
    df["cond_label"] = cfg.cond_labels[spec_key]
    return df


def _long_format(agent_dfs, specs, role, col, cfg=None):
    """Build a long DataFrame of `col` for `role` across `specs`."""
    if cfg is None:
        cfg = _DEFAULT_CFG
    rows = []
    for spec in specs:
        df = agent_dfs.get(spec)
        if df is None:
            continue
        sub = df[df["role"] == role] if role else df
        ep = sub.groupby(["env_id", "episode_index"])[col].sum().reset_index()
        ep["cond_label"] = cfg.cond_labels[spec]
        rows.append(ep)
    return pd.concat(rows, ignore_index=True) if rows else None


# ── panels ────────────────────────────────────────────────────────────────────

def plot_b_eats_pct(agent_dfs, out_dir, cfg=None):
    """Fraction of episodes where B ate ≥1 food, per condition."""
    if cfg is None:
        cfg = _DEFAULT_CFG
    set_style()
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
        results.append({"spec": spec, "label": cfg.cond_labels[spec], "pct": pct * 100, "n": n_ep})

    if not results:
        return
    df_r = pd.DataFrame(results)
    fig, ax = panel()
    colors = [cfg.cond_colors[r["spec"]] for r in results]
    ax.bar(range(len(df_r)), df_r["pct"], color=colors, alpha=0.8, width=0.6,
           edgecolor="white", linewidth=0.5)
    for i, row in df_r.iterrows():
        ax.text(i, row["pct"] + 1.5, f"N={int(row['n'])}", ha="center", fontsize=6)
    ax.set_xticks(range(len(df_r)))
    ax.set_xticklabels(df_r["label"], rotation=20, ha="right")
    ax.set_ylabel("B eats (% of episodes)")
    ax.set_ylim(0, 115)
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "b_eats_pct.pdf"))


def plot_b_food_by_condition(agent_dfs, out_dir, cfg=None):
    """Food eaten by B per episode, boxplot + stripplot per condition."""
    if cfg is None:
        cfg = _DEFAULT_CFG
    set_style()
    conditions = cfg.main_conditions + [cfg.control_b]
    long = _long_format(agent_dfs, conditions, "B", "food_eaten", cfg=cfg)
    if long is None:
        return
    order = [cfg.cond_labels[s] for s in conditions if agent_dfs.get(s) is not None]
    palette = {cfg.cond_labels[s]: cfg.cond_colors[s] for s in conditions}
    fig, ax = panel()
    condition_plot(ax, long, "cond_label", "food_eaten", palette, order=order,
                   ylabel="Food eaten by B", rotation=20)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "b_food_by_condition.pdf"))


def _plot_social_vs_alone(ax, soc_vals, alone_vals, label_s, label_a, color):
    """Boxplot+strip for 2-condition social-vs-alone comparison. Returns (U, p) or (None, None)."""
    df_long = pd.DataFrame({
        "food": np.concatenate([soc_vals, alone_vals]),
        "condition": [label_s] * len(soc_vals) + [label_a] * len(alone_vals),
    })
    palette = {label_s: color, label_a: "#AAAAAA"}
    condition_plot(ax, df_long, "condition", "food", palette,
                   order=[label_s, label_a])
    if len(soc_vals) > 2 and len(alone_vals) > 2:
        U, p = stats.mannwhitneyu(soc_vals, alone_vals, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.set_title(sig, fontsize=8)
        return float(U), float(p)
    return None, None


def plot_social_vs_alone(agent_dfs, out_dir, cfg=None):
    """A: AeqB vs control_a; B: AeqB vs control_b."""
    if cfg is None:
        cfg = _DEFAULT_CFG
    set_style()
    aeqb = agent_dfs.get(f"{cfg.main_conditions[1]}")  # AeqB is index 1
    ctrl_a = agent_dfs.get(cfg.control_a)
    ctrl_b = agent_dfs.get(cfg.control_b)
    mwu_rows = []

    if aeqb is not None and ctrl_a is not None:
        a_soc = aeqb[aeqb["role"] == "A"].groupby(["env_id", "episode_index"])["food_eaten"].sum().values
        a_alone = ctrl_a.groupby(["env_id", "episode_index"])["food_eaten"].sum().values
        fig, ax = panel()
        U, p = _plot_social_vs_alone(ax, a_soc, a_alone, "A + B", "A alone", _ROLE_COLORS["A"])
        ax.set_ylabel("Food eaten by A")
        plt.tight_layout(pad=0.3)
        save(fig, os.path.join(out_dir, "a_food_social_vs_alone.pdf"))
        if p is not None:
            mwu_rows.append({"role": "A", "group_social": "A+B", "group_alone": "A alone",
                              "n_social": len(a_soc), "n_alone": len(a_alone), "U_stat": U, "p": p})

    if aeqb is not None and ctrl_b is not None:
        b_soc = aeqb[aeqb["role"] == "B"].groupby(["env_id", "episode_index"])["food_eaten"].sum().values
        b_alone = ctrl_b.groupby(["env_id", "episode_index"])["food_eaten"].sum().values
        fig, ax = panel()
        U, p = _plot_social_vs_alone(ax, b_soc, b_alone, "A + B", "B alone", _ROLE_COLORS["B"])
        ax.set_ylabel("Food eaten by B")
        plt.tight_layout(pad=0.3)
        save(fig, os.path.join(out_dir, "b_food_social_vs_alone.pdf"))
        if p is not None:
            mwu_rows.append({"role": "B", "group_social": "A+B", "group_alone": "B alone",
                              "n_social": len(b_soc), "n_alone": len(b_alone), "U_stat": U, "p": p})

    if mwu_rows:
        pd.DataFrame(mwu_rows).to_csv(
            os.path.join(out_dir, "social_vs_alone_mwu.csv"), index=False)


def plot_social_vs_alone_pct(agent_dfs, out_dir, cfg=None):
    """% of episodes where agent ate ≥1 food item: social (AeqB) vs alone.

    Bar chart with binomial SE error bars and Fisher's exact test.
    Outputs: a_food_pct_social_vs_alone.pdf, b_food_pct_social_vs_alone.pdf
    """
    if cfg is None:
        cfg = _DEFAULT_CFG
    set_style()
    aeqb   = agent_dfs.get(cfg.main_conditions[1])   # AeqB
    ctrl_a = agent_dfs.get(cfg.control_a)
    ctrl_b = agent_dfs.get(cfg.control_b)

    def _ate_indicator(df, role=None):
        """1 per episode if agent ate ≥1 food, else 0."""
        sub = df[df["role"] == role] if role else df
        per_ep = sub.groupby(["env_id", "episode_index"])["food_eaten"].sum()
        return (per_ep > 0).astype(int).values

    def _draw(ax, arr_s, arr_a, label_s, label_a, color):
        pct_s = arr_s.mean() * 100
        pct_a = arr_a.mean() * 100
        se_s  = np.sqrt(arr_s.mean() * (1 - arr_s.mean()) / len(arr_s)) * 100
        se_a  = np.sqrt(arr_a.mean() * (1 - arr_a.mean()) / len(arr_a)) * 100
        x = [0, 1]
        ax.bar(x, [pct_s, pct_a], color=[color, "#AAAAAA"],
               edgecolor="k", linewidth=0.5, width=0.5)
        ax.errorbar(x, [pct_s, pct_a], yerr=[se_s, se_a],
                    fmt="none", color="k", capsize=3, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([label_s, label_a], fontsize=7)
        ax.set_ylabel("Episodes with food eaten (%)")
        ax.set_ylim(0, 110)
        _, p = stats.fisher_exact(
            [[arr_s.sum(), len(arr_s) - arr_s.sum()],
             [arr_a.sum(), len(arr_a) - arr_a.sum()]]
        )
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        y_top = max(pct_s + se_s, pct_a + se_a)
        ax.text(0.5, y_top + 4, sig, ha="center", va="bottom", fontsize=8)
        return float(p)

    pct_rows = []
    if aeqb is not None and ctrl_a is not None:
        a_soc   = _ate_indicator(aeqb, role="A")
        a_alone = _ate_indicator(ctrl_a)
        fig, ax = panel()
        p = _draw(ax, a_soc, a_alone, "A + B", "A alone", _ROLE_COLORS["A"])
        sns.despine(ax=ax)
        plt.tight_layout(pad=0.3)
        save(fig, os.path.join(out_dir, "a_food_pct_social_vs_alone.pdf"))
        pct_rows.append({"role": "A", "n_social": len(a_soc), "n_alone": len(a_alone),
                          "pct_social": a_soc.mean() * 100, "pct_alone": a_alone.mean() * 100,
                          "p_fisher": p})

    if aeqb is not None and ctrl_b is not None:
        b_soc   = _ate_indicator(aeqb, role="B")
        b_alone = _ate_indicator(ctrl_b)
        fig, ax = panel()
        p = _draw(ax, b_soc, b_alone, "A + B", "B alone", _ROLE_COLORS["B"])
        sns.despine(ax=ax)
        plt.tight_layout(pad=0.3)
        save(fig, os.path.join(out_dir, "b_food_pct_social_vs_alone.pdf"))
        pct_rows.append({"role": "B", "n_social": len(b_soc), "n_alone": len(b_alone),
                          "pct_social": b_soc.mean() * 100, "pct_alone": b_alone.mean() * 100,
                          "p_fisher": p})

    if pct_rows:
        pd.DataFrame(pct_rows).to_csv(
            os.path.join(out_dir, "social_vs_alone_pct_fisher.csv"), index=False)


def plot_time_to_first_food_b(step_dfs, out_dir, cfg=None):
    """
    Steps until B's first eating event per episode, by condition.
    Episodes where B never ate are censored at max_steps.
    """
    if cfg is None:
        cfg = _DEFAULT_CFG
    set_style()
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
        merged["t_ms"] = merged["t_first"].fillna(max_steps) * TIME_STEP_MS
        merged["cond_label"] = cfg.cond_labels[spec]
        rows.append(merged[["cond_label", "t_ms"]])

    if not rows:
        return
    long = pd.concat(rows, ignore_index=True)
    order = [cfg.cond_labels[s] for s in conditions if step_dfs.get(s) is not None]
    palette = {cfg.cond_labels[s]: cfg.cond_colors[s] for s in conditions}
    fig, ax = panel()
    condition_plot(ax, long, "cond_label", "t_ms", palette, order=order,
                   ylabel="Time to first food — B (ms)", rotation=20)
    if max_steps is not None:
        ax.axhline(max_steps * TIME_STEP_MS, color="gray", lw=0.8, ls="--", alpha=0.6)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "time_to_first_food_b.pdf"))


def plot_time_to_first_food_b_eaters(step_dfs, out_dir, cfg=None):
    """
    Steps until B's first eating event, restricted to episodes where B ate ≥1 food.
    Episodes where B never ate are excluded entirely (not censored).

    Statistical note
    ----------------
    Excluding non-eaters conditions this plot on B having found food at all, which
    introduces selection bias: if P(B eats) varies across conditions (e.g. lower when
    A is large), the surviving episodes are not a random sample — AgtB episodes where
    B does eat may represent unusually favourable starting states.  Cross-condition
    comparisons of latency among eaters are therefore confounded with that selection.

    If pairwise tests are run across the 4 conditions (3 main + control_b) there are
    C(4,2)=6 comparisons; Bonferroni or Benjamini-Hochberg FDR correction is needed.
    The uncensored version (time_to_first_food_b.png) has the same multiplicity issue.

    Preferred alternative: a Kaplan-Meier survival curve with log-rank test handles
    censoring correctly and avoids the selection bias entirely.
    """
    if cfg is None:
        cfg = _DEFAULT_CFG
    set_style()
    conditions = cfg.main_conditions + [cfg.control_b]
    rows = []
    n_eaters = {}
    for spec in conditions:
        df = step_dfs.get(spec)
        if df is None:
            continue
        b_eat = df[(df["role"] == "B") & (df["eating_event"].astype(bool))]
        first = b_eat.groupby(["env_id", "episode_index"])["time_step"].min()
        eater_eps = first.reset_index().rename(columns={"time_step": "t_first"})
        eater_eps["t_ms"] = eater_eps["t_first"] * TIME_STEP_MS
        eater_eps["cond_label"] = cfg.cond_labels[spec]
        n_eaters[spec] = len(eater_eps)
        rows.append(eater_eps[["cond_label", "t_ms"]])

    if not rows:
        return
    long = pd.concat(rows, ignore_index=True)
    order = [cfg.cond_labels[s] for s in conditions if step_dfs.get(s) is not None]
    palette = {cfg.cond_labels[s]: cfg.cond_colors[s] for s in conditions}
    fig, ax = panel()
    condition_plot(ax, long, "cond_label", "t_ms", palette, order=order,
                   ylabel="Time to first food — B (ms, eaters only)", rotation=20)
    for i, spec in enumerate([s for s in conditions if step_dfs.get(s) is not None]):
        ax.text(i, ax.get_ylim()[0], f"N={n_eaters[spec]}",
                ha="center", va="bottom", fontsize=6, color="gray")
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "time_to_first_food_b_eaters.pdf"))


# ── stat helpers ─────────────────────────────────────────────────────────────

def _bh_correct(pvalues):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values in original order."""
    m = len(pvalues)
    if m == 0:
        return np.array([])
    p = np.array(pvalues, dtype=float)
    order = np.argsort(p)
    p_sorted = p[order]
    adjusted = p_sorted * m / np.arange(1, m + 1)
    for i in range(m - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    adjusted = np.minimum(adjusted, 1.0)
    result = np.empty(m)
    result[order] = adjusted
    return result


def _km_curve(times, events):
    """
    Kaplan-Meier estimator with pointwise 95% CI (Greenwood formula).
    events: 1 = event occurred, 0 = right-censored.
    Returns (t, S, S_lo, S_hi) as step-function arrays for plotting.
    """
    times, events = np.asarray(times, float), np.asarray(events, int)
    event_times = np.sort(np.unique(times[events == 1]))
    S = 1.0
    G = 0.0  # Greenwood cumulative sum: sum d / (n * (n - d))
    t_km = [0.0];  s_km = [1.0];  lo_km = [1.0];  hi_km = [1.0]
    for t in event_times:
        n_risk  = int((times >= t).sum())
        n_event = int(((times == t) & (events == 1)).sum())
        S *= 1 - n_event / n_risk
        if n_risk > n_event:
            G += n_event / (n_risk * (n_risk - n_event))
        se = S * np.sqrt(G)
        lo, hi = max(0.0, S - 1.96 * se), min(1.0, S + 1.96 * se)
        t_km.extend([t, t]);  s_km.extend([s_km[-1], S])
        lo_km.extend([lo_km[-1], lo]);  hi_km.extend([hi_km[-1], hi])
    if len(times):
        t_km.append(float(times.max()))
        for lst in (s_km, lo_km, hi_km):
            lst.append(lst[-1])
    return (np.array(t_km), np.array(s_km),
            np.array(lo_km), np.array(hi_km))


def _logrank_p(t1, e1, t2, e2):
    """Two-sample log-rank test p-value (chi-squared, 1 df)."""
    from scipy.stats import chi2 as _chi2
    t1, e1, t2, e2 = map(np.asarray, [t1, e1, t2, e2])
    event_times = np.unique(np.concatenate([t1[e1 == 1], t2[e2 == 1]]))
    O = E = V = 0.0
    for t in event_times:
        n1, n2 = int((t1 >= t).sum()), int((t2 >= t).sum())
        d1, d2 = int(((t1 == t) & (e1 == 1)).sum()), int(((t2 == t) & (e2 == 1)).sum())
        N, D = n1 + n2, d1 + d2
        if N < 2 or D == 0:
            continue
        E += n1 * D / N
        V += n1 * n2 * D * (N - D) / (N ** 2 * (N - 1))
        O += d1
    if V <= 0:
        return 1.0
    return float(_chi2.sf((O - E) ** 2 / V, df=1))


# ── annotated time-to-first plots ────────────────────────────────────────────

def _annotate_mwu_bh(ax, long, x_col, y_col, order):
    """
    Run all pairwise Mann-Whitney U tests, apply BH FDR, and annotate only
    significant pairs (p_BH < 0.05) using statannotations.
    Returns list of dicts with full results for all pairs.
    """
    from statannotations.Annotator import Annotator
    all_pairs = [(a, b) for i, a in enumerate(order) for b in order[i + 1:]]
    raw_ps = []
    for a, b in all_pairs:
        va = long.loc[long[x_col] == a, y_col].dropna().values
        vb = long.loc[long[x_col] == b, y_col].dropna().values
        if len(va) < 2 or len(vb) < 2:
            raw_ps.append(1.0)
        else:
            _, p = stats.mannwhitneyu(va, vb, alternative="two-sided")
            raw_ps.append(float(p))
    adj_ps = _bh_correct(raw_ps)
    result_rows = [
        {"cond_a": a, "cond_b": b, "p_raw": float(pr), "p_bh": float(pa),
         "sig": "***" if pa < 0.001 else "**" if pa < 0.01 else "*" if pa < 0.05 else "n.s."}
        for (a, b), pr, pa in zip(all_pairs, raw_ps, adj_ps)
    ]
    sig_idx = [i for i, q in enumerate(adj_ps) if q < 0.05]
    if sig_idx:
        sig_pairs = [all_pairs[i] for i in sig_idx]
        sig_adj   = [adj_ps[i]    for i in sig_idx]
        ann = Annotator(ax, sig_pairs, data=long, x=x_col, y=y_col, order=order)
        ann.configure(test=None, text_format="star", loc="outside", verbose=False)
        ann.set_pvalues_and_annotate(sig_adj)
    return result_rows


def plot_time_to_first_food_b_stats(step_dfs, out_dir, conditions=None, cfg=None):
    """
    Censored time-to-first-food for B with BH-corrected pairwise MWU brackets.
    Only significant pairs annotated.  Pass conditions=cfg.main_conditions to produce
    a main-only variant (omits control_b; saves as *_main.pdf).

    Caveat: censoring at max_steps creates a point mass; MWU is valid but has reduced
    sensitivity.  The survival plot (survival_b.pdf) is the preferred inferential tool.
    """
    if cfg is None:
        cfg = _DEFAULT_CFG
    set_style()
    if conditions is None:
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
        merged["t_ms"] = merged["t_first"].fillna(max_steps) * TIME_STEP_MS
        merged["cond_label"] = cfg.cond_labels[spec]
        rows.append(merged[["cond_label", "t_ms"]])
    if not rows:
        return
    long = pd.concat(rows, ignore_index=True)
    order = [cfg.cond_labels[s] for s in conditions if step_dfs.get(s) is not None]
    palette = {cfg.cond_labels[s]: cfg.cond_colors[s] for s in conditions}
    fig, ax = panel()
    condition_plot(ax, long, "cond_label", "t_ms", palette, order=order,
                   ylabel="Time to first food — B (ms)", rotation=20)
    if max_steps is not None:
        ax.axhline(max_steps * TIME_STEP_MS, color="gray", lw=0.8, ls="--", alpha=0.6)
    mwu_rows = _annotate_mwu_bh(ax, long, "cond_label", "t_ms", order)
    plt.tight_layout(pad=0.3)
    suffix = "_main" if set(conditions) == set(cfg.main_conditions) else ""
    save(fig, os.path.join(out_dir, f"time_to_first_food_b_stats{suffix}.pdf"))
    if mwu_rows:
        pd.DataFrame(mwu_rows).to_csv(
            os.path.join(out_dir, f"time_to_first_food_b_mwu_bh{suffix}.csv"), index=False)


def plot_time_to_first_food_b_eaters_stats(step_dfs, out_dir, conditions=None, cfg=None):
    """
    Eaters-only time-to-first-food for B with BH-corrected pairwise MWU brackets.
    Pass conditions=cfg.main_conditions to produce a main-only variant (*_main.pdf).
    See plot_time_to_first_food_b_eaters docstring for selection-bias caveat.
    """
    if cfg is None:
        cfg = _DEFAULT_CFG
    set_style()
    if conditions is None:
        conditions = cfg.main_conditions + [cfg.control_b]
    rows, n_eaters = [], {}
    for spec in conditions:
        df = step_dfs.get(spec)
        if df is None:
            continue
        b_eat = df[(df["role"] == "B") & (df["eating_event"].astype(bool))]
        first = b_eat.groupby(["env_id", "episode_index"])["time_step"].min()
        eater_eps = first.reset_index().rename(columns={"time_step": "t_first"})
        eater_eps["t_ms"] = eater_eps["t_first"] * TIME_STEP_MS
        eater_eps["cond_label"] = cfg.cond_labels[spec]
        n_eaters[spec] = len(eater_eps)
        rows.append(eater_eps[["cond_label", "t_ms"]])
    if not rows:
        return
    long = pd.concat(rows, ignore_index=True)
    order = [cfg.cond_labels[s] for s in conditions if step_dfs.get(s) is not None]
    palette = {cfg.cond_labels[s]: cfg.cond_colors[s] for s in conditions}
    fig, ax = panel()
    condition_plot(ax, long, "cond_label", "t_ms", palette, order=order,
                   ylabel="Time to first food — B (ms, eaters only)", rotation=20)
    for i, spec in enumerate([s for s in conditions if step_dfs.get(s) is not None]):
        ax.text(i, ax.get_ylim()[0], f"N={n_eaters[spec]}",
                ha="center", va="bottom", fontsize=6, color="gray")
    mwu_rows = _annotate_mwu_bh(ax, long, "cond_label", "t_ms", order)
    plt.tight_layout(pad=0.3)
    suffix = "_main" if set(conditions) == set(cfg.main_conditions) else ""
    save(fig, os.path.join(out_dir, f"time_to_first_food_b_eaters_stats{suffix}.pdf"))
    if mwu_rows:
        pd.DataFrame(mwu_rows).to_csv(
            os.path.join(out_dir, f"time_to_first_food_b_eaters_mwu_bh{suffix}.csv"), index=False)


# ── survival plot ─────────────────────────────────────────────────────────────

def plot_survival_b(step_dfs, out_dir, conditions=None, cfg=None):
    """
    Kaplan-Meier survival curves for B (P(not yet eaten) vs time), one curve per
    condition.  Tick marks on each curve show censored observations.

    Pairwise log-rank tests (BH-corrected) appear in a text box.  Pass
    conditions=cfg.main_conditions for a main-only variant (*_main.pdf) that
    excludes control_b (n≈2 eaters makes its survival curve uninformative).
    This is the preferred inferential complement to the boxplots because it
    handles censoring without discarding episodes or creating a point mass.
    """
    if cfg is None:
        cfg = _DEFAULT_CFG
    set_style()
    if conditions is None:
        conditions = cfg.main_conditions + [cfg.control_b]
    avail = [s for s in conditions if step_dfs.get(s) is not None]
    if not avail:
        return

    # Build per-condition (times_ms, events) arrays
    cond_data = {}
    for spec in avail:
        df = step_dfs[spec]
        max_steps = df["time_step"].max()
        b_eat = df[(df["role"] == "B") & (df["eating_event"].astype(bool))]
        first = b_eat.groupby(["env_id", "episode_index"])["time_step"].min()
        all_eps = df[df["role"] == "B"][["env_id", "episode_index"]].drop_duplicates()
        merged = all_eps.merge(first.rename("t_first").reset_index(),
                               on=["env_id", "episode_index"], how="left")
        times  = merged["t_first"].fillna(max_steps).values * TIME_STEP_MS
        events = merged["t_first"].notna().astype(int).values
        cond_data[spec] = (times, events)

    # Pairwise log-rank with BH
    pairs = [(a, b) for i, a in enumerate(avail) for b in avail[i + 1:]]
    raw_ps = [_logrank_p(cond_data[a][0], cond_data[a][1],
                         cond_data[b][0], cond_data[b][1])
              for a, b in pairs]
    adj_ps = _bh_correct(raw_ps)
    sig_str = {(a, b): ("***" if q < 0.001 else "**" if q < 0.01 else
                         "*" if q < 0.05 else "n.s.")
               for (a, b), q in zip(pairs, adj_ps)}

    fig, ax = panel()
    for spec in avail:
        times, events = cond_data[spec]
        t_km, s_km, lo_km, hi_km = _km_curve(times, events)
        color = cfg.cond_colors[spec]
        ax.plot(t_km, s_km, color=color, lw=1.4, label=cfg.cond_labels[spec])
        ax.fill_between(t_km, lo_km, hi_km, color=color, alpha=0.15)
        censored_t = times[events == 0] * 1.0
        if len(censored_t):
            s_at_censor = np.interp(censored_t, t_km, s_km)
            ax.plot(censored_t, s_at_censor, "+", color=color,
                    ms=4, mew=0.8, alpha=0.6, zorder=3)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("P(B has not eaten)")
    ax.set_ylim(-0.05, 1.05)
    sns.despine(ax=ax)

    p_lines = ["Pairwise log-rank (BH):"]
    for (a, b), q, sig in zip(pairs, adj_ps, [sig_str[p] for p in pairs]):
        p_lines.append(f"  {cfg.cond_labels[a]} vs {cfg.cond_labels[b]}: p={q:.3f} {sig}")
    ax.legend(fontsize=6, frameon=False, loc="upper right",
              handlelength=1.1, handletextpad=0.35,
              borderaxespad=0.25, labelspacing=0.2)
    ax.text(0.03, 0.03, "\n".join(p_lines), transform=ax.transAxes,
            fontsize=6, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="none", lw=0, alpha=0.8))
    plt.tight_layout(pad=0.3)
    suffix = "_main" if set(conditions) == set(cfg.main_conditions) else ""
    save(fig, os.path.join(out_dir, f"survival_b{suffix}.pdf"))
    logrank_rows = [
        {"cond_a": cfg.cond_labels[a], "cond_b": cfg.cond_labels[b],
         "p_raw": float(pr), "p_bh": float(pa),
         "sig": sig_str[(a, b)]}
        for (a, b), pr, pa in zip(pairs, raw_ps, adj_ps)
    ]
    if logrank_rows:
        pd.DataFrame(logrank_rows).to_csv(
            os.path.join(out_dir, f"survival_b_logrank{suffix}.csv"), index=False)


def plot_nn_distance_by_condition(ep_dfs, out_dir, cfg=None):
    """Mean nearest-neighbour distance per episode, by condition."""
    if cfg is None:
        cfg = _DEFAULT_CFG
    set_style()
    rows = []
    for spec in cfg.main_conditions:
        df = ep_dfs.get(spec)
        if df is None or "mean_nn_distance_cm" not in df.columns:
            continue
        tmp = df[["mean_nn_distance_cm"]].copy()
        tmp["cond_label"] = cfg.cond_labels[spec]
        rows.append(tmp)
    if not rows:
        return
    long = pd.concat(rows, ignore_index=True)
    order = [cfg.cond_labels[s] for s in cfg.main_conditions if ep_dfs.get(s) is not None]
    palette = {cfg.cond_labels[s]: cfg.cond_colors[s] for s in cfg.main_conditions}
    fig, ax = panel()
    condition_plot(ax, long, "cond_label", "mean_nn_distance_cm", palette,
                   order=order, ylabel="Mean NN distance (cm)")
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "nn_distance_by_condition.pdf"))


def plot_peod_by_role_condition(agent_dfs, out_dir, cfg=None):
    """Boxplot+strip: P(emit EOD) for roles A and B, per condition."""
    if cfg is None:
        cfg = _DEFAULT_CFG
    set_style()
    conditions = [s for s in cfg.main_conditions if agent_dfs.get(s) is not None]
    n = len(conditions)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 2.0), sharey=True)
    if n == 1:
        axes = [axes]
    role_order = ["A", "B"]
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
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "peod_by_role_condition.pdf"))


def _load_patch_centers(raw_dir: str) -> pd.DataFrame:
    """
    Load *_arena.pkl files and return patch centroid (mean food position at t=0)
    per (env_id, episode_index).  Returns empty DataFrame on missing data.
    """
    import glob as _glob
    arena_files = sorted(_glob.glob(os.path.join(raw_dir, "*_arena.pkl")))
    if not arena_files:
        return pd.DataFrame(columns=["env_id", "episode_index", "patch_cx", "patch_cy"])
    arena_df = pd.concat([pd.read_pickle(f) for f in arena_files], ignore_index=True)
    # keep only the first recorded time step per episode
    start_arena = (
        arena_df.sort_values("time_step")
        .groupby(["env_id", "episode_index"], sort=False)
        .first()
        .reset_index()[["env_id", "episode_index", "food_positions"]]
    )
    rows = []
    for _, row in start_arena.iterrows():
        fp = np.asarray(row["food_positions"])
        if len(fp) > 0:
            cx, cy = float(fp[:, 0].mean()), float(fp[:, 1].mean())
        else:
            cx, cy = np.nan, np.nan
        rows.append({"env_id": row["env_id"], "episode_index": row["episode_index"],
                     "patch_cx": cx, "patch_cy": cy})
    return pd.DataFrame(rows)


def plot_dist_to_patch_center_by_condition(step_dfs, evals_dir, out_dir, cfg=None):
    """Mean per-episode distance to food-patch centroid for roles A and B, by condition."""
    if cfg is None:
        cfg = _DEFAULT_CFG
    set_style()
    conditions = [s for s in cfg.main_conditions if step_dfs.get(s) is not None]
    n = len(conditions)
    if n == 0:
        return

    all_rows = []
    for spec in conditions:
        df = step_dfs[spec]
        raw_dir = os.path.join(evals_dir, spec, "raw")
        patch_centers = _load_patch_centers(raw_dir)
        if patch_centers.empty:
            continue
        merged = df.merge(patch_centers, on=["env_id", "episode_index"], how="left")
        merged = merged.copy()
        merged["dist_to_patch"] = np.sqrt(
            (merged["position_x"] - merged["patch_cx"]) ** 2 +
            (merged["position_y"] - merged["patch_cy"]) ** 2
        )
        mean_dist = (
            merged.groupby(["env_id", "episode_index", "agent_id"])["dist_to_patch"]
            .mean()
            .reset_index()
        )
        mean_dist["role"] = mean_dist["agent_id"].map({0: "A", 1: "B"})
        mean_dist["cond_label"] = cfg.cond_labels[spec]
        all_rows.append(mean_dist)

    if not all_rows:
        return
    long = pd.concat(all_rows, ignore_index=True)

    avail_conditions = [s for s in conditions if step_dfs.get(s) is not None]
    fig, axes = plt.subplots(1, len(avail_conditions),
                             figsize=(2.0 * len(avail_conditions), 2.0), sharey=True)
    if len(avail_conditions) == 1:
        axes = [axes]
    role_order = ["A", "B"]
    for ax, spec in zip(axes, avail_conditions):
        sub = long[long["cond_label"] == cfg.cond_labels[spec]]
        n_per_role = sub.groupby("role")["dist_to_patch"].count()
        n_map = {r: int(n_per_role.get(r, 0)) for r in role_order}
        sns.boxplot(data=sub, x="role", y="dist_to_patch", order=role_order,
                    palette={"A": _ROLE_COLORS["A"], "B": _ROLE_COLORS["B"]},
                    ax=ax, width=0.55, linewidth=0.8, fliersize=0,
                    boxprops=dict(alpha=0.75))
        if max(n_map.values(), default=0) < PLOT_POINTS_BELOW_N:
            sns.stripplot(data=sub, x="role", y="dist_to_patch", order=role_order,
                          color="black", alpha=0.5, jitter=0.2, size=3, ax=ax)
        ax.set_xticks(range(len(role_order)))
        ax.set_xticklabels([f"{r}\n(N={n_map[r]})" for r in role_order])
        ax.set_title(cfg.cond_labels[spec], fontsize=7)
        ax.set_xlabel("")
        sns.despine(ax=ax)
    axes[0].set_ylabel("Mean dist to patch center (cm)")
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "dist_to_patch_center_by_condition.pdf"))


def plot_b_dist_to_patch_dunnett(step_dfs, evals_dir, out_dir, cfg=None):
    """
    Mean per-episode distance to food-patch centroid for B only, across the three
    main conditions.  Dunnett's test vs the A=B condition (control); significant
    pairs annotated; full results saved as CSV.
    """
    from scipy.stats import dunnett as _dunnett
    from statannotations.Annotator import Annotator
    if cfg is None:
        cfg = _DEFAULT_CFG
    set_style()

    rows = []
    for spec in cfg.main_conditions:
        df = step_dfs.get(spec)
        if df is None:
            continue
        raw_dir = os.path.join(evals_dir, spec, "raw")
        patch_centers = _load_patch_centers(raw_dir)
        if patch_centers.empty:
            continue
        b_df = df[df["role"] == "B"].merge(
            patch_centers, on=["env_id", "episode_index"], how="left"
        ).copy()
        b_df["dist_to_patch"] = np.sqrt(
            (b_df["position_x"] - b_df["patch_cx"]) ** 2 +
            (b_df["position_y"] - b_df["patch_cy"]) ** 2
        )
        mean_dist = (
            b_df.groupby(["env_id", "episode_index"])["dist_to_patch"]
            .mean()
            .reset_index()
        )
        mean_dist["cond_label"] = cfg.cond_labels[spec]
        rows.append(mean_dist)

    if not rows:
        return
    long = pd.concat(rows, ignore_index=True)
    order = [cfg.cond_labels[s] for s in cfg.main_conditions
             if step_dfs.get(s) is not None]
    palette = {cfg.cond_labels[s]: cfg.cond_colors[s] for s in cfg.main_conditions}

    fig, ax = panel()
    condition_plot(ax, long, "cond_label", "dist_to_patch", palette,
                   order=order, ylabel="Mean dist to patch center — B (cm)", rotation=20)

    # Dunnett's test: AltB and AgtB vs AeqB (control = index 1)
    ctrl_spec = cfg.main_conditions[1]  # AeqB
    ctrl_label = cfg.cond_labels[ctrl_spec]
    ctrl_vals = long.loc[long["cond_label"] == ctrl_label, "dist_to_patch"].dropna().values
    treat_specs = [cfg.main_conditions[0], cfg.main_conditions[2]]  # AltB, AgtB
    treat_groups = [
        long.loc[long["cond_label"] == cfg.cond_labels[s], "dist_to_patch"].dropna().values
        for s in treat_specs
    ]
    result = _dunnett(*treat_groups, control=ctrl_vals)

    # bracket annotation for significant pairs
    sig_pairs = [(cfg.cond_labels[s], ctrl_label)
                 for s, p in zip(treat_specs, result.pvalue) if p < 0.05]
    sig_pvals = [float(p) for p in result.pvalue if p < 0.05]
    if sig_pairs:
        ann = Annotator(ax, sig_pairs, data=long, x="cond_label",
                        y="dist_to_patch", order=order)
        ann.configure(test=None, text_format="star", loc="outside", verbose=False)
        ann.set_pvalues_and_annotate(sig_pvals)

    # text box summary
    p_lines = [f"Dunnett vs {ctrl_label}:"]
    dunnett_rows = []
    for s, p in zip(treat_specs, result.pvalue):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        p_lines.append(f"  {cfg.cond_labels[s]}: p={p:.3f} {sig}")
        dunnett_rows.append({"treatment": cfg.cond_labels[s], "control": ctrl_label,
                             "p": float(p), "sig": sig})
    ax.text(0.02, 0.98, "\n".join(p_lines), transform=ax.transAxes,
            fontsize=7, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="gray", lw=0.5, alpha=0.8))

    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "b_dist_to_patch_dunnett.pdf"))
    pd.DataFrame(dunnett_rows).to_csv(
        os.path.join(out_dir, "b_dist_to_patch_dunnett.csv"), index=False)


def plot_food_vs_size(agent_dfs, out_dir, cfg=None):
    """Food eaten vs agent size — open-circle scatter + regression per role."""
    from statsmodels.api import OLS, add_constant
    if cfg is None:
        cfg = _DEFAULT_CFG
    set_style()
    fig, ax = panel()
    ols_rows = []
    for role, color in _ROLE_COLORS.items():
        xs, ys = [], []
        for spec in cfg.main_conditions:
            df = agent_dfs.get(spec)
            if df is None:
                continue
            sub = df[df["role"] == role]
            xs.extend(sub["agent_size"].values)
            ys.extend(sub["food_eaten"].values)
        xs, ys = np.array(xs, dtype=float), np.array(ys, dtype=float)
        valid = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[valid], ys[valid]
        ax.scatter(xs, ys, facecolors="none", edgecolors=color,
                   s=18, linewidths=0.8, alpha=0.7, label=f"Role {role}")
        if len(xs) > 3:
            X = add_constant(xs)
            model = OLS(ys, X).fit()
            xs_s = np.sort(xs)
            ax.plot(xs_s, model.predict(add_constant(xs_s)), color=color, lw=1.2, ls="--")
            if len(model.pvalues) > 1:
                ols_rows.append({
                    "role": role,
                    "x_var": "agent_size", "y_var": "food_eaten",
                    "slope": float(model.params[1]),
                    "intercept": float(model.params[0]),
                    "r2": float(model.rsquared),
                    "p": float(model.pvalues[1]),
                    "n": int(len(xs)),
                })
    ax.set_xlabel("Agent size")
    ax.set_ylabel("Food eaten")
    ax.legend(frameon=False, fontsize=7, handlelength=1.2)
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "food_vs_size.pdf"))
    if ols_rows:
        pd.DataFrame(ols_rows).to_csv(
            os.path.join(out_dir, "food_vs_size_ols.csv"), index=False)


def _build_event_times(step_dfs, conditions, event_fn, cfg=None):
    """Return (condition_times, palette) for the given event function."""
    if cfg is None:
        cfg = _DEFAULT_CFG
    condition_times, palette = [], {}
    for spec in conditions:
        df = step_dfs.get(spec)
        if df is None:
            continue
        times = event_fn(df)
        if times:
            label = cfg.cond_labels[spec]
            condition_times.append((label, times))
            palette[label] = cfg.cond_colors[spec]
    return condition_times, palette


def plot_time_to_consumption_by_condition(step_dfs, out_dir, cfg=None):
    """Time-to-nth-food curves, one line per condition (A>B / A=B / A<B / B-only)."""
    if cfg is None:
        cfg = _DEFAULT_CFG
    ct, palette = _build_event_times(step_dfs, cfg.main_conditions + [cfg.control_b],
                                     consumption_times_from_step_df, cfg=cfg)
    save_event_time_plot_set(ct, palette, out_dir, "time_to_consumption",
                             event_label="Food consumed (A + B)")


def plot_time_to_consumption_by_role(step_dfs, out_dir, cfg=None):
    """Role-separated food-consumption timecourses: one pair for A, one pair for B.

    Each pair has two variants:
      excl_zeros — only episodes where that agent ate (current default behaviour)
      incl_zeros — all episodes; non-eating episodes contribute a flat zero line
    Outputs: time_to_consumption_{A,B}_{excl,incl}_zeros_timecourse_linear.pdf
    """
    if cfg is None:
        cfg = _DEFAULT_CFG
    from analysis_style import X_timecourse_linear

    for role, agent_id in [("A", 0), ("B", 1)]:
        for incl in [False, True]:
            suffix = "incl_zeros" if incl else "excl_zeros"
            condition_times, palette = [], {}
            for spec in cfg.main_conditions:
                df = step_dfs.get(spec)
                if df is None:
                    continue
                role_df = df[df["agent_id"] == agent_id]
                # episodes with ≥1 eating event for this agent
                eating_eps = (
                    role_df[role_df["eating_event"].astype(bool)][["env_id", "episode_index"]]
                    .drop_duplicates()
                )
                all_eps = role_df[["env_id", "episode_index"]].drop_duplicates()
                ep_arrays = consumption_times_from_step_df(role_df)
                if incl:
                    n_zeros = len(all_eps) - len(eating_eps)
                    ep_arrays = ep_arrays + [np.array([])] * n_zeros
                label = cfg.cond_labels[spec]
                condition_times.append((label, ep_arrays))
                palette[label] = cfg.cond_colors[spec]
            if len(condition_times) < 2:
                continue
            out_path = os.path.join(
                out_dir, f"time_to_consumption_{role}_{suffix}_timecourse_linear.pdf"
            )
            X_timecourse_linear(
                condition_times, palette, out_path,
                event_label=f"Food consumed ({role})",
                time_label="Time (s)",
                include_zeros=incl,
            )


def plot_time_to_bitten_by_condition(step_dfs, out_dir, cfg=None):
    """Time-to-nth-biting-event curves, one line per condition."""
    if cfg is None:
        cfg = _DEFAULT_CFG
    ct, palette = _build_event_times(step_dfs, cfg.main_conditions + [cfg.control_b],
                                     biting_times_from_step_df, cfg=cfg)
    save_event_time_plot_set(ct, palette, out_dir, "time_to_bitten",
                              event_label="Biting events",
                              time_label="Time to biting event (s)")


def _select_traj_samples(df, n=N_TRAJ_SAMPLES):
    """Return up to n (env_id, episode_index) pairs, sorted descending by eating events."""
    totals = (df.groupby(["env_id", "episode_index"])["eating_event"]
              .sum()
              .sort_values(ascending=False))
    return list(totals.head(n).index)


class _HandlerColormapGradient(HandlerBase):
    """Renders a colormap as a gradient rectangle in the legend."""
    def __init__(self, cmap, n=32, **kwargs):
        super().__init__(**kwargs)
        self.cmap = cmap
        self.n = n

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        artists = []
        for i in range(self.n):
            x0 = xdescent + width * i / self.n
            w = width / self.n
            color = self.cmap(i / (self.n - 1))
            rect = mpatches.Rectangle((x0, ydescent), w, height,
                                      transform=trans, facecolor=color,
                                      edgecolor=color, linewidth=0)
            artists.append(rect)
        return artists


class _ColormapLegendHandle:
    def __init__(self, cmap):
        self.cmap = cmap


_CMAP_A = cm.Blues
_CMAP_B = cm.Oranges


def _save_trajectory_2f1p(sub, out_path, with_legend, max_steps=None, title=None):
    """Single-episode trajectory: time-colored scatter matching original utils_2f1p aesthetic.

    max_steps: if set, truncate trajectory to this many steps and fix colormap range to [0, max_steps].
    title: optional condition label shown as a centered title (fontsize 5.5, matching Fig 4 panels).
    """
    set_style()
    arena_size = sub["arena_size"].iloc[0]
    w = arena_size[0] if hasattr(arena_size, "__len__") else 150
    h = arena_size[1] if hasattr(arena_size, "__len__") else 150
    if max_steps is not None:
        sub = sub[sub["time_step"] < max_steps]
        norm_max = max_steps
    else:
        norm_max = int(sub["time_step"].max()) if len(sub) else 400
    norm = mcolors.Normalize(vmin=0, vmax=norm_max)

    fig, ax = panel()

    for role, cmap, eat_color in [("A", _CMAP_A, "blue"), ("B", _CMAP_B, "red")]:
        r = sub[sub["role"] == role].sort_values("time_step")
        if len(r) == 0:
            continue
        ax.scatter(r["position_x"], r["position_y"],
                   c=r["time_step"], cmap=cmap, norm=norm,
                   s=10, alpha=0.8)
        eat = r[r["eating_event"].astype(bool)]
        if len(eat):
            ax.scatter(eat["position_x"], eat["position_y"],
                       marker="x", s=50, linewidths=2, color=eat_color, zorder=5)

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, linestyle="--", alpha=0.3)
    if title is not None:
        ax.set_title(title, fontsize=7, pad=2, loc="center")

    if with_legend:
        h_A = _ColormapLegendHandle(_CMAP_A)
        h_B = _ColormapLegendHandle(_CMAP_B)
        ax.legend(
            handles=[
                h_A, h_B,
                mlines.Line2D([0], [0], marker="x", color="blue", linestyle="None",
                              markersize=8, markeredgewidth=2),
                mlines.Line2D([0], [0], marker="x", color="red", linestyle="None",
                              markersize=8, markeredgewidth=2),
            ],
            labels=["Agent A", "Agent B", "Eating (A)", "Eating (B)"],
            handler_map={h_A: _HandlerColormapGradient(_CMAP_A),
                         h_B: _HandlerColormapGradient(_CMAP_B)},
            fontsize=6,
        )

    plt.tight_layout(pad=0.3)
    save(fig, out_path)
    plt.close(fig)


def plot_trajectory_samples_by_condition(step_dfs, out_dir, n=N_TRAJ_SAMPLES,
                                         max_steps=400, cfg=None):
    """Per-condition folder of N sample trajectory plots (with and without legend).

    max_steps: truncate each trajectory to this many steps and fix the colormap
               range to [0, max_steps] so colours are comparable across episodes.
               Pass None for full-length, per-episode-normalised plots.
    """
    if cfg is None:
        cfg = _DEFAULT_CFG
    all_specs = cfg.main_conditions + [cfg.control_a, cfg.control_b]
    for spec in all_specs:
        df = step_dfs.get(spec)
        if df is None:
            continue
        spec_dir = os.path.join(out_dir, "trajectories", spec)
        os.makedirs(spec_dir, exist_ok=True)
        cond_label = cfg.cond_labels.get(spec)
        samples = _select_traj_samples(df, n=n)
        print(f"  {spec}: {len(samples)} trajectory samples")
        for env_id, ep_idx in samples:
            sub = df[(df["env_id"] == env_id) & (df["episode_index"] == ep_idx)]
            stem = os.path.join(spec_dir, f"trajectory_env{env_id}_ep{ep_idx}")
            _save_trajectory_2f1p(sub, stem + ".pdf", with_legend=False,
                                  max_steps=max_steps, title=cond_label)
            _save_trajectory_2f1p(sub, stem + "_legend.pdf", with_legend=True,
                                  max_steps=max_steps, title=cond_label)


# ── main ─────────────────────────────────────────────────────────────────────

def run(evals_dir, out_dir=None, spec_prefix="2f1p", save_trajectories=True):
    cfg = _make_cfg(spec_prefix)
    if out_dir is None:
        out_dir = os.path.join(evals_dir, "..", "multi_eval", spec_prefix)
    out_dir = os.path.normpath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    all_specs = cfg.main_conditions + [cfg.control_a, cfg.control_b]
    agent_dfs = {s: load_agent(evals_dir, s, cfg=cfg) for s in all_specs}
    step_dfs  = {s: load_step(evals_dir, s)           for s in all_specs}
    ep_dfs    = {s: load_ep(evals_dir, s, cfg=cfg)    for s in all_specs}

    found = [s for s in all_specs if agent_dfs[s] is not None]
    print(f"Found {len(found)}/{len(all_specs)} {spec_prefix} specs: {found}")
    if not found:
        sys.exit(f"No {spec_prefix} data found.")

    plot_b_eats_pct(agent_dfs, out_dir, cfg=cfg)
    plot_b_food_by_condition(agent_dfs, out_dir, cfg=cfg)
    plot_social_vs_alone(agent_dfs, out_dir, cfg=cfg)
    plot_social_vs_alone_pct(agent_dfs, out_dir, cfg=cfg)
    plot_time_to_first_food_b(step_dfs, out_dir, cfg=cfg)
    plot_time_to_first_food_b_stats(step_dfs, out_dir, cfg=cfg)
    plot_time_to_first_food_b_stats(step_dfs, out_dir, conditions=cfg.main_conditions, cfg=cfg)
    plot_time_to_first_food_b_eaters(step_dfs, out_dir, cfg=cfg)
    plot_time_to_first_food_b_eaters_stats(step_dfs, out_dir, cfg=cfg)
    plot_time_to_first_food_b_eaters_stats(step_dfs, out_dir, conditions=cfg.main_conditions, cfg=cfg)
    plot_survival_b(step_dfs, out_dir, cfg=cfg)
    plot_survival_b(step_dfs, out_dir, conditions=cfg.main_conditions, cfg=cfg)
    plot_time_to_consumption_by_condition(step_dfs, out_dir, cfg=cfg)
    plot_time_to_consumption_by_role(step_dfs, out_dir, cfg=cfg)
    plot_time_to_bitten_by_condition(step_dfs, out_dir, cfg=cfg)
    plot_nn_distance_by_condition(ep_dfs, out_dir, cfg=cfg)
    plot_peod_by_role_condition(agent_dfs, out_dir, cfg=cfg)
    plot_dist_to_patch_center_by_condition(step_dfs, evals_dir, out_dir, cfg=cfg)
    plot_b_dist_to_patch_dunnett(step_dfs, evals_dir, out_dir, cfg=cfg)
    plot_food_vs_size(agent_dfs, out_dir, cfg=cfg)
    if save_trajectories:
        plot_trajectory_samples_by_condition(step_dfs, out_dir, cfg=cfg)
    print("Done.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--spec_prefix", default="2f1p",
                    help="Spec-key prefix, e.g. '2f1p' (default) or '2f1p_k0'")
    ap.add_argument("--skip_trajectories", action="store_true",
                    help="Skip per-episode trajectory PDF generation")
    args = ap.parse_args(argv)
    run(args.evals_dir, out_dir=args.out_dir, spec_prefix=args.spec_prefix,
        save_trajectories=not args.skip_trajectories)


if __name__ == "__main__":
    main()
