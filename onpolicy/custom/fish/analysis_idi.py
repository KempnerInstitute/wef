"""
IDI (inter-discharge interval) distributions and power-law fits.

Usage:
    python analysis_idi.py --spec_dir path/to/evals/m1a1k1_patchy_square

Outputs to {spec_dir}/analyses/idi/
  idi_histogram.pdf        — log-log IDI histogram: per-agent + real-fish overlay
  idi_powerlaw.pdf         — power-law fits: agents + real fish on same axes
  idi_by_agent_size.pdf    — IDI histograms pooled by size bin (0–0.33 / 0.33–0.66 / 0.66–1.0)
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import powerlaw

# allow running from fish/ directory
sys.path.insert(0, os.path.dirname(__file__))
from analysis_style import (
    set_style, panel, save,
    AGENT_COLORS, TIME_STEP_MS,
)
from utils_features import compute_idis
from cfg import REAL_FISH_IDI_CSV

REAL_FISH_COLOR = "#888888"
AGENT_POOL_COLOR = "#1F77B4"

OUT_SUBDIR = "analyses/idi"


# ── helpers ─────────────────────────────────────────────────────────────────

def _resolve_real_fish_csv():
    """Return the real-fish CSV path that exists, preferring local."""
    for key in ("local", "cluster"):
        p = REAL_FISH_IDI_CSV[key]
        if os.path.exists(p):
            return p
    return None


def _load_real_fish_idis():
    """Load real-fish IDIs (ms) from the SPI CSV.  Returns None if file not found."""
    path = _resolve_real_fish_csv()
    if path is None:
        print("  real-fish CSV not found — skipping real-fish overlay")
        return None
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c != "time"]
    vals = pd.concat([df[c].dropna() for c in cols]).values
    # CSV stores SPIs in ms
    return vals[(vals > 0) & np.isfinite(vals)]


def _idi_histogram_bins():
    """Log-spaced bins from 10 ms to 3 000 ms."""
    return np.logspace(np.log10(10), np.log10(3000), 50)


def _plot_log_hist(ax, idis, color, label, bins, density=True, lw=1.5, ls="-"):
    counts, edges = np.histogram(idis, bins=bins, density=density)
    centres = np.sqrt(edges[:-1] * edges[1:])   # geometric midpoint
    mask = counts > 0
    ax.plot(centres[mask], counts[mask], color=color, lw=lw, ls=ls, label=label)


# ── panels ───────────────────────────────────────────────────────────────────

def _plot_12ms_histogram(ax, idis_ms, max_plot_ms=204, title=None):
    """12 ms-bin histogram, truncated at max_plot_ms, log y-axis (Gebhardt style)."""
    idis = idis_ms[idis_ms < max_plot_ms]
    bin_width = 12
    edges = np.arange(0, idis.max() + bin_width, bin_width) - bin_width / 2
    ax.hist(idis, bins=edges)
    ax.set_yscale("log")
    ax.set_xlabel("IDI (ms)")
    ax.set_ylabel("Count")
    centers = (edges[:-1] + edges[1:]) / 2
    ax.set_xticks(centers[::2])
    ax.set_xticklabels([f"{int(c)}" for c in centers[::2]], rotation=45)
    ax.tick_params(axis="x", length=0)
    if title:
        ax.text(0.97, 0.97, title, transform=ax.transAxes,
                ha="right", va="top", fontsize=8, fontstyle="italic")
    sns.despine(ax=ax)


def plot_idi_histogram(df, out_dir):
    """Log-log histogram: one curve per agent + real-fish overlay."""
    set_style()
    bins = _idi_histogram_bins()
    rf_idis = _load_real_fish_idis()

    fig, ax = panel(2.0, 1.0)

    for i, agent_id in enumerate(sorted(df["agent_id"].unique())):
        sub = df[df["agent_id"] == agent_id]
        idis = compute_idis(sub)
        if len(idis) == 0:
            continue
        _plot_log_hist(ax, idis, AGENT_COLORS[i % len(AGENT_COLORS)],
                       f"Agent {agent_id}", bins, lw=1.2)

    if rf_idis is not None:
        _plot_log_hist(ax, rf_idis, REAL_FISH_COLOR, "Real fish", bins,
                       lw=1.5, ls="--")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("IDI (ms)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, handlelength=1.2)
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "idi_histogram.pdf"))


def _fit_powerlaw(idis_ms, xmin_ms=24):
    """Fit a power-law to IDIs (ms).  Returns (fit, alpha) or (None, None)."""
    idis = idis_ms[idis_ms >= 12]
    if len(idis) < 10:
        return None, None
    try:
        fit = powerlaw.Fit(idis, xmin=xmin_ms, discrete=True, verbose=False)
        return fit, fit.power_law.alpha
    except (ValueError, RuntimeError) as e:
        print(f"  powerlaw fit skipped ({e})")
        return None, None


def _draw_powerlaw_line(ax, fit, idis, color, label):
    """Overlay a normalised power-law line on ax."""
    bins = _idi_histogram_bins()
    alpha = fit.power_law.alpha
    x_min = fit.xmin
    x_max = np.percentile(idis, 99)
    xs = np.logspace(np.log10(x_min), np.log10(x_max), 100)
    counts, edges = np.histogram(idis, bins=bins, density=True)
    centres = np.sqrt(edges[:-1] * edges[1:])
    close = np.argmin(np.abs(centres - x_min))
    if counts[close] > 0:
        scale = counts[close] / (x_min ** (-alpha))
        ys = scale * xs ** (-alpha)
        ax.plot(xs, ys, color=color, ls="--", lw=1.2,
                label=rf"{label} $\alpha={alpha:.2f}$")


def _plot_powerlaw_fit(ax, idis_ms, label, title=None,
                       bin_width_ms=TIME_STEP_MS):
    """Power-law fit: blue data, red fit line, fixed manuscript-style axes.

    label is used only for the console print (e.g. "Agents", "dyad").
    Fixed axes match Gebhardt-paper style: xlim (12, 300), ylim (4e-5, 4e-1).
    Data bins are centred on the discrete IDI support so that the powerlaw
    library's automatic log bins do not create empty-bin gaps.  The fitted
    distribution is rendered via the powerlaw library's plot_pdf.
    """
    fit = powerlaw.Fit(idis_ms, xmin=12, discrete=True, verbose=False)
    alpha = fit.power_law.alpha
    max_idi = np.max(idis_ms)
    bins = np.arange(0.5 * bin_width_ms,
                     max_idi + 1.5 * bin_width_ms,
                     bin_width_ms)
    fit.plot_pdf(
        color="b", ax=ax, original_data=True, label="Data",
        bins=bins, linear_bins=True,
    )
    fit.power_law.plot_pdf(
        color="r", linestyle="--", ax=ax,
        label=rf"Fit, $\alpha={alpha:.2f}$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.2e1, 3.0e2)
    ax.set_ylim(4e-5, 4e-1)
    ax.set_xticks([2e1, 1e2, 2e2])
    ax.set_yticks([1e-2])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xticklabels([r"$2{\cdot}10^1$", r"$10^2$", r"$2{\cdot}10^2$"])
    ax.set_yticklabels([r"$10^{-2}$"])
    ax.set_xlabel("IDI (ms)", labelpad=1)
    ax.set_ylabel("Density", labelpad=1)
    ax.legend(frameon=False, fontsize=8, loc="lower left",
              handlelength=1.4, handletextpad=0.35,
              borderaxespad=0.2, labelspacing=0.2)
    ax.tick_params(axis="both", which="major", labelsize=7, pad=1)
    if title:
        ax.text(0.97, 0.97, title, transform=ax.transAxes,
                ha="right", va="top", fontsize=8, fontstyle="italic")
    sns.despine(ax=ax)
    print(f"  {label} power-law α = {alpha:.3f}  (xmin = {fit.xmin} ms)")
    return fit


def _plot_gebhardt(ax, idis_ms, max_spi=300, bin_width=12):
    """Horizontal bar chart of relative occurrence (%), reproducing Gebhardt et al. Fig. 4C style."""
    idis = np.asarray(idis_ms)
    idis = idis[np.isfinite(idis) & (idis <= max_spi)]
    edges = np.arange(0, idis.max() + bin_width, bin_width) - bin_width / 2
    counts, edges = np.histogram(idis, bins=edges)
    percent = counts / counts.sum() * 100
    centers = (edges[:-1] + edges[1:]) / 2
    ax.barh(centers, percent, height=bin_width * 0.9,
            color="black", edgecolor="black", linewidth=0.2)
    ax.set_xlabel("Relative occurrence (%)")
    ax.set_ylabel("IDI (ms)")
    ax.set_ylim(0, idis.max() + bin_width)
    ax.set_xlim(left=0)
    ax.text(0.62, 0.92,
            f"N = {len(idis)} IDIs\nMin.: {int(idis.min())} ms\nMax.: {int(idis.max())} ms",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5})
    sns.despine(ax=ax)


def plot_idi_histogram_12ms(df, out_dir):
    """12 ms-bin histogram of pooled-agent IDIs (matches real_fish_dyad_ipi_histogram style)."""
    set_style()
    agent_idis = compute_idis(df)
    fig, ax = panel(2.0, 1.25)
    _plot_12ms_histogram(ax, agent_idis, title="Agent")
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "idi_histogram_12ms.pdf"))


def plot_idi_gebhardt(df, out_dir):
    """Gebhardt-style horizontal relative-occurrence chart of pooled-agent IDIs."""
    set_style()
    agent_idis = compute_idis(df)
    fig, ax = panel(2.0, 2.0)
    _plot_gebhardt(ax, agent_idis)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "idi_gebhardt.pdf"))


def plot_idi_powerlaw(df, out_dir):
    """Power-law fit of pooled-agent IDIs using manuscript-style fixed axes."""
    set_style()
    agent_idis = compute_idis(df)
    fig, ax = panel(2.0, 1.25)
    _plot_powerlaw_fit(ax, agent_idis, "Agents", title="Agent")
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "idi_powerlaw.pdf"))


_SIZE_BINS = [(0.0, 0.33), (0.33, 0.66), (0.66, 1.0)]
_SIZE_BIN_LABELS = ["Small (0–0.33)", "Medium (0.33–0.66)", "Large (0.66–1.0)"]


def plot_idi_by_agent_size(df, spec_dir, out_dir):
    """IDI distributions pooled by agent size bin (0–0.33 / 0.33–0.66 / 0.66–1.0)."""
    if "agent_size" in df.columns:
        sizes = (df[["env_id", "episode_index", "agent_id", "agent_size"]]
                 .drop_duplicates())
    else:
        agent_pkl = os.path.join(spec_dir, "derived", "per_env_ep_agent.pkl")
        adf = pd.read_pickle(agent_pkl)
        sizes = adf[["env_id", "episode_index", "agent_id", "agent_size"]].drop_duplicates()

    df = df.merge(sizes, on=["env_id", "episode_index", "agent_id"], how="left",
                  suffixes=("", "_sz"))
    if "agent_size_sz" in df.columns:
        df["agent_size"] = df["agent_size_sz"]
        df = df.drop(columns=["agent_size_sz"])

    set_style()
    bins = _idi_histogram_bins()
    fig, ax = panel(2.0, 1.0)

    for (lo, hi), label, color in zip(_SIZE_BINS, _SIZE_BIN_LABELS, AGENT_COLORS):
        mask = (df["agent_size"] >= lo) & (df["agent_size"] <= hi)
        sub = df[mask]
        idis = compute_idis(sub)
        if len(idis) == 0:
            print(f"  no IDIs for size bin {label} — skipped")
            continue
        _plot_log_hist(ax, idis, color, label, bins, lw=1.2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("IDI (ms)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, handlelength=1.2)
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, os.path.join(out_dir, "idi_by_agent_size.pdf"))


# ── main ─────────────────────────────────────────────────────────────────────

def load(spec_dir):
    step_pkl = os.path.join(spec_dir, "derived", "per_env_ep_agent_step.pkl")
    if not os.path.exists(step_pkl):
        sys.exit(f"Not found: {step_pkl}")
    out_dir = os.path.join(spec_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Loading {step_pkl} \u2026")
    df = pd.read_pickle(step_pkl)
    print(f"  {len(df):,} rows, {df['agent_id'].nunique()} agents")
    return {"df": df, "out_dir": out_dir, "spec_dir": spec_dir}


def run(spec_dir):
    derived = os.path.join(spec_dir, "derived")
    step_pkl = os.path.join(derived, "per_env_ep_agent_step.pkl")
    if not os.path.exists(step_pkl):
        sys.exit(f"Not found: {step_pkl}")

    out_dir = os.path.join(spec_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {step_pkl} …")
    df = pd.read_pickle(step_pkl)
    print(f"  {len(df):,} rows, {df['agent_id'].nunique()} agents")

    plot_idi_histogram(df, out_dir)
    plot_idi_histogram_12ms(df, out_dir)
    plot_idi_gebhardt(df, out_dir)
    plot_idi_powerlaw(df, out_dir)
    plot_idi_by_agent_size(df, spec_dir, out_dir)
    print("Done.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec_dir", required=True,
                    help="Path to eval spec directory (contains derived/)")
    args = ap.parse_args(argv)
    run(args.spec_dir)


if __name__ == "__main__":
    main()
