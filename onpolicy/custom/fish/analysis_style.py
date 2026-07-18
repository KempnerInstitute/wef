"""Shared style utilities for analysis_*.py scripts.

Matches utils_figstyle.set_nature_style() exactly, plus shared helpers.
"""
import contextlib
import inspect
import os
import tempfile
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
import pandas as pd
from cfg import SIM_FPS, EOD_RATE_WINDOW, PLOT_POINTS_BELOW_N, AGENT_COLORS, SEMANTIC_COLORS

# ── fonts ──────────────────────────────────────────────────────────────────
_FONT_PATH = os.path.join(os.path.dirname(__file__), "fonts", "Arial.ttf")
if os.path.exists(_FONT_PATH):
    fm.fontManager.addfont(_FONT_PATH)
    _FONT_NAME = fm.FontProperties(fname=_FONT_PATH).get_name()
else:
    _FONT_NAME = "sans-serif"

# ── palette ────────────────────────────────────────────────────────────────
# AGENT_COLORS imported from cfg

# Ethogram state colours matching utils_behavior.py exactly
STATE_COLORS = {
    "eating":   "#4CAF50",   # green   — eating_event
    "bitten":   "#E53935",   # red     — was_bitten
    "biting":   "#7B3294",   # purple  — bite_other_fish
    "meeting":  "#FFB74D",   # orange  — has_nearby
    "emitting": "#90CAF9",   # pale blue — emit_eod (no direct existing equiv.)
    "silent":   "#EEEEEE",   # light gray — none
}

# Intervention / condition palette (up to 6 conditions)
COND_PALETTE = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377", "#66CCEE"]


def set_style():
    """Exact match to utils_figstyle.set_nature_style()."""
    sns.set(style="ticks")   # not sns.set_style — sns.set also resets color cycle
    plt.rcParams.update({
        "font.family":      _FONT_NAME,
        "axes.labelsize":   8,
        "xtick.labelsize":  7,
        "ytick.labelsize":  7,
        "legend.fontsize":  7,
        "axes.linewidth":   1,
        "xtick.direction":  "out",
        "ytick.direction":  "out",
        "figure.dpi":       300,
        "savefig.dpi":      300,
    })


def panel(w=2.0, h=2.0):
    """Return a (fig, ax) pair at the standard 2.0×2.0 panel size."""
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


def size_pct_heatmap(row_sizes, col_sizes, out_path, *, cmap="Blues",
                     xlabel="", ylabel="", title="", n_bins=4):
    """Quantile-binned 2-D size heatmap rendered as % of observations.

    Bins both axes on shared quantile edges over the pooled ``row_sizes`` and
    ``col_sizes`` (matching the analysis_bite_network counts heatmap), counts
    the joint distribution, and renders it as a percentage of all observations
    (cells sum to 100%). Total observation count N is shown in the title.

    Returns the integer count matrix (DataFrame), or None if there is too
    little data / size variance to bin.
    """
    row = pd.Series(np.asarray(row_sizes, dtype=float)).reset_index(drop=True)
    col = pd.Series(np.asarray(col_sizes, dtype=float)).reset_index(drop=True)
    valid = row.notna() & col.notna()
    row, col = row[valid], col[valid]
    n_obs = int(len(row))
    if n_obs < 2:
        return None

    pooled = pd.concat([row, col])
    edges = np.unique(np.quantile(pooled, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return None
    labels = [f"{e:.2f}" for e in edges[:-1]]
    row_bin = pd.cut(row, bins=edges, labels=labels, include_lowest=True)
    col_bin = pd.cut(col, bins=edges, labels=labels, include_lowest=True)

    counts = (pd.DataFrame({"row": row_bin, "col": col_bin})
              .groupby(["row", "col"], observed=False).size()
              .unstack(fill_value=0)
              .reindex(index=labels, columns=labels, fill_value=0))
    pct = 100.0 * counts / max(counts.values.sum(), 1)

    set_style()
    fig, ax = panel(w=2.0, h=2.0)
    sns.heatmap(pct, ax=ax,  
                cmap=cmap,
                annot=True,
                # annot_kws={"fontsize": 7}, 
                fmt=".1f", 
                linewidths=0.5,
                cbar=False,
                # cbar_kws={"label": "% of observations", "shrink": 0.8}
                )  
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}  (N={n_obs})" if title else f"N={n_obs}", fontsize=7)
    plt.tight_layout(pad=0.3)
    save(fig, out_path)
    return counts


def _format_count_label(counts):
    if not counts:
        return None
    if len(set(counts)) == 1:
        return f"N={counts[0]}"
    return f"N={min(counts)}-{max(counts)}"


def _add_top_left_count_label(ax, text):
    if not text:
        return
    ax.text(0.03, 0.95, text, transform=ax.transAxes,
            ha="left", va="top", fontsize=6, color="#333333",
            bbox={"facecolor": "white", "alpha": 0.68,
                  "edgecolor": "none", "pad": 1.8})


def _add_bottom_right_count_label(ax, text):
    if not text:
        return
    ax.text(0.97, 0.05, text, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6, color="#333333",
            bbox={"facecolor": "white", "alpha": 0.68,
                  "edgecolor": "none", "pad": 1.8})


def format_n_fish(values):
    """Format a set/iterable of per-condition fish counts as e.g. '4 fish' or '1–4 fish'."""
    vals = sorted({int(v) for v in values if v is not None and not pd.isna(v)})
    if not vals:
        return None
    return f"{vals[0]} fish" if len(vals) == 1 else f"{vals[0]}–{vals[-1]} fish"


def add_right_count_title(ax, n_obs_text=None, n_fish=None, fontsize=7):
    """Right-aligned title summarizing fish count and observation count.

    Used to move the N-observation annotation out of the plot body and into a
    title (font size matches legend.fontsize, default 7). ``n_fish`` is a
    pre-formatted string such as '4 fish'; ``n_obs_text`` e.g. 'N=18–20'.
    """
    parts = [p for p in (n_fish, n_obs_text) if p]
    if parts:
        ax.set_title(", ".join(parts), loc="right", fontsize=fontsize)


# Non-None while inside an inline_display() context.
_inline_tmpdir = None


@contextlib.contextmanager
def inline_display(tmpdir=None):
    """Redirect save() to PNG + IPython inline display for the duration of the block.

    Usage in notebooks::

        from analysis_style import inline_display
        with inline_display():
            time_to_X_log(...)   # displays inline instead of writing to disk
    """
    global _inline_tmpdir
    _inline_tmpdir = tmpdir or tempfile.mkdtemp()
    try:
        yield _inline_tmpdir
    finally:
        _inline_tmpdir = None


def save(fig, path, tight=True, formats=(".pdf",)):
    """Save figure. Default is PDF only; pass formats=(".png", ".pdf") to add raster.

    Inside an inline_display() context, saves as PNG to a temp dir and calls
    IPython.display instead of writing to path.
    """
    if _inline_tmpdir is not None:
        from IPython.display import display as _display, Image as _Image
        stem = os.path.splitext(os.path.basename(path))[0]
        caller_frame = inspect.currentframe().f_back
        caller_name = caller_frame.f_code.co_name if caller_frame is not None else "<unknown>"
        tmp_png = os.path.join(_inline_tmpdir, f"{stem}.png")
        kw = dict(dpi=150, bbox_inches="tight") if tight else dict(dpi=150)
        fig.savefig(tmp_png, **kw)
        plt.close(fig)
        print(f"  {caller_name} -> {stem}.png")
        _display(_Image(tmp_png))
        del caller_frame
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    base = path
    for ext in (".png", ".pdf", ".svg"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    kw = dict(dpi=300, bbox_inches="tight") if tight else dict(dpi=300)
    for ext in formats:
        fig.savefig(f"{base}{ext}", **kw)
    plt.close(fig)
    print(f"  saved → {base}{formats[0]}")


def condition_plot(ax, data, x_col, y_col, palette, order=None, ylabel=None, rotation=0,
                   count_label="per_group", n_fish=None):
    """Seaborn boxplot with N labels and conditional point overlay.

    Points are overlaid only when every group has fewer than PLOT_POINTS_BELOW_N
    observations, to avoid overloaded PDFs for large datasets.
    Returns a dict {group_label: n} for callers that want to include N in CSVs.

    count_label:
      - "per_group": append ``(n=...)`` to each x tick label.
      - "corner": keep x tick labels clean and add one top-left count box.
      - "bottom_right": keep x tick labels clean and add one bottom-right count box.
      - "title": clean x tick labels; show the count (and ``n_fish``) in a
        right-aligned title instead of inside the plot body.
      - "none": omit count labels from the plot.

    n_fish: pre-formatted fish-count string (e.g. '4 fish') shown in the title
    when ``count_label='title'``.
    """
    data = data.dropna(subset=[y_col])
    sns.boxplot(data=data, x=x_col, y=y_col, hue=x_col, order=order,
                palette=palette, legend=False,
                ax=ax, width=0.55, linewidth=0.8, fliersize=0)
    for patch in ax.patches:
        patch.set_alpha(0.75)

    # Per-group counts
    n_per_group = data.groupby(x_col)[y_col].count()
    tick_labels = list(order) if order is not None else list(n_per_group.index)
    n_map = {lbl: int(n_per_group.get(lbl, 0)) for lbl in tick_labels}
    if count_label == "per_group":
        n_labels = [f"{lbl}\n(N={n_map[lbl]})" for lbl in tick_labels]
    else:
        n_labels = tick_labels
    ax.set_xticks(range(len(tick_labels)))
    ax.set_xticklabels(n_labels,
                       rotation=rotation,
                       ha="right" if rotation else "center")

    if count_label == "corner" and n_map:
        _add_top_left_count_label(ax, _format_count_label(list(n_map.values())))
    elif count_label == "bottom_right" and n_map:
        _add_bottom_right_count_label(ax, _format_count_label(list(n_map.values())))
    elif count_label == "title" and (n_map or n_fish):
        add_right_count_title(
            ax,
            _format_count_label(list(n_map.values())) if n_map else None,
            n_fish=n_fish,
        )

    # Conditional stripplot
    if max(n_map.values(), default=0) < PLOT_POINTS_BELOW_N:
        sns.stripplot(data=data, x=x_col, y=y_col, order=order, ax=ax,
                      color="black", alpha=0.5, jitter=0.2, size=3)

    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    sns.despine(ax=ax)
    return n_map


# ── re-exported helpers (moved to dedicated modules) ─────────────────────────
# Kept here for backward compatibility with existing
# `from analysis_style import ...` call sites.  Only light-weight homes are
# re-exported; svd_evr/SVD_BACKEND live in utils_rnn and are imported directly
# by their consumers to avoid pulling its heavy deps into this styling module.
from cfg import TIME_STEP_MS  # noqa: E402,F401
from utils_features import (  # noqa: E402,F401
    add_size_advantage,
    compute_wall_distance,
    compute_idis,
)
from utils_event_times import (  # noqa: E402,F401
    consumption_times_from_step_df,
    biting_times_from_step_df,
    load_consumption_times,
    load_biting_times,
)


def _time_to_x_impl(condition_times, palette, out_path,
                    log_scale, show_traces, xlabel, ylabel,
                    min_episodes=3, alpha_trace=0.18, count_label="none"):
    """X = nth event, Y = time to reach it (mean ± SEM across episodes)."""
    set_style()
    fig, ax = plt.subplots(figsize=(2.0, 2.0))

    for label, ep_arrays in condition_times:
        color = palette.get(label, "#888888")
        if not ep_arrays:
            continue
        max_n = max(len(a) for a in ep_arrays)

        if show_traces:
            for arr in ep_arrays:
                ax.plot(np.arange(1, len(arr) + 1), arr,
                        color=color, lw=0.5, alpha=alpha_trace)

        xs_mean, ys_mean, ys_sem = [], [], []
        for n in range(1, max_n + 1):
            vals = [a[n - 1] for a in ep_arrays if len(a) >= n]
            if len(vals) < min_episodes:
                continue
            xs_mean.append(n)
            ys_mean.append(np.mean(vals))
            ys_sem.append(np.std(vals) / np.sqrt(len(vals)))

        if xs_mean:
            xs_mean = np.array(xs_mean)
            ys_mean = np.array(ys_mean)
            ys_sem  = np.array(ys_sem)
            ax.fill_between(xs_mean, ys_mean - ys_sem, ys_mean + ys_sem,
                            color=color, alpha=0.25)
            ax.plot(xs_mean, ys_mean, color=color, lw=2.0,
                    marker="o" if show_traces else None, ms=4, label=label)

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if count_label == "corner":
        _add_top_left_count_label(
            ax,
            _format_count_label([len(ep_arrays) for _, ep_arrays in condition_times]),
        )
    elif count_label == "bottom_right":
        _add_bottom_right_count_label(
            ax,
            _format_count_label([len(ep_arrays) for _, ep_arrays in condition_times]),
        )
    if len(condition_times) > 1:
        legend_loc = "upper right" if count_label in {"corner", "bottom_right"} else "upper left"
        ax.legend(frameon=False, fontsize=6, handlelength=1.5, loc=legend_loc)
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, out_path)


def _x_timecourse_impl(condition_times, palette, out_path,
                       log_scale, show_traces, xlabel, ylabel,
                       min_episodes=3, alpha_trace=0.18, count_label="none",
                       include_zeros=False):
    """X = time, Y = cumulative event count (mean ± SEM). log_scale applies to X axis.

    include_zeros: if True, episodes with no events contribute a flat zero line to
    the mean/SEM (N in legend = all episodes). If False (default), only episodes
    with ≥1 event are used.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(2.0, 2.0))

    for label, ep_arrays in condition_times:
        n_obs = len(ep_arrays)
        color = palette.get(label, "#888888")
        nonempty = [a for a in ep_arrays if len(a) > 0]
        if len(nonempty) < min_episodes:
            continue
        max_t = max(a[-1] for a in nonempty)
        if log_scale:
            min_t = min(a[0] for a in nonempty)
            t_grid = np.logspace(np.log10(max(min_t * 0.5, 1e-3)),
                                 np.log10(max_t * 1.05), 300)
        else:
            t_grid = np.linspace(0, max_t * 1.05, 300)

        arrays_for_counts = ep_arrays if include_zeros else nonempty
        counts = np.array(
            [np.searchsorted(a, t_grid, side="right") for a in arrays_for_counts],
            dtype=float,
        )

        if show_traces:
            for cc in counts:
                ax.plot(t_grid, cc, color=color, lw=0.5, alpha=alpha_trace)

        mean_c = counts.mean(axis=0)
        sem_c  = counts.std(axis=0) / np.sqrt(len(counts))
        ax.fill_between(t_grid, np.clip(mean_c - sem_c, 0, None), mean_c + sem_c,
                        color=color, alpha=0.25)
        ax.plot(t_grid, mean_c, color=color, lw=2.0, label=f"{label} (N={n_obs})")

    if log_scale:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if count_label == "corner":
        _add_top_left_count_label(
            ax,
            _format_count_label([len(ep_arrays) for _, ep_arrays in condition_times]),
        )
    elif count_label == "bottom_right":
        _add_bottom_right_count_label(
            ax,
            _format_count_label([len(ep_arrays) for _, ep_arrays in condition_times]),
        )
    if len(condition_times) > 1:
        ax.legend(frameon=False, fontsize=6, handlelength=1.5, loc="upper left")
    sns.despine(ax=ax)
    plt.tight_layout(pad=0.3)
    save(fig, out_path)


def time_to_X_log(condition_times, palette, out_path,
                   event_label="Events", time_label="Time (s)",
                   min_episodes=3, alpha_trace=0.18, count_label="none"):
    """nth-event vs time-to-reach-it, log Y, with per-episode traces."""
    _time_to_x_impl(condition_times, palette, out_path,
                    log_scale=True, show_traces=True,
                    xlabel=event_label, ylabel=time_label,
                    min_episodes=min_episodes, alpha_trace=alpha_trace,
                    count_label=count_label)


def time_to_X_linear(condition_times, palette, out_path,
                      event_label="Events", time_label="Time (s)",
                      min_episodes=3, alpha_trace=0.18, count_label="none"):
    """nth-event vs time-to-reach-it, linear Y, mean ± SEM only."""
    _time_to_x_impl(condition_times, palette, out_path,
                    log_scale=False, show_traces=False,
                    xlabel=event_label, ylabel=time_label,
                    min_episodes=min_episodes, alpha_trace=alpha_trace,
                    count_label=count_label)


def X_timecourse_log(condition_times, palette, out_path,
                      event_label="Events", time_label="Time (s)",
                      min_episodes=3, alpha_trace=0.18, count_label="none"):
    """Cumulative event count over time, log X, with per-episode traces."""
    _x_timecourse_impl(condition_times, palette, out_path,
                       log_scale=True, show_traces=True,
                       xlabel=time_label, ylabel=event_label,
                       min_episodes=min_episodes, alpha_trace=alpha_trace,
                       count_label=count_label)


def X_timecourse_linear(condition_times, palette, out_path,
                         event_label="Events", time_label="Time (s)",
                         min_episodes=3, alpha_trace=0.18, count_label="none",
                         include_zeros=False):
    """Cumulative event count over time, linear X, mean ± SEM only."""
    _x_timecourse_impl(condition_times, palette, out_path,
                       log_scale=False, show_traces=False,
                       xlabel=time_label, ylabel=event_label,
                       min_episodes=min_episodes, alpha_trace=alpha_trace,
                       count_label=count_label, include_zeros=include_zeros)


def save_event_time_plot_set(condition_times, palette, out_dir, stem,
                              event_label="Food consumed",
                              time_label="Time (s)",
                              count_label="none"):
    """Save all four plot variants for one event type."""
    if len(condition_times) < 2:
        return
    time_to_X_log(
        condition_times, palette, os.path.join(out_dir, f"{stem}_log.pdf"),
        event_label=event_label, time_label=time_label,
        count_label=count_label)
    time_to_X_linear(
        condition_times, palette, os.path.join(out_dir, f"{stem}_linear.pdf"),
        event_label=event_label, time_label=time_label,
        count_label=count_label)
    X_timecourse_log(
        condition_times, palette, os.path.join(out_dir, f"{stem}_timecourse_log.pdf"),
        event_label=event_label, time_label=time_label,
        count_label=count_label)
    X_timecourse_linear(
        condition_times, palette, os.path.join(out_dir, f"{stem}_timecourse_linear.pdf"),
        event_label=event_label, time_label=time_label,
        count_label=count_label)
