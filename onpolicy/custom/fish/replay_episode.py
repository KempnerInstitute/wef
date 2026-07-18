#!/usr/bin/env python3
"""
replay_episode.py — regenerate an MP4 or GIF animation from saved eval data.

Reads from raw/agg_flat.pkl and raw/ep{N}_arena.pkl; no live environment
or policy needed.  Electric-field streamlines are skipped (not recorded).

Usage (run from onpolicy/custom/fish/):
    python replay_episode.py EVAL_SPEC_DIR
    python replay_episode.py EVAL_SPEC_DIR --env-id 0 --episode 0
    python replay_episode.py EVAL_SPEC_DIR --env-id 2 --episode 1 --t-start 200 --t-end 600
    python replay_episode.py EVAL_SPEC_DIR --output clip.mp4
    python replay_episode.py EVAL_SPEC_DIR --output clip.gif --compression 3
    python replay_episode.py EVAL_SPEC_DIR --gif --fps 15 --compression 2
    python replay_episode.py EVAL_SPEC_DIR --rnn-pca
    python replay_episode.py EVAL_SPEC_DIR --rnn-pca --rnn-pcs 3

Output format: inferred from --output extension (.mp4 / .gif), or use --gif flag.
--compression 1..5 controls GIF palette depth (1=256 colours, 5=16 colours).
--rnn-pca adds a 2×2 grid of per-agent RNN PC trajectories (right panel).

EVAL_SPEC_DIR is the path to evals/{spec_key}/, e.g.
    results/K2_Enum/K2_EnumSeed1/20260611_095911/evals/m1a1k1_patchy_square/
"""

import argparse
import json
import os
import sys

import imageio
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cfg import ENV_PARAMS, AGENT_PARAMS
from renderer import FishRenderer


# ------------------------------------------------------------------ #
# Stub classes that satisfy the FishRenderer interface
# ------------------------------------------------------------------ #

class _AgentStub:
    """Minimal stand-in for EFishAgent, populated from recorded data."""
    def __init__(self, agent_id, arena_size):
        self.agent_id = agent_id
        self.active = True
        self.position = np.array([arena_size[0] / 2.0, arena_size[1] / 2.0])
        self.orientation = 0.0
        self.energy = 50.0
        self.cumulative_reward = 0.0
        self.agent_size = 0.5
        self.was_bitten = False
        self.bite_other_fish = False
        self.emit_eod = False
        self.asym_eating = True
        self.body_radius = AGENT_PARAMS["body_radius_cm"]
        self.eating_radius = AGENT_PARAMS["eating_radius_cm"]
        self.eating_angle = AGENT_PARAMS["eating_angle"]
        self.morm_food_detection_range = AGENT_PARAMS["morm_food_detection_range_m"]
        self.bounding_radius = 0.0
        self.trajectory = []
        self.eod_history = []
        self.was_bitten_history = []
        self.bite_history = []
        self.energy_history = []


class _ArenaStub:
    def __init__(self):
        self.food_positions = np.empty((0, 2))
        self.food_radius = ENV_PARAMS["food_radius_cm"]


class _EnvStub:
    """Minimal stand-in for MultiAgentFishEnv, populated from recorded data."""
    def __init__(self, arena_size, num_agents, task="foraging", homing_mode=False):
        self.arena_size = list(arena_size)
        self.num_agents = num_agents
        self.curr_time = 0
        self.step_count = 0
        self.homing_mode = homing_mode
        self.homing_distance = 5.0
        self.task = task
        self.active_agent_ids = list(range(num_agents))
        self.all_args = {}
        self.save_vid = False
        self.np_random = np.random.default_rng()
        self.arena = _ArenaStub()
        self.agent_objects = [_AgentStub(i, arena_size) for i in range(num_agents)]


class _ReplayRenderer(FishRenderer):
    """FishRenderer with electric-field overlay disabled (field not recorded)."""
    def _plot_electric_field(self, ax):
        pass


# ------------------------------------------------------------------ #
# RNN PCA renderer
# ------------------------------------------------------------------ #

class _RNNPCARenderer(_ReplayRenderer):
    """
    Extends _ReplayRenderer with a right-side 2×2 panel showing per-agent
    RNN trajectories in shared PC space.

    Aesthetic matches archive/anim_rnn.py / utils_rnn.animate_2d:
      - Static grey scatter cloud of ALL states as background context
      - Animated 10-step coloured trace (recent window)
      - Animated current-position dot

    Uses set_data / set_data_3d per frame — no ax.clear().
    """
    _TRACE_LEN = 10  # frames in the moving tail, matching legacy code

    def __init__(self, env, rnn_projections, var_explained, n_pcs):
        super().__init__(env)
        self.rnn_projections = rnn_projections   # list of (T, n_pcs) arrays
        self.var_explained   = var_explained
        self.n_pcs           = n_pcs
        self._rnn_axes       = None
        self._traces         = []   # animated Line2D / Line3D for recent trail
        self._dots           = []   # animated Line2D / Line3D for current dot
        self._frame_idx      = 0
        self._n_rnn_agents   = len(rnn_projections)

    def _init_layout(self):
        """Build figure: arena+eod left, 2×2 PC panels right."""
        n_agents    = self._n_rnn_agents
        n_rnn_cols  = 2
        n_rnn_rows  = (n_agents + 1) // 2
        use_3d      = (self.n_pcs == 3)

        dpi = ENV_PARAMS.get("render_dpi", 72)
        fig = plt.figure(figsize=(14, 8), dpi=dpi)
        fig.subplots_adjust(left=0.05, right=0.97, top=0.97, bottom=0.06)

        gs_outer = gridspec.GridSpec(
            1, 2, figure=fig, width_ratios=[1.0, 0.7], wspace=0.06,
        )
        gs_left = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs_outer[0, 0], height_ratios=[4, 1], hspace=0.08,
        )
        ax_arena = fig.add_subplot(gs_left[0])
        ax_eod   = fig.add_subplot(gs_left[1])

        gs_right = gridspec.GridSpecFromSubplotSpec(
            n_rnn_rows, n_rnn_cols, subplot_spec=gs_outer[0, 1],
            wspace=0.35, hspace=0.38,
        )

        rnn_axes = []
        for i in range(n_agents):
            kw = {"projection": "3d"} if use_3d else {}
            ax = fig.add_subplot(gs_right[i // n_rnn_cols, i % n_rnn_cols], **kw)
            rnn_axes.append(ax)

        ve = self.var_explained * 100
        for i, ax in enumerate(rnn_axes):
            proj = self.rnn_projections[i]   # (T, n_pcs)

            # --- static grey cloud of all states ---
            if use_3d:
                ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2],
                           color="grey", alpha=0.5, s=2)
            else:
                ax.scatter(proj[:, 0], proj[:, 1],
                           color="grey", alpha=0.5, s=2)

            # --- animated artists (empty at t=0) ---
            color = f"C{i}"
            if use_3d:
                (trace,) = ax.plot([], [], [], color=color, linewidth=1)
                (dot,)   = ax.plot([], [], [], f"C{i}o", markersize=4)
            else:
                (trace,) = ax.plot([], [], color=color, linewidth=1)
                (dot,)   = ax.plot([], [], f"C{i}o", markersize=4)
            self._traces.append(trace)
            self._dots.append(dot)

            # --- static decorations ---
            ax.set_title(f"Agent {i}", fontsize=10, color=color)
            if use_3d:
                ax.set_xlabel(f"PC1 ({ve[0]:.2f}% var)", fontsize=7, labelpad=2)
                ax.set_ylabel(f"PC2 ({ve[1]:.2f}% var)", fontsize=7, labelpad=2)
                ax.set_zlabel(f"PC3 ({ve[2]:.2f}% var)", fontsize=7, labelpad=2)
                ax.tick_params(labelsize=5)
            else:
                ax.set_xlabel(f"PC1 ({ve[0]:.2f}% var)", fontsize=7)
                ax.set_ylabel(f"PC2 ({ve[1]:.2f}% var)", fontsize=7)
                ax.tick_params(labelsize=6)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

            # Fix axis limits from full data range (with 10% padding)
            pad = 0.1
            for pc_i, setter in enumerate(
                ([ax.set_xlim, ax.set_ylim] if not use_3d
                 else [ax.set_xlim, ax.set_ylim, ax.set_zlim])
            ):
                col = proj[:, pc_i]
                lo, hi = col.min(), col.max()
                span = max(hi - lo, 1e-6)
                setter(lo - pad * span, hi + pad * span)

            # Force physically square axes box regardless of data ranges
            ax.set_box_aspect(1 if not use_3d else [1, 1, 1])

        # Wire into FishRenderer
        self.fig      = fig
        self.axes     = np.array([ax_arena, ax_eod])
        self.ax1      = ax_arena
        self.aux_axes = np.array([ax_eod])
        self._rnn_axes = rnn_axes

    def _update_pca_artists(self):
        t     = self._frame_idx
        use3d = (self.n_pcs == 3)
        for i, (trace, dot) in enumerate(zip(self._traces, self._dots)):
            proj  = self.rnn_projections[i]
            t_now = min(t + 1, len(proj))
            if t_now == 0:
                continue
            w0 = max(0, t_now - self._TRACE_LEN)
            x  = proj[w0:t_now, 0]
            y  = proj[w0:t_now, 1]
            cx = proj[t_now - 1, 0]
            cy = proj[t_now - 1, 1]
            if use3d:
                z  = proj[w0:t_now, 2]
                cz = proj[t_now - 1, 2]
                trace.set_data(x, y);   trace.set_3d_properties(z)
                dot.set_data([cx], [cy]); dot.set_3d_properties([cz])
            else:
                trace.set_data(x, y)
                dot.set_data([cx], [cy])

    def render(self, mode=None, auxs=None, attentions=None):
        if self.fig is None:
            self._init_layout()

        # Draw arena + EOD into the pre-created axes
        super().render(mode=None, auxs=auxs, attentions=attentions)

        self._update_pca_artists()
        self._frame_idx += 1

        if mode == "rgb_array":
            self.fig.canvas.draw()
            w, h = self.fig.canvas.get_width_height()
            return np.frombuffer(
                self.fig.canvas.buffer_rgba(), dtype=np.uint8
            ).reshape(h, w, 4)[..., :3]


# ------------------------------------------------------------------ #
# Data loading
# ------------------------------------------------------------------ #

def _load_raw(raw_dir, episode_idx):
    flat_path  = os.path.join(raw_dir, "agg_flat.pkl")
    arena_path = os.path.join(raw_dir, f"ep{episode_idx}_arena.pkl")
    eps_path   = os.path.join(raw_dir, f"ep{episode_idx}_episodes.json")
    missing = [p for p in (flat_path, arena_path, eps_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing files in raw/:\n" + "\n".join(f"  {p}" for p in missing))
    bdf = pd.read_pickle(flat_path)
    adf = pd.read_pickle(arena_path)
    with open(eps_path) as f:
        eps = json.load(f)
    return bdf, adf, eps


def _episode_meta(eps, env_id, episode_idx):
    for e in eps:
        if e["env_id"] == env_id and e["episode_index"] == episode_idx:
            return e
    raise KeyError(
        f"No episode metadata for env_id={env_id}, episode_index={episode_idx}. "
        f"Available: {[(e['env_id'], e['episode_index']) for e in eps]}"
    )


def _load_rnn_projections(raw_dir, episode_idx, env_id, t_start, t_end, n_pcs):
    """
    Load ep{episode_idx}_rnn.npy, extract (env_id) slice, fit common PCA
    across all agents, return (projections, var_explained).

    rnn.npy shape: (T, n_envs, n_agents, recurrent_N, hidden_size)
    Returns:
        projections  : list of n_agents arrays, each shape (T_window, n_pcs)
        var_explained: array of shape (n_pcs,)
    """
    from sklearn.decomposition import PCA

    rnn_path = os.path.join(raw_dir, f"ep{episode_idx}_rnn.npy")
    if not os.path.exists(rnn_path):
        raise FileNotFoundError(f"RNN file not found: {rnn_path}\n"
                                 "Re-run eval with save_rnn=True in the EvalSpec.")

    rnn = np.load(rnn_path)  # (T, n_envs, n_agents, recurrent_N, hidden_size)
    # Extract this env, collapse recurrent_N layer (index 0 = first/only layer)
    rnn_ep = rnn[:, env_id, :, 0, :].astype(np.float32)  # (T, n_agents, hidden_size)

    # Slice to replay window
    rnn_ep = rnn_ep[t_start : t_end + 1]  # (T_window, n_agents, hidden_size)
    T, n_agents, H = rnn_ep.shape

    # Fit PCA on combined data (all agents stacked)
    combined = rnn_ep.reshape(T * n_agents, H)
    pca = PCA(n_components=n_pcs)
    pca.fit(combined)
    print(f"  PCA variance explained: {pca.explained_variance_ratio_ * 100}")

    projections = [pca.transform(rnn_ep[:, a, :]) for a in range(n_agents)]
    return projections, pca.explained_variance_ratio_


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("eval_spec_dir", help="Path to evals/{spec_key}/ directory")
    ap.add_argument("--env-id",  type=int, default=0)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--t-start", type=int, default=None)
    ap.add_argument("--t-end",   type=int, default=None)
    ap.add_argument("--output",  type=str, default=None,
                    help="Output path (.mp4 or .gif). Default: <eval_spec_dir>/replay/env<N>_ep<N>.mp4")
    ap.add_argument("--gif", action="store_true",
                    help="Output as GIF (also inferred from .gif extension in --output)")
    ap.add_argument("--compression", type=int, default=2, choices=range(1, 6), metavar="1-5",
                    help="GIF palette depth: 1=256 colours (best), 5=16 colours (smallest). Default 2.")
    ap.add_argument("--fps", type=int, default=None,
                    help=f"Output FPS (default: {ENV_PARAMS['fps_video']} for MP4, 15 for GIF)")
    ap.add_argument("--rnn-pca", action="store_true",
                    help="Overlay per-agent RNN trajectories in shared PC space (right panel)")
    ap.add_argument("--rnn-pcs", type=int, default=2, choices=[2, 3],
                    help="Number of PCs to extract (2 or 3). With 3, PC3 is encoded as colour. Default 2.")
    ap.add_argument("--no-homing-clip", action="store_true",
                    help="For homing tasks, disable automatic clip at first homing success.")
    args = ap.parse_args(argv)

    eval_spec_dir = os.path.abspath(args.eval_spec_dir)
    raw_dir = os.path.join(eval_spec_dir, "raw")

    print(f"Loading data from {raw_dir} ...")
    bdf, adf, eps = _load_raw(raw_dir, args.episode)

    ep_meta    = _episode_meta(eps, args.env_id, args.episode)
    arena_size = ep_meta["arena_size"]

    b = bdf[(bdf["env_id"] == args.env_id) & (bdf["episode_index"] == args.episode)].copy()
    a = adf[(adf["env_id"] == args.env_id) & (adf["episode_index"] == args.episode)].copy()

    if len(b) == 0:
        raise RuntimeError(
            f"No behavior rows for env_id={args.env_id}, episode={args.episode}.\n"
            f"Available: {sorted(bdf[['env_id','episode_index']].drop_duplicates().itertuples(index=False, name=None))}"
        )

    num_agents    = b["agent_id"].nunique()
    all_timesteps = sorted(b["time_step"].unique())
    t_start = args.t_start if args.t_start is not None else all_timesteps[0]
    t_end   = args.t_end   if args.t_end   is not None else all_timesteps[-1]
    timesteps = [t for t in all_timesteps if t_start <= t <= t_end]

    if not timesteps:
        raise RuntimeError(f"No timesteps in range [{t_start}, {t_end}]")

    spec_key = os.path.basename(eval_spec_dir)
    task = "1rw1f1p" if "1rw1f1p" in spec_key else ("homing" if "homing" in spec_key else "foraging")

    # For homing replays, clip at the end of the first successful trial.
    # trial_start=True marks the first step of each new trial (fires after the env
    # auto-resets on homing success).  The last frame of the successful trial is
    # therefore (next_trial_start_t - 1).
    if task == "homing" and not args.no_homing_clip and args.t_end is None:
        homer_rows = b[b["agent_id"] == 1].copy()
        next_trial = homer_rows.loc[
            homer_rows["trial_start"] & (homer_rows["time_step"] > t_start),
            "time_step",
        ]
        if len(next_trial) > 0:
            reset_t = int(next_trial.min())
            t_end = reset_t - 1
            timesteps = [t for t in timesteps if t <= t_end]
            print(f"  Homing success: next trial_start at t={reset_t}; clipping to t=[{t_start}..{t_end}] ({len(timesteps)} steps)")
        else:
            print("  No homing success detected in episode; playing full episode")

    print(
        f"Replay: spec={spec_key!r}  env_id={args.env_id}  episode={args.episode}  "
        f"t=[{t_start}..{t_end}] ({len(timesteps)} steps)  agents={num_agents}"
    )

    env = _EnvStub(arena_size=arena_size, num_agents=num_agents,
                   task=task, homing_mode=(task == "homing"))
    active_ids = ep_meta.get("active_agent_ids")
    if active_ids is not None:
        env.active_agent_ids = active_ids

    # Optionally load RNN PCA projections
    rnn_projections = var_explained = None
    if args.rnn_pca:
        print(f"Computing RNN PCA ({args.rnn_pcs} PCs) ...")
        rnn_projections, var_explained = _load_rnn_projections(
            raw_dir, args.episode, args.env_id, t_start, t_end, args.rnn_pcs
        )

    b_by_t    = {t: grp for t, grp in b.groupby("time_step")}
    a_indexed = a.set_index("time_step")

    use_gif = args.gif or (args.output and args.output.lower().endswith(".gif"))
    if args.output:
        out_path = args.output
    else:
        replay_dir = os.path.join(eval_spec_dir, "replay")
        os.makedirs(replay_dir, exist_ok=True)
        t_suffix   = f"_t{t_start}-{t_end}" if (args.t_start or args.t_end) else ""
        rnn_suffix = f"_pca{args.rnn_pcs}" if args.rnn_pca else ""
        out_path = os.path.join(
            replay_dir,
            f"env{args.env_id}_ep{args.episode}{t_suffix}{rnn_suffix}.{'gif' if use_gif else 'mp4'}"
        )

    fps        = args.fps or (15 if use_gif else ENV_PARAMS["fps_video"])
    palettesize = max(16, 256 >> (args.compression - 1))

    if args.rnn_pca:
        renderer = _RNNPCARenderer(env, rnn_projections, var_explained, args.rnn_pcs)
    else:
        renderer = _ReplayRenderer(env)

    gif_frames: list = []
    video_writer = None if use_gif else imageio.get_writer(out_path, format="ffmpeg", mode="I", fps=fps)

    fmt_label = f"GIF compression={args.compression} ({palettesize} colours)" if use_gif else "MP4"
    print(f"Writing {out_path} @ {fps} fps [{fmt_label}] ...")

    try:
        for t in timesteps:
            brows = b_by_t.get(t)
            if brows is None:
                continue

            env.curr_time  = t
            env.step_count = t

            if t in a_indexed.index:
                arow = a_indexed.loc[t]
                fp = arow["food_positions"]
                env.arena.food_positions = (
                    np.array(fp) if fp is not None and len(fp) > 0 else np.empty((0, 2))
                )
                ai = arow.get("active_agent_ids")
                if ai is not None:
                    env.active_agent_ids = list(ai)

            for _, row in brows.iterrows():
                agent = env.agent_objects[int(row["agent_id"])]
                pos   = np.asarray(row["position"], dtype=float)
                agent.position       = pos
                agent.orientation    = float(row["orientation"])
                agent.energy         = float(row["energy"])
                agent.agent_size     = float(row["agent_size"])
                agent.active         = bool(row["active"])
                agent.was_bitten     = bool(row["was_bitten"])
                agent.bite_other_fish = bool(row["bite_other_fish"])
                agent.emit_eod       = bool(row["emit_eod"])
                agent.cumulative_reward += float(row["rewards"])
                agent.trajectory.append(pos.copy())
                agent.eod_history.append(1.0 if agent.emit_eod else 0.0)
                agent.was_bitten_history.append(1.0 if agent.was_bitten else 0.0)
                agent.bite_history.append(1.0 if agent.bite_other_fish else 0.0)
                agent.energy_history.append(agent.energy)

            frame = renderer.render(mode="rgb_array", auxs=["eods"])
            if use_gif:
                gif_frames.append(frame)
            else:
                video_writer.append_data(frame)

    finally:
        if video_writer is not None:
            video_writer.close()
        renderer.close()

    if use_gif:
        print(f"Encoding GIF ({len(gif_frames)} frames, palettesize={palettesize}) ...")
        imageio.mimwrite(out_path, gif_frames, format="GIF", fps=fps, loop=0,
                         palettesize=palettesize, quantizer="nq")

    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    main()
