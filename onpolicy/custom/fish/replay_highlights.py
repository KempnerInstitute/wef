#!/usr/bin/env python3
"""
replay_highlights.py — find and clip interesting behavioural events from eval data.

Scans an eval spec folder, detects events matching a named scenario, then calls
replay_episode for each of the top N matches.

Usage (from onpolicy/custom/fish/):
    python replay_highlights.py EVAL_SPEC_DIR --scenario chasing
    python replay_highlights.py EVAL_SPEC_DIR --scenario chasing --n-clips 5 --window 150
    python replay_highlights.py EVAL_SPEC_DIR --scenario size_reversal --gif
    python replay_highlights.py EVAL_SPEC_DIR --scenario eating --n-clips 3 --gif --compression 3

Scenarios
---------
  chasing         — one fish oriented toward and closing on another (sustained ≥20 steps)
  eating          — any eating event (optionally under social pressure)
  coordination    — 2+ fish eating within a 40-step window of each other
  size_reversal   — smaller fish eats more food than larger fish in the same episode (2f1p)
  high_eod        — sustained burst of above-average EOD rate by any agent

Output lands in <eval_spec_dir>/highlights/{scenario}/ as clip_0.mp4 (or .gif).
"""

import argparse
import json
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

import replay_episode as _re


# ------------------------------------------------------------------ #
# Shared data loader
# ------------------------------------------------------------------ #

def _load_all(raw_dir):
    """Return (behavior_df, arena_df, episodes_list) merging all batches."""
    flat_path = os.path.join(raw_dir, "agg_flat.pkl")
    if not os.path.exists(flat_path):
        raise FileNotFoundError(f"agg_flat.pkl not found in {raw_dir}")
    bdf = pd.read_pickle(flat_path)

    arena_files = sorted([
        f for f in os.listdir(raw_dir) if f.startswith("ep") and f.endswith("_arena.pkl")
    ])
    if not arena_files:
        raise FileNotFoundError(f"No ep*_arena.pkl files in {raw_dir}")
    adfs = []
    all_eps = []
    for af in arena_files:
        ep_idx = int(af.split("_")[0][2:])
        adfs.append(pd.read_pickle(os.path.join(raw_dir, af)))
        eps_path = os.path.join(raw_dir, af.replace("_arena.pkl", "_episodes.json"))
        if os.path.exists(eps_path):
            with open(eps_path) as f:
                all_eps.extend(json.load(f))
    adf = pd.concat(adfs, ignore_index=True)
    return bdf, adf, all_eps


# ------------------------------------------------------------------ #
# Position helpers
# ------------------------------------------------------------------ #

def _positions_wide(ep_df):
    """
    Return dict mapping timestep → np.array of shape (n_agents, 2).
    Only agents present at that timestep are included.
    """
    result = {}
    for t, grp in ep_df.groupby("time_step"):
        rows = grp.sort_values("agent_id")
        result[t] = {int(r["agent_id"]): np.asarray(r["position"]) for _, r in rows.iterrows()}
    return result


def _bearing_angle(from_pos, to_pos):
    """Angle of (to_pos - from_pos) in radians."""
    d = to_pos - from_pos
    return np.arctan2(d[1], d[0])


def _angle_diff(a, b):
    """Signed angular difference a-b, wrapped to (-π, π]."""
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return d


# ------------------------------------------------------------------ #
# Scenario detectors
# Each returns a list of dicts:
#   {env_id, episode, t_center, score, description}
# sorted by score descending.
# ------------------------------------------------------------------ #

def _detect_chasing(bdf, min_duration=20, dist_thresh=14.0, angle_thresh=np.pi / 3):
    """
    Fish i chasing fish j: close together, i oriented toward j, sustained.
    Returns events keyed by the best sustained run length.
    """
    events = []
    for (env_id, ep), ep_df in bdf.groupby(["env_id", "episode_index"]):
        agent_ids = sorted(ep_df["agent_id"].unique())
        if len(agent_ids) < 2:
            continue

        # Build (timestep → {agent_id: position, orientation})
        t_data = {}
        for _, row in ep_df.iterrows():
            t = int(row["time_step"])
            if t not in t_data:
                t_data[t] = {}
            t_data[t][int(row["agent_id"])] = {
                "pos": np.asarray(row["position"]),
                "ori": float(row["orientation"]),
            }

        # For each ordered pair (pursuer, target), find runs
        for pursuer, target in [(a, b) for a in agent_ids for b in agent_ids if a != b]:
            run_start = None
            run_len = 0
            timesteps = sorted(t_data.keys())
            for t in timesteps:
                if pursuer not in t_data[t] or target not in t_data[t]:
                    run_start = None
                    run_len = 0
                    continue
                pos_p = t_data[t][pursuer]["pos"]
                pos_t = t_data[t][target]["pos"]
                dist = float(np.linalg.norm(pos_p - pos_t))
                bearing = _bearing_angle(pos_p, pos_t)
                angle_err = abs(_angle_diff(t_data[t][pursuer]["ori"], bearing))

                if dist < dist_thresh and angle_err < angle_thresh:
                    if run_start is None:
                        run_start = t
                    run_len += 1
                else:
                    if run_len >= min_duration:
                        t_center = run_start + run_len // 2
                        events.append({
                            "env_id": env_id, "episode": ep,
                            "t_center": t_center, "score": run_len,
                            "description": f"agent{pursuer}→agent{target} chasing ({run_len} steps)",
                        })
                    run_start = None
                    run_len = 0
            # flush last run
            if run_len >= min_duration and run_start is not None:
                t_center = run_start + run_len // 2
                events.append({
                    "env_id": env_id, "episode": ep,
                    "t_center": t_center, "score": run_len,
                    "description": f"agent{pursuer}→agent{target} chasing ({run_len} steps)",
                })

    events.sort(key=lambda x: -x["score"])
    return events


def _detect_eating(bdf, under_pressure=False):
    """
    Eating events, optionally filtered to only those where the eating agent
    was also nearby another agent (has_nearby) or being bitten.
    """
    events = []
    for (env_id, ep), ep_df in bdf.groupby(["env_id", "episode_index"]):
        eat = ep_df[ep_df["eating_event"] == True]
        if under_pressure:
            eat = eat[(eat["has_nearby"] == True) | (eat["was_bitten"] == True)]
        for _, row in eat.iterrows():
            events.append({
                "env_id": env_id, "episode": ep,
                "t_center": int(row["time_step"]),
                "score": 1.0,
                "description": f"agent{int(row['agent_id'])} eats at t={int(row['time_step'])}",
            })
    # Deduplicate: keep events >50 steps apart per (env, ep)
    events.sort(key=lambda x: (x["env_id"], x["episode"], x["t_center"]))
    deduped = []
    last_key = {}
    for e in events:
        key = (e["env_id"], e["episode"])
        if key not in last_key or e["t_center"] - last_key[key] > 50:
            deduped.append(e)
            last_key[key] = e["t_center"]
    return deduped


def _detect_coordination(bdf, window=40, min_agents=2):
    """
    Multiple agents eating within a short time window.
    Score = number of distinct agents eating within ±window steps of t_center.
    """
    events = []
    for (env_id, ep), ep_df in bdf.groupby(["env_id", "episode_index"]):
        eat = ep_df[ep_df["eating_event"] == True][["time_step", "agent_id"]].drop_duplicates()
        if len(eat) == 0:
            continue
        eat_times = sorted(eat["time_step"].unique())
        for t in eat_times:
            nearby = eat[
                (eat["time_step"] >= t - window) & (eat["time_step"] <= t + window)
            ]
            n_agents = nearby["agent_id"].nunique()
            if n_agents >= min_agents:
                events.append({
                    "env_id": env_id, "episode": ep,
                    "t_center": int(t), "score": float(n_agents),
                    "description": f"{n_agents} agents eat within {window}-step window of t={t}",
                })
    # Deduplicate: keep best event per (env, ep, coarse_t)
    events.sort(key=lambda x: -x["score"])
    deduped = []
    seen = set()
    for e in events:
        key = (e["env_id"], e["episode"], e["t_center"] // 60)
        if key not in seen:
            deduped.append(e)
            seen.add(key)
    deduped.sort(key=lambda x: -x["score"])
    return deduped


def _detect_size_reversal(bdf, all_eps):
    """
    Episodes where the *smaller* agent accumulates more food than the larger one.
    Works on any spec with 2 agents; particularly meaningful for 2f1p_AgtB specs.
    Score = food_eaten(small) - food_eaten(large).
    Returns one entry per matching (env_id, episode); t_center is last eating event.
    """
    events = []
    for (env_id, ep), ep_df in bdf.groupby(["env_id", "episode_index"]):
        agent_ids = sorted(ep_df["agent_id"].unique())
        if len(agent_ids) != 2:
            continue

        # Agent sizes — use first timestep value (fixed per episode)
        first_t = ep_df["time_step"].min()
        sizes = {}
        food = {}
        for aid in agent_ids:
            ag = ep_df[ep_df["agent_id"] == aid]
            sizes[aid] = float(ag[ag["time_step"] == first_t]["agent_size"].iloc[0])
            food[aid] = int(ag["food_eaten_count"].sum())

        large = max(agent_ids, key=lambda a: sizes[a])
        small = min(agent_ids, key=lambda a: sizes[a])

        if sizes[large] == sizes[small]:
            continue  # equal size, skip

        diff = food[small] - food[large]
        if diff > 0:
            # Find the last eating event by the small agent
            small_eats = ep_df[
                (ep_df["agent_id"] == small) & (ep_df["eating_event"] == True)
            ]["time_step"]
            t_center = int(small_eats.max()) if len(small_eats) > 0 else int(
                ep_df["time_step"].max()
            )
            events.append({
                "env_id": env_id, "episode": ep,
                "t_center": t_center, "score": float(diff),
                "description": (
                    f"small(a{small},sz={sizes[small]:.2f}) ate {food[small]}, "
                    f"large(a{large},sz={sizes[large]:.2f}) ate {food[large]}"
                ),
            })

    events.sort(key=lambda x: -x["score"])
    return events


def _detect_high_eod(bdf, window=50, min_rate_multiplier=2.0):
    """
    Find windows where any agent's EOD firing rate exceeds 2× their own episode mean.
    Score = peak rate / mean rate.
    """
    events = []
    for (env_id, ep), ep_df in bdf.groupby(["env_id", "episode_index"]):
        for aid in sorted(ep_df["agent_id"].unique()):
            ag = ep_df[ep_df["agent_id"] == aid].sort_values("time_step")
            eod = ag["emit_eod"].astype(float).values
            times = ag["time_step"].values

            if len(eod) < window:
                continue

            mean_rate = eod.mean()
            if mean_rate < 0.02:
                continue  # agent barely fires, skip

            # Rolling sum over window
            kernel = np.ones(window) / window
            rolling = np.convolve(eod, kernel, mode="valid")
            peak_idx = int(np.argmax(rolling))
            peak_rate = float(rolling[peak_idx])
            ratio = peak_rate / mean_rate

            if ratio >= min_rate_multiplier:
                t_center = int(times[peak_idx + window // 2])
                events.append({
                    "env_id": env_id, "episode": ep,
                    "t_center": t_center, "score": ratio,
                    "description": (
                        f"agent{aid} EOD burst: {peak_rate:.2f}/step "
                        f"({ratio:.1f}× mean) at t={t_center}"
                    ),
                })

    events.sort(key=lambda x: -x["score"])
    return events


DETECTORS = {
    "chasing":       _detect_chasing,
    "eating":        _detect_eating,
    "coordination":  _detect_coordination,
    "size_reversal": _detect_size_reversal,
    "high_eod":      _detect_high_eod,
}


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("eval_spec_dir")
    ap.add_argument("--scenario", required=True, choices=list(DETECTORS),
                    help="Behavioural scenario to search for")
    ap.add_argument("--n-clips",  type=int, default=3,
                    help="Number of clips to generate (default 3)")
    ap.add_argument("--window",   type=int, default=120,
                    help="Half-window around t_center in timesteps (default 120 → 240-step clip)")
    ap.add_argument("--min-duration", type=int, default=20,
                    help="Min sustained duration for chasing detector (default 20)")
    ap.add_argument("--under-pressure", action="store_true",
                    help="For eating: only show events with nearby/biting pressure")
    ap.add_argument("--gif",         action="store_true")
    ap.add_argument("--compression", type=int, default=2, choices=range(1, 6), metavar="1-5")
    ap.add_argument("--fps",         type=int, default=None)
    args = ap.parse_args(argv)

    eval_spec_dir = os.path.abspath(args.eval_spec_dir)
    raw_dir = os.path.join(eval_spec_dir, "raw")

    print(f"Loading data from {raw_dir} ...")
    bdf, adf, all_eps = _load_all(raw_dir)

    print(f"Detecting '{args.scenario}' events ...")
    detector = DETECTORS[args.scenario]
    if args.scenario == "chasing":
        events = detector(bdf, min_duration=args.min_duration)
    elif args.scenario == "eating":
        events = detector(bdf, under_pressure=args.under_pressure)
    elif args.scenario == "size_reversal":
        events = detector(bdf, all_eps)
    else:
        events = detector(bdf)

    if not events:
        print("No matching events found.")
        return

    print(f"Found {len(events)} events; generating top {min(args.n_clips, len(events))} clips.")

    out_dir = os.path.join(eval_spec_dir, "highlights", args.scenario)
    os.makedirs(out_dir, exist_ok=True)
    ext = "gif" if args.gif else "mp4"

    for i, ev in enumerate(events[: args.n_clips]):
        t_start = max(0, ev["t_center"] - args.window)
        # t_end: cap at last available timestep for this (env, ep)
        ep_ts = bdf[
            (bdf["env_id"] == ev["env_id"]) & (bdf["episode_index"] == ev["episode"])
        ]["time_step"]
        t_end = min(int(ep_ts.max()), ev["t_center"] + args.window)

        out_path = os.path.join(out_dir, f"clip_{i:02d}.{ext}")

        print(
            f"\n  [{i+1}/{min(args.n_clips, len(events))}] {ev['description']}\n"
            f"    env={ev['env_id']} ep={ev['episode']} t=[{t_start}..{t_end}] → {out_path}"
        )

        replay_argv = [
            eval_spec_dir,
            "--env-id",  str(ev["env_id"]),
            "--episode", str(ev["episode"]),
            "--t-start", str(t_start),
            "--t-end",   str(t_end),
            "--output",  out_path,
            "--compression", str(args.compression),
        ]
        if args.gif:
            replay_argv.append("--gif")
        if args.fps:
            replay_argv += ["--fps", str(args.fps)]

        _re.main(replay_argv)

    print(f"\nAll clips saved to {out_dir}/")


if __name__ == "__main__":
    main()
