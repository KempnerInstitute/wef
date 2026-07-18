"""Synthetic directory builders for analysis smoke tests.

make_spec_dir      → builds a full spec_dir with all derived/*.pkl and optional rnn/obs npy
make_homing_spec_dir → homing variant with 2 agents (target + homing agent)
make_evals_dir     → creates multiple spec dirs under an evals/ folder
"""

import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# --- synthetic DataFrame builders -------------------------------------------

def _make_step_df(n_envs, n_episodes, n_steps, n_agents, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for env_id in range(n_envs):
        for ep in range(n_episodes):
            for t in range(n_steps):
                for agent_id in range(n_agents):
                    angle = float(agent_id) * 2 * np.pi / n_agents
                    rows.append({
                        "env_id":           env_id,
                        "episode_index":    ep,
                        "time_step":        t,
                        "agent_id":         agent_id,
                        "agent_size":       0.3 + agent_id * 0.1,
                        "emit_eod":         bool(rng.integers(0, 2)),
                        "has_nearby":       t > 0 and t < n_steps - 1,
                        "meeting_event":    t == 1,
                        "eating_event":     1 if (t % 5 == 0 and agent_id == 0) else 0,
                        "was_bitten":       1 if (t % 7 == 0 and agent_id == 1) else 0,
                        "was_bitten_by_agent_id": (
                            float(0) if (t % 7 == 0 and agent_id == 1) else float("nan")
                        ),
                        "bite_other_fish":  1 if (t % 7 == 0 and agent_id == 0) else 0,
                        "position_x":       float(agent_id * 5 + env_id),
                        "position_y":       float(ep * 2),
                        "orientation":      angle,
                        "displacement":     0.5 + rng.uniform(0, 0.1),
                        "displacement_x":   0.3,
                        "displacement_y":   0.4,
                        "linear_velocity":  1.0 + rng.uniform(0, 0.5),
                        "angular_velocity": rng.uniform(-0.2, 0.2),
                        "distance_to_nearest_agent": 5.0 if n_agents > 1 else float("nan"),
                        "size_of_nearest_agent":     0.4,
                        "angle_to_closest_agent":    angle + 0.1,
                        "nearest_agent_id":          float((agent_id + 1) % n_agents),
                        "distance_to_closest_food":  8.0,
                        "angle_to_closest_food":     0.0,
                        "has_food_in_food_sensing_range": t % 3 == 0,
                        "food_observed":    t % 3 == 0,
                        "position":         np.array([float(agent_id * 5 + env_id), float(ep * 2)]),
                        "arena_size":       (70, 70),
                        "arena_type":       "patchy",
                    })
    return pd.DataFrame(rows)


def _make_agent_df(n_envs, n_episodes, n_agents, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for env_id in range(n_envs):
        for ep in range(n_episodes):
            for agent_id in range(n_agents):
                rows.append({
                    "run_id":                    "test_run",
                    "env_id":                    env_id,
                    "episode_index":             ep,
                    "agent_id":                  agent_id,
                    "food_eaten":                int(rng.integers(0, 5)),
                    "p_emit_eod":                float(rng.uniform(0.3, 0.7)),
                    "num_biting_events":         int(rng.integers(0, 3)),
                    "mean_displacement":         0.5,
                    "agent_size":                0.3 + agent_id * 0.1,
                    "distance_to_nearest_agent": 5.0,
                    "arena_size":                (70, 70),
                    "arena_type":                "patchy",
                })
    return pd.DataFrame(rows)


def _make_ep_df(n_envs, n_episodes, n_agents, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for env_id in range(n_envs):
        for ep in range(n_episodes):
            rows.append({
                "run_id":             "test_run",
                "env_id":             env_id,
                "episode_index":      ep,
                "food_eaten":         int(rng.integers(1, 10)),
                "food_eaten_theil":   float(rng.uniform(0.1, 0.5)),
                "p_emit_eod":         float(rng.uniform(0.3, 0.7)),
                "num_interactions":   int(rng.integers(5, 30)),
                "num_biting_events":  int(rng.integers(0, 5)),
                "mean_nn_distance_cm": float(rng.uniform(5.0, 20.0)),
                "polarization":       float(rng.uniform(0.3, 0.9)),
                "num_agents":         n_agents,
                "arena_type":         "patchy",
                "arena_size":         (70, 70),
            })
    return pd.DataFrame(rows)


def _make_step_summary_df(n_envs, n_episodes, n_steps, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for env_id in range(n_envs):
        for ep in range(n_episodes):
            for t in range(n_steps):
                rows.append({
                    "run_id":                        "test_run",
                    "env_id":                        env_id,
                    "episode_index":                 ep,
                    "time_step":                     t,
                    "food_eaten":                    1 if t % 5 == 0 else 0,
                    "p_emit_eod":                    float(rng.uniform(0.3, 0.7)),
                    "mean_distance_to_nearest_agent": float(rng.uniform(5.0, 20.0)),
                })
    return pd.DataFrame(rows)


def _make_food_dispersion_df(n_envs, n_episodes, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for env_id in range(n_envs):
        for ep in range(n_episodes):
            rows.append({
                "env_id":              env_id,
                "episode_index":       ep,
                "food_count_start":    20,
                "food_count_mid":      15,
                "food_count_end":      10,
                "food_dispersion_start": float(rng.uniform(10, 20)),
                "food_dispersion_mid":   float(rng.uniform(10, 20)),
                "food_dispersion_end":   float(rng.uniform(10, 20)),
                "mean_nn_food_start":    float(rng.uniform(5, 15)),
                "mean_nn_food_mid":      float(rng.uniform(5, 15)),
                "mean_nn_food_end":      float(rng.uniform(5, 15)),
            })
    return pd.DataFrame(rows)


# --- directory builders ------------------------------------------------------

def make_spec_dir(
    tmp_path,
    spec_key="test_spec",
    n_envs=1,
    n_episodes=3,
    n_steps=20,
    n_agents=4,
    seed=0,
    include_rnn=False,
    include_obs=False,
    include_food_dispersion=False,
    include_env_args=False,
    include_episodes_json=False,
    rnn_dim=64,
    obs_dim=96,
):
    """Build a minimal per-spec directory with all standard derived pkls.

    Returns the spec_dir path as a string.
    """
    spec_dir = tmp_path / spec_key
    derived = spec_dir / "derived"
    derived.mkdir(parents=True)
    raw_dir = spec_dir / "raw"
    raw_dir.mkdir(parents=True)

    _make_step_df(n_envs, n_episodes, n_steps, n_agents, seed).to_pickle(
        derived / "per_env_ep_agent_step.pkl"
    )
    _make_agent_df(n_envs, n_episodes, n_agents, seed).to_pickle(
        derived / "per_env_ep_agent.pkl"
    )
    _make_ep_df(n_envs, n_episodes, n_agents, seed).to_pickle(
        derived / "per_env_ep.pkl"
    )
    _make_step_summary_df(n_envs, n_episodes, n_steps, seed).to_pickle(
        derived / "per_env_ep_step.pkl"
    )

    if include_food_dispersion:
        _make_food_dispersion_df(n_envs, n_episodes, seed).to_pickle(
            derived / "food_dispersion.pkl"
        )

    rng = np.random.default_rng(seed)
    if include_rnn:
        for ep in range(n_episodes):
            # shape (T, E, A, 1, H) — 1 is the GRU recurrent-layer dim; loader squeezes it
            arr = rng.standard_normal((n_steps, n_envs, n_agents, 1, rnn_dim)).astype(np.float32)
            np.save(raw_dir / f"ep{ep}_rnn.npy", arr)

    if include_obs:
        for ep in range(n_episodes):
            arr = rng.standard_normal((n_steps, n_envs, n_agents, obs_dim)).astype(np.float32)
            np.save(raw_dir / f"ep{ep}_obs.npy", arr)

    if include_env_args:
        logs_dir = spec_dir / "logs"
        logs_dir.mkdir(parents=True)
        env_args = {"agent_env_args": {"morm_food_detection_range_m": 0.10}}
        with open(logs_dir / "env_args.json", "w") as f:
            json.dump(env_args, f)

    if include_episodes_json:
        # Minimal episodes JSON for analysis_1f1rw1p (rw_* bot params per episode).
        # Use valid EOD_RATES values and include both frozen/moving to exercise all code paths.
        _VALID_EOD_RATES = [0.0, 0.25, 0.5, 0.75, 1.0]
        ep_rows = []
        for env_id in range(n_envs):
            for ep in range(n_episodes):
                ep_rows.append({
                    "env_id":        env_id,
                    "episode_index": ep,
                    "rw_eod_A":      _VALID_EOD_RATES[ep % len(_VALID_EOD_RATES)],
                    "rw_freeze_A":   ep % 2 == 0,
                    "rw_size_A":     0.3 if ep % 3 != 2 else 0.5,
                    "rw_size_B":     0.4,
                    "rw_trial_id":   ep % 10,
                })
        with open(raw_dir / "episodes_000_episodes.json", "w") as f:
            json.dump(ep_rows, f)

    return str(spec_dir)


def make_homing_spec_dir(
    tmp_path,
    spec_key="homing",
    n_envs=1,
    n_episodes=3,
    n_steps=20,
    seed=0,
):
    """Homing-specific spec dir: 2 agents (agent_id=0 = stationary target, 1 = homing).

    Adds target_x and target_y columns that analysis_homing._prepare_homing_dff expects.
    Returns spec_dir path as a string.
    """
    spec_dir = tmp_path / spec_key
    derived = spec_dir / "derived"
    derived.mkdir(parents=True)
    raw_dir = spec_dir / "raw"
    raw_dir.mkdir(parents=True)

    n_agents = 2
    rng = np.random.default_rng(seed)
    rows = []
    for env_id in range(n_envs):
        for ep in range(n_episodes):
            target_x = float(rng.uniform(10, 60))
            target_y = float(rng.uniform(10, 60))
            for t in range(n_steps):
                # agent 0 = stationary target (stays at target position)
                rows.append({
                    "env_id":           env_id,
                    "episode_index":    ep,
                    "time_step":        t,
                    "agent_id":         0,
                    "position_x":       target_x,
                    "position_y":       target_y,
                    "position":         np.array([target_x, target_y]),
                    "agent_size":       0.3,
                    "orientation":      0.0,
                    "displacement":     0.0,
                    "displacement_x":   0.0,
                    "displacement_y":   0.0,
                    "linear_velocity":  0.0,
                    "angular_velocity": 0.0,
                    "emit_eod":         False,
                    "has_nearby":       False,
                    "meeting_event":    False,
                    "eating_event":     0,
                    "was_bitten":       0,
                    "was_bitten_by_agent_id": float("nan"),
                    "bite_other_fish":  0,
                    "distance_to_nearest_agent": 5.0,
                    "size_of_nearest_agent":     0.4,
                    "angle_to_closest_agent":    0.0,
                    "nearest_agent_id":          1.0,
                    "distance_to_closest_food":  float("nan"),
                    "angle_to_closest_food":     float("nan"),
                    "trial_start":      t == 0,
                    "has_food_in_food_sensing_range": False,
                    "arena_size":       (70, 70),
                    "arena_type":       "patchy",
                })
                # agent 1 = homing agent (moves toward target)
                frac = t / max(n_steps - 1, 1)
                hx = target_x * frac + 35.0 * (1 - frac)
                hy = target_y * frac + 35.0 * (1 - frac)
                rows.append({
                    "env_id":           env_id,
                    "episode_index":    ep,
                    "time_step":        t,
                    "agent_id":         1,
                    "position_x":       hx,
                    "position_y":       hy,
                    "position":         np.array([hx, hy]),
                    "agent_size":       0.4,
                    "orientation":      np.arctan2(target_y - hy, target_x - hx),
                    "displacement":     0.5,
                    "displacement_x":   0.3,
                    "displacement_y":   0.4,
                    "linear_velocity":  1.0,
                    "angular_velocity": 0.05,
                    "emit_eod":         bool(rng.integers(0, 2)),
                    "has_nearby":       t > 0,
                    "meeting_event":    t == 1,
                    "eating_event":     0,
                    "was_bitten":       0,
                    "was_bitten_by_agent_id": float("nan"),
                    "bite_other_fish":  0,
                    "distance_to_nearest_agent": 5.0 * (1 - frac),
                    "size_of_nearest_agent":     0.3,
                    "angle_to_closest_agent":    0.0,
                    "nearest_agent_id":          0.0,
                    "distance_to_closest_food":  float("nan"),
                    "angle_to_closest_food":     float("nan"),
                    "trial_start":               t == 0,
                    "knollen_error_angle_nearest": float(rng.uniform(-np.pi, np.pi)),
                    "mormyromast_error_angle":     float(rng.uniform(-np.pi, np.pi)),
                    "ampullary_error_angle":       float(rng.uniform(-np.pi, np.pi)),
                    "has_food_in_food_sensing_range": False,
                    "arena_size":       (70, 70),
                    "arena_type":       "patchy",
                })
    step_df = pd.DataFrame(rows)
    step_df.to_pickle(derived / "per_env_ep_agent_step.pkl")

    # also write a minimal agg_flat as fallback
    step_df.to_pickle(raw_dir / "agg_flat.pkl")

    return str(spec_dir)


def make_evals_dir(tmp_path, spec_keys, **make_spec_kwargs):
    """Create an evals/ directory containing one spec subdir per key.

    Returns (evals_dir_str, {spec_key: spec_dir_str}).
    """
    evals_dir = tmp_path / "evals"
    evals_dir.mkdir(parents=True)
    spec_dirs = {}
    for key in spec_keys:
        sd = make_spec_dir(evals_dir, spec_key=key, **make_spec_kwargs)
        spec_dirs[key] = sd
    return str(evals_dir), spec_dirs
