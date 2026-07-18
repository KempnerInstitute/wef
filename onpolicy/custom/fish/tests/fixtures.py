"""Shared synthetic data builders for pipeline tests.

make_agg_flat  → minimal agg_flat.pkl-equivalent DataFrame (preprocess_features input)
make_features_df → minimal per_env_ep_agent_step-equivalent DataFrame (summaries input)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd


def make_agg_flat(
    n_envs: int = 1,
    n_episodes: int = 2,
    n_steps: int = 4,
    n_agents: int = 2,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic DataFrame mimicking agg_flat.pkl (output of preprocess_flatten).

    Guarantees:
      - Uniform (env_id, episode_index, time_step) groups each have exactly n_agents rows.
      - Agent 0 is always at (0, 0); Agent 1 at (3, 4) → distance = 5.0 cm.
      - has_nearby follows a known pattern per episode for meeting_event tests.
      - food_positions = one pellet at (10, 0) for every row.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for env_id in range(n_envs):
        for ep in range(n_episodes):
            # has_nearby pattern: [False, True, True, False, ...] (first n_steps values)
            nearby_pattern = [False, True, True, False] + [False] * max(0, n_steps - 4)
            nearby_pattern = nearby_pattern[:n_steps]

            for t in range(n_steps):
                for agent_id in range(n_agents):
                    # Agent 0 at origin, Agent 1 at (3,4) → dist=5
                    if agent_id == 0:
                        pos = np.array([0.0, 0.0])
                    else:
                        pos = np.array([3.0, 4.0])

                    rows.append({
                        "env_id":          env_id,
                        "episode_index":   ep,
                        "time_step":       t,
                        "agent_id":        agent_id,
                        "position":        pos.copy(),
                        "orientation":     float(agent_id) * np.pi / 4,
                        "agent_size":      0.3 + agent_id * 0.2,
                        "has_nearby":      nearby_pattern[t],
                        "eating_event":    1 if (t == 1 and agent_id == 0) else 0,
                        "was_bitten":      1 if (t == 2 and agent_id == 1) else 0,
                        "emit_eod":        bool(rng.integers(0, 2)),
                        "displacement":    0.5,
                        "displacement_ground": np.array([0.3, 0.4]),
                        "food_positions":  np.array([[10.0, 0.0]]),
                        "arena_size":      (70, 70),
                        "arena_type":      "patchy",
                    })

    return pd.DataFrame(rows)


def make_features_df(
    n_envs: int = 1,
    n_episodes: int = 1,
    n_steps: int = 4,
    n_agents: int = 2,
    seed: int = 0,
) -> pd.DataFrame:
    """Minimal per_env_ep_agent_step DataFrame for summaries tests.

    Designed for exact arithmetic verification:
      Agent 0: eating_event = [1,0,0,1], emit_eod = [1,0,1,0]  → food=2, p_eod=0.5
      Agent 1: eating_event = [0,1,0,0], emit_eod = [0,1,0,1]  → food=1, p_eod=0.5
      Run total food_eaten = 3, run p_emit_eod = 0.5
    """
    rows = []
    eating = [[1, 0, 0, 1], [0, 1, 0, 0]]
    eod    = [[1, 0, 1, 0], [0, 1, 0, 1]]
    for env_id in range(n_envs):
        for ep in range(n_episodes):
            for t in range(n_steps):
                for agent_id in range(n_agents):
                    rows.append({
                        "env_id":          env_id,
                        "episode_index":   ep,
                        "time_step":       t,
                        "agent_id":        agent_id,
                        "agent_size":      0.3 + agent_id * 0.2,
                        "eating_event":    eating[agent_id][t],
                        "emit_eod":        bool(eod[agent_id][t]),
                        "has_nearby":      t == 1,
                        "meeting_event":   t == 1,
                        "was_bitten":      0,
                        "displacement":    0.5,
                        "position_x":      float(agent_id * 3),
                        "position_y":      float(agent_id * 4),
                        "position":        np.array([float(agent_id * 3), float(agent_id * 4)]),
                        "distance_to_nearest_agent": 5.0,
                        "distance_to_closest_food":  7.07,
                        "arena_size":      (70, 70),
                        "arena_type":      "patchy",
                    })
    return pd.DataFrame(rows)
