"""P2: aggregation correctness in preprocess_summaries.py.

Builds a tiny synthetic features DataFrame with known values and asserts that
build_agent_run_df, build_run_df, and build_step_df produce the expected aggregates.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from fixtures import make_features_df
from utils_stats_general import build_agent_run_df, build_run_df, build_step_df


# Fixture summary for 2 agents, 1 env, 1 episode, 4 steps:
#   Agent 0: eating=[1,0,0,1], eod=[1,0,1,0]  → food_eaten=2, p_eod=0.50
#   Agent 1: eating=[0,1,0,0], eod=[0,1,0,1]  → food_eaten=1, p_eod=0.50
#   Run total food_eaten=3, run p_emit_eod=0.50


def _agent_df():
    return build_agent_run_df(make_features_df())


def _run_df():
    return build_run_df(make_features_df())


def _step_df():
    return build_step_df(make_features_df())


# ---------------------------------------------------------------------------
# build_agent_run_df
# ---------------------------------------------------------------------------

def test_agent_df_row_count():
    """One row per (env, episode, agent) = 2 rows for 1 env × 1 ep × 2 agents."""
    df = _agent_df()
    assert len(df) == 2, f"Expected 2 rows, got {len(df)}"


def test_food_eaten_is_sum_of_eating_events():
    """food_eaten per agent = sum of eating_event over timesteps."""
    df = _agent_df().set_index("agent_id")
    assert df.loc[0, "food_eaten"] == pytest.approx(2), "Agent 0 food_eaten should be 2"
    assert df.loc[1, "food_eaten"] == pytest.approx(1), "Agent 1 food_eaten should be 1"


def test_p_emit_eod_is_mean():
    """p_emit_eod per agent = mean of emit_eod (bool) over timesteps."""
    df = _agent_df().set_index("agent_id")
    # Agent 0: [T, F, T, F] → mean=0.5; Agent 1: [F, T, F, T] → mean=0.5
    assert df.loc[0, "p_emit_eod"] == pytest.approx(0.5, abs=1e-6)
    assert df.loc[1, "p_emit_eod"] == pytest.approx(0.5, abs=1e-6)


def test_agent_size_preserved():
    """agent_size is carried through (constant per agent)."""
    df = _agent_df().set_index("agent_id")
    assert df.loc[0, "agent_size"] == pytest.approx(0.3, abs=1e-6)
    assert df.loc[1, "agent_size"] == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# build_run_df
# ---------------------------------------------------------------------------

def test_run_df_row_count():
    """One row per (env, episode) = 1 row for 1 env × 1 episode."""
    df = _run_df()
    assert len(df) == 1, f"Expected 1 row, got {len(df)}"


def test_run_df_food_eaten_sums_all_agents():
    """Episode food_eaten = total eating events across all agents and timesteps."""
    df = _run_df()
    assert df.iloc[0]["food_eaten"] == pytest.approx(3), (
        f"Expected episode food_eaten=3, got {df.iloc[0]['food_eaten']}"
    )


def test_run_df_p_emit_eod_is_global_mean():
    """p_emit_eod at run level = mean of emit_eod over all agent-timesteps."""
    df = _run_df()
    # 4 steps × 2 agents = 8 values: [T,F,T,F, F,T,F,T] → 4/8 = 0.5
    assert df.iloc[0]["p_emit_eod"] == pytest.approx(0.5, abs=1e-6), (
        f"Expected p_emit_eod=0.5, got {df.iloc[0]['p_emit_eod']}"
    )


def test_run_df_num_agents():
    """num_agents column reflects the actual count of distinct agent_ids."""
    df = _run_df()
    assert int(df.iloc[0]["num_agents"]) == 2


def test_run_df_num_timesteps():
    """num_time_steps reflects the number of distinct timesteps."""
    df = _run_df()
    assert int(df.iloc[0]["num_time_steps"]) == 4


# ---------------------------------------------------------------------------
# build_step_df
# ---------------------------------------------------------------------------

def test_step_df_row_count():
    """One row per (env, episode, timestep) = 4 rows for 4 steps."""
    df = _step_df()
    assert len(df) == 4, f"Expected 4 rows (one per timestep), got {len(df)}"


def test_step_df_food_eaten_per_step():
    """food_eaten at step level = sum over agents at that timestep."""
    df = _step_df().sort_values("time_step").reset_index(drop=True)
    # t=0: agent0 eats (1), agent1 doesn't (0) → 1
    # t=1: agent0 0, agent1 1 → 1
    # t=2: 0+0 → 0
    # t=3: agent0 1, agent1 0 → 1
    expected = [1, 1, 0, 1]
    actual = df["food_eaten"].tolist()
    assert actual == expected, f"Step food_eaten: expected {expected}, got {actual}"


def test_step_df_columns_present():
    """build_step_df produces key expected columns."""
    df = _step_df()
    for col in ("time_step", "food_eaten", "p_emit_eod", "num_agents"):
        assert col in df.columns, f"Column '{col}' missing from build_step_df output"


# ---------------------------------------------------------------------------
# Multi-episode correctness
# ---------------------------------------------------------------------------

def test_run_df_multi_episode():
    """With 2 episodes, build_run_df produces 2 rows each with food_eaten=3."""
    dff = make_features_df(n_episodes=2)
    df = build_run_df(dff)
    assert len(df) == 2, f"Expected 2 rows for 2 episodes, got {len(df)}"
    assert (df["food_eaten"] == 3).all(), (
        f"Both episodes should have food_eaten=3, got {df['food_eaten'].tolist()}"
    )


# ---------------------------------------------------------------------------
# Runner (for direct execution without pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_agent_df_row_count,
        test_food_eaten_is_sum_of_eating_events,
        test_p_emit_eod_is_mean,
        test_agent_size_preserved,
        test_run_df_row_count,
        test_run_df_food_eaten_sums_all_agents,
        test_run_df_p_emit_eod_is_global_mean,
        test_run_df_num_agents,
        test_run_df_num_timesteps,
        test_step_df_row_count,
        test_step_df_food_eaten_per_step,
        test_step_df_columns_present,
        test_run_df_multi_episode,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed+failed} passed")
    sys.exit(0 if failed == 0 else 1)
