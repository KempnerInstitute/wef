"""P1: feature engineering correctness (get_df_with_candidate_vars).

Each test builds a small synthetic agg_flat DataFrame with known geometry,
runs the feature pipeline, and asserts the computed columns match expected values.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from fixtures import make_agg_flat

import cfg

_FOOD_RANGE_CM    = cfg.AGENT_PARAMS["morm_food_detection_range_m"] * 100
_KNOLLEN_RANGE_CM = cfg.AGENT_PARAMS["knollen_agent_detection_range_m"] * 100


def _run_features(dff=None, **kwargs):
    """Run feature pipeline; pass sensing ranges explicitly to suppress warnings."""
    from utils_preprocess import get_df_with_candidate_vars
    if dff is None:
        dff = make_agg_flat(**kwargs)
    return get_df_with_candidate_vars(
        dff,
        food_detection_range_cm=_FOOD_RANGE_CM,
        knollen_detection_range_cm=_KNOLLEN_RANGE_CM,
        include_counters=False,  # skip slow counters; not under test here
    )


# ---------------------------------------------------------------------------
# 1. meeting_event fires only on the rising edge of has_nearby
# ---------------------------------------------------------------------------

def test_meeting_event_rising_edge():
    """meeting_event=True only on the False→True transition, not sustained True."""
    # has_nearby pattern per episode (4 steps): [F, T, T, F]
    # Expected meeting_event:                   [F, T, F, F]
    dff = make_agg_flat(n_episodes=1, n_steps=4, n_agents=2)
    out = _run_features(dff=dff)

    agent0 = out[(out["agent_id"] == 0) & (out["episode_index"] == 0)].sort_values("time_step")
    expected_meeting = [False, True, False, False]
    actual_meeting   = agent0["meeting_event"].tolist()
    assert actual_meeting == expected_meeting, (
        f"meeting_event wrong: expected {expected_meeting}, got {actual_meeting}"
    )


def test_meeting_event_fires_at_step0_when_starts_nearby():
    """At step 0 the shift fill_value=False means starting adjacent triggers a meeting.

    This is the current implemented behavior: if agents begin the episode already
    within range (has_nearby=True at t=0), the rising-edge logic fires once because
    the 'previous' step is treated as not-nearby (fill_value=False).
    """
    dff = make_agg_flat(n_episodes=1, n_steps=4, n_agents=2)
    dff["has_nearby"] = True  # all steps nearby from the start
    out = _run_features(dff=dff)

    # Step 0 should have meeting_event=True (rising edge from fill_value=False)
    step0 = out[out["time_step"] == 0]
    assert step0["meeting_event"].all(), (
        "meeting_event should fire at step 0 when has_nearby starts True "
        "(shift fill_value=False treats start-of-episode as not-nearby)"
    )
    # Steps 1+ should be False (sustained True → no further rising edges)
    later = out[out["time_step"] > 0]
    assert not later["meeting_event"].any(), (
        "meeting_event should not fire on sustained has_nearby=True (only rising edge)"
    )


# ---------------------------------------------------------------------------
# 2. position_x / position_y extracted correctly
# ---------------------------------------------------------------------------

def test_position_xy_extracted():
    """position_x and position_y are taken from position[0] and position[1]."""
    dff = make_agg_flat(n_episodes=1, n_steps=2, n_agents=2)
    out = _run_features(dff=dff)

    for _, row in out.iterrows():
        assert row["position_x"] == pytest.approx(row["position"][0], abs=1e-6)
        assert row["position_y"] == pytest.approx(row["position"][1], abs=1e-6)


# ---------------------------------------------------------------------------
# 3. displacement_x / displacement_y from displacement_ground
# ---------------------------------------------------------------------------

def test_displacement_components_from_ground():
    """dx/dy are extracted from displacement_ground when that column is present."""
    dff = make_agg_flat(n_episodes=1, n_steps=2, n_agents=2)
    # displacement_ground = [0.3, 0.4] in fixture
    out = _run_features(dff=dff)

    assert "displacement_x" in out.columns, "displacement_x not added"
    assert "displacement_y" in out.columns, "displacement_y not added"
    np.testing.assert_allclose(out["displacement_x"].to_numpy(float), 0.3, atol=1e-5)
    np.testing.assert_allclose(out["displacement_y"].to_numpy(float), 0.4, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. nearest agent distance is geometrically correct
# ---------------------------------------------------------------------------

def test_nearest_agent_distance_correct():
    """Agent 0 at (0,0) and Agent 1 at (3,4) → distance_to_nearest_agent = 5.0."""
    dff = make_agg_flat(n_episodes=1, n_steps=2, n_agents=2)
    out = _run_features(dff=dff)

    assert "distance_to_nearest_agent" in out.columns
    # Both agents should see the other at distance 5.0 (Pythagorean)
    vals = out["distance_to_nearest_agent"].to_numpy(float)
    np.testing.assert_allclose(vals, 5.0, atol=1e-4,
        err_msg=f"Expected all distances≈5.0, got: {np.unique(vals)}"
    )


def test_single_agent_distance_is_nan():
    """With only one agent there is no nearest other — column should be NaN."""
    dff = make_agg_flat(n_episodes=1, n_steps=2, n_agents=1)
    out = _run_features(dff=dff)
    assert out["distance_to_nearest_agent"].isna().all(), (
        "distance_to_nearest_agent should be NaN when n_agents=1"
    )


# ---------------------------------------------------------------------------
# 5. angle_to_closest_agent_observed masked to NaN when has_nearby=False
# ---------------------------------------------------------------------------

def test_angle_observed_nan_when_no_nearby():
    """angle_to_closest_agent_observed is NaN where has_nearby=False."""
    dff = make_agg_flat(n_episodes=1, n_steps=4, n_agents=2)
    out = _run_features(dff=dff)

    no_nearby = out[~out["has_nearby"]]
    yes_nearby = out[out["has_nearby"]]

    assert no_nearby["angle_to_closest_agent_observed"].isna().all(), (
        "angle_to_closest_agent_observed must be NaN when has_nearby=False"
    )
    # When has_nearby=True the column should have a real value
    assert yes_nearby["angle_to_closest_agent_observed"].notna().all(), (
        "angle_to_closest_agent_observed should not be NaN when has_nearby=True"
    )


# ---------------------------------------------------------------------------
# 6. food features added when food_positions present
# ---------------------------------------------------------------------------

def test_food_features_added_when_food_present():
    """distance_to_closest_food column is added and ≥ 0 when food_positions is set."""
    dff = make_agg_flat(n_episodes=1, n_steps=2, n_agents=2)
    out = _run_features(dff=dff)

    assert "distance_to_closest_food" in out.columns, (
        "distance_to_closest_food not added despite food_positions being present"
    )
    assert (out["distance_to_closest_food"] >= 0).all(), (
        "distance_to_closest_food should be non-negative"
    )


def test_food_distance_matches_geometry():
    """Agent 0 at (0,0) with food at (10,0) → distance = 10 cm."""
    dff = make_agg_flat(n_episodes=1, n_steps=1, n_agents=1)
    # Agent 0 is at (0,0) in the fixture; food at (10,0) → dist=10
    out = _run_features(dff=dff)
    agent0 = out[out["agent_id"] == 0]
    vals = agent0["distance_to_closest_food"].to_numpy(dtype=float)
    np.testing.assert_allclose(vals, 10.0, atol=0.1,
        err_msg=f"Expected distance_to_closest_food≈10.0, got {vals}"
    )


@pytest.mark.xfail(
    reason="BUG in utils_preprocess.py: in-range flags block (line ~126) unconditionally "
           "accesses 'distance_to_closest_food' even when food_positions is absent, "
           "raising KeyError. Fix: guard with `if 'distance_to_closest_food' in merged_df`.",
    raises=KeyError,
    strict=True,
)
def test_food_features_skipped_when_no_positions():
    """When food_positions column is absent, pipeline should not crash.

    Currently FAILS with KeyError — see xfail reason above.
    """
    dff = make_agg_flat(n_episodes=1, n_steps=2, n_agents=2)
    dff = dff.drop(columns=["food_positions"])
    out = _run_features(dff=dff)  # should not raise
    assert out is not None


# ---------------------------------------------------------------------------
# Runner (for direct execution without pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_meeting_event_rising_edge,
        test_meeting_event_fires_at_step0_when_starts_nearby,
        test_position_xy_extracted,
        test_displacement_components_from_ground,
        test_nearest_agent_distance_correct,
        test_single_agent_distance_is_nan,
        test_angle_observed_nan_when_no_nearby,
        test_food_features_added_when_food_present,
        test_food_distance_matches_geometry,
    ]
    # xfail tests (expect a specific exception — treated as pass when they raise it)
    xfail_tests = {
        test_food_features_skipped_when_no_positions: KeyError,
    }

    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS   {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL   {t.__name__}: {e}")
            failed += 1
    for t, exc_type in xfail_tests.items():
        try:
            t()
            print(f"  XPASS  {t.__name__} (expected failure but passed)")
            passed += 1
        except exc_type:
            print(f"  XFAIL  {t.__name__} (known bug — see docstring)")
            passed += 1
        except Exception as e:
            print(f"  FAIL   {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed+len(xfail_tests)+failed} passed")
    sys.exit(0 if failed == 0 else 1)
