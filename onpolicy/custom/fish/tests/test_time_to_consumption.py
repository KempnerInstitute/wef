"""Tests for analysis_style time-to-event plot functions and helpers.

Functions under test:
  time_to_X_log, time_to_X_linear  — nth-event vs time (log/linear Y)
  X_timecourse_log, X_timecourse_linear — cumulative events over time (log/linear X)
  save_event_time_plot_set — writes all four variants to disk
  compute_idis, consumption_times_from_step_df, biting_times_from_step_df
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from analysis_style import (
    time_to_X_log,
    time_to_X_linear,
    X_timecourse_log,
    X_timecourse_linear,
    save_event_time_plot_set,
    consumption_times_from_step_df,
    biting_times_from_step_df,
)
from utils_features import compute_idis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ep_arrays(n_episodes=10, n_events=5, seed=0):
    """Return a list of sorted time arrays (seconds), one per episode."""
    rng = np.random.default_rng(seed)
    return [np.sort(rng.uniform(0, 60, size=n_events)) for _ in range(n_episodes)]


def _make_condition_times(labels=("A", "B"), n_episodes=10, n_events=5):
    return [(lbl, _make_ep_arrays(n_episodes, n_events, seed=i))
            for i, lbl in enumerate(labels)]


def _palette(labels):
    colors = ["#4477AA", "#EE6677", "#228833", "#CCBB44"]
    return {lbl: colors[i % len(colors)] for i, lbl in enumerate(labels)}


# ---------------------------------------------------------------------------
# time_to_X_log / time_to_X_linear
# ---------------------------------------------------------------------------

class TestTimeToXLog:

    def test_creates_pdf(self, tmp_path):
        ct = _make_condition_times(["A", "B"])
        out = str(tmp_path / "ttc_log.pdf")
        time_to_X_log(ct, _palette(["A", "B"]), out)
        assert os.path.exists(out)

    def test_custom_labels(self, tmp_path):
        ct = _make_condition_times(["A", "B"])
        out = str(tmp_path / "ttc_log_labels.pdf")
        time_to_X_log(ct, _palette(["A", "B"]), out,
                      event_label="Biting events", time_label="Time to biting (s)")
        assert os.path.exists(out)

    def test_single_condition(self, tmp_path):
        ct = _make_condition_times(["solo"])
        out = str(tmp_path / "ttc_log_solo.pdf")
        time_to_X_log(ct, _palette(["solo"]), out)
        assert os.path.exists(out)

    def test_empty_condition_list(self, tmp_path):
        out = str(tmp_path / "ttc_log_empty.pdf")
        time_to_X_log([], {}, out)
        assert os.path.exists(out)

    def test_below_min_episodes_skipped(self, tmp_path):
        ct = [("sparse", _make_ep_arrays(n_episodes=2))]
        out = str(tmp_path / "ttc_log_sparse.pdf")
        time_to_X_log(ct, _palette(["sparse"]), out)
        assert os.path.exists(out)

    def test_empty_arrays_skipped(self, tmp_path):
        ct = [("empty_eps", [np.array([]) for _ in range(5)])]
        out = str(tmp_path / "ttc_log_empty_eps.pdf")
        time_to_X_log(ct, _palette(["empty_eps"]), out)
        assert os.path.exists(out)

    def test_variable_length_episodes(self, tmp_path):
        rng = np.random.default_rng(42)
        arrays = [np.sort(rng.uniform(0, 60, size=rng.integers(1, 10)))
                  for _ in range(12)]
        ct = [("var", arrays)]
        out = str(tmp_path / "ttc_log_var.pdf")
        time_to_X_log(ct, _palette(["var"]), out)
        assert os.path.exists(out)


class TestTimeToXLinear:

    def test_creates_pdf(self, tmp_path):
        ct = _make_condition_times(["A", "B"])
        out = str(tmp_path / "ttc_linear.pdf")
        time_to_X_linear(ct, _palette(["A", "B"]), out)
        assert os.path.exists(out)

    def test_empty_condition_list(self, tmp_path):
        out = str(tmp_path / "ttc_linear_empty.pdf")
        time_to_X_linear([], {}, out)
        assert os.path.exists(out)

    def test_single_condition(self, tmp_path):
        ct = _make_condition_times(["solo"])
        out = str(tmp_path / "ttc_linear_solo.pdf")
        time_to_X_linear(ct, _palette(["solo"]), out)
        assert os.path.exists(out)


# ---------------------------------------------------------------------------
# X_timecourse_log / X_timecourse_linear
# ---------------------------------------------------------------------------

class TestXTimecourseLog:

    def test_creates_pdf(self, tmp_path):
        ct = _make_condition_times(["A", "B"])
        out = str(tmp_path / "tc_log.pdf")
        X_timecourse_log(ct, _palette(["A", "B"]), out)
        assert os.path.exists(out)

    def test_custom_labels(self, tmp_path):
        ct = _make_condition_times(["A", "B"])
        out = str(tmp_path / "tc_log_labels.pdf")
        X_timecourse_log(ct, _palette(["A", "B"]), out,
                          event_label="Biting events", time_label="Time (s)")
        assert os.path.exists(out)

    def test_below_min_episodes_skipped(self, tmp_path):
        ct = [("sparse", _make_ep_arrays(n_episodes=1))]
        out = str(tmp_path / "tc_log_sparse.pdf")
        X_timecourse_log(ct, _palette(["sparse"]), out)
        assert os.path.exists(out)

    def test_empty_condition_list(self, tmp_path):
        out = str(tmp_path / "tc_log_empty.pdf")
        X_timecourse_log([], {}, out)
        assert os.path.exists(out)


class TestXTimecourseLinear:

    def test_creates_pdf(self, tmp_path):
        ct = _make_condition_times(["A", "B"])
        out = str(tmp_path / "tc_linear.pdf")
        X_timecourse_linear(ct, _palette(["A", "B"]), out)
        assert os.path.exists(out)

    def test_empty_condition_list(self, tmp_path):
        out = str(tmp_path / "tc_linear_empty.pdf")
        X_timecourse_linear([], {}, out)
        assert os.path.exists(out)

    def test_single_condition(self, tmp_path):
        ct = _make_condition_times(["solo"])
        out = str(tmp_path / "tc_linear_solo.pdf")
        X_timecourse_linear(ct, _palette(["solo"]), out)
        assert os.path.exists(out)


# ---------------------------------------------------------------------------
# save_event_time_plot_set
# ---------------------------------------------------------------------------

class TestSaveEventTimePlotSet:

    def test_writes_four_files(self, tmp_path):
        ct = _make_condition_times(["A", "B"])
        save_event_time_plot_set(ct, _palette(["A", "B"]), str(tmp_path), "eating")
        expected = [
            "eating_log.pdf",
            "eating_linear.pdf",
            "eating_timecourse_log.pdf",
            "eating_timecourse_linear.pdf",
        ]
        for fname in expected:
            assert os.path.exists(tmp_path / fname), f"missing {fname}"

    def test_custom_labels_accepted(self, tmp_path):
        ct = _make_condition_times(["A", "B"])
        save_event_time_plot_set(ct, _palette(["A", "B"]), str(tmp_path), "biting",
                                  event_label="Biting events",
                                  time_label="Time to biting event (s)")
        assert os.path.exists(tmp_path / "biting_log.pdf")

    def test_single_condition_skipped(self, tmp_path):
        # < 2 conditions → should not write any files and not crash
        ct = _make_condition_times(["solo"])
        save_event_time_plot_set(ct, _palette(["solo"]), str(tmp_path), "solo_event")
        assert not any(f.startswith("solo_event") for f in os.listdir(tmp_path))

    def test_empty_skipped(self, tmp_path):
        save_event_time_plot_set([], {}, str(tmp_path), "no_event")
        assert not any(f.startswith("no_event") for f in os.listdir(tmp_path))


# ---------------------------------------------------------------------------
# Helper: consumption_times_from_step_df
# ---------------------------------------------------------------------------

class TestConsumptionTimesFromStepDf:

    def _make_df(self, n_envs=2, n_eps=3, n_steps=20, eat_every=4):
        rows = []
        for env_id in range(n_envs):
            for ep in range(n_eps):
                for t in range(n_steps):
                    rows.append({
                        "env_id":        env_id,
                        "episode_index": ep,
                        "time_step":     t,
                        "eating_event":  1 if t % eat_every == 0 and t > 0 else 0,
                    })
        return pd.DataFrame(rows)

    def test_returns_list_of_arrays(self):
        df = self._make_df()
        result = consumption_times_from_step_df(df)
        assert isinstance(result, list)
        assert all(isinstance(a, np.ndarray) for a in result)

    def test_arrays_are_sorted(self):
        df = self._make_df()
        for arr in consumption_times_from_step_df(df):
            assert np.all(np.diff(arr) >= 0)

    def test_no_events_returns_empty_list(self):
        df = self._make_df(eat_every=999)  # no events within n_steps=20
        result = consumption_times_from_step_df(df)
        assert result == []

    def test_count_matches_events_per_episode(self):
        n_steps = 20
        eat_every = 4
        # eating_event = 1 when t % eat_every == 0 and t > 0 → t=4,8,12,16 → 4 events
        df = self._make_df(n_envs=1, n_eps=2, n_steps=n_steps, eat_every=eat_every)
        result = consumption_times_from_step_df(df)
        expected_count = len([t for t in range(n_steps) if t % eat_every == 0 and t > 0])
        for arr in result:
            assert len(arr) == expected_count


# ---------------------------------------------------------------------------
# Helper: biting_times_from_step_df
# ---------------------------------------------------------------------------

class TestBitingTimesFromStepDf:

    def _make_df(self, n_envs=1, n_eps=2, n_steps=15, bite_every=3):
        rows = []
        for env_id in range(n_envs):
            for ep in range(n_eps):
                for t in range(n_steps):
                    rows.append({
                        "env_id":        env_id,
                        "episode_index": ep,
                        "time_step":     t,
                        "was_bitten":    1 if t % bite_every == 0 and t > 0 else 0,
                    })
        return pd.DataFrame(rows)

    def test_returns_list(self):
        result = biting_times_from_step_df(self._make_df())
        assert isinstance(result, list)

    def test_missing_column_returns_empty(self):
        df = self._make_df().drop(columns=["was_bitten"])
        assert biting_times_from_step_df(df) == []


# ---------------------------------------------------------------------------
# Helper: compute_idis
# ---------------------------------------------------------------------------

class TestComputeIdis:

    def _make_df(self, emit_pattern):
        """emit_pattern: list of bools for time_step 0..N-1 for one (env,ep,agent)."""
        rows = [{"env_id": 0, "episode_index": 0, "agent_id": 0,
                 "time_step": t, "emit_eod": int(v)}
                for t, v in enumerate(emit_pattern)]
        return pd.DataFrame(rows)

    def test_two_emits_gives_one_idi(self):
        df = self._make_df([False, True, False, False, True])
        idis = compute_idis(df)
        assert len(idis) == 1

    def test_single_emit_no_idi(self):
        df = self._make_df([False, True, False])
        idis = compute_idis(df)
        assert len(idis) == 0

    def test_no_emits(self):
        df = self._make_df([False] * 10)
        idis = compute_idis(df)
        assert len(idis) == 0

    def test_idi_values_positive(self):
        df = self._make_df([True, False, True, False, False, True])
        idis = compute_idis(df)
        assert np.all(idis > 0)

    def test_idi_units_ms(self):
        # emits at t=0 and t=1 → IDI should be 1 step * TIME_STEP_MS
        from analysis_style import TIME_STEP_MS
        df = self._make_df([True, True])
        idis = compute_idis(df)
        assert len(idis) == 1
        assert abs(idis[0] - TIME_STEP_MS) < 1e-6
