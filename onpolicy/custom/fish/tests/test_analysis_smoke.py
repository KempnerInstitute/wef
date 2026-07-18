"""P5: smoke tests for all analysis_*.py scripts.

Each test builds a minimal synthetic spec directory and calls run() to verify that
the script imports, finds required data columns, and completes without crashing.
Tests do NOT verify output correctness — only absence of exceptions.

All tests use `tmp_path` (pytest fixture) so no disk state bleeds between tests.
Analysis modules are imported inside each test via importlib to keep failures isolated.
"""

import importlib
import sys
import os
import matplotlib
matplotlib.use("Agg")  # suppress any display before any analysis import

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from analysis_fixtures import make_spec_dir, make_homing_spec_dir, make_evals_dir


def _import(name):
    return importlib.import_module(name)


# ---------------------------------------------------------------------------
# Group A: per_env_ep_agent_step.pkl only
# ---------------------------------------------------------------------------

def test_behavior_smoke(tmp_path):
    spec_dir = make_spec_dir(tmp_path)
    m = _import("analysis_behavior")
    m.run(spec_dir)


def test_eod_smoke(tmp_path):
    spec_dir = make_spec_dir(tmp_path)
    m = _import("analysis_eod")
    m.run(spec_dir)


def test_idi_smoke(tmp_path):
    spec_dir = make_spec_dir(tmp_path)
    m = _import("analysis_idi")
    m.run(spec_dir)


def test_pairwise_smoke(tmp_path):
    # pairwise returns gracefully when interactions are sparse
    spec_dir = make_spec_dir(tmp_path)
    m = _import("analysis_pairwise")
    m.run(spec_dir)


# ---------------------------------------------------------------------------
# Group B: per_env_ep_agent.pkl only (general also reads step pkl optionally)
# ---------------------------------------------------------------------------

def test_general_smoke(tmp_path):
    spec_dir = make_spec_dir(tmp_path)
    m = _import("analysis_general")
    m.run(spec_dir)


# ---------------------------------------------------------------------------
# Group C: per_env_ep_agent_step.pkl + per_env_ep_agent.pkl
# ---------------------------------------------------------------------------

def test_bite_network_smoke(tmp_path):
    # merged module emits both biter-centric and victim-centric plot sets
    spec_dir = make_spec_dir(tmp_path)
    m = _import("analysis_bite_network")
    m.run(spec_dir)


# ---------------------------------------------------------------------------
# Group D: per_env_ep.pkl + per_env_ep_step.pkl + per_env_ep_agent_step.pkl
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Group E: food_dispersion.pkl + per_env_ep.pkl
# ---------------------------------------------------------------------------

def test_food_distribution_smoke(tmp_path):
    spec_dir = make_spec_dir(tmp_path, include_food_dispersion=True)
    m = _import("analysis_food_distribution")
    m.run(spec_dir)


# ---------------------------------------------------------------------------
# Group F: per_env_ep_agent_step.pkl + ep*_obs.npy
# ---------------------------------------------------------------------------

def test_rollout_diagnostics_smoke(tmp_path):
    spec_dir = make_spec_dir(tmp_path, include_obs=True)
    m = _import("analysis_rollout_diagnostics")
    m.run(spec_dir)


# ---------------------------------------------------------------------------
# Group G: ep*_rnn.npy only
# ---------------------------------------------------------------------------

def test_rnn_dim_smoke(tmp_path):
    spec_dir = make_spec_dir(tmp_path, include_rnn=True)
    m = _import("analysis_rnn_dim")
    m.run(spec_dir)


def test_rnn_dim_smoke_no_rnn(tmp_path):
    """rnn_dim gracefully skips when no rnn files are present."""
    spec_dir = make_spec_dir(tmp_path)
    m = _import("analysis_rnn_dim")
    m.run(spec_dir)  # should not raise — prints skipping message


# ---------------------------------------------------------------------------
# Group H: per_env_ep_agent_step.pkl + ep*_rnn.npy
# ---------------------------------------------------------------------------

def test_rnn_psd_smoke(tmp_path):
    spec_dir = make_spec_dir(tmp_path, include_rnn=True)
    m = _import("analysis_rnn_psd")
    m.run(spec_dir)


def test_rnn_plsc_smoke(tmp_path):
    spec_dir = make_spec_dir(tmp_path, include_rnn=True)
    m = _import("analysis_rnn_plsc")
    m.run(spec_dir)


def test_rnn_decoding_smoke(tmp_path):
    spec_dir = make_spec_dir(tmp_path, include_rnn=True, include_env_args=True)
    m = _import("analysis_rnn_decoding")
    m.run(spec_dir)


# ---------------------------------------------------------------------------
# Group I: homing-specific
# ---------------------------------------------------------------------------

def test_homing_smoke(tmp_path):
    spec_dir = make_homing_spec_dir(tmp_path)
    m = _import("analysis_homing")
    m.run(spec_dir)


# ---------------------------------------------------------------------------
# Group J: multi-spec (evals_dir)
# ---------------------------------------------------------------------------

def test_interventions_smoke(tmp_path):
    """Interventions gracefully skips any spec dirs that don't exist."""
    # Provide only the "sensor" group's first spec — others are silently skipped.
    evals_dir, _ = make_evals_dir(
        tmp_path, ["m1a1k1_patchy_square", "m0a0k0_patchy_square"]
    )
    out_dir = str(tmp_path / "out_interventions")
    m = _import("analysis_interventions")
    m.run(evals_dir, out_dir=out_dir)


def test_nfish_smoke_graceful_too_few(tmp_path):
    """nfish returns gracefully when fewer than 2 k1 nfish specs found."""
    evals_dir, _ = make_evals_dir(
        tmp_path, ["nfish4_m1a1k1_patchy_square"]
    )
    out_dir = str(tmp_path / "out_nfish")
    m = _import("analysis_nfish")
    m.run(evals_dir, out_dir=out_dir)


def test_nfish_smoke_two_specs(tmp_path):
    """nfish runs with exactly 2 k1 nfish specs (the minimum required).

    Each spec dir must have n_agents matching the N in its name (nfish4 → 4, nfish8 → 8)
    so that the rnn reshape inside analysis_nfish._load_rnn_episodes succeeds.
    """
    evals_dir = str(tmp_path / "evals")
    os.makedirs(evals_dir, exist_ok=True)
    make_spec_dir(tmp_path / "evals", spec_key="nfish4_m1a1k1_patchy_square",
                  n_agents=4, include_rnn=True)
    make_spec_dir(tmp_path / "evals", spec_key="nfish8_m1a1k1_patchy_square",
                  n_agents=8, include_rnn=True)
    out_dir = str(tmp_path / "out_nfish")
    m = _import("analysis_nfish")
    m.run(evals_dir, out_dir=out_dir)


def test_2f1p_smoke(tmp_path):
    """2f1p multispec with all 5 condition dirs present."""
    specs = ["2f1p_AltB", "2f1p_AeqB", "2f1p_AgtB", "2f1p_control_a", "2f1p_control_b"]
    evals_dir, _ = make_evals_dir(tmp_path, specs, n_agents=2)
    out_dir = str(tmp_path / "out_2f1p")
    m = _import("analysis_2f1p_multispec")
    m.run(evals_dir, out_dir=out_dir)


def test_1f1rw1p_smoke(tmp_path):
    """1f1rw1p returns gracefully when 1rw1f1p_grid spec dir is absent."""
    # spec_dir doesn't exist → build_grid returns None → graceful exit
    spec_dir = str(tmp_path / "evals" / "1rw1f1p_grid")
    m = _import("analysis_1f1rw1p")
    m.run(spec_dir)


def test_1f1rw1p_smoke_with_data(tmp_path):
    """1f1rw1p with the expected spec key and episodes JSON present."""
    spec_dir = make_spec_dir(
        tmp_path,
        spec_key="1rw1f1p_grid",
        n_agents=2,
        include_episodes_json=True,
    )
    m = _import("analysis_1f1rw1p")
    m.run(spec_dir)
