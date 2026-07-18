"""P3: pipeline dispatch consistency and EvalSpec invariants.

Tests that eval_registry.py and pipeline.py remain consistent with each other,
plus a small filesystem-backed orchestration check for analysis marker behavior.
"""

import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval_registry import EVAL_REGISTRY, SPEC_ANALYSES


# Analysis keys in SPEC_ANALYSES that are acknowledged as not-yet-implemented.
# Update this set when a new analysis is implemented and wired into _ANALYSIS_SCRIPT_MAP.
_KNOWN_UNMAPPED: set = {
    "predict_action",
    "attention",
    "2fish",
    "rnn_timescales",
    "2f1p_grid",
}

# Analyses that require the RNN hidden-state files to be written.
_RNN_ANALYSES = {"rnn_dim", "rnn_psd", "rnn_plsc"}

# Analyses that require the observation files to be written.
_OBS_ANALYSES = {"decoding"}


def _load_script_map():
    """Import _ANALYSIS_SCRIPT_MAP from pipeline without triggering main()."""
    import importlib
    pipeline = importlib.import_module("pipeline")
    return pipeline._ANALYSIS_SCRIPT_MAP


def test_no_novel_unmapped_keys():
    """Every analysis key in SPEC_ANALYSES is either mapped or in the known-gaps set."""
    script_map = _load_script_map()
    mapped = set(script_map.keys())

    novel_unmapped = {}
    for spec_key, analyses in SPEC_ANALYSES.items():
        for akey in analyses:
            if akey not in mapped and akey not in _KNOWN_UNMAPPED:
                novel_unmapped.setdefault(spec_key, []).append(akey)

    assert not novel_unmapped, (
        f"Unknown analysis keys found in SPEC_ANALYSES (not in _ANALYSIS_SCRIPT_MAP "
        f"and not in _KNOWN_UNMAPPED):\n"
        + "\n".join(f"  {s}: {ks}" for s, ks in sorted(novel_unmapped.items()))
        + "\nAdd to _KNOWN_UNMAPPED in this test if intentionally planned but unimplemented, "
        "or implement the analysis and register it in _ANALYSIS_SCRIPT_MAP."
    )


def test_eod_and_idi_are_separate_keys():
    """eod and idi are now separate dispatch keys (split from the old combined entry)."""
    script_map = _load_script_map()
    assert "eod" in script_map, "'eod' missing from _ANALYSIS_SCRIPT_MAP"
    assert "idi" in script_map, (
        "'idi' missing from _ANALYSIS_SCRIPT_MAP — analysis_idi.py will not run "
        "for any spec. Add 'idi' as a separate key."
    )
    # Each key must dispatch its own script only (not both)
    assert script_map["eod"] == ["analysis_eod"], (
        f"'eod' should dispatch only analysis_eod, got {script_map['eod']}"
    )
    assert script_map["idi"] == ["analysis_idi"], (
        f"'idi' should dispatch only analysis_idi, got {script_map['idi']}"
    )


def test_rnn_analyses_require_save_rnn():
    """Specs that include rnn_dim/rnn_psd/rnn_plsc must have save_rnn=True."""
    bad = {}
    for spec_key, spec in EVAL_REGISTRY.items():
        if any(a in _RNN_ANALYSES for a in spec.analyses):
            if not spec.save_rnn:
                bad[spec_key] = spec.analyses
    assert not bad, (
        f"Specs with RNN analyses but save_rnn=False (rnn .npy files won't be written):\n"
        + "\n".join(f"  {k}: {v}" for k, v in sorted(bad.items()))
    )


def test_decoding_requires_save_obs():
    """Specs that include the decoding analysis must have save_obs=True."""
    bad = {}
    for spec_key, spec in EVAL_REGISTRY.items():
        if "decoding" in spec.analyses:
            if not spec.save_obs:
                bad[spec_key] = spec.analyses
    assert not bad, (
        f"Specs with decoding analysis but save_obs=False (obs .npy files won't be written):\n"
        + "\n".join(f"  {k}: {v}" for k, v in sorted(bad.items()))
    )


def test_homing_spec_invariants():
    """The homing spec must have allow_aggression=0 and task='homing'."""
    spec = EVAL_REGISTRY.get("homing")
    assert spec is not None, "'homing' not found in EVAL_REGISTRY"
    assert spec.task == "homing", f"homing spec task={spec.task!r}, expected 'homing'"
    assert spec.allow_aggression == 0, (
        f"homing spec allow_aggression={spec.allow_aggression}, expected 0"
    )


def test_all_spec_keys_in_registry_match_field():
    """Every EVAL_REGISTRY entry's spec_key matches its dict key."""
    mismatches = {k: s.spec_key for k, s in EVAL_REGISTRY.items() if k != s.spec_key}
    assert not mismatches, (
        f"EVAL_REGISTRY dict key ≠ spec.spec_key for: {mismatches}"
    )


def test_force_analyses_ignores_done_marker_without_forcing_preprocess(tmp_path, monkeypatch):
    """--force-analyses should rerun marked analyses without acting like --force."""
    import pipeline

    spec_key = "unit_spec"
    spec_dir = tmp_path / "evals" / spec_key
    (spec_dir / "raw").mkdir(parents=True)
    (spec_dir / "derived").mkdir()
    analyses_dir = spec_dir / "analyses"
    analyses_dir.mkdir()
    (spec_dir / "raw" / "agg_flat.pkl").touch()
    (spec_dir / "derived" / "per_env_ep_agent_step.pkl").touch()
    (spec_dir / "derived" / "per_env_ep.pkl").touch()
    (analyses_dir / ".analysis_done_general").touch()

    calls = []

    def event_times_run(*, spec_dir, force):
        calls.append(("event_times", force))

    def analysis_run(*, spec_dir):
        calls.append(("analysis", spec_dir))

    monkeypatch.setitem(
        sys.modules,
        "preprocess_event_times",
        SimpleNamespace(run=event_times_run),
    )
    monkeypatch.setitem(
        sys.modules,
        "analysis_general",
        SimpleNamespace(run=analysis_run),
    )

    spec = SimpleNamespace(spec_key=spec_key, analyses=["general"])
    kwargs = dict(
        run_dir=str(tmp_path),
        spec=spec,
        analyses=spec.analyses,
        dry_run=False,
        force_eval=False,
        force_preprocess=False,
        run_flatten=True,
        run_analyses=True,
        analyses_filter=None,
        analyses_add=None,
        render_episodes_override=None,
        episode_length_override=None,
        eval_rollout_threads_override=None,
    )

    pipeline.run_one_spec(force_analyses=False, **kwargs)
    assert ("event_times", False) in calls
    assert not any(call[0] == "analysis" for call in calls)

    calls.clear()
    pipeline.run_one_spec(force_analyses=True, **kwargs)
    assert ("event_times", False) in calls
    assert any(call[0] == "analysis" for call in calls)


def test_force_preprocess_reruns_derived_steps_without_forcing_eval(tmp_path, monkeypatch):
    """--force-preprocess should refresh derived pkls but still skip existing evals."""
    import pipeline

    spec_key = "unit_spec"
    spec_dir = tmp_path / "evals" / spec_key
    (spec_dir / "raw").mkdir(parents=True)
    derived_dir = spec_dir / "derived"
    derived_dir.mkdir()
    (spec_dir / "raw" / "agg_flat.pkl").touch()
    (derived_dir / "per_env_ep_agent_step.pkl").touch()
    (derived_dir / "per_env_ep.pkl").touch()

    calls = []

    def features_run(*, flat_pkl, output_dir, skip_counters, force):
        calls.append(("features", force))

    def summaries_run(*, features_pkl, output_dir, force):
        calls.append(("summaries", force))

    def event_times_run(*, spec_dir, force):
        calls.append(("event_times", force))

    monkeypatch.setitem(
        sys.modules,
        "preprocess_features",
        SimpleNamespace(run=features_run),
    )
    monkeypatch.setitem(
        sys.modules,
        "preprocess_summaries",
        SimpleNamespace(run=summaries_run),
    )
    monkeypatch.setitem(
        sys.modules,
        "preprocess_event_times",
        SimpleNamespace(run=event_times_run),
    )

    spec = SimpleNamespace(spec_key=spec_key, analyses=[])
    pipeline.run_one_spec(
        run_dir=str(tmp_path),
        spec=spec,
        analyses=[],
        dry_run=False,
        force_eval=False,
        force_preprocess=True,
        force_analyses=False,
        run_flatten=True,
        run_analyses=False,
        analyses_filter=None,
        analyses_add=None,
        render_episodes_override=None,
        episode_length_override=None,
        eval_rollout_threads_override=None,
    )

    assert calls == [
        ("features", True),
        ("summaries", True),
        ("event_times", True),
    ]


if __name__ == "__main__":
    tests = [
        test_no_novel_unmapped_keys,
        test_eod_and_idi_are_separate_keys,
        test_rnn_analyses_require_save_rnn,
        test_decoding_requires_save_obs,
        test_homing_spec_invariants,
        test_all_spec_keys_in_registry_match_field,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed+failed} passed")
    sys.exit(0 if failed == 0 else 1)
