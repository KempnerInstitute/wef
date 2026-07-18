# Testing Plan — Feature Generation and Analyses

_Created 2026-06-09. Priority order matches the ranking in the conversation that produced this plan._

## Current coverage

The existing tests cover the **sensor/electric layer only**:

| File | What it tests |
|------|--------------|
| `test_refactor.py` | cfg integrity, SensingParams, AgentElectrics virtual counts, obs partitions |
| `sensing_dynamic_test.py` | DynamicBaselineModel sense_mormyromast_all output shapes |
| `sensing_fracrand_test.py` | FracRandModel sense_mormyromast_all output shapes |
| `electric_scene_test.py` | ElectricScene field primitives |
| `arena_randomness_test.py` | Arena food-placement randomness |

**Zero coverage** exists for:
- `preprocess_features.py` / `utils_preprocess.py` (feature engineering)
- `preprocess_summaries.py` / `utils_stats_general.py` (aggregations)
- `eval_registry.py` (spec-level invariants)
- `pipeline.py` dispatch consistency
- Any `analysis_*.py` script
- `rnn_loader.py`

---

## Priority ranking

### P1 — `test_preprocess_features.py`
**Risk**: `get_df_with_candidate_vars` is called before every analysis. A silent bug (wrong column, wrong edge detection, wrong distance formula) propagates to every figure without any error. Currently zero tests.

**Tests**:
- `test_meeting_event_rising_edge` — meeting_event fires only at the False→True transition of has_nearby, not on sustained True
- `test_position_xy_extracted` — position_x/y match position array elements
- `test_displacement_components_from_ground` — dx/dy extracted from displacement_ground vector
- `test_nearest_agent_distance_correct` — distance_to_nearest_agent matches geometry (Pythagorean)
- `test_angle_observed_nan_when_no_nearby` — angle_to_closest_agent_observed is NaN when has_nearby=False
- `test_food_features_added_when_positions_present` — distance_to_closest_food column appears and is ≥ 0

**Synthetic data**: 2 agents, 1 env, 2 episodes, 4 steps each (16 rows). See `fixtures.py:make_agg_flat`.

---

### P2 — `test_preprocess_summaries.py`
**Risk**: `per_env_ep.pkl` is consumed by all multi-spec analyses. A wrong aggregation (e.g., mean vs sum for food_eaten) silently produces wrong multi-spec figures.

**Tests**:
- `test_food_eaten_is_sum` — build_agent_run_df: food_eaten = sum of eating_event per agent
- `test_p_emit_eod_is_mean` — build_agent_run_df: p_emit_eod = mean of emit_eod per agent
- `test_run_df_food_eaten_sums_all_agents` — build_run_df: episode food_eaten = total across agents
- `test_run_df_p_emit_eod_is_mean_across_all` — build_run_df: p_emit_eod = mean across all agent-steps
- `test_step_df_has_one_row_per_timestep` — build_step_df: output has n_steps rows (agents collapsed)

**Synthetic data**: 2 agents, 1 env, 1 episode, 4 steps (8 rows) with known eating and eod values. See `fixtures.py:make_features_df`.

---

### P3 — `test_pipeline_registry.py` (dispatch consistency)
**Risk**: Analysis keys in `SPEC_ANALYSES` (e.g., `predict_action`, `2fish`) that are missing from `_ANALYSIS_SCRIPT_MAP` are silently skipped. A production run can silently omit analyses with only a `[WARN]` line.

**Tests**:
- `test_no_novel_unmapped_keys` — every key in SPEC_ANALYSES either appears in `_ANALYSIS_SCRIPT_MAP` or in the explicit acknowledged-gaps set (`{"predict_action", "attention", "2fish", "rnn_timescales", "2f1p_grid"}`)
- `test_eod_idi_are_separate_keys` — both "eod" and "idi" are present in `_ANALYSIS_SCRIPT_MAP` as distinct keys (split from the old combined `"eod": ["analysis_eod", "analysis_idi"]`)
- `test_rnn_analyses_require_save_rnn` — every spec in EVAL_REGISTRY whose `analyses` includes any of `{"rnn_dim","rnn_psd","rnn_plsc","decoding"}` must have `save_rnn=True`
- `test_decoding_requires_save_obs` — every spec with `"decoding"` in `analyses` must have `save_obs=True`
- `test_homing_spec_has_correct_task` — homing spec has `task="homing"` and `allow_aggression=0`

---

### P4 — `test_rnn_loader.py`
**Risk**: `iter_rnn_episodes` is the sole data source for all four RNN analysis scripts. A broken yield order or shape mismatch would silently corrupt PCA/PSD/PLSC/decoding results.

**Tests**:
- `test_iter_yields_in_index_order` — write ep0_rnn.npy and ep2_rnn.npy (skipping ep1); confirm yield order is 0, 2
- `test_iter_shapes` — each yielded rnn_arr has shape (T, 1, A, H) matching the written array
- `test_load_single_episode` — `load_rnn_episode(raw_dir, k)` returns the correct array for a given k
- `test_iter_empty_dir_yields_nothing` — empty raw_dir yields no items without crashing

---

### P5 — `test_analysis_smoke.py` (import + run smoke tests)
**Risk**: A broken import or missing column in any `analysis_*.py` is only discovered at full-pipeline time (hours in). A smoke test with a tiny synthetic dataset catches this at `pytest` time.

**Approach**: For each dispatched analysis script, write a minimal synthetic `derived/` pkl (one episode, 2 agents, 4 steps) to a temp directory, then call `script.run(run_dir)`. The test passes if `run()` returns without exception (not whether figures are correct).

**Analyses to cover** (ordered by manuscript importance):
1. `analysis_general` — needs per_env_ep_agent.pkl + per_env_ep.pkl
2. `analysis_behavior` — needs per_env_ep_agent_step.pkl + samples subdir
3. `analysis_eod` — needs per_env_ep_agent_step.pkl with eod columns
4. `analysis_bite_network` — needs per_env_ep_agent_step.pkl with was_bitten (dispatched by both `biting_network` and `bitten_network` keys)
5. `analysis_rollout_diagnostics` — needs raw/agg_flat.pkl with obs arrays
6. `analysis_rnn_dim` — needs raw/ep*_rnn.npy files + per_env_ep_agent_step.pkl

Effort: ~2 hrs per script. Batch 1 covers items 1–3 as the first PR.

---

## Implementation files

| File | Priority | Status |
|------|----------|--------|
| `fixtures.py` | shared | ✅ implemented |
| `test_preprocess_features.py` | P1 | ✅ implemented |
| `test_preprocess_summaries.py` | P2 | ✅ implemented |
| `test_pipeline_registry.py` | P3 | ✅ implemented (2 known failures as of 2026-06-30) |
| `test_rnn_loader.py` | P4 | not yet written |
| `analysis_fixtures.py` | P5 shared | ✅ implemented |
| `test_analysis_smoke.py` | P5 | ✅ implemented (multiple known failures — see BACKLOG.md) |
| `ANALYSIS_SMOKE_PLAN.md` | P5 docs | ✅ implemented |
| `test_time_to_consumption.py` | — | ✅ implemented (324 lines) |
| `test_rnn_decoding_features.py` | — | ✅ implemented |
| `test_plsc.py` | — | ⚠️ collection error: imports `compute_plsc` from `analysis_rnn_plsc` — that function was removed; needs update to use `compute_plsc_gpu` |

---

## Running tests

```bash
cd onpolicy/custom/fish
mamba run --name mfrefactor python -m pytest tests/ -v
```

Or run a single file:
```bash
mamba run --name mfrefactor python -m pytest tests/test_preprocess_features.py -v
```

Or run the old-style tests directly:
```bash
mamba run --name mfrefactor python tests/test_refactor.py
```

---

## Notes

- `_ANALYSIS_SCRIPT_MAP` now has separate `"eod"` and `"idi"` keys (split from the old single `"eod"` key that dispatched both). Specs in `SPEC_ANALYSES` that list only `"eod"` will **not** run `analysis_idi` until they also add `"idi"`. This is a known behavior change (intentional split).
- All test fixtures avoid disk I/O where possible; where a temp dir is needed (smoke tests), use `tmp_path` from pytest.
- Do not import any analysis script at module level — use `importlib` inside the test function to keep failures isolated.
