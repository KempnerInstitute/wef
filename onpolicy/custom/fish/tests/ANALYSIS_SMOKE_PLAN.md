# Analysis Smoke-Test Plan

_Created 2026-06-09._

## Goal

One smoke test per `analysis_*.py` script. Each test:
1. Builds a minimal synthetic spec-directory in `tmp_path` (via `analysis_fixtures.py`).
2. Calls `run(spec_dir)` (or `run(evals_dir, out_dir)` for multi-spec scripts).
3. Asserts no exception is raised.

Tests do **not** verify figure correctness — only that imports succeed, required files are
found, and no crash occurs on minimal data. This catches: missing imports, bad column names,
wrong path assumptions, and regressions after refactoring.

---

## Fixture helpers (`analysis_fixtures.py`)

| Helper | Purpose |
|--------|---------|
| `make_spec_dir(tmp_path, **kw)` | Full per-spec directory: `derived/*.pkl`, `raw/ep0_rnn.npy`, `raw/ep0_obs.npy` (opt-in), `logs/env_args.json` (opt-in) |
| `make_homing_spec_dir(tmp_path, **kw)` | Homing variant: 2 agents (agent 0=target, agent 1=homing), target_x/target_y columns |
| `make_evals_dir(tmp_path, spec_keys, **kw)` | Creates multiple spec dirs under `evals/`; returns `(evals_dir, {key: spec_dir})` |

Synthetic defaults: `n_envs=1`, `n_episodes=3`, `n_steps=20`, `n_agents=4`, `rnn_dim=64`, `obs_dim=36`.

---

## Per-spec analysis tests (`test_analysis_smoke.py`)

### Group A — `per_env_ep_agent_step.pkl` only

| Test | Script | Notes |
|------|--------|-------|
| `test_behavior_smoke` | `analysis_behavior` | writes `analyses/behavior/` |
| `test_eod_smoke` | `analysis_eod` | needs `emit_eod` column |
| `test_idi_smoke` | `analysis_idi` | needs `emit_eod` column |
| `test_pairwise_smoke` | `analysis_pairwise` | graceful return if no biting interactions |
| `test_velocity_profiles_smoke` | `analysis_velocity_profiles` | needs `linear_velocity`, `angular_velocity`, `has_food_in_food_sensing_range` |

### Group B — `per_env_ep_agent.pkl` only

| Test | Script | Notes |
|------|--------|-------|
| `test_general_smoke` | `analysis_general` | also optionally loads step pkl via `load_consumption_times` |

### Group C — `per_env_ep_agent_step.pkl` + `per_env_ep_agent.pkl`

| Test | Script | Notes |
|------|--------|-------|
| `test_biting_network_smoke` | `analysis_bite_network` | needs `was_bitten_by_agent_id`, `bite_other_fish`; biter-centric plot set |
| `test_bitten_network_smoke` | `analysis_bite_network` | same columns as biting; victim-centric plot set |

### Group D — three pkls

| Test | Script | Notes |
|------|--------|-------|
| `test_group_spacing_smoke` | `analysis_group_spacing` | needs `per_env_ep`, `per_env_ep_step`, `per_env_ep_agent_step` with `orientation` column |

### Group E — `food_dispersion.pkl` + `per_env_ep.pkl`

| Test | Script | Notes |
|------|--------|-------|
| `test_food_distribution_smoke` | `analysis_food_distribution` | `make_spec_dir(..., include_food_dispersion=True)` |

### Group F — `per_env_ep_agent_step.pkl` + `ep*_obs.npy`

| Test | Script | Notes |
|------|--------|-------|
| `test_rollout_diagnostics_smoke` | `analysis_rollout_diagnostics` | `make_spec_dir(..., include_obs=True)` |

### Group G — `ep*_rnn.npy` only (no pkl required for compute)

| Test | Script | Notes |
|------|--------|-------|
| `test_rnn_dim_smoke` | `analysis_rnn_dim` | `make_spec_dir(..., include_rnn=True)`; graceful if no rnn files |

### Group H — `per_env_ep_agent_step.pkl` + `ep*_rnn.npy`

| Test | Script | Notes |
|------|--------|-------|
| `test_rnn_psd_smoke` | `analysis_rnn_psd` | graceful skip if no rnn files |
| `test_rnn_plsc_smoke` | `analysis_rnn_plsc` | graceful skip if no rnn files |
| `test_rnn_decoding_smoke` | `analysis_rnn_decoding` | also needs `logs/env_args.json`; `make_spec_dir(..., include_rnn=True, include_env_args=True)` |

### Group I — Homing-specific

| Test | Script | Notes |
|------|--------|-------|
| `test_homing_smoke` | `analysis_homing` | `make_homing_spec_dir()`; 2 agents; needs `target_x`, `target_y` columns |

### Group J — Multi-spec (`evals_dir`)

| Test | Script | Notes |
|------|--------|-------|
| `test_interventions_smoke` | `analysis_interventions` | `make_evals_dir` with ≥1 spec; graceful skip of missing specs |
| `test_nfish_smoke` | `analysis_nfish` | needs `nfish4_*` + `nfish8_*` spec dirs with rnn files; graceful if <2 nfish specs |
| `test_2f1p_smoke` | `analysis_2f1p_multispec` | needs `2f1p_AltB`, `2f1p_AeqB`, `2f1p_AgtB`, `2f1p_control_a`, `2f1p_control_b` spec dirs |
| `test_1f1rw1p_smoke` | `analysis_1f1rw1p` | needs `1rw1f1p_grid` spec dir; graceful return if not found |

---

## Required columns per synthetic DataFrame

### `per_env_ep_agent_step.pkl`
Keys: `env_id`, `episode_index`, `time_step`, `agent_id`

| Column | Type | Notes |
|--------|------|-------|
| `agent_size` | float | |
| `emit_eod` | bool | |
| `has_nearby` | bool | |
| `meeting_event` | bool | |
| `eating_event` | int | 0/1 |
| `was_bitten` | int | 0/1 |
| `was_bitten_by_agent_id` | float | NaN when no biting |
| `bite_other_fish` | int | 0/1 |
| `position_x` | float | |
| `position_y` | float | |
| `orientation` | float | radians |
| `displacement` | float | |
| `displacement_x` | float | |
| `displacement_y` | float | |
| `linear_velocity` | float | |
| `angular_velocity` | float | |
| `distance_to_nearest_agent` | float | NaN for 1-agent case |
| `size_of_nearest_agent` | float | |
| `angle_to_closest_agent` | float | |
| `nearest_agent_id` | float | |
| `distance_to_closest_food` | float | |
| `angle_to_closest_food` | float | |
| `has_food_in_food_sensing_range` | bool | |
| `arena_size` | tuple | `(70, 70)` |
| `arena_type` | str | `"patchy"` |

### `per_env_ep_agent.pkl`
Keys: `env_id`, `episode_index`, `agent_id`

| Column | Type |
|--------|------|
| `run_id` | str |
| `food_eaten` | int |
| `p_emit_eod` | float |
| `num_biting_events` | int |
| `mean_displacement` | float |
| `agent_size` | float |
| `distance_to_nearest_agent` | float |
| `arena_size` | tuple |
| `arena_type` | str |

### `per_env_ep.pkl`
Keys: `env_id`, `episode_index`

| Column | Type |
|--------|------|
| `run_id` | str |
| `food_eaten` | int |
| `food_eaten_theil` | float |
| `p_emit_eod` | float |
| `num_interactions` | int |
| `num_biting_events` | int |
| `mean_nn_distance_cm` | float |
| `polarization` | float |
| `num_agents` | int |
| `arena_type` | str |
| `arena_size` | tuple |

### `per_env_ep_step.pkl`
Keys: `env_id`, `episode_index`, `time_step`

| Column | Type |
|--------|------|
| `run_id` | str |
| `food_eaten` | int |
| `p_emit_eod` | float |
| `mean_distance_to_nearest_agent` | float |

### `food_dispersion.pkl`
Keys: `env_id`, `episode_index`

| Column | Type |
|--------|------|
| `food_count_start` | int |
| `food_count_mid` | int |
| `food_count_end` | int |
| `food_dispersion_start` | float |
| `food_dispersion_mid` | float |
| `food_dispersion_end` | float |
| `mean_nn_food_start` | float |
| `mean_nn_food_mid` | float |
| `mean_nn_food_end` | float |

---

## Running the smoke tests

```bash
cd onpolicy/custom/fish
mamba run --name mfrefactor python -m pytest tests/test_analysis_smoke.py -v
```

Single test:
```bash
mamba run --name mfrefactor python -m pytest tests/test_analysis_smoke.py::test_behavior_smoke -v
```

---

## Implementation status

| File | Status |
|------|--------|
| `analysis_fixtures.py` | ✅ implemented |
| `test_analysis_smoke.py` | ✅ implemented |
