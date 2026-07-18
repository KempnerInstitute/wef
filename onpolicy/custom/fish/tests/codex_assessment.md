# Red-Team Assessment of `onpolicy/custom/fish/tests`

Date: 2026-06-09

## Bottom Line

The tests are useful but do not yet prove the train-eval-preprocess-analysis pipeline works end to end. The strongest coverage is at the unit/smoke level: feature geometry, summary aggregations, registry invariants, analysis imports, and a few sensing/electric checks. The weakest coverage is the actual contract the pipeline claims to provide: a trained run flows through eval, flatten, feature generation, summaries, optional food dispersion, per-spec analyses, and multi-spec analyses, with expected files and markers produced.

The current test suite also is not clean under pytest: from `onpolicy/custom/fish`, the selected modern tests fail with 3 failures, and from the repo root collection fails before execution because `fonts/Arial.ttf` is resolved relative to the current working directory.

## Commands Run

From repo root:

```bash
pytest --collect-only -q onpolicy/custom/fish/tests
```

Result: collection failed on `tests/test_preprocess_summaries.py` because `utils_figstyle.py` tries to load `fonts/Arial.ttf` relative to the repo root.

From `onpolicy/custom/fish`:

```bash
pytest --collect-only -q tests
```

Result: 78 tests collected.

From `onpolicy/custom/fish`:

```bash
pytest -q tests/test_pipeline_registry.py tests/test_preprocess_features.py tests/test_preprocess_summaries.py tests/test_analysis_smoke.py tests/test_refactor.py
```

Result: 54 passed, 1 xfailed, 3 failed. Failures:

- `test_analysis_smoke.py::test_1f1rw1p_smoke`: `analysis_1f1rw1p.run()` does not accept `out_dir`.
- `test_analysis_smoke.py::test_1f1rw1p_smoke_with_data`: same API mismatch.
- `test_refactor.py::test_sensing_params`: expects `SensingParams.morm_sensor_min`, which no longer exists.

## What The Tests Do Well

`test_preprocess_features.py` does real value checking on small, deterministic data. It validates rising-edge `meeting_event`, position extraction, displacement component extraction, nearest-agent distance, food distance, single-agent NaN behavior, and masking of observed closest-agent angles. This is meaningful coverage for a high-blast-radius feature step.

`test_preprocess_summaries.py` checks aggregation semantics rather than just shape. It verifies food counts are sums, EOD rates are means, row counts match grouping levels, agent size is preserved, and multi-episode grouping works. This catches a class of silent scientific errors.

`test_pipeline_registry.py` protects some important registry contracts: analysis keys must either be dispatchable or explicitly known gaps, RNN/obs analyses require corresponding eval outputs, homing has the right task/aggression settings, and registry keys match `spec.spec_key`.

`test_analysis_smoke.py` is useful as a broad import/schema smoke layer. It exercises many analysis modules against synthetic `derived/` and `raw/` files, so missing imports and obvious missing-column crashes surface quickly.

## Current Failures And Mismatches

The documented testing plan says `test_analysis_smoke.py` is implemented and passing, but it currently fails for `analysis_1f1rw1p`. The test calls `run(evals_dir, out_dir=...)`, while `analysis_1f1rw1p.py` exposes `run(spec_dir, plot_only=False)`. This is either a test bug or an analysis API inconsistency. It matters because the smoke suite is meant to protect dispatch signatures.

`test_refactor.py` appears stale relative to `SensingParams`: it expects `morm_sensor_min`. If this field was intentionally renamed or removed, the test is no longer testing the intended contract. If it was not intentional, the code has regressed.

Pytest invocation from the repo root is broken by a current-working-directory assumption. `utils_figstyle.py` loads `fonts/Arial.ttf` as a relative path. The test plan says to run from `onpolicy/custom/fish`, and `pipeline.py` has the same cwd assumption, but the tests do not encode that in config or fixtures.

The older `*_test.py` files are collected by pytest. Several are closer to calibration/plotting scripts than strict tests. For example, `arena_randomness_test.py` defines pytest-collected tests with an `arena_case` argument but no fixture; collection succeeds, but execution would require explicit parameterization or a fixture. This can make full-suite behavior surprising.

## Pipeline Coverage Assessment

The pipeline sequence in `pipeline.py` is:

1. `eval_fish.main(...)`
2. `preprocess_flatten.run(...)`
3. `preprocess_features.run(...)`
4. `preprocess_summaries.run(...)`
5. `preprocess_food_dispersion.run(...)` for patchy specs
6. per-spec `analysis_*.run(spec_dir=...)`
7. multi-spec `analysis_*.run(evals_dir=..., out_dir=...)`

That sequence is not tested as a sequence. There is no test that calls `pipeline.run_one_spec` with faked eval/flatten/features/summaries modules and asserts call order, skip behavior, force behavior, output checks, and marker creation.

The skip and failure semantics are mostly untested. `pipeline.py` tolerates failures by calling several stages with `allow_fail=True`, then deciding whether to continue based on file existence. Tests do not verify the intended behavior when flatten fails, features fails, summaries fails, an analysis raises, an analysis is unmapped, or stale marker files exist.

The smoke train script does not fill this gap. `scripts/run_smoke.sh` claims to verify the full train-eval-pipeline stack, but it invokes `pipeline.py` with `--no-analyses`, so it only covers train/eval/preprocess stages, and it is a script rather than an automated pytest with artifact assertions.

There is no test that a minimal real or mocked `run_dir` produces the expected files:

- `evals/<spec>/raw/agg_flat.pkl`
- `evals/<spec>/derived/per_env_ep_agent_step.pkl`
- `evals/<spec>/derived/per_env_ep_agent.pkl`
- `evals/<spec>/derived/per_env_ep.pkl`
- `evals/<spec>/derived/per_env_ep_step.pkl`
- patchy `derived/food_dispersion.pkl`
- analysis output PDFs/PNGs
- `.analysis_done_<name>` markers
- `multi_eval/<key>/...` outputs

There is no test for `preprocess_flatten.py` at all, so the raw eval recording format to flat dataframe boundary is uncovered.

`preprocess_features.run` and `preprocess_summaries.run` are only indirectly represented by lower-level helper tests. The tests validate some helper math, but not CLI/run-level behavior such as reading/writing pickle paths, force/skip behavior, output filenames, or compatibility with actual `agg_flat.pkl` columns.

`rnn_loader.py` is explicitly planned but not implemented. That leaves RNN dim/PSD/PLSC/decoding analyses dependent on untested file discovery, ordering, shape normalization, and empty-directory behavior.

The analysis smoke tests assert no exception, not correctness. Many tests also accept graceful skips. That is fine for a smoke layer, but it means a broken analysis can pass if it silently returns early or emits no figures. The suite does not generally assert that expected analysis artifacts exist or are non-empty.

Multi-spec analysis coverage is shallow. `interventions`, `nfish`, and `2f1p` are smoke-tested with synthetic directories, but not through `pipeline.run_multi_analyses`, not against registry groups, and not with assertions on produced outputs.

## Specific Blind Spots

- Real train/eval boundary: no pytest trains or fakes a policy checkpoint and verifies `eval_fish` writes the raw files that downstream stages expect.
- Raw flattening: no unit or integration tests for `preprocess_flatten.run`.
- Run-level preprocessing: helper math is tested, but not `preprocess_features.run` / `preprocess_summaries.run` file contracts.
- Pipeline orchestration: no direct test of `run_one_spec` or `main` argument resolution.
- Error policy: no assertions for warning-and-skip versus hard failure.
- Analysis artifact contract: no systematic checks that each analysis emits at least one expected output and that markers reflect actual success.
- Registry completeness: known unmapped analysis keys are allowed forever unless manually revisited.
- Working-directory contract: tests and pipeline depend on being run from `onpolicy/custom/fish`.
- Old diagnostic scripts: pytest collects non-idiomatic tests that may not be intended CI targets.
- Performance/CI profile: full analysis smoke tests take tens of seconds and emit many warnings; no fast/slow markers separate cheap contract tests from plotting-heavy diagnostics.

## Does The Suite Do What It Intends?

Partially.

For P1/P2 helper correctness, yes: the feature and summary tests are targeted and meaningful.

For P3 registry consistency, mostly yes: the tests catch missing dispatch mappings and some eval-output prerequisites. They do not test command-line group resolution or multi-spec dispatch.

For P5 analysis smoke coverage, only partially: most modules are exercised, but the suite currently fails for `analysis_1f1rw1p`, and passing smoke tests do not imply output correctness or even output existence.

For the broader claim that the train-eval-preprocess-analysis pipeline is sufficiently tested, no. The suite currently tests islands of behavior around the pipeline, not the integrated pipeline contract.

## Highest-Value Next Tests

1. Add a pure orchestration test for `pipeline.run_one_spec` using monkeypatched fake modules/functions. Assert eval, flatten, features, summaries, food dispersion, analyses, and markers happen in the intended order.

2. Add a minimal filesystem contract test for `preprocess_features.run` and `preprocess_summaries.run`: write synthetic input pickle, call `run`, assert exact output filenames, required columns, and skip/force behavior.

3. Add `test_preprocess_flatten.py` with a tiny synthetic raw eval directory that exercises the actual flatten boundary.

4. Fix or quarantine stale/broken tests: update `test_refactor.py`, fix the `analysis_1f1rw1p` smoke API mismatch, and either parameterize or rename old diagnostic `*_test.py` scripts if they are not intended pytest targets.

5. Add artifact assertions to analysis smoke tests for representative scripts: at least one expected PDF/PNG exists and is non-empty after `run`.

6. Implement the planned `test_rnn_loader.py`.

7. Add pytest config or path handling so tests can run from repo root. At minimum, fix font loading to resolve relative to the fish package directory.

8. Add one small end-to-end smoke test that runs `pipeline.py` over a prebuilt tiny fake `run_dir` with analyses enabled and asserts derived outputs plus markers. This should be separate from the expensive `scripts/run_smoke.sh` training smoke.

## Logic-Focused Robustness Plan

The priority should be validating scientific and pipeline logic, not copied figures or final manuscript packaging. A much more robust suite would have four layers:

1. deterministic data-contract tests,
2. property/invariant tests for derived features,
3. pipeline state-machine tests,
4. analysis-level semantic tests on tiny controlled datasets.

### 1. Strengthen Data-Contract Tests

Add explicit schema tests for each stage boundary. These should not just check "file exists"; they should assert required columns, dtypes/categories where meaningful, grouping keys, uniqueness, and row-count relationships.

Recommended contracts:

- `agg_flat.pkl` -> `per_env_ep_agent_step.pkl`
  - keys `env_id, episode_index, time_step, agent_id` are present and unique per row.
  - every `(env_id, episode_index, time_step)` has the expected number of agent rows unless active-agent filtering is intentional.
  - position-like columns have consistent dimensions.
  - boolean event columns are boolean or 0/1, not arbitrary numeric.
  - food arrays may be empty, but empty food must not crash feature generation.

- `per_env_ep_agent_step.pkl` -> summaries
  - `per_env_ep_agent.pkl` has one row per `(env_id, episode_index, agent_id)`.
  - `per_env_ep.pkl` has one row per `(env_id, episode_index)`.
  - `per_env_ep_step.pkl` has one row per `(env_id, episode_index, time_step)`.
  - aggregation conservation holds: episode `food_eaten` equals sum of agent `food_eaten`, which equals sum of step `food_eaten`.
  - `num_agents` equals distinct `agent_id` count after filtering.

- raw RNN/obs files -> analysis loaders
  - episode indices are sorted numerically, not lexicographically.
  - shapes are normalized consistently for `(T, E, A, H)` or whatever canonical shape is intended.
  - missing episode files are either allowed with a warning or forbidden, but the test should encode the decision.

These tests catch logic drift even when no figures are generated.

### 2. Add Invariant Tests For Feature Engineering

The feature tests should be expanded from a few exact examples to invariants over several small geometries. These can be plain parametrized tests; property-testing libraries are optional.

High-value invariants:

- Nearest-agent distance is symmetric for two agents and never negative.
- With one agent, all nearest-agent fields are NaN or explicitly absent by contract.
- `meeting_event` count equals the number of `False -> True` transitions in `has_nearby` per `(env_id, episode_index, agent_id)`.
- `angle_to_closest_agent_observed` is populated only when `has_nearby` is true.
- Food distance is nonnegative and NaN/absent when no food exists.
- Food-in-range flags are equivalent to `distance_to_closest_food <= configured_range`.
- Orientation-derived angles are wrapped to the intended interval.
- Reordering input rows does not change output after sorting by keys.
- Multiple envs and multiple episodes do not leak state across group boundaries.

The last two are especially important. Many pandas bugs pass with one env, one episode, and pre-sorted input.

### 3. Test Pipeline As A State Machine

`pipeline.run_one_spec` should have tests that monkeypatch the expensive stages and create/delete sentinel files. The goal is to prove the control flow is right.

Cases to cover:

- clean run: eval -> flatten -> features -> summaries -> food dispersion -> selected analyses.
- existing `agg_flat.pkl` and no `--force`: eval and flatten skipped, features/summaries still considered.
- existing features and no `--force`: feature generation skipped, summaries still considered.
- existing summaries and no `--force`: summaries skipped, analyses still run unless markers exist.
- `--force`: all stages rerun despite existing products.
- flatten failure/no output: features and analyses are skipped.
- features failure/no output: summaries and analyses are skipped.
- summaries failure/no output: analyses are skipped.
- one analysis raises: its marker is not written; later analyses either continue or stop according to the intended policy.
- unmapped analysis key: test whether "skip with info" is acceptable or should fail loudly.
- `--analyses` and `--analyses-add`: filter/add semantics are correct and stable.

This does not require real training or eval. Fake modules can write minimal sentinel pickles into `tmp_path`.

### 4. Add Semantic Analysis Tests

Keep smoke tests, but add a small number of semantic tests for analyses whose conclusions matter. These should use tiny synthetic datasets where the expected result is obvious.

Examples:

- `analysis_general`: if larger fish have strictly more `food_eaten`, the computed size-vs-food summary should have positive slope/order.
- `analysis_eod`: if one agent emits every timestep and another never emits, per-agent EOD rates should be 1.0 and 0.0.
- `analysis_idi`: for a constructed pulse train with fixed interval, the IDI distribution should concentrate at that interval.
- `analysis_pairwise`: for a constructed chase/flee geometry, classified pairwise directionality should match the geometry.
- `analysis_biting_network` / `analysis_bitten_network`: a known biter/victim matrix should produce the expected counts.
- `analysis_homing`: an agent moving monotonically toward the target should have decreasing target distance and positive success/approach metrics.
- RNN analyses: constant RNN activity should not produce spurious dimensionality; simple two-state activity should produce low-dimensional structure.

These tests should target helper functions where possible. If an analysis script currently only writes figures and hides intermediate tables, factor out pure helper functions that return dataframes/stats. That makes the logic testable without asserting pixels or PDF details.

### 5. Make Fixtures More Adversarial

Current fixtures are intentionally simple. Add fixture variants designed to break accidental assumptions:

- unsorted rows,
- two envs and two episodes,
- missing food,
- empty food arrays,
- one agent,
- active-agent subset,
- all sensors off,
- all events false,
- all events true,
- NaNs in optional columns,
- nonconsecutive episode indices,
- nonconsecutive agent ids,
- variable number of food pellets per step,
- patchy vs uniform arena labels,
- homing-specific two-agent target layout.

Each fixture variant should be small, deterministic, and named for the assumption it attacks.

### 6. Encode Conservation Laws

Add tests for quantities that should be conserved across aggregation levels:

- Total food eaten is consistent across agent, step, and episode summaries.
- Bite counts are consistent between biter-side and victim-side representations when both are present.
- Event counts do not change when rows are shuffled.
- Per-episode means are weighted correctly; avoid accidental "mean of means" when group sizes differ.
- Episode duration / timestep counts match source data.

These tests are more valuable than broad smoke tests because they catch silent wrong figures.

### 7. Add Golden Tiny Pipeline Dataset

Create one tiny checked-in or generated-on-the-fly dataset that represents a complete eval output without training. It should be small enough to understand by inspection.

Use it to test:

- flatten -> features -> summaries,
- a couple of representative analyses,
- multi-spec analysis over two or three fake specs,
- expected summary values.

Avoid making this a "copy output files" test. The useful assertions are the intermediate tables and computed statistics.

### 8. Improve Failure Strictness Around Logic

Right now several stages and analyses can "gracefully skip." That is useful for exploratory scripts, but dangerous for logic assurance.

Recommendation:

- keep permissive behavior in production pipeline if desired,
- add strict test mode or helper wrappers that fail if required inputs are missing,
- have analysis tests assert whether a skip was expected,
- do not count "returned without exception" as success when the intended analysis had enough data to run.

For example, `test_rnn_dim_smoke_no_rnn` is fine as a missing-input behavior test, but it should be paired with a positive test that verifies a known RNN input produces a non-empty result table or output artifact.

### 9. Prioritized Implementation Order

1. Fix the currently failing tests or mark stale ones explicitly.
2. Add run-level contract tests for `preprocess_features.run` and `preprocess_summaries.run`.
3. Add conservation-law tests across summary outputs.
4. Add pipeline state-machine tests with monkeypatched stages.
5. Add adversarial fixture variants for unsorted rows, multiple envs/episodes, one-agent, and no-food cases.
6. Implement `test_rnn_loader.py`.
7. Refactor the most important analyses to expose pure stat/table helpers, then test those helpers semantically.
8. Add one tiny generated end-to-end pipeline dataset with analyses enabled.

If only a small amount of time is available, do items 2-4 first. They give the best protection against logically wrong downstream results.
