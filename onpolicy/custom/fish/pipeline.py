"""Orchestrate eval runs from EVAL_REGISTRY.

Replaces run_existing_evals_skip_if_exists.sh. Must be run from
onpolicy/custom/fish/ (same requirement as eval_fish.py).

Usage
-----
  python3 pipeline.py <run_dir> [options]

Key options
-----------
  --specs KEY [KEY ...]        run only these spec_keys (default: all non-nfish)
  --group GROUP [GROUP ...]    run all specs in named group(s)
  --render-episodes N          override render_episodes for every spec
  --dry-run                    print what would happen without running
  --force, --force-all         re-run eval, preprocess, and analyses
  --force-eval                 re-run eval even if flat output exists
  --force-preprocess           re-run derived preprocessing from agg_flat.pkl
  --force-analyses             re-run analyses even if done markers exist
  --no-flatten                 skip preprocess_flatten.py
  --no-analyses                skip all figure generation (per-spec and multi-spec)
  --analyses NAME [NAME ...]   run only these named analyses (overrides spec registry)
  --analyses-add NAME [NAME ...] append extra analyses on top of each spec's registry list
  --no-multi-spec              skip multi-spec analyses only
  --multi-spec KEY [KEY ...]   restrict multi-spec analyses to these keys
                               (default: all — interventions, 2f1p, nfish, food_grid_iso)

Notes
-----
Skip rule: a spec is skipped when
  <run_dir>/evals/<spec_key>/raw/*_agg_flat.pkl  already exists.

render_episodes in the registry are the defaults; --render-episodes
overrides all of them globally (e.g. for quick smoke tests).

nfish specs (nfish1_*, nfish2_*, ...) are excluded from the default run.
"""

import argparse
import glob
import importlib
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from eval_registry import EVAL_REGISTRY, SPEC_GROUPS, EvalSpec


# ---------------------------------------------------------------------------
# Analysis script dispatch tables
# ---------------------------------------------------------------------------

# Per-spec: analysis name (from spec.analyses) → list of script module names
_ANALYSIS_SCRIPT_MAP = {
    "general":      ["analysis_general"],
    "behavior":     ["analysis_behavior"],
    "eod":          ["analysis_eod"],
    "idi":          ["analysis_idi"],
    "rnn_dim":      ["analysis_rnn_dim"],
    "rnn_psd":      ["analysis_rnn_psd"],
    "rnn_plsc":     ["analysis_rnn_plsc"],
    "decoding":     ["analysis_rnn_decoding"],
    "pairwise":         ["analysis_pairwise"],
    # biting_network (biter-centric) + bitten_network (victim-centric) merged into
    # one module; both keys dispatch it. The per-spec dedup guard below runs it once.
    "biting_network":   ["analysis_bite_network"],
    "bitten_network":   ["analysis_bite_network"],
    "food_distribution":    ["analysis_food_distribution"],
    # Homing: prefers derived/per_env_ep_agent_step.pkl (needs agent_ids_by_dist etc.);
    # falls back to raw/agg_flat.pkl. agent_id=0 is stationary target, agent_id=1 homing agent.
    "homing":           ["analysis_homing"],
    "rollout_diagnostics": ["analysis_rollout_diagnostics"],
    "1f1rw1p":          ["analysis_1f1rw1p"],
    # Not yet re-implemented in new style — pipeline will warn and skip:
    # "predict_action", "attention", "2f1p", "2fish",
    # "rnn_timescales", "2f1p_grid"
}

# Multi-spec: key → (script module name, output subdirectory under multi_eval/)
MULTI_SPEC_ALL = ["interventions", "2f1p", "2f1p_k0", "2f1p_k0k1", "nfish",
                  "food_grid_iso", "patchy_vs_uniform"]
_MULTI_SPEC_SCRIPT_MAP = {
    "interventions":  ("analysis_interventions",     "interventions"),
    "2f1p":           ("analysis_2f1p_multispec",    "2f1p"),
    "2f1p_k0":        ("analysis_2f1p_k0_multispec", "2f1p_k0"),
    "2f1p_k0k1":      ("analysis_2f1p_k0k1_compare", "2f1p_k0k1"),
    "nfish":          ("analysis_nfish",              "nfish"),
    "food_grid_iso":  ("analysis_food_grid_iso",      "food_grid_iso"),
    "patchy_vs_uniform": ("analysis_patchy_vs_uniform", "patchy_vs_uniform"),
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _eval_subdir(run_dir: str, spec_key: str, subdir: str) -> str:
    return os.path.join(run_dir, "evals", spec_key, subdir)


def _flat_pkls(run_dir: str, spec_key: str, split: bool = False) -> list:
    raw_dir = _eval_subdir(run_dir, spec_key, "raw")
    if split:
        all_flats = sorted(glob.glob(os.path.join(raw_dir, "*_flat.pkl")))
        return [f for f in all_flats if os.path.basename(f) != "agg_flat.pkl"]
    agg = os.path.join(raw_dir, "agg_flat.pkl")
    return [agg] if os.path.exists(agg) else []


def _features_pkls(run_dir: str, spec_key: str, split: bool = False) -> list:
    derived = _eval_subdir(run_dir, spec_key, "derived")
    if split:
        return sorted(glob.glob(os.path.join(derived, "*_features_step.pkl")))
    agg = os.path.join(derived, "per_env_ep_agent_step.pkl")
    return [agg] if os.path.exists(agg) else []


def _summary_pkls(run_dir: str, spec_key: str) -> list:
    p = os.path.join(_eval_subdir(run_dir, spec_key, "derived"), "per_env_ep.pkl")
    return [p] if os.path.exists(p) else []


def _call(func, dry_run: bool, *, label: str = "", allow_fail: bool = False, **kwargs) -> bool:
    """Call func(**kwargs) directly. Returns True on success."""
    prefix = "[DRY RUN]" if dry_run else "[RUN]"
    tag = f" ({label})" if label else ""
    kw_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    print(f"{prefix}{tag} {func.__module__}.{func.__qualname__}({kw_str})", flush=True)
    if dry_run:
        return True
    t0 = time.perf_counter()
    try:
        func(**kwargs)
        elapsed = time.perf_counter() - t0
        print(f"  [DONE {label}: {elapsed:.1f}s]", flush=True)
        return True
    except SystemExit as e:
        elapsed = time.perf_counter() - t0
        if e.code not in (None, 0):
            print(f"[WARN] {label} exited with code {e.code} ({elapsed:.1f}s)", flush=True)
            if allow_fail:
                return False
            raise
        print(f"  [DONE {label}: {elapsed:.1f}s]", flush=True)
        return True
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"[WARN] {label} raised: {e} ({elapsed:.1f}s)", flush=True)
        if allow_fail:
            return False
        raise


def _run_eval(eval_args, episode_configs, dry_run: bool) -> bool:
    """Call eval_fish.main() directly. Returns True on success."""
    import eval_fish as _eval_mod

    label = f"eval_fish.main({eval_args.eval_run_name})"
    if dry_run:
        print(f"[DRY RUN] {label}", flush=True)
        return True
    print(f"[RUN] {label}", flush=True)
    t0 = time.perf_counter()
    try:
        _eval_mod.main(eval_args, episode_configs=episode_configs)
        print(f"  [DONE {label}: {time.perf_counter() - t0:.1f}s]", flush=True)
        return True
    except SystemExit as e:
        elapsed = time.perf_counter() - t0
        if e.code not in (None, 0):
            print(f"[WARN] {label} exited with code {e.code} ({elapsed:.1f}s)", flush=True)
            return False
        print(f"  [DONE {label}: {elapsed:.1f}s]", flush=True)
        return True
    except Exception as e:
        print(f"[WARN] {label} raised: {e} ({time.perf_counter() - t0:.1f}s)", flush=True)
        return False


# ---------------------------------------------------------------------------
# per-spec runner
# ---------------------------------------------------------------------------

def run_one_spec(
    run_dir: str,
    spec: EvalSpec,
    analyses: List[str],
    *,
    dry_run: bool,
    force_eval: bool,
    force_preprocess: bool,
    force_analyses: bool,
    run_flatten: bool,
    run_analyses: bool,
    analyses_filter: Optional[List[str]],
    analyses_add: Optional[List[str]],
    render_episodes_override: Optional[int],
    episode_length_override: Optional[int],
    eval_rollout_threads_override: Optional[int] = None,
) -> None:
    raw_dir      = _eval_subdir(run_dir, spec.spec_key, "raw")
    derived_dir  = _eval_subdir(run_dir, spec.spec_key, "derived")
    spec_dir     = os.path.join(run_dir, "evals", spec.spec_key)
    os.makedirs(raw_dir, exist_ok=True)
    is_split = getattr(spec, "flatten_split", False)

    # --- eval (skip if flat pkl already exists) ---
    existing = _flat_pkls(run_dir, spec.spec_key, split=is_split)
    if existing and not force_eval:
        print(f"[SKIP] {spec.spec_key}: flat pkl exists: {existing[0]}", flush=True)
        flat_exists = True
    else:
        eval_args = spec.to_eval_args(run_dir)
        eval_args.output_subdir = os.path.join("evals", spec.spec_key, "raw")
        if render_episodes_override is not None:
            eval_args.eval_render_episodes = render_episodes_override
        if episode_length_override is not None:
            eval_args.eval_episode_length = episode_length_override
        if eval_rollout_threads_override is not None:
            eval_args.n_rollout_threads = eval_rollout_threads_override

        ok = _run_eval(eval_args, spec.episode_configs, dry_run)
        if not ok:
            print(f"[WARN] eval failed for {spec.spec_key}; skipping flatten/analyses.", flush=True)
            return

        if run_flatten:
            import preprocess_flatten

            _call(preprocess_flatten.run, dry_run, label="flatten", allow_fail=True,
                  run_dir=run_dir, outputs_folder=raw_dir, force=force_eval,
                  delete_raw=True, no_aggregate=is_split)

        # after flatten, re-check
        flat_exists = dry_run or bool(_flat_pkls(run_dir, spec.spec_key, split=is_split))
        if not flat_exists and run_flatten:
            print(f"[WARN] flatten produced no output for {spec.spec_key}; skipping analyses.", flush=True)
            return

    if not flat_exists:
        return

    # --- features ---
    existing_features = _features_pkls(run_dir, spec.spec_key, split=is_split)
    if existing_features and not force_preprocess:
        print(f"[SKIP] {spec.spec_key}: features exist ({len(existing_features)} file(s))", flush=True)
        features_pkls = existing_features
    else:
        flat_pkls_list = _flat_pkls(run_dir, spec.spec_key, split=is_split) if not dry_run else []
        import preprocess_features

        if is_split:
            _call(preprocess_features.run_split, dry_run, label="features", allow_fail=True,
                  flat_pkls=flat_pkls_list, output_dir=derived_dir, skip_counters=True,
                  force=force_preprocess)
        else:
            _call(preprocess_features.run, dry_run, label="features", allow_fail=True,
                  flat_pkl=flat_pkls_list[0] if flat_pkls_list else "", output_dir=derived_dir,
                  skip_counters=True, force=force_preprocess)
        found = _features_pkls(run_dir, spec.spec_key, split=is_split)
        if not found and not dry_run:
            print(f"[WARN] features produced no output for {spec.spec_key}; skipping summaries/analyses.", flush=True)
            return
        features_pkls = found

    features_pkl = features_pkls[0] if features_pkls else ""

    # --- summaries ---
    if _summary_pkls(run_dir, spec.spec_key) and not force_preprocess:
        print(f"[SKIP] {spec.spec_key}: summaries exist", flush=True)
    else:
        import preprocess_summaries

        if is_split:
            _call(preprocess_summaries.run_split, dry_run, label="summaries", allow_fail=True,
                  features_pkls=features_pkls, output_dir=derived_dir, force=force_preprocess)
        else:
            _call(preprocess_summaries.run, dry_run, label="summaries", allow_fail=True,
                  features_pkl=features_pkl, output_dir=derived_dir, force=force_preprocess)
        if not dry_run and not _summary_pkls(run_dir, spec.spec_key):
            print(f"[WARN] summaries produced no output for {spec.spec_key}; skipping analyses.", flush=True)
            return

    # --- event times (consumption + biting) ---
    import preprocess_event_times
    if is_split:
        _call(preprocess_event_times.run_split, dry_run, label="event_times",
              allow_fail=True, features_pkls=features_pkls, spec_dir=spec_dir,
              force=force_preprocess)
    else:
        _call(preprocess_event_times.run, dry_run, label="event_times",
              allow_fail=True, spec_dir=spec_dir, force=force_preprocess)

    # --- food dispersion (patchy specs only) ---
    if "patchy" in spec.spec_key:
        import preprocess_food_dispersion
        _call(preprocess_food_dispersion.run, dry_run, label="food_dispersion",
              allow_fail=True, spec_dir=spec_dir, force=force_preprocess)

    # --- per-spec figures ---
    if analyses_filter is not None:
        analyses = [a for a in analyses if a in analyses_filter]
    if analyses_add is not None:
        seen = set(analyses)
        analyses = analyses + [a for a in analyses_add if a not in seen]

    if not run_analyses or not analyses:
        return

    analyses_dir = os.path.join(spec_dir, "analyses")
    os.makedirs(analyses_dir, exist_ok=True)

    ran_scripts: set = set()  # scripts already run this spec (some keys share a module)
    for analysis_name in analyses:
        marker = os.path.join(analyses_dir, f".analysis_done_{analysis_name}")
        if os.path.exists(marker) and not force_analyses:
            print(f"  [SKIP] {spec.spec_key}/{analysis_name}: marker exists", flush=True)
            continue
        script_names = _ANALYSIS_SCRIPT_MAP.get(analysis_name)
        if script_names is None:
            print(f"  [INFO] '{analysis_name}' not yet in new analysis style — skipped", flush=True)
            continue
        extra_kwargs = spec.analysis_kwargs.get(analysis_name, {})
        ok = True
        for script_name in script_names:
            if script_name in ran_scripts:
                # e.g. biting_network + bitten_network both dispatch analysis_bite_network,
                # which emits both plot sets in one pass — don't run it twice.
                print(f"  [DEDUP] {script_name} already ran this spec — reusing outputs", flush=True)
                continue
            mod = importlib.import_module(script_name)
            if not _call(mod.run, dry_run, label=script_name, allow_fail=True,
                         spec_dir=spec_dir, **extra_kwargs):
                ok = False
            elif not dry_run:
                ran_scripts.add(script_name)
        if ok and not dry_run:
            Path(marker).touch()


# ---------------------------------------------------------------------------
# multi-spec analyses (run after all specs complete)
# ---------------------------------------------------------------------------

def run_multi_analyses(run_dir: str, keys: List[str], dry_run: bool) -> None:
    evals_dir = os.path.join(run_dir, "evals")
    multi_dir = os.path.join(run_dir, "multi_eval")
    print(f"\n--- multi-spec analyses: {keys} ---", flush=True)
    for key in keys:
        script_name, out_subdir = _MULTI_SPEC_SCRIPT_MAP[key]
        out_dir = os.path.join(multi_dir, out_subdir)
        mod = importlib.import_module(script_name)
        _call(mod.run, dry_run, label=script_name, allow_fail=True,
              evals_dir=evals_dir, out_dir=out_dir)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Run fish evals from EVAL_REGISTRY. Run from onpolicy/custom/fish/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("run_dir", help="Path to a training run directory")
    p.add_argument(
        "--specs", nargs="+", metavar="SPEC", default=[],
        help="Run these spec_keys",
    )
    p.add_argument(
        "--group", nargs="+", metavar="GROUP", default=[],
        help=f"Run all specs in named group(s) from SPEC_GROUPS. Known groups: {sorted(SPEC_GROUPS)}",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    p.add_argument(
        "--force", "--force-all", dest="force_all", action="store_true",
        help="Re-run eval, preprocess, and analyses even if outputs or markers exist.",
    )
    p.add_argument(
        "--force-eval", action="store_true",
        help="Re-run eval even if evals/<spec>/raw/agg_flat.pkl exists.",
    )
    p.add_argument(
        "--force-preprocess", action="store_true",
        help="Re-run derived preprocessing from agg_flat.pkl; does not force eval.",
    )
    p.add_argument(
        "--force-analyses", action="store_true",
        help="Re-run analyses even if .analysis_done_* markers exist; does not force eval/preprocess.",
    )
    p.add_argument(
        "--no-flatten", dest="run_flatten", action="store_false", default=True,
        help="Skip preprocess_flatten.py",
    )
    p.add_argument(
        "--no-analyses", dest="run_analyses", action="store_false", default=True,
        help="Skip all figure generation (per-spec and multi-spec)",
    )
    p.add_argument(
        "--analyses", nargs="+", metavar="NAME", default=None,
        dest="analyses_filter",
        help="Restrict to these analysis names (intersected with each spec's configured analyses). "
             f"Known: {', '.join(sorted(_ANALYSIS_SCRIPT_MAP))}",
    )
    p.add_argument(
        "--analyses-add", nargs="+", metavar="NAME", default=None,
        dest="analyses_add",
        help="Also run these analysis names for every spec, even if not in its configured analyses. "
             f"Known: {', '.join(sorted(_ANALYSIS_SCRIPT_MAP))}",
    )
    p.add_argument(
        "--no-multi-spec", dest="run_multi_spec", action="store_false", default=True,
        help="Skip multi-spec analyses (per-spec analyses still run)",
    )
    p.add_argument(
        "--multi-spec", nargs="+", metavar="KEY",
        choices=MULTI_SPEC_ALL, default=None,
        dest="multi_spec_keys",
        help=f"Restrict multi-spec analyses to these keys (default: all). "
             f"Known: {', '.join(MULTI_SPEC_ALL)}",
    )
    p.add_argument(
        "--render-episodes", type=int, default=None, metavar="N",
        help="Override render_episodes (= num_eval_rollouts) for every spec. Must be >=1 or no pkls are written.",
    )
    p.add_argument(
        "--episode-length", type=int, default=None, metavar="N",
        help="Override episode_length for every spec (useful for short test runs)",
    )
    p.add_argument(
        "--eval-rollout-threads", type=int, default=None, metavar="N",
        help="Override n_rollout_threads for eval (total rollouts = threads × render_episodes). Default: registry value (10).",
    )
    args = p.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        sys.exit(f"ERROR: run_dir not found: {run_dir}")

    # Resolve --group keys into spec lists
    unknown_groups = set(args.group) - set(SPEC_GROUPS)
    if unknown_groups:
        sys.exit(f"ERROR: unknown group(s): {sorted(unknown_groups)}. Known: {sorted(SPEC_GROUPS)}")
    group_keys: List[str] = []
    for g in args.group:
        group_keys.extend(SPEC_GROUPS[g])

    # Merge group specs + explicit --specs; preserve order, deduplicate
    all_keys: List[str] = list(dict.fromkeys(group_keys + list(args.specs)))

    if all_keys:
        unknown_specs = set(all_keys) - set(EVAL_REGISTRY)
        if unknown_specs:
            sys.exit(f"ERROR: unknown spec(s): {sorted(unknown_specs)}")
        specs_to_run = [EVAL_REGISTRY[k] for k in all_keys]
    else:
        # No --specs or --group: run all non-nfish specs by default
        specs_to_run = [s for s in EVAL_REGISTRY.values() if not s.spec_key.startswith("nfish")]

    if not specs_to_run:
        sys.exit("ERROR: no specs matched --specs filter")

    multi_keys = args.multi_spec_keys if args.multi_spec_keys else MULTI_SPEC_ALL
    force_eval = args.force_eval or args.force_all
    force_preprocess = args.force_preprocess or args.force_all
    force_analyses = args.force_analyses or args.force_all

    print(f"run_dir:          {run_dir}")
    print(f"specs ({len(specs_to_run)}):        {[spec.spec_key for spec in specs_to_run]}")
    print(f"render_episodes:  {args.render_episodes or '(from registry)'}")
    print(f"episode_length:   {args.episode_length or '(from registry)'}")
    print(f"eval_rollout_threads: {args.eval_rollout_threads or '(from registry)'}")
    print(f"run_analyses:     {args.run_analyses}")
    print(f"analyses_filter:  {args.analyses_filter or '(all)'}")
    print(f"analyses_add:     {args.analyses_add or '(none)'}")
    print(f"force_all:        {args.force_all}")
    print(f"force_eval:       {force_eval}")
    print(f"force_preprocess: {force_preprocess}")
    print(f"force_analyses:   {force_analyses}")
    print(f"multi_spec:       {multi_keys if args.run_multi_spec and args.run_analyses else 'skipped'}")
    print(f"dry_run:          {args.dry_run}")

    t_total = time.perf_counter()
    for spec in specs_to_run:
        print(f"\n--- spec: {spec.spec_key} ---", flush=True)
        t_spec = time.perf_counter()
        run_one_spec(
            run_dir, spec, spec.analyses,
            dry_run=args.dry_run,
            force_eval=force_eval,
            force_preprocess=force_preprocess,
            force_analyses=force_analyses,
            run_flatten=args.run_flatten,
            run_analyses=args.run_analyses,
            analyses_filter=args.analyses_filter,
            analyses_add=args.analyses_add,
            render_episodes_override=args.render_episodes,
            episode_length_override=args.episode_length,
            eval_rollout_threads_override=args.eval_rollout_threads,
        )
        print(f"  [spec done: {time.perf_counter() - t_spec:.1f}s]", flush=True)

    if args.run_analyses and args.run_multi_spec:
        run_multi_analyses(run_dir, multi_keys, args.dry_run)

    print(f"\nAll done. Total: {time.perf_counter() - t_total:.1f}s")


if __name__ == "__main__":
    main()
