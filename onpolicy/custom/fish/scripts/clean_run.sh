#!/usr/bin/env bash
# Clean analysis outputs from a run directory so the pipeline can be re-run.
#
# Usage:
#   bash scripts/clean_run.sh <run_dir> [--level N] [--dry-run] [--yes]
#
# Levels:
#   1  (default) analyses + figs only
#            deletes: evals/*/analyses/, evals/*/.analysis_done_*, multi_eval/, figs/
#   2  + derived features
#            also deletes: evals/*/derived/
#   3  + flattened pkl
#            also deletes: evals/*/raw/agg_flat.pkl
#   4  + all raw eval data (requires full re-eval from the policy checkpoint)
#            also deletes: evals/
#
# --dry-run  Print what would be removed without deleting anything.
# --yes      Skip confirmation prompt.

set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
LEVEL=1
DRY_RUN=false
YES=false
RUN_DIR=""

# ── arg parse ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --level)   LEVEL="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --yes)     YES=true; shift ;;
        -*)        echo "Unknown flag: $1" >&2; exit 1 ;;
        *)         RUN_DIR="$1"; shift ;;
    esac
done

if [[ -z "$RUN_DIR" ]]; then
    echo "Usage: bash scripts/clean_run.sh <run_dir> [--level 1-4] [--dry-run] [--yes]" >&2
    exit 1
fi

RUN_DIR="$(realpath "$RUN_DIR")"
if [[ ! -d "$RUN_DIR" ]]; then
    echo "ERROR: run_dir not found: $RUN_DIR" >&2
    exit 1
fi

if [[ ! "$LEVEL" =~ ^[1-4]$ ]]; then
    echo "ERROR: --level must be 1, 2, 3, or 4" >&2
    exit 1
fi

# ── build target list ─────────────────────────────────────────────────────────
# Each entry: "label|path_or_pattern|type"  (type: dir | file | glob)
TARGETS=()

# Level 1: analyses + figs
TARGETS+=("analyses dirs|${RUN_DIR}/evals/*/analyses|dir-glob")
TARGETS+=("analysis_done markers|${RUN_DIR}/evals/*/.analysis_done_*|file-glob")
TARGETS+=("multi_eval/|${RUN_DIR}/multi_eval|dir")
TARGETS+=("figs/|${RUN_DIR}/figs|dir")

if [[ $LEVEL -ge 2 ]]; then
    TARGETS+=("derived/ dirs|${RUN_DIR}/evals/*/derived|dir-glob")
fi

if [[ $LEVEL -ge 3 ]]; then
    TARGETS+=("agg_flat.pkl files|${RUN_DIR}/evals/*/raw/agg_flat.pkl|file-glob")
fi

if [[ $LEVEL -ge 4 ]]; then
    # Replace the individual evals/* entries with the whole evals/ dir
    TARGETS+=("evals/ (ALL raw rollout data)|${RUN_DIR}/evals|dir")
fi

# ── preview ───────────────────────────────────────────────────────────────────
echo "run_dir : $RUN_DIR"
echo "level   : $LEVEL"
$DRY_RUN && echo "(dry run — nothing will be deleted)"
echo

echo "Targets:"
ANY=false
for entry in "${TARGETS[@]}"; do
    label="${entry%%|*}"
    rest="${entry#*|}"
    pattern="${rest%%|*}"
    type="${rest##*|}"

    case "$type" in
        dir)
            if [[ -d "$pattern" ]]; then
                echo "  [dir]  $pattern"
                ANY=true
            else
                echo "  [dir]  $pattern  (not found — skip)"
            fi
            ;;
        dir-glob)
            found=false
            for p in $pattern; do
                [[ -d "$p" ]] && echo "  [dir]  $p" && ANY=true && found=true
            done
            $found || echo "  [dir]  $pattern  (no matches — skip)"
            ;;
        file-glob)
            found=false
            for p in $pattern; do
                [[ -e "$p" ]] && echo "  [file] $p" && ANY=true && found=true
            done
            $found || echo "  [file] $pattern  (no matches — skip)"
            ;;
    esac
done

if ! $ANY; then
    echo
    echo "Nothing to delete."
    exit 0
fi

# ── confirm ───────────────────────────────────────────────────────────────────
if ! $DRY_RUN; then
    if ! $YES; then
        echo
        printf "Delete the above? [y/N] "
        read -r reply
        [[ "$reply" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
    fi

    # ── delete ────────────────────────────────────────────────────────────────
    echo
    for entry in "${TARGETS[@]}"; do
        rest="${entry#*|}"
        pattern="${rest%%|*}"
        type="${rest##*|}"

        case "$type" in
            dir)
                if [[ -d "$pattern" ]]; then
                    rm -rf "$pattern"
                    echo "  removed $pattern"
                fi
                ;;
            dir-glob)
                for p in $pattern; do
                    if [[ -d "$p" ]]; then
                        rm -rf "$p"
                        echo "  removed $p"
                    fi
                done
                ;;
            file-glob)
                for p in $pattern; do
                    if [[ -e "$p" ]]; then
                        rm -f "$p"
                        echo "  removed $p"
                    fi
                done
                ;;
        esac
    done
    echo
    echo "Done."
fi
