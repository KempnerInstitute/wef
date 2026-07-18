#!/usr/bin/env bash
# Eval-only pipeline on an existing run directory (no training).
# Runs: eval → flatten → features → summaries → analyses → copy_to_figs → figseed.pdf.
#
# Usage:
#   RUN_DIR=/path/to/run bash scripts/run_eval.sh
#   RUN_DIRS="/run1 /run2" bash scripts/run_eval.sh
#   RUN_EVAL_NFISH=0 RUN_DIR=/path/to/run bash scripts/run_eval.sh

set -uo pipefail

RUN_DIR="${RUN_DIR:-${1:-}}"
RUN_DIRS="${RUN_DIRS:-}"

if [[ -z "$RUN_DIR" && -z "$RUN_DIRS" ]]; then
    echo "ERROR: Set RUN_DIR=/path/to/run or RUN_DIRS=\"/r1 /r2\"."
    exit 1
fi

RUN_EVAL_BASIC="${RUN_EVAL_BASIC:-1}"
RUN_EVAL_2F1P="${RUN_EVAL_2F1P:-1}"
RUN_EVAL_1RW1F1P="${RUN_EVAL_1RW1F1P:-1}"
RUN_COMPARISON="${RUN_COMPARISON:-1}"
RUN_EVAL_2FSQUARE="${RUN_EVAL_2FSQUARE:-1}"
RUN_EVAL_NFISH="${RUN_EVAL_NFISH:-1}"
RUN_EVAL_NFISH_K0="${RUN_EVAL_NFISH_K0:-1}"   # knollen-off nfish for k0 vs k1 comparison in analysis_nfish
RUN_EVAL_2FWIDE="${RUN_EVAL_2FWIDE:-1}"
RUN_EVAL_NPATCH="${RUN_EVAL_NPATCH:-1}"        # m1a1k1_1patch: "num_patches" group in analysis_interventions
RUN_EVAL_FOOD05="${RUN_EVAL_FOOD05:-1}"        # food05_*: "sensor_food05" group in analysis_interventions
RUN_EVAL_FOOD025="${RUN_EVAL_FOOD025:-1}"      # food025_*: "sensor_food025"/"food_abundance" groups
RUN_EVAL_SMALL_CS="${RUN_EVAL_SMALL_CS:-1}"    # small_cs0/1/2: "collective_sensing" group
RUN_EVAL_2F1P_K0="${RUN_EVAL_2F1P_K0:-1}"     # 2f1p_k0_*: knollen-off 2f1p for analysis_2f1p_k0_multispec + _k0k1_compare
RUN_EVAL_FOOD_GRID="${RUN_EVAL_FOOD_GRID:-1}"  # iso_p*/free_p*: patch sweep for analysis_food_grid_iso

build_specs() {
    local specs=""
    [[ $RUN_EVAL_BASIC      -eq 1 ]] && specs="$specs m1a1k1_patchy_square m1a1k1_uniform_square"
    [[ $RUN_COMPARISON      -eq 1 ]] && specs="$specs m1a1k1_patchy_square m1a1k1_uniform_square \
        m0a1k1_patchy_square m1a0k1_patchy_square m1a1k0_patchy_square m0a0k1_patchy_square m0a0k0_patchy_square \
        1fish_m0a1k1_patchy_square 1fish_m1a0k1_patchy_square 1fish_m1a1k0_patchy_square 1fish_m1a1k1_patchy_square \
        m1a0k1_uniform_square"
    [[ $RUN_EVAL_2FWIDE     -eq 1 ]] && specs="$specs 2fish_m1a1k1_uniform_wide"
    [[ $RUN_EVAL_2FSQUARE   -eq 1 ]] && specs="$specs 2fish_m1a1k1_uniform_square"
    [[ $RUN_EVAL_NFISH      -eq 1 ]] && specs="$specs nfish1_m1a1k1_patchy_square nfish2_m1a1k1_patchy_square nfish3_m1a1k1_patchy_square nfish4_m1a1k1_patchy_square"
    [[ $RUN_EVAL_NFISH_K0   -eq 1 ]] && specs="$specs nfish1_m1a1k0_patchy_square nfish2_m1a1k0_patchy_square nfish3_m1a1k0_patchy_square nfish4_m1a1k0_patchy_square"
    [[ $RUN_EVAL_2F1P       -eq 1 ]] && specs="$specs 2f1p_AltB 2f1p_AeqB 2f1p_AgtB 2f1p_control_a 2f1p_control_b"
    [[ $RUN_EVAL_2F1P_K0    -eq 1 ]] && specs="$specs 2f1p_k0_AltB 2f1p_k0_AeqB 2f1p_k0_AgtB 2f1p_k0_control_a 2f1p_k0_control_b"
    [[ $RUN_EVAL_1RW1F1P    -eq 1 ]] && specs="$specs 1rw1f1p_grid"
    [[ $RUN_EVAL_NPATCH     -eq 1 ]] && specs="$specs m1a1k1_1patch"
    [[ $RUN_EVAL_FOOD05     -eq 1 ]] && specs="$specs food05_m1a1k1_patchy_square food05_m0a1k1_patchy_square food05_m1a0k1_patchy_square food05_m1a1k0_patchy_square"
    [[ $RUN_EVAL_FOOD025    -eq 1 ]] && specs="$specs food025_m1a1k1_patchy_square food025_m0a1k1_patchy_square food025_m1a0k1_patchy_square food025_m1a1k0_patchy_square"
    [[ $RUN_EVAL_SMALL_CS   -eq 1 ]] && specs="$specs small_cs0 small_cs1 small_cs2"
    [[ $RUN_EVAL_FOOD_GRID  -eq 1 ]] && specs="$specs iso_p1_m1 iso_p2_m05 iso_p3_m0333 iso_p4_m025 iso_p5_m02 free_p2_m1 free_p3_m1 free_p4_m1 free_p5_m1"
    echo "$specs"
}

do_one() {
    local rd="$1"
    local specs
    specs=$(build_specs)

    echo "============================================================"
    echo "Processing: $rd"
    echo "============================================================"

    if [[ -z "$specs" ]]; then
        echo "No eval specs selected — all RUN_EVAL_* flags are 0."
        return
    fi

    # shellcheck disable=SC2086
    python3 pipeline.py "$rd" --specs $specs --no-multi-spec \
        --analyses general behavior eod idi pairwise biting_network bitten_network food_distribution rollout_diagnostics 1f1rw1p

    # multi-spec (interventions, 2f1p, nfish): per-spec analyses skipped via done markers
    # shellcheck disable=SC2086
    python3 pipeline.py "$rd" --specs $specs --no-flatten

    # rnn_psd and rnn_plsc are restricted to 2fish_m1a1k1_uniform_square via SPEC_ANALYSES in eval_registry.py
    # shellcheck disable=SC2086
    python3 pipeline.py "$rd" --specs $specs --no-multi-spec \
        --analyses rnn_dim rnn_psd rnn_plsc decoding

    python3 copy_to_figs.py "$rd"
    bash scripts/regenerate_latex.sh "$rd"
}

[[ -n "$RUN_DIR"  ]] && do_one "$RUN_DIR"
if [[ -n "$RUN_DIRS" ]]; then
    for rd in $RUN_DIRS; do do_one "$rd"; done
fi
