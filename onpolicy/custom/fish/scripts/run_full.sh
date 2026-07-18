#!/usr/bin/env bash
# Full training run: 3M steps, all eval specs, quick + slow analyses.
# Canonical production run for paper figures.
#
# Usage:
#   bash scripts/run_full.sh
#   SEEDS="1 2 3" bash scripts/run_full.sh
#   NUM_TRAIN_STEPS=1000000 bash scripts/run_full.sh
#   RUN_EVAL_NFISH=0 RUN_EVAL_2F1P=0 bash scripts/run_full.sh

set -uo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
RESULTS_PARENT_DIR="${RESULTS_PARENT_DIR:-./}"

NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-5000000}"
SEEDS="${SEEDS:-1 2}"
echo "SEEDS: $SEEDS"

RENDER_EPISODES="${RENDER_EPISODES:-1}"
EPISODE_LENGTH="${EPISODE_LENGTH:-1600}"
N_ROLLOUT_THREADS="${N_ROLLOUT_THREADS:-16}"
AGENT_SIZE_MODE="random"; TRAIN_DROPOUT=0.0; SIZE_SPEED_EXPONENT=1

SENSING_MODEL_TYPE="${SENSING_MODEL_TYPE:-dynamic}"
SENSOR_NOISE_FRAC="${SENSOR_NOISE_FRAC:-0.05}"
AMP_CONS_EOD_NOISE_FRAC="${AMP_CONS_EOD_NOISE_FRAC:-1.0}"
MORMYROMAST_MODE=1; KNOLLEN_MODE="${KNOLLEN_MODE:-1}"; AMPULLARY_MODE=1
KNOLLEN_PROCESSING="${KNOLLEN_PROCESSING:-binarize}"

RNN_TYPE="GRU"; WIDTH=512; GAMMA=0.995; DCL=100
MM=1; ML=2.0; MA=4.0
BASE_FOOD_MULTIPLIER=1.0; FOOD_ORIENTATION_DRIFT="${FOOD_ORIENTATION_DRIFT:-0.1}"
ELECTRIC_BACKEND="${ELECTRIC_BACKEND:-original}"
export ELECTRIC_BACKEND
BE_TAG=""; [[ "$ELECTRIC_BACKEND" != "original" ]] && BE_TAG="BEnumba"
EAT_COOLDOWN_RATE="${EAT_COOLDOWN_RATE:-}"
ECR_TAG=""; [[ -n "$EAT_COOLDOWN_RATE" ]] && ECR_TAG="ECR${EAT_COOLDOWN_RATE}"
GROUP="${GROUP:-}"  # optional short group name, e.g. GROUP=K2_Enum
PRANDOM=5; URANDOM=1; PNPATCH=0

RUN_EVAL_BASIC="${RUN_EVAL_BASIC:-1}"
RUN_EVAL_2F1P="${RUN_EVAL_2F1P:-1}"
RUN_EVAL_1RW1F1P="${RUN_EVAL_1RW1F1P:-1}"
RUN_COMPARISON="${RUN_COMPARISON:-1}"
RUN_EVAL_2FSQUARE="${RUN_EVAL_2FSQUARE:-1}"
RUN_EVAL_NFISH="${RUN_EVAL_NFISH:-1}"
RUN_EVAL_NFISH_K0="${RUN_EVAL_NFISH_K0:-1}"
RUN_EVAL_2FWIDE="${RUN_EVAL_2FWIDE:-1}"
RUN_EVAL_NPATCH="${RUN_EVAL_NPATCH:-1}"
RUN_EVAL_FOOD05="${RUN_EVAL_FOOD05:-1}"
RUN_EVAL_FOOD025="${RUN_EVAL_FOOD025:-1}"
RUN_EVAL_SMALL_CS="${RUN_EVAL_SMALL_CS:-1}"
RUN_EVAL_2F1P_K0="${RUN_EVAL_2F1P_K0:-1}"
RUN_EVAL_FOOD_GRID="${RUN_EVAL_FOOD_GRID:-1}"

for SEED in $SEEDS; do
    DATESTAMP=$(date +"%Y%m%d"); TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    NUM_TRAIN_STEPS_H=$(numfmt --to=si --format="%.0f" $NUM_TRAIN_STEPS)

    AUTO_GROUP_NAME="ConsNoise${DATESTAMP}${SENSING_MODEL_TYPE}T${NUM_TRAIN_STEPS_H}FO${FOOD_ORIENTATION_DRIFT}FX${BASE_FOOD_MULTIPLIER}Order${MM}LinearX${ML}AngularX${MA}Gamma${GAMMA}DCL${DCL}TD${TRAIN_DROPOUT}PR${PRANDOM}UR${URANDOM}NP${PNPATCH}A${AMPULLARY_MODE}K${KNOLLEN_MODE}M${MORMYROMAST_MODE}${RNN_TYPE}${BE_TAG}${ECR_TAG}"
    GROUP_FOLDER_NAME="${GROUP:-${AUTO_GROUP_NAME}}"
    EXP_FOLDER_NAME="${GROUP_FOLDER_NAME}Seed${SEED}"
    RUN_DIR="${RESULTS_PARENT_DIR}/results/${GROUP_FOLDER_NAME}/${EXP_FOLDER_NAME}/${TIMESTAMP}"
    echo "RUN_DIR: $RUN_DIR"

    python train_fish.py --experiment_name "${GROUP_FOLDER_NAME}/${EXP_FOLDER_NAME}" \
        --num_agents 4 \
        --results_parent_dir "$RESULTS_PARENT_DIR" \
        --timestamp "${TIMESTAMP}" \
        --num_env_steps "$NUM_TRAIN_STEPS" \
        --episode_length "$EPISODE_LENGTH" \
        --max_episode_length "$EPISODE_LENGTH" \
        --n_rollout_threads "$N_ROLLOUT_THREADS" \
        --render_episodes "$RENDER_EPISODES" \
        --num_render_envs "$N_ROLLOUT_THREADS" \
        --allow_aggression 1 --agent_size_mode "$AGENT_SIZE_MODE" \
        --weight_decay 0.000001 \
        --sensing_model_type "$SENSING_MODEL_TYPE" \
        --noise_frac_morm "$SENSOR_NOISE_FRAC" \
        --noise_frac_amp "$SENSOR_NOISE_FRAC" \
        --noise_frac_amp_cons_eod "$AMP_CONS_EOD_NOISE_FRAC" \
        --noise_frac_knollen "$SENSOR_NOISE_FRAC" \
        --size_speed_exponent $SIZE_SPEED_EXPONENT \
        --train_sensor_dropout_p $TRAIN_DROPOUT \
        --p_init_closeby 0.0 \
        --base_food_multiplier "${BASE_FOOD_MULTIPLIER}" \
        --hidden_size $WIDTH \
        --max_food_sensing_radius 15 \
        --save_interval 1000 \
        --train_food_scaling_min 0.5 \
        --train_food_scaling_max 2 \
        --use_bite_cooldown \
        --mormyromast_mode $MORMYROMAST_MODE \
        --ampullary_mode $AMPULLARY_MODE \
        --knollen_mode $KNOLLEN_MODE \
        --knollen_processing $KNOLLEN_PROCESSING \
        --knollen_metadata_mode relative \
        --collective_sensing_mode 1 \
        --data_chunk_length $DCL \
        --use_orthogonal \
        --rnn_type $RNN_TYPE \
        --motion_order $MM \
        --multiplier_linear $ML \
        --multiplier_angular $MA \
        --ampullary_intrinsic_only \
        --auxs eods \
        --asym_eating \
        --food_orientation_drift $FOOD_ORIENTATION_DRIFT \
        --penalize_effort_over_frac 0.5 \
        --prandom "$PRANDOM" --urandom "$URANDOM" --prob_n_patch $PNPATCH --pfeeder 0 \
        --feedback_displacement --gamma $GAMMA --seed $SEED \
        ${EAT_COOLDOWN_RATE:+--eat_cooldown_rate "$EAT_COOLDOWN_RATE"}

    RUN_DIR="$RUN_DIR" \
    RUN_EVAL_BASIC=$RUN_EVAL_BASIC \
    RUN_EVAL_2F1P=$RUN_EVAL_2F1P \
    RUN_EVAL_1RW1F1P=$RUN_EVAL_1RW1F1P \
    RUN_COMPARISON=$RUN_COMPARISON \
    RUN_EVAL_2FSQUARE=$RUN_EVAL_2FSQUARE \
    RUN_EVAL_NFISH=$RUN_EVAL_NFISH \
    RUN_EVAL_NFISH_K0=$RUN_EVAL_NFISH_K0 \
    RUN_EVAL_2FWIDE=$RUN_EVAL_2FWIDE \
    RUN_EVAL_NPATCH=$RUN_EVAL_NPATCH \
    RUN_EVAL_FOOD05=$RUN_EVAL_FOOD05 \
    RUN_EVAL_FOOD025=$RUN_EVAL_FOOD025 \
    RUN_EVAL_SMALL_CS=$RUN_EVAL_SMALL_CS \
    RUN_EVAL_2F1P_K0=$RUN_EVAL_2F1P_K0 \
    RUN_EVAL_FOOD_GRID=$RUN_EVAL_FOOD_GRID \
    bash scripts/run_eval.sh
done
