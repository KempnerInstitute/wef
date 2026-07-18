#!/usr/bin/env bash
# Short training run: 500k steps, basic specs, quick analyses only.
# Good for development iteration — enough steps to see emergent behavior,
# faster than the full 3M-step run.
#
# Usage:
#   bash scripts/run_short.sh
#   SEEDS="1 2" bash scripts/run_short.sh
#   NUM_TRAIN_STEPS=200000 bash scripts/run_short.sh

set -uo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
RESULTS_PARENT_DIR="${RESULTS_PARENT_DIR:-./}"

NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-500000}"
SEEDS="${SEEDS:-1}"
echo "SEEDS: $SEEDS"

EPISODE_LENGTH=800
N_ROLLOUT_THREADS="${N_ROLLOUT_THREADS:-10}"
RENDER_EPISODES=1
AGENT_SIZE_MODE="random"; TRAIN_DROPOUT=0.0; SIZE_SPEED_EXPONENT=1

SENSING_MODEL_TYPE="${SENSING_MODEL_TYPE:-dynamic}"
SENSOR_NOISE_FRAC=0.05; AMP_CONS_EOD_NOISE_FRAC=0.5
MORMYROMAST_MODE=1; AMPULLARY_MODE=1; KNOLLEN_MODE=1
RNN_TYPE="GRU"; WIDTH=512; GAMMA=0.995; DCL=100
MM=1; ML=2.0; MA=4.0
BASE_FOOD_MULTIPLIER=1.0; FOOD_ORIENTATION_DRIFT=0.1
PRANDOM=5; URANDOM=1; PNPATCH=0
GROUP="${GROUP:-}"  # optional short group name, e.g. GROUP=MyShortRun

for SEED in $SEEDS; do
    DATESTAMP=$(date +"%Y%m%d"); TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    NUM_TRAIN_STEPS_H=$(numfmt --to=si --format="%.0f" $NUM_TRAIN_STEPS)

    AUTO_GROUP_NAME="Short${DATESTAMP}${SENSING_MODEL_TYPE}T${NUM_TRAIN_STEPS_H}FO${FOOD_ORIENTATION_DRIFT}FX${BASE_FOOD_MULTIPLIER}Order${MM}LinearX${ML}AngularX${MA}Gamma${GAMMA}DCL${DCL}TD${TRAIN_DROPOUT}PR${PRANDOM}UR${URANDOM}NP${PNPATCH}A${AMPULLARY_MODE}K${KNOLLEN_MODE}M${MORMYROMAST_MODE}${RNN_TYPE}"
    GROUP_FOLDER_NAME="${GROUP:-${AUTO_GROUP_NAME}}"
    EXP_FOLDER_NAME="${GROUP_FOLDER_NAME}Seed${SEED}"
    RUN_DIR="${RESULTS_PARENT_DIR}/results/${GROUP_FOLDER_NAME}/${EXP_FOLDER_NAME}/${TIMESTAMP}"
    echo "RUN_DIR: $RUN_DIR"

    python train_fish.py --experiment_name "${GROUP_FOLDER_NAME}/${EXP_FOLDER_NAME}" \
        --num_agents 4 \
        --results_parent_dir "$RESULTS_PARENT_DIR" \
        --timestamp "${TIMESTAMP}" \
        --num_env_steps "$NUM_TRAIN_STEPS" \
        --render_episodes "$RENDER_EPISODES" \
        --episode_length "$EPISODE_LENGTH" \
        --max_episode_length "$EPISODE_LENGTH" \
        --n_rollout_threads "$N_ROLLOUT_THREADS" \
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
        --feedback_displacement --gamma $GAMMA --seed $SEED

    # Quick analyses only (general, behavior, eod, idi); skip slow RNN analyses and multi-spec
    # shellcheck disable=SC2086
    python3 pipeline.py "$RUN_DIR" \
        --specs m1a1k1_patchy_square 2fish_m1a1k1_uniform_wide \
        --analyses general behavior eod idi \
        --no-multi-spec

    python3 copy_to_figs.py "$RUN_DIR"
done
