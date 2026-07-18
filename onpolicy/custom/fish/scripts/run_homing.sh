#!/usr/bin/env bash
# Homing task: train a model to home to target, then run homing eval pipeline.
# Pass RUN_DIR to skip training and re-run eval/analysis on an existing run.
#
# Usage:
#   bash scripts/run_homing.sh
#   SEEDS="1 2 3" bash scripts/run_homing.sh
#   RUN_DIR=/path/to/run bash scripts/run_homing.sh
#   HOMING2=1 bash scripts/run_homing.sh   # always-frozen target variant

set -uo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1
RESULTS_PARENT_DIR="${RESULTS_PARENT_DIR:-./}"
POST_TRAIN_RUN_DIR="${1:-${RUN_DIR:-}}"

SEEDS="${SEEDS:-1}"
echo "SEEDS: $SEEDS"

SENSING_MODEL_TYPE="${SENSING_MODEL_TYPE:-dynamic}"
SENSOR_NOISE_FRAC="${SENSOR_NOISE_FRAC:-0.05}"
AMP_CONS_EOD_NOISE_FRAC="${AMP_CONS_EOD_NOISE_FRAC:-1.0}"
BASE_FOOD_MULTIPLIER=0.0
EPISODE_LENGTH="${EPISODE_LENGTH:-800}"
N_ROLLOUT_THREADS="${N_ROLLOUT_THREADS:-10}"
EVAL_RENDER_EPISODES="${EVAL_RENDER_EPISODES:-10}"
EVAL_ROLLOUT_THREADS="${EVAL_ROLLOUT_THREADS:-$N_ROLLOUT_THREADS}"
AGENT_SIZE_MODE="${AGENT_SIZE_MODE:-AeqB}"
KNOLLEN_MODE="${KNOLLEN_MODE:-1}"
KNOLLEN_PROCESSING="${KNOLLEN_PROCESSING:-binarize}"
ELECTRIC_BACKEND="${ELECTRIC_BACKEND:-original}"
export ELECTRIC_BACKEND
GROUP="${GROUP:-}"  # optional short group name, e.g. GROUP=HomingFrac

# Homing2 variant: always freeze the target during training (fixes train/eval mismatch).
# Only difference from standard: --always_freeze_agent0.
HOMING2="${HOMING2:-0}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-2000000}"
RNN_TYPE="GRU"
ALWAYS_FREEZE_FLAGS=""
[[ "$HOMING2" == "1" ]] && ALWAYS_FREEZE_FLAGS="--always_freeze_agent0"
echo "HOMING2: $HOMING2  NUM_TRAIN_STEPS: $NUM_TRAIN_STEPS  RNN_TYPE: $RNN_TYPE"

run_post_training_steps() {
    local rd="$1"
    python3 pipeline.py "$rd" \
        --group homing \
        --render-episodes "$EVAL_RENDER_EPISODES" \
        --episode-length "$EPISODE_LENGTH" \
        --eval-rollout-threads "$EVAL_ROLLOUT_THREADS" \
        --no-multi-spec

    mkdir -p "$rd/figs"
    find $rd/evals/ \( -name "*.png" -o -name "*.pdf" \) -exec cp {} "$rd/figs/" \;
}

if [[ -n "$POST_TRAIN_RUN_DIR" ]]; then
    echo "RUN_DIR provided — skipping training."
    run_post_training_steps "$POST_TRAIN_RUN_DIR"
    exit 0
fi

for SEED in $SEEDS; do
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S"); DATESTAMP=$(date +"%Y%m%d")
    NUM_TRAIN_STEPS_H=$(numfmt --to=si --format="%.0f" $NUM_TRAIN_STEPS)

    HOMING_VARIANT="${HOMING2:+2}"  # appends "2" to "Homing" when HOMING2=1
    AUTO_GROUP_NAME="Homing${HOMING_VARIANT:-}${DATESTAMP}ConsNoise${AMP_CONS_EOD_NOISE_FRAC}${SENSING_MODEL_TYPE}T${NUM_TRAIN_STEPS_H}Food${BASE_FOOD_MULTIPLIER}"
    GROUP_FOLDER_NAME="${GROUP:-${AUTO_GROUP_NAME}}"
    EXP_FOLDER_NAME="${GROUP_FOLDER_NAME}Seed${SEED}"
    RUN_DIR="${RESULTS_PARENT_DIR}/results/${GROUP_FOLDER_NAME}/${EXP_FOLDER_NAME}/${TIMESTAMP}"
    echo "RUN_DIR: $RUN_DIR"

    python train_fish.py --experiment_name "${GROUP_FOLDER_NAME}/${EXP_FOLDER_NAME}" \
        --results_parent_dir "$RESULTS_PARENT_DIR" \
        --timestamp "${TIMESTAMP}" \
        --num_env_steps "$NUM_TRAIN_STEPS" \
        --render_episodes 0 \
        --episode_length "$EPISODE_LENGTH" \
        --max_episode_length "$EPISODE_LENGTH" \
        --n_rollout_threads "$N_ROLLOUT_THREADS" \
        --sensing_model_type "$SENSING_MODEL_TYPE" \
        --noise_frac_morm "$SENSOR_NOISE_FRAC" \
        --noise_frac_amp "$SENSOR_NOISE_FRAC" \
        --noise_frac_amp_cons_eod "$AMP_CONS_EOD_NOISE_FRAC" \
        --noise_frac_knollen "$SENSOR_NOISE_FRAC" \
        --pfeeder 0 --prandom 0 --urandom 1 \
        --allow_aggression 0 --agent_size_mode "$AGENT_SIZE_MODE" \
        --homing_mode --required_homing_steps 5 \
        --train_sensor_dropout_p 0.0 \
        --p_init_closeby 0.0 \
        --base_food_multiplier "${BASE_FOOD_MULTIPLIER}" \
        --hidden_size 512 \
        --max_food_sensing_radius 15 \
        --save_interval 1000 \
        --use_bite_cooldown \
        --knollen_metadata_mode relative \
        --asym_eating \
        --use_orthogonal \
        --rnn_type $RNN_TYPE \
        --knollen_mode $KNOLLEN_MODE \
        --knollen_processing $KNOLLEN_PROCESSING \
        --ampullary_intrinsic_only \
        --auxs eods \
        --penalize_effort_over_frac 0.5 \
        --feedback_displacement --gamma 0.995 --weight_decay 0.000001 \
        $ALWAYS_FREEZE_FLAGS \
        --seed $SEED

    run_post_training_steps "$RUN_DIR"
done
