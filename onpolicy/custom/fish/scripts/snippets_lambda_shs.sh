# Not ready for prime-time
echo "Not for batch execution; copy-paste to run..."
exit 0

# ============================================================
# UTILITIES
# ============================================================

rsync -avz --progress results/ /srv/marl/${USER}/marl_fish/
# Delete older than 240 days (8 months)
find /srv/marl/${USER}/marl_fish/ -maxdepth 1 -mindepth 1 -type d -mtime +240 -print # -exec rm -rf {} \;

# Extract timestamp prefix from filenames
find /srv/marl/${USER:-$(whoami)}/ -name "*start_success_rate_heatmap*.png" \
  | grep -o 'MAFish_[0-9]\+_[0-9]\+_' | sort -u

# Git log a single file by month
git log --since="2025-04-01" --until="2025-05-01" \
  -p --pretty=format:"---%n%H|%an|%ad|%s%n" --date=short -- MAEFish.py \
  > MAEFish_04.txt

for m in {01..12}; do
    since="2025-$m-01"
    until=$(date -d "$since +1 month" +%Y-%m-%d)
    echo "Processing $since → $until"
    git log --since="$since" --until="$until" \
        -p --pretty=format:"---%n%H|%an|%ad|%s%n" --date=short \
        -- MAEFish.py > "MAEFish_${m}.txt"
done

git log -p --pretty=format:"---%n%H|%an|%ad|%s%n" --date=short -- ../../algorithms/utils/*.py > attn_log.txt


# ============================================================
# PIPELINE — canonical train → eval → analysis workflow
# ============================================================

RUN_DIR=results/20260201Degen.../20260201_193106/

# Full pipeline for all registered specs (eval → flatten → features → summaries → analyses)
python3 pipeline.py $RUN_DIR

# Dry run first to see what will execute
python3 pipeline.py $RUN_DIR --dry-run

# Specific specs only
python3 pipeline.py $RUN_DIR --specs m1a1k1_patchy_square m1a1k1_uniform_square

# By group
python3 pipeline.py $RUN_DIR --group ablations

# Override render_episodes (number of eval rollouts written as pkls)
python3 pipeline.py $RUN_DIR --render-episodes 15

# Skip analyses — just flatten + features + summaries
python3 pipeline.py $RUN_DIR --no-analyses

# Skip flattening (eval data already present) — features + summaries + analyses only
python3 pipeline.py $RUN_DIR --no-flatten --specs m1a1k1_patchy_square

# Skip multi-spec analyses (interventions / 2f1p / 1f1rw1p / nfish)
python3 pipeline.py $RUN_DIR --no-multi-spec

# Run only specific analyses (filter must match names in spec's SPEC_ANALYSES list)
python3 pipeline.py $RUN_DIR --specs m1a1k1_patchy_square --no-multi-spec --analyses general behavior
python3 pipeline.py $RUN_DIR --specs m1a1k1_patchy_square --no-multi-spec --analyses rollout_diagnostics
python3 pipeline.py $RUN_DIR --specs m1a1k1_patchy_square --no-multi-spec --analyses rnn_dim rnn_psd rnn_plsc decoding

# Fast analyses only — skip slow RNN analyses (recommended for local runs)
python3 pipeline.py $RUN_DIR --no-multi-spec \
  --analyses general behavior eod idi pairwise biting_network bitten_network \
            group_spacing food_distribution rollout_diagnostics

# Force re-run all stages even if outputs exist (WARNING: re-triggers eval — needs logs/all_args.json)
python3 pipeline.py $RUN_DIR --specs m1a1k1_patchy_square --force-all

# Re-run analyses only, ignoring .analysis_done_* markers without forcing eval/preprocess
python3 pipeline.py $RUN_DIR --specs m1a1k1_patchy_square --no-multi-spec \
  --analyses rollout_diagnostics --force-analyses

# Re-run derived preprocessing only from existing agg_flat.pkl
python3 pipeline.py $RUN_DIR --specs m1a1k1_patchy_square --no-multi-spec \
  --no-analyses --force-preprocess


# ============================================================
# PER-SPEC ANALYSES — run one analysis script directly
# ============================================================

SPEC_DIR=$RUN_DIR/evals/m1a1k1_patchy_square

# Core / behaviour
python3 analysis_general.py           --spec_dir $SPEC_DIR
python3 analysis_behavior.py          --spec_dir $SPEC_DIR
python3 analysis_eod.py               --spec_dir $SPEC_DIR
python3 analysis_idi.py               --spec_dir $SPEC_DIR

# Social / spatial
python3 analysis_pairwise.py          --spec_dir $SPEC_DIR
python3 analysis_bite_network.py      --spec_dir $SPEC_DIR  # biter + victim plot sets
python3 analysis_food_distribution.py --spec_dir $SPEC_DIR

# Homing (separate spec; agent_id=0 is target, agent_id=1 is homing agent)
python3 analysis_homing.py            --spec_dir $RUN_DIR/evals/homing

# Sensor diagnostics / observation sanity checks
python3 analysis_rollout_diagnostics.py --spec_dir $SPEC_DIR

# RNN (slow — skip locally unless specifically needed)
python3 analysis_rnn_dim.py           --spec_dir $SPEC_DIR
python3 analysis_rnn_psd.py           --spec_dir $SPEC_DIR
python3 analysis_rnn_plsc.py          --spec_dir $SPEC_DIR
python3 analysis_rnn_decoding.py      --spec_dir $SPEC_DIR


# ============================================================
# MULTI-SPEC ANALYSES — cross-spec comparisons
# ============================================================

python3 analysis_interventions.py  --evals_dir $RUN_DIR/evals --out_dir $RUN_DIR/multi_eval/interventions
python3 analysis_2f1p_multispec.py --evals_dir $RUN_DIR/evals --out_dir $RUN_DIR/multi_eval/2f1p
python3 analysis_1f1rw1p.py        --evals_dir $RUN_DIR/evals --out_dir $RUN_DIR/multi_eval/1f1rw1p
python3 analysis_nfish.py          --evals_dir $RUN_DIR/evals --out_dir $RUN_DIR/multi_eval/nfish


# ============================================================
# PREPROCESS — individual steps (when not using pipeline.py)
# ============================================================

FLAT_PKL=$RUN_DIR/evals/m1a1k1_patchy_square/raw/agg_flat.pkl
DERIVED_DIR=$RUN_DIR/evals/m1a1k1_patchy_square/derived/
RAW_DIR=$RUN_DIR/evals/m1a1k1_patchy_square/raw/

# preprocess_flatten: positional arg is run_dir; --outputs_folder is the raw subdir
python3 preprocess_flatten.py $RUN_DIR --outputs_folder $RAW_DIR
python3 preprocess_features.py $FLAT_PKL --output_dir $DERIVED_DIR
python3 preprocess_summaries.py $DERIVED_DIR/per_env_ep_agent_step.pkl --output_dir $DERIVED_DIR

# Batch: flatten all timestamp runs under an experiment dir (rmappo-... level)
EXP_DIR=results/.../
python3 preprocess_all_exps_in_dir.py $EXP_DIR


# ============================================================
# ONE-OFF EVAL + PIPELINE (custom spec not in registry)
# ============================================================

RUN_DIR=results/.../20260211_092450/
EVAL_RUN_NAME="OneOff50x50Food1x"

EVAL_RENDER_EPISODES=10
EVAL_NUM_VIDS_TO_SAVE=0
EVAL_ROLLOUT_THREADS=10
EVAL_EP_LENGTH=1200
EVAL_ALLOW_AGGRESSION=1
EVAL_SEED=1
EVAL_COLLSENSE_MODE=1
EVAL_MORMYROMAST_MODE=1
EVAL_AMPULLARY_MODE=1
EVAL_KNOLLEN_MODE=1
BASE_FOOD_MULTIPLIER=1.0
EVAL_TASK="foraging"

python3 eval_fish.py $RUN_DIR \
    --n_rollout_threads $EVAL_ROLLOUT_THREADS \
    --eval_episode_length $EVAL_EP_LENGTH \
    --eval_render_episodes $EVAL_RENDER_EPISODES \
    --save_vids --num_vids_to_save $EVAL_NUM_VIDS_TO_SAVE \
    --eval_run_name $EVAL_RUN_NAME \
    --eval_seed $EVAL_SEED \
    --task $EVAL_TASK \
    --eval_pfeeder 0 \
    --eval_prandom 1 \
    --eval_urandom 1 \
    --eval_collective_sensing_mode $EVAL_COLLSENSE_MODE \
    --eval_mormyromast_mode $EVAL_MORMYROMAST_MODE \
    --eval_ampullary_mode $EVAL_AMPULLARY_MODE \
    --eval_knollen_mode $EVAL_KNOLLEN_MODE \
    --eval_allow_aggression $EVAL_ALLOW_AGGRESSION \
    --eval_base_food_multiplier ${BASE_FOOD_MULTIPLIER} \
    --eval_mute_k 0 \
    --eval_arena_size_max "(50,50)" \
    --eval_arena_size_min "(50,50)" \
    --eval_active_agent_ids "(0,1)"

# Flatten + features + summaries + analyses (skips eval since flat pkl now exists)
python3 pipeline.py $RUN_DIR --specs $EVAL_RUN_NAME


# ============================================================
# ANIMATIONS (developer tools, not in pipeline)
# ============================================================

python3 anim_rnn.py --pkl $RUN_DIR/evals/m1a1k1_patchy_square/raw/agg_flat.pkl --pca --umap

python3 anim_stack_bhv_rnn.py \
  --behavior $RUN_DIR/evals/m1a1k1cs1sd0patchydiverse/raw/MAFish_behavior_*.mp4 \
  --rnn ./neural_pca_agent_collage_ep0_env0.mp4 \
  --output ./MAFish_behavior_collage.mp4


# ============================================================
# TEST ENVIRONMENT
# ============================================================

# Test environment in parallel
for i in $(seq 5); do python MAEFish.py & done; wait


# ============================================================
# TRAIN
# ============================================================

python train_fish.py

python train_fish.py --num_agents 4

# Shorter training for testing
python train_fish.py --experiment_name 1K --num_env_steps 1000 --episode_length 20 \
  --render_episodes 2 --n_rollout_threads 12 --weight_decay 0.000001

python train_fish.py --experiment_name 10K --num_env_steps 10000 --episode_length 400 \
  --render_episodes 2 --n_rollout_threads 12 --weight_decay 0.000001 --save_interval 100

python train_fish.py --experiment_name 100K --num_env_steps 100000 --episode_length 400 \
  --render_episodes 2 --n_rollout_threads 12 --weight_decay 0.000001

python train_fish.py --experiment_name 1M --num_env_steps 1000000 --episode_length 400 \
  --render_episodes 2 --weight_decay 0.000001 --n_rollout_threads 12 --backwards \
  --allow_aggression 1 --agent_size_mode "random" \
  --train_sensor_dropout_p 0.25

python train_fish.py --experiment_name 2M --num_env_steps 2000000 --episode_length 400 \
  --render_episodes 5 --n_rollout_threads 12 --backwards --weight_decay 0.000001 \
  --pfeeder 1 --prandom 1 --urandom 4 \
  --allow_aggression 1 --agent_size_mode "random"

python train_fish.py --experiment_name 3M --num_env_steps 3000000 --episode_length 400 \
  --render_episodes 5 --n_rollout_threads 12 --backwards --weight_decay 0.000001 \
  --pfeeder 1 --prandom 1 --urandom 4 \
  --allow_aggression 1 --agent_size_mode "random"


# ============================================================
# HOMING
# ============================================================

python train_fish.py --experiment_name H5K --num_env_steps 5000 \
  --homing_mode --required_homing_steps 1  \
  --n_rollout_threads 12 --save_interval 100 --episode_length 400

python train_fish.py --experiment_name H1M --num_env_steps 1000000 \
  --episode_length 400 --max_episode_length 400 \
  --save_interval 100 \
  --p_init_closeby 0.0 \
  --render_episodes 10 --weight_decay 0.000001 --n_rollout_threads 12 \
  --homing_mode --required_homing_steps 1

python train_fish.py --experiment_name H500K_matchhanson --num_env_steps 500000 \
  --render_episodes 2 --episode_length 400 --weight_decay 0.000001 --n_rollout_threads 12 \
  --mormyromast_mode 0 --ampullary_mode 0 \
  --knollen_mode 1 --knollen_processing log \
  --homing_mode --required_homing_steps 10 --backwards

for K in 1 0; do
 for M in 1 0; do
   for A in 1 0; do
    EXPT_NAME="homing_K${K}M${M}A${A}"
    python train_fish.py --experiment_name $EXPT_NAME \
      --num_env_steps 500000 \
      --render_episodes 5 \
      --episode_length 400 \
      --n_rollout_threads 8 \
      --homing_mode \
      --required_homing_steps 5 \
      --knollen_mode $K \
      --mormyromast_mode $M \
      --ampullary_mode $A \
      --weight_decay 0.000001 > ${EXPT_NAME}.log 2>&1 &
    sleep 1
    done
  done
done


# ============================================================
# OOM / RESOURCE DIAGNOSTICS
# ============================================================

dmesg | grep -i 'killed process'
ulimit -a
# ulimit -m unlimited
# ulimit -v unlimited


# ============================================================
# VIDEO CONVERSION
# ============================================================

for G in $(ls ./*.gif); do
  ffmpeg -i $G ${G%.gif}.mp4
done


# ============================================================
# LATEX GENERATION
# ============================================================
RUN_DIR=results/ConsNoise20260608dynamicT5MFO0.1FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRU/ConsNoise20260608dynamicT5MFO0.1FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRUSeed1/20260608_201817/
bash scripts/copy_figs_to_tex.sh "$RUN_DIR"

# OR run remotely on the cluster
# The */20*/ glob picks up the timestamped subdir inside each seed folder in one shot — no array indexing needed.
GROUP_NAME=ConsNoise20260608dynamicT5MFO0.1FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRU
GROUP_BASE=~/cluster_lab/${USER:-$(whoami)}/marl_fish_storage/results/$GROUP_NAME
for run in "$GROUP_BASE"/*/20*/; do
    bash scripts/copy_figs_to_tex.sh "${run%/}"
done
