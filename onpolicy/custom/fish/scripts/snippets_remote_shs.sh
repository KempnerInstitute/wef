# KICK OFF ON  LAMBDA -- BUT RUN ON CLUSTER

# Not for batch execution; copy-paste to run...
echo "Not for batch execution; copy-paste to run..."
exit 0


# ============================================================
# SSH / TUNNEL
# ============================================================

# Establish ControlMaster tunnel + sshfs mounts (interactive, needs password + 2FA)
bash scripts/fasrc_connect_shs.sh
# or from Claude Code prompt:
# ! bash scripts/fasrc_connect_shs.sh

# Mounted locally at:
#   ~/cluster_lab/  → /n/holylfs06/LABS/krajan_lab/Lab/
#   ~/cluster_home/ → ~/  (cluster home)

DIR_MOUNT=~/cluster_lab/${USER:-$(whoami)}/marl_fish_storage/results/
cd $DIR_MOUNT


# Install Python prereqs on cluster (only needed once or after requirements change)
bash scripts/fasrc_setup_env_shs.sh

# ============================================================
# JOB SUBMISSION (SLURM)
# ============================================================

# Submit foraging jobs
bash scripts/fasrc_submit_shs.sh                   # default seeds 1 2
SEEDS="3 4 5" bash scripts/fasrc_submit_shs.sh     # custom seeds
HOMING=1 bash scripts/fasrc_submit_shs.sh          # homing variant

# Each seed: 1 sbatch job (24h, 1 GPU, 256 GB RAM, 24 CPUs)
# Script: scripts/lambda_minimal_shs.sh → train + eval + pipeline




# ============================================================
# MONITORING
# ============================================================

ssh fasrc "squeue -u ${USER:-$(whoami)}"
ssh fasrc "squeue -u ${USER:-$(whoami)} --format='%.18i %.9P %.30j %.8T %.10M %.6D %R'"
ssh fasrc "scancel <job_id>"
ssh fasrc "scontrol show job <job_id>"

# Tail latest output log
ssh fasrc "cat ~/mfrefactor/onpolicy/custom/fish/logs/minimal_*/minimal_*.out" | tail -50

# Interactive GPU session
salloc --partition=kempner --account=kempner_krajan_lab \
       --time=2:00:00 --mem=28G --gres=gpu:1 --cpus-per-task=12 \
       srun --pty bash -i

salloc -p kempner_interactive --account=kempner_krajan_lab \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G --time=00-2:30:00 \
  srun --pty bash -i

# On cluster: load modules + env
module load cuda/11.8.0-fasrc01 Mambaforge/23.3.1-fasrc01
source ~/.bashrc
mamba activate mfrefactor


# ============================================================
# RSYNC — pull results to local
# ============================================================

# Pull a specific run (excludes raw pkl bundles — keeps derived + analyses)
DIR_CLUSTER=/n/holylfs06/LABS/krajan_lab/Lab/${USER:-$(whoami)}/marl_fish_storage/results/<EXP>/<TIMESTAMP>/
DIR_LOCAL=/srv/marl/${USER:-$(whoami)}/marl_fish/

rsync -avz --progress \
  --exclude='*raw*.pkl' \
  ${USER:-$(whoami)}@login.rc.fas.harvard.edu:${DIR_CLUSTER} ${DIR_LOCAL}/

# Pull flattened pkl only (skip raw pkl bundles and analysis outputs)
rsync -avz --progress \
  --include='*/' \
  --include='*agg_flat.pkl' \
  --include='derived/' \
  --include='*.pkl' \
  --exclude='*raw*.pkl' \
  --exclude='*.mp4' \
  --exclude='*.npy' \
  ${USER:-$(whoami)}@login.rc.fas.harvard.edu:${DIR_CLUSTER} ${DIR_LOCAL}/

# Pull only analysis PDFs (no pkl, no video)
rsync -avz --progress \
  --include='*/' \
  --include='*.pdf' \
  --exclude='*' \
  ${USER:-$(whoami)}@login.rc.fas.harvard.edu:${DIR_CLUSTER} ${DIR_LOCAL}/

# Browse results mounted locally (no rsync needed if tunnel is live)
ls ~/cluster_lab/${USER:-$(whoami)}/marl_fish_storage/results/


# ============================================================
# One offs
# ============================================================

RUN_DIR=./results/ConsNoise20260610dynamicT5MFO0.1FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRUBEorig/ConsNoise20260610dynamicT5MFO0.1FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRUBEorigSeed1/20260610_020447/
SPEC_DIR=$RUN_DIR/evals/m1a1k1_patchy_square/
python analysis_behavior.py --spec_dir $SPEC_DIR --n_samples 1


SPEC_DIR=results/ConsNoise20260610dynamicT5MFO0.1FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRUBEorig/ConsNoise20260610dynamicT5MFO0.1FX1.0Order1LinearX2.0AngularX4.0Gamma0.995DCL100TD0.0PR5UR1NP0A1K1M1GRUBEorigSeed1/20260610_020447/evals/2fish_m1a1k1_uniform_wide
python analysis_eod.py --spec_dir $SPEC_DIR
