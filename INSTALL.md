# Installation

These instructions assume a Linux machine with a CUDA 12.x-capable GPU. The setup
below was validated with Python 3.10.20, PyTorch 2.5.1+cu121, and CUDA 12.4.

## Prerequisites

- `mamba` or `conda`
- CUDA 12.x drivers available on the machine
- `ffmpeg` for video rendering

On the Harvard FASRC SLURM cluster, load the modules first:

```bash
module load cuda/11.8.0-fasrc01
module load Mambaforge/23.3.1-fasrc01
```

## Create the environment

From the repository root:

```bash
mamba create -n mfrefactor python=3.10 -c conda-forge
mamba activate mfrefactor
mamba install -c conda-forge ffmpeg
```

## Install Python dependencies

Install PyTorch first (requires the PyTorch index for CUDA wheels), then the rest
of the requirements, then install the local `onpolicy` package in editable mode:

```bash
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

## libcudnn path (required for GPU ops)

The environment needs `libcudnn.so.9` on `LD_LIBRARY_PATH`. The conda activation
hook sets this automatically if you activate via `mamba activate mfrefactor`. For
non-interactive shells (SLURM jobs, `mamba run`), set it manually:

```bash
export LD_LIBRARY_PATH=/home/$USER/miniforge3/envs/mfrefactor/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```

Adjust the prefix to match your miniforge installation path.

## Smoke tests

Verify the package imports:

```bash
python -c "import onpolicy; print(onpolicy.__version__)"
```

Run a short training job (**from `onpolicy/custom/fish/`**):

```bash
cd onpolicy/custom/fish
python train_fish.py \
  --experiment_name install_smoke_test \
  --num_env_steps 1000 \
  --episode_length 20 \
  --max_episode_length 20 \
  --n_rollout_threads 1 \
  --render_episodes 0
```

Verify the env and arena independently:

```bash
python arena.py      # saves arena_*.png visualisations
python MAEFish.py    # runs 40 steps, saves an MP4
```

Outputs are written under `onpolicy/custom/fish/results/`.

## Notes

- All scripts must be run from `onpolicy/custom/fish/` — imports are local/relative.
- `environment.yaml` is a legacy environment snapshot from the original MAPPO codebase.
  Use the commands above instead.
- `ffmpeg` is only required for rendering videos, but installing it up front avoids
  runtime failures in pipeline and animation scripts.
- On FASRC, `mamba activate` fails in non-interactive shells. Use:
  ```bash
  /home/$USER/miniforge3/bin/mamba run --name mfrefactor bash scripts/run_full.sh
  ```

## Fonts and external data (not included in this release)

Two assets used by some figure/analysis scripts are intentionally **not** distributed
with this repository:

- **Arial font.** The figure style uses Arial, which is a proprietary Monotype font
  and cannot be redistributed here. Plotting code will fall back to Matplotlib's
  default font if Arial is unavailable. To reproduce the paper's typography, install
  Arial from a licensed source onto your system.

- **Real weakly-electric-fish recordings.** The `onpolicy/custom/fish/real_fish_data/`
  directory contains inter-discharge-interval data derived from the supplementary material
  of:

  > Chrtkova, et al (2025). Unsupervised electric signal separation for linking behavior and
  > electrocommunication in *Gnathonemus petersii*. *Scientific Reports*.