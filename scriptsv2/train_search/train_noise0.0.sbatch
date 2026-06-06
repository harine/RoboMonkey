#!/bin/bash
#SBATCH --job-name=search_noise0.0
#SBATCH --account=robotics
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=data/train/noise_sweep/slurm-%j.out
#SBATCH --error=data/train/noise_sweep/slurm-%j.out
#
# Train the conditional-diffusion Learning-to-Search policy at ZERO observation
# noise (obs_noise_level=0.0) -- the clean baseline that matches the existing
# noise0.4 / noise0.8 / tmrl checkpoints (same config, same in-process verifier,
# same 7000-step cap). Single a40 GPU.
#
# Submit:
#   cd /mmfs1/home/harine/RoboMonkey
#   sbatch scriptsv2/train_search/train_noise0.0.sbatch
#
# Override env: MAX_GRADIENT_STEPS (default 7000), CONDA_ENV (monkey-verifier).
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
mkdir -p data/train/noise_sweep

echo "[sbatch] host=$(hostname)  cuda_visible=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi -L || true

export CONDA_ENV="${CONDA_ENV:-monkey-verifier}"
MAX_GRADIENT_STEPS="${MAX_GRADIENT_STEPS:-7000}"

# task.dataset.dataset_dir: the task config hardcodes a /home path that does not
# exist on compute nodes; point at the curated complete-shard dir (images +
# reward labels). success_only=true is already set in the task config.
export EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-+training.max_gradient_steps=${MAX_GRADIENT_STEPS} task.dataset.use_cache=true task.dataset.dataset_dir=/gscratch/robotics/harine/data/eggplant_in_basket_complete}"

# Single unconditioned tunable-corruption run at obs_noise_level=0.0 (the
# noise-conditioned tmrl model is skipped via RUN_COND=0).
DEVICE=cuda:0 RUN_COND=0 LEVELS=0.0 \
    bash scriptsv2/train_search/train_noise_sweep.sh

echo "[sbatch] training (noise0.0) done."
