#!/bin/bash
#SBATCH --job-name=eggplant_bc
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=data/train/bc/slurm-%j.out
#SBATCH --error=data/train/bc/slurm-%j.out
#
# Train a BC diffusion UNet on the eggplant task.
# Architecture matches the search policies (same obs space, DDIM 8 steps,
# n_obs_steps=2, horizon=16) so the resulting checkpoint can be evaluated
# with eval_search_bc.sh using inference-time N-sample + verifier selection.
# No verifier is used at training time.
#
# Submit:
#   cd /mmfs1/home/harine/RoboMonkey
#   sbatch scriptsv2/train_search/train_bc.sbatch
#
# Override env:
#   MAX_GRADIENT_STEPS (default: 7000)
#   BATCH_SIZE         (default: config default 256)
#   CONDA_ENV          (default: monkey-verifier)

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
mkdir -p data/train/bc

echo "[sbatch] host=$(hostname)  cuda_visible=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi -L || true

CONDA_ENV="${CONDA_ENV:-monkey-verifier}"
# This workspace stops on num_epochs, not a gradient-step cap (max_gradient_steps
# is unused). num_epochs lives in the config (260 ~= 20k steps). Override here
# only if you want a different length.
NUM_EPOCHS="${NUM_EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
DP_ROOT="${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}/diffusion_policy"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONPATH="${DP_ROOT}:${PYTHONPATH:-}"
cd "$DP_ROOT"

OVERRIDES=(
    "training.device=cuda:0"
    "task.dataset.use_cache=true"
    "task.dataset.dataset_dir=/gscratch/robotics/harine/data/eggplant_in_basket_complete"
)
if [[ -n "${NUM_EPOCHS:-}" ]]; then
    OVERRIDES+=("training.num_epochs=${NUM_EPOCHS}")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
    OVERRIDES+=(
        "dataloader.batch_size=${BATCH_SIZE}"
        "val_dataloader.batch_size=${BATCH_SIZE}"
    )
fi

echo "[sbatch] overrides: ${OVERRIDES[*]}"
python diffusion_policy/workspace/train_diffusion_unet_image_workspace.py \
    --config-name=robomonkey_eggplant_bc \
    "${OVERRIDES[@]}"

echo "[sbatch] training done."
