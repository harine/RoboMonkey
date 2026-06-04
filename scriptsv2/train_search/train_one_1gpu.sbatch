#!/bin/bash
#SBATCH --job-name=noise_one
#SBATCH --account=robotics
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=80G
#SBATCH --time=72:00:00
#SBATCH --output=data/train/noise_sweep/slurm-%j.out
#SBATCH --error=data/train/noise_sweep/slurm-%j.out
#
# One search-policy run on one a40. Select the run via --export:
#   tmrl/noise_conditioned:  --export=ALL,RUN_COND=1,LEVELS=
#   tunable_corruption@L:    --export=ALL,RUN_COND=0,LEVELS=0.4   (or 0.8)
# Resubmit with a longer --time if 7000 steps need >72h (each ~49h at ~25s/it).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
mkdir -p data/train/noise_sweep
: "${RUN_COND:?set RUN_COND}"; LEVELS="${LEVELS-}"
echo "[sbatch] host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-?} RUN_COND=$RUN_COND LEVELS='${LEVELS}'"
nvidia-smi -L || true
export CONDA_ENV="${CONDA_ENV:-monkey-verifier}"
export EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-+training.max_gradient_steps=7000 task.dataset.use_cache=true task.dataset.dataset_dir=/gscratch/robotics/harine/data/eggplant_in_basket_complete}"
TS="$(date +%Y_%m_%d-%H_%M_%S)"
if [[ "$RUN_COND" == "1" ]]; then tag="tmrl"; else tag="tcorrupt${LEVELS}"; fi
LOG="data/train/noise_sweep/${tag}-$TS-job$SLURM_JOB_ID.log"
DEVICE=cuda:0 RUN_COND="$RUN_COND" LEVELS="$LEVELS" bash scriptsv2/train_search/train_noise_sweep.sh > "$LOG" 2>&1
echo "[sbatch] done; log=$LOG"
