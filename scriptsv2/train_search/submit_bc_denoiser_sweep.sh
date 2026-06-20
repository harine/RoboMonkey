#!/bin/bash
# Queue the 4 eggplant BC trainings (MLP / encoder-decoder denoiser x horizon
# 16/8) on weirdlab L40, 7k steps each. Each run has a fixed output dir and is
# resumable (re-run this script, or re-submit with a larger MAX_GRADIENT_STEPS,
# to continue from latest.ckpt).
#
#   cd /mmfs1/home/harine/RoboMonkey
#   scriptsv2/train_search/submit_bc_denoiser_sweep.sh
#
# Optional: MAX_GRADIENT_STEPS=12000 scriptsv2/train_search/submit_bc_denoiser_sweep.sh
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mmfs1/home/harine/RoboMonkey}"
cd "$REPO_ROOT"
SBATCH_FILE=scriptsv2/train_search/train_bc_denoiser.sbatch
# SLURM opens --output before the job body runs, so the log dir must pre-exist.
mkdir -p data/train/bc_denoiser/_logs

MAX_GRADIENT_STEPS="${MAX_GRADIENT_STEPS:-7000}"

# denoiser  horizon  job-name
RUNS=(
  "mlp          16  bc_mlp_h16"
  "mlp           8  bc_mlp_h8"
  "transformer  16  bc_xenc_h16"
  "transformer   8  bc_xenc_h8"
)

for r in "${RUNS[@]}"; do
  read -r DENOISER HORIZON JOBNAME <<< "$r"
  echo "[submit] $JOBNAME : denoiser=$DENOISER horizon=$HORIZON steps=$MAX_GRADIENT_STEPS"
  sbatch \
    --job-name="$JOBNAME" \
    --export=ALL,DENOISER="$DENOISER",HORIZON="$HORIZON",MAX_GRADIENT_STEPS="$MAX_GRADIENT_STEPS" \
    "$SBATCH_FILE"
done

echo "[submit] queued ${#RUNS[@]} jobs. Track with: squeue -u $USER"
