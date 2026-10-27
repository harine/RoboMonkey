#!/usr/bin/env bash
# One BC UNet (H8, n_obs 1, predict 8 / execute 4, 50k steps: the
# eggplant_bc_unet_h8_50k_filtered recipe) per split manifest of a task.
#
#   bash scriptsv2/train_search/submit_bc_splits.sh eggplant          # dry run
#   SUBMIT=1 bash scriptsv2/train_search/submit_bc_splits.sh eggplant
#   RULES="random quadrant" SUBMIT=1 bash ... stack
#   SBATCH_ARGS="--account=robotics --partition=gpu-a40 --gres=gpu:a40:1" SUBMIT=1 bash ... eggplant
#
# Idempotent: a manifest whose job name is already queued/running is skipped.
set -euo pipefail
TASK="${1:?task: eggplant|stack}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SPLIT_DIR="$REPO/diffusion_policy/diffusion_policy/config/splits/simpler/$TASK"
SUBMIT="${SUBMIT:-0}"
RULES="${RULES:-}"
STEPS="${MAX_GRADIENT_STEPS:-50000}"
# extra sbatch options, e.g. SBATCH_ARGS="--account=robotics --partition=gpu-a40 --gres=gpu:a40:1"
# default: robotics A40 (alternatives: cse gpu-l40s/gpu-a100, ckpt scavenger with --requeue)
SBATCH_ARGS="${SBATCH_ARGS:---account=robotics --partition=gpu-a40 --gres=gpu:a40:1}"

[ -d "$SPLIT_DIR" ] || { echo "no manifests under $SPLIT_DIR (run scriptsv2/splits/make_simpler_splits.py)"; exit 1; }

queued="$(squeue -u "$USER" -h -o %j 2>/dev/null || true)"
for mf in "$SPLIT_DIR"/*.json; do
  name="$(basename "$mf" .json)"           # <rule>_train<N>
  rule="${name%%_train*}"
  if [ -n "$RULES" ] && ! grep -qw "$rule" <<<"$RULES"; then continue; fi
  DATASET_DIR="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["dataset_dir"])' "$mf")"
  run="bc_unet_h8_${STEPS}_${TASK}_${name}"
  if grep -qx "$run" <<<"$queued"; then echo "skip (queued): $run"; continue; fi
  echo "submit: $run  <- $mf"
  if [ "$SUBMIT" = "1" ]; then
    sbatch --job-name="$run" $SBATCH_ARGS \
      --export=ALL,DENOISER=unet,HORIZON=8,N_OBS_STEPS=1,MAX_GRADIENT_STEPS="$STEPS",LR_DECAY_STEPS=40000,RUN_NAME="$run",DATASET_DIR="$DATASET_DIR",SPLIT_FILE="$mf",WANDB_PROJECT="BC_${TASK}_splits",WANDB_NAME="$run" \
      "$REPO/scriptsv2/train_search/train_bc_denoiser.sbatch"
  fi
done
[ "$SUBMIT" = "1" ] || echo "(dry run; SUBMIT=1 to submit)"
