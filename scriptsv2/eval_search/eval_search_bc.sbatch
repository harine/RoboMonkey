#!/bin/bash
#SBATCH --job-name=search_bc_sweep
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:a40:4
#SBATCH --cpus-per-task=20
#SBATCH --mem=320G
#SBATCH --time=48:00:00
#SBATCH --output=data/eval/search_bc_sweep/_summaries/slurm-%j.out
#SBATCH --error=data/eval/search_bc_sweep/_summaries/slurm-%j.out
#
# Softmax N-sample sweep for the BC diffusion UNet baseline.
# The BC checkpoint is wrapped at eval time with BCSearchWrapper + in-process
# LLaVA-7b verifier (no verifier used at training time).
#
# 8 cells (N in {1,2,4,8,16,32,64,128}), 4 GPUs → 2 cells per GPU.
# Each cell: 50 episodes, softmax selection.
#
# Submit (after BC training completes):
#   cd /mmfs1/home/harine/RoboMonkey
#   CHECKPOINT=<path/to/bc/latest.ckpt> \
#     sbatch scriptsv2/eval_search/eval_search_bc.sbatch
#
# Or update CHECKPOINT below and submit without env override.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
mkdir -p data/eval/search_bc_sweep/_summaries

echo "[sbatch] host=$(hostname)  cuda_visible=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi -L || true

export DIFFUSION_POLICY_ROOT="${DIFFUSION_POLICY_ROOT:-$SLURM_SUBMIT_DIR/diffusion_policy}"
export MONKEY_VERIFIER_SRC="${MONKEY_VERIFIER_SRC:-$SLURM_SUBMIT_DIR/monkey-verifier/src}"
export MODEL_DIR="${MODEL_DIR:-$SLURM_SUBMIT_DIR/monkey-verifier/model_dir}"
export CONDA_ENV="${CONDA_ENV:-monkey-verifier}"

# Same verifier speedup config as search sweep.
export ROBOMONKEY_KV_PREFIX="${ROBOMONKEY_KV_PREFIX:-0}"
export ROBOMONKEY_KV_CHUNK="${ROBOMONKEY_KV_CHUNK:-16}"
export ROBOMONKEY_IMAGE_FEAT_CACHE_SIZE="${ROBOMONKEY_IMAGE_FEAT_CACHE_SIZE:-512}"
export ROBOMONKEY_PROC_IMAGE_CACHE_SIZE="${ROBOMONKEY_PROC_IMAGE_CACHE_SIZE:-512}"

# CHECKPOINT must be provided. The sbatch --export=ALL,CHECKPOINT=... pattern
# is preferred; fall back to the most-recent bc run dir if not set.
if [[ -z "${CHECKPOINT:-}" ]]; then
    # Auto-locate latest bc checkpoint from training output.
    CHECKPOINT="$(ls -td "$SLURM_SUBMIT_DIR"/diffusion_policy/data/outputs/*/*/robomonkey_eggplant_bc_*/checkpoints/latest.ckpt 2>/dev/null | head -1 || true)"
    if [[ -z "$CHECKPOINT" ]]; then
        echo "ERROR: CHECKPOINT not set and no bc checkpoint auto-located." >&2
        echo "  Pass via: CHECKPOINT=<path> sbatch eval_search_bc.sbatch" >&2
        exit 1
    fi
    echo "[sbatch] auto-located checkpoint: $CHECKPOINT"
fi
export CHECKPOINT

NUM_EPISODES="${NUM_EPISODES:-50}" \
    N_LIST="${N_LIST:-1 2 4 8 16 32 64 128 256 512 1024}" \
    START_SEED="${START_SEED:-1000}" \
    SOFTMAX_TEMP="${SOFTMAX_TEMP:-1.0}" \
    NUM_WORKERS="${NUM_WORKERS:-4}" \
    bash scriptsv2/eval_search/eval_search_bc.sh
