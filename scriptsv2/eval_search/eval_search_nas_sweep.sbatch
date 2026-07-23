#!/bin/bash
#SBATCH --job-name=search_nas4
#SBATCH --account=robotics
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --output=data/eval/search_nas_sweep/_summaries/slurm-%j.out
#SBATCH --error=data/eval/search_nas_sweep/_summaries/slurm-%j.out
#
# Ablate the executed action horizon: eval a trained search policy at
# N_ACTION_STEPS executed steps/replan (the policy still predicts the full
# horizon=16). Mirrors eval_search_noise_sweep but writes to
# data/eval/search_nas_sweep/ so it never collides with the n_action_steps=8
# runs. Defaults to noise0.0 @ N_ACTION_STEPS=4.
#
# Submit:
#   cd /mmfs1/home/harine/RoboMonkey
#   sbatch scriptsv2/eval_search/eval_search_nas_sweep.sbatch
#   # or override, e.g.:
#   N_ACTION_STEPS=4 N_LIST="1 2 4 8 16 32" sbatch --export=ALL scriptsv2/eval_search/eval_search_nas_sweep.sbatch
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
mkdir -p data/eval/search_nas_sweep/_summaries

echo "[sbatch] host=$(hostname)  cuda_visible=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi -L || true

export MODEL_DIR="${MODEL_DIR:-$SLURM_SUBMIT_DIR/monkey-verifier/model_dir}"
export MONKEY_VERIFIER_SRC="${MONKEY_VERIFIER_SRC:-$SLURM_SUBMIT_DIR/monkey-verifier/src}"
export DIFFUSION_POLICY_ROOT="${DIFFUSION_POLICY_ROOT:-$SLURM_SUBMIT_DIR/diffusion_policy}"
export CONDA_ENV="${CONDA_ENV:-monkey-verifier}"

export ROBOMONKEY_KV_PREFIX="${ROBOMONKEY_KV_PREFIX:-0}"
export ROBOMONKEY_KV_CHUNK="${ROBOMONKEY_KV_CHUNK:-16}"
export ROBOMONKEY_IMAGE_FEAT_CACHE_SIZE="${ROBOMONKEY_IMAGE_FEAT_CACHE_SIZE:-512}"
export ROBOMONKEY_PROC_IMAGE_CACHE_SIZE="${ROBOMONKEY_PROC_IMAGE_CACHE_SIZE:-512}"

N_ACTION_STEPS="${N_ACTION_STEPS:-4}" \
    NUM_EPISODES="${NUM_EPISODES:-50}" \
    N_LIST="${N_LIST:-1 2 4 8 16 32}" \
    POLICIES="${POLICIES:-noise0.0}" \
    START_SEED="${START_SEED:-1000}" \
    SOFTMAX_TEMP="${SOFTMAX_TEMP:-1.0}" \
    NUM_WORKERS="${NUM_WORKERS:-2}" \
    bash scriptsv2/eval_search/eval_search_nas_sweep.sh
