#!/bin/bash
#SBATCH --job-name=search_nsweep
#SBATCH --account=socialrl
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:8
#SBATCH --cpus-per-task=40
#SBATCH --mem=640G
#SBATCH --time=72:00:00
#SBATCH --output=data/eval/search_noise_sweep/_summaries/slurm-%j.out
#SBATCH --error=data/eval/search_noise_sweep/_summaries/slurm-%j.out
#
# Softmax N-sample sweep over the three noise-comparison search policies
# (noise0.4, noise0.8, tmrl) x N_SAMPLES in {1,2,4,...,1024}, MODE=softmax.
# Diffusion candidate sampling is seeded per-replan (deterministic candidates);
# softmax selection stays stochastic. tmrl additionally sweeps its noise-level
# input tcont over TCONT_LIST (default 0.4 0.8 0.0); the tunable_corruption
# policies ignore tcont.
#
# Submit:
#   cd /mmfs1/home/harine/RoboMonkey
#   sbatch scriptsv2/eval_search/eval_search_noise_sweep.sbatch
#
# Reserves 8 l40s GPUs (5 cpu / 80G per GPU). One worker per GPU pulls cells
# from the shared queue. The in-process LLaVA-7b verifier shares each GPU with
# the search policy.
#
# Override env (via --export=ALL,VAR=...): NUM_EPISODES, N_LIST, POLICIES,
# START_SEED, SOFTMAX_TEMP, NUM_WORKERS, FORCE.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
mkdir -p data/eval/search_noise_sweep/_summaries

echo "[sbatch] host=$(hostname)  cuda_visible=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi -L || true

# In-process verifier: resolves MODEL_DIR to monkey-verifier/model_dir and
# imports verifier_client from monkey-verifier/src. Set both explicitly.
export MODEL_DIR="${MODEL_DIR:-$SLURM_SUBMIT_DIR/monkey-verifier/model_dir}"
export MONKEY_VERIFIER_SRC="${MONKEY_VERIFIER_SRC:-$SLURM_SUBMIT_DIR/monkey-verifier/src}"
# eval_search.sh's default diffusion_policy root is a nonexistent /home path on
# klone; pin the real one so child imports resolve.
export DIFFUSION_POLICY_ROOT="${DIFFUSION_POLICY_ROOT:-$SLURM_SUBMIT_DIR/diffusion_policy}"
export CONDA_ENV="${CONDA_ENV:-monkey-verifier}"

# In-process verifier speedup — same as training: image-feature / processed-
# image caches in the default score_paired path. KV_PREFIX (score_paired_kvprefix)
# is a separate scorer training never used and currently crashes in this env, so
# it stays OFF (set =1 only after that verifier bug is fixed).
export ROBOMONKEY_KV_PREFIX="${ROBOMONKEY_KV_PREFIX:-0}"
export ROBOMONKEY_KV_CHUNK="${ROBOMONKEY_KV_CHUNK:-16}"
export ROBOMONKEY_IMAGE_FEAT_CACHE_SIZE="${ROBOMONKEY_IMAGE_FEAT_CACHE_SIZE:-512}"
export ROBOMONKEY_PROC_IMAGE_CACHE_SIZE="${ROBOMONKEY_PROC_IMAGE_CACHE_SIZE:-512}"

NUM_EPISODES="${NUM_EPISODES:-50}" \
    N_LIST="${N_LIST:-1 2 4 8 16 32 64 128 256 512 1024}" \
    TCONT_LIST="${TCONT_LIST:-0.4 0.8 0.0}" \
    POLICIES="${POLICIES:-noise0.4 noise0.8 tmrl}" \
    START_SEED="${START_SEED:-1000}" \
    SOFTMAX_TEMP="${SOFTMAX_TEMP:-1.0}" \
    NUM_WORKERS="${NUM_WORKERS:-8}" \
    bash scriptsv2/eval_search/eval_search_noise_sweep.sh
