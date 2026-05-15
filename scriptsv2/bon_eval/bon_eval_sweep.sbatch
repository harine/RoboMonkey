#!/bin/bash
#SBATCH --job-name=bon_sweep
#SBATCH --account=robotics
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=13
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=data/eval/bon/_summaries/slurm-%j.out
#SBATCH --error=data/eval/bon/_summaries/slurm-%j.out
#
# Submit:
#   cd /mmfs1/home/harine/RoboMonkey
#   sbatch scriptsv2/bon_eval/bon_eval_sweep.sbatch
#
# Reserves 2 a40 GPUs + 13 CPUs + 160G mem (matches the robotics QOS cap for
# 2-GPU jobs: cpu=13, mem=175G). Runs the full
# (start_seed x k) grid in parallel with one worker per GPU. The in-process
# improved verifier shares each GPU with the diffusion policy.
#
# Default sweep: k in {1, 2, 4, 8, 16, 32, 64} (2^0..2^6) across 3 seeds.
# Override env: KS, START_SEEDS, NUM_EPISODES, NUM_WORKERS, FORCE.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p data/eval/bon/_summaries

echo "[sbatch] host=$(hostname)  cuda_visible=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L || true

# In-process verifier: loaded inside the eval Python process on the same GPU
# as the diffusion policy. No HTTP server is started. Override REWARD_SERVER_PORT
# to a positive value to fall back to the HTTP infer_server on that port.
export REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-0}"
# Point at the LLaVA checkpoint dir; falls back to monkey-verifier/model_dir.
export MODEL_DIR="${MODEL_DIR:-$SLURM_SUBMIT_DIR/monkey-verifier/model_dir}"
# Run eval inside the monkey-verifier env (now augmented with SimplerEnv +
# diffusion_policy deps by scripts/install_verifier_eval_deps.sh) so the
# in-process verifier can import llava/transformers from the same Python.
export CONDA_ENV="${CONDA_ENV:-monkey-verifier}"

NUM_EPISODES="${NUM_EPISODES:-100}" \
    KS="${KS:-1 2 4 8 16 32 64}" \
    START_SEEDS="${START_SEEDS:-17 38 99}" \
    NUM_WORKERS="${NUM_WORKERS:-2}" \
    bash scriptsv2/bon_eval/bon_eval_sweep.sh
