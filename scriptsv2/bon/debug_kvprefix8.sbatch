#!/bin/bash
#SBATCH --job-name=kvdbg8
#SBATCH --account=robotics
#SBATCH --partition=ckpt
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/mmfs1/home/harine/RoboMonkey/scriptsv2/bon/kvdbg8-%j.out
#SBATCH --error=/mmfs1/home/harine/RoboMonkey/scriptsv2/bon/kvdbg8-%j.out
set -euo pipefail
cd /mmfs1/home/harine/RoboMonkey
echo "[dbg8] host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi -L || true
source ~/miniconda3/etc/profile.d/conda.sh
conda activate monkey-verifier
export MODEL_DIR=/mmfs1/home/harine/RoboMonkey/monkey-verifier/model_dir
find monkey-verifier -name '*.pyc' -delete 2>/dev/null || true

echo "===== RUN A: default (no CUBLAS_WORKSPACE_CONFIG) ====="
unset CUBLAS_WORKSPACE_CONFIG || true
python scriptsv2/bon/debug_kvprefix8.py

echo "===== RUN B: CUBLAS_WORKSPACE_CONFIG=:4096:8 ====="
CUBLAS_WORKSPACE_CONFIG=:4096:8 python scriptsv2/bon/debug_kvprefix8.py
echo "[dbg8] exit=$?"
