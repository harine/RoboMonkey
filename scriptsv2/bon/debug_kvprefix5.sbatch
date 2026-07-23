#!/bin/bash
#SBATCH --job-name=kvdbg5
#SBATCH --account=robotics
#SBATCH --partition=ckpt
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mmfs1/home/harine/RoboMonkey/scriptsv2/bon/kvdbg5-%j.out
#SBATCH --error=/mmfs1/home/harine/RoboMonkey/scriptsv2/bon/kvdbg5-%j.out
set -euo pipefail
cd /mmfs1/home/harine/RoboMonkey
echo "[dbg] host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi -L || true
source ~/miniconda3/etc/profile.d/conda.sh
conda activate monkey-verifier
export MODEL_DIR=/mmfs1/home/harine/RoboMonkey/monkey-verifier/model_dir
find monkey-verifier -name '*.pyc' -delete 2>/dev/null || true
python scriptsv2/bon/debug_kvprefix5.py
echo "[dbg] exit=$?"
