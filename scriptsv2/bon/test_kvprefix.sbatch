#!/bin/bash
#SBATCH --job-name=kvprefix_test
#SBATCH --account=robotics
#SBATCH --partition=ckpt
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/mmfs1/home/harine/RoboMonkey/scriptsv2/bon/kvprefix_test-%j.out
#SBATCH --error=/mmfs1/home/harine/RoboMonkey/scriptsv2/bon/kvprefix_test-%j.out
#
# Validate the per-image prefix-KV verifier path: exactness vs the full-forward
# baseline + speedup. Submit:
#   sbatch scriptsv2/bon/test_kvprefix.sbatch
set -euo pipefail
cd /mmfs1/home/harine/RoboMonkey
echo "[test] host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi -L || true
source ~/miniconda3/etc/profile.d/conda.sh
conda activate monkey-verifier
export MODEL_DIR=/mmfs1/home/harine/RoboMonkey/monkey-verifier/model_dir
# Drop stale bytecode so the edited flash-attention source is picked up.
find monkey-verifier/llava_setup/LLaVA/llava/model/language_model -name '*.pyc' -delete 2>/dev/null || true
KV_C="${KV_C:-16}" KV_ROUNDS="${KV_ROUNDS:-15}" \
  python scriptsv2/bon/test_verifier_kvprefix.py
echo "[test] exit=$?"
