#!/bin/bash
#SBATCH --job-name=kvsmoke
#SBATCH --account=robotics
#SBATCH --partition=ckpt
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --time=00:40:00
#SBATCH --output=/mmfs1/home/harine/RoboMonkey/data/train/kvsmoke/slurm-%j.out
#SBATCH --error=/mmfs1/home/harine/RoboMonkey/data/train/kvsmoke/slurm-%j.out
#
# Smoke test the chunk-sum search-policy training step WITH the per-image
# prefix-KV verifier enabled. Runs a handful of gradient steps to a throwaway
# dir so it can't clobber a real run. Confirms: KV prefix engages, no crash/OOM,
# and the per-step time drops (verifier loop ~25s -> a few s).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
mkdir -p data/train/kvsmoke
echo "[smoke] host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi -L || true
find monkey-verifier -name '*.pyc' -delete 2>/dev/null || true

export CONDA_ENV="${CONDA_ENV:-monkey-verifier}"
# THE knob under test: per-image prefix-KV reuse on.
export ROBOMONKEY_KV_PREFIX=1
export ROBOMONKEY_KV_CHUNK="${ROBOMONKEY_KV_CHUNK:-16}"
export ROBOMONKEY_KV_FWD_CHUNK="${ROBOMONKEY_KV_FWD_CHUNK:-16}"
export ROBOMONKEY_IMAGE_FEAT_CACHE_SIZE="${ROBOMONKEY_IMAGE_FEAT_CACHE_SIZE:-512}"
export ROBOMONKEY_PROC_IMAGE_CACHE_SIZE="${ROBOMONKEY_PROC_IMAGE_CACHE_SIZE:-512}"
export ROBOMONKEY_PAIRED_CHUNK_SIZE="${ROBOMONKEY_PAIRED_CHUNK_SIZE:-8}"

STEPS="${STEPS:-40}"
export EXTRA_OVERRIDES="+training.max_gradient_steps=${STEPS} \
task.dataset.use_cache=true \
task.dataset.dataset_dir=/gscratch/robotics/harine/data/eggplant_in_basket_complete \
+policy.verifier.score_executed_chunk=true \
+policy.verifier.score_agg=sum \
hydra.run.dir=data/train/kvsmoke/run"

DEVICE=cuda:0 LEVELS=0.0 RUN_COND=0 \
    bash scriptsv2/train_search/train_noise_sweep.sh
echo "[smoke] done."
