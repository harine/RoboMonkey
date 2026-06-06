#!/bin/bash
#SBATCH --job-name=tr_chunksum
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=data/train/chunksum/slurm-%j.out
#SBATCH --error=data/train/chunksum/slurm-%j.out
#
# Train a search policy with CHUNK-SUM verifier scoring: the in-process
# RoboMonkey verifier scores the 8 EXECUTED actions (horizon steps [1..8]) vs
# the current frame and SUMS them into one chunk value, in TRAINING (here) and
# inference (the override is baked into the checkpoint config). Replaces the
# legacy single-step (index -1, non-executed) scoring.
#
# POLICY env selects the run: noise0.0 | noise0.4 | noise0.8 | tmrl
# Account / partition / gres come from the submit command (--account -p --gres).
#
# Submit (example):
#   POLICY=noise0.0 sbatch --account=socialrl --partition=gpu-l40 \
#       --gres=gpu:l40:1 scriptsv2/train_search/train_chunksum.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
mkdir -p data/train/chunksum

POLICY="${POLICY:?set POLICY=noise0.0|noise0.4|noise0.8|tmrl}"
echo "[sbatch] host=$(hostname)  cuda=${CUDA_VISIBLE_DEVICES:-?}  POLICY=$POLICY"
nvidia-smi -L || true

case "$POLICY" in
    noise0.0) LEVELS=0.0; RUN_COND=0;;
    noise0.4) LEVELS=0.4; RUN_COND=0;;
    noise0.8) LEVELS=0.8; RUN_COND=0;;
    tmrl)     LEVELS="";  RUN_COND=1;;
    *) echo "ERROR: bad POLICY '$POLICY'"; exit 1;;
esac
OUTDIR="data/train/chunksum/${POLICY}"   # relative to diffusion_policy/ (train cd's there)

export CONDA_ENV="${CONDA_ENV:-monkey-verifier}"
MAX_GRADIENT_STEPS="${MAX_GRADIENT_STEPS:-7000}"

# In-process verifier speedups (same set the eval sweep uses). The image-feature
# / processed-image caches make the chunk-sum scoring's repeated single-frame
# scorings reuse one vision-tower encode per frame across the 8 executed steps x
# N candidates. KV_PREFIX stays OFF (known _slice_kv crash); flip to 1 only once
# that verifier bug is fixed (would collapse the per-candidate LLM forward too).
export ROBOMONKEY_KV_PREFIX="${ROBOMONKEY_KV_PREFIX:-0}"
export ROBOMONKEY_KV_CHUNK="${ROBOMONKEY_KV_CHUNK:-16}"
export ROBOMONKEY_IMAGE_FEAT_CACHE_SIZE="${ROBOMONKEY_IMAGE_FEAT_CACHE_SIZE:-512}"
export ROBOMONKEY_PROC_IMAGE_CACHE_SIZE="${ROBOMONKEY_PROC_IMAGE_CACHE_SIZE:-512}"

# Chunk-sum scoring baked into the policy's verifier config (-> saved in the
# checkpoint, so inference uses the SAME scoring). score_indices = the executed
# window [1..n_action_steps]; score_agg=sum.
export EXTRA_OVERRIDES="+training.max_gradient_steps=${MAX_GRADIENT_STEPS} \
task.dataset.use_cache=true \
task.dataset.dataset_dir=/gscratch/robotics/harine/data/eggplant_in_basket_complete \
+policy.verifier.score_indices=[1,2,3,4,5,6,7,8] \
+policy.verifier.score_agg=sum \
hydra.run.dir=${OUTDIR}"

DEVICE=cuda:0 LEVELS="$LEVELS" RUN_COND="$RUN_COND" \
    bash scriptsv2/train_search/train_noise_sweep.sh

echo "[sbatch] chunk-sum training ($POLICY) done -> diffusion_policy/${OUTDIR}/checkpoints/latest.ckpt"
