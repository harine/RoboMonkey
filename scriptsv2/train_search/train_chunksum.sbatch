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
# Write checkpoints to gscratch by default (home has a ~10GB quota that
# corrupted/killed saves -- e.g. noise0.8 died with "Disk quota exceeded").
# Absolute hydra.run.dir works regardless of the train cd into diffusion_policy/.
OUT_ROOT="${OUT_ROOT:-/gscratch/robotics/harine/RoboMonkey_data/train/chunksum}"
OUTDIR="${OUT_ROOT}/${POLICY}"

export CONDA_ENV="${CONDA_ENV:-monkey-verifier}"
MAX_GRADIENT_STEPS="${MAX_GRADIENT_STEPS:-7000}"
# Search breadth (candidates the policy conditions on). 16 = MAZE-consistent
# (all maze_search* configs + the eggplant yaml use max_actions=16). This gives
# cond_pos_emb = n_obs_steps(2) + (max_actions-1) = 17. NOTE: the old chunk-sum
# checkpoints were trained at 8 (cond_pos_emb=9, a memory-reduction override) and
# are NOT loadable at 16 -> training at 16 means retraining from scratch.
MAX_ACTIONS="${MAX_ACTIONS:-16}"

# In-process verifier speedups (same set the eval sweep uses). The image-feature
# / processed-image caches make the chunk-sum scoring's repeated single-frame
# scorings reuse one vision-tower encode per frame across the 8 executed steps x
# N candidates. KV_PREFIX is now ON: each frame's image+instruction prefix is
# encoded once and its KV reused across the search rounds, so only the short
# action suffix is forwarded per score. Measured 3.65x on the chunk-sum verifier
# loop (scriptsv2/bon/bench_chunksum_kv.py: 109 -> 30 ms/score). NOT bit-exact vs
# the full forward (~0.1 bf16 noise that compounds over 32 layers) but the action
# the search SELECTS is unchanged (100% top-1 agreement; see validate_kvprefix.py
# + README_L2S.md). KV_FWD_CHUNK bounds the prefill/suffix sub-batch memory.
export ROBOMONKEY_KV_PREFIX="${ROBOMONKEY_KV_PREFIX:-1}"
export ROBOMONKEY_KV_CHUNK="${ROBOMONKEY_KV_CHUNK:-16}"
export ROBOMONKEY_KV_FWD_CHUNK="${ROBOMONKEY_KV_FWD_CHUNK:-16}"
export ROBOMONKEY_IMAGE_FEAT_CACHE_SIZE="${ROBOMONKEY_IMAGE_FEAT_CACHE_SIZE:-512}"
export ROBOMONKEY_PROC_IMAGE_CACHE_SIZE="${ROBOMONKEY_PROC_IMAGE_CACHE_SIZE:-512}"
# Batch the chunk-sum scoring: each search get_value call scores the 8 executed
# steps as ONE batched verifier forward (default chunk size 4 -> 2 forwards).
# =8 collapses it to a single forward per candidate (~2x on the scoring path).
export ROBOMONKEY_PAIRED_CHUNK_SIZE="${ROBOMONKEY_PAIRED_CHUNK_SIZE:-8}"

# Chunk-sum scoring baked into the policy's verifier config (-> saved in the
# checkpoint, so inference uses the SAME scoring). score_executed_chunk=true lets
# the search policy derive the scored window = the EXECUTED action steps
# [n_obs_steps-1 .. +n_action_steps], so it tracks n_action_steps automatically
# (instead of a hardcoded index list). score_agg=sum.
export EXTRA_OVERRIDES="+training.max_gradient_steps=${MAX_GRADIENT_STEPS} \
task.dataset.use_cache=true \
task.dataset.dataset_dir=/gscratch/robotics/harine/data/eggplant_in_basket_complete \
+policy.verifier.score_executed_chunk=true \
+policy.verifier.score_agg=sum \
policy.max_actions=${MAX_ACTIONS} \
hydra.run.dir=${OUTDIR}"

DEVICE=cuda:0 LEVELS="$LEVELS" RUN_COND="$RUN_COND" \
    bash scriptsv2/train_search/train_noise_sweep.sh

echo "[sbatch] chunk-sum training ($POLICY) done -> diffusion_policy/${OUTDIR}/checkpoints/latest.ckpt"
