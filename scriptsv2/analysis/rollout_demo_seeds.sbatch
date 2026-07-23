#!/bin/bash
#SBATCH --job-name=demoroll
#SBATCH --cpus-per-task=6
#SBATCH --mem=56G
#SBATCH --time=6:00:00
#SBATCH --output=/mmfs1/home/harine/RoboMonkey/data/eval/train_test_analysis/_logs/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# Closed-loop rollout of the search policy from the DEMO seeds (train or test),
# dumping viz_q npz (frames + candidates + Q + selected_index + cam) for the
# expert-overlay renderer.
#
# The seed list is read from a FILE and exported as SEEDS inside this script, so the
# comma-separated value never passes through sbatch's comma-delimited --export.
#
# Required: CKPT, TASK, SEEDS_FILE, OUT_DIR.  Optional: N_SAMPLES, MAX_STEPS, NUM_VIDEOS.
#   sbatch --account=robotics --partition=ckpt-all --gres=gpu:a40:1 \
#     --export=ALL,CKPT=...,TASK=widowx_spoon_on_towel,SEEDS_FILE=...,OUT_DIR=... \
#     scriptsv2/analysis/rollout_demo_seeds.sbatch
# -----------------------------------------------------------------------------
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
REPO="$PWD"

: "${CKPT:?set CKPT}"; : "${TASK:?set TASK}"; : "${SEEDS_FILE:?set SEEDS_FILE}"; : "${OUT_DIR:?set OUT_DIR}"
NUM_VIDEOS="${NUM_VIDEOS:-10}"

# first NUM_VIDEOS seeds, comma-joined -- built HERE, never through --export
SEEDS="$(head -n "$NUM_VIDEOS" "$SEEDS_FILE" | paste -sd, -)"
export SEEDS
echo "[demoroll] task=$TASK  seeds=$SEEDS"
echo "[demoroll] out=$OUT_DIR"

export CONDA_ENV="${CONDA_ENV:-dqc_jax}"
export DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
# eval_search_dqc_chunk.sbatch expects CHECKPOINT (not CKPT)
export CHECKPOINT="$CKPT"

# viz path: non-tiled (frames are only dumped there), per-replan npz; the expert
# overlay is rendered afterwards by render_rollout_vs_expert.py.
MODE="${MODE:-argmax}" \
N_SAMPLES="${N_SAMPLES:-16}" \
NUM_EPISODES="$NUM_VIDEOS" \
MAX_STEPS="${MAX_STEPS:-120}" \
TILED_ENVS=1 \
VIZ_Q=1 VIZ_Q_EPISODES="$NUM_VIDEOS" \
RENDER_Q=0 SAVE_VIDEOS=0 \
TASK="$TASK" \
DQC_RESTORE_EPOCH=100000 \
OUT_DIR="$OUT_DIR" \
  bash scriptsv2/eval_search/eval_search_dqc_chunk.sbatch

echo "[demoroll] done -> $OUT_DIR/search_q"
