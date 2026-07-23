#!/bin/bash
#SBATCH --job-name=srchdqc
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/mmfs1/home/harine/RoboMonkey/data/eval/search_dqc/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# Evaluate the CONDITIONAL DIFFUSION search policy (SearchPolicyRoboMonkeyDiffusion)
# with the DQC Q-critic verifier: the autoregressive best-of-N search scores each
# candidate by the DISCOUNTED CHUNK SUM of DQC Q-values over the executed chunk
# (sum_t gamma^t Q(frame, a_t), gamma=0.98) and argmax-selects (MODE=argmax).
# eval_search.py swaps the policy's baked LLaVA verifier for DQCVerifier when
# VERIFIER_TYPE=dqc. Runs in simpler_env (JAX[DQC] + torch[policy/SAPIEN]).
#   sbatch --export=ALL,CHECKPOINT=<search ckpt>,N_SAMPLES=16,NUM_EPISODES=50 eval_search_dqc.sbatch
# -----------------------------------------------------------------------------
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
: "${CHECKPOINT:?set CHECKPOINT (diffusion search policy)}"
DQC_CKPT_DIR="${DQC_CKPT_DIR:-/gscratch/robotics/harine/dqc_checkpoint/sd000s_137913.0.20260605_163017}"
DQC_TEXT_EMB="${DQC_TEXT_EMB:-$SLURM_SUBMIT_DIR/scriptsv2/q_fn_eval/task_emb.npy}"
NUM_EPISODES="${NUM_EPISODES:-50}"
N_SAMPLES="${N_SAMPLES:-16}"
OUT_DIR="${OUT_DIR:-data/eval/search_dqc/n${N_SAMPLES}_disc098}"
mkdir -p "$(dirname "$OUT_DIR")" data/eval/search_dqc

# Put the repo's diffusion_policy first on PYTHONPATH — in simpler_env the bare
# `diffusion_policy` is an empty namespace package, so the custom search-policy
# module (SearchPolicyRoboMonkeyDiffusion) is only importable from the repo copy.
export DIFFUSION_POLICY_ROOT="${DIFFUSION_POLICY_ROOT:-$SLURM_SUBMIT_DIR/diffusion_policy}"
# JAX (DQC) must not preallocate the GPU — torch (the diffusion policy) shares it.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}"

echo "[search_dqc] policy = $CHECKPOINT"
echo "[search_dqc] dqc    = $DQC_CKPT_DIR  (discounted_sum, gamma=${DQC_DISCOUNT:-0.98})"
echo "[search_dqc] N=$N_SAMPLES  nas=8  episodes=$NUM_EPISODES  mode=argmax  -> $OUT_DIR"

CONDA_ENV="${CONDA_ENV:-simpler_env}" DEVICE=cuda:0 \
  VERIFIER_TYPE=dqc \
  DQC_CKPT_DIR="$DQC_CKPT_DIR" DQC_TEXT_EMB="$DQC_TEXT_EMB" DQC_DISCOUNT="${DQC_DISCOUNT:-0.98}" \
  VERIFIER_INSTRUCTION="${VERIFIER_INSTRUCTION:-put the eggplant in the basket}" \
  VERIFIER_IMAGE_KEY="${VERIFIER_IMAGE_KEY:-agentview_image}" \
  MODE=argmax SCORE_WINDOW=executed \
  N_SAMPLES="$N_SAMPLES" N_ACTION_STEPS=8 \
  START_SEED="${START_SEED:-1000}" MAX_STEPS="${MAX_STEPS:-120}" USE_EMA=1 \
  TASK="${TASK:-widowx_put_eggplant_in_basket}" SAVE_VIDEOS=0 \
  bash scriptsv2/eval_search/eval_search.sh "$CHECKPOINT" "$NUM_EPISODES" "$OUT_DIR"

echo "[search_dqc] done -> $OUT_DIR/eval_log.json"
