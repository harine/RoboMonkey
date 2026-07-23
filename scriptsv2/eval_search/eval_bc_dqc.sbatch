#!/bin/bash
#SBATCH --job-name=bc_dqc
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=data/eval/bc_dqc/slurm-%x-%j.out
#SBATCH --error=data/eval/bc_dqc/slurm-%x-%j.out
#
# Evaluate the 16-horizon / 8-executed-step BC policy with the DQC Q-function
# verifier doing best-of-N selection by DISCOUNTED chunk sum.
#
#   - horizon 16, n_action_steps=8 (execute 8 of the 16 predicted)
#   - N_SAMPLES candidate chunks per replan (default 16), argmax over DQC value
#   - DQC scores each of the 8 executed steps vs the current frame, aggregated
#     as sum_t 0.98^t * Q(frame, a_t)  (discounted chunk sum)
#   - BC_SAMPLER=ddpm so the 16 candidates are genuinely diverse (DDIM-8
#     collapses them, making best-of-N a no-op).
#   - runs in simpler_env (the only env with both JAX [DQC] and torch [policy]).
#
# Submit:
#   sbatch --export=ALL,CHECKPOINT=<bc latest.ckpt>,NUM_EPISODES=50 \
#       scriptsv2/eval_search/eval_bc_dqc.sbatch
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
mkdir -p data/eval/bc_dqc

# Default to the most-recent eggplant BC checkpoint (best.ckpt — matches the
# existing bc_h8 RoboMonkey-verifier eval for direct comparability).
if [[ -z "${CHECKPOINT:-}" ]]; then
    CHECKPOINT="$(ls -td diffusion_policy/data/outputs/*/*robomonkey_eggplant_bc_*/checkpoints/best.ckpt 2>/dev/null | head -1 || true)"
fi
: "${CHECKPOINT:?set CHECKPOINT (no bc checkpoint auto-located)}"

DQC_CKPT_DIR="${DQC_CKPT_DIR:-/gscratch/robotics/harine/dqc_checkpoint/sd000s_137913.0.20260605_163017}"
DQC_TEXT_EMB="${DQC_TEXT_EMB:-$SLURM_SUBMIT_DIR/scriptsv2/q_fn_eval/task_emb.npy}"
NUM_EPISODES="${NUM_EPISODES:-50}"
N_SAMPLES="${N_SAMPLES:-16}"
OUT_DIR="${OUT_DIR:-data/eval/bc_dqc/h16_n${N_SAMPLES}_disc098}"

echo "[bc_dqc] host=$(hostname)"
echo "[bc_dqc] bc ckpt   = $CHECKPOINT"
echo "[bc_dqc] dqc ckpt  = $DQC_CKPT_DIR"
echo "[bc_dqc] text emb  = $DQC_TEXT_EMB"
echo "[bc_dqc] N=$N_SAMPLES  nas=8  episodes=$NUM_EPISODES  agg=discounted_sum gamma=0.98"
echo "[bc_dqc] out        = $OUT_DIR"
nvidia-smi -L || true

export DIFFUSION_POLICY_ROOT="${DIFFUSION_POLICY_ROOT:-$SLURM_SUBMIT_DIR/diffusion_policy}"
# JAX must not preallocate the GPU — torch (the diffusion policy) shares it.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}"

CONDA_ENV=simpler_env DEVICE=cuda:0 \
  VERIFIER_TYPE=dqc \
  DQC_CKPT_DIR="$DQC_CKPT_DIR" DQC_TEXT_EMB="$DQC_TEXT_EMB" DQC_DISCOUNT="${DQC_DISCOUNT:-0.98}" \
  VERIFIER_INSTRUCTION="${VERIFIER_INSTRUCTION:-put the eggplant in the basket}" \
  VERIFIER_IMAGE_KEY="${VERIFIER_IMAGE_KEY:-agentview_image}" \
  MODE="${MODE:-softmax}" SOFTMAX_TEMP="${SOFTMAX_TEMP:-1.0}" \
  BC_SAMPLER="${BC_SAMPLER:-ddpm}" \
  BC_VERIFIER_SCORE=chunk BC_VERIFIER_AGG=discounted_sum \
  N_SAMPLES="$N_SAMPLES" N_ACTION_STEPS=8 \
  START_SEED="${START_SEED:-1000}" MAX_STEPS="${MAX_STEPS:-120}" USE_EMA=1 \
  TASK="${TASK:-widowx_put_eggplant_in_basket}" \
  SAVE_VIDEOS="${SAVE_VIDEOS:-0}" \
  bash scriptsv2/eval_search/eval_search.sh "$CHECKPOINT" "$NUM_EPISODES" "$OUT_DIR"

echo "[bc_dqc] done -> $OUT_DIR/eval_log.json"
