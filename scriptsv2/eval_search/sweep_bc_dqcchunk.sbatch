#!/bin/bash
# BC diffusion ckpt + REAL DQC CHUNK critic verifier, softmax N-sweep, rollout 8.
#   Wraps the frozen BC ckpt with BCSearchWrapper + in-process DQC chunk critic
#   (BridgeDQCChunkAgent; scores the whole 8-step executed window in one forward).
#   Sweeps N in 2^0..2^9; softmax selection; 50 ep/cell; ddpm sampler (diverse).
#   Runs in the `dqc` env (here dqc_repo: torch + jax0.4.28/flax0.8.5). JAX shares
#   each GPU with the torch BC policy.
# DefaultTime is 1h -- MUST set --time (cap < Jul-14 maintenance -> 20d).
#SBATCH --job-name=sweep_bc_dqcchunk
#SBATCH --cpus-per-task=15
#SBATCH --mem=300G
#SBATCH --time=20-00:00:00
#SBATCH --output=data/eval/search_bc_sweep/_summaries/slurm-dqcchunk-%j.out
#SBATCH --error=data/eval/search_bc_sweep/_summaries/slurm-dqcchunk-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
mkdir -p data/eval/search_bc_sweep/_summaries

echo "[sweep_bc_dqcchunk] host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi -L || true

export DIFFUSION_POLICY_ROOT="${DIFFUSION_POLICY_ROOT:-$SLURM_SUBMIT_DIR/diffusion_policy}"
export CONDA_ENV="${CONDA_ENV:-dqc_repo}"   # torch + jax0.4.28/flax0.8.5 + real verifier

# DQC chunk critic config (read by eval_search.py via inherited env).
export VERIFIER_TYPE="${VERIFIER_TYPE:-dqc}"
export DQC_CRITIC_TYPE="${DQC_CRITIC_TYPE:-chunk}"
export DQC_CKPT_DIR="${DQC_CKPT_DIR:-/gscratch/robotics/harine/dqc_checkpoint/sd000s_137913.0.20260605_163017}"
export DQC_TEXT_EMB="${DQC_TEXT_EMB:-$SLURM_SUBMIT_DIR/scriptsv2/q_fn_eval/task_emb_use3.npy}"
export N_ACTION_STEPS="${N_ACTION_STEPS:-8}"          # rollout 8 (chunk critic needs 8)
export BC_VERIFIER_SCORE="${BC_VERIFIER_SCORE:-chunk_critic}"
export VERIFIER_INSTRUCTION="${VERIFIER_INSTRUCTION:-put the eggplant in the basket}"
export VERIFIER_IMAGE_KEY="${VERIFIER_IMAGE_KEY:-agentview_image}"
# DQC (JAX) shares the GPU with the torch BC policy.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"

CHECKPOINT="${CHECKPOINT:-$SLURM_SUBMIT_DIR/diffusion_policy/data/outputs/2026.06.11/13.52.09_robomonkey_eggplant_bc_robomonkey_eggplant_state/checkpoints/best.ckpt}"
export CHECKPOINT
if [[ ! -f "$CHECKPOINT" ]]; then echo "ERROR: ckpt not found: $CHECKPOINT" >&2; exit 1; fi

MODE="${MODE:-softmax}" \
    SOFTMAX_TEMP="${SOFTMAX_TEMP:-1.0}" \
    BC_SAMPLER="${BC_SAMPLER:-ddpm}" \
    VIZ_Q="${VIZ_Q:-0}" \
    BC_LABEL="${BC_LABEL:-bc_dqcchunk_h16_na8}" \
    NUM_EPISODES="${NUM_EPISODES:-50}" \
    N_LIST="${N_LIST:-1 2 4 8 16 32 64 128 256 512}" \
    START_SEED="${START_SEED:-1000}" \
    NUM_WORKERS="${NUM_WORKERS:-3}" \
    bash scriptsv2/eval_search/eval_search_bc.sh

echo "[sweep_bc_dqcchunk] done"
