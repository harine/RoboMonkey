#!/bin/bash
# TASK 3: BC diffusion ckpt + LLaVA verifier, softmax N-sweep, rollout 1.
#   Wraps the frozen BC ckpt with BCSearchWrapper + in-process LLaVA verifier;
#   sweeps N in 2^0..2^9; softmax selection; 50 episodes/cell; ddpm sampler so the
#   N candidates are diverse. 5-GPU worker pool on weirdlab gpu-l40s.
# DefaultTime is 1h -- MUST set --time.
#SBATCH --job-name=sweep_bc_llava
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=20-00:00:00
#SBATCH --output=data/eval/search_bc_sweep/_summaries/slurm-llava-%j.out
#SBATCH --error=data/eval/search_bc_sweep/_summaries/slurm-llava-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
mkdir -p data/eval/search_bc_sweep/_summaries

echo "[sweep_bc_llava] host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi -L || true

export DIFFUSION_POLICY_ROOT="${DIFFUSION_POLICY_ROOT:-$SLURM_SUBMIT_DIR/diffusion_policy}"
export MONKEY_VERIFIER_SRC="${MONKEY_VERIFIER_SRC:-$SLURM_SUBMIT_DIR/monkey-verifier/src}"
export MODEL_DIR="${MODEL_DIR:-$SLURM_SUBMIT_DIR/monkey-verifier/model_dir}"
export CONDA_ENV="${CONDA_ENV:-monkey-verifier}"

# Verifier + rollout config read by eval_search.py via the env (propagated through
# eval_search_bc.sh's run_cell, which uses `env ... bash` and preserves the env).
export VERIFIER_TYPE="${VERIFIER_TYPE:-robomonkey}"   # LLaVA
export N_ACTION_STEPS="${N_ACTION_STEPS:-1}"          # rollout 1
export BC_VERIFIER_SCORE="${BC_VERIFIER_SCORE:-chunk}"
export VERIFIER_INSTRUCTION="${VERIFIER_INSTRUCTION:-put the eggplant in the basket}"
export VERIFIER_IMAGE_KEY="${VERIFIER_IMAGE_KEY:-agentview_image}"

CHECKPOINT="${CHECKPOINT:-$SLURM_SUBMIT_DIR/diffusion_policy/data/outputs/2026.06.11/13.52.09_robomonkey_eggplant_bc_robomonkey_eggplant_state/checkpoints/best.ckpt}"
export CHECKPOINT
if [[ ! -f "$CHECKPOINT" ]]; then echo "ERROR: ckpt not found: $CHECKPOINT" >&2; exit 1; fi

MODE="${MODE:-softmax}" \
    SOFTMAX_TEMP="${SOFTMAX_TEMP:-1.0}" \
    BC_SAMPLER="${BC_SAMPLER:-ddpm}" \
    VIZ_Q="${VIZ_Q:-0}" \
    BC_LABEL="${BC_LABEL:-bc_llava_h16_na1}" \
    NUM_EPISODES="${NUM_EPISODES:-50}" \
    N_LIST="${N_LIST:-1 2 4 8 16 32 64 128 256 512}" \
    START_SEED="${START_SEED:-1000}" \
    NUM_WORKERS="${NUM_WORKERS:-4}" \
    bash scriptsv2/eval_search/eval_search_bc.sh

echo "[sweep_bc_llava] done"
