#!/bin/bash
# TASK 1: train a conditional DIFFUSION search policy
# (SearchPolicyRoboMonkeyDiffusion) with the RoboMonkey LLaVA verifier.
#   - config robomonkey_eggplant_search_state_diffusion (verifier = RoboMonkeyVerifier, in_process)
#   - horizon 16, n_action_steps 1 (ROLLOUT 1 -- execute 1 of 16)
#   - score_executed_chunk=true -> verifier scores the single executed step
#   - policy.max_actions=16 (search breadth)
#   - runs in monkey-verifier conda env (LLaVA-7b in process)
# DefaultTime is 1h on these partitions -- MUST set --time.
#SBATCH --job-name=tr_diff_llava_na1
#SBATCH --account=robotics
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --time=20-00:00:00
#SBATCH --output=data/train/diff_llava_na1/slurm-%j.out
#SBATCH --error=data/train/diff_llava_na1/slurm-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
mkdir -p data/train/diff_llava_na1

echo "[tr_diff_llava_na1] host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi -L || true

CONDA_ENV="${CONDA_ENV:-monkey-verifier}"
# 20k-step budget with the POLYNOMIAL DECOUPLED LR schedule — same as the DQC
# verifier search-policy training (train_dqc.sbatch): linear warmup 0->1e-4 over
# LR_WARMUP, then polynomial decay to lr_end over LR_DECAY_STEPS, then HOLD lr_end
# for the rest. The decay is DECOUPLED from num_epochs (cosine bakes the span from
# len(dataloader)*num_epochs and would ramp back up past its endpoint), so
# num_epochs only sets the total budget and MAX_GRADIENT_STEPS caps it. Checkpoints
# land every 1000 steps.
MAX_GRADIENT_STEPS="${MAX_GRADIENT_STEPS:-20000}"
# polynomial LR decoupled from num_epochs -> set num_epochs high enough that
# max_gradient_steps is the cap (~50 steps/epoch on this run; 450*50 ~= 22.5k > 20k).
NUM_EPOCHS="${NUM_EPOCHS:-450}"
LR_SCHEDULER="${LR_SCHEDULER:-polynomial}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-500}"
LR_DECAY_STEPS="${LR_DECAY_STEPS:-10000}"   # decay to lr_end by step 10k, then HOLD
MAX_ACTIONS="${MAX_ACTIONS:-16}"
N_ACTION_STEPS="${N_ACTION_STEPS:-1}"
DATASET_DIR="${DATASET_DIR:-/gscratch/robotics/harine/data/eggplant_in_basket_complete}"
OUT_ROOT="${OUT_ROOT:-/gscratch/robotics/harine/RoboMonkey_data/train/diff_llava_na1}"
LOG_PROJECT="${LOG_PROJECT:-robomonkey_search_policy}"
# run-name convention: arch _ verifier _ hyak _ predhorizon<H> _ rollout<R>.
# This is the conditional-diffusion + LLaVA search policy (predhorizon from the
# config = 16 here; rollout = n_action_steps).
PRED_HORIZON="${PRED_HORIZON:-16}"
RUN_NAME="${RUN_NAME:-eggplant_diffusion_llava_hyak_predhorizon${PRED_HORIZON}_rollout${N_ACTION_STEPS}}"
OUTDIR="${OUTDIR:-${OUT_ROOT}/${RUN_NAME}}"

export DIFFUSION_POLICY_ROOT="${DIFFUSION_POLICY_ROOT:-$SLURM_SUBMIT_DIR/diffusion_policy}"
export MONKEY_VERIFIER_SRC="${MONKEY_VERIFIER_SRC:-$SLURM_SUBMIT_DIR/monkey-verifier/src}"
export MODEL_DIR="${MODEL_DIR:-$SLURM_SUBMIT_DIR/monkey-verifier/model_dir}"
# RoboMonkey verifier speedups (in-process LLaVA): KV-prefix cache (image+prompt
# prefix reused across candidates) + image-feature / processed-image caches.
export ROBOMONKEY_KV_PREFIX="${ROBOMONKEY_KV_PREFIX:-1}"
export ROBOMONKEY_KV_CHUNK="${ROBOMONKEY_KV_CHUNK:-16}"
export ROBOMONKEY_KV_FWD_CHUNK="${ROBOMONKEY_KV_FWD_CHUNK:-16}"
export ROBOMONKEY_IMAGE_FEAT_CACHE_SIZE="${ROBOMONKEY_IMAGE_FEAT_CACHE_SIZE:-512}"
export ROBOMONKEY_PROC_IMAGE_CACHE_SIZE="${ROBOMONKEY_PROC_IMAGE_CACHE_SIZE:-512}"
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
export DISPLAY=""

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="${HOME}/miniconda3/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${DIFFUSION_POLICY_ROOT}:${MONKEY_VERIFIER_SRC}:${PYTHONPATH:-}"
cd "$DIFFUSION_POLICY_ROOT"

echo "[tr_diff_llava_na1] config=robomonkey_eggplant_search_state_diffusion n_action_steps=$N_ACTION_STEPS"
echo "[tr_diff_llava_na1] max_actions=$MAX_ACTIONS max_grad_steps=$MAX_GRADIENT_STEPS num_epochs=$NUM_EPOCHS out=$OUTDIR"
echo "[tr_diff_llava_na1] wandb=l2sml/${LOG_PROJECT} run=${RUN_NAME}"

python diffusion_policy/workspace/train_mlp_image_workspace.py \
    --config-name=robomonkey_eggplant_search_state_diffusion \
    n_action_steps="${N_ACTION_STEPS}" \
    policy.max_actions="${MAX_ACTIONS}" \
    policy.verifier.server_url=in_process \
    +policy.verifier.score_executed_chunk=true \
    training.device=cuda:0 \
    training.num_epochs="${NUM_EPOCHS}" \
    +training.max_gradient_steps="${MAX_GRADIENT_STEPS}" \
    training.lr_scheduler="${LR_SCHEDULER}" \
    training.lr_warmup_steps="${LR_WARMUP_STEPS}" \
    +training.lr_decay_steps="${LR_DECAY_STEPS}" \
    task.dataset.use_cache=true \
    task.dataset.dataset_dir="${DATASET_DIR}" \
    logging.project="${LOG_PROJECT}" \
    logging.name="${RUN_NAME}" \
    +logging.entity=l2sml \
    hydra.run.dir="${OUTDIR}"

echo "[tr_diff_llava_na1] done -> ${OUTDIR}/checkpoints/latest.ckpt"
