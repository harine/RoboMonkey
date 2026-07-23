#!/bin/bash
# Train a search policy supervised by the REAL DQC CHUNK Q-function verifier
# (BridgeDQCChunkAgent, flax) — horizon 8, rollout 4, the chunk critic scores all
# 8 predicted actions as one chunk a_{t:t+8}.
#
#   CONFIG=robomonkey_eggplant_search_dqc           -> conditional diffusion (default)
#   CONFIG=robomonkey_eggplant_search_dqc_gaussian  -> transformer/gaussian
#
# Runs in the `dqc` conda env (torch + jax0.4.x/flax0.8.x + distrax for the real
# verifier). JAX shares the GPU with the torch policy (no prealloc; CUDA graphs off).
# DefaultTime is 1h on these partitions -- MUST set --time (job blocked by the
# Jul-14 maintenance reservation above ~25d, so cap at 20d).
#SBATCH --job-name=tr_dqc_chunk
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --time=20-00:00:00
#SBATCH --output=data/train/dqc_chunk/slurm-%x-%j.out
#SBATCH --error=data/train/dqc_chunk/slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
mkdir -p data/train/dqc_chunk

echo "[tr_dqc_chunk] host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi -L || true

CONDA_ENV="${CONDA_ENV:-dqc_repo}"   # dqc_repo extended with torch + diffusion deps
CONFIG="${CONFIG:-robomonkey_eggplant_search_dqc}"
# Real ckpt lives on gscratch (the config's /home path doesn't resolve here).
DQC_CKPT_DIR="${DQC_CKPT_DIR:-/gscratch/robotics/harine/dqc_checkpoint/sd000s_137913.0.20260605_163017}"
MAX_GRADIENT_STEPS="${MAX_GRADIENT_STEPS:-50000}"
# cosine LR is baked at init from len(dataloader)*num_epochs -> size num_epochs to
# the step budget (~76 steps/epoch on eggplant). 50k/76 ~= 658.
NUM_EPOCHS="${NUM_EPOCHS:-658}"
MAX_ACTIONS="${MAX_ACTIONS:-16}"
OBS_NOISE_LEVEL="${OBS_NOISE_LEVEL:-0.0}"
DATASET_DIR="${DATASET_DIR:-/gscratch/robotics/harine/data/eggplant_in_basket_complete}"
OUT_ROOT="${OUT_ROOT:-/gscratch/robotics/harine/RoboMonkey_data/train/dqc_chunk}"
LOG_PROJECT="${LOG_PROJECT:-dqc_search_policy}"

# --- run-name convention: arch _ verifier _ hyak _ predhorizon<H> _ rollout<R> ---
# arch: gaussian | diffusion (from CONFIG); verifier: dqc_chunk8 (8-step chunk
# critic) | dqc_chunk1 (single-step action critic) | llava. predhorizon/rollout
# are name-only (the config sets the real values; keep these in sync if you change it).
case "$CONFIG" in
  *gaussian*) ARCH=gaussian ;;
  *)          ARCH=diffusion ;;
esac
CRITIC_TYPE="${CRITIC_TYPE:-chunk}"             # chunk (8-step) | action (single-step)
if [ "$CRITIC_TYPE" = "action" ]; then VTAG=dqc_chunk1; else VTAG=dqc_chunk8; fi
PRED_HORIZON="${PRED_HORIZON:-8}"
ROLLOUT="${ROLLOUT:-4}"
RUN_NAME="${RUN_NAME:-eggplant_${ARCH}_${VTAG}_hyak_predhorizon${PRED_HORIZON}_rollout${ROLLOUT}}"
OUTDIR="${OUTDIR:-${OUT_ROOT}/${RUN_NAME}}"

export DIFFUSION_POLICY_ROOT="${DIFFUSION_POLICY_ROOT:-$SLURM_SUBMIT_DIR/diffusion_policy}"
# Real DQC verifier (JAX) shares the GPU with the torch policy.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
export DISPLAY=""

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="${HOME}/miniconda3/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${DIFFUSION_POLICY_ROOT}:${PYTHONPATH:-}"
cd "$DIFFUSION_POLICY_ROOT"

echo "[tr_dqc_chunk] env=$CONDA_ENV config=$CONFIG ckpt=$DQC_CKPT_DIR"
python -c "import jax,torch; print('   jax',jax.__version__,'devs',jax.devices(),'torch',torch.__version__)" 2>&1 | grep -viE "warning|gpu may" | tail -2
echo "[tr_dqc_chunk] max_actions=$MAX_ACTIONS max_grad_steps=$MAX_GRADIENT_STEPS num_epochs=$NUM_EPOCHS out=$OUTDIR"

python diffusion_policy/workspace/train_mlp_image_workspace.py \
    --config-name="${CONFIG}" \
    policy.obs_noise_level="${OBS_NOISE_LEVEL}" \
    policy.max_actions="${MAX_ACTIONS}" \
    policy.verifier.ckpt_dir="${DQC_CKPT_DIR}" \
    policy.verifier.critic_type="${CRITIC_TYPE}" \
    training.device=cuda:0 \
    training.num_epochs="${NUM_EPOCHS}" \
    +training.max_gradient_steps="${MAX_GRADIENT_STEPS}" \
    task.dataset.use_cache=true \
    task.dataset.dataset_dir="${DATASET_DIR}" \
    logging.project="${LOG_PROJECT}" \
    logging.name="${RUN_NAME}" \
    +logging.entity=l2sml \
    hydra.run.dir="${OUTDIR}"

echo "[tr_dqc_chunk] done -> ${OUTDIR}/checkpoints/latest.ckpt"
