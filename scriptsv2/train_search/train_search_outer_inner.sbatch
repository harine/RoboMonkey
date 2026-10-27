#!/bin/bash
#SBATCH --job-name=tr_oi
#SBATCH --account=robotics
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --output=data/train/outer_inner/slurm-%x-%j.out
#SBATCH --error=data/train/outer_inner/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# Train a diffusion SEARCH policy with the OUTER/INNER trainer.
#
# Same policy, same data, same loss as train_search_pg.sbatch with PG_LAMBDA=0 --
# only the training LOOP differs. The offline trainer regenerates the search
# context inside compute_loss on EVERY gradient step (max_actions-1 = 15 DDIM
# samples + 15 DQC verifier calls per batch, then discarded). Here the context is
# generated once per outer step for a pool of OUTER_BATCH_SIZE windows and reused
# for INNER_EPOCHS passes, dividing the search cost per update by INNER_EPOCHS.
#
# The cost is staleness: the buffered context comes from weights up to num_inner
# updates old. A frozen collector snapshot is kept and train_drift_mse_eps reports
# how far the live policy has moved from it -- watch it rise across an inner loop
# and reset at each outer step. If it grows large, lower INNER_EPOCHS.
#
# Per-task knobs (REQUIRED for stack/spoon -- see the launcher):
#   DATASET_DIR   task zarr root
#   TEXT_EMB      per-task MUSE/USE-3 embedding for the language-conditioned critic.
#                 Needed for TRAINING, not just eval: generate_search_context calls
#                 the verifier every outer step to build the context the policy
#                 conditions on, so a wrong embedding poisons the context.
#   INSTRUCTION   cosmetic when TEXT_EMB resolves (the .npy wins), but it names the
#                 task in the verifier health-check log.
#
# LR schedule matches the offline multitask runs so the arms stay comparable:
# polynomial, warmup 500, decay over LR_DECAY_STEPS then hold.
# Runs in dqc_jax (the JAX critic shares the GPU with the torch policy).
# resume=True + a fixed hydra.run.dir -> wall/preempt safe.
# -----------------------------------------------------------------------------
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
mkdir -p data/train/outer_inner

CONFIG="${CONFIG:-robomonkey_eggplant_search_dqc_outer_inner}"
CRITIC_DIR="${CRITIC_DIR:-/gscratch/robotics/harine/dqc_checkpoint/sd000s_137913.0.20260605_163017}"
RESTORE_EPOCH="${RESTORE_EPOCH:-100000}"
DATASET_DIR="${DATASET_DIR:-/gscratch/robotics/harine/data/eggplant_in_basket_complete}"
TEXT_EMB="${TEXT_EMB:-/mmfs1/home/harine/RoboMonkey/scriptsv2/q_fn_eval/task_emb_use3.npy}"
INSTRUCTION="${INSTRUCTION:-put the eggplant in the basket}"
MAX_ACTIONS="${MAX_ACTIONS:-16}"

# ---- outer/inner geometry ---------------------------------------------------
MAX_GRADIENT_STEPS="${MAX_GRADIENT_STEPS:-20000}"
OUTER_BATCH_SIZE="${OUTER_BATCH_SIZE:-512}"
INNER_BATCH_SIZE="${INNER_BATCH_SIZE:-256}"
INNER_EPOCHS="${INNER_EPOCHS:-4}"
DRIFT_EVERY="${DRIFT_EVERY:-4}"
# cadences in GRADIENT STEPS (this workspace has no epoch-based cadences)
ROLLOUT_EVERY_STEPS="${ROLLOUT_EVERY_STEPS:-5000}"
VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-1000}"
SAMPLE_EVERY_STEPS="${SAMPLE_EVERY_STEPS:-1000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
NRMSE_MAX_BATCHES="${NRMSE_MAX_BATCHES:-4}"
MAX_VAL_STEPS="${MAX_VAL_STEPS:-20}"

# ---- LR (matches what the offline multitask runs ACTUALLY did) --------------
# polynomial: warmup 500 -> lr, decay over LR_DECAY_STEPS, then hold.
# NOTE deliberately NOT passing lr_end / lr_power. train_search_pg.sbatch passes
# `+training.lr_end=1e-5 +training.lr_power=1.0` and its header claims a 1e-5 floor,
# but no workspace forwards those keys to get_scheduler (only lr_scheduler_kwargs is
# forwarded) -- a grep finds lr_end in comments only. So the offline runs really
# decayed to diffusers' default 1e-7. Passing them here would silently change the
# schedule relative to the arms this is meant to be compared against. See
# diffusion_policy/AUDIT.md P2-9.
LR_SCHEDULER="${LR_SCHEDULER:-polynomial}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-500}"
LR_DECAY_STEPS="${LR_DECAY_STEPS:-10000}"

NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_PROJECT="${LOG_PROJECT:-dqc_search_outer_inner}"
RUN_NAME="${RUN_NAME:-oi_eggplant_ie${INNER_EPOCHS}}"
OUT_ROOT="${OUT_ROOT:-/gscratch/robotics/harine/RoboMonkey_data/train/outer_inner}"
OUTDIR="${OUTDIR:-${OUT_ROOT}/${RUN_NAME}}"
CONDA_ENV="${CONDA_ENV:-dqc_jax}"
DEBUG="${DEBUG:-False}"

export DIFFUSION_POLICY_ROOT="${DIFFUSION_POLICY_ROOT:-${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}/diffusion_policy}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"
export DQC_RESTORE_EPOCH="${RESTORE_EPOCH}"
export MUJOCO_GL=${MUJOCO_GL:-osmesa}; export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}; export DISPLAY=""

source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="${HOME}/miniconda3/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${DIFFUSION_POLICY_ROOT}:${PYTHONPATH:-}"
cd "$DIFFUSION_POLICY_ROOT"

echo "[tr_oi] host=$(hostname) config=$CONFIG debug=$DEBUG"
echo "[tr_oi] data=$DATASET_DIR"
echo "[tr_oi] emb=$(basename "$TEXT_EMB")  instr='$INSTRUCTION'  critic=$CRITIC_DIR (restore $RESTORE_EPOCH)"
echo "[tr_oi] outer_bs=$OUTER_BATCH_SIZE inner_bs=$INNER_BATCH_SIZE inner_epochs=$INNER_EPOCHS -> ${INNER_EPOCHS}x fewer searches/update"
echo "[tr_oi] max_grad_steps=$MAX_GRADIENT_STEPS  out=$OUTDIR"
python -c "import jax; print('   jax devices:', jax.devices())" 2>&1 | tail -1

EXTRA=()
VAL_DATASET_DIR="${VAL_DATASET_DIR:-}"
if [ -n "$VAL_DATASET_DIR" ]; then
    EXTRA+=( "+task.dataset.val_dataset_dir=${VAL_DATASET_DIR}" "task.dataset.val_ratio=0.0" )
fi
[ -n "${TAGS:-}" ] && EXTRA+=( "logging.tags=${TAGS}" )

python diffusion_policy/workspace/train_search_outer_inner_robomonkey_workspace.py \
    --config-name="${CONFIG}" \
    policy.obs_noise_level=0.0 \
    policy.max_actions="${MAX_ACTIONS}" \
    policy.verifier.ckpt_dir="${CRITIC_DIR}" \
    policy.verifier.text_emb_file="${TEXT_EMB}" \
    policy.verifier.instruction="${INSTRUCTION}" \
    training.debug="${DEBUG}" \
    training.max_gradient_steps="${MAX_GRADIENT_STEPS}" \
    training.outer_batch_size="${OUTER_BATCH_SIZE}" \
    training.inner_batch_size="${INNER_BATCH_SIZE}" \
    training.inner_epochs="${INNER_EPOCHS}" \
    training.drift_every="${DRIFT_EVERY}" \
    training.rollout_every_steps="${ROLLOUT_EVERY_STEPS}" \
    training.val_every_steps="${VAL_EVERY_STEPS}" \
    training.sample_every_steps="${SAMPLE_EVERY_STEPS}" \
    training.checkpoint_every="${CHECKPOINT_EVERY}" \
    training.nrmse_max_batches="${NRMSE_MAX_BATCHES}" \
    training.max_val_steps="${MAX_VAL_STEPS}" \
    training.lr_scheduler="${LR_SCHEDULER}" \
    training.lr_warmup_steps="${LR_WARMUP_STEPS}" \
    +training.lr_decay_steps="${LR_DECAY_STEPS}" \
    dataloader.num_workers="${NUM_WORKERS}" \
    val_dataloader.num_workers=2 \
    task.dataset.use_cache=true \
    task.dataset.dataset_dir="${DATASET_DIR}" \
    logging.project="${LOG_PROJECT}" \
    logging.name="${RUN_NAME}" \
    +logging.entity=l2sml \
    hydra.run.dir="${OUTDIR}" \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "[tr_oi] done -> ${OUTDIR}/checkpoints/latest.ckpt"
