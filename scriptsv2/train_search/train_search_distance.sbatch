#!/bin/bash
#SBATCH --job-name=tr_dist
#SBATCH --account=robotics
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=data/train/distance_search/slurm-%x-%j.out
#SBATCH --error=data/train/distance_search/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# Train the eggplant diffusion SEARCH policy against the closed-form SHAPED
# DISTANCE verifier (no DQC critic, no JAX) on the 50-demo strict-distance subset.
#
#   value = -( ||ee - eggplant|| + ||eggplant - basket|| )  after the 8-action chunk
#
# Geometry is unchanged from the DQC arm: predict 8, score all 8, execute 4,
# re-search. The ONLY difference between this arm and oi_eggplant_ie4 is the verifier
# — which is the comparison the run exists to make.
#
# CALIBRATION IS NOT OPTIONAL. action_pos_scale (metres of EE travel per action unit)
# and value_mean/value_std (the scale the policy conditions on) are dataset constants.
# Wrong values do not crash — they quietly make every candidate look identical, so the
# search context carries no ranking signal and the run silently degenerates to plain
# BC. The launcher runs calibrate_distance_verifier.py and passes them in; submitting
# this sbatch directly without them trips the verifier's startup warnings.
#
# MEM is 32G, not the 64G the DQC arm used: no JAX critic shares this GPU and the
# sub50 zarr is 1223 steps. resume=True + a FIXED hydra.run.dir -> wall/preempt safe.
# -----------------------------------------------------------------------------
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
mkdir -p data/train/distance_search

CONFIG="${CONFIG:-robomonkey_eggplant_search_distance_outer_inner}"
DATASET_DIR="${DATASET_DIR:-/gscratch/robotics/harine/data/strict_distance_success_sub50_diverse_start}"
MAX_ACTIONS="${MAX_ACTIONS:-16}"

# ---- verifier constants (from calibrate_distance_verifier.py) ---------------
# PER-AXIS list, passed through to hydra as [x,y,z]. A bare scalar is accepted by the
# verifier but re-introduces the pooled-scalar bug — see calibrate_distance_verifier.py.
ACTION_POS_SCALE="${ACTION_POS_SCALE:-[1.0,1.0,1.0]}"
VALUE_MEAN="${VALUE_MEAN:-0.0}"
VALUE_STD="${VALUE_STD:-1.0}"
VALUE_NORMALIZE="${VALUE_NORMALIZE:-fixed}"
GRASP_KEY="${GRASP_KEY:-last_gripper_action}"
# chunk = gripper-gated (candidate's own gripper channel decides how far the object is
# carried); obs = grasp frozen at time t, the ablation.
GRASP_MODE="${GRASP_MODE:-chunk}"
W_REACH="${W_REACH:-1.0}"
W_PLACE="${W_PLACE:-1.0}"

# ---- outer/inner geometry ---------------------------------------------------
MAX_GRADIENT_STEPS="${MAX_GRADIENT_STEPS:-20000}"
OUTER_BATCH_SIZE="${OUTER_BATCH_SIZE:-512}"
INNER_BATCH_SIZE="${INNER_BATCH_SIZE:-256}"
INNER_EPOCHS="${INNER_EPOCHS:-4}"
DRIFT_EVERY="${DRIFT_EVERY:-4}"
ROLLOUT_EVERY_STEPS="${ROLLOUT_EVERY_STEPS:-5000}"
VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-1000}"
SAMPLE_EVERY_STEPS="${SAMPLE_EVERY_STEPS:-1000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
NRMSE_MAX_BATCHES="${NRMSE_MAX_BATCHES:-8}"
MAX_VAL_STEPS="${MAX_VAL_STEPS:-20}"

# LR matched to the DQC outer/inner arm so the two are comparable: polynomial,
# warmup 500, decay over LR_DECAY_STEPS then hold. See train_search_outer_inner.sbatch
# for why lr_end/lr_power are deliberately NOT passed (no workspace forwards them).
LR_SCHEDULER="${LR_SCHEDULER:-polynomial}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-500}"
LR_DECAY_STEPS="${LR_DECAY_STEPS:-10000}"

NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_PROJECT="${LOG_PROJECT:-distance_search_sub50}"
RUN_NAME="${RUN_NAME:-dist_eggplant_sub50_ie${INNER_EPOCHS}}"
OUT_ROOT="${OUT_ROOT:-/gscratch/robotics/harine/RoboMonkey_data/train/distance_search}"
OUTDIR="${OUTDIR:-${OUT_ROOT}/${RUN_NAME}}"
CONDA_ENV="${CONDA_ENV:-dqc_jax}"
DEBUG="${DEBUG:-False}"

export DIFFUSION_POLICY_ROOT="${DIFFUSION_POLICY_ROOT:-${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}/diffusion_policy}"
export MUJOCO_GL=${MUJOCO_GL:-osmesa}; export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}; export DISPLAY=""

source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="${HOME}/miniconda3/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${DIFFUSION_POLICY_ROOT}:${PYTHONPATH:-}"
cd "$DIFFUSION_POLICY_ROOT"

echo "[tr_dist] host=$(hostname) config=$CONFIG debug=$DEBUG"
echo "[tr_dist] data=$DATASET_DIR"
echo "[tr_dist] verifier: distance  scale=$ACTION_POS_SCALE  norm=$VALUE_NORMALIZE(mean=$VALUE_MEAN, std=$VALUE_STD)  grasp=$GRASP_KEY"
echo "[tr_dist] outer_bs=$OUTER_BATCH_SIZE inner_bs=$INNER_BATCH_SIZE inner_epochs=$INNER_EPOCHS"
echo "[tr_dist] max_grad_steps=$MAX_GRADIENT_STEPS  out=$OUTDIR"

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
    policy.verifier.action_pos_scale="${ACTION_POS_SCALE}" \
    policy.verifier.normalize="${VALUE_NORMALIZE}" \
    policy.verifier.value_mean="${VALUE_MEAN}" \
    policy.verifier.value_std="${VALUE_STD}" \
    policy.verifier.grasp_key="${GRASP_KEY}" \
    policy.verifier.grasp_mode="${GRASP_MODE}" \
    policy.verifier.w_reach="${W_REACH}" \
    policy.verifier.w_place="${W_PLACE}" \
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

echo "[tr_dist] done -> ${OUTDIR}/checkpoints/latest.ckpt"
