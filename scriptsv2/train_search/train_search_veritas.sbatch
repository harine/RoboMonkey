#!/bin/bash
#SBATCH --job-name=tr_veritas
#SBATCH --account=robotics
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=data/train/veritas_search/slurm-%x-%j.out
#SBATCH --error=data/train/veritas_search/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# Train the eggplant diffusion SEARCH policy against the Veritas PIXEL-WAYPOINT verifier
# (outer/inner loop), on a strict-split manifest with precomputed per-frame targets.
#
# Required: SPLIT_FILE (config/splits/simpler/<task>/<rule>_train<N>.json),
#           PLAN_DIR (holds veritas_targets.npy/.json from build_veritas_targets.py).
# Optional: EXTRA_OVERRIDES (extra hydra overrides, e.g. logging.mode=offline),
#           ACTION_POS_SCALE (null | [x,y,z]), VALUE_MEAN / VALUE_STD (defaults read from
#           PLAN_DIR/veritas_targets.json), CAMERA_FILE, RUN_NAME, DATASET_DIR.
#           OBS_NOISE_LEVEL (0.0 clean, null random timestep, >0 fixed noise level).
# Geometry: predict 8, score and execute 4 before replanning.
# -----------------------------------------------------------------------------
set -euo pipefail
# Gemini key: from the environment, else the user's key file (never passed via --export,
# which would expose it in the job's recorded environment).
if [ -z "${GEMINI_API_KEY:-}" ] && [ -f "$HOME/.config/gemini/api_key" ]; then
  export GEMINI_API_KEY="$(tr -d '[:space:]' < "$HOME/.config/gemini/api_key")"
fi
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
mkdir -p data/train/veritas_search

CONFIG="${CONFIG:-robomonkey_eggplant_search_veritas_outer_inner}"
TASK="${TASK:-widowx_put_eggplant_in_basket}"
SPLIT_FILE="${SPLIT_FILE:?}"
DATASET_DIR="${DATASET_DIR:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["dataset_dir"])' "$SPLIT_FILE")}"
PLAN_DIR="${PLAN_DIR:?}"
TARGETS_NPY="${TARGETS_NPY:-$PLAN_DIR/veritas_targets.npy}"
MAX_ACTIONS="${MAX_ACTIONS:-16}"
ACTION_POS_SCALE="${ACTION_POS_SCALE:-null}"
[ -f "$TARGETS_NPY" ] || { echo "missing $TARGETS_NPY (run build_veritas_targets.py)"; exit 1; }
CALIBRATION_FILE="$(realpath "${PLAN_DIR}/veritas_targets.json")"
OBS_NOISE_LEVEL="${OBS_NOISE_LEVEL:-0.0}" # 0 = clean; null = random DDPM timestep

MAX_GRADIENT_STEPS="${MAX_GRADIENT_STEPS:-20000}"
OUTER_BATCH_SIZE="${OUTER_BATCH_SIZE:-512}"
INNER_BATCH_SIZE="${INNER_BATCH_SIZE:-256}"
INNER_EPOCHS="${INNER_EPOCHS:-4}"
DRIFT_EVERY="${DRIFT_EVERY:-4}"
VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-1000}"
SAMPLE_EVERY_STEPS="${SAMPLE_EVERY_STEPS:-1000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
NRMSE_MAX_BATCHES="${NRMSE_MAX_BATCHES:-8}"
MAX_VAL_STEPS="${MAX_VAL_STEPS:-20}"
LR_SCHEDULER="${LR_SCHEDULER:-polynomial}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-500}"
LR_DECAY_STEPS="${LR_DECAY_STEPS:-10000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_PROJECT="${LOG_PROJECT:-veritas_verifier_simpler}"
RUN_NAME="${RUN_NAME:-veritas_$(basename "$SPLIT_FILE" .json)_ie${INNER_EPOCHS}}"
OUT_ROOT="${OUT_ROOT:-/gscratch/robotics/harine/RoboMonkey_data/train/veritas_search}"
OUTDIR="${OUTDIR:-${OUT_ROOT}/${RUN_NAME}}"
CONDA_ENV="${CONDA_ENV:-dqc_jax}"
DEBUG="${DEBUG:-False}"

export DIFFUSION_POLICY_ROOT="${DIFFUSION_POLICY_ROOT:-${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}/diffusion_policy}"
CAMERA_FILE="${CAMERA_FILE:-$DIFFUSION_POLICY_ROOT/diffusion_policy/config/veritas/camera_${TASK}.json}"
export MUJOCO_GL=${MUJOCO_GL:-osmesa}; export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}; export DISPLAY=""
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="${HOME}/miniconda3/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${DIFFUSION_POLICY_ROOT}:${PYTHONPATH:-}"
SPLIT_ABS="$(realpath "$SPLIT_FILE")"; TARGETS_ABS="$(realpath "$TARGETS_NPY")"
CAMERA_FILE="$(realpath "$CAMERA_FILE")"
cd "$DIFFUSION_POLICY_ROOT"

# Reject old/all-split statistics instead of silently training with incompatible feedback.
CALIBRATION_VALUES=$(python - "$CALIBRATION_FILE" "$SPLIT_ABS" "$ACTION_POS_SCALE" "$CAMERA_FILE" <<'PY'
import json, sys
from diffusion_policy.common.veritas_calibration import validate_calibration
meta, manifest = (json.load(open(p)) for p in sys.argv[1:3])
validate_calibration(meta, manifest, 4, json.loads(sys.argv[3]), sys.argv[4])
print(meta['value_mean'], meta['value_std'])
PY
)
read -r CALIBRATED_MEAN CALIBRATED_STD <<< "$CALIBRATION_VALUES"
VALUE_MEAN="${VALUE_MEAN:-$CALIBRATED_MEAN}"
VALUE_STD="${VALUE_STD:-$CALIBRATED_STD}"

echo "[tr_veritas] host=$(hostname) config=$CONFIG split=$SPLIT_ABS targets=$TARGETS_ABS"
echo "[tr_veritas] verifier: veritas scale=$ACTION_POS_SCALE value mean=$VALUE_MEAN std=$VALUE_STD camera=$CAMERA_FILE"
echo "[tr_veritas] outer_bs=$OUTER_BATCH_SIZE inner_bs=$INNER_BATCH_SIZE inner_epochs=$INNER_EPOCHS steps=$MAX_GRADIENT_STEPS out=$OUTDIR"

python diffusion_policy/workspace/train_search_outer_inner_robomonkey_workspace.py \
    --config-name="${CONFIG}" \
    policy.obs_noise_level="${OBS_NOISE_LEVEL}" \
    policy.max_actions="${MAX_ACTIONS}" \
    policy.verifier.camera_file="${CAMERA_FILE}" \
    policy.verifier.action_pos_scale="${ACTION_POS_SCALE}" \
    policy.verifier.value_mean="${VALUE_MEAN}" \
    policy.verifier.value_std="${VALUE_STD}" \
    task.dataset.dataset_dir="${DATASET_DIR}" \
    task.dataset.split_file="${SPLIT_ABS}" \
    task.dataset.extra_obs_arrays.veritas_targets="${TARGETS_ABS}" \
    task.dataset.use_cache=true \
    training.debug="${DEBUG}" \
    training.max_gradient_steps="${MAX_GRADIENT_STEPS}" \
    training.outer_batch_size="${OUTER_BATCH_SIZE}" \
    training.inner_batch_size="${INNER_BATCH_SIZE}" \
    training.inner_epochs="${INNER_EPOCHS}" \
    training.drift_every="${DRIFT_EVERY}" \
    training.rollout_every_steps="${ROLLOUT_EVERY_STEPS:-5000}" \
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
    logging.project="${LOG_PROJECT}" \
    logging.name="${RUN_NAME}" \
    +logging.entity=l2sml \
    hydra.run.dir="${OUTDIR}" \
    ${EXTRA_OVERRIDES:-}

echo "[tr_veritas] done -> ${OUTDIR}/checkpoints/latest.ckpt"
