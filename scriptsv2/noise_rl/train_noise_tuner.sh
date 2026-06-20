#!/bin/bash
# Train the tmrl-style noise-level SAC tuner on top of a frozen
# SearchPolicyRoboMonkeyDiffusionNoiseCond checkpoint.
#
# The base policy + verifier are not touched. Only a small (actor + twin
# critic) head learns to pick tcont_context per replan from the current
# state. Eval at deploy time: read tcont from `tuner.act(state)` and pass
# it to `policy.predict_n_actions(..., tcont_context=tcont)`.
#
# Usage
# -----
#   bash scriptsv2/noise_rl/train_noise_tuner.sh <noise_cond_ckpt> [num_env_steps] [output_dir]
#
# Example
# -------
#   bash scriptsv2/noise_rl/train_noise_tuner.sh \
#       diffusion_policy/data/outputs/2026.06.0X/.../checkpoints/latest.ckpt \
#       20000 \
#       data/noise_tuner/eggplant
#
# Env vars (override on cmdline):
#   DEVICE         (default: cuda:0)
#   CONDA_ENV      (default: monkey-verifier)
#   TASK           (default: widowx_put_eggplant_in_basket)
#   MAX_STEPS      (default: 120)
#   N_SAMPLES      (default: 16; chunks scored per replan)
#   BUFFER_SIZE    (default: 50000)
#   BATCH_SIZE     (default: 256)
#   WARMUP         (default: 200; SAC updates start once buffer >= this)
#   UPDATES_PER_CHUNK (default: 1)
#   INIT_ALPHA     (default: 0.1)
#   GAMMA          (default: 0.99)
#   TAU            (default: 0.005)
#   START_SEED     (default: 2000)
#   USE_EMA        (default: 1; set 0 to load raw weights)

set -euo pipefail

CKPT="${1:-}"
NUM_ENV_STEPS="${2:-20000}"
OUT_DIR_ARG="${3:-}"

if [[ -z "$CKPT" ]]; then
    echo "usage: bash $0 <noise_cond_ckpt> [num_env_steps] [output_dir]" >&2
    exit 1
fi
if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: checkpoint not found: $CKPT" >&2
    exit 1
fi

if [[ -z "$OUT_DIR_ARG" ]]; then
    RUN_DIR="$(dirname "$(dirname "$CKPT")")"
    RUN_NAME="$(basename "$RUN_DIR")"
    OUT_DIR="data/noise_tuner/${RUN_NAME}"
else
    OUT_DIR="$OUT_DIR_ARG"
fi

DEVICE="${DEVICE:-cuda:0}"
CONDA_ENV="${CONDA_ENV:-monkey-verifier}"
TASK="${TASK:-widowx_put_eggplant_in_basket}"
MAX_STEPS="${MAX_STEPS:-120}"
N_SAMPLES="${N_SAMPLES:-16}"
BUFFER_SIZE="${BUFFER_SIZE:-50000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
WARMUP="${WARMUP:-200}"
UPDATES_PER_CHUNK="${UPDATES_PER_CHUNK:-1}"
INIT_ALPHA="${INIT_ALPHA:-0.1}"
GAMMA="${GAMMA:-0.99}"
TAU="${TAU:-0.005}"
START_SEED="${START_SEED:-2000}"
USE_EMA="${USE_EMA:-1}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
export DISPLAY=""
export LD_LIBRARY_PATH="${HOME}/miniconda3/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

full_path="$(realpath "$0")"
dir_path="$(dirname "$full_path")"
repo_root="$(cd "$dir_path/../.." && pwd)"
cd "$repo_root"

DP_ROOT="${DIFFUSION_POLICY_ROOT:-${repo_root}/diffusion_policy}"
export PYTHONPATH="${DP_ROOT}:${PYTHONPATH:-}"

mkdir -p "$OUT_DIR"

EXTRA_FLAGS=()
if [[ "$USE_EMA" == "0" ]]; then
    EXTRA_FLAGS+=(--no-ema)
fi

echo "============================================================"
echo "  noise-tuner SAC training"
echo "  checkpoint   : $CKPT"
echo "  task         : $TASK"
echo "  num_env_steps: $NUM_ENV_STEPS"
echo "  output_dir   : $OUT_DIR"
echo "  device       : $DEVICE"
echo "  n_samples    : $N_SAMPLES"
echo "  buffer_size  : $BUFFER_SIZE  batch_size=$BATCH_SIZE  warmup=$WARMUP"
echo "  sac          : alpha=$INIT_ALPHA  gamma=$GAMMA  tau=$TAU"
echo "============================================================"

xvfb-run --auto-servernum -s "-screen 0 640x480x24" \
    python "$dir_path/train_noise_tuner.py" \
        --checkpoint "$CKPT" \
        --output-dir "$OUT_DIR" \
        --task "$TASK" \
        --num-env-steps "$NUM_ENV_STEPS" \
        --max-episode-steps "$MAX_STEPS" \
        --n-samples "$N_SAMPLES" \
        --buffer-size "$BUFFER_SIZE" \
        --batch-size "$BATCH_SIZE" \
        --warmup-transitions "$WARMUP" \
        --updates-per-chunk "$UPDATES_PER_CHUNK" \
        --init-alpha "$INIT_ALPHA" \
        --gamma "$GAMMA" \
        --tau "$TAU" \
        --start-seed "$START_SEED" \
        --device "$DEVICE" \
        "${EXTRA_FLAGS[@]}"

echo
echo "[noise_tuner] done. Tuner ckpts -> $OUT_DIR"
