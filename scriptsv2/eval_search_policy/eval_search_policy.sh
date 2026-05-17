#!/bin/bash
# Evaluate a SearchPolicyRoboMonkey checkpoint (state-based L2S policy
# with in-process RoboMonkey verifier) over N rollouts in SimplerEnv.
#
# Usage
# -----
#   bash scriptsv2/eval_search_policy/eval_search_policy.sh <checkpoint> [num_episodes] [output_dir]
#
# Examples
# --------
#   bash scriptsv2/eval_search_policy/eval_search_policy.sh \
#       /home/harine/diffusion_policy/data/outputs/2026.05.11/17.40.43_robomonkey_eggplant_search_state_corrupt_robomonkey_eggplant_state/checkpoints/latest.ckpt \
#       100 \
#       data/eval/eggplant_search_state_corrupt
#
# Env vars (override on cmdline):
#   DEVICE        (default: cuda:0)
#   START_SEED    (default: 1000)
#   SEEDS         (default: unset; comma-separated explicit seeds override
#                 START_SEED and num_episodes)
#   MAX_STEPS     (default: 120)
#   USE_EMA       (default: 1; set 0 to use raw model weights)
#   SAVE_VIDEOS   (default: 0; save MP4s for the first N episodes)
#   VIDEO_FPS     (default: 10)
#   VIZ_Q         (default: 0; when 1, save per-replan sampled candidate
#                 actions + verifier values + frame to
#                 <output_dir>/search_q/ep<idx>_seed<seed>.npz)
#   TASK          (default: widowx_put_eggplant_in_basket)
#   CONDA_ENV     (default: simpler_env)

set -euo pipefail

CKPT="${1:-}"
NUM_EPISODES="${2:-100}"
OUT_DIR_ARG="${3:-}"

if [[ -z "$CKPT" ]]; then
    echo "usage: bash $0 <checkpoint> [num_episodes] [output_dir]" >&2
    exit 1
fi
if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: checkpoint not found: $CKPT" >&2
    exit 1
fi

if [[ -z "$OUT_DIR_ARG" ]]; then
    RUN_DIR="$(dirname "$(dirname "$CKPT")")"
    RUN_NAME="$(basename "$RUN_DIR")"
    OUT_DIR="data/eval/${RUN_NAME}"
else
    OUT_DIR="$OUT_DIR_ARG"
fi

DEVICE="${DEVICE:-cuda:0}"
START_SEED="${START_SEED:-1000}"
SEEDS="${SEEDS:-}"
MAX_STEPS="${MAX_STEPS:-120}"
USE_EMA="${USE_EMA:-1}"
CONDA_ENV="${CONDA_ENV:-simpler_env}"
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"
VIDEO_FPS="${VIDEO_FPS:-10}"
TASK="${TASK:-widowx_put_eggplant_in_basket}"
MODE="${MODE:-argmax}"
VIZ_Q="${VIZ_Q:-0}"               # 1 = dump sampled actions + values to search_q/
REPEAT_SEED="${REPEAT_SEED:-0}"   # 1 = every episode uses START_SEED

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# SimplerEnv / SAPIEN env vars
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
export DISPLAY=""

# Loader path for bitsandbytes (CUDA-13 libnvJitLink under the conda env)
export LD_LIBRARY_PATH="${HOME}/miniconda3/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

DP_ROOT="${DIFFUSION_POLICY_ROOT:-${repo_root:-/home/harine/RoboMonkey}/diffusion_policy}"
export PYTHONPATH="${DP_ROOT}:${PYTHONPATH:-}"

full_path="$(realpath "$0")"
dir_path="$(dirname "$full_path")"
repo_root="$(cd "$dir_path/../.." && pwd)"
cd "$repo_root"

mkdir -p "$OUT_DIR"

EXTRA_FLAGS=()
if [[ "$USE_EMA" == "0" ]]; then
    EXTRA_FLAGS+=(--no-ema)
fi
if [[ "$SAVE_VIDEOS" -gt 0 ]]; then
    EXTRA_FLAGS+=(--save-videos "$SAVE_VIDEOS" --video-fps "$VIDEO_FPS")
fi
EXTRA_FLAGS+=(--task "$TASK")
EXTRA_FLAGS+=(--mode "$MODE")
if [[ "$VIZ_Q" == "1" ]]; then
    EXTRA_FLAGS+=(--viz-q)
fi
if [[ "$REPEAT_SEED" == "1" ]]; then
    EXTRA_FLAGS+=(--repeat-seed)
fi
if [[ -n "$SEEDS" ]]; then
    EXTRA_FLAGS+=(--seeds "$SEEDS")
fi

echo "============================================================"
echo "  search-policy SimplerEnv eval"
echo "  checkpoint   : $CKPT"
echo "  task         : $TASK"
echo "  num_episodes : $NUM_EPISODES"
echo "  output_dir   : $OUT_DIR"
echo "  device       : $DEVICE"
echo "  use_ema      : $USE_EMA"
if [[ -n "$SEEDS" ]]; then
    echo "  seeds        : $SEEDS  (overrides start_seed + num_episodes)"
else
    echo "  start_seed   : $START_SEED"
fi
echo "  max_steps    : $MAX_STEPS"
echo "  save_videos  : $SAVE_VIDEOS  (fps=$VIDEO_FPS)"
echo "  mode         : $MODE"
echo "============================================================"

xvfb-run --auto-servernum -s "-screen 0 640x480x24" \
    python "$dir_path/eval_search_policy.py" \
        --checkpoint "$CKPT" \
        --num-episodes "$NUM_EPISODES" \
        --start-seed "$START_SEED" \
        --max-steps "$MAX_STEPS" \
        --device "$DEVICE" \
        --output-dir "$OUT_DIR" \
        "${EXTRA_FLAGS[@]}"

echo
echo "[run_eval] done. Summary -> $OUT_DIR/eval_log.json"
if [[ -f "$OUT_DIR/eval_log.json" ]]; then
    python "$repo_root/scriptsv2/eval_diffusion/eval_summary.py" "$OUT_DIR/eval_log.json"
fi
