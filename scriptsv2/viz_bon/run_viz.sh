#!/bin/bash
# Augment + visualize bon_q .npz files for a given cell.
#
#   bash scriptsv2/viz_bon/run_viz.sh <bon_q_dir> [out_dir] [limit_files] [every]
#
# Defaults:
#   out_dir      = data/eval/bon/_summaries/bon_viz/<basename of parent of bon_q dir>
#   limit_files  = 5
#   every        = 1   (plot every replan branch)
#
# Env vars:
#   TASK         = widowx_put_eggplant_in_basket
#   CONDA_ENV    = simpler_env
#   VIDEO        = 1     (write one MP4 per episode instead of per-branch PNGs)
#   FPS          = 2     (fps for --video; ignored when VIDEO=0)

set -euo pipefail

BON_Q_DIR="${1:-}"
if [[ -z "$BON_Q_DIR" ]]; then
    echo "usage: bash $0 <bon_q_dir> [out_dir] [limit_files] [every]" >&2
    exit 1
fi
OUT_DIR="${2:-}"
LIMIT="${3:-5}"
EVERY="${4:-1}"
TASK="${TASK:-widowx_put_eggplant_in_basket}"
CONDA_ENV="${CONDA_ENV:-simpler_env}"
VIDEO="${VIDEO:-1}"
FPS="${FPS:-2}"

if [[ -z "$OUT_DIR" ]]; then
    CELL_DIR="$(realpath "$BON_Q_DIR/..")"
    OUT_DIR="data/eval/bon/_summaries/bon_viz/$(basename "$CELL_DIR")"
fi
mkdir -p "$OUT_DIR"

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo_root"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# SimplerEnv / SAPIEN env vars
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
export DISPLAY=""

if command -v xvfb-run >/dev/null 2>&1; then
    XVFB=(xvfb-run --auto-servernum -s "-screen 0 640x480x24")
else
    XVFB=()
fi

echo "============================================================"
echo "  viz_bon run"
echo "  bon_q_dir   : $BON_Q_DIR"
echo "  out_dir     : $OUT_DIR"
echo "  limit_files : $LIMIT"
echo "  every       : $EVERY"
echo "  task        : $TASK"
echo "  conda env   : $CONDA_ENV"
echo "  video       : $VIDEO  (fps=$FPS)"
echo "============================================================"

# Step 1: augment npz files in-place (under <BON_Q_DIR>) with EE pose + cam.
"${XVFB[@]}" python scriptsv2/viz_bon/augment_bon_q.py "$BON_Q_DIR" \
    --task "$TASK" \
    --limit "$LIMIT"

# Step 2: render. By default writes one MP4 per episode (no PNGs). Set
# VIDEO=0 to write per-branch PNGs instead.
VIDEO_FLAGS=()
if [[ "$VIDEO" != "0" ]]; then
    VIDEO_FLAGS=(--video --fps "$FPS")
fi
python scriptsv2/viz_bon/viz_action_branches.py "$BON_Q_DIR" \
    --out-dir "$OUT_DIR" \
    --limit-files "$LIMIT" \
    --every "$EVERY" \
    "${VIDEO_FLAGS[@]}"

echo
echo "[viz_bon] done -> $OUT_DIR"
