#!/bin/bash
# Run policy + verifier on a training-data zarr to produce bon_q-style NPZs,
# then render the same Q-overlay PNGs.
#
#   bash scriptsv2/viz_bon/run_viz_training.sh <zarr> <ckpt> [out_dir] [num_eps] [bon_k]
#
# Defaults:
#   out_dir = data/eval/training_viz/<zarr_stem>
#   num_eps = 5
#   bon_k   = 8
#
# Env vars:
#   DEVICE                 (default: cuda:0)
#   SCORE_NUM_ACTIONS      (default: 4)
#   BRANCH_EVERY           (default: 4)
#   REWARD_SERVER_PORT     (default: 0  -> in-process verifier)
#   REWARD_BATCH_SIZE      (default: 16)
#   INSTRUCTION            (default: "put eggplant into yellow basket")
#   TASK                   (default: widowx_put_eggplant_in_basket)
#   CONDA_ENV              (default: monkey-verifier)

set -euo pipefail

ZARR="${1:-}"
CKPT="${2:-}"
OUT_DIR="${3:-}"
NUM_EPS="${4:-5}"
BON_K="${5:-8}"

if [[ -z "$ZARR" || -z "$CKPT" ]]; then
    echo "usage: bash $0 <zarr_path> <checkpoint> [out_dir] [num_eps] [bon_k]" >&2
    exit 1
fi
if [[ ! -d "$ZARR" ]]; then
    echo "ERROR: zarr not found: $ZARR" >&2
    exit 1
fi
if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: checkpoint not found: $CKPT" >&2
    exit 1
fi

if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="data/eval/training_viz/$(basename "$ZARR" .zarr)"
fi
mkdir -p "$OUT_DIR"

DEVICE="${DEVICE:-cuda:0}"
SCORE_NUM_ACTIONS="${SCORE_NUM_ACTIONS:-4}"
BRANCH_EVERY="${BRANCH_EVERY:-4}"
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-0}"
REWARD_BATCH_SIZE="${REWARD_BATCH_SIZE:-16}"
INSTRUCTION="${INSTRUCTION:-put eggplant into yellow basket}"
TASK="${TASK:-widowx_put_eggplant_in_basket}"
CONDA_ENV="${CONDA_ENV:-monkey-verifier}"

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo_root"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Make `import diffusion_policy` work without needing to pip-install it.
DP_ROOT="${DIFFUSION_POLICY_ROOT:-/mmfs1/home/harine/diffusion_policy}"
export PYTHONPATH="${DP_ROOT}:${PYTHONPATH:-}"

# SAPIEN env vars (needed only for the one-time camera-param query).
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
export DISPLAY=""

if command -v xvfb-run >/dev/null 2>&1; then
    XVFB=(xvfb-run --auto-servernum -s "-screen 0 640x480x24")
else
    XVFB=()
fi

echo "============================================================"
echo "  viz_bon training-data run"
echo "  zarr        : $ZARR"
echo "  checkpoint  : $CKPT"
echo "  out_dir     : $OUT_DIR"
echo "  num_eps     : $NUM_EPS"
echo "  bon_k       : $BON_K"
echo "  device      : $DEVICE"
echo "  reward_port : $REWARD_SERVER_PORT  (0 = in-process)"
echo "  conda env   : $CONDA_ENV"
echo "============================================================"

"${XVFB[@]}" python scriptsv2/viz_bon/viz_training_data.py \
    --zarr "$ZARR" \
    --checkpoint "$CKPT" \
    --out-dir "$OUT_DIR" \
    --task "$TASK" \
    --instruction "$INSTRUCTION" \
    --num-episodes "$NUM_EPS" \
    --bon-k "$BON_K" \
    --score-num-actions "$SCORE_NUM_ACTIONS" \
    --branch-every "$BRANCH_EVERY" \
    --device "$DEVICE" \
    --reward-server-port "$REWARD_SERVER_PORT" \
    --reward-batch-size "$REWARD_BATCH_SIZE"

# Render overlays
python scriptsv2/viz_bon/viz_action_branches.py "$OUT_DIR" \
    --out-dir "${OUT_DIR}/_plots"

echo
echo "[viz_bon] training viz -> ${OUT_DIR}/_plots"
