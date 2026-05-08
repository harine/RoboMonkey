#!/bin/bash
# Evaluate BOTH the Diffusion U-Net and the MLP `bridge_v2_carrot_lowdim`
# checkpoints on `widowx_carrot_on_plate`, then print a combined summary.
#
# Usage
# -----
#   bash scriptsv2/run_eval_both_architectures.sh [<unet_ckpt> <mlp_ckpt>] [num_episodes]
#
# Examples
# --------
#   # Use defaults from /home/harine/diffusion_policy/data/outputs/2026.04.28
#   bash scriptsv2/run_eval_both_architectures.sh
#
#   # Custom checkpoints, 50 rollouts each
#   bash scriptsv2/run_eval_both_architectures.sh \
#       /path/to/unet/latest.ckpt \
#       /path/to/mlp/latest.ckpt \
#       50
#
# Env vars (forwarded to `run_eval_diffusion_policy.sh`):
#   DEVICE / START_SEED / MAX_STEPS / NUM_INFERENCE_STEPS / CONDA_ENV
#   SAVE_VIDEOS / VIDEO_FPS  -- e.g. SAVE_VIDEOS=5 saves MP4s for the first
#                              5 episodes of EACH architecture.

set -euo pipefail

DEFAULT_UNET="/home/harine/diffusion_policy/data/outputs/2026.04.28/16.22.54_train_diffusion_unet_bridge_v2_carrot_lowdim_bridge_v2_carrot_lowdim/checkpoints/latest.ckpt"
DEFAULT_MLP="/home/harine/diffusion_policy/data/outputs/2026.04.28/16.14.29_train_mlp_bridge_v2_carrot_lowdim_bridge_v2_carrot_lowdim/checkpoints/latest.ckpt"

# Allow `bash run_eval_both.sh 100` (single positional == num_episodes)
if [[ $# -eq 0 ]]; then
    UNET_CKPT="$DEFAULT_UNET"
    MLP_CKPT="$DEFAULT_MLP"
    NUM_EPISODES="100"
elif [[ $# -eq 1 ]]; then
    UNET_CKPT="$DEFAULT_UNET"
    MLP_CKPT="$DEFAULT_MLP"
    NUM_EPISODES="$1"
elif [[ $# -eq 2 ]]; then
    UNET_CKPT="$1"
    MLP_CKPT="$2"
    NUM_EPISODES="100"
else
    UNET_CKPT="$1"
    MLP_CKPT="$2"
    NUM_EPISODES="$3"
fi

if [[ ! -f "$UNET_CKPT" ]]; then
    echo "ERROR: U-Net checkpoint not found: $UNET_CKPT" >&2
    exit 1
fi
if [[ ! -f "$MLP_CKPT" ]]; then
    echo "ERROR: MLP checkpoint not found: $MLP_CKPT" >&2
    exit 1
fi

full_path="$(realpath "$0")"
dir_path="$(dirname "$full_path")"
repo_root="$(dirname "$dir_path")"
cd "$repo_root"

UNET_NAME="$(basename "$(dirname "$(dirname "$UNET_CKPT")")")"
MLP_NAME="$(basename "$(dirname "$(dirname "$MLP_CKPT")")")"
UNET_OUT="data/eval/${UNET_NAME}"
MLP_OUT="data/eval/${MLP_NAME}"

echo "============================================================"
echo "  diffusion_policy SimplerEnv eval — BOTH ARCHITECTURES"
echo "  num_episodes : $NUM_EPISODES"
echo "  U-Net ckpt   : $UNET_CKPT"
echo "         out   : $UNET_OUT"
echo "  MLP   ckpt   : $MLP_CKPT"
echo "         out   : $MLP_OUT"
echo "============================================================"
echo

echo "[run_eval_both] >>> Diffusion U-Net (use_ema=1)"
USE_EMA=1 bash "$dir_path/run_eval_diffusion_policy.sh" \
    "$UNET_CKPT" "$NUM_EPISODES" "$UNET_OUT"
echo
echo "[run_eval_both] >>> MLP (use_ema=0; trained without EMA)"
USE_EMA=0 bash "$dir_path/run_eval_diffusion_policy.sh" \
    "$MLP_CKPT" "$NUM_EPISODES" "$MLP_OUT"
echo

# Use the same conda env so we have a Python interpreter available.
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-simpler_env}"

python "$dir_path/summarize_evals.py" \
    --labels "diffusion_unet,mlp" \
    "$UNET_OUT/eval_log.json" \
    "$MLP_OUT/eval_log.json"
