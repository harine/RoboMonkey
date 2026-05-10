#!/bin/bash
# Evaluate BOTH the Diffusion U-Net and the MLP checkpoint for a task, then
# print a side-by-side summary. Replaces the per-task run_eval_both_* scripts.
#
# Tasks supported (pick via TASK env var, short name accepted):
#   * eggplant (default) -> widowx_put_eggplant_in_basket
#   * carrot             -> widowx_carrot_on_plate
#
# Usage
# -----
#   bash scriptsv2/eval_diffusion/eval_diffusion_both.sh                    # eggplant defaults, 100 eps
#   TASK=carrot bash scriptsv2/eval_diffusion/eval_diffusion_both.sh        # carrot defaults
#   bash scriptsv2/eval_diffusion/eval_diffusion_both.sh 50                 # 50 eps
#   bash scriptsv2/eval_diffusion/eval_diffusion_both.sh <unet> <mlp> [N]   # custom ckpts
#
# Env vars (forwarded to eval_diffusion.sh):
#   DEVICE / START_SEED / MAX_STEPS / NUM_INFERENCE_STEPS / CONDA_ENV
#   SAVE_VIDEOS / VIDEO_FPS

set -euo pipefail

_TASK_SHORT="${TASK:-eggplant}"
case "$_TASK_SHORT" in
    carrot|widowx_carrot_on_plate)
        export TASK="widowx_carrot_on_plate"
        DEFAULT_UNET="/home/harine/diffusion_policy/data/outputs/2026.04.28/16.22.54_train_diffusion_unet_bridge_v2_carrot_lowdim_bridge_v2_carrot_lowdim/checkpoints/latest.ckpt"
        DEFAULT_MLP="/home/harine/diffusion_policy/data/outputs/2026.04.28/16.14.29_train_mlp_bridge_v2_carrot_lowdim_bridge_v2_carrot_lowdim/checkpoints/latest.ckpt"
        ;;
    eggplant|widowx_put_eggplant_in_basket|*)
        export TASK="widowx_put_eggplant_in_basket"
        DEFAULT_UNET="/home/harine/diffusion_policy/data/outputs/2026.04.29/16.45.01_train_diffusion_unet_eggplant_in_basket_lowdim_eggplant_in_basket_lowdim/checkpoints/latest.ckpt"
        DEFAULT_MLP="/home/harine/diffusion_policy/data/outputs/2026.04.29/16.40.54_train_mlp_eggplant_in_basket_lowdim_eggplant_in_basket_lowdim/checkpoints/latest.ckpt"
        ;;
esac

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

dir_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$dir_path/../.." && pwd)"
cd "$repo_root"

UNET_NAME="$(basename "$(dirname "$(dirname "$UNET_CKPT")")")"
MLP_NAME="$(basename "$(dirname "$(dirname "$MLP_CKPT")")")"
UNET_OUT="data/eval/${UNET_NAME}"
MLP_OUT="data/eval/${MLP_NAME}"

echo "============================================================"
echo "  diffusion_policy SimplerEnv eval — BOTH ARCHITECTURES"
echo "  task         : $TASK"
echo "  num_episodes : $NUM_EPISODES"
echo "  U-Net ckpt   : $UNET_CKPT"
echo "         out   : $UNET_OUT"
echo "  MLP   ckpt   : $MLP_CKPT"
echo "         out   : $MLP_OUT"
echo "============================================================"
echo

echo "[eval_both] >>> Diffusion U-Net (use_ema=1)"
USE_EMA=1 bash "$dir_path/eval_diffusion.sh" \
    "$UNET_CKPT" "$NUM_EPISODES" "$UNET_OUT"
echo
echo "[eval_both] >>> MLP (use_ema=0; trained without EMA)"
USE_EMA=0 bash "$dir_path/eval_diffusion.sh" \
    "$MLP_CKPT" "$NUM_EPISODES" "$MLP_OUT"
echo

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-simpler_env}"

python "$dir_path/eval_summary.py" \
    --labels "diffusion_unet,mlp" \
    "$UNET_OUT/eval_log.json" \
    "$MLP_OUT/eval_log.json"
