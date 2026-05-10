#!/bin/bash
# Ordered Best-of-N sweep on widowx_carrot_on_plate.
# All 6 runs use:
#   BON_REPLAN_EVERY_N_STEPS=4   (replan after every 4 executed actions)
#   BON_SCORE_NUM_ACTIONS=4      (verifier scores+averages first 4 actions per candidate)
# So the verifier judges exactly the actions you'll execute before the next decision.
#
# Order of runs (U-Net only by default; set SKIP_MLP=0 to also include the
# MLP rows interleaved at the end):
#    1. U-Net k=2
#    2. U-Net k=4
#    3. U-Net k=8
#    4. U-Net k=16
#    5. U-Net k=32
#    6. U-Net k=64
#  (with SKIP_MLP=0, MLP k=2..64 are appended after the U-Net rows)
#
# Usage
# -----
#   bash scriptsv2/bon_eval/bon_eval.sh [<unet_ckpt>] [num_episodes]
#   SKIP_MLP=0 bash scriptsv2/bon_eval/bon_eval.sh <unet_ckpt> <mlp_ckpt> [num_episodes]
#
# Examples
# --------
#   # Default U-Net-only sweep on the eggplant default ckpt, 100 eps each:
#   bash scriptsv2/bon_eval/bon_eval.sh
#
#   # Smoke test (5 eps per cell):
#   bash scriptsv2/bon_eval/bon_eval.sh 5
#
#   # Sweep one custom U-Net checkpoint:
#   bash scriptsv2/bon_eval/bon_eval.sh /path/to/checkpoints/latest.ckpt
#
#   # Custom seeds + save Q-values for branching viz:
#   SEEDS="17,50,3,9" VIZ_Q=1 \
#       bash scriptsv2/bon_eval/bon_eval.sh /path/to/checkpoints/latest.ckpt
#
#   # Include MLP rows too (legacy mode):
#   SKIP_MLP=0 bash scriptsv2/bon_eval/bon_eval.sh \
#       /path/to/unet/checkpoints/latest.ckpt \
#       /path/to/mlp/checkpoints/latest.ckpt
#
# Task shortcut:
#   TASK=eggplant    use widowx_put_eggplant_in_basket + eggplant checkpoints (default)
#   TASK=carrot      use widowx_carrot_on_plate + carrot checkpoints
#
# Skip filter:
#   ONLY_K="15"      run only k=15 rows
#   ONLY_K="15 5"    run only k=15 and k=5 rows
#   SKIP_MLP=1       (default) skip every "mlp_*" row
#   SKIP_MLP=0       include MLP rows; requires <mlp_ckpt> argument
#
# Custom seeds:
#   SEEDS="17,50,3,9"  run one episode per listed seed instead of
#                      START_SEED..START_SEED+N. Output dirs are namespaced
#                      with `_seeds<tag>` so they don't collide with default
#                      sweeps.
#
# Save Q-values for visualizing BoN branching:
#   VIZ_Q=1            for every BoN row, write per-replan candidate actions
#                      and verifier rewards to
#                      <out_dir>/bon_q/ep<idx>_seed<seed>.npz
#
# Other env vars forwarded to eval_diffusion.sh:
#   DEVICE / START_SEED / MAX_STEPS / NUM_INFERENCE_STEPS / CONDA_ENV
#   TASK / REWARD_SERVER_PORT / REWARD_BATCH_SIZE / SAVE_VIDEOS / VIDEO_FPS
#   SEEDS / VIZ_Q

set -euo pipefail

# Task shortcuts: set TASK=eggplant to use the eggplant task + checkpoints.
# Full task names also work: TASK=widowx_put_eggplant_in_basket
_TASK_SHORT="${TASK:-eggplant}"
if [[ "$_TASK_SHORT" == "carrot" || "$_TASK_SHORT" == "widowx_carrot_on_plate" ]]; then
    export TASK="widowx_carrot_on_plate"
    DEFAULT_UNET="/home/harine/diffusion_policy/data/outputs/2026.04.28/16.22.54_train_diffusion_unet_bridge_v2_carrot_lowdim_bridge_v2_carrot_lowdim/checkpoints/latest.ckpt"
    DEFAULT_MLP="/home/harine/diffusion_policy/data/outputs/2026.04.28/16.14.29_train_mlp_bridge_v2_carrot_lowdim_bridge_v2_carrot_lowdim/checkpoints/latest.ckpt"
else
    export TASK="widowx_put_eggplant_in_basket"
    DEFAULT_UNET="/home/harine/diffusion_policy/data/outputs/2026.04.29/16.45.01_train_diffusion_unet_eggplant_in_basket_lowdim_eggplant_in_basket_lowdim/checkpoints/latest.ckpt"
    DEFAULT_MLP="/home/harine/diffusion_policy/data/outputs/2026.04.29/16.40.54_train_mlp_eggplant_in_basket_lowdim_eggplant_in_basket_lowdim/checkpoints/latest.ckpt"
fi

SKIP_MLP="${SKIP_MLP:-1}"

# Arg parsing.
#   With SKIP_MLP=1, we accept (in order): [unet_ckpt] [num_episodes].
#   Otherwise we accept (in order): [unet_ckpt mlp_ckpt] [num_episodes].
if [[ "$SKIP_MLP" == "1" ]]; then
    if [[ $# -eq 0 ]]; then
        UNET_CKPT="$DEFAULT_UNET"
        NUM_EPISODES="100"
    elif [[ $# -eq 1 ]]; then
        # If $1 is a file path, treat as checkpoint; else treat as num_episodes.
        if [[ -f "$1" ]]; then
            UNET_CKPT="$1"
            NUM_EPISODES="100"
        else
            UNET_CKPT="$DEFAULT_UNET"
            NUM_EPISODES="$1"
        fi
    else
        UNET_CKPT="$1"
        NUM_EPISODES="$2"
    fi
    MLP_CKPT=""
else
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
        NUM_EPISODES="50"
    else
        UNET_CKPT="$1"
        MLP_CKPT="$2"
        NUM_EPISODES="$3"
    fi
fi

if [[ ! -f "$UNET_CKPT" ]]; then
    echo "ERROR: U-Net checkpoint not found: $UNET_CKPT" >&2
    exit 1
fi
if [[ "$SKIP_MLP" != "1" && ! -f "$MLP_CKPT" ]]; then
    echo "ERROR: MLP checkpoint not found: $MLP_CKPT" >&2
    echo "  (set SKIP_MLP=1 to sweep U-Net only)" >&2
    exit 1
fi

full_path="$(realpath "$0")"
dir_path="$(dirname "$full_path")"
repo_root="$(cd "$dir_path/../.." && pwd)"
EVAL_DIR="$repo_root/scriptsv2/eval_diffusion"
cd "$repo_root"

UNET_NAME="$(basename "$(dirname "$(dirname "$UNET_CKPT")")")"
if [[ "$SKIP_MLP" == "1" ]]; then
    MLP_NAME=""
else
    MLP_NAME="$(basename "$(dirname "$(dirname "$MLP_CKPT")")")"
fi

ONLY_K="${ONLY_K:-}"
SEEDS="${SEEDS:-}"
VIZ_Q="${VIZ_Q:-0}"

# Namespace output dirs when running an explicit seed list, so a custom-seed
# sweep doesn't overwrite (or get short-circuited by) prior default-seed runs.
SEED_TAG=""
if [[ -n "$SEEDS" ]]; then
    SEED_TAG="_seeds${SEEDS//,/_}"
fi

# Each row: label | which (UNET|MLP) | use_ema | k
# Order: all U-Net (k=2→32), then all MLP (k=2→32), then alternating for k=64+
RUNS=(
    "unet_k2 |UNET|1|2"
    "unet_k4 |UNET|1|4"
    "unet_k8 |UNET|1|8"
    "unet_k16|UNET|1|16"
    "unet_k32|UNET|1|32"
    "mlp_k2  |MLP |0|2"
    "mlp_k4  |MLP |0|4"
    "mlp_k8  |MLP |0|8"
    "mlp_k16 |MLP |0|16"
    "mlp_k32 |MLP |0|32"
    "unet_k64|UNET|1|64"
    "mlp_k64 |MLP |0|64"
)

REPLAN_N=4
SCORE_N=4

echo "============================================================"
echo "  diffusion_policy RoboMonkey BON ordered sweep"
echo "  task         : ${TASK:-widowx_carrot_on_plate}"
if [[ -n "$SEEDS" ]]; then
    echo "  seeds        : $SEEDS  (overrides num_episodes)"
else
    echo "  num_episodes : $NUM_EPISODES"
fi
echo "  replan_every_n_steps=${REPLAN_N}  score_num_actions=${SCORE_N}"
echo "  only_k       : ${ONLY_K:-(all)}"
echo "  skip_mlp     : $SKIP_MLP"
echo "  viz_q        : $VIZ_Q"
echo "  U-Net ckpt   : $UNET_CKPT"
if [[ "$SKIP_MLP" != "1" ]]; then
    echo "  MLP ckpt     : $MLP_CKPT"
fi
echo "  reward_port  : ${REWARD_SERVER_PORT:-3100}"
echo "  reward_batch : ${REWARD_BATCH_SIZE:-16} (rows per verifier GPU call; raise for speed, lower if OOM)"
echo "============================================================"
echo
echo "Make sure the RoboMonkey verifier server is running with a clean"
echo "(uncorrupted) CUDA context. If you've seen 'device-side assert'"
echo "errors recently, restart the verifier first."
echo

SUMMARY_FILES=()
SUMMARY_LABELS=()

for row in "${RUNS[@]}"; do
    IFS='|' read -r label which use_ema k <<< "$row"
    label="${label// /}"
    k="${k// /}"

    if [[ -n "$ONLY_K" ]]; then
        match=0
        for kf in $ONLY_K; do
            if [[ "$k" == "$kf" ]]; then match=1; break; fi
        done
        if [[ "$match" == "0" ]]; then
            echo "[bon_eval] skipping ${label} (only_k=${ONLY_K})"
            continue
        fi
    fi

    if [[ "$which" == "MLP" && "$SKIP_MLP" == "1" ]]; then
        echo "[bon_eval] skipping ${label} (SKIP_MLP=1)"
        continue
    fi

    if [[ "$which" == "UNET" ]]; then
        ckpt="$UNET_CKPT"
        run_name="$UNET_NAME"
    else
        ckpt="$MLP_CKPT"
        run_name="$MLP_NAME"
    fi

    out_dir="data/eval/bon/${run_name}/replan${REPLAN_N}_k${k}_score${SCORE_N}${SEED_TAG}"
    log_file="$out_dir/eval_log.json"

    # Resume support: if eval_log.json already exists, include it in the
    # summary but skip the actual run. Set FORCE=1 to overwrite.
    if [[ -f "$log_file" && "${FORCE:-0}" != "1" ]]; then
        echo "[bon_eval] ${label} already complete -> ${log_file}"
        echo "  (set FORCE=1 to re-run)"
        SUMMARY_FILES+=("$log_file")
        SUMMARY_LABELS+=("$label")
        echo
        continue
    fi

    echo "============================================================"
    echo "  >>> ${label}"
    echo "      k=${k}  replan_every_n_steps=${REPLAN_N}  score_num_actions=${SCORE_N}"
    echo "      use_ema=${use_ema}"
    echo "      ckpt=${ckpt}"
    echo "      out=${out_dir}"
    echo "============================================================"
    echo

    BON_K="$k" \
        BON_REPLAN_EVERY_N_STEPS="$REPLAN_N" \
        BON_SCORE_NUM_ACTIONS="$SCORE_N" \
        USE_EMA="$use_ema" \
        SEEDS="$SEEDS" \
        VIZ_Q="$VIZ_Q" \
        bash "$EVAL_DIR/eval_diffusion.sh" \
            "$ckpt" "$NUM_EPISODES" "$out_dir"

    SUMMARY_FILES+=("$log_file")
    SUMMARY_LABELS+=("$label")
    echo
done

if (( ${#SUMMARY_FILES[@]} == 0 )); then
    echo "[bon_eval] no runs executed."
    exit 0
fi

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-simpler_env}"

LABEL_CSV="$(IFS=,; echo "${SUMMARY_LABELS[*]}")"
SUMMARY_DIR="data/eval/bon/_summaries"
mkdir -p "$SUMMARY_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
SUMMARY_PATH="${SUMMARY_DIR}/summary_${TASK}_${TS}.txt"

# Print to stdout AND save to file via tee.
python "$EVAL_DIR/eval_summary.py" \
    --labels "$LABEL_CSV" \
    "${SUMMARY_FILES[@]}" \
    | tee "$SUMMARY_PATH"

echo
echo "[bon_eval] summary saved to: $SUMMARY_PATH"
