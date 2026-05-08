#!/bin/bash
# Ordered Best-of-N sweep on widowx_carrot_on_plate.
# All 6 runs use:
#   BON_REPLAN_EVERY_N_STEPS=4   (replan after every 4 executed actions)
#   BON_SCORE_NUM_ACTIONS=4      (verifier scores+averages first 4 actions per candidate)
# So the verifier judges exactly the actions you'll execute before the next decision.
#
# Order of runs:
#    1. U-Net k=2
#    2. U-Net k=4
#    3. U-Net k=8
#    4. U-Net k=16
#    5. U-Net k=32
#    6. MLP   k=2
#    7. MLP   k=4
#    8. MLP   k=8
#    9. MLP   k=16
#   10. MLP   k=32
# (then alternating for any k > 32, e.g. k=64)
#
# Usage
# -----
#   bash scriptsv2/run_bon_ordered_sweep.sh [<unet_ckpt> <mlp_ckpt>] [num_episodes]
#
# Examples
# --------
#   bash scriptsv2/run_bon_ordered_sweep.sh                # default ckpts, 100 eps
#   bash scriptsv2/run_bon_ordered_sweep.sh 5              # smoke test, 5 eps each
#
# Task shortcut:
#   TASK=eggplant    use widowx_put_eggplant_in_basket + eggplant checkpoints (default)
#   TASK=carrot      use widowx_carrot_on_plate + carrot checkpoints
#
# Skip filter:
#   ONLY_K="15"      run only k=15 rows
#   ONLY_K="15 5"    run only k=15 and k=5 rows
#
# Other env vars forwarded to run_eval_diffusion_policy.sh:
#   DEVICE / START_SEED / MAX_STEPS / NUM_INFERENCE_STEPS / CONDA_ENV
#   TASK / REWARD_SERVER_PORT / REWARD_BATCH_SIZE / SAVE_VIDEOS / VIDEO_FPS

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

ONLY_K="${ONLY_K:-}"

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
echo "  num_episodes : $NUM_EPISODES"
echo "  replan_every_n_steps=${REPLAN_N}  score_num_actions=${SCORE_N}"
echo "  only_k       : ${ONLY_K:-(all)}"
echo "  U-Net ckpt   : $UNET_CKPT"
echo "  MLP ckpt     : $MLP_CKPT"
echo "  reward_port  : ${REWARD_SERVER_PORT:-3100}"
echo "  reward_batch : ${REWARD_BATCH_SIZE:-8} (rows per verifier GPU call; raise for speed, lower if OOM)"
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
            echo "[run_bon_ordered_sweep] skipping ${label} (only_k=${ONLY_K})"
            continue
        fi
    fi

    if [[ "$which" == "UNET" ]]; then
        ckpt="$UNET_CKPT"
        run_name="$UNET_NAME"
    else
        ckpt="$MLP_CKPT"
        run_name="$MLP_NAME"
    fi

    out_dir="data/eval/bon/${run_name}/replan${REPLAN_N}_k${k}_score${SCORE_N}"
    log_file="$out_dir/eval_log.json"

    # Resume support: if eval_log.json already exists, include it in the
    # summary but skip the actual run. Set FORCE=1 to overwrite.
    if [[ -f "$log_file" && "${FORCE:-0}" != "1" ]]; then
        echo "[run_bon_ordered_sweep] ${label} already complete -> ${log_file}"
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
        bash "$dir_path/run_eval_diffusion_policy.sh" \
            "$ckpt" "$NUM_EPISODES" "$out_dir"

    SUMMARY_FILES+=("$log_file")
    SUMMARY_LABELS+=("$label")
    echo
done

if (( ${#SUMMARY_FILES[@]} == 0 )); then
    echo "[run_bon_ordered_sweep] no runs executed."
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
python "$dir_path/summarize_evals.py" \
    --labels "$LABEL_CSV" \
    "${SUMMARY_FILES[@]}" \
    | tee "$SUMMARY_PATH"

echo
echo "[run_bon_ordered_sweep] summary saved to: $SUMMARY_PATH"
