#!/bin/bash
# Sweep RoboMonkey verifier Best-of-N for the latest MLP and Diffusion U-Net
# diffusion_policy checkpoints on the carrot-on-plate SimplerEnv task.
#
# Sweeps:
#   * k in K_VALUES (default: 5 15 20)
#   * mode in {a, b}:
#       (a) replan every env step, score 1 leading action per candidate
#       (b) replan per chunk        , score 3 leading actions per candidate
#   * architecture in {U-Net, MLP}
#
# Usage
# -----
#   bash scriptsv2/run_bon_mlp_unet_sweep.sh [<unet_ckpt> <mlp_ckpt>] [num_episodes]
#
# Examples
# --------
#   # Default sweep: K=5,15,20 x mode={a,b} x {unet,mlp}, 100 episodes each.
#   bash scriptsv2/run_bon_mlp_unet_sweep.sh
#
#   # Quick smoke sweep with 5 episodes per cell.
#   K_VALUES="5" bash scriptsv2/run_bon_mlp_unet_sweep.sh 5
#
# Env vars forwarded to run_eval_diffusion_policy.sh:
#   DEVICE / START_SEED / MAX_STEPS / NUM_INFERENCE_STEPS / CONDA_ENV
#   TASK / REWARD_SERVER_PORT / REWARD_BATCH_SIZE / SAVE_VIDEOS / VIDEO_FPS
#
# Sweep controls:
#   K_VALUES   whitespace-separated list (default: "5 15 20")
#   MODES      subset of "a b" (default: "a b")

set -euo pipefail

DEFAULT_UNET="/home/harine/diffusion_policy/data/outputs/2026.04.28/16.22.54_train_diffusion_unet_bridge_v2_carrot_lowdim_bridge_v2_carrot_lowdim/checkpoints/latest.ckpt"
DEFAULT_MLP="/home/harine/diffusion_policy/data/outputs/2026.04.28/16.14.29_train_mlp_bridge_v2_carrot_lowdim_bridge_v2_carrot_lowdim/checkpoints/latest.ckpt"

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

read -r -a KS <<< "${K_VALUES:-5 15 20}"
read -r -a MODE_LIST <<< "${MODES:-a b}"

# mode -> (replan_every_step, score_num_actions)
declare -A MODE_REPLAN=(["a"]="1" ["b"]="0")
declare -A MODE_SCORE_N=(["a"]="1" ["b"]="3")
declare -A MODE_DESC=(
    ["a"]="replan every env step, score 1 action"
    ["b"]="replan per chunk, score 3 actions (averaged)"
)

echo "============================================================"
echo "  diffusion_policy RoboMonkey BON sweep"
echo "  task         : ${TASK:-widowx_carrot_on_plate}"
echo "  num_episodes : $NUM_EPISODES"
echo "  k values     : ${KS[*]}"
echo "  modes        : ${MODE_LIST[*]}"
echo "  U-Net ckpt   : $UNET_CKPT"
echo "  MLP ckpt     : $MLP_CKPT"
echo "  reward_port  : ${REWARD_SERVER_PORT:-3100}"
echo "============================================================"
echo
echo "Make sure the RoboMonkey verifier server is running before this sweep."
echo

SUMMARY_FILES=()
SUMMARY_LABELS=()

for mode in "${MODE_LIST[@]}"; do
    if [[ -z "${MODE_REPLAN[$mode]:-}" ]]; then
        echo "ERROR: unknown mode '$mode' (valid: a, b)" >&2
        exit 1
    fi
    replan="${MODE_REPLAN[$mode]}"
    score_n="${MODE_SCORE_N[$mode]}"
    desc="${MODE_DESC[$mode]}"

    echo "============================================================"
    echo "  >>> mode=${mode} (${desc})"
    echo "      replan_every_step=${replan}  score_num_actions=${score_n}"
    echo "============================================================"
    echo

    for k in "${KS[@]}"; do
        UNET_OUT="data/eval/bon/${UNET_NAME}/mode_${mode}/k_${k}"
        MLP_OUT="data/eval/bon/${MLP_NAME}/mode_${mode}/k_${k}"

        echo "[run_bon_sweep] >>> mode=${mode} k=${k} Diffusion U-Net (use_ema=1)"
        BON_K="$k" \
            BON_REPLAN_EVERY_STEP="$replan" \
            BON_SCORE_NUM_ACTIONS="$score_n" \
            USE_EMA=1 \
            bash "$dir_path/run_eval_diffusion_policy.sh" \
                "$UNET_CKPT" "$NUM_EPISODES" "$UNET_OUT"
        SUMMARY_FILES+=("$UNET_OUT/eval_log.json")
        SUMMARY_LABELS+=("unet_mode${mode}_k${k}")
        echo

        echo "[run_bon_sweep] >>> mode=${mode} k=${k} MLP (use_ema=0)"
        BON_K="$k" \
            BON_REPLAN_EVERY_STEP="$replan" \
            BON_SCORE_NUM_ACTIONS="$score_n" \
            USE_EMA=0 \
            bash "$dir_path/run_eval_diffusion_policy.sh" \
                "$MLP_CKPT" "$NUM_EPISODES" "$MLP_OUT"
        SUMMARY_FILES+=("$MLP_OUT/eval_log.json")
        SUMMARY_LABELS+=("mlp_mode${mode}_k${k}")
        echo
    done
done

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-simpler_env}"

LABEL_CSV="$(IFS=,; echo "${SUMMARY_LABELS[*]}")"
python "$dir_path/summarize_evals.py" \
    --labels "$LABEL_CSV" \
    "${SUMMARY_FILES[@]}"
