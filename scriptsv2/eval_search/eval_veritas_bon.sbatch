#!/bin/bash
#SBATCH --job-name=veritas_bon
#SBATCH --account=ckpt-robotics
#SBATCH --partition=ckpt
#SBATCH --qos=ckpt-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|l40s|l40|a100
#SBATCH --requeue
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=data/eval/veritas_bon/_logs/slurm-%x-%j.out
#SBATCH --error=data/eval/veritas_bon/_logs/slurm-%x-%j.out
# One best-of-N cell of a BC UNet under the Veritas pixel-waypoint verifier.
# Required env: CKPT, SPLIT_FILE, PLAN_DIR, CAMERA_FILE, N, OUT_DIR, TASK.
# Optional: ACTION_POS_SCALE (null | "[x,y,z]"), MODE (argmax | softmax; softmax uses
# SOFTMAX_NORM=zscore, SOFTMAX_TEMP=1), SPLIT_NAME (test), MAX_STEPS (120),
# NUM_EPISODES (all), VERITAS_ALLOW_LIVE (0; 1 = Gemini plan at each reset, cached in PLAN_DIR),
# EVAL_OBS_NOISE_LEVEL (empty = clean; else eval-only BC state noise at that level).
set -euo pipefail
# Gemini key: from the environment, else the user's key file (never passed via --export,
# which would expose it in the job's recorded environment). GEMINI_KEY_FILE picks the file
# and makes it the ONLY key the client loads (else it also loads every api_key* file).
if [ -n "${GEMINI_KEY_FILE:-}" ]; then
  export GEMINI_API_KEY="$(tr -d '[:space:]' < "$GEMINI_KEY_FILE")"
  export GEMINI_KEY_GLOB="$GEMINI_KEY_FILE"
elif [ -z "${GEMINI_API_KEY:-}" ] && [ -f "$HOME/.config/gemini/api_key" ]; then
  export GEMINI_API_KEY="$(tr -d '[:space:]' < "$HOME/.config/gemini/api_key")"
fi
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
mkdir -p data/eval/veritas_bon/_logs

export DIFFUSION_POLICY_ROOT="$PWD/diffusion_policy"
export VERIFIER_TYPE=veritas
export VERITAS_CAMERA_FILE="$(realpath "${CAMERA_FILE:?}")"
mkdir -p "${PLAN_DIR:?}"
export VERITAS_PLAN_DIR="$(realpath "$PLAN_DIR")"
SPLIT_FILE="$(realpath "${SPLIT_FILE:?}")"
export VERITAS_ACTION_POS_SCALE="${ACTION_POS_SCALE:-null}"
export VERITAS_ALLOW_LIVE="${VERITAS_ALLOW_LIVE:-0}"
# score each candidate's endpoint after the actions actually executed (upstream: the
# scored chunk IS the executed chunk); 4 = n_action_steps of the H8 BC UNets
export VERITAS_HORIZON="${VERITAS_HORIZON:-${N_ACTION_STEPS:-4}}"
# For reproducing old results only: VERITAS_PLAN_DONE_MODE=source_object_legacy.
export VERITAS_PLAN_DONE_MODE="${VERITAS_PLAN_DONE_MODE:-zero}"
export BC_SAMPLER="${BC_SAMPLER:-ddpm}"
export BC_VERIFIER_SCORE=full
export BC_BATCH_CANDIDATES="${BC_BATCH_CANDIDATES:-1}"
export N_SAMPLES="${N:?}"
export MODE="${MODE:-argmax}"            # argmax | softmax
export SOFTMAX_TEMP="${SOFTMAX_TEMP:-1.0}"
export SOFTMAX_NORM="${SOFTMAX_NORM:-zscore}"  # softmax over per-replan z-scored values
export SPLIT_FILE="${SPLIT_FILE:?}"
export SPLIT_NAME="${SPLIT_NAME:-test}"
export TASK="${TASK:?}"
export MAX_STEPS="${MAX_STEPS:-120}"
export SCORE_WINDOW=last
export CONDA_ENV="${CONDA_ENV:-simpler_env}"
export EVAL_OBS_NOISE_LEVEL="${EVAL_OBS_NOISE_LEVEL:-}"
export VERITAS_HINTS="${VERITAS_HINTS:-sim}"   # sim | gsam2 (upstream Grounded-SAM-2 object hints)
if [ "$VERITAS_HINTS" = "gsam2" ]; then
  export GROUNDED_SAM2_PATH="${GROUNDED_SAM2_PATH:-/gscratch/robotics/harine/Grounded-SAM-2}"
  # the tracking server runs as a subprocess in its own env (torch 2.12 cu130 + sam2 +
  # transformers 4.47.1, upstream's versions); simpler_env itself is unchanged
  export GROUNDED_SAM2_PYTHON="${GROUNDED_SAM2_PYTHON:-$PWD/scriptsv2/veritas/gsam2_python.sh}"
  export GDINO_MODEL_ID="${GDINO_MODEL_ID:-IDEA-Research/grounding-dino-base}"
  export HF_HOME="${HF_HOME:-/gscratch/robotics/harine/huggingface}"
fi

echo "[veritas_bon] ckpt=$CKPT n=$N mode=$MODE(norm=$SOFTMAX_NORM,T=$SOFTMAX_TEMP) scale=$VERITAS_ACTION_POS_SCALE split=$SPLIT_FILE[$SPLIT_NAME] eps=${NUM_EPISODES:-all} live=$VERITAS_ALLOW_LIVE obs_noise=${EVAL_OBS_NOISE_LEVEL:-off} hints=$VERITAS_HINTS horizon=$VERITAS_HORIZON out=$OUT_DIR"
bash scriptsv2/eval_search/eval_search.sh "${CKPT:?}" "${NUM_EPISODES:-all}" "${OUT_DIR:?}"
