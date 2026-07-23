#!/bin/bash
#SBATCH --job-name=rmky_acterr
#SBATCH --account=socialrl
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/mmfs1/home/harine/RoboMonkey/data/eval/robomonkey_action_error/slurm-%j.out
#SBATCH --error=/mmfs1/home/harine/RoboMonkey/data/eval/robomonkey_action_error/slurm-%j.out
# -----------------------------------------------------------------------------
# Offline action-error (RMSE) eval for the RoboMonkey reproduction, N=2^0..2^6.
# Single H200: sglang OpenVLA policy server (mem-fraction capped) co-located with
# the eval process (in-process LLaVA verifier + TF-CPU image preprocessing).
# No SIMPLER/SAPIEN here -- we replay frames from the success-filtered val set.
#
# Submit (smoke, ~20 frames to time the pipeline):
#   sbatch --export=ALL,N_FRAMES=20 scriptsv2/robomonkey_repro/eval_action_error.sbatch
# Full run (subsample 2000 frames):
#   sbatch scriptsv2/robomonkey_repro/eval_action_error.sbatch
# -----------------------------------------------------------------------------
set -u
REPO_ROOT=/mmfs1/home/harine/RoboMonkey
SCRIPT_DIR="$REPO_ROOT/scriptsv2/robomonkey_repro"
OUT_ROOT="$REPO_ROOT/data/eval/robomonkey_action_error"
mkdir -p "$OUT_ROOT"

DATASET="${DATASET:-/gscratch/robotics/harine/data/eggplant_in_basket_val_success/val_success.zarr}"
MAX_N="${MAX_N:-64}"
N_FRAMES="${N_FRAMES:-2000}"
SEED="${SEED:-0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
ACTION_PORT=3200

export HF_HOME="${HF_HOME:-/gscratch/robotics/harine/huggingface}"
export MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/model_dir}"
export MONKEY_VERIFIER_SRC="${MONKEY_VERIFIER_SRC:-$REPO_ROOT/monkey-verifier/src}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"
# Co-located: cap sglang static mem so the in-process LLaVA verifier has room.
export VLA_MEM_FRACTION="${VLA_MEM_FRACTION:-0.55}"

echo "============================================================"
echo " RoboMonkey action-error eval (RMSE, N=2^0..2^6)"
echo "   host=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-?}  VLA_MEM_FRACTION=$VLA_MEM_FRACTION"
echo "   dataset=$DATASET"
echo "   MAX_N=$MAX_N  N_FRAMES=$N_FRAMES  SEED=$SEED  TEMP=$TEMPERATURE"
echo "============================================================"
nvidia-smi -L || true

source "$HOME/miniconda3/etc/profile.d/conda.sh"

# --- launch OpenVLA sglang policy server (sglang-vla env) -------------------
VLA_LOG="$OUT_ROOT/vla_server-${SLURM_JOB_ID:-local}.log"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" SEED="$SEED" \
    bash "$SCRIPT_DIR/run_openvla_server_h200.sh" > "$VLA_LOG" 2>&1 &
VLA_PID=$!
echo "[vla] server pid=$VLA_PID  log=$VLA_LOG"

cleanup() {
    echo "[cleanup] stopping server $VLA_PID"
    kill "$VLA_PID" 2>/dev/null
    pkill -f "openvla_server.py" 2>/dev/null
    sleep 2
}
trap cleanup EXIT INT TERM

# --- wait for health (up to 15 min: H200 FlashInfer JIT on first launch) -----
echo -n "[wait] openvla_server :$ACTION_PORT/health "
for i in $(seq 1 300); do
    if curl -sf -o /dev/null "http://127.0.0.1:$ACTION_PORT/health" 2>/dev/null; then
        echo " READY (${i}*3s)"; break
    fi
    if ! kill -0 "$VLA_PID" 2>/dev/null; then
        echo " SERVER DIED"; tail -40 "$VLA_LOG"; exit 1
    fi
    sleep 3; echo -n "."
done
curl -sf -o /dev/null "http://127.0.0.1:$ACTION_PORT/health" 2>/dev/null \
    || { echo "[wait] TIMEOUT; tail:"; tail -40 "$VLA_LOG"; exit 1; }

# --- run the eval (monkey-verifier env, in-process verifier) -----------------
conda activate monkey-verifier
export PYTHONPATH="$MONKEY_VERIFIER_SRC:${PYTHONPATH:-}"

python "$SCRIPT_DIR/eval_action_error.py" \
    --dataset "$DATASET" \
    --out-dir "$OUT_ROOT" \
    --max-n "$MAX_N" \
    --n-frames "$N_FRAMES" \
    --seed "$SEED" \
    --temperature "$TEMPERATURE" \
    --action-server-port "$ACTION_PORT"
RC=$?

echo "[done] eval rc=$RC"
exit "$RC"
