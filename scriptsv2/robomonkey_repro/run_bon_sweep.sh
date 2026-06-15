#!/bin/bash
# =============================================================================
# RoboMonkey Best-of-N test-time scaling sweep with the VLA policy + IN-PROCESS
# verifier. For each N in {1,2,4,...,1024} we run run_simpler_eval with
#   initial_samples = INIT_SAMPLES (fixed, fits the Gaussian proposal)
#   augmented_samples = N           (candidates the verifier scores; pick best)
# so the x-axis ("number of action samples") is N, the VLA cost is constant per
# step, and N=1 is the near-baseline. Verifier runs in-process (reward port 0).
#
# One sglang OpenVLA server stays up for the whole sweep; each N is a separate
# eval invocation (the in-process verifier reloads per invocation, ~1 min).
# Everything co-locates on a single GPU (sglang mem-fraction capped).
#
# Usage: bash scriptsv2/robomonkey_repro/run_bon_sweep.sh
# Env knobs (defaults reproduce the eggplant scaling curve):
#   TASK=simpler_put_eggplant_in_basket  SEED=1  INIT_SAMPLES=9
#   NUM_TRIALS=50  N_LIST="1 2 4 8 16 32 64 128 256 512 1024"
#   REWARD_BATCH_SIZE=64   (verifier chunk; RoboMonkey knob, tuned for H200)
# =============================================================================
set -u

TASK="${TASK:-simpler_put_eggplant_in_basket}"
SEED="${SEED:-1}"
INIT_SAMPLES="${INIT_SAMPLES:-9}"
NUM_TRIALS="${NUM_TRIALS:-50}"
N_LIST="${N_LIST:-1 2 4 8 16 32 64 128 256 512 1024}"
VLA_ENV="${VLA_ENV:-sglang-vla}"
EVAL_ENV="${EVAL_ENV:-monkey-verifier}"
export REWARD_BATCH_SIZE="${REWARD_BATCH_SIZE:-64}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OPENVLA_DIR="$REPO_ROOT/openvla-mini"
RUN_DIR="${OUT_ROOT:-$REPO_ROOT/data/eval/robomonkey_repro/bon_sweep}"
mkdir -p "$RUN_DIR"
ACTION_PORT=3200

export HF_HOME="${HF_HOME:-/gscratch/robotics/harine/huggingface}"
export MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/model_dir}"
export MONKEY_VERIFIER_SRC="${MONKEY_VERIFIER_SRC:-$REPO_ROOT/monkey-verifier/src}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export DISPLAY="${DISPLAY:-}"

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
GPU_VLA="${GPUS[0]}"
GPU_EVAL="${GPUS[1]:-${GPUS[0]}}"
# co-located single-GPU run -> cap sglang static memory so the in-process
# verifier + SAPIEN have room on the same card.
if [ "$GPU_EVAL" = "$GPU_VLA" ]; then
  export VLA_MEM_FRACTION="${VLA_MEM_FRACTION:-0.5}"
fi

source "$HOME/miniconda3/etc/profile.d/conda.sh"

echo "============================================================"
echo " RoboMonkey BoN sweep (VLA policy + in-process verifier)"
echo "   TASK=$TASK seed=$SEED init=$INIT_SAMPLES trials=$NUM_TRIALS"
echo "   N_LIST=[$N_LIST]  REWARD_BATCH_SIZE=$REWARD_BATCH_SIZE"
echo "   GPUs=${CUDA_VISIBLE_DEVICES:-<none>}  VLA=$GPU_VLA EVAL=$GPU_EVAL ${VLA_MEM_FRACTION:+mem_frac=$VLA_MEM_FRACTION}"
echo "   out: $RUN_DIR"
echo "============================================================"

VLA_PID=""
cleanup() {
  echo "[cleanup] stopping server..."
  [ -n "$VLA_PID" ] && kill "$VLA_PID" 2>/dev/null
  pkill -f "openvla_server.py" 2>/dev/null
  sleep 2
}
trap cleanup EXIT INT TERM

wait_http() {
  local url="$1" timeout="$2" label="$3" t=0
  echo -n "[wait] $label "
  while ! curl -sf -o /dev/null "$url" 2>/dev/null; do
    sleep 3; t=$((t+3)); echo -n "."
    if [ "$t" -ge "$timeout" ]; then echo " TIMEOUT ${timeout}s"; return 1; fi
  done
  echo " READY (${t}s)"; return 0
}

# --- start the sglang OpenVLA server once for the whole sweep ---------------
echo "[vla] launching openvla_server on GPU $GPU_VLA (seed=$SEED)"
CUDA_VISIBLE_DEVICES="$GPU_VLA" SEED="$SEED" \
  bash "$SCRIPT_DIR/run_openvla_server_h200.sh" \
  > "$RUN_DIR/vla_server.log" 2>&1 &
VLA_PID=$!
wait_http "http://127.0.0.1:${ACTION_PORT}/health" 900 "openvla_server" || {
  echo "[vla] server failed; tail:"; tail -40 "$RUN_DIR/vla_server.log"; exit 1; }

# --- sweep N (resumable + appendable) ---------------------------------------
# progress.json accumulates (success, episodes) per N across relaunches. On
# relaunch we skip N values already at NUM_TRIALS and, for the rest, run only
# the *remaining* episodes (seed_offset = episodes already done) and add them
# in. Raise NUM_TRIALS and relaunch to append more episodes; FORCE=1 redoes all.
PROGRESS="$RUN_DIR/progress.json"
SUMMARY="$RUN_DIR/sweep_summary.txt"
HEADER="RoboMonkey BoN sweep (VLA policy + in-process verifier) | task=$TASK seed=$SEED init=$INIT_SAMPLES target_trials=$NUM_TRIALS reward_batch=$REWARD_BATCH_SIZE"
PROG_PY="$SCRIPT_DIR/bon_progress.py"

conda activate "$EVAL_ENV"
cd "$OPENVLA_DIR"
export PRISMATIC_DATA_ROOT=. PYTHONPATH=.

# Checkpoint every BLOCK_TRIALS episodes so a wall-clock timeout mid-N loses at
# most BLOCK_TRIALS episodes (important for the ~12h N=1024 point).
BLOCK_TRIALS="${BLOCK_TRIALS:-10}"

for N in $N_LIST; do
  while : ; do
    DONE=$(python "$PROG_PY" get_done "$PROGRESS" "$N" 2>/dev/null || echo 0)
    if [ "$DONE" -ge "$NUM_TRIALS" ]; then
      echo "[eval] N=$N has $DONE/$NUM_TRIALS episodes -> done"
      break
    fi
    REMAIN=$(( NUM_TRIALS - DONE ))
    BLK=$(( REMAIN < BLOCK_TRIALS ? REMAIN : BLOCK_TRIALS ))
    LOG="$RUN_DIR/eval_n${N}_t${DONE}-$((DONE+BLK-1)).log"
    echo "[eval] N=$N (init=$INIT_SAMPLES aug=$N) episodes $DONE..$((DONE+BLK-1)) (+$BLK) -> $LOG"
    CUDA_VISIBLE_DEVICES="$GPU_EVAL" \
    python experiments/robot/simpler/run_simpler_eval.py \
        --task_suite_name "$TASK" \
        --initial_samples "$INIT_SAMPLES" \
        --augmented_samples "$N" \
        --num_trials_per_task "$BLK" \
        --seed_offset "$DONE" \
        --seed "$SEED" \
        --action_server_port "$ACTION_PORT" \
        --reward_server_port 0 \
        --local_log_dir "$RUN_DIR/n${N}_logs" \
        --run_id_note "bon_n${N}_t${DONE}" \
        > "$LOG" 2>&1
    RC=$?
    if [ "$RC" -ne 0 ]; then
      echo "[eval] N=$N block FAILED (rc=$RC). tail:"; tail -25 "$LOG"
      break
    fi
    RES=$(python "$PROG_PY" update "$PROGRESS" "$N" "$LOG")
    echo "[eval] N=$N cumulative -> $RES (success episodes rate%)"
    python "$PROG_PY" summary "$PROGRESS" "$SUMMARY" "$HEADER" >/dev/null
  done
done

echo "============================================================"
echo "BoN SWEEP SUMMARY:"; python "$PROG_PY" summary "$PROGRESS" "$SUMMARY" "$HEADER"
echo "============================================================"
