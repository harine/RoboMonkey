#!/bin/bash
# =============================================================================
# Reproduce the original RoboMonkey result: OpenVLA-7B as the *policy* (served
# by sglang) + the LLaVA-7B action *verifier*, Best-of-N over
# initial_samples + augmented_samples, evaluated in SIMPLER.
#
# Two verifier backends, selected by MODE:
#   MODE=http       -> the README path: a separate infer_server.py HTTP verifier
#                      (reward_server_port=3100).
#   MODE=inprocess  -> "the in-process verifier we are using": the eval process
#                      loads RobotRewardModel directly via VerifierClient and
#                      scores all candidates in one batched GPU forward
#                      (reward_server_port=0). No HTTP, no disk image hop.
#
# The eval itself runs in the `monkey-verifier` conda env in BOTH modes, so the
# only thing that differs between the two runs is the verifier backend -> the
# results are directly comparable. The OpenVLA policy is always served by the
# sglang `openvla_server.py` (sglang-vla env) and is re-seeded per eval seed so
# the 3 seed columns match the README's Seed 1/2/3.
#
# Usage (direct or via the sibling .sbatch wrappers):
#   MODE=inprocess bash scriptsv2/robomonkey_repro/run_robomonkey_eval.sh
#
# Env knobs (all optional, defaults reproduce the README eggplant row):
#   MODE            http | inprocess        (default inprocess)
#   TASK            simpler_put_eggplant_in_basket (default)
#   SEEDS           "1 2 3"                 (default; one VLA-server restart each)
#   INIT_SAMPLES    9                       (initial policy samples)
#   AUG_SAMPLES     32                      (Gaussian-augmented samples)
#   NUM_TRIALS      50                      (rollouts per task)
#   OUT_ROOT        data/eval/robomonkey_repro
# =============================================================================
set -u

MODE="${MODE:-inprocess}"
TASK="${TASK:-simpler_put_eggplant_in_basket}"
SEEDS="${SEEDS:-1 2 3}"
INIT_SAMPLES="${INIT_SAMPLES:-9}"
AUG_SAMPLES="${AUG_SAMPLES:-32}"
NUM_TRIALS="${NUM_TRIALS:-50}"
VLA_ENV="${VLA_ENV:-sglang-vla}"            # sglang OpenVLA policy server
VERIFIER_ENV="${VERIFIER_ENV:-monkey-verifier}"  # HTTP infer_server.py
# Eval env: README runs run_simpler_eval in `simpler_env`. The in-process
# verifier needs transformers 4.31, which only exists in `monkey-verifier`, so
# the in-process eval runs there instead (the LLaVA verifier loads in-process).
if [ "${MODE}" = "inprocess" ]; then
  EVAL_ENV="${EVAL_ENV:-monkey-verifier}"
else
  EVAL_ENV="${EVAL_ENV:-simpler_env}"
fi

# --- locate repo root -------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OPENVLA_DIR="$REPO_ROOT/openvla-mini"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/data/eval/robomonkey_repro}"
RUN_DIR="$OUT_ROOT/$MODE"
mkdir -p "$RUN_DIR"

case "$MODE" in
  http)      REWARD_PORT=3100
             # On H200 score all candidates per step in one HTTP batch (vs the
             # upstream 4090 default of 2). Result-identical, ~20x fewer calls.
             export REWARD_BATCH_SIZE="${REWARD_BATCH_SIZE:-64}" ;;
  inprocess) REWARD_PORT=0 ;;
  *) echo "ERROR: MODE must be 'http' or 'inprocess', got '$MODE'"; exit 2 ;;
esac
ACTION_PORT=3200

export HF_HOME="${HF_HOME:-/gscratch/robotics/harine/huggingface}"
export MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/model_dir}"
export MONKEY_VERIFIER_SRC="${MONKEY_VERIFIER_SRC:-$REPO_ROOT/monkey-verifier/src}"
# loopback must bypass any cluster HTTP proxy for the local health checks
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"
# SimplerEnv / SAPIEN headless rendering (matches scriptsv2/eval_search/eval_search.sh,
# the proven config on this cluster). SAPIEN auto-discovers the NVIDIA Vulkan ICD.
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export DISPLAY="${DISPLAY:-}"

# --- GPU assignment ---------------------------------------------------------
# Use whatever SLURM gave us (CUDA_VISIBLE_DEVICES is the allocated set). We
# pin each child by selecting individual entries from that list.
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
GPU_VLA="${GPUS[0]}"
if [ "$MODE" = "http" ]; then
  GPU_VERIFIER="${GPUS[1]:-${GPUS[0]}}"
  GPU_EVAL="${GPUS[2]:-${GPUS[1]:-${GPUS[0]}}}"
else
  GPU_EVAL="${GPUS[1]:-${GPUS[0]}}"
fi

# If the eval shares a GPU with the sglang server (single-GPU co-location),
# cap the server's static memory fraction so the in-process LLaVA verifier +
# SAPIEN have room on the same card.
if [ "$GPU_EVAL" = "$GPU_VLA" ]; then
  export VLA_MEM_FRACTION="${VLA_MEM_FRACTION:-0.5}"
  echo "   co-located on GPU $GPU_VLA -> VLA_MEM_FRACTION=$VLA_MEM_FRACTION"
fi

source "$HOME/miniconda3/etc/profile.d/conda.sh"

echo "============================================================"
echo " RoboMonkey reproduction"
echo "   MODE=$MODE  reward_port=$REWARD_PORT  action_port=$ACTION_PORT"
echo "   TASK=$TASK  seeds=[$SEEDS]  init=$INIT_SAMPLES aug=$AUG_SAMPLES trials=$NUM_TRIALS"
echo "   envs: vla=$VLA_ENV  eval=$EVAL_ENV ${MODE:+verifier=$VERIFIER_ENV}"
echo "   GPUs visible: ${CUDA_VISIBLE_DEVICES:-<none>}"
echo "   GPU_VLA=$GPU_VLA  GPU_EVAL=$GPU_EVAL ${GPU_VERIFIER:+GPU_VERIFIER=$GPU_VERIFIER}"
echo "   out: $RUN_DIR"
echo "============================================================"

# --- process bookkeeping / teardown ----------------------------------------
VLA_PID=""
VERIFIER_PID=""
cleanup() {
  echo "[cleanup] stopping servers..."
  [ -n "$VLA_PID" ] && kill "$VLA_PID" 2>/dev/null
  [ -n "$VERIFIER_PID" ] && kill "$VERIFIER_PID" 2>/dev/null
  # nuke any lingering children bound to our ports
  pkill -f "openvla_server.py" 2>/dev/null
  [ "$MODE" = "http" ] && pkill -f "infer_server.py" 2>/dev/null
  sleep 2
}
trap cleanup EXIT INT TERM

wait_http() { # url timeout_s label
  local url="$1" timeout="$2" label="$3" t=0
  echo -n "[wait] $label ($url) "
  while ! curl -sf -o /dev/null "$url" 2>/dev/null; do
    sleep 3; t=$((t+3))
    echo -n "."
    if [ "$t" -ge "$timeout" ]; then echo " TIMEOUT after ${timeout}s"; return 1; fi
  done
  echo " READY (${t}s)"; return 0
}

start_vla_server() { # seed
  local seed="$1"
  echo "[vla] launching openvla_server on GPU $GPU_VLA (seed=$seed)"
  CUDA_VISIBLE_DEVICES="$GPU_VLA" SEED="$seed" \
    bash "$SCRIPT_DIR/run_openvla_server_h200.sh" \
    > "$RUN_DIR/vla_server_seed${seed}.log" 2>&1 &
  VLA_PID=$!
  if ! wait_http "http://127.0.0.1:${ACTION_PORT}/health" 900 "openvla_server"; then
    echo "[vla] server failed to come up; tail of log:"; tail -40 "$RUN_DIR/vla_server_seed${seed}.log"
    return 1
  fi
}
stop_vla_server() {
  [ -n "$VLA_PID" ] && kill "$VLA_PID" 2>/dev/null
  pkill -f "openvla_server.py" 2>/dev/null
  VLA_PID=""
  sleep 3
}

start_verifier_server() {
  echo "[verifier] launching infer_server (HTTP) on GPU $GPU_VERIFIER"
  ( cd "$REPO_ROOT/monkey-verifier/src" && \
    conda activate "$VERIFIER_ENV" && \
    CUDA_VISIBLE_DEVICES="$GPU_VERIFIER" MODEL_DIR="$MODEL_DIR" \
      python infer_server.py ) \
    > "$RUN_DIR/verifier_server.log" 2>&1 &
  VERIFIER_PID=$!
  if ! wait_http "http://127.0.0.1:${REWARD_PORT}/" 900 "infer_server"; then
    echo "[verifier] server failed to come up; tail of log:"; tail -40 "$RUN_DIR/verifier_server.log"
    return 1
  fi
}

# --- HTTP verifier: one long-lived server for all seeds ---------------------
if [ "$MODE" = "http" ]; then
  start_verifier_server || exit 1
fi

# --- xvfb wrapper for headless SAPIEN rendering -----------------------------
if command -v xvfb-run >/dev/null 2>&1; then
  XVFB=(xvfb-run --auto-servernum -s "-screen 0 640x480x24")
else
  XVFB=()
fi

SUMMARY="$RUN_DIR/summary.txt"
echo "RoboMonkey reproduction summary  (MODE=$MODE, task=$TASK, init=$INIT_SAMPLES, aug=$AUG_SAMPLES, trials=$NUM_TRIALS)" > "$SUMMARY"
echo "seed    success_rate    n_success/n_episodes    eval_log" >> "$SUMMARY"

# --- per-seed eval ----------------------------------------------------------
conda activate "$EVAL_ENV"
cd "$OPENVLA_DIR"
export PRISMATIC_DATA_ROOT=. PYTHONPATH=.

OVERALL_RC=0
for SEED in $SEEDS; do
  start_vla_server "$SEED" || { OVERALL_RC=1; break; }

  EVAL_LOG="$RUN_DIR/eval_seed${SEED}.log"
  echo "[eval] seed=$SEED  reward_port=$REWARD_PORT  -> $EVAL_LOG"
  CUDA_VISIBLE_DEVICES="$GPU_EVAL" \
  "${XVFB[@]}" python experiments/robot/simpler/run_simpler_eval.py \
      --task_suite_name "$TASK" \
      --initial_samples "$INIT_SAMPLES" \
      --augmented_samples "$AUG_SAMPLES" \
      --num_trials_per_task "$NUM_TRIALS" \
      --seed "$SEED" \
      --action_server_port "$ACTION_PORT" \
      --reward_server_port "$REWARD_PORT" \
      --local_log_dir "$RUN_DIR/seed${SEED}_logs" \
      --run_id_note "${MODE}_seed${SEED}" \
      > "$EVAL_LOG" 2>&1
  RC=$?

  stop_vla_server

  # parse the last reported total success rate
  SR=$(grep -aoE "Current total success rate: [0-9.]+" "$EVAL_LOG" | tail -1 | grep -oE "[0-9.]+$")
  NS=$(grep -aoE "# successes: [0-9]+ \([0-9.]+%\)" "$EVAL_LOG" | tail -1)
  if [ "$RC" -ne 0 ] || [ -z "$SR" ]; then
    echo "$SEED    FAILED(rc=$RC)    -    $EVAL_LOG" >> "$SUMMARY"
    echo "[eval] seed=$SEED FAILED (rc=$RC). tail:"; tail -30 "$EVAL_LOG"
    OVERALL_RC=1
  else
    PCT=$(python -c "print(f'{$SR*100:.1f}%')" 2>/dev/null || echo "${SR}")
    echo "$SEED    $PCT    ${NS#\# successes: }    $EVAL_LOG" >> "$SUMMARY"
    echo "[eval] seed=$SEED success_rate=$PCT"
  fi
done

echo "============================================================"
echo "SUMMARY ($MODE):"
cat "$SUMMARY"
echo "============================================================"
exit "$OVERALL_RC"
