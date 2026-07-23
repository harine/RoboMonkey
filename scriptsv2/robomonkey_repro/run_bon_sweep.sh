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
# VERIFIER_TYPE: robomonkey (LLaVA token verifier, default) | dqc (in-process
# JAX Q-critic). Both score the SAME N candidates per step (augmented_samples=N)
# and argmax-select, so the two sweeps are directly comparable.
VERIFIER_TYPE="${VERIFIER_TYPE:-robomonkey}"
if [ "$VERIFIER_TYPE" = "dqc" ]; then
  EVAL_ENV="${EVAL_ENV:-simpler_env}"   # only env with both JAX (DQC) + torch (SAPIEN)
else
  EVAL_ENV="${EVAL_ENV:-monkey-verifier}"
fi
export REWARD_BATCH_SIZE="${REWARD_BATCH_SIZE:-64}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OPENVLA_DIR="$REPO_ROOT/openvla-mini"
if [ "$VERIFIER_TYPE" = "dqc" ]; then
  RUN_DIR="${OUT_ROOT:-$REPO_ROOT/data/eval/robomonkey_repro/bon_sweep_dqc}"
else
  RUN_DIR="${OUT_ROOT:-$REPO_ROOT/data/eval/robomonkey_repro/bon_sweep}"
fi
mkdir -p "$RUN_DIR"
# Action server port: overridable + exported so openvla_server.py binds it.
# Lets several BoN-cell jobs run on the SAME node without colliding on :3200.
ACTION_PORT="${ACTION_PORT:-3200}"
export ACTION_PORT

export HF_HOME="${HF_HOME:-/gscratch/robotics/harine/huggingface}"
export MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/model_dir}"
export MONKEY_VERIFIER_SRC="${MONKEY_VERIFIER_SRC:-$REPO_ROOT/monkey-verifier/src}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export DISPLAY="${DISPLAY:-}"

# --- DQC verifier config (VERIFIER_TYPE=dqc) --------------------------------
# get_vla_action routes the continuous candidates to the in-process DQC critic.
# The image-encoding LRU cache means the N candidates per replan share ONE
# ResNet encode (cheap heads only), so high N stays fast even on CPU JAX.
if [ "$VERIFIER_TYPE" = "dqc" ]; then
  export VERIFIER_TYPE
  export DQC_CKPT_DIR="${DQC_CKPT_DIR:-/gscratch/robotics/harine/dqc_checkpoint/sd000s_137913.0.20260605_163017}"
  export DQC_TEXT_EMB="${DQC_TEXT_EMB:-$REPO_ROOT/scriptsv2/q_fn_eval/task_emb.npy}"
  export DQC_DISCOUNT="${DQC_DISCOUNT:-0.98}"
  export VERIFIER_INSTRUCTION="${VERIFIER_INSTRUCTION:-put the eggplant in the basket}"
  export VERIFIER_IMAGE_KEY="${VERIFIER_IMAGE_KEY:-agentview_image}"
  export DIFFUSION_POLICY_ROOT="${DIFFUSION_POLICY_ROOT:-$REPO_ROOT/diffusion_policy}"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}"
  export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
  [ -d "$DQC_CKPT_DIR" ] || { echo "ERROR: DQC_CKPT_DIR not found: $DQC_CKPT_DIR"; exit 2; }
fi

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
echo " RoboMonkey BoN sweep (VLA policy + in-process verifier=$VERIFIER_TYPE)"
echo "   TASK=$TASK seed=$SEED init=$INIT_SAMPLES trials=$NUM_TRIALS"
echo "   N_LIST=[$N_LIST]  REWARD_BATCH_SIZE=$REWARD_BATCH_SIZE"
echo "   GPUs=${CUDA_VISIBLE_DEVICES:-<none>}  VLA=$GPU_VLA EVAL=$GPU_EVAL ${VLA_MEM_FRACTION:+mem_frac=$VLA_MEM_FRACTION}"
echo "   out: $RUN_DIR"
echo "============================================================"

VLA_PID=""
cleanup() {
  echo "[cleanup] stopping server (port ${ACTION_PORT})..."
  [ -n "$VLA_PID" ] && kill "$VLA_PID" 2>/dev/null
  # Job-scoped teardown: free ONLY this job's ACTION_PORT. Do NOT pkill
  # openvla_server.py node-wide -- co-located sibling cells run their own
  # servers on other ports, and a node-wide pkill kills them mid-sweep.
  fuser -k "${ACTION_PORT}/tcp" 2>/dev/null \
    || kill $(lsof -ti "tcp:${ACTION_PORT}" 2>/dev/null) 2>/dev/null
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

# --- sglang OpenVLA server (function so it can be restarted mid-sweep) -------
# sglang very occasionally returns a batch error ("'NoneType' object is not
# subscriptable"); rather than abandon a whole cell we restart the server and
# retry the failed block (see the loop below).
start_vla() {
  if [ -n "$VLA_PID" ]; then
    kill "$VLA_PID" 2>/dev/null
    fuser -k "${ACTION_PORT}/tcp" 2>/dev/null
    sleep 3
  fi
  echo "[vla] launching openvla_server on GPU $GPU_VLA port $ACTION_PORT (seed=$SEED)"
  CUDA_VISIBLE_DEVICES="$GPU_VLA" SEED="$SEED" \
    bash "$SCRIPT_DIR/run_openvla_server_h200.sh" \
    >> "$RUN_DIR/vla_server.log" 2>&1 &
  VLA_PID=$!
  wait_http "http://127.0.0.1:${ACTION_PORT}/health" 900 "openvla_server"
}
start_vla || { echo "[vla] server failed; tail:"; tail -40 "$RUN_DIR/vla_server.log"; exit 1; }

# --- sweep N (resumable + appendable) ---------------------------------------
# progress.json accumulates (success, episodes) per N across relaunches. On
# relaunch we skip N values already at NUM_TRIALS and, for the rest, run only
# the *remaining* episodes (seed_offset = episodes already done) and add them
# in. Raise NUM_TRIALS and relaunch to append more episodes; FORCE=1 redoes all.
PROGRESS="$RUN_DIR/progress.json"
SUMMARY="$RUN_DIR/sweep_summary.txt"
HEADER="RoboMonkey BoN sweep (VLA policy + in-process verifier=$VERIFIER_TYPE) | task=$TASK seed=$SEED init=$INIT_SAMPLES target_trials=$NUM_TRIALS reward_batch=$REWARD_BATCH_SIZE"
PROG_PY="$SCRIPT_DIR/bon_progress.py"

conda activate "$EVAL_ENV"
# Per-job working dir for ./transfer_images/reward_img.jpg etc. Distinct EVAL_CWD
# lets this sweep run concurrently with the other RoboMonkey jobs without racing
# on that shared file. Default = OPENVLA_DIR (unchanged). Imports + the eval
# script resolve via absolute OPENVLA_DIR, so any CWD works.
EVAL_CWD="${EVAL_CWD:-$OPENVLA_DIR}"
mkdir -p "$EVAL_CWD"; cd "$EVAL_CWD"
export PRISMATIC_DATA_ROOT="$OPENVLA_DIR" PYTHONPATH="$OPENVLA_DIR"
echo "[eval] CWD=$EVAL_CWD"

# Checkpoint every BLOCK_TRIALS episodes so a wall-clock timeout mid-N loses at
# most BLOCK_TRIALS episodes (important for the ~12h N=1024 point).
BLOCK_TRIALS="${BLOCK_TRIALS:-10}"

for N in $N_LIST; do
  fails=0
  while : ; do
    DONE=$(python "$PROG_PY" get_done "$PROGRESS" "$N" 2>/dev/null || echo 0)
    if [ "$DONE" -ge "$NUM_TRIALS" ]; then
      echo "[eval] N=$N has $DONE/$NUM_TRIALS episodes -> done"
      break
    fi
    # Make sure the server is up before spending a block on it.
    if ! curl -sf -o /dev/null "http://127.0.0.1:${ACTION_PORT}/health" 2>/dev/null; then
      echo "[vla] server not healthy before N=$N block -> restarting"
      start_vla || { echo "[vla] restart failed; aborting N=$N"; break; }
    fi
    REMAIN=$(( NUM_TRIALS - DONE ))
    BLK=$(( REMAIN < BLOCK_TRIALS ? REMAIN : BLOCK_TRIALS ))
    LOG="$RUN_DIR/eval_n${N}_t${DONE}-$((DONE+BLK-1)).log"
    echo "[eval] N=$N (init=$INIT_SAMPLES aug=$N) episodes $DONE..$((DONE+BLK-1)) (+$BLK) -> $LOG"
    CUDA_VISIBLE_DEVICES="$GPU_EVAL" \
    python "$OPENVLA_DIR/experiments/robot/simpler/run_simpler_eval.py" \
        --task_suite_name "$TASK" \
        --initial_samples "$INIT_SAMPLES" \
        --augmented_samples "$N" \
        --num_trials_per_task "$BLK" \
        --seed_offset "$(( ${SEED_OFFSET_BASE:-0} + DONE ))" \
        --seed "$SEED" \
        --action_server_port "$ACTION_PORT" \
        --reward_server_port 0 \
        --local_log_dir "$RUN_DIR/n${N}_logs" \
        --run_id_note "bon_n${N}_t${DONE}" \
        > "$LOG" 2>&1
    RC=$?
    if [ "$RC" -ne 0 ]; then
      fails=$(( fails + 1 ))
      echo "[eval] N=$N block FAILED (rc=$RC, fail $fails/3). tail:"; tail -20 "$LOG"
      if [ "$fails" -ge 3 ]; then
        echo "[eval] N=$N: giving up after $fails consecutive failures."; break
      fi
      echo "[eval] restarting sglang server, retrying N=$N ..."
      start_vla || { echo "[vla] restart failed; aborting N=$N"; break; }
      continue
    fi
    fails=0
    RES=$(python "$PROG_PY" update "$PROGRESS" "$N" "$LOG")
    echo "[eval] N=$N cumulative -> $RES (success episodes rate%)"
    python "$PROG_PY" summary "$PROGRESS" "$SUMMARY" "$HEADER" >/dev/null
  done
done

echo "============================================================"
echo "BoN SWEEP SUMMARY:"; python "$PROG_PY" summary "$PROGRESS" "$SUMMARY" "$HEADER"
echo "============================================================"
