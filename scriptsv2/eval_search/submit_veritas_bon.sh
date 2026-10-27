#!/usr/bin/env bash
# Best-of-N sweep of one BC checkpoint under the Veritas verifier: one job per
# (arm, N). Arms: scale_null (1:1, upstream) and scale_fit (per-axis H=8 fit).
#
#   CKPT=<latest.ckpt> SPLIT_FILE=<manifest> PLAN_DIR=<plans> \
#   bash scriptsv2/eval_search/submit_veritas_bon.sh eggplant [SUBMIT=1]
#
# Env: NS="1 2 4 8 16 32 64", ARMS="scale_null scale_fit", SCALE_FIT="[0.72,0.74,0.87]"
#      (from calibrate_distance_verifier.py --fit horizon_cumulative --frame base),
#      OUT_ROOT=data/eval/veritas_bon, NUM_EPISODES=all (cap on the split's seeds).
#      LIVE=1: plan with Gemini on each rollout's own reset frame (upstream behaviour)
#      instead of the offline PLAN_DIR cache; plans land in <cell>/plans.
#      OBS_NOISE=<level>: eval-only BC state-obs noise (eval_search --eval-obs-noise-level);
#      cells go to arm dir <arm>_obsnoise<level> so clean cells keep their paths.
#      HINTS=gsam2: upstream object hints (Gemini objects -> Grounded-SAM-2 -> Gemini bbox
#      fallback) + exact upstream prompt formatting; arm dir <arm>_gsam2. Needs LIVE=1.
# Cells whose eval_log.json exists are skipped.
set -euo pipefail
TASK_SHORT="${1:?task: eggplant|stack}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LIVE="${LIVE:-0}"
CKPT="${CKPT:?}"; SPLIT_FILE="${SPLIT_FILE:?}"
[ "$LIVE" = "1" ] || PLAN_DIR="${PLAN_DIR:?}"
NUM_EPISODES="${NUM_EPISODES:-all}"
SUBMIT="${SUBMIT:-0}"
# extra sbatch options, e.g. the ckpt scavenger partition
SBATCH_ARGS="${SBATCH_ARGS:-}"
NS="${NS:-1 2 4 8 16 32 64}"
ARMS="${ARMS:-scale_null scale_fit}"
MODES="${MODES:-argmax softmax}"   # softmax = per-replan z-scored values, T=1
SCALE_FIT="${SCALE_FIT:-}"
OUT_ROOT="${OUT_ROOT:-$REPO/data/eval/veritas_bon}"
OBS_NOISE="${OBS_NOISE:-}"
arm_sfx=""; [ -n "$OBS_NOISE" ] && arm_sfx="_obsnoise${OBS_NOISE}"
HINTS="${HINTS:-sim}"     # sim | gsam2 (upstream Grounded-SAM-2 object hints; LIVE=1 only)
[ "$HINTS" = "gsam2" ] && { [ "$LIVE" = "1" ] || { echo "HINTS=gsam2 needs LIVE=1"; exit 1; }; arm_sfx="_gsam2${arm_sfx}"; }
case "$TASK_SHORT" in
  eggplant) TASK=widowx_put_eggplant_in_basket ;;
  stack)    TASK=widowx_stack_cube ;;
  *) echo "unknown task $TASK_SHORT"; exit 1 ;;
esac
CAMERA_FILE="${CAMERA_FILE:-$REPO/diffusion_policy/diffusion_policy/config/veritas/camera_${TASK}.json}"
run="$(basename "$(dirname "$(dirname "$CKPT")")")"
step="${STEP_NAME:-$(basename "$CKPT" .ckpt)}"   # STEP_NAME labels latest.ckpt by its step
rule="$(basename "$SPLIT_FILE" .json)"
queued="$(squeue -u "$USER" -h -o %j 2>/dev/null || true)"
# every test episode needs a cached plan (eval fails fast on a miss); live cells plan at reset
missing=0
[ "$LIVE" = "1" ] || missing="$(python3 - "$SPLIT_FILE" "$PLAN_DIR" <<'PY'
import json, os, sys
m = json.load(open(sys.argv[1]))
print(sum(not os.path.exists(os.path.join(sys.argv[2], f"seed{m['episodes'][str(i)]['seed']}.json"))
          for i in m['test']))
PY
)"
if [ "$missing" != "0" ]; then
  echo "skip: $(basename "$SPLIT_FILE") has $missing test episode(s) without a plan in $PLAN_DIR"
  exit 0
fi
for arm in $ARMS; do
  case "$arm" in
    scale_null) scale=null ;;
    scale_fit)  [ -n "$SCALE_FIT" ] || { echo "scale_fit needs SCALE_FIT=[x,y,z]"; exit 1; }; scale="$SCALE_FIT" ;;
    *) echo "unknown arm $arm"; exit 1 ;;
  esac
  for mode in $MODES; do
  for n in $NS; do
    # N=1 has a single candidate: argmax and softmax are the same rollout, run it once
    if [ "$n" = "1" ] && [ "$mode" != "argmax" ]; then continue; fi
    out="$OUT_ROOT/$TASK_SHORT/$rule/$run/$step/$arm$arm_sfx/$mode/n$n"
    job="vbon_${TASK_SHORT}_${rule}_${step}_${arm}${arm_sfx}_${mode}_n${n}"
    if [ -f "$out/eval_log.json" ]; then echo "done: $out"; continue; fi
    if grep -qx "$job" <<<"$queued"; then echo "queued: $job"; continue; fi
    hrs=$(( n >= 32 ? 10 : (n >= 8 ? 6 : 3) ))
    echo "submit: $job -> $out"
    plan_dir="${PLAN_DIR:-}"; [ "$LIVE" = "1" ] && plan_dir="$out/plans"
    if [ "$SUBMIT" = "1" ]; then
      sbatch --job-name="$job" --time="${hrs}:00:00" $SBATCH_ARGS \
        --export=ALL,CKPT="$CKPT",SPLIT_FILE="$SPLIT_FILE",PLAN_DIR="$plan_dir",VERITAS_ALLOW_LIVE="$LIVE",NUM_EPISODES="$NUM_EPISODES",CAMERA_FILE="$CAMERA_FILE",N="$n",OUT_DIR="$out",TASK="$TASK",ACTION_POS_SCALE="$scale",MODE="$mode",EVAL_OBS_NOISE_LEVEL="$OBS_NOISE",VERITAS_HINTS="$HINTS" \
        "$REPO/scriptsv2/eval_search/eval_veritas_bon.sbatch"
    fi
  done
  done
done
[ "$SUBMIT" = "1" ] || echo "(dry run; SUBMIT=1 to submit)"
