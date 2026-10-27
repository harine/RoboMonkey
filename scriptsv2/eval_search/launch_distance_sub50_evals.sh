#!/bin/bash
# -----------------------------------------------------------------------------
# Success-rate eval of the distance-verifier search policy, per 5k milestone.
#
# Protocol is the canonical eggplant one so this arm is directly comparable to every
# other run under data/eval/: the test-split demo seeds (SPLIT=test), DDIM
# (deterministic) sampler, argmax selection. "The same 50 test episodes" is exactly
# that seed range — a reproducible object, not a saved episode list.
#
# N_LIST="1 16" evaluates BOTH arms of the comparison that matters:
#   n=1   one sample, no selection  -> the no-search baseline
#   n=16  best-of-16 under the distance verifier -> the search arm
# (n=16 - n=1) is the success-rate analogue of the (first - min) nRMSE gain the
# trainer logs. Reporting n=16 alone cannot separate "search helps" from "this
# checkpoint is simply better", so both are always run against the same seeds.
#
# The verifier is instantiated FROM THE CHECKPOINT's own config, so no verifier
# environment variables are passed here — the calibrated action_pos_scale and value
# normalization travel inside the checkpoint. A checkpoint trained with wrong
# calibration therefore evaluates with the same wrong calibration, which is the honest
# read of that run.
#
#   bash scriptsv2/eval_search/launch_distance_sub50_evals.sh
#   DRY=1 ...  |  MILESTONES="5000 10000" ...  |  SAVE_VIDEOS=5 ...
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey

SB="${SB:-scriptsv2/eval_search/eval_search_bc.sbatch}"
RUN_NAME="${RUN_NAME:-dist_eggplant_sub50_ie4}"
OUT_ROOT="${OUT_ROOT:-/gscratch/robotics/harine/RoboMonkey_data/train/distance_search}"
CKPT_DIR="${CKPT_DIR:-${OUT_ROOT}/${RUN_NAME}/checkpoints}"
LABEL_PREFIX="${LABEL_PREFIX:-dist_sub50}"
MILESTONES="${MILESTONES:-5000 10000 15000 20000}"
N_LIST="${N_LIST:-1 16}"
NUM_EPISODES="${NUM_EPISODES:-50}"
SPLIT="${SPLIT:-test}"
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"
VIDEO_FPS="${VIDEO_FPS:-10}"
CKPT_TOL="${CKPT_TOL:-2000}"
DRY="${DRY:-0}"

ROUTES=( "robotics|gpu-a40|gpu:a40:1" "cse|gpu-l40s|gpu:l40s:1" "weirdlab|gpu-l40s|gpu:l40s:1" "weirdlab|gpu-l40|gpu:l40:1" )
MAXSTEP="$(echo $MILESTONES | tr ' ' '\n' | sort -n | tail -1)"

# checkpoint_every is 1000, so every 5k milestone lands exactly — but keep the tolerant
# match (and the latest.ckpt mapping for the final milestone) so this still works if the
# run is resumed onto a shifted cadence or ends between step_*.ckpt writes.
find_ckpt () {
  local M="$1"
  if [[ "$M" == "$MAXSTEP" && -f "$CKPT_DIR/latest.ckpt" ]]; then
    local newest
    newest=$(ls "$CKPT_DIR"/step_*.ckpt 2>/dev/null | sed -E 's/.*step_0*([0-9]+)\.ckpt/\1/' | sort -n | tail -1)
    newest=${newest:-0}
    # Only call latest.ckpt "the final milestone" once training has actually got there;
    # otherwise it is a mid-training checkpoint and would be mislabeled.
    if (( newest >= MAXSTEP - CKPT_TOL )); then echo "$CKPT_DIR/latest.ckpt"; fi
    return 0
  fi
  ls "$CKPT_DIR"/step_*.ckpt 2>/dev/null | awk -v M="$M" -v TOL="$CKPT_TOL" '
    function abs(x){return x<0?-x:x}
    { n=$0; sub(/.*step_0*/,"",n); sub(/\.ckpt/,"",n); n+=0; d=abs(n-M);
      if (d<=TOL && (b==""||d<bd)){b=$0; bd=d} }
    END{ if(b!="") print b }'
}

# A cell counts as done only when EVERY n in N_LIST has NUM_EPISODES episodes. One job
# writes them all, so a partially-written cell must re-submit rather than be skipped.
cell_done () {
  local label="$1"
  for n in $N_LIST; do
    local f="data/eval/search_bc_sweep/${label}/argmax_ddim_n${n}_${SPLIT}_ep${NUM_EPISODES}/eval_log.json"
    python3 - "$f" "$NUM_EPISODES" <<'PY' 2>/dev/null || return 1
import json,sys
try:
    d=json.load(open(sys.argv[1])); sys.exit(0 if int(d.get("num_episodes",0))>=int(sys.argv[2]) else 1)
except Exception: sys.exit(1)
PY
  done
  return 0
}

QNAMES="$(squeue -u "$USER" -h -o '%j' 2>/dev/null)"
i=0; sub=0; sd=0; sq=0; noc=0
NL_TAG="$(echo $N_LIST | tr ' ' '-')"
for M in $MILESTONES; do
  tag="$((M/1000))k"
  label="${LABEL_PREFIX}_${tag}"
  ckpt="$(find_ckpt "$M")"
  if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then echo "[skip] $tag: no ckpt yet (within ±${CKPT_TOL} of $M)"; noc=$((noc+1)); continue; fi
  jn="${label}_n${NL_TAG}"
  if cell_done "$label"; then echo "[done] $tag already complete for n=$N_LIST"; sd=$((sd+1)); continue; fi
  if grep -qxF "$jn" <<< "$QNAMES"; then echo "[queued] $jn"; sq=$((sq+1)); continue; fi
  IFS='|' read -r a p gres <<< "${ROUTES[$((i % ${#ROUTES[@]}))]}"; i=$((i+1))
  if [[ "$DRY" == "1" ]]; then echo "WOULD submit $jn ($ckpt, $a/$p)"; sub=$((sub+1)); continue; fi
  jid=$(sbatch --parsable \
        --job-name="$jn" --account="$a" --partition="$p" --gres="$gres" \
        --cpus-per-task=2 --mem=16G --time=06:00:00 \
        --export=ALL,CHECKPOINT="$ckpt",N_LIST="$N_LIST",NUM_EPISODES="$NUM_EPISODES",SPLIT="$SPLIT",NUM_WORKERS=1,MODE=argmax,BC_SAMPLER=ddim,BC_LABEL="$label",VIZ_Q=0,SAVE_VIDEOS="$SAVE_VIDEOS",VIDEO_FPS="$VIDEO_FPS" \
        "$SB")
  echo "submitted $jn -> $jid  ($a/$p, $ckpt)"; sub=$((sub+1))
  sleep 2   # stagger: the eval driver's date-stamped WORK_DIR must differ per cell
done
echo "=== ${LABEL_PREFIX} evals: submitted=$sub skip_done=$sd skip_queued=$sq no_ckpt=$noc ==="
echo "results -> data/eval/search_bc_sweep/${LABEL_PREFIX}_<tag>/argmax_ddim_n<N>_${SPLIT}_ep${NUM_EPISODES}/eval_log.json"
