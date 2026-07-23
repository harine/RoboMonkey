#!/bin/bash
# -----------------------------------------------------------------------------
# n=1 success-rate eval for a 50k-step U-Net BC run (eggplant), per checkpoint.
# Evaluates the milestone checkpoints in MILESTONES of the training run
# diffusion_policy/data/train/bc_denoiser/${RUN_NAME}.
#
# n=1 => one BC sample per replan, no search/selection (the wrapped verifier
# loads but its score is irrelevant with a single candidate). Protocol matches
# the canonical BC evals: 50 episodes, seeds 1000-1049, EMA weights, DDIM
# (deterministic) sampler, MAX_STEPS=120 -> directly comparable.
#
# Each checkpoint gets a distinct BC_LABEL so cells (keyed by sampler/N/seed,
# NOT by checkpoint) never collide:
#   data/eval/search_bc_sweep/${LABEL_PREFIX}_<tag>/argmax_ddim_n1_seed1000_ep50/
#
# Idempotent: a cell already complete (>=NUM_EPISODES) or a job already queued
# for that tag is skipped; a milestone with no checkpoint yet is skipped (picked
# up on a later run). Submits are staggered 2s so eval_search_bc.sh's
# date-stamped WORK_DIR doesn't collide between concurrent cells.
#
#   bash scriptsv2/eval_search/launch_unet50k_n1_evals.sh        # original run
#   RUN_NAME=eggplant_bc_unet_h16_50k_filtered LABEL_PREFIX=unet50k_filt \
#     MILESTONES="10000 20000 30000 40000 50000" \
#     bash scriptsv2/eval_search/launch_unet50k_n1_evals.sh      # filtered run
#   DRY=1 ... bash .../launch_unet50k_n1_evals.sh                # report only
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey

SB=scriptsv2/eval_search/eval_search_bc.sbatch
RUN_NAME="${RUN_NAME:-eggplant_bc_unet_h16_50k}"
LABEL_PREFIX="${LABEL_PREFIX:-unet50k}"
CKPT_DIR="diffusion_policy/data/train/bc_denoiser/${RUN_NAME}/checkpoints"
MILESTONES="${MILESTONES:-10000 20000 30000 40000 50000}"
NUM_EPISODES="${NUM_EPISODES:-50}"
START_SEED="${START_SEED:-1000}"
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"   # >0 = save MP4s for the first N episodes
VIDEO_FPS="${VIDEO_FPS:-10}"      # videos land in <cell>/argmax_ddim_n1_.../videos/
DRY="${DRY:-0}"

# Free GPU routes. 1 GPU + lean CPU/mem per cell (n=1, NUM_WORKERS=1).
# Use any free GPU: round-robin across all my account/partition combos so a cell
# starts wherever an AssocGrpGRES cap has headroom (all 48GB cards fit LLaVA).
ROUTES=( "robotics|gpu-a40|gpu:a40:1" "cse|gpu-l40s|gpu:l40s:1" "weirdlab|gpu-l40s|gpu:l40s:1" "weirdlab|gpu-l40|gpu:l40:1" )

QNAMES="$(squeue -u "$USER" -h -o '%j')"
i=0; sub=0; sd=0; sq=0; noc=0

# Checkpoint cadence isn't always exact thousands (epoch boundaries push later
# saves to step_X001), and the final step lands in latest.ckpt (step_*.ckpt may
# stop ~1k short, e.g. 49001). So match the nearest step_*.ckpt within CKPT_TOL,
# and map the largest milestone to latest.ckpt when present.
CKPT_TOL="${CKPT_TOL:-2000}"
MAXSTEP="$(echo $MILESTONES | tr ' ' '\n' | sort -n | tail -1)"
find_ckpt () {  # $1=milestone -> echo ckpt path or empty
  local M="$1"
  # Map the largest milestone to latest.ckpt ONLY once training has actually
  # reached the end — otherwise latest.ckpt is a mid-training checkpoint and
  # would be mislabeled as the final step. Gate on the newest step_*.ckpt being
  # within CKPT_TOL of MAXSTEP (training essentially done).
  if [[ "$M" == "$MAXSTEP" && -f "$CKPT_DIR/latest.ckpt" ]]; then
    local newest
    newest=$(ls "$CKPT_DIR"/step_*.ckpt 2>/dev/null | sed -E 's/.*step_0*([0-9]+)\.ckpt/\1/' | sort -n | tail -1)
    newest=${newest:-0}
    if (( newest >= MAXSTEP - CKPT_TOL )); then
      echo "$CKPT_DIR/latest.ckpt"; return 0
    fi
    return 0  # not done yet -> no checkpoint for the top milestone
  fi
  ls "$CKPT_DIR"/step_*.ckpt 2>/dev/null | awk -v M="$M" -v TOL="$CKPT_TOL" '
    function abs(x){return x<0?-x:x}
    { n=$0; sub(/.*step_0*/,"",n); sub(/\.ckpt/,"",n); n+=0; d=abs(n-M);
      if (d<=TOL && (b==""||d<bd)){b=$0; bd=d} }
    END{ if(b!="") print b }'
}

cell_done () {  # $1=label -> 0 if cell has >=NUM_EPISODES episodes
  local f="data/eval/search_bc_sweep/$1/argmax_ddim_n1_seed${START_SEED}_ep${NUM_EPISODES}/eval_log.json"
  python3 - "$f" "$NUM_EPISODES" <<'PY' 2>/dev/null
import json,sys
try:
    d=json.load(open(sys.argv[1])); sys.exit(0 if int(d.get("num_episodes",0))>=int(sys.argv[2]) else 1)
except Exception: sys.exit(1)
PY
}

for M in $MILESTONES; do
  tag="$((M/1000))k"
  label="${LABEL_PREFIX}_${tag}"
  ckpt="$(find_ckpt "$M")"
  if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then echo "[skip] $tag: no ckpt yet (within ±${CKPT_TOL} of $M)"; noc=$((noc+1)); continue; fi
  jn="${label}_n1"
  if cell_done "$label"; then echo "[done] $tag already has >=$NUM_EPISODES ep"; sd=$((sd+1)); continue; fi
  if grep -qxF "$jn" <<< "$QNAMES"; then echo "[queued] $jn"; sq=$((sq+1)); continue; fi
  IFS='|' read -r a p gres <<< "${ROUTES[$((i % ${#ROUTES[@]}))]}"; i=$((i+1))
  if [[ "$DRY" == "1" ]]; then echo "WOULD submit $jn ($ckpt, $a/$p)"; sub=$((sub+1)); continue; fi
  jid=$(sbatch --parsable \
        --job-name="$jn" --account="$a" --partition="$p" --gres="$gres" \
        --cpus-per-task=2 --mem=16G --time=04:00:00 \
        --export=ALL,CHECKPOINT="$PWD/$ckpt",N_LIST=1,NUM_EPISODES="$NUM_EPISODES",START_SEED="$START_SEED",NUM_WORKERS=1,MODE=argmax,BC_SAMPLER=ddim,BC_LABEL="$label",VIZ_Q=0,SAVE_VIDEOS="$SAVE_VIDEOS",VIDEO_FPS="$VIDEO_FPS" \
        "$SB")
  echo "submitted $jn -> $jid  ($a/$p, $ckpt)"; sub=$((sub+1))
  sleep 2   # stagger so eval_search_bc.sh's date-stamped WORK_DIR differs per cell
done
echo "=== ${LABEL_PREFIX} n=1 evals: submitted=$sub skip_done=$sd skip_queued=$sq no_ckpt=$noc ==="
echo "results -> data/eval/search_bc_sweep/${LABEL_PREFIX}_<tag>/argmax_ddim_n1_seed${START_SEED}_ep${NUM_EPISODES}/eval_log.json"
