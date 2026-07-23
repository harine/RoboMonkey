#!/bin/bash
# -----------------------------------------------------------------------------
# Idempotent launcher: softmax search BoN eval (N=1..64) on the 4 PG-trained
# search-policy checkpoints (BC + advantage-PG, lambda=0.1) as they come in:
#   pg_gauss_base / pg_gauss_v2  (SearchPolicyRoboMonkey  + base/v2 DQC chunk critic)
#   pg_diff_base  / pg_diff_v2   (SearchPolicyRoboMonkeyDiffusion + base/v2 critic)
#
# Same eval protocol as the prior search sweeps: MODE=softmax, predict-8/rollout-4,
# DQC chunk verifier (BAKED into each ckpt), 50 ep, seeds 1000-1049 -> directly
# comparable. Evals the checkpoint nearest each 10k milestone (10k/20k/30k/40k/50k).
#
# Idempotent & safe to re-run from cron: a milestone with no checkpoint yet is
# skipped (picked up later); a cell already complete (>=50 ep) or PD/R in squeue
# is skipped; only gaps are submitted, onto free weirdlab l40/l40s (2 CPU/16G).
#
#   bash scriptsv2/eval_search/launch_pg_ckpt_evals.sh        # submit ready gaps
#   DRY=1 bash .../launch_pg_ckpt_evals.sh                    # report only
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
SB=scriptsv2/eval_search/eval_search_dqc_chunk.sbatch
PGROOT=/gscratch/robotics/harine/RoboMonkey_data/train/pg
N_LIST="1 2 4 8 16 32 64"
MILESTONES="10000 20000 30000 40000 50000"
# v2-critic runs DROPPED: the v2 (finetuned) DQC critic is a broken BoN verifier —
# both PG+v2 runs collapse to ~0% by N>=2-8 at 10k AND 20k (selecting by v2 score
# picks failures; more training does not fix it). Only base-critic runs are evaluated.
RUNS="pg_gauss_base pg_diff_base"
DRY="${DRY:-0}"
# robotics gpu-a40 listed first/twice: weirdlab hits its AssocGrpGRES cap, robotics a40
# is free -> bias cells onto robotics, weirdlab l40/l40s as overflow.
ROUTES=( "robotics|gpu-a40|gpu:a40:1" "weirdlab|gpu-l40s|gpu:l40s:1" "robotics|gpu-a40|gpu:a40:1" "weirdlab|gpu-l40|gpu:l40:1" )

QNAMES="$(squeue -u "$USER" -h -o '%j')"
i=0; sub=0; sd=0; sq=0; noc=0
done_cell () {  # $1=outprefix $2=N
  python3 - "$1" "$2" <<'PY' 2>/dev/null
import json,sys
p=f"data/eval/search_dqc_chunk/{sys.argv[1]}_n{sys.argv[2]}_seed1000_ep50/eval_log.json"
try:
    d=json.load(open(p)); sys.exit(0 if int(d.get("num_episodes",0))>=50 else 1)
except Exception: sys.exit(1)
PY
}
# nearest checkpoint to a milestone step within [M-200, M+1200]
find_ckpt () {  # $1=run $2=milestone -> prints ckpt path or empty
  local d="$PGROOT/$1/checkpoints"
  [ -d "$d" ] || return 0
  ls "$d"/step_*.ckpt 2>/dev/null | awk -v M="$2" '
    { s=$0; n=$0; sub(/.*step_0*/,"",n); sub(/\.ckpt/,"",n); n+=0;
      if (n>=M-200 && n<=M+1200) { dd=(n>M)?n-M:M-n; if (best=="" || dd<bd){best=s; bd=dd} } }
    END{ if(best!="") print best }'
}
for run in $RUNS; do
  # v2-critic runs restart epoch numbering at 0 (params up to 50000); base = 100000.
  case "$run" in *_v2) RESTORE=50000;; *) RESTORE=100000;; esac
  for M in $MILESTONES; do
    ckpt="$(find_ckpt "$run" "$M")"
    tag="$((M/1000))k"
    outpfx="${run}_${tag}_chunk8_predh8_roll4_softmax"
    if [ -z "$ckpt" ]; then noc=$((noc+1)); continue; fi
    for N in $N_LIST; do
      jn="${run}_${tag}_n${N}"; jn="${jn//pg_/}"   # shorten job name
      if done_cell "$outpfx" "$N"; then sd=$((sd+1)); continue; fi
      if grep -qxF "$jn" <<< "$QNAMES"; then sq=$((sq+1)); continue; fi
      IFS='|' read -r a p gres <<< "${ROUTES[$((i % ${#ROUTES[@]}))]}"; i=$((i+1))
      if [ "$DRY" = "1" ]; then echo "WOULD submit $jn ($run @$tag, $a/$p)"; sub=$((sub+1)); continue; fi
      jid=$(sbatch --parsable --job-name="$jn" --account="$a" --partition="$p" --gres="$gres" \
            --cpus-per-task=2 --mem=16G \
            --export=ALL,CHECKPOINT="$ckpt",N_SAMPLES="$N",MODE=softmax,SOFTMAX_TEMP=1.0,NUM_EPISODES=50,START_SEED=1000,DQC_RESTORE_EPOCH="$RESTORE",SAVE_VIDEOS="${SAVE_VIDEOS:-0}",VIDEO_FPS=10,VIZ_Q="${VIZ_Q:-1}",VIZ_Q_EPISODES="${VIZ_Q_EPISODES:-10}",RENDER_Q="${RENDER_Q:-0}",OUT_DIR="data/eval/search_dqc_chunk/${outpfx}_n${N}_seed1000_ep50" \
            "$SB")
      echo "submitted $jn N=$N -> $jid ($run @$tag)"; sub=$((sub+1))
    done
  done
done
echo "=== PG-ckpt evals: submitted=$sub skip_done=$sd skip_queued=$sq ckpt_not_yet=$noc ==="