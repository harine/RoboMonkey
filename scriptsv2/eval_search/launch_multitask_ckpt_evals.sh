#!/bin/bash
# -----------------------------------------------------------------------------
# Idempotent BoN eval launcher for the NEW-TASK diffusion search policies
# (stack_cube + spoon_on_towel, sub100/50/25 diverse-start) at the 5k/10k/20k
# checkpoints. Mirrors launch_rmse_ckpt_evals.sh exactly: MODE=softmax, N=1..64,
# predict-8/rollout-4, 50 ep, seeds 1000-1049, + Q-overlay videos (VIZ_Q+RENDER_Q,
# 10 ep). The DQC chunk verifier is BAKED into each ckpt with the TASK-CORRECT
# MUSE embedding (task_emb_{stack,spoon}_use3.npy) from training, so NO embedding
# override is needed here -- eval reconstructs the right verifier automatically.
#
# Idempotent & cron-safe: a milestone with no ckpt yet is skipped; a cell already
# complete (>=50 ep) or PD/R in squeue is skipped; only gaps are submitted. 2 CPU/16G.
#   bash scriptsv2/eval_search/launch_multitask_ckpt_evals.sh     # submit ready gaps
#   DRY=1 bash .../launch_multitask_ckpt_evals.sh                 # report only
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
SB=scriptsv2/eval_search/eval_search_dqc_chunk.sbatch
CKROOT=/gscratch/robotics/harine/RoboMonkey_data/train/rmse_multitask
N_LIST="1 2 4 8 16 32 64"
MILESTONES="10000 20000 30000 40000 50000 60000 70000 80000 90000 100000"
RUNS="stack_sub100 stack_sub50 stack_sub25 spoon_sub100 spoon_sub50 spoon_sub25"
RESTORE=100000                         # base DQC chunk critic (params_100000)
DRY="${DRY:-0}"
# a40 + a100 have headroom (l40s busy with the 6 trainings). Spread across accounts.
# Route to whatever hyakalloc shows FREE (2026-07-05: a40/a100 are 100% full; weirdlab
# gpu-l40 ×5 + cse gpu-l40s ×7 are free). eval sbatch --time=12h fits cse's 1-day cap.
# When evals stall (all PD), re-check `hyakalloc` and repoint this list at the free rows.
ROUTES=( "weirdlab|gpu-l40|gpu:l40:1" "cse|gpu-l40s|gpu:l40s:1" )

QNAMES="$(squeue -u "$USER" -h -o '%j')"
i=0; sub=0; sd=0; sq=0; noc=0
done_cell () {  # $1=odir $2=outprefix $3=N
  python3 - "$1" "$2" "$3" <<'PY' 2>/dev/null
import json,sys
p=f"data/eval/{sys.argv[1]}/{sys.argv[2]}_n{sys.argv[3]}_seed1000_ep50/eval_log.json"
try:
    d=json.load(open(p)); sys.exit(0 if int(d.get("num_episodes",0))>=50 else 1)
except Exception: sys.exit(1)
PY
}
find_ckpt () {  # $1=run $2=milestone -> nearest ckpt within [M-200, M+1200]
  local d="$CKROOT/diff_search_rmse_$1_diverse/checkpoints"
  [ -d "$d" ] || return 0
  ls "$d"/step_*.ckpt 2>/dev/null | awk -v M="$2" '
    { s=$0; n=$0; sub(/.*step_0*/,"",n); sub(/\.ckpt/,"",n); n+=0;
      if (n>=M-200 && n<=M+1200) { dd=(n>M)?n-M:M-n; if (best=="" || dd<bd){best=s; bd=dd} } }
    END{ if(best!="") print best }'
}
for run in $RUNS; do
  case "$run" in
    stack_*) odir=stack_cube_search; task=widowx_stack_cube;;      # eval in the STACK env
    spoon_*) odir=spoon_towel_search; task=widowx_spoon_on_towel;;  # eval in the SPOON env
    *)       odir=multitask_search; task=widowx_put_eggplant_in_basket;;
  esac
  label="$run"
  for M in $MILESTONES; do
    ckpt="$(find_ckpt "$run" "$M")"
    tag="$((M/1000))k"
    outpfx="${label}_${tag}_chunk8_predh8_roll4_softmax"
    if [ -z "$ckpt" ]; then noc=$((noc+1)); continue; fi
    for N in $N_LIST; do
      jn="${label}_${tag}_n${N}"
      if done_cell "$odir" "$outpfx" "$N"; then sd=$((sd+1)); continue; fi
      if grep -qxF "$jn" <<< "$QNAMES"; then sq=$((sq+1)); continue; fi
      IFS='|' read -r a p gres <<< "${ROUTES[$((i % ${#ROUTES[@]}))]}"; i=$((i+1))
      if [ "$DRY" = "1" ]; then echo "WOULD submit $jn ($run @$tag, $a/$p)"; sub=$((sub+1)); continue; fi
      jid=$(sbatch --parsable --job-name="$jn" --account="$a" --partition="$p" --gres="$gres" \
            --cpus-per-task=2 --mem=16G \
            --export=ALL,CHECKPOINT="$ckpt",N_SAMPLES="$N",MODE=softmax,SOFTMAX_TEMP=1.0,NUM_EPISODES=50,START_SEED=1000,DQC_RESTORE_EPOCH="$RESTORE",TASK="$task",SAVE_VIDEOS=0,VIDEO_FPS=10,VIZ_Q=1,VIZ_Q_EPISODES=10,RENDER_Q=1,OUT_DIR="data/eval/${odir}/${outpfx}_n${N}_seed1000_ep50" \
            "$SB")
      echo "submitted $jn N=$N -> $jid ($run @$tag)"; sub=$((sub+1))
    done
  done
done
echo "=== multitask ckpt evals: submitted=$sub skip_done=$sd skip_queued=$sq ckpt_not_yet=$noc ==="
