#!/bin/bash
# -----------------------------------------------------------------------------
# Idempotent launcher: softmax search BoN eval (N=1..64) on the 2 rmse/BC-only
# search policies trained on the FILTERED dataset (strict_distance_success), as
# their checkpoints reach each 10k milestone (10k/20k/30k/40k/50k):
#   diff_search_rmse_filtered_data  (SearchPolicyRoboMonkeyDiffusion, base critic)
#   gauss_search_rmse_filtered_data (SearchPolicyRoboMonkey,          base critic)
#
# SAME eval conditions as the PG campaign: MODE=softmax, predict-8/rollout-4, DQC
# chunk verifier (BAKED into each ckpt), 50 ep, seeds 1000-1049 -> directly
# comparable. PLUS Q-overlay videos: VIZ_Q=1 + RENDER_Q=1 render 10 episodes/cell
# with the candidate action trajectories + verifier Q-values into search_q/videos/.
#
# Idempotent & cron-safe: a milestone with no checkpoint yet is skipped; a cell
# already complete (>=50 ep) or PD/R in squeue is skipped; only gaps submitted,
# biased onto free robotics-a40 (weirdlab l40/l40s overflow), 2 CPU/16G.
#   bash scriptsv2/eval_search/launch_rmse_ckpt_evals.sh     # submit ready gaps
#   DRY=1 bash .../launch_rmse_ckpt_evals.sh                 # report only
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
SB=scriptsv2/eval_search/eval_search_dqc_chunk.sbatch
CKROOT=/gscratch/robotics/harine/RoboMonkey_data/train/rmse_filtered
N_LIST="1 2 4 8 16 32 64"
# sub50/sub100 extended to the full N=1..512 tail (per user request); filtered &
# sub25 keep N<=64 (sub25 is below the knee; filtered's tail is not needed).
N_EXT="1 2 4 8 16 32 64 128 256 512"
MILESTONES="5000 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000"
# gauss dropped (training stopped; its 10k/20k evals are done + in the doc).
RUNS="diff_search_rmse_filtered_data diff_search_rmse_sub100_diverse_data diff_search_rmse_sub50_diverse_data diff_search_rmse_sub25_diverse_data"
RESTORE=100000                         # both runs use the base DQC chunk critic
DRY="${DRY:-0}"
# per hyakalloc (2026-06-30): weirdlab gpu-a40 has the most headroom (5 free),
# weirdlab l40/l40s are full. Bias onto weirdlab a40; robotics a40 + cse a100 overflow.
# Route to FREE GPUs (2026-07-05: a40/a100 full; weirdlab gpu-l40 + cse gpu-l40s free).
# Re-check `hyakalloc` and repoint here whenever eval cells stall PD.
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
# nearest checkpoint to a milestone step within [M-200, M+1200]
find_ckpt () {  # $1=run $2=milestone -> prints ckpt path or empty
  local d="$CKROOT/$1/checkpoints"
  [ -d "$d" ] || return 0
  ls "$d"/step_*.ckpt 2>/dev/null | awk -v M="$2" '
    { s=$0; n=$0; sub(/.*step_0*/,"",n); sub(/\.ckpt/,"",n); n+=0;
      if (n>=M-200 && n<=M+1200) { dd=(n>M)?n-M:M-n; if (best=="" || dd<bd){best=s; bd=dd} } }
    END{ if(best!="") print best }'
}
for run in $RUNS; do
  case "$run" in
    diff_search_rmse_filtered*)  label=rmse_diff;   odir=search_dqc_chunk;                    nlist="$N_LIST";;  # 699-ep filtered
    diff_search_rmse_sub100*)    label=rmse_sub100; odir=search_dqc_chunk;                    nlist="$N_EXT";;   # 100-ep diverse-start -> N=1..512
    diff_search_rmse_sub50*)     label=rmse_sub50;  odir=filtered_sub50_diverse_eggplant;     nlist="$N_EXT";;   # 50-ep -> own dir, N=1..512
    diff_search_rmse_sub25*)     label=rmse_sub25;  odir=filtered_sub25_diverse_eggplant;     nlist="$N_LIST";;  # 25-ep -> own data+video dir
    gauss_search_rmse_filtered*) label=rmse_gauss;  odir=search_dqc_chunk;                    nlist="$N_LIST";;
    *) label="$run"; odir=search_dqc_chunk; nlist="$N_LIST";;
  esac
  for M in $MILESTONES; do
    ckpt="$(find_ckpt "$run" "$M")"
    tag="$((M/1000))k"
    outpfx="${label}_${tag}_chunk8_predh8_roll4_softmax"
    if [ -z "$ckpt" ]; then noc=$((noc+1)); continue; fi
    for N in $nlist; do
      jn="${label}_${tag}_n${N}"
      if done_cell "$odir" "$outpfx" "$N"; then sd=$((sd+1)); continue; fi
      if grep -qxF "$jn" <<< "$QNAMES"; then sq=$((sq+1)); continue; fi
      IFS='|' read -r a p gres <<< "${ROUTES[$((i % ${#ROUTES[@]}))]}"; i=$((i+1))
      # high-N tail needs more host RAM (N candidates buffered); CPU stays at 2.
      mem=16G; [ "$N" -ge 128 ] && mem=32G
      if [ "$DRY" = "1" ]; then echo "WOULD submit $jn ($run @$tag, $a/$p, $mem)"; sub=$((sub+1)); continue; fi
      jid=$(sbatch --parsable --job-name="$jn" --account="$a" --partition="$p" --gres="$gres" \
            --cpus-per-task=2 --mem="$mem" \
            --export=ALL,CHECKPOINT="$ckpt",N_SAMPLES="$N",MODE=softmax,SOFTMAX_TEMP=1.0,NUM_EPISODES=50,START_SEED=1000,DQC_RESTORE_EPOCH="$RESTORE",SAVE_VIDEOS=0,VIDEO_FPS=10,VIZ_Q=1,VIZ_Q_EPISODES=10,RENDER_Q=1,OUT_DIR="data/eval/${odir}/${outpfx}_n${N}_seed1000_ep50" \
            "$SB")
      echo "submitted $jn N=$N -> $jid ($run @$tag)"; sub=$((sub+1))
    done
  done
done
echo "=== rmse-filtered ckpt evals: submitted=$sub skip_done=$sd skip_queued=$sq ckpt_not_yet=$noc ==="
