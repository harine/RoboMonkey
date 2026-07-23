#!/bin/bash
# -----------------------------------------------------------------------------
# Idempotent monitor: render the N-sweep Q-overlay videos for the advantage/PG
# (policy-gradient, lambda=0.1) search-policy checkpoints at each 10k milestone
# AS THEY LAND, into the consolidated folder
#   data/eval/june_19-24/videos/advantage_pg/<policy><step>/n<N>/
# matching the 10k cells already there (same render_q.py top-right-box legend).
#
# Each run does two things, both idempotent & safe to repeat from cron:
#   (1) STAGE: any finished eval job whose search_q/videos are rendered is copied
#       into its consolidated n-folder.
#   (2) SUBMIT: for each (policy, milestone, N) gap with a checkpoint present and
#       NO video yet, submit one eval_search_dqc_chunk job (VIZ_Q + RENDER_Q) —
#       but ONLY onto a partition that currently has an idle/mix GPU node, so we
#       never queue behind a full cluster. Milestones with no checkpoint yet are
#       skipped and picked up on a later tick.
#
#   bash scriptsv2/eval_search/render_pg_milestone_videos.sh      # stage + submit
#   DRY=1 bash .../render_pg_milestone_videos.sh                  # report only
#   MILES="20000 30000" N_LIST="8 16" bash .../...               # subset
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
SB=scriptsv2/eval_search/eval_search_dqc_chunk.sbatch
PGROOT=/gscratch/robotics/harine/RoboMonkey_data/train/pg
P=data/eval/search_dqc_chunk
DST=data/eval/june_19-24/videos/advantage_pg
N_LIST="${N_LIST:-1 8 16 32 64}"
MILES="${MILES:-20000 30000 40000 50000}"     # 10k already rendered by hand
DRY="${DRY:-0}"
declare -A POL=( [pg_diff_base]=diff [pg_gauss_base]=gauss )

# nearest checkpoint to a milestone step within [M-200, M+1200]
find_ckpt () {  # $1=run $2=milestone -> prints ckpt path or empty
  local d="$PGROOT/$1/checkpoints"; [ -d "$d" ] || return 0
  ls "$d"/step_*.ckpt 2>/dev/null | awk -v M="$2" '
    { s=$0;n=$0;sub(/.*step_0*/,"",n);sub(/\.ckpt/,"",n);n+=0;
      if(n>=M-200&&n<=M+1200){dd=(n>M)?n-M:M-n; if(best==""||dd<bd){best=s;bd=dd}} }
    END{ if(best!="")print best }'
}

# first route whose partition has >=1 idle/mix GPU node (else empty -> defer)
free_route () {
  local routes=( "cse|gpu-l40s|gpu:l40s:1" "cse|gpu-a100|gpu:a100:1" \
                 "robotics|gpu-a40|gpu:a40:1" "weirdlab|gpu-l40|gpu:l40:1" )
  for r in "${routes[@]}"; do
    IFS='|' read -r a p g <<< "$r"
    local avail=$(sinfo -h -p "$p" -t idle,mix -o '%D' 2>/dev/null | awk '{s+=$1} END{print s+0}')
    [ "${avail:-0}" -ge 1 ] && { echo "$r"; return 0; }
  done
  return 1
}

QN="$(squeue -u "$USER" -h -o '%j' 2>/dev/null)"
staged=0; subbed=0; skip_done=0; skip_q=0; nockpt=0; nogpu=0
for run in pg_diff_base pg_gauss_base; do
  pol="${POL[$run]}"
  for M in $MILES; do
    ck="$(find_ckpt "$run" "$M")"; tag="$((M/1000))k"
    [ -z "$ck" ] && { nockpt=$((nockpt+1)); continue; }
    for N in $N_LIST; do
      out="$P/pg_${pol}_base_${tag}_chunk8_predh8_roll4_softmax_n${N}_seed1000_ep10_vidq"
      tgt="$DST/${pol}${tag}/n${N}"
      jn="pgvid_${pol}${tag}_n${N}"
      [ "$(ls "$tgt"/*.mp4 2>/dev/null | wc -l)" -ge 10 ] && { skip_done=$((skip_done+1)); continue; }
      if [ "$(ls "$out/search_q/videos"/*.mp4 2>/dev/null | wc -l)" -ge 10 ]; then
        if [ "$DRY" = "1" ]; then echo "WOULD stage $jn -> ${tgt#"$DST"/}"; else
          mkdir -p "$tgt"; rm -f "$tgt"/*.mp4; cp "$out/search_q/videos"/*.mp4 "$tgt"/; fi
        staged=$((staged+1)); continue
      fi
      grep -qxF "$jn" <<< "$QN" && { skip_q=$((skip_q+1)); continue; }
      route="$(free_route)" || { nogpu=$((nogpu+1)); continue; }
      IFS='|' read -r a p g <<< "$route"
      if [ "$DRY" = "1" ]; then echo "WOULD submit $jn ($a/$p) ckpt=$(basename "$ck")"; subbed=$((subbed+1)); continue; fi
      jid=$(sbatch --parsable --job-name="$jn" --account="$a" --partition="$p" --gres="$g" \
            --cpus-per-task=2 --mem=16G --time=03:00:00 \
            --export=ALL,CHECKPOINT="$ck",N_SAMPLES="$N",OUT_DIR="$out",MODE=softmax,SOFTMAX_TEMP=1.0,NUM_EPISODES=10,START_SEED=1000,SAVE_VIDEOS=0,VIDEO_FPS=3,VIZ_Q=1,VIZ_Q_EPISODES=10,RENDER_Q=1 \
            "$SB") && { echo "submitted $jn -> $jid ($a/$p)"; subbed=$((subbed+1)); QN="$QN"$'\n'"$jn"; } || echo "FAIL $jn"
    done
  done
done
echo "=== pg_milestone_videos: staged=$staged submitted=$subbed skip_done=$skip_done skip_queued=$skip_q no_ckpt=$nockpt no_free_gpu=$nogpu ==="
