#!/bin/bash
# -----------------------------------------------------------------------------
# For each stack/spoon subset policy: once all 5 n=1 evals are complete, pick the
# best checkpoint (highest n=1 SR) and save 10 rollout videos (seeds 1000-1009,
# the SAME seeds across all policies). Idempotent; self-removes its crontab line
# once all 6 policies have their videos.
#
# Two phases per policy (marker-gated):
#   1. submit a 10-episode video eval (SAVE_VIDEOS=10) of the best checkpoint,
#   2. when that finishes, copy its videos to data/eval/best_ckpt_videos/<pol>/.
#
# Install (hourly at :58), preserving existing:
#   ( crontab -l 2>/dev/null; echo "58 * * * * /bin/bash /mmfs1/home/harine/RoboMonkey/scriptsv2/eval_search/save_best_videos.sh # BESTVID" ) | crontab -
# Remove:  crontab -l | grep -v BESTVID | crontab -
# Log: data/eval/best_ckpt_videos/_log.txt
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
DST=data/eval/best_ckpt_videos
mkdir -p "$DST"
LOG="$DST/_log.txt"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(ts)] === tick ===" >> "$LOG"
QN="$(squeue -u "$USER" -h -o '%j' 2>/dev/null)"
CKTOL=2000
ROUTES=( "robotics|gpu-a40|gpu:a40:1" "weirdlab|gpu-l40|gpu:l40:1" "weirdlab|gpu-l40s|gpu:l40s:1" )
ri=0; all_done=1

for pair in "stack:widowx_stack_cube" "spoon:widowx_spoon_on_towel"; do
  IFS=: read -r tk tenv <<< "$pair"
  for sub in sub100 sub50 sub25; do
    lp="unet50k_h8_${tk}_${sub}_nobs1"
    ckdir="diffusion_policy/data/train/bc_denoiser/bc_unet_h8_50k_${tk}_${sub}_nobs1/checkpoints"
    donemk="$DST/.done_${tk}_${sub}"
    [[ -f "$donemk" ]] && continue
    all_done=0

    # best checkpoint over the 5 milestone n=1 evals (need all 5 complete)
    best_tag=""; best_sr=-1; nd=0
    for tag in 10k 20k 30k 40k 50k; do
      jf="data/eval/search_bc_sweep/${lp}_${tag}/argmax_ddim_n1_seed1000_ep50/eval_log.json"
      read -r ne sr < <(python3 -c "import json
try:
    d=json.load(open('$jf')); print(int(d.get('num_episodes',0)), f\"{d.get('success_rate',0):.4f}\")
except Exception: print(0,'-')" 2>/dev/null)
      if [[ "${ne:-0}" -ge 50 ]]; then nd=$((nd+1)); awk "BEGIN{exit !($sr > $best_sr)}" && { best_sr="$sr"; best_tag="$tag"; }; fi
    done
    # save videos for the BEST-SO-FAR (>=1 eval done); it re-picks/updates as more
    # milestones (e.g. 30k) complete — vlabel encodes best_tag so a changed best
    # triggers a fresh video eval. Finalize (mark done) only once all 5 are in.
    if [[ "$nd" -lt 1 ]]; then echo "[$(ts)] ${tk}_${sub}: 0 evals done (waiting)" >> "$LOG"; continue; fi

    # resolve best checkpoint file
    M=$(( ${best_tag%k} * 1000 ))
    if [[ "$M" -ge 49000 && -f "$ckdir/latest.ckpt" ]]; then
      ck="$PWD/$ckdir/latest.ckpt"
    else
      ck=$(ls "$ckdir"/step_*.ckpt 2>/dev/null | awk -v M="$M" -v T="$CKTOL" '
        function abs(x){return x<0?-x:x}{n=$0;sub(/.*step_0*/,"",n);sub(/\.ckpt/,"",n);n+=0;d=abs(n-M);
        if(d<=T&&(b==""||d<bd)){b=$0;bd=d}}END{if(b!="")print b}')
      [[ -n "$ck" ]] && ck="$PWD/$ck"
    fi
    [[ -z "$ck" || ! -f "$ck" ]] && { echo "[$(ts)] ${tk}_${sub}: best=$best_tag but no ckpt file" >> "$LOG"; continue; }

    vlabel="bestvid_${tk}_${sub}_${best_tag}"
    out="data/eval/search_bc_sweep/${vlabel}/argmax_ddim_n1_seed1000_ep10"
    jn="vid_${tk}_${sub}"

    if [[ ! -f "$out/eval_log.json" ]]; then
      # phase 1: submit the 10-video eval of the best checkpoint (unless queued)
      if ! grep -qxF "$jn" <<< "$QN"; then
        IFS='|' read -r a p g <<< "${ROUTES[$((ri % ${#ROUTES[@]}))]}"; ri=$((ri+1))
        sbatch --job-name="$jn" --account="$a" --partition="$p" --gres="$g" \
          --cpus-per-task=2 --mem=16G --time=04:00:00 \
          --export=ALL,CHECKPOINT="$ck",N_LIST=1,NUM_EPISODES=10,START_SEED=1000,NUM_WORKERS=1,MODE=argmax,BC_SAMPLER=ddim,BC_LABEL="$vlabel",TASK="$tenv",VIZ_Q=0,SAVE_VIDEOS=10,VIDEO_FPS=10 \
          scriptsv2/eval_search/eval_search_bc.sbatch >> "$LOG" 2>&1
        echo "[$(ts)] ${tk}_${sub}: submitted video eval (best=$best_tag SR=$best_sr, $a/$p)" >> "$LOG"
      fi
      continue
    fi

    # phase 2: video eval done -> collect
    vd="$DST/${tk}_${sub}_best_${best_tag}"; mkdir -p "$vd"
    cp -f "$out/videos/"*.mp4 "$vd/" 2>/dev/null
    n=$(ls "$vd"/*.mp4 2>/dev/null | wc -l)
    if [[ "$nd" -ge 5 ]]; then
      touch "$donemk"
      echo "[$(ts)] ${tk}_${sub}: FINAL best=$best_tag (SR=$best_sr) -> $n videos in $vd" >> "$LOG"
    else
      echo "[$(ts)] ${tk}_${sub}: best-so-far=$best_tag (SR=$best_sr, ${nd}/5 evals) -> $n videos in $vd; will update after 30k" >> "$LOG"
    fi
  done
done

if [[ "$all_done" == "1" ]]; then
  echo "[$(ts)] all 6 policies have best-ckpt videos -> removing cron" >> "$LOG"
  crontab -l 2>/dev/null | grep -v 'BESTVID' | crontab - 2>/dev/null || true
fi
