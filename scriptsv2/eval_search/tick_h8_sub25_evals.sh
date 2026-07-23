#!/bin/bash
# -----------------------------------------------------------------------------
# Hourly cron tick: advance the H8 sub25_diverse-start BC n=1 eval pipeline.
# Idempotent. Each tick:
#   (1) logs training step,
#   (2) submits n=1 evals (with SAVE_VIDEOS=5) for any ready milestone checkpoint,
#   (3) copies eval DATA (eval_log.json + episodes.jsonl) and the 5 rollout VIDEOS
#       for each ready checkpoint into
#       data/eval/filtered_sub25_diverse_eggplant/<tag>/,
#   (4) once all 5 milestone evals are complete (50/50), logs the SR-vs-step curve
#       to wandb, reports the best checkpoint, and removes its own cron entry.
#
# Install (hourly at :20), preserving existing entries:
#   ( crontab -l 2>/dev/null; echo "20 * * * * /bin/bash /mmfs1/home/harine/RoboMonkey/scriptsv2/eval_search/tick_h8_sub25_evals.sh" ) | crontab -
# Remove early:
#   crontab -l | grep -v tick_h8_sub25_evals.sh | crontab -
#
# Status log: data/eval/filtered_sub25_diverse_eggplant/_tick.log
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true

RUN_NAME=eggplant_bc_unet_h8_50k_sub25_diverse
RUN_DIR="diffusion_policy/data/train/bc_denoiser/${RUN_NAME}"
LABEL_PREFIX=unet50k_h8_sub25
MILESTONES="10000 20000 30000 40000 50000"
WANDB_RUN=unet_denoiser_pred8_roll4_50k_sub25_diverse
NUM_EPISODES=50
START_SEED=1000
DST=data/eval/filtered_sub25_diverse_eggplant
LOG="$DST/_tick.log"
mkdir -p "$DST"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(ts)] === tick start ===" >> "$LOG"

# (1) training step
step=$(tail -1 "$RUN_DIR/logs.json.txt" 2>/dev/null | python3 -c "import sys,json
try: print(json.loads(sys.stdin.read()).get('global_step'))
except Exception: print('?')" 2>/dev/null || echo '?')
echo "[$(ts)] train step=$step/50000" >> "$LOG"

# (2) submit ready milestone evals (idempotent; with videos)
RUN_NAME="$RUN_NAME" LABEL_PREFIX="$LABEL_PREFIX" MILESTONES="$MILESTONES" SAVE_VIDEOS=5 \
  bash scriptsv2/eval_search/launch_unet50k_n1_evals.sh >> "$LOG" 2>&1

# (3) collect data + 5 videos per ready ckpt
alldone=1; srline=""; best_tag=""; best_sr=-1
for M in $MILESTONES; do
  tag="$((M/1000))k"
  cell="data/eval/search_bc_sweep/${LABEL_PREFIX}_${tag}/argmax_ddim_n1_seed${START_SEED}_ep${NUM_EPISODES}"
  jf="$cell/eval_log.json"
  read -r ne sr < <(python3 -c "import json
try:
    d=json.load(open('$jf')); print(int(d.get('num_episodes',0)), f\"{d.get('success_rate',0):.4f}\")
except Exception: print(0,'-')" 2>/dev/null)
  if [[ "${ne:-0}" -ge "$NUM_EPISODES" ]]; then
    mkdir -p "$DST/$tag"
    cp -f "$jf" "$cell/episodes.jsonl" "$DST/$tag/" 2>/dev/null
    [[ -d "$cell/videos" ]] && cp -f "$cell/videos/"*.mp4 "$DST/$tag/" 2>/dev/null
    awk "BEGIN{exit !($sr > $best_sr)}" && { best_sr="$sr"; best_tag="$tag"; }
  else
    alldone=0
  fi
  srline+=" ${tag}=${sr}(${ne}ep)"
done
echo "[$(ts)] SR:$srline" >> "$LOG"

# (4) all done -> log wandb + report best + self-remove
if [[ "$alldone" == "1" ]]; then
  conda activate monkey-verifier 2>/dev/null || true
  WANDB_SILENT=true python scriptsv2/eval_search/log_eval_sr_to_wandb.py \
    --run-id "$WANDB_RUN" --label-prefix "$LABEL_PREFIX" --milestones $MILESTONES >> "$LOG" 2>&1
  echo "[$(ts)] ALL DONE -> SR curve logged to wandb ($WANDB_RUN). BEST CHECKPOINT: ${best_tag} (SR=${best_sr}). data+videos in $DST/<tag>/" >> "$LOG"
  crontab -l 2>/dev/null | grep -v 'tick_h8_sub25_evals.sh' | crontab - 2>/dev/null || true
fi
echo "[$(ts)] === tick end (alldone=$alldone best=${best_tag}:${best_sr}) ===" >> "$LOG"
