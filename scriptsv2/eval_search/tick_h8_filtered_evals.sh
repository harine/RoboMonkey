#!/bin/bash
# -----------------------------------------------------------------------------
# Hourly cron tick: advance the H8 filtered BC eval pipeline. Idempotent & safe
# to run repeatedly. Each tick:
#   (1) logs H8 training step,
#   (2) submits n=1 evals for any ready milestone checkpoint (idempotent launcher),
#   (3) copies the 5 videos/ckpt into data/eval/videos/june30_filtered_h8/<tag>/,
#   (4) once all 5 milestone evals are complete (50/50), logs the SR curve to the
#       wandb run and removes its own crontab entry (job done).
#
# Append to crontab (hourly at :00), preserving existing entries:
#   ( crontab -l 2>/dev/null; echo "0 * * * * /bin/bash /mmfs1/home/harine/RoboMonkey/scriptsv2/eval_search/tick_h8_filtered_evals.sh" ) | crontab -
# Remove early:
#   crontab -l | grep -v tick_h8_filtered_evals.sh | crontab -
#
# Status log: data/eval/videos/june30_filtered_h8/_tick.log
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true

RUN_NAME=eggplant_bc_unet_h8_50k_filtered
RUN_DIR="diffusion_policy/data/train/bc_denoiser/${RUN_NAME}"
LABEL_PREFIX=unet50k_h8_filt
MILESTONES="10000 20000 30000 40000 50000"
WANDB_RUN=unet_denoiser_pred8_roll4_50k_filtered
DST=data/eval/videos/june30_filtered_h8
LOG="$DST/_tick.log"
mkdir -p "$DST"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(ts)] === tick start ===" >> "$LOG"

# (1) training step
step=$(tail -1 "$RUN_DIR/logs.json.txt" 2>/dev/null | python3 -c "import sys,json
try: print(json.loads(sys.stdin.read()).get('global_step'))
except Exception: print('?')" 2>/dev/null || echo '?')
echo "[$(ts)] train step=$step/50000" >> "$LOG"

# (2) submit ready milestone evals (idempotent: skips done/queued/no-ckpt)
RUN_NAME="$RUN_NAME" LABEL_PREFIX="$LABEL_PREFIX" MILESTONES="$MILESTONES" SAVE_VIDEOS=5 \
  bash scriptsv2/eval_search/launch_unet50k_n1_evals.sh >> "$LOG" 2>&1

# (3) collect videos for any cell that has its 5
for M in $MILESTONES; do
  tag="$((M/1000))k"
  src="data/eval/search_bc_sweep/${LABEL_PREFIX}_${tag}/argmax_ddim_n1_seed1000_ep50/videos"
  n=$(ls "$src"/*.mp4 2>/dev/null | wc -l)
  if [[ "$n" -ge 5 ]]; then mkdir -p "$DST/$tag"; cp -f "$src"/*.mp4 "$DST/$tag/" 2>/dev/null; fi
done

# (4) all 5 evals complete? -> log wandb (idempotent marker) + self-remove cron
alldone=1; srline=""
for M in $MILESTONES; do
  tag="$((M/1000))k"
  jf="data/eval/search_bc_sweep/${LABEL_PREFIX}_${tag}/argmax_ddim_n1_seed1000_ep50/eval_log.json"
  read -r ne sr < <(python3 -c "import json
try:
    d=json.load(open('$jf')); print(int(d.get('num_episodes',0)), f\"{d.get('success_rate',0):.3f}\")
except Exception: print(0, '-')" 2>/dev/null)
  [[ "${ne:-0}" -ge 50 ]] || alldone=0
  srline+=" ${tag}=${sr}(${ne}ep)"
done
echo "[$(ts)] SR:$srline" >> "$LOG"

if [[ "$alldone" == "1" ]]; then
  conda activate monkey-verifier 2>/dev/null || true
  WANDB_SILENT=true python scriptsv2/eval_search/log_eval_sr_to_wandb.py \
    --run-id "$WANDB_RUN" --label-prefix "$LABEL_PREFIX" --milestones $MILESTONES >> "$LOG" 2>&1
  echo "[$(ts)] ALL DONE -> SR logged to wandb ($WANDB_RUN); removing cron entry" >> "$LOG"
  crontab -l 2>/dev/null | grep -v 'tick_h8_filtered_evals.sh' | crontab - 2>/dev/null || true
fi
echo "[$(ts)] === tick end (alldone=$alldone) ===" >> "$LOG"
