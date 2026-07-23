#!/bin/bash
# -----------------------------------------------------------------------------
# Hourly cron tick: advance the H8 sub100_diverse-start BC n=1 eval pipeline.
# Idempotent. Each tick:
#   (1) logs training step,
#   (2) submits n=1 evals for any ready milestone checkpoint (idempotent launcher),
#   (3) once all 5 milestone evals are complete (50/50), logs the SR-vs-step curve
#       to the wandb run, reports the best checkpoint, and removes its own cron entry.
# No videos (SAVE_VIDEOS=0).
#
# Install (hourly at :10), preserving existing entries:
#   ( crontab -l 2>/dev/null; echo "10 * * * * /bin/bash /mmfs1/home/harine/RoboMonkey/scriptsv2/eval_search/tick_h8_sub100_evals.sh" ) | crontab -
# Remove early:
#   crontab -l | grep -v tick_h8_sub100_evals.sh | crontab -
#
# Status log: data/eval/search_bc_sweep/_tick_h8_sub100.log
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true

RUN_NAME=eggplant_bc_unet_h8_50k_sub100_diverse
RUN_DIR="diffusion_policy/data/train/bc_denoiser/${RUN_NAME}"
LABEL_PREFIX=unet50k_h8_sub100
MILESTONES="10000 20000 30000 40000 50000"
WANDB_RUN=unet_denoiser_pred8_roll4_50k_sub100_diverse
NUM_EPISODES=50
START_SEED=1000
LOG=data/eval/search_bc_sweep/_tick_h8_sub100.log
mkdir -p data/eval/search_bc_sweep
ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(ts)] === tick start ===" >> "$LOG"

# (1) training step
step=$(tail -1 "$RUN_DIR/logs.json.txt" 2>/dev/null | python3 -c "import sys,json
try: print(json.loads(sys.stdin.read()).get('global_step'))
except Exception: print('?')" 2>/dev/null || echo '?')
echo "[$(ts)] train step=$step/50000" >> "$LOG"

# (2) submit ready milestone evals (idempotent; no videos)
RUN_NAME="$RUN_NAME" LABEL_PREFIX="$LABEL_PREFIX" MILESTONES="$MILESTONES" SAVE_VIDEOS=0 \
  bash scriptsv2/eval_search/launch_unet50k_n1_evals.sh >> "$LOG" 2>&1

# (3) status per milestone
alldone=1; srline=""; best_tag=""; best_sr=-1
for M in $MILESTONES; do
  tag="$((M/1000))k"
  jf="data/eval/search_bc_sweep/${LABEL_PREFIX}_${tag}/argmax_ddim_n1_seed${START_SEED}_ep${NUM_EPISODES}/eval_log.json"
  read -r ne sr < <(python3 -c "import json
try:
    d=json.load(open('$jf')); print(int(d.get('num_episodes',0)), f\"{d.get('success_rate',0):.4f}\")
except Exception: print(0,'-')" 2>/dev/null)
  [[ "${ne:-0}" -ge "$NUM_EPISODES" ]] || alldone=0
  srline+=" ${tag}=${sr}(${ne}ep)"
  if [[ "$sr" != "-" && "${ne:-0}" -ge "$NUM_EPISODES" ]]; then
    awk "BEGIN{exit !($sr > $best_sr)}" && { best_sr="$sr"; best_tag="$tag"; }
  fi
done
echo "[$(ts)] SR:$srline" >> "$LOG"

# (4) all done -> log wandb + report best + self-remove
if [[ "$alldone" == "1" ]]; then
  conda activate monkey-verifier 2>/dev/null || true
  WANDB_SILENT=true python scriptsv2/eval_search/log_eval_sr_to_wandb.py \
    --run-id "$WANDB_RUN" --label-prefix "$LABEL_PREFIX" --milestones $MILESTONES >> "$LOG" 2>&1
  echo "[$(ts)] ALL DONE -> SR curve logged to wandb ($WANDB_RUN). BEST CHECKPOINT: ${best_tag} (SR=${best_sr})" >> "$LOG"
  crontab -l 2>/dev/null | grep -v 'tick_h8_sub100_evals.sh' | crontab - 2>/dev/null || true
fi
echo "[$(ts)] === tick end (alldone=$alldone best=${best_tag}:${best_sr}) ===" >> "$LOG"
