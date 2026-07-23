#!/bin/bash
# -----------------------------------------------------------------------------
# Hourly cron tick: advance the H8-filtered DQC Best-of-N sweep. Idempotent.
# Each tick:
#   (1) (re)submits any missing/failed N cell (launcher skips done/queued/running),
#   (2) logs current SR per N,
#   (3) once all N are 50/50, logs the BoN curve (SR vs N) to the wandb run and
#       removes its own crontab entry.
#
# Install (hourly at :05, offset from the n=1 tick), preserving existing entries:
#   ( crontab -l 2>/dev/null; echo "5 * * * * /bin/bash /mmfs1/home/harine/RoboMonkey/scriptsv2/eval_search/tick_h8filt_dqc_bon.sh" ) | crontab -
# Remove early:
#   crontab -l | grep -v tick_h8filt_dqc_bon.sh | crontab -
#
# Status log: data/eval/bc_dqc_chunk/_tick_h8filt_dqc.log
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true

LABEL=bcH8filt50k_chunk8_predh8_roll4_ddpm
N_LIST="1 2 4 8 16 32 64"
NUM_EPISODES=50
START_SEED=1000
WANDB_RUN=unet_denoiser_pred8_roll4_50k_filtered
LOG=data/eval/bc_dqc_chunk/_tick_h8filt_dqc.log
mkdir -p data/eval/bc_dqc_chunk
ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(ts)] === tick start ===" >> "$LOG"

# (1) submit any gaps (idempotent: skips done/queued/running)
N_LIST="$N_LIST" NUM_EPISODES="$NUM_EPISODES" START_SEED="$START_SEED" LABEL="$LABEL" \
  bash scriptsv2/eval_search/launch_h8filt_dqc_bon.sh >> "$LOG" 2>&1

# (2) status per N
alldone=1; line=""
for N in $N_LIST; do
  jf="data/eval/bc_dqc_chunk/${LABEL}_n${N}_seed${START_SEED}_ep${NUM_EPISODES}/eval_log.json"
  read -r ne sr < <(python3 -c "import json
try:
    d=json.load(open('$jf')); print(int(d.get('num_episodes',0)), f\"{d.get('success_rate',0):.3f}\")
except Exception: print(0,'-')" 2>/dev/null)
  [[ "${ne:-0}" -ge "$NUM_EPISODES" ]] || alldone=0
  line+=" n${N}=${sr}(${ne}ep)"
done
echo "[$(ts)] BoN:$line" >> "$LOG"

# (3) all N done -> log wandb (idempotent marker) + self-remove
if [[ "$alldone" == "1" ]]; then
  conda activate monkey-verifier 2>/dev/null || true
  WANDB_SILENT=true python scriptsv2/eval_search/log_dqc_bon_to_wandb.py \
    --run-id "$WANDB_RUN" --label "$LABEL" --n-list "$N_LIST" \
    --num-episodes "$NUM_EPISODES" --start-seed "$START_SEED" >> "$LOG" 2>&1
  echo "[$(ts)] ALL DONE -> BoN curve logged to wandb ($WANDB_RUN); removing cron entry" >> "$LOG"
  crontab -l 2>/dev/null | grep -v 'tick_h8filt_dqc_bon.sh' | crontab - 2>/dev/null || true
fi
echo "[$(ts)] === tick end (alldone=$alldone) ===" >> "$LOG"
