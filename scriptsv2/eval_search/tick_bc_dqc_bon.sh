#!/bin/bash
# -----------------------------------------------------------------------------
# General hourly DQC-BoN tick — one script for any BC checkpoint's BoN sweep.
# Env (set inline in the crontab line):
#   CKPT (required) LABEL (required) JOBTAG (required) WANDB_RUN (required)
#   N_LIST (opt) MODE (opt, default softmax)
# Each tick: resubmits any missing/failed N cell (idempotent launcher, routes
# robotics/a40), logs current SR per N, and once all N are 50/50 logs the BoN
# curve (SR vs N) to wandb and removes its OWN crontab line (matched by the
# `# DQCBONTICK=<JOBTAG>` end-marker).
# Status log: data/eval/bc_dqc_chunk/_tick_<JOBTAG>.log
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true

: "${CKPT:?}"; : "${LABEL:?}"; : "${JOBTAG:?}"; : "${WANDB_RUN:?}"
N_LIST="${N_LIST:-1 2 4 8 16 32 64}"
MODE="${MODE:-softmax}"
NUM_EPISODES=50
START_SEED=1000
LOG="data/eval/bc_dqc_chunk/_tick_${JOBTAG}.log"
mkdir -p data/eval/bc_dqc_chunk
ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(ts)] === tick start ($LABEL) ===" >> "$LOG"

CKPT="$CKPT" LABEL="$LABEL" JOBTAG="$JOBTAG" N_LIST="$N_LIST" MODE="$MODE" \
  bash scriptsv2/eval_search/launch_bc_dqc_bon.sh >> "$LOG" 2>&1

alldone=1; line=""
for N in $N_LIST; do
  jf="data/eval/bc_dqc_chunk/${LABEL}_n${N}_seed${START_SEED}_ep${NUM_EPISODES}/eval_log.json"
  read -r ne sr < <(python3 -c "import json
try:
    d=json.load(open('$jf')); print(int(d.get('num_episodes',0)), f\"{d.get('success_rate',0):.3f}\")
except Exception: print(0,'-')" 2>/dev/null)
  [[ "${ne:-0}" -ge "$NUM_EPISODES" ]] || alldone=0
  line+=" n${N}=${sr}(${ne})"
done
echo "[$(ts)] BoN:$line" >> "$LOG"

if [[ "$alldone" == "1" ]]; then
  conda activate monkey-verifier 2>/dev/null || true
  WANDB_SILENT=true python scriptsv2/eval_search/log_dqc_bon_to_wandb.py \
    --run-id "$WANDB_RUN" --project "${WANDB_PROJECT:-BC_eggplant}" --label "$LABEL" --n-list "$N_LIST" >> "$LOG" 2>&1
  echo "[$(ts)] ALL DONE -> BoN curve logged to wandb ($WANDB_RUN); removing cron" >> "$LOG"
  crontab -l 2>/dev/null | grep -v "DQCBONTICK=${JOBTAG}\$" | crontab - 2>/dev/null || true
fi
echo "[$(ts)] === tick end (alldone=$alldone) ===" >> "$LOG"
