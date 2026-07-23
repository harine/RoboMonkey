#!/bin/bash
# -----------------------------------------------------------------------------
# General hourly BC n=1 eval tick — parameterized (one script for any BC run).
# Params via env (set inline in the crontab line):
#   RUN_NAME     (required) bc_denoiser subdir
#   LABEL_PREFIX (required) eval cell/label prefix + tick-log key
#   WANDB_RUN    (required) wandb run id to log the SR curve into
#   MILESTONES   (opt, default "10000 20000 30000 40000 50000")
#   START_SEED   (opt, default 1000)
#   SAVE_VIDEOS  (opt, default 0)   >0 = save N rollout videos per ckpt
#   DST          (opt)              if set, copy eval_log.json+episodes.jsonl(+videos) there per tag
#
# Each tick: logs train step, submits n=1 evals for ready milestones (idempotent
# launcher, routes to robotics/a40), optionally collects data+videos, and once
# all milestones are 50/50 logs the SR curve to wandb, reports best, and removes
# its OWN crontab line (matched by the `# BCTICK=<LABEL_PREFIX>` end-marker).
#
# Install (unique marker per run), preserving existing entries:
#   ( crontab -l 2>/dev/null; echo "25 * * * * RUN_NAME=... LABEL_PREFIX=foo WANDB_RUN=... /bin/bash $PWD/scriptsv2/eval_search/tick_bc_n1.sh # BCTICK=foo" ) | crontab -
# Remove early:
#   crontab -l | grep -v 'BCTICK=foo$' | crontab -
# Status log: data/eval/search_bc_sweep/_tick_<LABEL_PREFIX>.log
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true

: "${RUN_NAME:?set RUN_NAME}"; : "${LABEL_PREFIX:?set LABEL_PREFIX}"; : "${WANDB_RUN:?set WANDB_RUN}"
MILESTONES="${MILESTONES:-10000 20000 30000 40000 50000}"
START_SEED="${START_SEED:-1000}"
NUM_EPISODES=50
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"
DST="${DST:-}"
TASK="${TASK:-widowx_put_eggplant_in_basket}"       # SimplerEnv rollout task
WANDB_PROJECT="${WANDB_PROJECT:-BC_eggplant}"        # wandb project for the SR curve
RUN_DIR="diffusion_policy/data/train/bc_denoiser/${RUN_NAME}"
LOG="data/eval/search_bc_sweep/_tick_${LABEL_PREFIX}.log"
mkdir -p data/eval/search_bc_sweep
ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(ts)] === tick start ($LABEL_PREFIX) ===" >> "$LOG"

step=$(tail -1 "$RUN_DIR/logs.json.txt" 2>/dev/null | python3 -c "import sys,json
try: print(json.loads(sys.stdin.read()).get('global_step'))
except Exception: print('?')" 2>/dev/null || echo '?')
echo "[$(ts)] train step=$step/50000" >> "$LOG"

RUN_NAME="$RUN_NAME" LABEL_PREFIX="$LABEL_PREFIX" MILESTONES="$MILESTONES" SAVE_VIDEOS="$SAVE_VIDEOS" TASK="$TASK" \
  bash scriptsv2/eval_search/launch_unet50k_n1_evals.sh >> "$LOG" 2>&1

alldone=1; srline=""; best_tag=""; best_sr=-1
for M in $MILESTONES; do
  tag="$((M/1000))k"
  cell="data/eval/search_bc_sweep/${LABEL_PREFIX}_${tag}/argmax_ddim_n1_seed${START_SEED}_ep${NUM_EPISODES}"
  read -r ne sr < <(python3 -c "import json
try:
    d=json.load(open('$cell/eval_log.json')); print(int(d.get('num_episodes',0)), f\"{d.get('success_rate',0):.4f}\")
except Exception: print(0,'-')" 2>/dev/null)
  if [[ "${ne:-0}" -ge "$NUM_EPISODES" ]]; then
    awk "BEGIN{exit !($sr > $best_sr)}" && { best_sr="$sr"; best_tag="$tag"; }
    if [[ -n "$DST" ]]; then
      mkdir -p "$DST/$tag"
      cp -f "$cell/eval_log.json" "$cell/episodes.jsonl" "$DST/$tag/" 2>/dev/null
      [[ -d "$cell/videos" ]] && cp -f "$cell/videos/"*.mp4 "$DST/$tag/" 2>/dev/null
    fi
  else
    alldone=0
  fi
  srline+=" ${tag}=${sr}(${ne}ep)"
done
echo "[$(ts)] SR:$srline" >> "$LOG"

if [[ "$alldone" == "1" ]]; then
  conda activate monkey-verifier 2>/dev/null || true
  WANDB_SILENT=true python scriptsv2/eval_search/log_eval_sr_to_wandb.py \
    --run-id "$WANDB_RUN" --project "$WANDB_PROJECT" --label-prefix "$LABEL_PREFIX" --milestones $MILESTONES >> "$LOG" 2>&1
  echo "[$(ts)] ALL DONE -> SR curve logged to wandb ($WANDB_RUN). BEST CHECKPOINT: ${best_tag} (SR=${best_sr})" >> "$LOG"
  crontab -l 2>/dev/null | grep -v "BCTICK=${LABEL_PREFIX}\$" | crontab - 2>/dev/null || true
fi
echo "[$(ts)] === tick end (alldone=$alldone best=${best_tag}:${best_sr}) ===" >> "$LOG"
