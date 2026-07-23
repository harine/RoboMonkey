#!/bin/bash
# -----------------------------------------------------------------------------
# Idempotent launcher: softmax search BoN eval (N=1..128) on the GAUSSIAN search
# policy checkpoints (job 36182914, eggplant_gaussian_dqc_chunk8 ... predh8_rollout4)
# as they come in during training. Targets the future 10k-step checkpoints
# (30k/40k/50k); 10k & 20k were already swept (gauss10k/gauss20k).
#
# Each cell = the baked DQC chunk verifier (predict-8/rollout-4, score_full_horizon,
# q_agg=min, USE_EMA=0 — gaussian trained use_ema=False), MODE=softmax, 50 ep,
# seeds 1000-1049. Same config as the gauss10k/20k sweeps, so directly comparable.
#
# Idempotent & safe to re-run from the hourly cron:
#   * a STEP whose step_00XXXXXX.ckpt does not exist yet is skipped (training hasn't
#     reached it) -> it's picked up automatically on a later pass.
#   * a cell already complete (eval_log >=50 ep) or already PD/R in squeue is skipped.
#   * only missing cells are submitted, round-robin onto free weirdlab l40/l40s.
#
#   bash scriptsv2/eval_search/launch_gaussian_ckpt_evals.sh           # submit ready gaps
#   DRY=1 bash .../launch_gaussian_ckpt_evals.sh                       # report only
#   STEPS="30000 40000 50000" bash .../launch_gaussian_ckpt_evals.sh  # override target steps
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
SB=scriptsv2/eval_search/eval_search_dqc_chunk.sbatch
GDIR=/gscratch/robotics/harine/RoboMonkey_data/train/dqc_chunk/eggplant_gaussian_dqc_chunk8_hyak_predhorizon8_rollout4/checkpoints
N_LIST="1 2 4 8 16 32 64 128"
STEPS="${STEPS:-30000 40000 50000}"      # future 10k multiples (10k/20k already done)
DRY="${DRY:-0}"
ROUTES=( "weirdlab|gpu-l40s|gpu:l40s:1" "weirdlab|gpu-l40|gpu:l40:1" )

QNAMES="$(squeue -u "$USER" -h -o '%j')"
i=0; sub=0; sd=0; sq=0; noc=0
done_cell () {
  python3 - "$1" "$2" <<'PY' 2>/dev/null
import json,sys
p=f"data/eval/search_dqc_chunk/{sys.argv[1]}_n{sys.argv[2]}_seed1000_ep50/eval_log.json"
try:
    d=json.load(open(p)); sys.exit(0 if int(d.get("num_episodes",0))>=50 else 1)
except Exception: sys.exit(1)
PY
}
for step in $STEPS; do
  ckpt="$GDIR/step_$(printf '%07d' "$step").ckpt"
  tag="$((step/1000))k"
  outpfx="gauss${tag}_chunk8_predh8_roll4_softmax"
  if [ ! -f "$ckpt" ]; then noc=$((noc+1)); continue; fi      # not trained to here yet
  for N in $N_LIST; do
    jn="g${tag}_n${N}"
    if done_cell "$outpfx" "$N"; then sd=$((sd+1)); continue; fi
    if grep -qxF "$jn" <<< "$QNAMES"; then sq=$((sq+1)); continue; fi
    IFS='|' read -r a p gres <<< "${ROUTES[$((i % 2))]}"; i=$((i+1))
    if [ "$DRY" = "1" ]; then echo "WOULD submit $jn ($a/$p)"; sub=$((sub+1)); continue; fi
    jid=$(sbatch --parsable --job-name="$jn" --account="$a" --partition="$p" --gres="$gres" \
          --cpus-per-task=2 --mem=16G \
          --export=ALL,CHECKPOINT="$ckpt",N_SAMPLES="$N",MODE=softmax,SOFTMAX_TEMP=1.0,NUM_EPISODES=50,START_SEED=1000,OUT_DIR="data/eval/search_dqc_chunk/${outpfx}_n${N}_seed1000_ep50" \
          "$SB")
    echo "submitted $jn N=$N $a/$p -> $jid"; sub=$((sub+1))
  done
done
echo "=== gaussian-ckpt softmax evals: submitted=$sub skip_done=$sd skip_queued=$sq ckpt_not_yet=$noc (steps: $STEPS) ==="