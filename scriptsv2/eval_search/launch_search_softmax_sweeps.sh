#!/bin/bash
# -----------------------------------------------------------------------------
# Idempotent launcher for the softmax search-policy BoN sweeps (DQC chunk verifier,
# predict-8 / rollout-4, 50 ep/cell, seeds 1000-1049):
#   * gaussian search 10k & 20k  (SearchPolicyRoboMonkey, baked DQC chunk verifier)
#   * diffusion search 10k/20k/33k (path-fixed ckpts, baked DQC chunk verifier)
# MODE=softmax (SOFTMAX_TEMP=1.0). The argmax counterparts live in the same dir
# under the no-"_softmax" prefixes; this script only manages the softmax cells.
#
# Idempotent: a cell is SKIPPED if its eval_log.json already has >=50 episodes, or
# if a job with its name is already PD/R in squeue. Only missing/dead cells are
# (re)submitted, round-robin across the FREE l40/l40s pools (ratio 16 -> 2 CPU is
# well under, no GPU stranding). Safe to re-run (e.g. from the monitoring cron).
#
#   bash scriptsv2/eval_search/launch_search_softmax_sweeps.sh          # submit gaps
#   DRY=1 bash scriptsv2/eval_search/launch_search_softmax_sweeps.sh    # just report
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
SB=scriptsv2/eval_search/eval_search_dqc_chunk.sbatch
GDIR=/gscratch/robotics/harine/RoboMonkey_data/train/dqc_chunk/eggplant_gaussian_dqc_chunk8_hyak_predhorizon8_rollout4/checkpoints
EDIR=/gscratch/robotics/harine/RoboMonkey_data/eval_ckpts
N_LIST="1 2 4 8 16 32 64 128 256 512"
DRY="${DRY:-0}"

# family: name_prefix | checkpoint | out_subprefix
FAMILIES=(
  "g10k|$GDIR/step_0010000.ckpt|gauss10k_chunk8_predh8_roll4_softmax"
  "g20k|$GDIR/step_0020000.ckpt|gauss20k_chunk8_predh8_roll4_softmax"
  "d1ksm|$EDIR/diffusion_dqc_h8_roll4_step_0001000_1k_pathfix.ckpt|diff1k_chunk8_predh8_roll4_softmax"
  "d2ksm|$EDIR/diffusion_dqc_h8_roll4_step_0002000_2k_pathfix.ckpt|diff2k_chunk8_predh8_roll4_softmax"
  "d5ksm|$EDIR/diffusion_dqc_h8_roll4_step_0005000_5k_pathfix.ckpt|diff5k_chunk8_predh8_roll4_softmax"
  "d10ksm|$EDIR/diffusion_dqc_h8_roll4_step_0010000_10k_pathfix.ckpt|diff10k_chunk8_predh8_roll4_softmax"
  "d20ksm|$EDIR/diffusion_dqc_h8_roll4_step_0020001_20k_pathfix.ckpt|diff20k_chunk8_predh8_roll4_softmax"
  "d33ksm|$EDIR/diffusion_dqc_h8_roll4_step33001_pathfix.ckpt|diffusion_chunk8_predh8_roll4_softmax"
)
# weirdlab l40/l40s have headroom (per-partition cap); cse/l40s is chronically over its
# account cap (14/8) and robotics/a100 is account-capped + held by the 5-day collection,
# so we route only to the two weirdlab pools that actually free up.
ROUTES=( "weirdlab|gpu-l40s|gpu:l40s:1" "weirdlab|gpu-l40|gpu:l40:1" )

QNAMES="$(squeue -u "$USER" -h -o '%j')"   # currently PD/R job names
i=0; sub=0; skip_done=0; skip_q=0
done_cell () {  # $1=out_subprefix $2=N  -> 0 if eval_log has >=50 ep
  python3 - "$1" "$2" <<'PY' 2>/dev/null
import json,sys
p=f"data/eval/search_dqc_chunk/{sys.argv[1]}_n{sys.argv[2]}_seed1000_ep50/eval_log.json"
try:
    d=json.load(open(p)); sys.exit(0 if int(d.get("num_episodes",0))>=50 else 1)
except Exception: sys.exit(1)
PY
}
for fam in "${FAMILIES[@]}"; do
  IFS='|' read -r pfx ckpt outpfx <<< "$fam"
  for N in $N_LIST; do
    jn="${pfx}_n${N}"
    if done_cell "$outpfx" "$N"; then skip_done=$((skip_done+1)); continue; fi
    if grep -qxF "$jn" <<< "$QNAMES"; then skip_q=$((skip_q+1)); continue; fi
    IFS='|' read -r acct part gres <<< "${ROUTES[$((i % ${#ROUTES[@]}))]}"; i=$((i+1))
    if [ "$DRY" = "1" ]; then printf 'WOULD submit %-12s %s/%s\n' "$jn" "$acct" "$part"; sub=$((sub+1)); continue; fi
    jid=$(sbatch --parsable --job-name="$jn" --account="$acct" --partition="$part" --gres="$gres" \
          --cpus-per-task=2 --mem=16G \
          --export=ALL,CHECKPOINT="$ckpt",N_SAMPLES="$N",MODE=softmax,SOFTMAX_TEMP=1.0,NUM_EPISODES=50,START_SEED=1000,OUT_DIR="data/eval/search_dqc_chunk/${outpfx}_n${N}_seed1000_ep50" \
          "$SB")
    printf 'submitted %-12s N=%-4s %-9s -> %s\n' "$jn" "$N" "$acct/$part" "$jid"; sub=$((sub+1))
  done
done
echo "=== submitted=$sub  skip_done=$skip_done  skip_queued=$skip_q ==="