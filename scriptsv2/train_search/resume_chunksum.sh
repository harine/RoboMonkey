#!/bin/bash
# Redirect the RESUME of running chunk-sum trainings to a chosen partition
# (e.g. h200) AND to gscratch output (fixes the home-quota corruption).
#
# Does NOT stop the running training: the resume link is submitted afterany the
# currently-RUNNING tr_<pol>, so it starts only when that link ends, bootstraps
# from the newest LOADABLE home checkpoint into gscratch, and continues there.
# The eval is re-pointed afterok the last resume link, reading gscratch ckpts.
#
# Usage:
#   POLICIES="noise0.0 noise0.4" \
#     RESUME_OPTS="--account=socialrl --partition=gpu-h200 --gres=gpu:h200:1 --cpus-per-task=8 --mem=120G --time=24:00:00" \
#     bash scriptsv2/train_search/resume_chunksum.sh
set -euo pipefail
repo_root="$(cd "$(dirname "$(realpath "$0")")/../.." && pwd)"; cd "$repo_root"

POLICIES="${POLICIES:?set POLICIES}"
GS_CKPT=/gscratch/robotics/harine/RoboMonkey_data/train/chunksum
RESUME_SBATCH=scriptsv2/train_search/train_chunksum_resume.sbatch
EVAL_SBATCH=scriptsv2/eval_search/eval_search_noise_sweep.sbatch
RESUME_OPTS="${RESUME_OPTS:---account=socialrl --partition=gpu-h200 --gres=gpu:h200:1 --cpus-per-task=8 --mem=120G --time=24:00:00}"
EVAL_OPTS="${EVAL_OPTS:---account=socialrl --partition=gpu-l40 --gres=gpu:l40:4 --cpus-per-task=20 --mem=320G --time=48:00:00}"
CHAIN_LEN="${TRAIN_CHAIN_LEN:-2}"
TRACK=data/eval/_chunksum_jobids.txt
mkdir -p "$(dirname "$TRACK")"; echo "# resume-redirect $(date)" >> "$TRACK"

for pol in $POLICIES; do
    echo "=== $pol -> resume ($RESUME_OPTS) + gscratch ==="
    l1=$(squeue -u "$USER" -h -t RUNNING -o "%i %j" 2>/dev/null | awk -v p="tr_$pol" '$2==p{print $1}' | head -1)
    if [[ -z "$l1" ]]; then echo "  no RUNNING tr_$pol (already ended?); using afterany on its pending link"; \
        l1=$(squeue -u "$USER" -h -o "%i %j" 2>/dev/null | awk -v p="tr_$pol" '$2==p{print $1}' | head -1); fi
    [[ -z "$l1" ]] && { echo "  no tr_$pol job at all; skip"; continue; }
    # cancel the OLD pending resume links + eval (keep the running/anchor link1)
    pend=$(squeue -u "$USER" -h -t PENDING -o "%i %j" 2>/dev/null | awk -v p="tr_$pol" -v e="ev_$pol" '($2==p||$2==e){print $1}')
    [[ -n "$pend" ]] && { scancel $pend; echo "  cancelled old pending: $(echo $pend|tr '\n' ' ')"; sleep 1; }

    export POLICY="$pol"
    prev="$l1"
    for ((k=1; k<=CHAIN_LEN; k++)); do
        jid=$(sbatch --parsable --dependency=afterany:"$prev" --job-name="tr_$pol" \
              --export=ALL $RESUME_OPTS "$RESUME_SBATCH")
        echo "  resume link $k/$CHAIN_LEN -> $jid (afterany:$prev)"
        echo "$jid resume $pol" >> "$TRACK"; prev="$jid"
    done
    export POLICIES="$pol" CKPT_DIR="$GS_CKPT" SCORE_AGG=sum NUM_WORKERS=4
    if [[ "$pol" == tmrl ]]; then export TCONT_LIST="0.0 0.4 0.8"; else unset TCONT_LIST; fi
    ejid=$(sbatch --parsable --dependency=afterok:"$prev" --job-name="ev_$pol" \
           --export=ALL $EVAL_OPTS "$EVAL_SBATCH")
    echo "  eval -> $ejid (afterok:$prev, reads gscratch)"
    echo "$ejid eval $pol" >> "$TRACK"
done
echo "[resume_chunksum] done. resume writes to $GS_CKPT/<pol>; evals read it."
