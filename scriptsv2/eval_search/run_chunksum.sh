#!/bin/bash
# Orchestrate the CHUNK-SUM retrain + eval of the search policies.
#
# For each policy: submit a 1-GPU TRAIN job (chunk-sum verifier scoring baked
# into the checkpoint config), then an EVAL job that depends (afterok) on it and
# reads the freshly-trained checkpoint from data/train/chunksum/<pol>.
#
# Chunk-sum = the verifier scores the 8 executed actions (horizon [1..8]) and
# SUMS them, in BOTH training and inference; BoN picks the highest-sum chunk.
#
# Resumable: evals run with FORCE=0 (skip done cells). Re-running this script
# re-submits; in-flight trainings/evals are not duplicated if you scope POLICIES.
#
# Usage:
#   bash scriptsv2/eval_search/run_chunksum.sh                       # all 4
#   POLICIES="noise0.0" bash scriptsv2/eval_search/run_chunksum.sh   # one
#   TRAIN_OPTS="--account=weirdlab --partition=gpu-l40s --gres=gpu:l40s:1 ..." \
#     EVAL_OPTS="..." bash ... run_chunksum.sh
set -euo pipefail
repo_root="$(cd "$(dirname "$(realpath "$0")")/../.." && pwd)"
cd "$repo_root"

POLICIES="${POLICIES:-noise0.0 noise0.4 noise0.8 tmrl}"
CKPT_DIR="${repo_root}/diffusion_policy/data/train/chunksum"
TRAIN_SBATCH=scriptsv2/train_search/train_chunksum.sbatch
EVAL_SBATCH=scriptsv2/eval_search/eval_search_noise_sweep.sbatch

TRAIN_OPTS="${TRAIN_OPTS:---account=socialrl --partition=gpu-l40 --gres=gpu:l40:1 --cpus-per-task=10 --mem=120G --time=24:00:00}"
EVAL_OPTS="${EVAL_OPTS:---account=socialrl --partition=gpu-l40 --gres=gpu:l40:4 --cpus-per-task=20 --mem=320G --time=48:00:00}"
NUM_WORKERS="${NUM_WORKERS:-4}"

TRACK="data/eval/_chunksum_jobids.txt"
mkdir -p "$(dirname "$TRACK")" data/train/chunksum
echo "# chunksum launch $(date)" >> "$TRACK"

for pol in $POLICIES; do
    echo "=== $pol ==="
    # --- TRAIN (1 GPU) ---
    export POLICY="$pol"
    tjid=$(sbatch --parsable --job-name="tr_${pol}" --export=ALL $TRAIN_OPTS "$TRAIN_SBATCH")
    echo "  train -> $tjid"
    echo "$tjid train $pol" >> "$TRACK"

    # --- EVAL (afterok train) ---
    # Export via environment so space-containing values (TCONT_LIST) survive.
    export POLICIES_ONE="$pol"
    export POLICIES="$pol" CKPT_DIR="$CKPT_DIR" SCORE_AGG="sum" NUM_WORKERS="$NUM_WORKERS"
    if [[ "$pol" == "tmrl" ]]; then export TCONT_LIST="0.0 0.4 0.8"; else unset TCONT_LIST; fi
    ejid=$(sbatch --parsable --dependency=afterok:"$tjid" --job-name="ev_${pol}" \
        --export=ALL $EVAL_OPTS "$EVAL_SBATCH")
    echo "  eval  -> $ejid (afterok:$tjid)"
    echo "$ejid eval $pol afterok:$tjid" >> "$TRACK"
    # reset POLICIES for next iter (loop var uses $pol, not $POLICIES)
done
unset POLICY POLICIES POLICIES_ONE CKPT_DIR SCORE_AGG TCONT_LIST 2>/dev/null || true

echo
echo "[run_chunksum] job IDs -> $TRACK"
echo "[run_chunksum] checkpoints -> diffusion_policy/data/train/chunksum/<pol>/checkpoints/latest.ckpt"
