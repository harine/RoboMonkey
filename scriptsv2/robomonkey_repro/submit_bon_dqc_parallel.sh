#!/bin/bash
# =============================================================================
# Submit the OpenVLA + DQC BoN scaling sweep (N=2^0..2^9) as PARALLEL per-cell
# jobs, one GPU each, so the whole curve finishes in ~one cell's wall-time
# instead of the sum. Each job:
#   * runs a disjoint subset of N (so together they cover the full N_LIST),
#   * has its OWN action-server port (ACTION_PORT) -> safe to co-locate on a node,
#   * has its OWN EVAL_CWD + OUT_ROOT -> no transfer_images / progress.json races,
#   * is independently RESUMABLE (progress.json per job; re-run this script to
#     top up after the maintenance window).
# Merge afterwards with: python bon_progress.py merge ... (or merge_bon_dqc.sh).
#
# Usage:  bash scriptsv2/robomonkey_repro/submit_bon_dqc_parallel.sh
# Knobs:  NUM_TRIALS (50), SEED (1), WALLTIME (06:00:00), ACCOUNT/PART/GRES,
#         CELLS (override the N split; "name=Nlist" entries, ';'-separated).
# =============================================================================
set -u
REPO=/mmfs1/home/harine/RoboMonkey
SWEEP_ROOT="$REPO/data/eval/robomonkey_repro/bon_sweep_dqc/par"
NUM_TRIALS="${NUM_TRIALS:-50}"
SEED="${SEED:-1}"
WALLTIME="${WALLTIME:-06:00:00}"
ACCOUNT="${ACCOUNT:-cse}"
PART="${PART:-gpu-l40s}"
GRES="${GRES:-gpu:l40s:1}"
BASE_PORT="${BASE_PORT:-3210}"

# Balanced N split (group label = COMMA-separated N list; commas avoid spaces in
# sbatch --export, the sbatch converts them back). Low-N cells are mostly done
# already (seeded below); the two slowest (256, 512) get their own job so the
# long pole is one big cell, not three.
# NB: do NOT name this GROUPS — that's a bash special read-only var (group IDs).
# N values within a cell are joined with '_' (NOT ',' — sbatch --export splits on
# commas, which would truncate the N list to its first value). The sbatch turns
# '_' back into spaces.
CELLS_DEFAULT="g0=1_2_4;g1=8_16;g2=32_64;g3=128_256;g4=512"
CELLS="${CELLS:-$CELLS_DEFAULT}"

mkdir -p "$SWEEP_ROOT"

# Seed group g0 with the N=1/2/4 episodes the sequential sweep already finished,
# so the parallel run reuses them instead of recomputing (~1h saved).
SAVED=/tmp/dqc_sweep_progress_saved.json
if [ -f "$SAVED" ]; then
  mkdir -p "$SWEEP_ROOT/g0"
  cp "$SAVED" "$SWEEP_ROOT/g0/progress.json"
  echo "[seed] g0 progress.json <- $SAVED"
fi

i=0
IFS=';' read -r -a GARR <<< "$CELLS"
for entry in "${GARR[@]}"; do
  label="${entry%%=*}"
  nlist="${entry#*=}"
  port=$(( BASE_PORT + i ))
  out="$SWEEP_ROOT/$label"
  cwd="$out/_cwd"
  mkdir -p "$cwd"
  jid=$(sbatch --parsable \
    --job-name="bondqc_${label}" \
    --account="$ACCOUNT" --partition="$PART" --gres="$GRES" \
    --time="$WALLTIME" \
    --output="$SWEEP_ROOT/slurm-${label}-%j.out" \
    --export=ALL,VERIFIER_TYPE=dqc,SEED="$SEED",INIT_SAMPLES=9,NUM_TRIALS="$NUM_TRIALS",N_LIST="$nlist",ACTION_PORT="$port",OUT_ROOT="$out",EVAL_CWD="$cwd" \
    "$REPO/scriptsv2/robomonkey_repro/bon_sweep_dqc.sbatch")
  echo "submitted $label  N=[$nlist]  port=$port  job=$jid  out=$out"
  i=$(( i + 1 ))
done
echo ""
echo "watch:  squeue -u \$USER -n bondqc_g0,bondqc_g1,bondqc_g2,bondqc_g3,bondqc_g4"
echo "merge:  bash $REPO/scriptsv2/robomonkey_repro/merge_bon_dqc.sh"
