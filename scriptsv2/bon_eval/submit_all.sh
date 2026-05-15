#!/bin/bash
# Submit the install job, then chain the 2-GPU parallel BoN sweep so it only
# starts after install succeeds.
#
# Usage:
#   cd /mmfs1/home/harine/RoboMonkey
#   bash scriptsv2/bon_eval/submit_all.sh
#
# Skip the install step (already done):
#   SKIP_INSTALL=1 bash scriptsv2/bon_eval/submit_all.sh
#
# Subset k:
#   KS="2 4 8" bash scriptsv2/bon_eval/submit_all.sh

set -euo pipefail
cd "$(dirname "$(realpath "$0")")/../.."

SBATCH_DIR="scriptsv2/bon_eval"
EXTRA_EXPORT=""
[[ -n "${KS:-}" ]] && EXTRA_EXPORT+=",KS=\"${KS}\""
[[ -n "${START_SEEDS:-}" ]] && EXTRA_EXPORT+=",START_SEEDS=\"${START_SEEDS}\""
[[ -n "${NUM_EPISODES:-}" ]] && EXTRA_EXPORT+=",NUM_EPISODES=${NUM_EPISODES}"
[[ -n "${NUM_WORKERS:-}" ]] && EXTRA_EXPORT+=",NUM_WORKERS=${NUM_WORKERS}"

if [[ "${SKIP_INSTALL:-0}" == "1" ]]; then
    sweep_jid=$(sbatch --parsable \
        --export="ALL${EXTRA_EXPORT}" \
        "$SBATCH_DIR/bon_eval_sweep.sbatch")
    echo "submitted sweep: $sweep_jid"
else
    install_jid=$(sbatch --parsable "$SBATCH_DIR/install_verifier_eval_deps.sbatch")
    echo "submitted install: $install_jid"
    sweep_jid=$(sbatch --parsable \
        --dependency=afterok:"$install_jid" \
        --export="ALL${EXTRA_EXPORT}" \
        "$SBATCH_DIR/bon_eval_sweep.sbatch")
    echo "submitted sweep:   $sweep_jid  (afterok:$install_jid)"
fi

echo
echo "monitor:"
echo "  squeue -u \$USER"
echo "  tail -f data/eval/bon/_summaries/slurm-${sweep_jid}.out"
echo "  tail -f data/eval/bon/_summaries/improved_verifier_sweep_*.txt"
