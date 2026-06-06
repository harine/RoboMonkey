#!/bin/bash
# Launch or RESUME the executed-window-scoring eval sweeps as self-chaining
# SLURM jobs.
#
# Resumability:
#   * FORCE=0 (default) -> every job skips cells that already have eval_log.json.
#   * Each sweep is submitted as a chain of CHAIN_LEN jobs linked by
#     --dependency=afterany. If a link ends early (timeout / preemption /
#     node failure / crash) the NEXT link auto-starts and resumes the remaining
#     cells. If the sweep already finished, later links no-op in minutes.
#   * Manual resume: just re-run this script. New jobs reuse completed cells.
#
# Sweeps (override which run via JOBS="noise tmrl bc"):
#   noise : noise0.4 + noise0.8            (eval_search_noise_sweep.sbatch)
#   tmrl  : tmrl @ tcont 0.4 / 0.8 / 0.0   (eval_search_noise_sweep.sbatch)
#   bc    : BC sweep                       (eval_search_bc.sbatch)
#
# All use executed-window verifier scoring (eval_search.sh default
# SCORE_WINDOW=executed) and save render data + videos for the first 10 fixed
# seeds (noise/tmrl: VIZ_Q defaults in the sweep .sh; bc: its own VIZ_Q=1).
#
# Usage:
#   bash scriptsv2/eval_search/run_exec_sweeps.sh
#   JOBS="noise tmrl" CHAIN_LEN=6 bash scriptsv2/eval_search/run_exec_sweeps.sh
#   # override resources per sweep, e.g.:
#   NOISE_OPTS="--account=socialrl --partition=gpu-h200 --gres=gpu:h200:4 \
#     --cpus-per-task=24 --mem=480G --time=72:00:00" bash ... run_exec_sweeps.sh
set -euo pipefail
repo_root="$(cd "$(dirname "$(realpath "$0")")/../.." && pwd)"
cd "$repo_root"

CHAIN_LEN="${CHAIN_LEN:-4}"
JOBS="${JOBS:-noise tmrl bc}"
NOISE_SBATCH=scriptsv2/eval_search/eval_search_noise_sweep.sbatch
BC_SBATCH=scriptsv2/eval_search/eval_search_bc.sbatch

# Per-sweep SLURM resources (override via env). Defaults spread the three
# sweeps across different QOS caps so they don't block each other.
NOISE_OPTS="${NOISE_OPTS:---account=socialrl --partition=gpu-l40s --gres=gpu:l40s:4 --cpus-per-task=20 --mem=320G --time=72:00:00}"
TMRL_OPTS="${TMRL_OPTS:---account=weirdlab --partition=gpu-l40 --gres=gpu:l40:4 --cpus-per-task=20 --mem=320G --time=72:00:00}"
BC_OPTS="${BC_OPTS:---account=socialrl --partition=gpu-l40 --gres=gpu:l40:4 --cpus-per-task=20 --mem=320G --time=48:00:00}"

TRACK="data/eval/_exec_sweep_jobids.txt"
mkdir -p "$(dirname "$TRACK")"
echo "# launched $(date)" >> "$TRACK"

submit_chain() {
    local name="$1" file="$2" opts="$3"
    # Caller exports the per-sweep env (POLICIES / TCONT_LIST / NUM_WORKERS /
    # CHECKPOINT); --export=ALL carries it into each job.
    local prev="" jid k
    for ((k=1; k<=CHAIN_LEN; k++)); do
        if [[ -z "$prev" ]]; then
            jid=$(sbatch --parsable --job-name="$name" --export=ALL $opts "$file")
        else
            jid=$(sbatch --parsable --dependency=afterany:"$prev" \
                  --job-name="$name" --export=ALL $opts "$file")
        fi
        echo "  $name  link $k/$CHAIN_LEN -> $jid${prev:+  (afterany:$prev)}"
        echo "$jid $name link$k" >> "$TRACK"
        prev="$jid"
    done
}

for job in $JOBS; do
    case "$job" in
        noise)
            echo "[noise0.4 + noise0.8]  -> $NOISE_OPTS"
            POLICIES="noise0.4 noise0.8" NUM_WORKERS=4 \
                submit_chain nsweep_noise "$NOISE_SBATCH" "$NOISE_OPTS"
            ;;
        tmrl)
            echo "[tmrl @ tcont 0.4/0.8/0.0]  -> $TMRL_OPTS"
            POLICIES="tmrl" TCONT_LIST="0.4 0.8 0.0" NUM_WORKERS=4 \
                submit_chain nsweep_tmrl "$NOISE_SBATCH" "$TMRL_OPTS"
            ;;
        bc)
            echo "[bc]  -> $BC_OPTS"
            NUM_WORKERS=4 submit_chain bc_sweep "$BC_SBATCH" "$BC_OPTS"
            ;;
        *)
            echo "unknown JOBS entry '$job' (use: noise tmrl bc)" >&2; exit 1;;
    esac
done

echo
echo "[run_exec_sweeps] job IDs tracked in $TRACK"
echo "[run_exec_sweeps] to STOP a sweep fully, scancel ALL its chain links"
echo "                  (afterany starts the next link even on cancel):"
echo "                  scancel --name=nsweep_noise   # or nsweep_tmrl / bc_sweep"
