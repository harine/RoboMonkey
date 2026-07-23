#!/bin/bash
# Run the STRICT success filter (one-try, no-failure, monotone approach, single
# held grasp) for one or more SIMPLER scenes. CPU-only + fast (reads zarr, numpy);
# runs fine on a login node or a small CPU srun.
#
# Usage:
#   bash scriptsv2/data_collect/filter_strict.sh                 # stack + spoon (default)
#   bash scriptsv2/data_collect/filter_strict.sh stack           # one scene
#   bash scriptsv2/data_collect/filter_strict.sh stack spoon carrot
#   DRY_RUN=1 bash scriptsv2/data_collect/filter_strict.sh stack # preview counts only
#
# Env knobs (passed through to the python; see --help for all): BACKTRACK_ABS,
# BACKTRACK_REL, END_OVER_D0_MAX, MAX_DGE_RISE, etc.
set -euo pipefail

SCENES=("$@")
[ ${#SCENES[@]} -eq 0 ] && SCENES=(stack spoon)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="$SCRIPT_DIR/filter_strict_success.py"

CONDA_BASE="${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}"
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null)}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate simpler_env
python -c "import zarr, numpy" 2>/dev/null || pip install zarr numcodecs

EXTRA=()
[ -n "${DRY_RUN:-}" ]        && EXTRA+=(--dry-run)
[ -n "${BACKTRACK_ABS:-}" ]  && EXTRA+=(--backtrack-abs "$BACKTRACK_ABS")
[ -n "${BACKTRACK_REL:-}" ]  && EXTRA+=(--backtrack-rel "$BACKTRACK_REL")
[ -n "${END_OVER_D0_MAX:-}" ] && EXTRA+=(--end-over-d0-max "$END_OVER_D0_MAX")
[ -n "${MAX_DGE_RISE:-}" ]   && EXTRA+=(--max-dge-rise "$MAX_DGE_RISE")

cd "$REPO_ROOT"
for scene in "${SCENES[@]}"; do
  echo "==================== strict filter: $scene ===================="
  python "$PY" --scene "$scene" "${EXTRA[@]}"
  echo
done
