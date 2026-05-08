#!/bin/bash
# Download a "carrot on plate" + "eggplant in basket" subset of the Bridge V2 LeRobot
# dataset (https://huggingface.co/datasets/jesbu1/bridge_v2_lerobot).
#
# The Bridge V2 task descriptions DO contain "put carrot on plate" (task 184) but
# do NOT contain the word "basket" anywhere near "eggplant". The script will warn
# loudly when a group matches 0 tasks. Override --task_groups below or pass a
# different EGGPLANT_GROUP env var to broaden the eggplant filter, e.g.:
#
#   EGGPLANT_GROUP="eggplant_in_pot=eggplant,pot" bash scriptsv2/download_carrot_and_eggplant.sh
#
# Usage:
#   bash scriptsv2/download_carrot_and_eggplant.sh                 # full download
#   DRY_RUN=1 bash scriptsv2/download_carrot_and_eggplant.sh       # match-only
#   OUT_DIR=/data/bridge bash scriptsv2/download_carrot_and_eggplant.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR=${OUT_DIR:-data/bridge_v2_filtered}
CARROT_GROUP=${CARROT_GROUP:-"carrot_on_plate=carrot,plate"}
EGGPLANT_GROUP=${EGGPLANT_GROUP:-"eggplant_in_basket=eggplant,basket"}
MAX_WORKERS=${MAX_WORKERS:-8}
EXCLUDES=${EXCLUDES:-"take carrot off"}

# Pick a Python: prefer the simpler_env conda env (already used in this repo) but
# fall back to system python.
PYTHON_BIN="python"
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx "simpler_env"; then
        conda activate simpler_env
        PYTHON_BIN="python"
    fi
fi

# Ensure huggingface_hub is available; install only if missing.
if ! "$PYTHON_BIN" -c "import huggingface_hub" 2>/dev/null; then
    echo "[setup] Installing huggingface_hub..."
    "$PYTHON_BIN" -m pip install --quiet "huggingface_hub"
fi

# Enable hf_transfer fast path only if the optional package is actually importable.
if "$PYTHON_BIN" -c "import hf_transfer" 2>/dev/null; then
    export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
else
    export HF_HUB_ENABLE_HF_TRANSFER=0
fi

EXTRA_ARGS=()
if [ "${DRY_RUN:-0}" = "1" ]; then
    EXTRA_ARGS+=(--dry_run)
fi
if [ -n "${MAX_EPISODES_PER_GROUP:-}" ]; then
    EXTRA_ARGS+=(--max_episodes_per_group "$MAX_EPISODES_PER_GROUP")
fi

echo "[run] output_dir=$OUT_DIR"
echo "[run] groups: $CARROT_GROUP | $EGGPLANT_GROUP"
echo "[run] excludes: $EXCLUDES"

"$PYTHON_BIN" scriptsv2/filter_bridge_v2.py \
    --output_dir "$OUT_DIR" \
    --task_groups "$CARROT_GROUP" "$EGGPLANT_GROUP" \
    --exclude "$EXCLUDES" \
    --max_workers "$MAX_WORKERS" \
    "${EXTRA_ARGS[@]}"
