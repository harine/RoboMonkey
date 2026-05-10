#!/bin/bash
# End-to-end Bridge V2 -> Zarr pipeline:
#   stage 1: download + filter the Bridge V2 LeRobot subset (filter_bridge_v2.py)
#   stage 2: convert the filtered subset into SIMPLER-style Zarr shards (bridge_to_zarr.py)
#
# Bridge V2: https://huggingface.co/datasets/jesbu1/bridge_v2_lerobot
#
# Defaults pull "put carrot on plate" + "put eggplant in basket". The eggplant
# group can be widened/swapped via EGGPLANT_GROUP, e.g.
#   EGGPLANT_GROUP="eggplant_in_pot=eggplant,pot" bash scriptsv2/bridge_to_zarr/bridge_to_zarr.sh
#
# Usage:
#   bash scriptsv2/bridge_to_zarr/bridge_to_zarr.sh             # filter+download+convert
#   DRY_RUN=1   bash scriptsv2/bridge_to_zarr/bridge_to_zarr.sh # match-only (skip download)
#   SKIP_CONVERT=1 bash scriptsv2/bridge_to_zarr/bridge_to_zarr.sh  # download only
#   OUT_DIR=/data/bridge ZARR_DIR=openvla-mini/data \
#       bash scriptsv2/bridge_to_zarr/bridge_to_zarr.sh
#
# Env vars:
#   OUT_DIR                  filtered LeRobot dir            (default: data/bridge_v2_filtered)
#   ZARR_DIR                 destination root for .zarr shards
#                            (default: openvla-mini/data)
#   CARROT_GROUP             filter spec for carrot task     (default: carrot_on_plate=carrot,plate)
#   EGGPLANT_GROUP           filter spec for eggplant task   (default: eggplant_in_basket=eggplant,basket)
#   EXCLUDES                 substrings disqualifying tasks  (default: "take carrot off")
#   MAX_WORKERS              parallel HF download workers    (default: 8)
#   MAX_EPISODES_PER_GROUP   cap episodes per group          (default: unset)
#   DRY_RUN=1                only print matches, don't download
#   SKIP_CONVERT=1           skip the .zarr conversion stage

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR=${OUT_DIR:-data/bridge_v2_filtered}
ZARR_DIR=${ZARR_DIR:-openvla-mini/data}
CARROT_GROUP=${CARROT_GROUP:-"carrot_on_plate=carrot,plate"}
EGGPLANT_GROUP=${EGGPLANT_GROUP:-"eggplant_in_basket=eggplant,basket"}
MAX_WORKERS=${MAX_WORKERS:-8}
EXCLUDES=${EXCLUDES:-"take carrot off"}

PYTHON_BIN="python"
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx "simpler_env"; then
        conda activate simpler_env
        PYTHON_BIN="python"
    fi
fi

if ! "$PYTHON_BIN" -c "import huggingface_hub" 2>/dev/null; then
    echo "[setup] Installing huggingface_hub..."
    "$PYTHON_BIN" -m pip install --quiet "huggingface_hub"
fi

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

echo "============================================================"
echo "  [1/2] filter + download Bridge V2 subset"
echo "    output_dir=$OUT_DIR"
echo "    groups=$CARROT_GROUP | $EGGPLANT_GROUP"
echo "    excludes=$EXCLUDES"
echo "============================================================"

"$PYTHON_BIN" "$SCRIPT_DIR/filter_bridge_v2.py" \
    --output_dir "$OUT_DIR" \
    --task_groups "$CARROT_GROUP" "$EGGPLANT_GROUP" \
    --exclude "$EXCLUDES" \
    --max_workers "$MAX_WORKERS" \
    "${EXTRA_ARGS[@]}"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[done] DRY_RUN=1 set; skipping convert stage."
    exit 0
fi
if [ "${SKIP_CONVERT:-0}" = "1" ]; then
    echo "[done] SKIP_CONVERT=1 set; filtered LeRobot dataset at $OUT_DIR"
    exit 0
fi

echo
echo "============================================================"
echo "  [2/2] convert filtered subset -> .zarr shards"
echo "    zarr_dir=$ZARR_DIR"
echo "============================================================"

"$PYTHON_BIN" "$SCRIPT_DIR/bridge_to_zarr.py" \
    --bridge_dir "$OUT_DIR" \
    --task_filter "put carrot on plate" \
    --out "$ZARR_DIR/carrot_on_plate/bridge_v2_carrot.zarr" \
    ${OVERWRITE_ZARR:+--overwrite}

"$PYTHON_BIN" "$SCRIPT_DIR/bridge_to_zarr.py" \
    --bridge_dir "$OUT_DIR" \
    --task_filter "put eggplant in basket" \
    --out "$ZARR_DIR/eggplant_in_basket/bridge_v2_eggplant.zarr" \
    ${OVERWRITE_ZARR:+--overwrite}

echo
echo "[done] zarr shards under: $ZARR_DIR/{carrot_on_plate,eggplant_in_basket}/"
