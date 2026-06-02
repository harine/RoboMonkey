#!/bin/bash
# Train the four noise-comparison checkpoints for the RoboMonkey eggplant
# search policy:
#
#   1) tunable_corruption @ obs_noise_level=0.1   (unconditioned, low noise)
#   2) tunable_corruption @ obs_noise_level=0.5   (unconditioned, mid noise)
#   3) tunable_corruption @ obs_noise_level=1.0   (unconditioned, max noise)
#   4) noise_conditioned                          (per-sample tcont ~ U[0,1],
#                                                  model sees the noise level)
#
# All four share the state encoder, conditional-diffusion head, and verifier
# from robomonkey_eggplant_search_state_diffusion.yaml. The only thing that
# changes is how much obs noise is applied and whether the model is told.
#
# Usage
# -----
#   bash scriptsv2/train_search/train_noise_sweep.sh [LEVELS_CSV]
#
# Examples
# --------
#   # All four runs sequentially:
#   bash scriptsv2/train_search/train_noise_sweep.sh
#
#   # Just the two unconditioned ones at 0.25 and 0.75:
#   LEVELS=0.25,0.75 RUN_COND=0 bash scriptsv2/train_search/train_noise_sweep.sh
#
#   # Just the conditioned model:
#   LEVELS= RUN_COND=1 bash scriptsv2/train_search/train_noise_sweep.sh
#
# Env vars (override on cmdline):
#   LEVELS         (default: 0.1,0.5,1.0; comma-separated obs_noise_level
#                   values for the unconditioned tunable runs. Set LEVELS=
#                   to skip the unconditioned sweep.)
#   RUN_COND       (default: 1; set 0 to skip the noise-conditioned model.)
#   DEVICE         (default: cuda:0)
#   CONDA_ENV      (default: robodiff)
#   NUM_EPOCHS     (default: unset = config default of 100)
#   BATCH_SIZE     (default: unset = config default of 256)
#   EXTRA_OVERRIDES (default: ""; passthrough Hydra overrides for every run.)

set -euo pipefail

LEVELS="${LEVELS:-0.1,0.5,1.0}"
RUN_COND="${RUN_COND:-1}"
DEVICE="${DEVICE:-cuda:0}"
CONDA_ENV="${CONDA_ENV:-robodiff}"
NUM_EPOCHS="${NUM_EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

full_path="$(realpath "$0")"
dir_path="$(dirname "$full_path")"
repo_root="$(cd "$dir_path/../.." && pwd)"
DP_ROOT="${DIFFUSION_POLICY_ROOT:-${repo_root}/diffusion_policy}"

export PYTHONPATH="${DP_ROOT}:${PYTHONPATH:-}"
cd "$DP_ROOT"

COMMON_OVERRIDES=(
    "training.device=${DEVICE}"
)
if [[ -n "$NUM_EPOCHS" ]]; then
    COMMON_OVERRIDES+=("training.num_epochs=${NUM_EPOCHS}")
fi
if [[ -n "$BATCH_SIZE" ]]; then
    COMMON_OVERRIDES+=("dataloader.batch_size=${BATCH_SIZE}" "val_dataloader.batch_size=${BATCH_SIZE}")
fi
if [[ -n "$EXTRA_OVERRIDES" ]]; then
    # shellcheck disable=SC2086
    COMMON_OVERRIDES+=( ${EXTRA_OVERRIDES} )
fi

run_one() {
    local label="$1"; shift
    echo
    echo "============================================================"
    echo "  $label"
    echo "  overrides: $*"
    echo "============================================================"
    python diffusion_policy/workspace/train_mlp_image_workspace.py "$@"
}

# --- Unconditioned tunable-corruption sweep -------------------------------
if [[ -n "$LEVELS" ]]; then
    IFS=',' read -r -a LEVEL_ARR <<< "$LEVELS"
    for L in "${LEVEL_ARR[@]}"; do
        run_one "tunable_corruption (no conditioning)  obs_noise_level=${L}" \
            --config-name=robomonkey_eggplant_search_tunable_corruption \
            "policy.obs_noise_level=${L}" \
            "${COMMON_OVERRIDES[@]}"
    done
fi

# --- Noise-conditioned model ----------------------------------------------
if [[ "$RUN_COND" == "1" ]]; then
    run_one "noise_conditioned (tmrl-style, tcont ~ U[0,1] in)" \
        --config-name=robomonkey_eggplant_search_noise_conditioned \
        "${COMMON_OVERRIDES[@]}"
fi

echo
echo "[train_noise_sweep] done."
