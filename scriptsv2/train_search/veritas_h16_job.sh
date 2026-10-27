#!/bin/bash
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
case "$2" in
  train) conda activate dqc_jax ;;
  *) conda activate simpler_env ;;
esac
export SLURM_EXPORT_ENV=ALL
export PYTHONPATH="$PWD/diffusion_policy:${PYTHONPATH:-}"
export DIFFUSION_POLICY_ROOT="$PWD/diffusion_policy"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa DISPLAY=""
python -u scriptsv2/train_search/veritas_h16_campaign.py "$@" &
PID=$!
trap 'kill -TERM "$PID" 2>/dev/null || true; wait "$PID" || true; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
wait "$PID"
