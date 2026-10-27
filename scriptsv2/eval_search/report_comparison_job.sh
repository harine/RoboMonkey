#!/bin/bash
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
if [ "$1" = replay ]; then conda activate dqc_jax; else conda activate simpler_env; fi
export SLURM_EXPORT_ENV=ALL
export PYTHONPATH="$PWD/diffusion_policy:${PYTHONPATH:-}"
export DIFFUSION_POLICY_ROOT="$PWD/diffusion_policy"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa DISPLAY=""
if [ "$1" = eval ]; then
 KEYF="$HOME/.config/gemini/api_key2"
 export GEMINI_API_KEY="$(tr -d '[:space:]' < "$KEYF")" GEMINI_KEY_GLOB="$KEYF"
fi
python -u scriptsv2/eval_search/report_comparison_campaign.py "$@" &
PID=$!
trap 'kill -TERM "$PID" 2>/dev/null || true; wait "$PID" || true; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
wait "$PID"
