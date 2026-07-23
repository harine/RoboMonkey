#!/bin/bash
#SBATCH --job-name=ckptcurve
#SBATCH --cpus-per-task=6
#SBATCH --mem=56G
#SBATCH --time=10:00:00
#SBATCH --output=/mmfs1/home/harine/RoboMonkey/data/eval/train_test_analysis/_logs/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# One shard of the best-of-N-RMSE-vs-training-step curve for a sub50 SEARCH policy.
# Driven by --export vars: CKPT_DIR, ZARR, TASK, INSTRUCTION, SPLIT, OUT, SHARD, NSHARDS
# (optional: N_SAMPLES, NUM_EPISODES, MIN_STEP, CKPT_STRIDE)
# -----------------------------------------------------------------------------
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
REPO="$PWD"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-dqc_jax}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}" DISPLAY=""
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/${CONDA_ENV:-dqc_jax}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO/diffusion_policy:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"

echo "[ckptcurve] $TASK $SPLIT shard $SHARD/$NSHARDS -> $OUT"
python scriptsv2/analysis/ckpt_rmse_curve.py \
  --ckpt-dir "$CKPT_DIR" --zarr "$ZARR" --task "$TASK" --instruction "$INSTRUCTION" \
  --split "$SPLIT" --out "$OUT" \
  --n-samples "${N_SAMPLES:-64}" --num-episodes "${NUM_EPISODES:-50}" \
  --min-step "${MIN_STEP:-5000}" --ckpt-stride "${CKPT_STRIDE:-5}" \
  --shard "$SHARD" --nshards "$NSHARDS"
echo "[ckptcurve] done -> $OUT"
