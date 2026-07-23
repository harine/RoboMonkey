#!/bin/bash
#SBATCH --job-name=offcand
#SBATCH --cpus-per-task=6
#SBATCH --mem=56G
#SBATCH --time=8:00:00
#SBATCH --output=/mmfs1/home/harine/RoboMonkey/data/eval/train_test_analysis/_logs/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# One offline candidate-sampling cell: dataset states -> N candidates -> Q -> a*.
# Driven entirely by --export vars (no commas in any value):
#   CKPT, ZARR, POLICY_KIND (search|bc), SPLIT (train|test), TASK, INSTRUCTION,
#   TEXT_EMB, N_SAMPLES, OUT_DIR, NUM_EPISODES, STRIDE, N_ACTION_STEPS
# Env mirrors viz_pipeline.sbatch (dqc_jax + osmesa; capture_camera builds a SimplerEnv).
# -----------------------------------------------------------------------------
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
REPO="$PWD"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-dqc_jax}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export DISPLAY=""
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/${CONDA_ENV:-dqc_jax}/lib:${LD_LIBRARY_PATH:-}"
export DIFFUSION_POLICY_ROOT="$REPO/diffusion_policy"
export PYTHONPATH="$DIFFUSION_POLICY_ROOT:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"
export DQC_CKPT_DIR="${DQC_CKPT_DIR:-/gscratch/robotics/harine/dqc_checkpoint/sd000s_137913.0.20260605_163017}"

echo "[offcand] $POLICY_KIND $SPLIT N=$N_SAMPLES task=$TASK"
echo "[offcand]   ckpt=$CKPT"
echo "[offcand]   zarr=$ZARR"
echo "[offcand]   text_emb=$TEXT_EMB   (used only for policy-kind=bc)"

python scriptsv2/q_fn_eval/sample_offline_candidates.py \
  --ckpt "$CKPT" \
  --zarr "$ZARR" \
  --policy-kind "$POLICY_KIND" \
  --split "$SPLIT" \
  --config-label "${POLICY_KIND}_${SPLIT}_N${N_SAMPLES}" \
  --out-dir "$OUT_DIR" \
  --n-samples "$N_SAMPLES" \
  --n-action-steps "${N_ACTION_STEPS:-4}" \
  --stride "${STRIDE:-4}" \
  --num-episodes "${NUM_EPISODES:-50}" \
  --task "$TASK" \
  --instruction "$INSTRUCTION" \
  --dqc-text-emb "$TEXT_EMB" \
  ${RAND_AUG_K:+--rand-aug-k "$RAND_AUG_K"}

echo "[offcand] done -> $OUT_DIR/search_q"
