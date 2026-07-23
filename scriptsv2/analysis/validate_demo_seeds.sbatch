#!/bin/bash
#SBATCH --job-name=seedgate
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/mmfs1/home/harine/RoboMonkey/data/eval/train_test_analysis/_seed_gate/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# SEED GATE: reset SimplerEnv at every reconstructed train/test demo seed and
# check the start pose against the dataset. Blocks the closed-loop videos.
#   sbatch --account=robotics --partition=ckpt-all --gres=gpu:a40:1 \
#          scriptsv2/analysis/validate_demo_seeds.sbatch
# Env mirrors viz_pipeline.sbatch (dqc_jax + osmesa software GL).
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

echo "[seedgate] regenerating seed lists (cheap, deterministic)"
python scriptsv2/analysis/reconstruct_demo_seeds.py --num "${NUM_DEMOS:-50}"

echo "[seedgate] validating against dataset start poses (tol=${TOL:-1e-4})"
python scriptsv2/analysis/validate_demo_seeds.py --tol "${TOL:-1e-4}"
rc=$?
echo "[seedgate] exit=$rc  (0 = PASS, 1 = FAIL)"
exit $rc
