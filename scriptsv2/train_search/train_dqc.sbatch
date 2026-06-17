#!/bin/bash
#SBATCH --job-name=tr_dqc
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=data/train/dqc/slurm-%j.out
#SBATCH --error=data/train/dqc/slurm-%j.out
#
# Train a NO-NOISE search policy (SearchPolicyRoboMonkeyDiffusion) using the
# frozen DQC Q-function verifier with a DISCOUNTED chunk sum.
#
#   - config robomonkey_eggplant_search_dqc (verifier = DQCVerifier, JAX)
#   - obs_noise_level=0.0      (no noise)
#   - horizon 16, n_action_steps 8 (execute 8 of 16)
#   - policy.max_actions=16    (search breadth / candidates)
#   - discounted chunk sum:  sum_t 0.98^t * Q(frame, a_t)  over the 8 executed
#     steps (score_executed_chunk=true, score_agg=discounted_sum, discount=0.98)
#   - runs in the dqc_jax conda env (cloned simpler_env + GPU jaxlib) so the
#     frozen DQC verifier scores candidates ON GPU during training.
#
# Submit:
#   sbatch scriptsv2/train_search/train_dqc.sbatch
#   # or override gpu:  sbatch --account=weirdlab --partition=gpu-l40 --gres=gpu:l40:1 ...
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
mkdir -p data/train/dqc

echo "[tr_dqc] host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi -L || true

CONDA_ENV="${CONDA_ENV:-dqc_jax}"
MAX_GRADIENT_STEPS="${MAX_GRADIENT_STEPS:-7000}"
MAX_ACTIONS="${MAX_ACTIONS:-16}"
OBS_NOISE_LEVEL="${OBS_NOISE_LEVEL:-0.0}"
DATASET_DIR="${DATASET_DIR:-/gscratch/robotics/harine/data/eggplant_in_basket_complete}"
OUT_ROOT="${OUT_ROOT:-/gscratch/robotics/harine/RoboMonkey_data/train/dqc}"
OUTDIR="${OUTDIR:-${OUT_ROOT}/noise${OBS_NOISE_LEVEL}}"

export DIFFUSION_POLICY_ROOT="${DIFFUSION_POLICY_ROOT:-$SLURM_SUBMIT_DIR/diffusion_policy}"
# Frozen DQC verifier (JAX) shares the GPU with the torch policy — do not let
# JAX preallocate the whole device.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}"
# SimplerEnv / SAPIEN (in case rollout_every triggers an eval rollout).
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
export DISPLAY=""

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="${HOME}/miniconda3/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${DIFFUSION_POLICY_ROOT}:${PYTHONPATH:-}"
cd "$DIFFUSION_POLICY_ROOT"

echo "[tr_dqc] env=$CONDA_ENV  jax devices:"
python -c "import jax; print('   ', jax.devices())" 2>&1 | grep -viE "warning|gpu may be present" | tail -2
echo "[tr_dqc] config=robomonkey_eggplant_search_dqc  obs_noise_level=$OBS_NOISE_LEVEL"
echo "[tr_dqc] max_actions=$MAX_ACTIONS  max_grad_steps=$MAX_GRADIENT_STEPS  out=$OUTDIR"

python diffusion_policy/workspace/train_mlp_image_workspace.py \
    --config-name=robomonkey_eggplant_search_dqc \
    policy.obs_noise_level="${OBS_NOISE_LEVEL}" \
    policy.max_actions="${MAX_ACTIONS}" \
    training.device=cuda:0 \
    +training.max_gradient_steps="${MAX_GRADIENT_STEPS}" \
    task.dataset.use_cache=true \
    task.dataset.dataset_dir="${DATASET_DIR}" \
    hydra.run.dir="${OUTDIR}"

echo "[tr_dqc] done -> ${OUTDIR}/checkpoints/latest.ckpt"
