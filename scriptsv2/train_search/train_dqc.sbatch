#!/bin/bash
# NOTE: 50k steps at ~6.7 s/step ~= 95-100h, which exceeds the cse assoc 24h
# MaxWall. weirdlab->gpu-l40s has no wall-time cap, so default there.
#SBATCH --job-name=tr_dqc
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --time=144:00:00
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
#   - discounted chunk sum:  sum_t DISCOUNT^t * Q(frame, a_t)  over the 8 executed
#     steps (score_executed_chunk=true, score_agg=discounted_sum, DISCOUNT=0.99)
#   - 50k gradient steps, logged to wandb (l2sml/robomonkey_l2s)
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
MAX_GRADIENT_STEPS="${MAX_GRADIENT_STEPS:-50000}"
# LR schedule = polynomial (diffusers): linear warmup 0->1e-4 over LR_WARMUP,
# then polynomial decay (LR_POWER=1.0 -> linear) 1e-4 -> LR_END over the window
# ending at LR_DECAY_STEPS, then HOLD LR_END constant for the rest of training
# (polynomial natively returns lr_end past num_training_steps; cosine would have
# ramped back up past its endpoint, which is why polynomial is used here). The
# decay is DECOUPLED from num_epochs/total steps, so the run can stop or extend
# at any point past LR_DECAY_STEPS at the constant floor. Decay sized to the
# productive-learning window (prior 7k run kept improving under a near-zero LR).
LR_SCHEDULER="${LR_SCHEDULER:-polynomial}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-500}"
LR_DECAY_STEPS="${LR_DECAY_STEPS:-10000}"   # decay ends (floor reached) at step 10k
LR_END="${LR_END:-1e-5}"                     # floor = 0.1 * 1e-4
LR_POWER="${LR_POWER:-1.0}"                  # 1.0 = linear decay
# num_epochs now only sets the TOTAL step budget (LR no longer depends on it);
# 658 epochs * ~76 steps/epoch ~= 50k. Capped by MAX_GRADIENT_STEPS.
NUM_EPOCHS="${NUM_EPOCHS:-658}"
# Discount for the discounted chunk sum  sum_t DISCOUNT^t * Q(frame, a_t) over
# the executed action window. Free hyperparam (independent of the frozen DQC
# Q-function's own training gamma).
DISCOUNT="${DISCOUNT:-0.99}"
MAX_ACTIONS="${MAX_ACTIONS:-16}"
# We only ever train the no-noise (obs_noise_level=0.0) search policy now, so it
# is not encoded in the run / output names.
OBS_NOISE_LEVEL="${OBS_NOISE_LEVEL:-0.0}"
DATASET_DIR="${DATASET_DIR:-/gscratch/robotics/harine/data/eggplant_in_basket_complete}"
OUT_ROOT="${OUT_ROOT:-/gscratch/robotics/harine/RoboMonkey_data/train/dqc}"
# wandb: l2sml/dqc_search_policy. Run name carries only the descriptive tag +
# the knobs that vary (discount, step budget) — no date, no noise level.
LOG_PROJECT="${LOG_PROJECT:-dqc_search_policy}"
RUN_NAME="${RUN_NAME:-eggplant_diffusion_hyak_disc${DISCOUNT}_$((MAX_GRADIENT_STEPS/1000))k}"
OUTDIR="${OUTDIR:-${OUT_ROOT}/${RUN_NAME}}"

export DIFFUSION_POLICY_ROOT="${DIFFUSION_POLICY_ROOT:-$SLURM_SUBMIT_DIR/diffusion_policy}"
# Frozen DQC verifier (JAX) shares the GPU with the torch policy — do not let
# JAX preallocate the whole device.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}"
# Disable XLA command buffers (CUDA graphs). When JAX shares the GPU with torch,
# CUDA-graph instantiation intermittently dies mid-run with
# "Failed to instantiate CUDA graph: CUDA_ERROR_UNKNOWN" (killed 36151277 at
# epoch 28). Graphs are only a launch-overhead optimization the DQC scorer does
# not need, and the GPU is search-bound anyway, so the cost is negligible.
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"
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
echo "[tr_dqc] config=robomonkey_eggplant_search_dqc  obs_noise_level=$OBS_NOISE_LEVEL  discount=$DISCOUNT"
echo "[tr_dqc] max_actions=$MAX_ACTIONS  max_grad_steps=$MAX_GRADIENT_STEPS  num_epochs=$NUM_EPOCHS  out=$OUTDIR"
echo "[tr_dqc] lr=$LR_SCHEDULER warmup=$LR_WARMUP_STEPS decay_steps=$LR_DECAY_STEPS lr_end=$LR_END power=$LR_POWER"
echo "[tr_dqc] wandb=l2sml/${LOG_PROJECT}  run=${RUN_NAME}"

python diffusion_policy/workspace/train_mlp_image_workspace.py \
    --config-name=robomonkey_eggplant_search_dqc \
    policy.obs_noise_level="${OBS_NOISE_LEVEL}" \
    policy.max_actions="${MAX_ACTIONS}" \
    policy.verifier.discount="${DISCOUNT}" \
    training.device=cuda:0 \
    training.num_epochs="${NUM_EPOCHS}" \
    +training.max_gradient_steps="${MAX_GRADIENT_STEPS}" \
    training.lr_scheduler="${LR_SCHEDULER}" \
    training.lr_warmup_steps="${LR_WARMUP_STEPS}" \
    +training.lr_decay_steps="${LR_DECAY_STEPS}" \
    +training.lr_end="${LR_END}" \
    +training.lr_power="${LR_POWER}" \
    task.dataset.use_cache=true \
    task.dataset.dataset_dir="${DATASET_DIR}" \
    logging.project="${LOG_PROJECT}" \
    logging.name="${RUN_NAME}" \
    +logging.entity=l2sml \
    hydra.run.dir="${OUTDIR}"

echo "[tr_dqc] done -> ${OUTDIR}/checkpoints/latest.ckpt"
