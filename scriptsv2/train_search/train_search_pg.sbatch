#!/bin/bash
#SBATCH --job-name=tr_pg
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --output=data/train/pg/slurm-%x-%j.out
#SBATCH --error=data/train/pg/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# Train a search policy (gaussian OR diffusion) with the opt-in advantage
# POLICY-GRADIENT term added to BC:  loss = loss_BC + PG_LAMBDA * loss_PG,
#   loss_PG  ~  E[ grad_logpi(a_pi) * (Q(s,a_pi) - Q(s,a*)) ]
# REINFORCE on the policy's own sample a_pi; expert a* is the baseline; Q is the
# FROZEN DQC chunk critic (scalar coefficient, no backprop through Q). The PG term
# lives in SearchPolicy{RoboMonkey,RoboMonkeyDiffusion}.compute_loss (pg_lambda>0).
#
# Parameterized by env (4 runs = {gaussian,diffusion} x {base,v2 critic}):
#   CONFIG       robomonkey_eggplant_search_dqc            (DIFFUSION + DQC chunk)
#                robomonkey_eggplant_search_dqc_gaussian   (GAUSSIAN  + DQC chunk)
#   CRITIC_DIR   base: /gscratch/robotics/harine/dqc_checkpoint/sd000s_137913.0.20260605_163017
#                v2:   /gscratch/robotics/harine/RoboMonkey_data/viz_q/ft/dqc_finetune/v2_all/v2_all
#   RESTORE_EPOCH base 100000 | v2 50000   (finetuned run restarts numbering at 0)
#   PG_LAMBDA    advantage-term weight (default 0.1)
#   RUN_NAME / MAX_GRADIENT_STEPS (default 50000)
#
# LR schedule matches the previous DQC search-policy training (polynomial: warmup
# 500 -> 1e-4, linear decay to 1e-5 by 10k, then hold). Runs in dqc_jax (JAX critic
# shares the GPU with the torch policy). Submit to any weirdlab GPU (no wall cap).
# -----------------------------------------------------------------------------
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
mkdir -p data/train/pg

CONFIG="${CONFIG:-robomonkey_eggplant_search_dqc}"
CRITIC_DIR="${CRITIC_DIR:-/gscratch/robotics/harine/dqc_checkpoint/sd000s_137913.0.20260605_163017}"
RESTORE_EPOCH="${RESTORE_EPOCH:-100000}"
PG_LAMBDA="${PG_LAMBDA:-0.1}"
PG_MODE="${PG_MODE:-reinforce}"           # reinforce (REINFORCE on a_pi) | awbc (imitate expert a*)
TAGS="${TAGS:-}"                          # optional hydra list for logging.tags, e.g. [eggplant,pg_loss,awbc,l0.1]
MAX_GRADIENT_STEPS="${MAX_GRADIENT_STEPS:-50000}"
MAX_ACTIONS="${MAX_ACTIONS:-16}"
NUM_EPOCHS="${NUM_EPOCHS:-658}"            # sized so MAX_GRADIENT_STEPS caps it
DATASET_DIR="${DATASET_DIR:-/gscratch/robotics/harine/data/eggplant_in_basket_complete}"
TEXT_EMB="${TEXT_EMB:-/mmfs1/home/harine/RoboMonkey/scriptsv2/q_fn_eval/task_emb_use3.npy}"
# Instruction is cosmetic when TEXT_EMB resolves (the .npy wins in _resolve_text_emb);
# still overridden per task so the verifier health-check log names the right task.
INSTRUCTION="${INSTRUCTION:-put the eggplant in the basket}"
# LR schedule (identical to train_dqc.sbatch).
LR_SCHEDULER="${LR_SCHEDULER:-polynomial}"; LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-500}"
LR_DECAY_STEPS="${LR_DECAY_STEPS:-10000}";  LR_END="${LR_END:-1e-5}"; LR_POWER="${LR_POWER:-1.0}"
NUM_WORKERS="${NUM_WORKERS:-4}"            # data is cached -> few workers; fits any GPU's CPU/GPU ratio
LOG_PROJECT="${LOG_PROJECT:-dqc_search_policy_pg}"
RUN_NAME="${RUN_NAME:-eggplant_pg_${CONFIG#robomonkey_eggplant_search_}_lam${PG_LAMBDA}}"
OUT_ROOT="${OUT_ROOT:-/gscratch/robotics/harine/RoboMonkey_data/train/pg}"
OUTDIR="${OUTDIR:-${OUT_ROOT}/${RUN_NAME}}"
CONDA_ENV="${CONDA_ENV:-dqc_jax}"

export DIFFUSION_POLICY_ROOT="${DIFFUSION_POLICY_ROOT:-$SLURM_SUBMIT_DIR/diffusion_policy}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"
export DQC_RESTORE_EPOCH="${RESTORE_EPOCH}"     # which params_<epoch>.pkl the critic loads
export MUJOCO_GL=${MUJOCO_GL:-osmesa}; export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}; export DISPLAY=""

source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="${HOME}/miniconda3/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${DIFFUSION_POLICY_ROOT}:${PYTHONPATH:-}"
cd "$DIFFUSION_POLICY_ROOT"

echo "[tr_pg] host=$(hostname) config=$CONFIG  PG_LAMBDA=$PG_LAMBDA  PG_MODE=$PG_MODE  critic=$CRITIC_DIR (restore $RESTORE_EPOCH)"
echo "[tr_pg] max_grad_steps=$MAX_GRADIENT_STEPS  max_actions=$MAX_ACTIONS  tags=${TAGS:-<default>}  out=$OUTDIR"
python -c "import jax; print('   jax devices:', jax.devices())" 2>&1 | tail -1

EXTRA=()
[ -n "$TAGS" ] && EXTRA+=( "logging.tags=${TAGS}" )

python diffusion_policy/workspace/train_mlp_image_workspace.py \
    --config-name="${CONFIG}" \
    policy.obs_noise_level=0.0 \
    policy.max_actions="${MAX_ACTIONS}" \
    +policy.pg_lambda="${PG_LAMBDA}" \
    +policy.pg_mode="${PG_MODE}" \
    policy.verifier.ckpt_dir="${CRITIC_DIR}" \
    policy.verifier.text_emb_file="${TEXT_EMB}" \
    policy.verifier.instruction="${INSTRUCTION}" \
    training.device=cuda:0 \
    training.num_epochs="${NUM_EPOCHS}" \
    +training.max_gradient_steps="${MAX_GRADIENT_STEPS}" \
    training.lr_scheduler="${LR_SCHEDULER}" \
    training.lr_warmup_steps="${LR_WARMUP_STEPS}" \
    +training.lr_decay_steps="${LR_DECAY_STEPS}" \
    +training.lr_end="${LR_END}" \
    +training.lr_power="${LR_POWER}" \
    dataloader.num_workers="${NUM_WORKERS}" \
    val_dataloader.num_workers=2 \
    task.dataset.use_cache=true \
    task.dataset.dataset_dir="${DATASET_DIR}" \
    logging.project="${LOG_PROJECT}" \
    logging.name="${RUN_NAME}" \
    +logging.entity=l2sml \
    hydra.run.dir="${OUTDIR}" \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "[tr_pg] done -> ${OUTDIR}/checkpoints/latest.ckpt"
