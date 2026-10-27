#!/bin/bash
#SBATCH --job-name=eggplant_bc_denoiser
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=data/train/bc_denoiser/_logs/slurm-%x-%j.out
#SBATCH --error=data/train/bc_denoiser/_logs/slurm-%x-%j.out
#
# BC diffusion policy on the eggplant SUCCESS data (success_only=true,
# eggplant_in_basket_complete), with a selectable denoiser architecture, trained
# for a fixed number of gradient steps on a weirdlab L40.
#
#   - Step checkpoints every 1000 steps + latest.ckpt -> <run>/checkpoints/
#     (no best.ckpt: val_loss min is NOT the best task policy here, so pick the
#      deploy checkpoint by n=1 eval, not by validation loss)
#   - Resumable: fixed (non-timestamped) output dir; re-submitting the same
#     RUN_NAME resumes from latest.ckpt. To train longer, raise
#     MAX_GRADIENT_STEPS and re-submit the same job.
#   - LR schedule (cosine_then_constant): linear warmup -> cosine decay to a
#     small floor by LR_DECAY_STEPS -> constant floor for the remaining steps.
#
# Submit one run:
#   cd /mmfs1/home/harine/RoboMonkey
#   sbatch --export=ALL,DENOISER=mlp,HORIZON=16 scriptsv2/train_search/train_bc_denoiser.sbatch
#
# Or queue all four with: scriptsv2/train_search/submit_bc_denoiser_sweep.sh
#
# Env overrides:
#   DENOISER            mlp | transformer | unet     (default: mlp)
#                       (unet = standard 1D-conv U-Net denoiser, config
#                        robomonkey_eggplant_bc -> DiffusionUnetImagePolicy)
#   HORIZON             predicted steps; rollout = HORIZON/2  (default: 8)
#   MAX_GRADIENT_STEPS  total gradient steps         (default: 7000)
#   LR_DECAY_STEPS      step at which LR hits floor  (default: 5000)
#   LR_WARMUP_STEPS     linear warmup steps          (default: 500)
#   LR_MIN_RATIO        floor LR = base_lr * ratio   (default: 0.1 -> 1e-5)
#   RUN_NAME            output subdir name  (default: eggplant_bc_${DENOISER}_h${HORIZON})
#   DATASET_DIR         training data dir (Sim2RealImageMultiDataset, *.zarr shards;
#                       success_only filtering still applies)
#                       (default: /gscratch/robotics/harine/data/eggplant_in_basket_complete)
#   SPLIT_FILE          split manifest; its train/val lists replace the val_ratio split
#                       (scriptsv2/train_search/submit_bc_splits.sh sets it)
#   WANDB_ENTITY        (default: l2sml)
#   WANDB_PROJECT       (default: BC_eggplant)
#   WANDB_NAME          wandb run name (default: <arch>_pred${HORIZON}_roll${HORIZON/2})
#   CONDA_ENV           (default: monkey-verifier)

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
cd "$REPO_ROOT"
mkdir -p data/train/bc_denoiser/_logs

DENOISER="${DENOISER:-mlp}"
HORIZON="${HORIZON:-8}"
N_OBS_STEPS="${N_OBS_STEPS:-2}"   # obs-conditioning window (2=config default; 1 matches the search policy / clean DQC chunk)
MAX_GRADIENT_STEPS="${MAX_GRADIENT_STEPS:-7000}"
LR_DECAY_STEPS="${LR_DECAY_STEPS:-5000}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-500}"
LR_MIN_RATIO="${LR_MIN_RATIO:-0.1}"
RUN_NAME="${RUN_NAME:-eggplant_bc_${DENOISER}_h${HORIZON}}"
DATASET_DIR="${DATASET_DIR:-/gscratch/robotics/harine/data/eggplant_in_basket_complete}"
SPLIT_FILE="${SPLIT_FILE:-}"       # split manifest (config/splits/simpler/...): train/val episodes
CONDA_ENV="${CONDA_ENV:-monkey-verifier}"
WANDB_ENTITY="${WANDB_ENTITY:-l2sml}"
WANDB_PROJECT="${WANDB_PROJECT:-BC_eggplant}"
ROLLOUT=$((HORIZON / 2))

case "$DENOISER" in
  mlp)         CONFIG=robomonkey_eggplant_bc_mlp;         ARCH_LABEL=mlp_denoiser ;;
  transformer) CONFIG=robomonkey_eggplant_bc_transformer; ARCH_LABEL=encdec_denoiser ;;
  unet)        CONFIG=robomonkey_eggplant_bc;             ARCH_LABEL=unet_denoiser ;;
  *) echo "[sbatch] ERROR: DENOISER=$DENOISER (want mlp|transformer|unet)"; exit 1 ;;
esac

# wandb run name encodes architecture + predicted horizon + rollout length.
WANDB_NAME="${WANDB_NAME:-${ARCH_LABEL}_pred${HORIZON}_roll${ROLLOUT}}"

DP_ROOT="$REPO_ROOT/diffusion_policy"
# Absolute, fixed output dir so resume (latest.ckpt) is unambiguous.
RUN_DIR="$DP_ROOT/data/train/bc_denoiser/${RUN_NAME}"

echo "[sbatch] host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
echo "[sbatch] denoiser=$DENOISER horizon=$HORIZON (rollout=$((HORIZON/2))) n_obs_steps=$N_OBS_STEPS steps=$MAX_GRADIENT_STEPS"
echo "[sbatch] lr: warmup=$LR_WARMUP_STEPS decay_to_floor_by=$LR_DECAY_STEPS min_ratio=$LR_MIN_RATIO"
echo "[sbatch] run_dir=$RUN_DIR"
echo "[sbatch] dataset_dir=$DATASET_DIR split_file=${SPLIT_FILE:-<val_ratio>}"
echo "[sbatch] wandb: $WANDB_ENTITY/$WANDB_PROJECT name=$WANDB_NAME"
nvidia-smi -L || true

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONPATH="${DP_ROOT}:${PYTHONPATH:-}"
cd "$DP_ROOT"
mkdir -p "$RUN_DIR/checkpoints"

# num_epochs is set well above what the step cap needs; max_gradient_steps is
# what actually stops the run (and bounds the LR schedule's constant tail).
python diffusion_policy/workspace/train_diffusion_unet_image_workspace.py \
    --config-name="$CONFIG" \
    hydra.run.dir="$RUN_DIR" \
    horizon="$HORIZON" \
    n_obs_steps="$N_OBS_STEPS" \
    training.num_epochs=100000 \
    training.resume=true \
    training.lr_scheduler=cosine_then_constant \
    training.lr_warmup_steps="$LR_WARMUP_STEPS" \
    +training.lr_decay_steps="$LR_DECAY_STEPS" \
    +training.lr_min_ratio="$LR_MIN_RATIO" \
    +training.max_gradient_steps="$MAX_GRADIENT_STEPS" \
    task.dataset.use_cache=true \
    task.dataset.dataset_dir="$DATASET_DIR" \
    ${SPLIT_FILE:+task.dataset.split_file="$SPLIT_FILE"} \
    training.device=cuda:0 \
    +logging.entity="$WANDB_ENTITY" \
    logging.project="$WANDB_PROJECT" \
    logging.name="$WANDB_NAME" \
    logging.group="$ARCH_LABEL" \
    logging.id="$WANDB_NAME" \
    logging.resume=allow

echo "[sbatch] training done: $RUN_DIR"
echo "[sbatch] checkpoints (step_*/latest): $RUN_DIR/checkpoints/"
