#!/bin/bash
#SBATCH --job-name=smoke_ust
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/mmfs1/home/harine/RoboMonkey/data/train/_smoke_unified_st/slurm-%j.out
#
# Short training smoke after merging harine/pushT + harine/maze and moving the Simpler ST to
# the shared architecture (cond_encoder=gpt2, causal_attn=True). Each arm runs a few dozen
# gradient steps with val / sample / checkpoint cadences inside the window, wandb offline.
#   1. eggplant DQC ST, outer/inner trainer      (TrainSearchOuterInnerRoboMonkeyWorkspace)
#   2. eggplant distance ST, outer/inner trainer
#   3. eggplant DQC ST, offline trainer          (TrainMLPImageWorkspace)
#   4. eggplant BC UNet                          (TrainDiffusionUnetImageWorkspace)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
ROOT="$PWD"
OUT=/gscratch/robotics/harine/RoboMonkey_data/train/_smoke_unified_st
mkdir -p "$OUT" data/train/_smoke_unified_st
export WANDB_MODE=offline

COMMON=(DEBUG=False MAX_GRADIENT_STEPS=40 OUTER_BATCH_SIZE=64 INNER_BATCH_SIZE=32
        INNER_EPOCHS=2 DRIFT_EVERY=2 ROLLOUT_EVERY_STEPS=1000000 VAL_EVERY_STEPS=20
        SAMPLE_EVERY_STEPS=20 CHECKPOINT_EVERY=20 NRMSE_MAX_BATCHES=1 MAX_VAL_STEPS=2
        NUM_WORKERS=2 LOG_PROJECT=smoke_unified_st OUT_ROOT="$OUT")

ARMS="${ARMS:-oi_dqc,oi_distance,offline_dqc,bc_unet}"   # comma-separated subset
want() { [[ ",$ARMS," == *",$1,"* ]]; }
declare -A RC
want oi_dqc && env "${COMMON[@]}" RUN_NAME=oi_dqc bash scriptsv2/train_search/train_search_outer_inner.sbatch
RC[oi_dqc]=$?
want oi_distance && env "${COMMON[@]}" RUN_NAME=oi_distance bash scriptsv2/train_search/train_search_distance.sbatch
RC[oi_distance]=$?

source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate dqc_jax
export LD_LIBRARY_PATH="${HOME}/miniconda3/envs/dqc_jax/lib:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa DISPLAY=""
export PYTHONPATH="$ROOT/diffusion_policy:${PYTHONPATH:-}"
cd "$ROOT/diffusion_policy"
DATA=/gscratch/robotics/harine/data/eggplant_in_basket_complete

want offline_dqc && python diffusion_policy/workspace/train_mlp_image_workspace.py \
    --config-name=robomonkey_eggplant_search_dqc \
    policy.obs_noise_level=0.0 \
    policy.verifier.ckpt_dir=/gscratch/robotics/harine/dqc_checkpoint/sd000s_137913.0.20260605_163017 \
    policy.verifier.text_emb_file="$ROOT/scriptsv2/q_fn_eval/task_emb_use3.npy" \
    task.dataset.dataset_dir="$DATA" \
    +training.max_gradient_steps=40 training.checkpoint_every=20 \
    +training.val_every_steps=20 +training.sample_every_steps=20 \
    +training.rollout_every_steps=1000000 training.max_val_steps=2 \
    +training.nrmse_max_batches=1 dataloader.batch_size=32 val_dataloader.batch_size=32 \
    dataloader.num_workers=2 val_dataloader.num_workers=1 \
    logging.mode=offline hydra.run.dir="$OUT/offline_dqc"
RC[offline_dqc]=$?

want bc_unet && python diffusion_policy/workspace/train_diffusion_unet_image_workspace.py \
    --config-name=robomonkey_eggplant_bc \
    task.dataset.dataset_dir="$DATA" \
    +training.max_gradient_steps=40 training.checkpoint_every=20 \
    training.max_val_steps=2 dataloader.batch_size=32 val_dataloader.batch_size=32 \
    dataloader.num_workers=2 val_dataloader.num_workers=1 \
    logging.mode=offline hydra.run.dir="$OUT/bc_unet"
RC[bc_unet]=$?

echo "=================== SMOKE SUMMARY ==================="
for k in ${ARMS//,/ }; do echo "$k exit=${RC[$k]}"; done
