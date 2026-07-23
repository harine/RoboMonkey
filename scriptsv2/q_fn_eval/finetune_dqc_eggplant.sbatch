#!/bin/bash
#SBATCH --job-name=ft_dqc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mmfs1/home/harine/RoboMonkey/data/viz_q/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# Finetune the DQC chunk Q on eggplant (native loss) from the base ckpt, for ONE
# reward variant. Saves params_<step>.pkl every save_interval + flags.json +
# action_normalization.json into a predictable run dir (rescored later).
#
# VARIANT in {v1_ramp, v2_all, v3_dist, v4_filt_ramp}. Submit with a right-sized GPU
# job (1 GPU; cpus<=cores/GPU so other GPUs on the node keep CPUs):
#   sbatch --account=gpu-l40-weirdlab --partition=gpu-l40 --qos=normal \
#          --export=ALL,VARIANT=v2_all scriptsv2/q_fn_eval/finetune_dqc_eggplant.sbatch
#   (or scavenger: --account=weirdlab-ckpt --partition=ckpt --qos=ckpt-gpu)
# v4_filt_ramp = same 0->1 ramp reward as v1_ramp but on the TRIPLE-FILTERED
# strict_distance_success data (699 success ep) instead of the unfiltered set.
# -----------------------------------------------------------------------------
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
REPO="$PWD"
VARIANT="${VARIANT:?set VARIANT=v1_ramp|v2_all|v3_dist|v4_filt_ramp}"
DATAROOT=/gscratch/robotics/harine/data
case "$VARIANT" in
  v1_ramp)      EGG_DIR="$DATAROOT/eggplant_success_ramp_lerobot" ;;
  v2_all)       EGG_DIR="$DATAROOT/eggplant_all_lerobot" ;;
  v3_dist)      EGG_DIR="$DATAROOT/eggplant_success_dist_lerobot" ;;
  v4_filt_ramp) EGG_DIR="$DATAROOT/eggplant_filtered_ramp_lerobot" ;;
  *) echo "bad VARIANT=$VARIANT"; exit 2 ;;
esac
[[ -d "$EGG_DIR" ]] || { echo "[ft] missing dataset $EGG_DIR"; exit 2; }

OFFLINE_STEPS="${OFFLINE_STEPS:-50000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
BATCH="${BATCH:-256}"           # agent+data batch_size (256=base; smaller fits l40 + faster/step)
# LR schedule. Default LR_DECAY_STEPS=0 = CONSTANT 3e-4 (the base/v1-v4 behavior; also REQUIRED
# when RESUMING a constant-lr ckpt — a scheduled optax.adam has an extra opt_state count, so its
# pytree differs and restore would fail). For a FRESH future run, set LR_DECAY_STEPS>0 (e.g.
# =OFFLINE_STEPS) to LINEARLY decay 3e-4 -> LR_END over LR_DECAY_STEPS then HOLD at LR_END
# ("decay then stabilize at a low lr"); optional linear warmup 0->3e-4 over LR_WARMUP_STEPS.
LR_DECAY_STEPS="${LR_DECAY_STEPS:-0}"
LR_END="${LR_END:-3e-5}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
BASE_CKPT="${BASE_CKPT:-/gscratch/robotics/harine/dqc_checkpoint/sd000s_137913.0.20260605_163017}"
# Checkpoints are large (~GB each) -> save on /gscratch (HOME has a tight quota).
SAVE_ROOT="${SAVE_ROOT:-/gscratch/robotics/harine/RoboMonkey_data/viz_q/ft}"
PROG_OUT="${PROG_OUT:-/gscratch/robotics/harine/RoboMonkey_data/viz_q/_prog}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-dqc_jax}"
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/${CONDA_ENV:-dqc_jax}/lib:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"
# Online wandb -> l2sml/dqc_finetune (auth via ~/.netrc / WANDB_API_KEY).
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-l2sml}"
export WANDB_PROJECT="${WANDB_PROJECT:-dqc_finetune}"
# Tag the run name with the actual GPU (so a100 vs h200 vs l40 runs are distinguishable).
GPU_TAG=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 \
          | grep -oiE "h200|a100|l40s|l40|a40|rtx6k|2080ti|p100" | head -1 | tr 'A-Z' 'a-z')
GPU_TAG="${GPU_TAG:-gpu}"
# Deterministic run name -> deterministic save dir -> auto-resumable. Override WANDB_NAME
# to resume a run created under a different name (e.g. the pre-tag a100 dirs v1_ramp/v2_all).
export WANDB_NAME="${WANDB_NAME:-${VARIANT}_b${BATCH}_${GPU_TAG}}"
export EGG_LEROBOT_DIR="$EGG_DIR"
export EGG_NUM_WORKERS="${EGG_NUM_WORKERS:-16}"
export EGG_BATCH="$BATCH"
# Fixed-instruction MUSE/USE-3 embedding (tensorflow_text unavailable on-cluster);
# numerically identical to live MUSE of "put the eggplant in the basket".
export DQC_TASK_EMB="${DQC_TASK_EMB:-$REPO/scriptsv2/q_fn_eval/task_emb_use3.npy}"
# Match base DQC training: infinite_horizon_backup=true for ALL variants (the base
# was trained with it true even with soar_failure episodes).
export EGG_INF_HORIZON="${EGG_INF_HORIZON:-1}"

# ---- Resumable: deterministic run dir; auto-resume from the latest checkpoint ----
# Save dir is save_dir/<project>/<run_group>/<run.name> and run.name==WANDB_NAME.
RUNDIR="$SAVE_ROOT/$WANDB_PROJECT/$VARIANT/$WANDB_NAME"
LATEST=$(ls "$RUNDIR"/params_*.pkl 2>/dev/null | sed 's#.*/params_##;s#\.pkl$##' | grep -E '^[0-9]+$' | sort -n | tail -1)
if [[ -n "$LATEST" && "$LATEST" -gt 0 ]]; then
  RESTORE_DIR="$RUNDIR"; RESTORE_EPOCH="$LATEST"; START_STEP="$LATEST"
  echo "[ft] RESUME $VARIANT from step $LATEST (full state: params+opt+count) dir=$RUNDIR"
else
  RESTORE_DIR="$BASE_CKPT"; RESTORE_EPOCH="${BASE_EPOCH:-100000}"; START_STEP=0
  echo "[ft] FRESH $VARIANT from base $BASE_CKPT"
fi

echo "[ft] VARIANT=$VARIANT data=$EGG_DIR inf_horizon=$EGG_INF_HORIZON start=$START_STEP -> $OFFLINE_STEPS save=$SAVE_INTERVAL"
echo "[ft] run dir -> $RUNDIR"

cd "$REPO/dqc"
# Config matches the base DQC training (sd000s.../flags.json): agent + data
# batch_size=256, infinite_horizon_backup=true, num_workers=16, lr 3e-4, disc 0.98,
# backup_horizon 8, q_agg min. No agent override needed (data batch_size=256 == agent).
python experiments/train_bridge_soar_dqc.py \
  --agent agents/bridge_dqc_chunk.py \
  --agent.text_processor=precomputed \
  --agent.batch_size="$BATCH" \
  --agent.lr_decay_steps="$LR_DECAY_STEPS" \
  --agent.lr_end="$LR_END" \
  --agent.lr_warmup_steps="$LR_WARMUP_STEPS" \
  --data experiments/configs/eggplant_ft_data_config.py \
  --restore_path "$RESTORE_DIR" --restore_epoch "$RESTORE_EPOCH" --start_step "$START_STEP" \
  --offline_steps "$OFFLINE_STEPS" --save_interval "$SAVE_INTERVAL" --log_interval 1000 \
  --run_group "$VARIANT" --save_dir "$SAVE_ROOT" \
  --seed 0
train_rc=$?
if [[ $train_rc -ne 0 ]]; then
  echo "[ft] $VARIANT TRAINING FAILED (rc=$train_rc) — skipping progression"; exit $train_rc
fi
echo "[ft] $VARIANT training done -> $RUNDIR"

# The critic is judged ONLY by BoN SUCCESS RATE in sim (the SimplerEnv sweep:
# scriptsv2/eval_search/launch_ft_verifier_bon.sh + build_bon_table.py) — NOT by how well
# its Q matches the expert action a*. The per-checkpoint advantage / RMSE-to-a* / rank
# progression (plot_ft_progression.py) was the WRONG objective; it is archived under
# scriptsv2/q_fn_eval/verifier_match_a_expert_eval/. Do not re-add an a*-match step here.
