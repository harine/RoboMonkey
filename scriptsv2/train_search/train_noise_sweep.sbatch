#!/bin/bash
#SBATCH --job-name=noise_sweep
#SBATCH --account=robotics
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=13
#SBATCH --mem=160G
#SBATCH --time=72:00:00
#SBATCH --output=data/train/noise_sweep/slurm-%j.out
#SBATCH --error=data/train/noise_sweep/slurm-%j.out
#
# Train the noise-comparison search-policy checkpoints across the two free a40
# GPUs on the robotics node, in PARALLEL:
#
#   GPU0 (cuda:0): noise_conditioned (tmrl-style)  THEN  tunable_corruption@0.4
#   GPU1 (cuda:1): tunable_corruption@0.8           (concurrent with GPU0)
#
# tmrl starts first; 0.8 runs alongside it on the second GPU; 0.4 runs after
# tmrl on GPU0. Reserves 2 a40 + 13 CPU + 160G (robotics 2-GPU QOS cap).
#
# Submit:
#   cd /mmfs1/home/harine/RoboMonkey
#   sbatch scriptsv2/train_search/train_noise_sweep.sbatch
#
# Override env (passed through to train_noise_sweep.sh):
#   CONDA_ENV       (default: monkey-verifier)
#   EXTRA_OVERRIDES (default: max_gradient_steps=7000 + dataset cache on)
#   BATCH_SIZE      (default: unset = config default of 256)

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
mkdir -p data/train/noise_sweep

echo "[sbatch] host=$(hostname)  cuda_visible=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi -L || true

export CONDA_ENV="${CONDA_ENV:-monkey-verifier}"
# task.dataset.dataset_dir: the task config hardcodes /home/harine/data/... which
# does not exist on compute nodes. Point at a curated dir holding symlinks to only
# the COMPLETE shards (state0, state1) that contain images (agentview_image) and
# reward labels; the raw dir also has an empty/partial state2.zarr (obs-only, no
# actions) that the *.zarr glob would otherwise pick up and crash on.
# success_only=true is already set in the task config, so training uses successful
# (terminal reward > 0) trajectories only.
export EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-+training.max_gradient_steps=7000 task.dataset.use_cache=true task.dataset.dataset_dir=/gscratch/robotics/harine/data/eggplant_in_basket_complete}"
# BATCH_SIZE intentionally left unset -> config default (256).

TS="$(date +%Y_%m_%d-%H_%M_%S)"
SWEEP="scriptsv2/train_search/train_noise_sweep.sh"
LOG_DIR="data/train/noise_sweep"

# Pin each stream to ONE physical GPU via CUDA_VISIBLE_DEVICES, and use cuda:0
# inside both. The in-process LLaVA verifier always loads onto cuda:0 and ignores
# training.device, so without pinning BOTH verifiers pile onto the first GPU
# (~22G each -> OOM) while the second GPU sits idle. Pinning makes each process
# see only its own GPU as cuda:0, so verifier + policy land on the right device.
# Indices are taken from the job's allocation list (robust to physical remapping).
IFS=',' read -r DEV_A DEV_B _ <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
: "${DEV_B:=$DEV_A}"   # fall back to single GPU if only one allocated
echo "[sbatch] pinning GPU0 stream -> CUDA_VISIBLE_DEVICES=$DEV_A ; GPU1 stream -> $DEV_B"

# --- GPU1 stream: tunable_corruption @ 0.8 (background, concurrent) --------
(
    export CUDA_VISIBLE_DEVICES="$DEV_B"
    DEVICE=cuda:0 RUN_COND=0 LEVELS=0.8 \
        bash "$SWEEP"
) > "$LOG_DIR/gpu1-tcorrupt0.8-$TS-job$SLURM_JOB_ID.log" 2>&1 &
PID_GPU1=$!
echo "[sbatch] GPU1 stream (0.8) pid=$PID_GPU1 on phys GPU $DEV_B"

# --- GPU0 stream: tmrl (noise_conditioned) THEN tunable_corruption @ 0.4 ---
(
    export CUDA_VISIBLE_DEVICES="$DEV_A"
    DEVICE=cuda:0 RUN_COND=1 LEVELS= \
        bash "$SWEEP"
    DEVICE=cuda:0 RUN_COND=0 LEVELS=0.4 \
        bash "$SWEEP"
) > "$LOG_DIR/gpu0-tmrl-then-0.4-$TS-job$SLURM_JOB_ID.log" 2>&1 &
PID_GPU0=$!
echo "[sbatch] GPU0 stream (tmrl -> 0.4) pid=$PID_GPU0 on phys GPU $DEV_A"

# --- wait for both streams; fail the job if either fails -------------------
rc=0
wait "$PID_GPU0" || { echo "[sbatch] GPU0 stream FAILED (rc=$?)"; rc=1; }
wait "$PID_GPU1" || { echo "[sbatch] GPU1 stream FAILED (rc=$?)"; rc=1; }

echo "[sbatch] all streams done (rc=$rc); logs in $LOG_DIR/"
exit "$rc"
