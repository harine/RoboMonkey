#!/bin/bash
#SBATCH --job-name=collect_simpler
#SBATCH --account=robotics
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/gscratch/robotics/harine/data/_collect_logs/slurm-%j.out
#SBATCH --error=/gscratch/robotics/harine/data/_collect_logs/slurm-%j.out
#
# General SIMPLER data-collection job for ANY WidowX task. Launches the OpenVLA
# sglang action server in the background, waits for it to become healthy, then
# runs the collector (collect_simpler.sh) which stores BOTH successes and
# failures plus the agentview RGB frames. On exit (timeout or completion) both
# processes are killed cleanly so the zarr shard is closed.
#
# Submit (carrot, one shard):
#   cd /mmfs1/home/harine/RoboMonkey
#   sbatch --export=ALL,TASK=carrot,START_INDEX=0,NUM=2000,SHARD=state0.zarr \
#       scriptsv2/data_collect/collect_simpler.sbatch
#
# Other tasks: TASK=eggplant | eggplant_wide | spoon | stack  (or a raw widowx_* id).
# Override the GPU pool at submit time, e.g. --partition=gpu-a40 --gres=gpu:a40:1.
#
# Chain shards back-to-back with --dependency=afterany so they queue in order:
#   J=$(sbatch --parsable --export=ALL,TASK=carrot,START_INDEX=0,NUM=2000,SHARD=state0.zarr     scriptsv2/data_collect/collect_simpler.sbatch)
#   J=$(sbatch --parsable --dependency=afterany:$J --export=ALL,TASK=carrot,START_INDEX=2000,NUM=2000,SHARD=state1.zarr scriptsv2/data_collect/collect_simpler.sbatch)
#   J=$(sbatch --parsable --dependency=afterany:$J --export=ALL,TASK=carrot,START_INDEX=4000,NUM=2000,SHARD=state2.zarr scriptsv2/data_collect/collect_simpler.sbatch)

set -euo pipefail

: "${TASK:?set TASK (carrot|eggplant|spoon|stack or a widowx_* id)}"
: "${START_INDEX:?set START_INDEX}"
: "${NUM:?set NUM}"
: "${SHARD:?set SHARD (e.g. state0.zarr)}"

export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"

# Resolve friendly task -> data subdir (mirrors collect_simpler.sh).
case "$TASK" in
  carrot|widowx_carrot_on_plate)          TASK_DIR=carrot_on_plate ;;
  eggplant|widowx_put_eggplant_in_basket) TASK_DIR=eggplant_in_basket ;;
  eggplant_wide|widowx_put_eggplant_in_basket_wide) TASK_DIR=eggplant_reset_core_x-0.22_-0.06_y0.02_0.26 ;;
  spoon|widowx_spoon_on_towel)            TASK_DIR=spoon_on_towel ;;
  stack|widowx_stack_cube)                TASK_DIR=stack_cube ;;
  *) echo "[sbatch] unknown TASK='$TASK'"; exit 1 ;;
esac
OUT_DIR=${OUT_DIR:-/gscratch/robotics/harine/data/$TASK_DIR}

echo "[sbatch] host=$(hostname)  gpu=$CUDA_VISIBLE_DEVICES"
echo "[sbatch] TASK=$TASK START_INDEX=$START_INDEX NUM=$NUM SHARD=$SHARD OUT_DIR=$OUT_DIR"
nvidia-smi -L || true

# flashinfer's JIT host-compile needs GCC>=9 (PyTorch C++17). klone's default is
# gcc 8.5.0, which makes sgl.Runtime crash at startup ("too old version of GCC").
# Load a modern gcc so nvcc's host compiler (PATH gcc/g++) satisfies it.
# Inherited by the setsid'd server subprocess.
source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
module load gcc/11.2.0 2>/dev/null || module load gcc/9.3.0 2>/dev/null || true
export CUDAHOSTCXX="$(command -v g++ || true)"
echo "[sbatch] gcc=$(gcc --version 2>/dev/null | head -1)  CUDAHOSTCXX=$CUDAHOSTCXX"

LOG_DIR="$OUT_DIR"
mkdir -p "$LOG_DIR"
TS="$(date +%Y_%m_%d-%H_%M_%S)"
SERVER_LOG="$LOG_DIR/openvla_server-$SLURM_JOB_ID.log"
COLLECT_LOG="$LOG_DIR/collect-$TASK_DIR-$TS-job$SLURM_JOB_ID.log"

# Cap sglang's static GPU-memory fraction so the co-located SAPIEN Vulkan renderer
# (~7.5 GiB on an L40) has headroom. sglang's ~0.83 default + the renderer fills a
# 44 GiB L40 and OOMs on long runs (a KV-cache growth tips it over). ~0.72 leaves
# ~5 GiB slack. Server reads VLA_MEM_FRACTION (sglang-vla/openvla_server.py).
export VLA_MEM_FRACTION="${VLA_MEM_FRACTION:-0.72}"
echo "[sbatch] VLA_MEM_FRACTION=$VLA_MEM_FRACTION"

# Stagger startup so co-located jobs (Slurm often packs a scene's shards onto one
# node) don't hammer the shared HF/model cache on NFS at the exact same instant.
STAGGER=$(( (SLURM_JOB_ID % 8) * 20 ))
echo "[sbatch] startup stagger: ${STAGGER}s"
sleep "$STAGGER"

# --- launch OpenVLA server in background ---
setsid bash scripts/run_openvla_server.sh > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[sbatch] openvla_server pid=$SERVER_PID  log=$SERVER_LOG"

cleanup() {
    echo "[sbatch] cleanup: killing server pgid $SERVER_PID"
    kill -TERM -- -$SERVER_PID 2>/dev/null || kill $SERVER_PID 2>/dev/null || true
    sleep 5
    kill -KILL -- -$SERVER_PID 2>/dev/null || true
}
trap cleanup EXIT TERM INT

# --- wait for server health (up to 30 min) ---
# OpenVLA-7B + sglang startup takes ~6 min alone and longer when two servers
# initialize concurrently on the same node (NFS model-cache I/O contention), so
# allow a generous window before giving up.
PORT="${ACTION_SERVER_PORT:-3200}"
for i in $(seq 1 360); do
    if curl -sS --max-time 3 "http://127.0.0.1:$PORT/get_model_info" >/dev/null 2>&1; then
        echo "[sbatch] server healthy after ${i}*5s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[sbatch] ERROR: server died before becoming healthy; see $SERVER_LOG"
        exit 1
    fi
    sleep 5
done
curl -sS --max-time 3 "http://127.0.0.1:$PORT/get_model_info" >/dev/null \
    || { echo "[sbatch] ERROR: server never became healthy"; exit 1; }

# --- run collector (success + fail, with images) ---
TASK="$TASK" OUT_DIR="$OUT_DIR" \
SAVE_IMAGES="${SAVE_IMAGES:-True}" ONLY_SUCCESSFUL="${ONLY_SUCCESSFUL:-False}" \
    bash scriptsv2/data_collect/collect_simpler.sh "$START_INDEX" "$NUM" "$SHARD" \
    2>&1 | tee "$COLLECT_LOG"

echo "[sbatch] collector exited; cleanup will run via trap"
