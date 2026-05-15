#!/bin/bash
#SBATCH --job-name=collect_eggplant
#SBATCH --account=robotics
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/gscratch/robotics/harine/data/eggplant_in_basket/slurm-%j.out
#SBATCH --error=/gscratch/robotics/harine/data/eggplant_in_basket/slurm-%j.out
#
# Collect eggplant trajectories on an A100. Runs the OpenVLA server in the
# background, waits for health, then launches the collector. On exit (timeout
# or completion) both processes are killed cleanly so the zarr shard is closed.
#
# Submit one shard:
#   cd /mmfs1/home/harine/RoboMonkey
#   sbatch --export=ALL,START_INDEX=2094,NUM=2000,SHARD=state2.zarr \
#       scriptsv2/data_collect/collect_eggplant_a100.sbatch
#
# Chain follow-ups with --dependency=afterany so they queue back-to-back:
#   J=$(sbatch --parsable --export=ALL,START_INDEX=2094,NUM=2000,SHARD=state2.zarr scriptsv2/data_collect/collect_eggplant_a100.sbatch)
#   J=$(sbatch --parsable --dependency=afterany:$J --export=ALL,START_INDEX=4094,NUM=2000,SHARD=state3.zarr scriptsv2/data_collect/collect_eggplant_a100.sbatch)
#   J=$(sbatch --parsable --dependency=afterany:$J --export=ALL,START_INDEX=6094,NUM=2000,SHARD=state4.zarr scriptsv2/data_collect/collect_eggplant_a100.sbatch)
#       sbatch         --dependency=afterany:$J --export=ALL,START_INDEX=8094,NUM=1906,SHARD=state5.zarr scriptsv2/data_collect/collect_eggplant_a100.sbatch

set -euo pipefail

: "${START_INDEX:?set START_INDEX}"
: "${NUM:?set NUM}"
: "${SHARD:?set SHARD (e.g. state2.zarr)}"

export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"

echo "[sbatch] host=$(hostname)  gpu=$CUDA_VISIBLE_DEVICES"
echo "[sbatch] START_INDEX=$START_INDEX NUM=$NUM SHARD=$SHARD"
nvidia-smi -L || true

LOG_DIR=/gscratch/robotics/harine/data/eggplant_in_basket
mkdir -p "$LOG_DIR"
TS="$(date +%Y_%m_%d-%H_%M_%S)"
SERVER_LOG="$LOG_DIR/openvla_server-$SLURM_JOB_ID.log"
COLLECT_LOG="$LOG_DIR/collect-$TS-job$SLURM_JOB_ID.log"

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

# --- wait for server health (up to 10 min) ---
PORT="${ACTION_SERVER_PORT:-3200}"
for i in $(seq 1 120); do
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

# --- run collector ---
bash scripts/collect_eggplant_in_basket.sh "$START_INDEX" "$NUM" "$SHARD" \
    2>&1 | tee "$COLLECT_LOG"

echo "[sbatch] collector exited; cleanup will run via trap"
