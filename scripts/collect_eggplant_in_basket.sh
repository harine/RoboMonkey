#!/bin/bash
# Collect N OpenVLA rollouts on `widowx_put_eggplant_in_basket` into a single Zarr shard.
#
# Requires an already-running OpenVLA sglang server (see repo README):
#   conda activate sglang-vla && cd sglang-vla
#   CUDA_VISIBLE_DEVICES=0 python openvla_server.py --seed 1
#
# Usage:
#   bash scripts/collect_eggplant_in_basket.sh                                  # 10k, shard state0
#   bash scripts/collect_eggplant_in_basket.sh 0     5000  state0.zarr          # shard A
#   bash scripts/collect_eggplant_in_basket.sh 5000  5000  state1.zarr          # shard B
#   OUT_DIR=data/eggplant bash scripts/collect_eggplant_in_basket.sh

set -e

# Ensure loopback traffic bypasses the cluster's HTTP proxy (e.g. Squid on
# Klone) so calls to the local action server at 127.0.0.1:$PORT aren't
# routed through it. Outbound proxy is preserved.
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

START_INDEX=${1:-0}
NUM=${2:-10000}
SHARD=${3:-state0.zarr}
OUT_DIR=${OUT_DIR:-/gscratch/robotics/harine/data/eggplant_in_basket}
PORT=${ACTION_SERVER_PORT:-3200}
N_SAMPLES=${INITIAL_SAMPLES:-4}

# Activate the simpler_env conda env (same env used by run_simpler_eval.py)
CONDA_BASE="${CONDA_EXE:+$(dirname $(dirname "$CONDA_EXE"))}"
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null)}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate simpler_env

# One-time: install Zarr writer deps if missing.
python -c "import zarr, numcodecs" 2>/dev/null || pip install zarr numcodecs

export PRISMATIC_DATA_ROOT=.
export PYTHONPATH=.
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
cd openvla-mini

if command -v xvfb-run >/dev/null 2>&1; then
    RUNNER=(xvfb-run --auto-servernum -s "-screen 0 640x480x24")
else
    echo "[collect] xvfb-run not available; relying on osmesa software rendering."
    RUNNER=()
fi

"${RUNNER[@]}" \
python experiments/robot/simpler/collect_trajectories.py \
  --task widowx_put_eggplant_in_basket \
  --num_trajectories "$NUM" \
  --start_index "$START_INDEX" \
  --output_dir "$OUT_DIR" \
  --shard_name "$SHARD" \
  --action_server_port "$PORT" \
  --initial_samples "$N_SAMPLES" \
  --augmented_samples 1 \
  --save_videos False \
  --save_images "${SAVE_IMAGES:-True}"
