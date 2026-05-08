#!/bin/bash
# Collect N OpenVLA rollouts on `widowx_carrot_on_plate` into a single Zarr shard.
#
# Requires an already-running OpenVLA sglang server (see repo README):
#   conda activate sglang-vla && cd sglang-vla
#   CUDA_VISIBLE_DEVICES=0 python openvla_server.py --seed 1
#
# Usage:
#   bash scripts/collect_carrot_on_plate.sh                                   # 10k, shard state0
#   bash scripts/collect_carrot_on_plate.sh 0     5000  state0.zarr           # shard A
#   bash scripts/collect_carrot_on_plate.sh 5000  5000  state1.zarr           # shard B
#   OUT_DIR=data/carrot bash scripts/collect_carrot_on_plate.sh

set -e

START_INDEX=${1:-0}
NUM=${2:-10000}
SHARD=${3:-state0.zarr}
OUT_DIR=${OUT_DIR:-data/carrot_on_plate}
PORT=${ACTION_SERVER_PORT:-3200}
N_SAMPLES=${INITIAL_SAMPLES:-4}

# Activate the simpler_env conda env (same env used by run_simpler_eval.py)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate simpler_env

# One-time: install Zarr writer deps if missing.
python -c "import zarr, numcodecs" 2>/dev/null || pip install zarr numcodecs

export PRISMATIC_DATA_ROOT=.
export PYTHONPATH=.
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
cd openvla-mini

xvfb-run --auto-servernum -s "-screen 0 640x480x24" \
python experiments/robot/simpler/collect_trajectories.py \
  --task widowx_carrot_on_plate \
  --num_trajectories "$NUM" \
  --start_index "$START_INDEX" \
  --output_dir "$OUT_DIR" \
  --shard_name "$SHARD" \
  --action_server_port "$PORT" \
  --initial_samples "$N_SAMPLES" \
  --augmented_samples 1 \
  --save_videos False
