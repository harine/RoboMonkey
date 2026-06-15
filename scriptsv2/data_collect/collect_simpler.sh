#!/bin/bash
# Collect N RoboMonkey/OpenVLA rollouts on a SIMPLER WidowX task into one Zarr shard.
#
# Generalizes scripts/collect_{carrot_on_plate,eggplant_in_basket}.sh into a single
# task-parameterized collector: pick the task with TASK=, everything else is shared.
# Stores BOTH successful and failed episodes (the default) and, by default, the
# agentview RGB frames so the shard can be used for image-conditioned training.
#
# Requires an already-running OpenVLA sglang action server (see repo README, or use
# the companion collect_simpler.sbatch which launches and health-checks one for you):
#   conda activate sglang-vla && cd sglang-vla
#   CUDA_VISIBLE_DEVICES=0 python openvla_server.py --seed 1
#
# Usage (positional: START_INDEX NUM SHARD):
#   TASK=carrot   bash scriptsv2/data_collect/collect_simpler.sh                       # 10k, shard state0
#   TASK=carrot   bash scriptsv2/data_collect/collect_simpler.sh 0     5000 state0.zarr
#   TASK=eggplant bash scriptsv2/data_collect/collect_simpler.sh 5000  5000 state1.zarr
#   TASK=spoon ONLY_SUCCESSFUL=True bash scriptsv2/data_collect/collect_simpler.sh
#
# Knobs (env):
#   TASK                carrot | eggplant | spoon | stack  (or a raw widowx_* id)  [default carrot]
#   OUT_DIR             output dir for the zarr shard      [default /gscratch/robotics/harine/data/<task>]
#   SAVE_IMAGES         store agentview RGB in the zarr     [default True]
#   ONLY_SUCCESSFUL     keep only successful episodes       [default False -> keep success + fail]
#   INITIAL_SAMPLES     VLA samples/step for mean/std       [default 4]
#   ACTION_SERVER_PORT  sglang action server port           [default 3200]

set -e

# Ensure loopback traffic bypasses the cluster's HTTP proxy (e.g. Squid on Klone)
# so calls to the local action server at 127.0.0.1:$PORT aren't routed through it.
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

START_INDEX=${1:-0}
NUM=${2:-10000}
SHARD=${3:-state0.zarr}

# --- task alias -> SIMPLER env id + default data subdir --------------------------
TASK=${TASK:-carrot}
case "$TASK" in
  carrot|widowx_carrot_on_plate)          SIM_TASK=widowx_carrot_on_plate;        TASK_DIR=carrot_on_plate ;;
  eggplant|widowx_put_eggplant_in_basket) SIM_TASK=widowx_put_eggplant_in_basket; TASK_DIR=eggplant_in_basket ;;
  spoon|widowx_spoon_on_towel)            SIM_TASK=widowx_spoon_on_towel;          TASK_DIR=spoon_on_towel ;;
  stack|widowx_stack_cube)                SIM_TASK=widowx_stack_cube;              TASK_DIR=stack_cube ;;
  *) echo "[collect] unknown TASK='$TASK' (want carrot|eggplant|spoon|stack or a widowx_* id)"; exit 1 ;;
esac

OUT_DIR=${OUT_DIR:-/gscratch/robotics/harine/data/$TASK_DIR}
PORT=${ACTION_SERVER_PORT:-3200}
N_SAMPLES=${INITIAL_SAMPLES:-4}
SAVE_IMAGES=${SAVE_IMAGES:-True}
ONLY_SUCCESSFUL=${ONLY_SUCCESSFUL:-False}

# Repo root = two levels up from this script (scriptsv2/data_collect/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Activate the simpler_env conda env (same env used by run_simpler_eval.py).
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
cd "$REPO_ROOT/openvla-mini"

echo "[collect] task=$SIM_TASK  out_dir=$OUT_DIR  shard=$SHARD  start=$START_INDEX num=$NUM"
echo "[collect] save_images=$SAVE_IMAGES  only_successful=$ONLY_SUCCESSFUL  initial_samples=$N_SAMPLES  port=$PORT"

if command -v xvfb-run >/dev/null 2>&1; then
    RUNNER=(xvfb-run --auto-servernum -s "-screen 0 640x480x24")
else
    echo "[collect] xvfb-run not available; relying on osmesa software rendering."
    RUNNER=()
fi

"${RUNNER[@]}" \
python experiments/robot/simpler/collect_trajectories.py \
  --task "$SIM_TASK" \
  --num_trajectories "$NUM" \
  --start_index "$START_INDEX" \
  --output_dir "$OUT_DIR" \
  --shard_name "$SHARD" \
  --action_server_port "$PORT" \
  --initial_samples "$N_SAMPLES" \
  --augmented_samples 1 \
  --save_videos False \
  --save_images "$SAVE_IMAGES" \
  --only_successful "$ONLY_SUCCESSFUL"
