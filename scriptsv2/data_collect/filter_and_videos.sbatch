#!/bin/bash
#SBATCH --job-name=filt_vid
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/gscratch/robotics/harine/data/_collect_logs/filtvid-%j.out
#SBATCH --error=/gscratch/robotics/harine/data/_collect_logs/filtvid-%j.out
#
# Runs the STRICT success filter for a scene and then saves N videos of the kept
# episodes into data/eval. CPU-only work (reads zarr, numpy, mp4 encode); it takes
# a small GPU slice only because harine has no CPU-only Slurm partition. Chain it
# after a scene's collection shards so it fires automatically when they finish:
#
#   sbatch --dependency=afterany:<shard0>:<shard1> \
#     --export=ALL,SCENE=stack,N=20 scriptsv2/data_collect/filter_and_videos.sbatch
#
set -uo pipefail
: "${SCENE:?set SCENE (stack|spoon|carrot|eggplant)}"
N="${N:-20}"
OUT_DIR="${OUT_DIR:-/mmfs1/home/harine/RoboMonkey/data/eval}"

cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
echo "[filtvid] host=$(hostname)  SCENE=$SCENE  N=$N  OUT_DIR=$OUT_DIR"

CONDA_BASE="${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}"
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null)}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate simpler_env
python -c "import zarr, numpy, imageio.v2" 2>/dev/null || pip install zarr numcodecs imageio imageio-ffmpeg

echo "[filtvid] ===== strict filter: $SCENE ====="
python scriptsv2/data_collect/filter_strict_success.py --scene "$SCENE"
rc=$?
if [ $rc -ne 0 ]; then
    echo "[filtvid] filter exited $rc (likely nothing kept / data missing); no videos."
    exit $rc
fi

echo "[filtvid] ===== save $N videos of kept $SCENE episodes ====="
python scriptsv2/data_collect/save_kept_videos.py --scene "$SCENE" --n "$N" --out-dir "$OUT_DIR"
echo "[filtvid] done."
