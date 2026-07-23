#!/bin/bash
#SBATCH --job-name=subsample
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/gscratch/robotics/harine/data/_collect_logs/subsample-%j.out
#SBATCH --error=/gscratch/robotics/harine/data/_collect_logs/subsample-%j.out
#
# One diverse-start subsample per job (CPU/IO-only; takes an unused GPU slice
# because harine has no CPU-only Slurm partition). Submit N of these in parallel.
#   sbatch --export=ALL,IN_DIR=/gscratch/robotics/harine/data/stack_cube_strict,IN_ZARR=stack_strict.zarr,N=100 \
#       scriptsv2/data_collect/subsample.sbatch
set -uo pipefail
: "${IN_DIR:?set IN_DIR}"; : "${IN_ZARR:?set IN_ZARR}"; : "${N:?set N}"
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
CONDA_BASE="${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}"
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null)}"
source "$CONDA_BASE/etc/profile.d/conda.sh"; conda activate simpler_env
python -c "import zarr, numpy" 2>/dev/null || pip install zarr numcodecs
export BLOSC_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "[subsample] host=$(hostname) IN_DIR=$IN_DIR IN_ZARR=$IN_ZARR N=$N"
python scriptsv2/data_collect/subsample_diverse_start.py \
    --in-dir "$IN_DIR" --in-zarr "$IN_ZARR" --n "$N" --tag "sub${N}_diverse_start"
echo "[subsample] done."
