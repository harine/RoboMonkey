#!/bin/bash
#SBATCH --job-name=make_embs
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=data/train/pg/make_embs-%j.out
#SBATCH --error=data/train/pg/make_embs-%j.out
# Generate the stack_cube + spoon_on_towel MUSE (USE-3) task embeddings on a
# COMPUTE node -- TensorFlow spawns a large thread pool at init and core-dumps on
# the login node (pthread_create EAGAIN vs the per-user thread cap). A compute node
# has no such cap and has internet for the one-time TF-Hub MUSE download.
set -uo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate dqc_jax
cd /mmfs1/home/harine/RoboMonkey
export TFHUB_CACHE_DIR=/gscratch/robotics/harine/tfhub_cache
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=2 OMP_NUM_THREADS=2
python scriptsv2/q_fn_eval/make_task_embs.py
echo "[make_embs] exit=$?"
