#!/bin/bash
#SBATCH --job-name=dqc_gpu_val
#SBATCH --account=cse
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/mmfs1/home/harine/RoboMonkey/data/eval/dqc_chunk_check/gpu_val-%j.out
#SBATCH --error=/mmfs1/home/harine/RoboMonkey/data/eval/dqc_chunk_check/gpu_val-%j.out
set -e; cd /mmfs1/home/harine/RoboMonkey; mkdir -p data/eval/dqc_chunk_check
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate dqc_repo
export LD_LIBRARY_PATH="${HOME}/miniconda3/envs/dqc_repo/lib:${LD_LIBRARY_PATH:-}"
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
python scriptsv2/robomonkey_repro/_dqc_gpu_validate.py
