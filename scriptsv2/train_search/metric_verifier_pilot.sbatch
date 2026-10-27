#!/bin/bash
#SBATCH --account=ckpt-robotics
#SBATCH --partition=ckpt
#SBATCH --qos=ckpt-gpu
#SBATCH --gpus=1
#SBATCH --constraint=a40|l40s|l40|a100
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=/gscratch/robotics/harine/RoboMonkey_data/experiments/metric_verifier_pilot_20261004/logs/%x-%j.out
set -eo pipefail
cd /mmfs1/home/harine/RoboMonkey
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate simpler_env
export SLURM_EXPORT_ENV=ALL
export PYTHONPATH="$PWD/diffusion_policy:${PYTHONPATH:-}"
exec python -u scriptsv2/train_search/metric_verifier_pilot.py "$@"
