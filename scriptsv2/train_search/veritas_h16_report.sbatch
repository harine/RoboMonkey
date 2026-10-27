#!/bin/bash
#SBATCH --job-name=report_veritas_h16
#SBATCH --account=ckpt-robotics
#SBATCH --partition=ckpt
#SBATCH --qos=ckpt
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=14-00:00:00
#SBATCH --output=/gscratch/robotics/harine/RoboMonkey_data/experiments/veritas_h16_20261002/logs/report-%j.out
#SBATCH --error=/gscratch/robotics/harine/RoboMonkey_data/experiments/veritas_h16_20261002/logs/report-%j.out
set -euo pipefail
cd /mmfs1/home/harine/RoboMonkey
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate dqc_jax
python -u scriptsv2/train_search/veritas_h16_report.py \
  /gscratch/robotics/harine/RoboMonkey_data/experiments/veritas_h16_20261002
