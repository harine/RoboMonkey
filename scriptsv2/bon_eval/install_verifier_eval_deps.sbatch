#!/bin/bash
#SBATCH --job-name=bon_install
#SBATCH --account=robotics
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=data/eval/bon/_summaries/install-%j.out
#SBATCH --error=data/eval/bon/_summaries/install-%j.out
#
# Install SimplerEnv + diffusion_policy deps into the monkey-verifier conda env.
# Submit ONCE before bon_eval_3seeds.sbatch (or chain via --dependency).
#
# Submit:
#   cd /mmfs1/home/harine/RoboMonkey
#   sbatch scriptsv2/bon_eval/install_verifier_eval_deps.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p data/eval/bon/_summaries
bash scripts/install_verifier_eval_deps.sh
