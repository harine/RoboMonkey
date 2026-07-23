#!/bin/bash
#SBATCH --job-name=recon_starts
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=data/eval/_verifier_comparison/_eval_starts/slurm-%x-%j.out
#SBATCH --error=data/eval/_verifier_comparison/_eval_starts/slurm-%x-%j.out
#
# Reconstruct eval-episode start positions (env.reset at the eval seeds) and
# regenerate the train-vs-eval start-position overlay with the full eval set.
# Submit (ckpt scavenger):
#   sbatch --account=ckpt-robotics --partition=ckpt --qos=ckpt-gpu \
#       --gres=gpu:1 --constraint='a40|l40s|l40|a100' \
#       scriptsv2/analysis/reconstruct_eval_starts.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
OUT=data/eval/_verifier_comparison/_eval_starts
mkdir -p "$OUT"

START_SEED="${START_SEED:-1000}"
NUM="${NUM:-50}"
VIZ="${VIZ:-8}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate simpler_env
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa DISPLAY=""
export LD_LIBRARY_PATH="${HOME}/miniconda3/envs/simpler_env/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/diffusion_policy:${PYTHONPATH:-}"
nvidia-smi -L || true

python scriptsv2/analysis/reconstruct_eval_starts.py \
    --start-seed "$START_SEED" --num "$NUM" --out "$OUT"

# Regenerate the overlay with the full eval set (matplotlib+zarr via monkey-verifier).
/gscratch/robotics/harine/miniconda3/envs/monkey-verifier/bin/python \
    scriptsv2/eval_search/plot_start_positions.py \
    --eval-dir "$OUT" --task eggplant --viz-episodes "$VIZ" \
    --out data/eval/_verifier_comparison/start_positions_train_vs_eval.png
echo "[recon] done -> start_positions_train_vs_eval.png"
