#!/bin/bash
#SBATCH --job-name=ft_interim
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mmfs1/home/harine/RoboMonkey/data/viz_q/slurm-%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"; REPO="$PWD"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate dqc_jax
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/dqc_jax/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO/diffusion_policy:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export DQC_TASK_EMB="$REPO/scriptsv2/q_fn_eval/task_emb_use3.npy"
GS=/gscratch/robotics/harine/RoboMonkey_data/viz_q/ft/dqc_finetune
OUT=/gscratch/robotics/harine/RoboMonkey_data/viz_q/_prog_interim
for V in v1_ramp v2_all; do
  echo "#### interim progression $V (epochs 0 5000) ####"
  python scriptsv2/q_fn_eval/plot_ft_progression.py \
    --ft-run-dir "$GS/$V/$V" --label "${V}_a100_interim" --epochs 0 5000 \
    --src-root data/viz --out "$OUT" || echo "[interim] $V FAILED"
done
echo "#### interim CSVs ####"; cat $OUT/progression_*_interim.csv 2>/dev/null
