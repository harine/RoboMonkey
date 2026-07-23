#!/bin/bash
#SBATCH --job-name=filt_vcomp
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mmfs1/home/harine/RoboMonkey/data/eval/filtered_success_eggplant/verifier_comp/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# How well do DQC chunk-8 vs DQC chunk-1 vs LLaVA separate the strict-filtered
# eggplant successes from the other (messy) successes and the failures?
#
# 3 stages, one GPU, sequential (separate processes -> no JAX/torch co-location):
#   1) filtered_sample_pairs.py  (dqc_jax)        -> pairs.npz   (3-group tagged)
#   2) filtered_score_dqc.py     (dqc_jax)        -> dqc_scores_filtered.csv
#   3) eval_robomonkey.py        (monkey-verifier)-> verifier_scores.npz (LLaVA)
#      filtered_plot_comp.py     (monkey-verifier)-> verifier_comp.{png,md,csv}
#
#   sbatch scriptsv2/q_fn_eval/run_filtered_verifier_comp.sbatch
# Knobs (env): MAX_EPS_PER_GROUP (80), STRIDE (4), SEED (42),
#              DO_SAMPLE/DO_DQC/DO_LLAVA/DO_PLOT (1), OUT_DIR.
# -----------------------------------------------------------------------------
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
REPO="$PWD"

OUT_DIR="${OUT_DIR:-$REPO/data/eval/filtered_success_eggplant/verifier_comp}"
MAX_EPS_PER_GROUP="${MAX_EPS_PER_GROUP:-80}"
STRIDE="${STRIDE:-4}"
SEED="${SEED:-42}"
INSTRUCTION="${INSTRUCTION:-put the eggplant in the basket}"
DATA_DIR="${DATA_DIR:-/gscratch/robotics/harine/data/eggplant_in_basket_complete}"
MANIFEST="${MANIFEST:-/gscratch/robotics/harine/data/strict_distance_success/filter_manifest.json}"
DQC_CKPT="${DQC_CKPT:-/gscratch/robotics/harine/dqc_checkpoint/sd000s_137913.0.20260605_163017}"
DO_SAMPLE="${DO_SAMPLE:-1}"; DO_DQC="${DO_DQC:-1}"; DO_LLAVA="${DO_LLAVA:-1}"; DO_PLOT="${DO_PLOT:-1}"
mkdir -p "$OUT_DIR"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
echo "============================================================"
echo " filtered verifier comp | out=$OUT_DIR"
echo "   max_eps_per_group=$MAX_EPS_PER_GROUP stride=$STRIDE seed=$SEED"
echo "   data=$DATA_DIR  manifest=$MANIFEST  dqc_ckpt=$DQC_CKPT"
echo "============================================================"

# ============================ STAGE 1+2: dqc_jax ========================== #
if [[ "$DO_SAMPLE" == "1" || "$DO_DQC" == "1" ]]; then
  conda activate dqc_jax
  export LD_LIBRARY_PATH="$HOME/miniconda3/envs/dqc_jax/lib:${LD_LIBRARY_PATH:-}"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.6}"
  export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"

  if [[ "$DO_SAMPLE" == "1" ]]; then
    echo "#### STAGE 1: sample 3-group pairs ####"
    python scriptsv2/q_fn_eval/filtered_sample_pairs.py \
      --data-dir "$DATA_DIR" --manifest "$MANIFEST" --out-dir "$OUT_DIR" \
      --instruction "$INSTRUCTION" --max-eps-per-group "$MAX_EPS_PER_GROUP" \
      --stride "$STRIDE" --seed "$SEED" \
      || { echo "[fvc] STAGE 1 sample FAILED"; exit 1; }
  fi

  if [[ "$DO_DQC" == "1" ]]; then
    echo "#### STAGE 2: DQC critics (chunk-8 / chunk-1 / action) ####"
    python scriptsv2/q_fn_eval/filtered_score_dqc.py \
      --out-dir "$OUT_DIR" --ckpt-dir "$DQC_CKPT" --instruction "$INSTRUCTION" \
      || { echo "[fvc] STAGE 2 DQC FAILED"; exit 1; }
  fi
fi

# ============================ STAGE 3: monkey-verifier ==================== #
if [[ "$DO_LLAVA" == "1" || "$DO_PLOT" == "1" ]]; then
  conda activate monkey-verifier
  export LD_LIBRARY_PATH="$HOME/miniconda3/envs/monkey-verifier/lib:${LD_LIBRARY_PATH:-}"

  if [[ "$DO_LLAVA" == "1" ]]; then
    echo "#### STAGE 3a: LLaVA / RoboMonkey verifier ####"
    python scriptsv2/q_fn_eval/eval_robomonkey.py \
      --out-dir "$OUT_DIR" --instruction "$INSTRUCTION" --reward-server-port 0 \
      || { echo "[fvc] STAGE 3a LLaVA FAILED"; exit 1; }
  fi

  if [[ "$DO_PLOT" == "1" ]]; then
    echo "#### STAGE 3b: 3-group comparison plot + AUC table ####"
    python scriptsv2/q_fn_eval/filtered_plot_comp.py --out-dir "$OUT_DIR" \
      || { echo "[fvc] STAGE 3b plot FAILED"; exit 1; }
  fi
fi

echo "[fvc] done -> $OUT_DIR"
ls -la "$OUT_DIR"
