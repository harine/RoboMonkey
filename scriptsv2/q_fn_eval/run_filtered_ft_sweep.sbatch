#!/bin/bash
#SBATCH --job-name=filt_ftsweep
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/mmfs1/home/harine/RoboMonkey/data/eval/filtered_success_eggplant/verifier_comp/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# Sweep the FINETUNED DQC checkpoints (v4_filt_ramp) over the SAME 3-group pairs
# the base run scored, to see how the filtered-data finetune changes the
# discrimination of filtered_success / other_success / failure eggplant episodes.
#
# Reuses the already-built, checkpoint-INDEPENDENT artifacts in OUT_DIR:
#   pairs.npz          (3-group sampled frames)      <- symlinked into each step dir
#   verifier_scores.npz (LLaVA baseline)             <- flat reference in progression
# and the base DQC scores (OUT_DIR/dqc_scores_filtered.csv) as the step-0 reference.
#
# Per finetune step S (auto-detected from params_<S>.pkl in FT_DIR, or set STEPS):
#   1) filtered_score_dqc.py --ckpt-dir FT_DIR --restore-epoch S  (dqc_jax)
#        -> OUT_DIR/ft/step_S/dqc_scores_filtered.csv
#   2) filtered_plot_comp.py (per-step strip plot; PLOT env)      [optional, DO_STEP_PLOT]
# Then once:
#   3) filtered_ft_progression.py -> OUT_DIR/ft_progression.{csv,md,png}
#
# Idempotent: a step whose dqc_scores_filtered.csv already exists is SKIPPED
# (set FORCE=1 to rescore). Re-run after more checkpoints land to extend to 50k.
#
#   sbatch scriptsv2/q_fn_eval/run_filtered_ft_sweep.sbatch
# Knobs (env): FT_DIR, OUT_DIR, STEPS ("5000 10000 ..."; default = all >0 in FT_DIR),
#              DO_SCORE(1) DO_STEP_PLOT(1) DO_PROG(1) FORCE(0) PLOT_ENV(monkey-verifier).
# -----------------------------------------------------------------------------
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
REPO="$PWD"

OUT_DIR="${OUT_DIR:-$REPO/data/eval/filtered_success_eggplant/verifier_comp}"
FT_DIR="${FT_DIR:-/gscratch/robotics/harine/RoboMonkey_data/viz_q/ft/dqc_finetune/v4_filt_ramp/v4_filt_ramp_b256_a100}"
INSTRUCTION="${INSTRUCTION:-put the eggplant in the basket}"
DO_SCORE="${DO_SCORE:-1}"; DO_STEP_PLOT="${DO_STEP_PLOT:-1}"; DO_PROG="${DO_PROG:-1}"
FORCE="${FORCE:-0}"
PLOT_ENV="${PLOT_ENV:-monkey-verifier}"

[[ -f "$OUT_DIR/pairs.npz" ]] || { echo "[ftsweep] missing $OUT_DIR/pairs.npz — run run_filtered_verifier_comp.sbatch first"; exit 1; }

# Steps to score: explicit STEPS, else every params_<S>.pkl in FT_DIR with S>0.
if [[ -n "${STEPS:-}" ]]; then
  STEP_LIST="$STEPS"
else
  STEP_LIST=$(ls "$FT_DIR"/params_*.pkl 2>/dev/null | sed 's#.*/params_##;s#\.pkl$##' \
              | grep -E '^[0-9]+$' | awk '$1>0' | sort -n | tr '\n' ' ')
fi
[[ -n "$STEP_LIST" ]] || { echo "[ftsweep] no finetune checkpoints (params_<S>.pkl, S>0) in $FT_DIR"; exit 1; }

source "$HOME/miniconda3/etc/profile.d/conda.sh"
echo "============================================================"
echo " filtered FT sweep | out=$OUT_DIR"
echo "   ft_dir=$FT_DIR"
echo "   steps: $STEP_LIST   force=$FORCE"
echo "============================================================"

# =============================== STAGE 1: score (dqc_jax) ================== #
if [[ "$DO_SCORE" == "1" ]]; then
  conda activate dqc_jax
  export LD_LIBRARY_PATH="$HOME/miniconda3/envs/dqc_jax/lib:${LD_LIBRARY_PATH:-}"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.6}"
  export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"

  for S in $STEP_LIST; do
    [[ -f "$FT_DIR/params_${S}.pkl" ]] || { echo "[ftsweep] no params_${S}.pkl, skip"; continue; }
    SD="$OUT_DIR/ft/step_${S}"; mkdir -p "$SD"
    ln -sf ../../pairs.npz "$SD/pairs.npz"
    if [[ "$FORCE" != "1" && -f "$SD/dqc_scores_filtered.csv" ]]; then
      echo "#### step $S: already scored, skip (FORCE=1 to redo) ####"; continue
    fi
    echo "#### step $S: score finetuned DQC critics ####"
    python scriptsv2/q_fn_eval/filtered_score_dqc.py \
      --out-dir "$SD" --ckpt-dir "$FT_DIR" --restore-epoch "$S" --instruction "$INSTRUCTION" \
      || { echo "[ftsweep] step $S score FAILED"; }
  done
fi

# ===================== STAGE 2+3: plots (PLOT_ENV, has pandas/mpl) ========= #
if [[ "$DO_STEP_PLOT" == "1" || "$DO_PROG" == "1" ]]; then
  conda activate "$PLOT_ENV"
  export LD_LIBRARY_PATH="$HOME/miniconda3/envs/$PLOT_ENV/lib:${LD_LIBRARY_PATH:-}"

  if [[ "$DO_STEP_PLOT" == "1" ]]; then
    for S in $STEP_LIST; do
      SD="$OUT_DIR/ft/step_${S}"
      [[ -f "$SD/dqc_scores_filtered.csv" ]] || continue
      ln -sf ../../verifier_scores.npz "$SD/verifier_scores.npz" 2>/dev/null || true
      echo "#### step $S: per-step 3-group strip plot ####"
      python scriptsv2/q_fn_eval/filtered_plot_comp.py --out-dir "$SD" \
        || echo "[ftsweep] step $S per-step plot FAILED"
    done
  fi

  if [[ "$DO_PROG" == "1" ]]; then
    echo "#### progression across finetune steps (+ base @100k, + LLaVA ref) ####"
    python scriptsv2/q_fn_eval/filtered_ft_progression.py --out-dir "$OUT_DIR" \
      || { echo "[ftsweep] progression FAILED"; exit 1; }
  fi
fi

echo "[ftsweep] done -> $OUT_DIR (ft_progression.{png,md,csv}, ft/step_*/)"
ls -la "$OUT_DIR" "$OUT_DIR/ft" 2>/dev/null