#!/bin/bash
#SBATCH --job-name=ft_plots
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/mmfs1/home/harine/RoboMonkey/data/viz_q/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# For a FINETUNED DQC variant, rescore the cached offline candidates with the
# finetuned Q at one or more intermediate checkpoints, then make the SAME
# advantage-classification (plot_advantage_investigation) and AUC/diagnostics
# (plot_offline_diagnostics) plots as the unfinetuned/base run -> directly
# comparable to data/viz_q/_adv/advantage_base_* and data/viz/_diag/*.
#
#   sbatch --account=weirdlab-ckpt --partition=ckpt-all --qos=ckpt-gpu \
#     --export=ALL,VARIANT=v2_all,EPOCHS="5000 30000" scriptsv2/q_fn_eval/ft_plots.sbatch
# -----------------------------------------------------------------------------
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"; REPO="$PWD"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate dqc_jax
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/dqc_jax/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO/diffusion_policy:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

VARIANT="${VARIANT:-v2_all}"
GS=/gscratch/robotics/harine/RoboMonkey_data/viz_q/ft/dqc_finetune
case "$VARIANT" in
  v1_ramp) CKPTDIR="${CKPTDIR:-$GS/v1_ramp/v1_ramp}" ;;
  v2_all)  CKPTDIR="${CKPTDIR:-$GS/v2_all/v2_all}" ;;
  v3_dist) CKPTDIR="${CKPTDIR:-$GS/v3_dist/v3_dist_b256_a100}" ;;
  *) CKPTDIR="${CKPTDIR:?set CKPTDIR}" ;;
esac
EPOCHS="${EPOCHS:-5000 30000}"
CONFIGS="${CONFIGS:-bc_h8 diff10k diff33k gauss20k gauss50k}"
ADV_OUT="${ADV_OUT:-data/viz_q/_adv_ft}"
DIAG_OUT="${DIAG_OUT:-data/viz_q/_diag_ft}"
RESC_ROOT="${RESC_ROOT:-data/viz_q/rescore}"
INSTR="put the eggplant in the basket"
TEMB="$REPO/scriptsv2/q_fn_eval/task_emb_use3.npy"
mkdir -p "$ADV_OUT" "$DIAG_OUT"

for EP in $EPOCHS; do
  [[ -f "$CKPTDIR/params_${EP}.pkl" ]] || { echo "[ftplots] MISSING $CKPTDIR/params_${EP}.pkl — skip"; continue; }
  TAG="${VARIANT}_s${EP}"
  echo "================ $TAG (ckpt $CKPTDIR @ $EP) ================"
  for split in train val; do
    RUNARGS=()
    for cfg in $CONFIGS; do
      SRC="data/viz/$cfg/$split/search_q"
      OUT="$RESC_ROOT/$TAG/$cfg/$split/search_q"
      [[ -d "$SRC" ]] || { echo "  skip missing $SRC"; continue; }
      n_src=$(ls "$SRC"/ep*.npz 2>/dev/null | wc -l)
      n_out=$(ls "$OUT"/ep*.npz 2>/dev/null | wc -l)
      if [[ "$n_out" -lt "$n_src" || "$n_src" -eq 0 ]]; then
        echo "  rescore $cfg/$split ($n_src eps) -> $OUT"
        python scriptsv2/q_fn_eval/rescore_offline_candidates.py \
          --src "$SRC" --out "$OUT" \
          --ckpt-dir "$CKPTDIR" --restore-epoch "$EP" \
          --instruction "$INSTR" --text-emb "$TEMB" --device cuda:0 \
          || { echo "  rescore FAILED $cfg/$split"; continue; }
      else
        echo "  reuse $cfg/$split ($n_out rescored npz)"
      fi
      RUNARGS+=(--run "$cfg:$RESC_ROOT/$TAG/$cfg/$split")
    done
    [[ ${#RUNARGS[@]} -eq 0 ]] && { echo "  no runs for $split"; continue; }
    echo "  -> advantage plot ($split, tag $TAG)"
    python scriptsv2/q_fn_eval/plot_advantage_investigation.py \
      --split "$split" --out "$ADV_OUT" "${RUNARGS[@]}" \
      --q-key values --aq-key a_star_q --tag "$TAG" \
      || echo "  advantage plot FAILED $split"
    echo "  -> AUC/diagnostics plot ($split, tag $TAG)"
    python scriptsv2/q_fn_eval/plot_offline_diagnostics.py \
      --split "$split" --out "$DIAG_OUT/$TAG" "${RUNARGS[@]}" \
      || echo "  diagnostics plot FAILED $split"
  done
done
echo "#### DONE ####"
echo "advantage  -> $ADV_OUT/advantage_${VARIANT}_s*_{train,val}.png  (base: data/viz_q/_adv/advantage_base_*)"
echo "auc/diag   -> $DIAG_OUT/<tag>/astar_rank_quality_*.png         (base: data/viz/_diag/astar_rank_quality_*)"
ls -1 "$ADV_OUT"/advantage_${VARIANT}_*.png 2>/dev/null
ls -1 "$DIAG_OUT"/*/astar_rank_quality_*.png 2>/dev/null
