#!/bin/bash
#SBATCH --job-name=ft_results
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/mmfs1/home/harine/RoboMonkey/data/viz_q/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# finetuned_dqc_june24 results: for ALL three finetuned variants at one step,
# rescore the cached offline candidates (passthrough frames -> videos work),
# then produce, comparable to the base/unfinetuned run:
#   * offline diagnostics (AUC / ranking)  -> data/viz_q/_diag_ft/<tag>/
#   * advantage classification             -> data/viz_q/_adv_ft/advantage_<tag>_*
#   * success-vs-fail verifier-Q over eps   -> data/viz_q/_sf/success_fail_q_<tag>_*
#   * re-rendered Q-overlay videos          -> rescore/<tag>/<cfg>/<split>/search_q/videos
#
#   sbatch --account=weirdlab-ckpt --partition=ckpt-all --qos=ckpt-gpu \
#     --export=ALL,EP=30000 scriptsv2/q_fn_eval/ft_results.sbatch
# -----------------------------------------------------------------------------
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"; REPO="$PWD"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate dqc_jax
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/dqc_jax/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO/diffusion_policy:$REPO/scriptsv2/q_fn_eval:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

EP="${EP:-30000}"
VARIANTS="${VARIANTS:-v1_ramp v2_all v3_dist}"
RENDER_VARIANT="${RENDER_VARIANT:-v2_all}"   # which variant's videos to re-render
CONFIGS="${CONFIGS:-bc_h8 diff10k diff33k gauss20k gauss50k}"
GS=/gscratch/robotics/harine/RoboMonkey_data/viz_q/ft/dqc_finetune
declare -A CKPT=( [v1_ramp]="$GS/v1_ramp/v1_ramp" [v2_all]="$GS/v2_all/v2_all" [v3_dist]="$GS/v3_dist/v3_dist_b256_a100" )
RESC=data/viz_q/rescore
SF=data/viz_q/_sf
INSTR="put the eggplant in the basket"
TEMB="$REPO/scriptsv2/q_fn_eval/task_emb_use3.npy"
mkdir -p "$SF"

for V in $VARIANTS; do
  CK="${CKPT[$V]}"
  if [[ ! -f "$CK/params_${EP}.pkl" ]]; then echo "[res] MISSING $CK/params_${EP}.pkl -> skip $V"; continue; fi
  TAG="${V}_s${EP}"
  echo "================================ $V @ $EP  ($CK) ================================"
  for split in train val; do
    RUNARGS=()
    for cfg in $CONFIGS; do
      SRC="data/viz/$cfg/$split/search_q"; OUT="$RESC/$TAG/$cfg/$split/search_q"
      [[ -d "$SRC" ]] || { echo "  skip missing $SRC"; continue; }
      echo "  rescore $V $cfg/$split -> $OUT"
      python scriptsv2/q_fn_eval/rescore_offline_candidates.py \
        --src "$SRC" --out "$OUT" --ckpt-dir "$CK" --restore-epoch "$EP" \
        --instruction "$INSTR" --text-emb "$TEMB" --device cuda:0 \
        || { echo "  RESCORE FAILED $cfg/$split"; continue; }
      RUNARGS+=(--run "$cfg:$RESC/$TAG/$cfg/$split")
    done
    [[ ${#RUNARGS[@]} -eq 0 ]] && continue
    # NOTE: a*-match eval (Q-vs-RMSE-to-a* diagnostics + advantage Q(a*)-Q(a_pred)) REMOVED —
    # the critic is judged by BoN SUCCESS RATE (launch_ft_verifier_bon.sh + build_bon_table.py),
    # not by match to the expert action. Archived: q_fn_eval/verifier_match_a_expert_eval/.
    # Rescored candidates above are kept only for the Q-overlay videos (render step below).
  done

  # success-vs-fail verifier-Q over episodes (TRAIN split has both outcome classes)
  SFRUNS=(); for cfg in $CONFIGS; do
    [[ -d "$RESC/$TAG/$cfg/train/search_q" ]] && SFRUNS+=(--run "$cfg:$RESC/$TAG/$cfg/train"); done
  [[ ${#SFRUNS[@]} -gt 0 ]] && python scriptsv2/q_fn_eval/plot_success_fail_q.py \
    --tag "$TAG" --out "$SF" "${SFRUNS[@]}" --source raw || echo "  sf-plot skipped"

  # re-render Q-overlay videos with the finetuned verifier (one chosen variant)
  if [[ "$V" == "$RENDER_VARIANT" ]]; then
    for cfg in $CONFIGS; do for split in train val; do
      d="$RESC/$TAG/$cfg/$split/search_q"
      [[ -d "$d" ]] && { echo "  render $cfg/$split"; python scriptsv2/eval_search/render_q.py "$d" --fps 3 || echo "  render FAILED $cfg/$split"; }
    done; done
  fi
done

echo "######## DONE (EP=$EP) ########"
echo "succ/fail:    $SF/success_fail_q_<tag>_*.png   (success-referenced, not a*-match)"
echo "videos:       $RESC/${RENDER_VARIANT}_s${EP}/<cfg>/<split>/search_q/videos/*.mp4"
echo "critic eval   -> BoN success rate: scriptsv2/eval_search/build_bon_table.py"
