#!/bin/bash
#SBATCH --job-name=evcell
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=18:00:00
#SBATCH --output=data/eval/search_nas_sweep/_summaries/cell-%x-%j.out
#SBATCH --error=data/eval/search_nas_sweep/_summaries/cell-%x-%j.out
#
# ONE search-policy eval cell (single N) on a single GPU, for parallel fan-out
# (one job per N) so a full N-sweep finishes fast. Softmax selection,
# executed-window verifier scoring. Set via env (sbatch --export=ALL,...):
#   CHECKPOINT   path to the search-policy latest.ckpt   (required)
#   N            n_samples for this cell                 (required)
#   OUT_DIR      output dir for this cell                (required)
#   SCORE_AGG    mean|sum|min over executed steps        (default sum)
#   N_ACTION_STEPS  executed steps/replan                (default 8)
#   NUM_EPISODES (default 50)  SOFTMAX_TEMP (1.0)  START_SEED (1000)  TASK
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
: "${CHECKPOINT:?set CHECKPOINT}"; : "${N:?set N}"; : "${OUT_DIR:?set OUT_DIR}"
mkdir -p "$(dirname "$OUT_DIR")" data/eval/search_nas_sweep/_summaries

echo "[cell] host=$(hostname) N=$N agg=${SCORE_AGG:-sum} nas=${N_ACTION_STEPS:-8} mode=softmax"
echo "[cell] ckpt=$CHECKPOINT"
echo "[cell] out=$OUT_DIR"
nvidia-smi -L || true

export MODEL_DIR="${MODEL_DIR:-$SLURM_SUBMIT_DIR/monkey-verifier/model_dir}"
export MONKEY_VERIFIER_SRC="${MONKEY_VERIFIER_SRC:-$SLURM_SUBMIT_DIR/monkey-verifier/src}"
export DIFFUSION_POLICY_ROOT="${DIFFUSION_POLICY_ROOT:-$SLURM_SUBMIT_DIR/diffusion_policy}"
export ROBOMONKEY_KV_PREFIX="${ROBOMONKEY_KV_PREFIX:-0}"
export ROBOMONKEY_IMAGE_FEAT_CACHE_SIZE="${ROBOMONKEY_IMAGE_FEAT_CACHE_SIZE:-512}"
export ROBOMONKEY_PROC_IMAGE_CACHE_SIZE="${ROBOMONKEY_PROC_IMAGE_CACHE_SIZE:-512}"

CONDA_ENV="${CONDA_ENV:-monkey-verifier}" DEVICE=cuda:0 \
  MODE=softmax SOFTMAX_TEMP="${SOFTMAX_TEMP:-1.0}" \
  SCORE_AGG="${SCORE_AGG:-sum}" SCORE_WINDOW=executed \
  SCORE_DISCOUNT="${SCORE_DISCOUNT:-}" \
  N_SAMPLES="$N" N_ACTION_STEPS="${N_ACTION_STEPS:-8}" \
  START_SEED="${START_SEED:-1000}" MAX_STEPS="${MAX_STEPS:-120}" USE_EMA=1 \
  TASK="${TASK:-widowx_put_eggplant_in_basket}" \
  VIZ_Q="${VIZ_Q:-0}" VIZ_Q_EPISODES="${VIZ_Q_EPISODES:-0}" SAVE_VIDEOS="${SAVE_VIDEOS:-0}" \
  bash scriptsv2/eval_search/eval_search.sh "$CHECKPOINT" "${NUM_EPISODES:-50}" "$OUT_DIR"

echo "[cell] done N=$N -> $OUT_DIR/eval_log.json"
