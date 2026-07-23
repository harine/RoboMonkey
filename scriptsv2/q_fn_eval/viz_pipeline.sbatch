#!/bin/bash
#SBATCH --job-name=viz_qa
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/mmfs1/home/harine/RoboMonkey/data/viz/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# Candidate-action + Q-value visualization for 5 policy configs:
#   bc_h8 (BC-H8 DDPM softmax), diff10k, diff33k (diffusion search),
#   gauss20k, gauss50k (gaussian search).  Q = DQC chunk verifier (the AUC scorer).
#
# OFFLINE (train/val zarrs):
#   sample_offline_candidates.py -> search_q/*.npz -> render_q.py (videos).
#   (a*-match diagnostics REMOVED — critic is judged by BoN success rate, not match to a*;
#    plot_offline_diagnostics.py archived in q_fn_eval/verifier_match_a_expert_eval/.)
# ONLINE (DO_ONLINE=1): same 10 seeds (1000-1009) rolled out in SimplerEnv via
#   eval_search.sh with VIZ_Q=1, then render_q.py. No a* online.
#
# Runs in dqc_jax (SimplerEnv sim + torch policy + JAX DQC verifier, one process).
# Dry run (1 episode, ~6 frames, one config, offline only):
#   sbatch --export=ALL,CONFIGS=diff33k,SPLITS=train,NUM_EPISODES=1,MAX_FRAMES=6,DO_PLOTS=0,DO_ONLINE=0 \
#          scriptsv2/q_fn_eval/viz_pipeline.sbatch
# Full run:
#   sbatch scriptsv2/q_fn_eval/viz_pipeline.sbatch
# -----------------------------------------------------------------------------
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
REPO="$PWD"

# ---- knobs ----
CONFIGS="${CONFIGS:-bc_h8 diff10k diff33k gauss20k gauss50k}"
SPLITS="${SPLITS:-train val}"
N_SAMPLES="${N_SAMPLES:-16}"
N_ACTION_STEPS="${N_ACTION_STEPS:-4}"
STRIDE="${STRIDE:-4}"
NUM_EPISODES="${NUM_EPISODES:-10}"
MAX_FRAMES="${MAX_FRAMES:-0}"          # 0 = all strided frames per episode
START_SEED="${START_SEED:-1000}"        # online eval seeds START_SEED..+NUM_EPISODES-1
DO_OFFLINE="${DO_OFFLINE:-1}"
DO_ONLINE="${DO_ONLINE:-0}"
DO_PLOTS="${DO_PLOTS:-1}"
VIDEO_FPS="${VIDEO_FPS:-3}"
OUT_ROOT="${OUT_ROOT:-data/viz}"

TRAIN_ZARR="${TRAIN_ZARR:-/gscratch/robotics/harine/data/eggplant_in_basket/state0.zarr}"
VAL_ZARR="${VAL_ZARR:-/gscratch/robotics/harine/data/eggplant_in_basket_val_success/val_success.zarr}"

# ---- config table: label -> "ckpt|kind|ema" ----
BC_CKPT="${BC_CKPT:-$(ls -td "$REPO"/diffusion_policy/data/outputs/2026.06.12/14.35.54_*robomonkey_eggplant_bc*/checkpoints/best.ckpt 2>/dev/null | head -1)}"
ECK=/gscratch/robotics/harine/RoboMonkey_data/eval_ckpts
GCK=/gscratch/robotics/harine/RoboMonkey_data/train/dqc_chunk/eggplant_gaussian_dqc_chunk8_hyak_predhorizon8_rollout4/checkpoints
declare -A CKPT KIND EMA
CKPT[bc_h8]="$BC_CKPT";                                          KIND[bc_h8]=bc;     EMA[bc_h8]=1
CKPT[diff10k]="$ECK/diffusion_dqc_h8_roll4_step_0010000_10k_pathfix.ckpt"; KIND[diff10k]=search; EMA[diff10k]=0
CKPT[diff33k]="$ECK/diffusion_dqc_h8_roll4_step33001_pathfix.ckpt";        KIND[diff33k]=search; EMA[diff33k]=0
CKPT[gauss20k]="$GCK/step_0020000.ckpt";                         KIND[gauss20k]=search; EMA[gauss20k]=0
CKPT[gauss50k]="$GCK/step_0050000.ckpt";                         KIND[gauss50k]=search; EMA[gauss50k]=0

zarr_for() { [[ "$1" == train ]] && echo "$TRAIN_ZARR" || echo "$VAL_ZARR"; }

# ---- env (mirror eval_bc_dqc_chunk.sbatch) ----
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-dqc_jax}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export DISPLAY=""
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/${CONDA_ENV:-dqc_jax}/lib:${LD_LIBRARY_PATH:-}"
export DIFFUSION_POLICY_ROOT="$REPO/diffusion_policy"
export PYTHONPATH="$DIFFUSION_POLICY_ROOT:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"
export DQC_CKPT_DIR="${DQC_CKPT_DIR:-/gscratch/robotics/harine/dqc_checkpoint/sd000s_137913.0.20260605_163017}"
export DQC_TEXT_EMB="${DQC_TEXT_EMB:-$REPO/scriptsv2/q_fn_eval/task_emb_use3.npy}"

echo "============================================================"
echo " viz pipeline | configs=[$CONFIGS] splits=[$SPLITS] N=$N_SAMPLES"
echo "   offline=$DO_OFFLINE online=$DO_ONLINE plots=$DO_PLOTS  out=$OUT_ROOT"
echo "   bc_ckpt=$BC_CKPT"
echo "============================================================"

# ============================ OFFLINE ====================================== #
if [[ "$DO_OFFLINE" == "1" ]]; then
  for cfg in $CONFIGS; do
    ck="${CKPT[$cfg]:-}"
    if [[ -z "$ck" || ! -e "$ck" ]]; then echo "[viz] SKIP $cfg: ckpt missing ($ck)"; continue; fi
    for split in $SPLITS; do
      zr="$(zarr_for "$split")"
      out="$OUT_ROOT/$cfg/$split"
      echo "#### OFFLINE $cfg/$split  ckpt=$ck  zarr=$zr ####"
      python scriptsv2/q_fn_eval/sample_offline_candidates.py \
        --ckpt "$ck" --policy-kind "${KIND[$cfg]}" --split "$split" \
        --config-label "$cfg" --zarr "$zr" --out-dir "$out" \
        --n-samples "$N_SAMPLES" --n-action-steps "$N_ACTION_STEPS" \
        --stride "$STRIDE" --num-episodes "$NUM_EPISODES" \
        --max-frames-per-ep "$MAX_FRAMES" --use-ema "${EMA[$cfg]}" \
        || { echo "[viz] sample FAILED for $cfg/$split"; continue; }
      python scriptsv2/eval_search/render_q.py "$out/search_q" --fps "$VIDEO_FPS" \
        || echo "[viz] render FAILED for $cfg/$split"
    done
  done
fi

# ============================ ONLINE (eval rollouts) ======================= #
if [[ "$DO_ONLINE" == "1" ]]; then
  for cfg in $CONFIGS; do
    ck="${CKPT[$cfg]:-}"
    if [[ -z "$ck" || ! -e "$ck" ]]; then echo "[viz] SKIP online $cfg: ckpt missing"; continue; fi
    out="$OUT_ROOT/$cfg/eval"
    echo "#### ONLINE $cfg  ckpt=$ck ####"
    if [[ "${KIND[$cfg]}" == bc ]]; then
      CONDA_ENV=dqc_jax DEVICE=cuda:0 \
        VERIFIER_TYPE=dqc DQC_CRITIC_TYPE=chunk BC_VERIFIER_SCORE=chunk_critic \
        BC_CHUNK_FROM_PRED=1 BC_SAMPLER=ddpm SCORE_WINDOW=last \
        MODE=softmax SOFTMAX_TEMP=1.0 USE_EMA=1 \
        N_SAMPLES="$N_SAMPLES" N_ACTION_STEPS="$N_ACTION_STEPS" \
        START_SEED="$START_SEED" MAX_STEPS="${MAX_STEPS:-120}" SAVE_VIDEOS=0 \
        VIZ_Q=1 VIZ_Q_EPISODES="$NUM_EPISODES" TASK=widowx_put_eggplant_in_basket \
        bash scriptsv2/eval_search/eval_search.sh "$ck" "$NUM_EPISODES" "$out" \
        || echo "[viz] online BC FAILED for $cfg"
    else
      CONDA_ENV=dqc_jax DEVICE=cuda:0 \
        MODE=softmax SOFTMAX_TEMP=1.0 SCORE_WINDOW=last USE_EMA="${EMA[$cfg]}" \
        N_SAMPLES="$N_SAMPLES" N_ACTION_STEPS="$N_ACTION_STEPS" \
        START_SEED="$START_SEED" MAX_STEPS="${MAX_STEPS:-120}" SAVE_VIDEOS=0 \
        VIZ_Q=1 VIZ_Q_EPISODES="$NUM_EPISODES" TASK=widowx_put_eggplant_in_basket \
        bash scriptsv2/eval_search/eval_search.sh "$ck" "$NUM_EPISODES" "$out" \
        || echo "[viz] online search FAILED for $cfg"
    fi
    python scriptsv2/eval_search/render_q.py "$out/search_q" --fps "$VIDEO_FPS" \
      || echo "[viz] render online FAILED for $cfg"
  done
fi

# ============================ DIAGNOSTIC PLOTS ============================= #
# Globs existing search_q dirs, so a plots-only job (DO_OFFLINE=0) can rebuild
# the faceted figures across all configs without re-sampling.
if [[ "$DO_PLOTS" == "1" ]]; then
  for split in $SPLITS; do
    runs=()
    for cfg in $CONFIGS; do
      d="$OUT_ROOT/$cfg/$split/search_q"
      [[ -d "$d" ]] && runs+=( --run "$cfg:$OUT_ROOT/$cfg/$split" )
    done
    if [[ ${#runs[@]} -gt 0 ]]; then
      # a*-match diagnostics (Q-vs-RMSE-to-a*, a*-rank/quality, ranking-correctness) REMOVED —
      # the critic is judged by BoN SUCCESS RATE in sim, not by match to the expert action a*.
      # Archived: scriptsv2/q_fn_eval/verifier_match_a_expert_eval/plot_offline_diagnostics.py.
      # This pipeline now only SAMPLES the offline candidates (above) + renders Q videos.
      echo "#### $split: ${#runs[@]} runs sampled (no a*-match plots; eval critic via BoN success rate) ####"
    fi
  done
fi

echo "[viz] done -> $OUT_ROOT"
