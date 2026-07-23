#!/bin/bash
#SBATCH --job-name=bc_dqcchunk
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/mmfs1/home/harine/RoboMonkey/data/eval/bc_dqc_chunk/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# Evaluate a plain BC diffusion policy (DiffusionUnetImagePolicy) with the DQC
# CHUNK verifier, mirroring the search eval: predict 8, rollout 4, score the full
# 8-step predicted chunk. For the H8 BC (best.ckpt, horizon 8, n_obs=2):
#   * VERIFIER_TYPE=dqc + DQC_CRITIC_TYPE=chunk -> eval_search.py wraps the BC with
#     BCSearchWrapper + a DQCVerifier(critic_type=chunk, score_full_horizon=True).
#   * BC_CHUNK_FROM_PRED=1 -> the wrapper scores the first chunk_h(=8) PREDICTED
#     actions as the chunk (decoupled from rollout), so rollout can be 4. (With
#     n_obs=2 the predicted chunk is ~1 step early vs a_{t:t+8}, but identical
#     across candidates -> BoN ranking is sound.)
#   * USE_EMA=1 (BC trained with use_ema=True).
#   * BC_SAMPLER selects ddim (trained, near-deterministic -> candidates collapse,
#     BoN ~no-op) or ddpm (100-step ancestral -> diverse candidates).
#   * MODE defaults to softmax (set MODE=argmax for best-of-N), seeds START_SEED..+NUM_EPISODES.
# Runs in dqc_jax (simpler_env sim + JAX[DQC]).
#   sbatch --export=ALL,CHECKPOINT=<bc ckpt>,BC_SAMPLER=ddpm,N_SAMPLES=4 eval_bc_dqc_chunk.sbatch
# -----------------------------------------------------------------------------
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
: "${CHECKPOINT:?set CHECKPOINT (BC diffusion policy)}"

BC_SAMPLER="${BC_SAMPLER:-ddpm}"
N_SAMPLES="${N_SAMPLES:-4}"
NUM_EPISODES="${NUM_EPISODES:-50}"
START_SEED="${START_SEED:-1000}"
N_ACTION_STEPS="${N_ACTION_STEPS:-4}"     # rollout 4 (predict 8)
OUT_DIR="${OUT_DIR:-data/eval/bc_dqc_chunk/bcH8_chunk8_predh8_roll4_${BC_SAMPLER}_n${N_SAMPLES}_seed${START_SEED}_ep${NUM_EPISODES}}"
mkdir -p data/eval/bc_dqc_chunk "$OUT_DIR"

export DIFFUSION_POLICY_ROOT="${DIFFUSION_POLICY_ROOT:-$SLURM_SUBMIT_DIR/diffusion_policy}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=} --xla_gpu_deterministic_ops=true"
# Determinism: cuBLAS needs a fixed workspace for reproducible GEMMs (set before CUDA init).
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# DQC chunk verifier (constructed by eval_search.py's BC path).
export VERIFIER_TYPE=dqc
export DQC_CRITIC_TYPE=chunk
export DQC_CKPT_DIR="${DQC_CKPT_DIR:-/gscratch/robotics/harine/dqc_checkpoint/sd000s_137913.0.20260605_163017}"
export DQC_TEXT_EMB="${DQC_TEXT_EMB:-$SLURM_SUBMIT_DIR/scriptsv2/q_fn_eval/task_emb_use3.npy}"
export BC_VERIFIER_SCORE=chunk_critic
export BC_CHUNK_FROM_PRED=1                # score full predicted 8-chunk, rollout<8 OK

echo "[bc_dqcchunk] host=$(hostname) ckpt=$CHECKPOINT"
echo "[bc_dqcchunk] sampler=$BC_SAMPLER N=$N_SAMPLES rollout=$N_ACTION_STEPS chunk_from_pred=1 mode=${MODE:-softmax} softmax_temp=${SOFTMAX_TEMP:-1.0}"
echo "[bc_dqcchunk] seeds=${START_SEED}..$((START_SEED+NUM_EPISODES-1)) episodes=$NUM_EPISODES -> $OUT_DIR"

_t0=$(date +%s)
CONDA_ENV="${CONDA_ENV:-dqc_jax}" DEVICE=cuda:0 \
  MODE="${MODE:-softmax}" SOFTMAX_TEMP="${SOFTMAX_TEMP:-1.0}" SCORE_WINDOW=last \
  BC_SAMPLER="$BC_SAMPLER" \
  N_SAMPLES="$N_SAMPLES" N_ACTION_STEPS="$N_ACTION_STEPS" \
  START_SEED="$START_SEED" MAX_STEPS="${MAX_STEPS:-120}" USE_EMA="${USE_EMA:-1}" \
  TASK="${TASK:-widowx_put_eggplant_in_basket}" \
  SAVE_VIDEOS="${SAVE_VIDEOS:-0}" VIDEO_FPS="${VIDEO_FPS:-10}" \
  VIZ_Q="${VIZ_Q:-0}" VIZ_Q_EPISODES="${VIZ_Q_EPISODES:-0}" \
  bash scriptsv2/eval_search/eval_search.sh "$CHECKPOINT" "$NUM_EPISODES" "$OUT_DIR"
_t1=$(date +%s); _dt=$((_t1-_t0))
printf 'sampler=%s N=%s episodes=%s wall_clock_seconds=%s (%dh%02dm%02ds)\n' \
  "$BC_SAMPLER" "$N_SAMPLES" "$NUM_EPISODES" "$_dt" $((_dt/3600)) $(((_dt%3600)/60)) $((_dt%60)) | tee "$OUT_DIR/time.txt"
if [[ "${RENDER_Q:-0}" == "1" && -d "$OUT_DIR/search_q" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate "${CONDA_ENV:-dqc_jax}"
  PYTHONPATH="$PWD/diffusion_policy:${PYTHONPATH:-}" \
    python scriptsv2/eval_search/render_q.py "$OUT_DIR/search_q" --fps "${VIDEO_FPS:-10}" \
    || echo "[bc_dqcchunk] render_q failed"
fi
echo "[bc_dqcchunk] done -> $OUT_DIR/eval_log.json"
