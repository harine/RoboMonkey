#!/bin/bash
#SBATCH --job-name=srch_dqcchunk
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/mmfs1/home/harine/RoboMonkey/data/eval/search_dqc_chunk/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# Evaluate the GAUSSIAN search policy (SearchPolicyRoboMonkey, task-2 training:
# eggplant_gaussian_dqc_chunk8 ... predhorizon8_rollout4) with its OWN baked
# DQC CHUNK verifier — i.e. verifier + encoder identical to training.
#
# Faithfulness:
#   * We do NOT set VERIFIER_TYPE=dqc. That env flag makes load_policy REPLACE the
#     baked verifier cfg with a score_executed_chunk / discounted_sum one — which
#     is NOT how this policy trained. Leaving it unset instantiates cfg.policy
#     (and its baked DQCVerifier: critic_type=chunk, score_full_horizon=True,
#     q_agg=min, the training ckpt_dir + task_emb_use3.npy) exactly as trained.
#   * SCORE_WINDOW=last so the eval doesn't slice score_indices; score_full_horizon
#     (baked) then scores ALL 8 predicted actions as the chunk (get_value uses
#     range(H), priority over score_indices). Prediction horizon 8, rollout 4.
#   * USE_EMA=0 — task-2 trained with training.use_ema=False (no ema_model in ckpt).
#   * Seeds = BC BoN eval seeds: START_SEED=1000, 50 episodes (seeds 1000..1049).
#
# Runs in the dqc_jax env (simpler_env sim + JAX[DQC] in-process).
#   sbatch --export=ALL,CHECKPOINT=<ckpt>,N_SAMPLES=4  eval_search_dqc_chunk.sbatch
#   sbatch --export=ALL,CHECKPOINT=<ckpt>,N_SAMPLES=16 eval_search_dqc_chunk.sbatch
# -----------------------------------------------------------------------------
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
: "${CHECKPOINT:?set CHECKPOINT (gaussian DQC-chunk search policy)}"

N_SAMPLES="${N_SAMPLES:-4}"
NUM_EPISODES="${NUM_EPISODES:-50}"
START_SEED="${START_SEED:-1000}"
N_ACTION_STEPS="${N_ACTION_STEPS:-4}"      # rollout 4 (matches training)
OUT_DIR="${OUT_DIR:-data/eval/search_dqc_chunk/gaussian_chunk8_predh8_roll4_n${N_SAMPLES}_seed${START_SEED}_ep${NUM_EPISODES}}"
mkdir -p data/eval/search_dqc_chunk "$OUT_DIR"

export DIFFUSION_POLICY_ROOT="${DIFFUSION_POLICY_ROOT:-$SLURM_SUBMIT_DIR/diffusion_policy}"
# JAX (baked DQC verifier) must not preallocate — torch (policy/SAPIEN) shares the GPU.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=} --xla_gpu_deterministic_ops=true"
# Determinism: cuBLAS needs a fixed workspace for reproducible GEMMs (set before CUDA init).
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

echo "[srch_dqcchunk] host=$(hostname) ckpt=$CHECKPOINT"
echo "[srch_dqcchunk] N=$N_SAMPLES  rollout(n_action_steps)=$N_ACTION_STEPS  pred_horizon=8(baked)"
echo "[srch_dqcchunk] verifier=BAKED DQC chunk (score_full_horizon, q_agg=min); VERIFIER_TYPE unset"
echo "[srch_dqcchunk] seeds=${START_SEED}..$((START_SEED+NUM_EPISODES-1))  episodes=$NUM_EPISODES  mode=${MODE:-softmax} softmax_temp=${SOFTMAX_TEMP:-1.0}  -> $OUT_DIR"

# Wall-clock timing of the whole eval (printed at the end + saved to time.txt).
_t0=$(date +%s)

CONDA_ENV="${CONDA_ENV:-dqc_jax}" DEVICE=cuda:0 \
  MODE="${MODE:-softmax}" SOFTMAX_TEMP="${SOFTMAX_TEMP:-1.0}" SCORE_WINDOW=last \
  N_SAMPLES="$N_SAMPLES" N_ACTION_STEPS="$N_ACTION_STEPS" \
  START_SEED="$START_SEED" MAX_STEPS="${MAX_STEPS:-120}" USE_EMA=0 \
  TASK="${TASK:-widowx_put_eggplant_in_basket}" \
  SAVE_VIDEOS="${SAVE_VIDEOS:-0}" VIDEO_FPS="${VIDEO_FPS:-10}" \
  VIZ_Q="${VIZ_Q:-0}" VIZ_Q_EPISODES="${VIZ_Q_EPISODES:-0}" \
  bash scriptsv2/eval_search/eval_search.sh "$CHECKPOINT" "$NUM_EPISODES" "$OUT_DIR"

_t1=$(date +%s); _dt=$((_t1-_t0))
printf 'N=%s episodes=%s wall_clock_seconds=%s (%dh%02dm%02ds)\n' \
  "$N_SAMPLES" "$NUM_EPISODES" "$_dt" $((_dt/3600)) $(((_dt%3600)/60)) $((_dt%60)) | tee "$OUT_DIR/time.txt"
# Optional Q-overlay videos (candidate trajectories colored by verifier-Q rank +
# the picked one) from the viz_q dumps — shows WHY the verifier selected as it did.
if [[ "${RENDER_Q:-0}" == "1" && -d "$OUT_DIR/search_q" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate "${CONDA_ENV:-dqc_jax}"
  PYTHONPATH="$PWD/diffusion_policy:${PYTHONPATH:-}" \
    python scriptsv2/eval_search/render_q.py "$OUT_DIR/search_q" --fps "${VIDEO_FPS:-10}" \
    || echo "[srch_dqcchunk] render_q failed"
fi
echo "[srch_dqcchunk] done -> $OUT_DIR/eval_log.json  (timing in $OUT_DIR/time.txt)"
