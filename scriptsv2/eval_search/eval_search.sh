#!/bin/bash
# Evaluate a SearchPolicyRoboMonkey checkpoint (Gaussian or Diffusion variant
# of the state-based L2S policy with in-process RoboMonkey verifier) over N
# rollouts in SimplerEnv.
#
# Usage
# -----
#   bash scriptsv2/eval_search/eval_search.sh <checkpoint> [num_episodes|all] [output_dir]
#
# Examples
# --------
#   bash scriptsv2/eval_search/eval_search.sh \
#       /home/harine/diffusion_policy/data/outputs/2026.05.11/17.40.43_robomonkey_eggplant_search_state_corrupt_robomonkey_eggplant_state/checkpoints/latest.ckpt \
#       all \
#       data/eval/eggplant_search_state_corrupt
#
# Env vars (override on cmdline):
#   DEVICE        (default: cuda:0)
#   SPLIT_FILE    (default: the checkpoint's task.dataset.split_file) split manifest
#                 (diffusion_policy/diffusion_policy/config/splits/simpler/<task>/...)
#   SPLIT         (default: test; SPLIT_NAME is an alias) which list of the manifest
#                 (test|val|train) is rolled out from its episodes' seeds
#   SEEDS         (default: unset; comma-separated explicit seeds override the manifest)
#   VERIFIER_TYPE=veritas needs VERITAS_CAMERA_FILE + VERITAS_PLAN_DIR
#                 (+ VERITAS_ACTION_POS_SCALE=null|"[x,y,z]", VERITAS_ALLOW_LIVE=0|1)
#   MAX_STEPS     (default: 120)
#   USE_EMA       (default: 1; set 0 to use raw model weights)
#   SAVE_VIDEOS   (default: 0; save MP4s for the first N episodes)
#   VIDEO_FPS     (default: 10)
#   VIZ_Q         (default: 0; when 1, save per-replan sampled candidate
#                 actions + verifier values + frame to
#                 <output_dir>/search_q/ep<idx>_seed<seed>.npz)
#   N_SAMPLES     (default: unset = policy.max_actions; override to sweep
#                 search width. For values > max_actions the eval slides a
#                 fixed (max_actions-1) context window — same context length
#                 as training, but argmax picks the best of N candidates.)
#   TASK          (default: widowx_put_eggplant_in_basket)
#   CONDA_ENV     (default: simpler_env)

set -euo pipefail

CKPT="${1:-}"
NUM_EPISODES="${2:-all}"          # cap on the split's seeds; 'all' = every seed
OUT_DIR_ARG="${3:-}"

if [[ -z "$CKPT" ]]; then
    echo "usage: bash $0 <checkpoint> [num_episodes] [output_dir]" >&2
    exit 1
fi
if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: checkpoint not found: $CKPT" >&2
    exit 1
fi

if [[ -z "$OUT_DIR_ARG" ]]; then
    RUN_DIR="$(dirname "$(dirname "$CKPT")")"
    RUN_NAME="$(basename "$RUN_DIR")"
    OUT_DIR="data/eval/${RUN_NAME}"
else
    OUT_DIR="$OUT_DIR_ARG"
fi

DEVICE="${DEVICE:-cuda:0}"
SPLIT="${SPLIT_NAME:-${SPLIT:-test}}"  # which list of the manifest to roll out
SEEDS="${SEEDS:-}"
SPLIT_FILE="${SPLIT_FILE:-}"      # empty = the checkpoint's own split manifest
MAX_STEPS="${MAX_STEPS:-120}"
USE_EMA="${USE_EMA:-1}"
CONDA_ENV="${CONDA_ENV:-simpler_env}"
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"
VIDEO_FPS="${VIDEO_FPS:-10}"
TASK="${TASK:-widowx_put_eggplant_in_basket}"
MODE="${MODE:-softmax}"             # default selection: softmax (set MODE=argmax for best-of-N)
SOFTMAX_TEMP="${SOFTMAX_TEMP:-1.0}"  # used when MODE=softmax
VIZ_Q="${VIZ_Q:-0}"               # 1 = dump sampled actions + values to search_q/
VIZ_Q_EPISODES="${VIZ_Q_EPISODES:-0}"  # when VIZ_Q=1, only save render npz for
                                  # the first K episodes (0 = all). The first K
                                  # episodes use the same seeds across every N.
N_SAMPLES="${N_SAMPLES:-}"        # empty = use policy.max_actions
OBS_CORRUPTION="${OBS_CORRUPTION:-as_trained}"  # as_trained | clean: whether a policy
                                  # trained with a fixed obs_noise_level sees that
                                  # (seeded) corruption at eval, or clean obs.
EVAL_OBS_NOISE_LEVEL="${EVAL_OBS_NOISE_LEVEL:-}"  # BC only: eval-only state obs noise
                                  # level (empty = off); see --eval-obs-noise-level.
SCORE_WINDOW="${SCORE_WINDOW:-executed}"  # 'executed' = verifier scores each
                                  # executed action step (steps start..start+
                                  # n_action_steps) and aggregates (RoboMonkey-
                                  # faithful: rate the actions that run). 'last'
                                  # = single action_chunk_index (legacy).
SCORE_AGG="${SCORE_AGG:-}"        # mean|sum|min over the executed-window scores;
                                  # empty (default) = use the checkpoint's baked
                                  # score_agg, so eval matches training.
SCORE_DISCOUNT="${SCORE_DISCOUNT:-}"  # geometric discount on later chunk steps
                                  # (step t weighted by SCORE_DISCOUNT**t);
                                  # empty = use ckpt baked (1.0 = uniform).

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# SimplerEnv / SAPIEN env vars
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
export DISPLAY=""

# Deterministic numerics for the policy (cuBLAS) and the JAX verifier (XLA). Both must be
# set before CUDA / XLA initialize, so they live here rather than in eval_search.py.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
case " ${XLA_FLAGS:-} " in
  *xla_gpu_deterministic_ops*) ;;
  *) export XLA_FLAGS="--xla_gpu_deterministic_ops=true ${XLA_FLAGS:-}" ;;
esac
# Full-precision (non-TF32) matmuls in the JAX DQC critic, read by jax.config at import.
export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-float32}"

# Loader path for bitsandbytes (CUDA-13 libnvJitLink under the conda env)
export LD_LIBRARY_PATH="${HOME}/miniconda3/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

DP_ROOT="${DIFFUSION_POLICY_ROOT:-${repo_root:-/home/harine/RoboMonkey}/diffusion_policy}"
export DIFFUSION_POLICY_ROOT="$DP_ROOT"   # else eval_diffusion.py prepends a stale checkout
export PYTHONPATH="${DP_ROOT}:${PYTHONPATH:-}"

full_path="$(realpath "$0")"
dir_path="$(dirname "$full_path")"
repo_root="$(cd "$dir_path/../.." && pwd)"
cd "$repo_root"

mkdir -p "$OUT_DIR"

EXTRA_FLAGS=()
if [[ "$USE_EMA" == "0" ]]; then
    EXTRA_FLAGS+=(--no-ema)
fi
if [[ "$SAVE_VIDEOS" -gt 0 ]]; then
    EXTRA_FLAGS+=(--save-videos "$SAVE_VIDEOS" --video-fps "$VIDEO_FPS")
fi
EXTRA_FLAGS+=(--task "$TASK")
EXTRA_FLAGS+=(--mode "$MODE")
if [[ "$MODE" == "softmax" ]]; then
    EXTRA_FLAGS+=(--softmax-temp "$SOFTMAX_TEMP")
fi
if [[ "$VIZ_Q" == "1" ]]; then
    EXTRA_FLAGS+=(--viz-q)
    if [[ -n "$VIZ_Q_EPISODES" && "$VIZ_Q_EPISODES" != "0" ]]; then
        EXTRA_FLAGS+=(--viz-q-episodes "$VIZ_Q_EPISODES")
    fi
fi
if [[ -n "$SEEDS" ]]; then
    EXTRA_FLAGS+=(--seeds "$SEEDS")
else
    EXTRA_FLAGS+=(--split "$SPLIT")
    [[ -n "$SPLIT_FILE" ]] && EXTRA_FLAGS+=(--split-file "$SPLIT_FILE")
fi
if [[ "$NUM_EPISODES" != "all" ]]; then
    EXTRA_FLAGS+=(--num-episodes "$NUM_EPISODES")
fi
if [[ -n "$N_SAMPLES" ]]; then
    EXTRA_FLAGS+=(--n-samples "$N_SAMPLES")
fi
EXTRA_FLAGS+=(--score-window "$SCORE_WINDOW")
EXTRA_FLAGS+=(--obs-corruption "$OBS_CORRUPTION")
if [[ -n "$EVAL_OBS_NOISE_LEVEL" ]]; then EXTRA_FLAGS+=(--eval-obs-noise-level "$EVAL_OBS_NOISE_LEVEL"); fi
if [[ -n "$SCORE_AGG" ]]; then EXTRA_FLAGS+=(--score-agg "$SCORE_AGG"); fi
if [[ -n "$SCORE_DISCOUNT" ]]; then EXTRA_FLAGS+=(--score-discount "$SCORE_DISCOUNT"); fi

echo "============================================================"
echo "  search-policy SimplerEnv eval"
echo "  checkpoint   : $CKPT"
echo "  task         : $TASK"
echo "  num_episodes : $NUM_EPISODES"
echo "  output_dir   : $OUT_DIR"
echo "  device       : $DEVICE"
echo "  use_ema      : $USE_EMA"
if [[ -n "$SEEDS" ]]; then
    echo "  seeds        : $SEEDS  (explicit, overrides SPLIT)"
else
    echo "  split        : ${SPLIT_FILE:-<checkpoint split_file>} [$SPLIT]"
fi
echo "  max_steps    : $MAX_STEPS"
echo "  save_videos  : $SAVE_VIDEOS  (fps=$VIDEO_FPS)"
echo "  mode         : $MODE"
if [[ "$MODE" == "softmax" ]]; then
    echo "  softmax_temp : $SOFTMAX_TEMP"
fi
echo "  n_samples    : ${N_SAMPLES:-<policy.max_actions>}"
echo "  sampling     : seeded per (episode seed, t)"
echo "  obs_corrupt  : $OBS_CORRUPTION"
echo "  obs_noise    : ${EVAL_OBS_NOISE_LEVEL:-off}  (BC eval-only state noise)"
echo "  score_window : $SCORE_WINDOW (agg=${SCORE_AGG:-<baked from ckpt>}, discount=${SCORE_DISCOUNT:-<baked>})"
echo "  viz_q        : $VIZ_Q"
echo "============================================================"

# OSMesa renders headless (no X server), so xvfb is optional. Use it when
# available, otherwise run python directly (mirrors eval_diffusion.sh).
if command -v xvfb-run >/dev/null 2>&1; then
    XVFB=(xvfb-run --auto-servernum -s "-screen 0 640x480x24")
else
    XVFB=()
fi
"${XVFB[@]}" python "$dir_path/eval_search.py" \
        --checkpoint "$CKPT" \
        --max-steps "$MAX_STEPS" \
        --device "$DEVICE" \
        --output-dir "$OUT_DIR" \
        "${EXTRA_FLAGS[@]}"

echo
echo "[run_eval] done. Summary -> $OUT_DIR/eval_log.json"
if [[ -f "$OUT_DIR/eval_log.json" ]]; then
    python "$repo_root/scriptsv2/eval_diffusion/eval_summary.py" "$OUT_DIR/eval_log.json"
fi
