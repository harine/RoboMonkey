#!/bin/bash
# Evaluate a diffusion_policy lowdim checkpoint (DiffusionUnet or MLP) trained
# on `bridge_v2_carrot_lowdim` over N rollouts of `widowx_carrot_on_plate`.
#
# Usage
# -----
#   bash scripts/run_eval_diffusion_policy.sh <checkpoint> [num_episodes] [output_dir]
#
# Defaults:
#   num_episodes = 100
#   output_dir   = data/eval/$(basename $(dirname $(dirname <checkpoint>)))
#
# Examples
# --------
#   # Evaluate the long-trained Diffusion U-Net run:
#   bash scripts/run_eval_diffusion_policy.sh \
#       /home/harine/diffusion_policy/data/outputs/2026.04.28/16.22.54_train_diffusion_unet_bridge_v2_carrot_lowdim_bridge_v2_carrot_lowdim/checkpoints/latest.ckpt
#
#   # Evaluate the MLP run with 50 rollouts and a custom output dir:
#   bash scripts/run_eval_diffusion_policy.sh \
#       /home/harine/diffusion_policy/data/outputs/2026.04.28/16.14.29_train_mlp_bridge_v2_carrot_lowdim_bridge_v2_carrot_lowdim/checkpoints/latest.ckpt \
#       50 \
#       data/eval/mlp_carrot
#
# Env vars (override on cmdline):
#   DEVICE            (default: cuda:0)
#   START_SEED        (default: 1000)
#   MAX_STEPS         (default: 120)
#   NUM_INFERENCE_STEPS  (default: cfg / 100, lower => faster)
#   USE_EMA           (default: 1; set 0 to use raw model weights)
#   SAVE_VIDEOS       (default: 0; save MP4s for the first N episodes)
#   VIDEO_FPS         (default: 10)
#   TASK              (default: widowx_carrot_on_plate)
#                     e.g. widowx_put_eggplant_in_basket
#   CONDA_ENV         (default: simpler_env)
#   BON_K                       (default: 1; >1 enables verifier Best-of-N)
#   BON_REPLAN_EVERY_N_STEPS    (default: 0 = chunk replan; N>0 replans after
#                               every N executed actions)
#   BON_SCORE_NUM_ACTIONS       (default: 1; average rewards over first N actions)
#   REWARD_SERVER_PORT          (default: 3100)
#   REWARD_BATCH_SIZE           (default: 2)

set -euo pipefail

CKPT="${1:-}"
NUM_EPISODES="${2:-100}"
OUT_DIR_ARG="${3:-}"

if [[ -z "$CKPT" ]]; then
    echo "usage: bash $0 <checkpoint> [num_episodes] [output_dir]" >&2
    exit 1
fi
if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: checkpoint not found: $CKPT" >&2
    exit 1
fi

# Default output dir = data/eval/<run_name>
if [[ -z "$OUT_DIR_ARG" ]]; then
    RUN_DIR="$(dirname "$(dirname "$CKPT")")"
    RUN_NAME="$(basename "$RUN_DIR")"
    OUT_DIR="data/eval/${RUN_NAME}"
else
    OUT_DIR="$OUT_DIR_ARG"
fi

DEVICE="${DEVICE:-cuda:0}"
START_SEED="${START_SEED:-1000}"
MAX_STEPS="${MAX_STEPS:-120}"
USE_EMA="${USE_EMA:-1}"
CONDA_ENV="${CONDA_ENV:-simpler_env}"
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"     # save MP4 for the first N episodes
VIDEO_FPS="${VIDEO_FPS:-10}"
TASK="${TASK:-widowx_put_eggplant_in_basket}"
BON_K="${BON_K:-1}"
BON_REPLAN_EVERY_N_STEPS="${BON_REPLAN_EVERY_N_STEPS:-0}"
BON_SCORE_NUM_ACTIONS="${BON_SCORE_NUM_ACTIONS:-1}"
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-3100}"
# Verifier batch size: all batches for a given replan are now dispatched in
# parallel threads, so raising this reduces round-trips (GPU memory allowing).
# 16 is safe for a 24GB GPU at any k. 32 OOMs the LLaVA verifier when a single
# batch contains 32 rows of full ~1280-token prompts (KV cache exceeds VRAM).
# Lower to 8 if you see verifier 500s; raise only after testing at your max k.
REWARD_BATCH_SIZE="${REWARD_BATCH_SIZE:-16}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# One-time install / upgrade of the diffusion_policy deps that simpler_env
# doesn't already have. Constraints:
#   * diffusers>=0.30 -- earlier versions import the legacy
#                       `huggingface_hub.cached_download` (removed in hf_hub>=0.26).
#   * diffusers<0.34  -- 0.34+ bumped the peft requirement to >=0.15, but
#                       simpler_env ships peft==0.11.1, so they fail at import.
#   => we want diffusers>=0.30,<0.34.
#
# We never `import diffusers` inside the Python that runs the pip install --
# a failed pre-install import would leave partial entries in sys.modules and
# the post-upgrade re-import would mismatch the new on-disk files. Instead we
# use `importlib.metadata.version` (which doesn't execute the package) and run
# the final smoke import in a fresh subprocess.
python - <<'PY'
import importlib.util, subprocess, sys
from importlib.metadata import PackageNotFoundError, version

DIFFUSERS_MIN = (0, 30)
DIFFUSERS_MAX = (0, 34)  # exclusive
DIFFUSERS_SPEC = "diffusers>=0.30,<0.34"

def _pip_install(specs):
    print(f"[run_eval] pip install: {specs}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *specs])

def _smoke():
    return subprocess.run([
        sys.executable, "-c",
        "from diffusers.schedulers.scheduling_ddpm import DDPMScheduler; "
        "import hydra, dill; "
        "print('[run_eval] deps ok: diffusers + hydra + dill')",
    ]).returncode == 0

def _diffusers_version():
    try:
        return version("diffusers")
    except PackageNotFoundError:
        return None

# 1) hydra + dill
to_install = []
for mod, pkg in [("hydra", "hydra-core==1.2.0"), ("dill", "dill")]:
    if importlib.util.find_spec(mod) is None:
        to_install.append(pkg)
if to_install:
    _pip_install(to_install)

# 2) diffusers in the supported window.
cur = _diffusers_version()
in_range = False
if cur is not None:
    try:
        major, minor = (int(x) for x in cur.split(".")[:2])
        if DIFFUSERS_MIN <= (major, minor) < DIFFUSERS_MAX:
            in_range = True
    except Exception:
        pass

if not in_range:
    print(f"[run_eval] diffusers={cur!s}; reinstalling to {DIFFUSERS_SPEC}")
    _pip_install([DIFFUSERS_SPEC])
else:
    print(f"[run_eval] diffusers={cur} (ok)")

# 3) Smoke import. If it fails (e.g. an existing in-range version is broken
#    by a separate dependency), force a clean reinstall and try once more.
if not _smoke():
    print("[run_eval] smoke check failed; force-reinstalling diffusers")
    _pip_install(["--force-reinstall", "--no-deps", DIFFUSERS_SPEC])
    if not _smoke():
        raise SystemExit(
            "[run_eval] diffusers post-install smoke check still failing; "
            "aborting. Check `pip show diffusers peft huggingface_hub`."
        )
PY

# SimplerEnv / SAPIEN env vars (same as collect_carrot_on_plate.sh)
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
export DISPLAY=""

# Make `import diffusion_policy` work without needing to pip-install it.
DP_ROOT="${DIFFUSION_POLICY_ROOT:-/home/harine/diffusion_policy}"
export PYTHONPATH="${DP_ROOT}:${PYTHONPATH:-}"

# Run from the SimplerEnv repo dir so relative asset paths resolve.
full_path="$(realpath "$0")"
dir_path="$(dirname "$full_path")"
cd "$dir_path/.."

mkdir -p "$OUT_DIR"

EXTRA_FLAGS=()
if [[ "$USE_EMA" == "0" ]]; then
    EXTRA_FLAGS+=(--no-ema)
fi
if [[ -n "${NUM_INFERENCE_STEPS:-}" ]]; then
    EXTRA_FLAGS+=(--num-inference-steps "$NUM_INFERENCE_STEPS")
fi
if [[ "$SAVE_VIDEOS" -gt 0 ]]; then
    EXTRA_FLAGS+=(--save-videos "$SAVE_VIDEOS" --video-fps "$VIDEO_FPS")
fi
EXTRA_FLAGS+=(--task "$TASK")
EXTRA_FLAGS+=(--bon-k "$BON_K")
EXTRA_FLAGS+=(--bon-score-num-actions "$BON_SCORE_NUM_ACTIONS")
EXTRA_FLAGS+=(--bon-replan-every-n-steps "$BON_REPLAN_EVERY_N_STEPS")
EXTRA_FLAGS+=(--reward-server-port "$REWARD_SERVER_PORT")
EXTRA_FLAGS+=(--reward-batch-size "$REWARD_BATCH_SIZE")

echo "============================================================"
echo "  diffusion_policy SimplerEnv eval"
echo "  checkpoint   : $CKPT"
echo "  task         : $TASK"
echo "  num_episodes : $NUM_EPISODES"
echo "  output_dir   : $OUT_DIR"
echo "  device       : $DEVICE"
echo "  use_ema      : $USE_EMA"
echo "  start_seed   : $START_SEED"
echo "  max_steps    : $MAX_STEPS"
echo "  save_videos  : $SAVE_VIDEOS  (fps=$VIDEO_FPS)"
echo "  bon_k        : $BON_K"
echo "  bon_replan_every_n_steps : $BON_REPLAN_EVERY_N_STEPS"
echo "  bon_score_num_actions    : $BON_SCORE_NUM_ACTIONS"
echo "  reward_port  : $REWARD_SERVER_PORT"
echo "============================================================"

xvfb-run --auto-servernum -s "-screen 0 640x480x24" \
    python scriptsv2/eval_diffusion_policy_carrot.py \
        --checkpoint "$CKPT" \
        --num-episodes "$NUM_EPISODES" \
        --start-seed "$START_SEED" \
        --max-steps "$MAX_STEPS" \
        --device "$DEVICE" \
        --output-dir "$OUT_DIR" \
        "${EXTRA_FLAGS[@]}"

echo
echo "[run_eval] done. Summary -> $OUT_DIR/eval_log.json"
if [[ -f "$OUT_DIR/eval_log.json" ]]; then
    python "$dir_path/summarize_evals.py" "$OUT_DIR/eval_log.json"
fi
