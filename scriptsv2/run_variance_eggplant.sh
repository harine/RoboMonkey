#!/bin/bash
# Measure action variance for both eggplant-in-basket policy checkpoints
# (Diffusion U-Net and MLP) and print a combined comparison table.
#
# Usage
# -----
#   bash scriptsv2/run_variance_eggplant.sh                # defaults
#   N_SAMPLES=64 bash scriptsv2/run_variance_eggplant.sh   # more samples
#
# Env vars
# --------
#   N_SAMPLES         samples per obs measurement (default 32)
#   N_ROLLOUT_STEPS   env steps per episode         (default 60)
#   N_EPISODES        number of env resets          (default 5)
#   DEVICE            (default cuda:0)
#   CONDA_ENV         (default simpler_env)

set -euo pipefail

UNET_CKPT="/home/harine/diffusion_policy/data/outputs/2026.04.29/16.45.01_train_diffusion_unet_eggplant_in_basket_lowdim_eggplant_in_basket_lowdim/checkpoints/latest.ckpt"
MLP_CKPT="/home/harine/diffusion_policy/data/outputs/2026.04.29/16.40.54_train_mlp_eggplant_in_basket_lowdim_eggplant_in_basket_lowdim/checkpoints/latest.ckpt"

N_SAMPLES="${N_SAMPLES:-32}"
N_ROLLOUT_STEPS="${N_ROLLOUT_STEPS:-60}"
N_EPISODES="${N_EPISODES:-5}"
SAVE_VIDEOS="${SAVE_VIDEOS:-5}"   # save annotated MP4s for first N episodes per arch
VIDEO_FPS="${VIDEO_FPS:-10}"
DEVICE="${DEVICE:-cuda:0}"
CONDA_ENV="${CONDA_ENV:-simpler_env}"

UNET_OUT="data/eval/variance_eggplant_unet"
MLP_OUT="data/eval/variance_eggplant_mlp"

full_path="$(realpath "$0")"
dir_path="$(dirname "$full_path")"
repo_root="$(dirname "$dir_path")"
cd "$repo_root"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Install diffusion_policy deps (same guard as run_eval_diffusion_policy.sh).
python - <<'PY'
import importlib.util, subprocess, sys
from importlib.metadata import PackageNotFoundError, version

DIFFUSERS_MIN = (0, 30); DIFFUSERS_MAX = (0, 34); SPEC = "diffusers>=0.30,<0.34"

def _pip(specs):
    print(f"[var] pip install: {specs}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *specs])

def _smoke():
    return subprocess.run([sys.executable, "-c",
        "from diffusers.schedulers.scheduling_ddpm import DDPMScheduler; "
        "import hydra, dill; print('[var] deps ok')"]).returncode == 0

needed = [pkg for mod, pkg in [("hydra","hydra-core==1.2.0"),("dill","dill")]
          if importlib.util.find_spec(mod) is None]
if needed: _pip(needed)

try: cur = version("diffusers")
except PackageNotFoundError: cur = None
in_range = cur and (DIFFUSERS_MIN <= tuple(int(x) for x in cur.split(".")[:2]) < DIFFUSERS_MAX)
if not in_range:
    print(f"[var] diffusers={cur}; reinstalling to {SPEC}")
    _pip([SPEC])
else:
    print(f"[var] diffusers={cur} (ok)")

if not _smoke():
    _pip(["--force-reinstall", "--no-deps", SPEC])
    if not _smoke():
        raise SystemExit("[var] diffusers smoke check still failing")
PY

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export DISPLAY=""
export PYTHONPATH="/home/harine/diffusion_policy:${PYTHONPATH:-}"

echo "============================================================"
echo "  action variance analysis — eggplant in basket"
echo "  n_samples       : $N_SAMPLES"
echo "  n_rollout_steps : $N_ROLLOUT_STEPS"
echo "  n_episodes      : $N_EPISODES"
echo "  save_videos     : $SAVE_VIDEOS  (fps=$VIDEO_FPS)"
echo "  device          : $DEVICE"
echo "============================================================"
echo

echo ">>> Diffusion U-Net"
xvfb-run --auto-servernum -s "-screen 0 640x480x24" \
    python scriptsv2/analyze_action_variance.py \
        --checkpoint "$UNET_CKPT" \
        --n-samples "$N_SAMPLES" \
        --n-rollout-steps "$N_ROLLOUT_STEPS" \
        --n-episodes "$N_EPISODES" \
        --save-videos "$SAVE_VIDEOS" \
        --video-fps "$VIDEO_FPS" \
        --device "$DEVICE" \
        --output-dir "$UNET_OUT"

echo
echo ">>> MLP"
xvfb-run --auto-servernum -s "-screen 0 640x480x24" \
    python scriptsv2/analyze_action_variance.py \
        --checkpoint "$MLP_CKPT" \
        --n-samples "$N_SAMPLES" \
        --n-rollout-steps "$N_ROLLOUT_STEPS" \
        --n-episodes "$N_EPISODES" \
        --save-videos "$SAVE_VIDEOS" \
        --video-fps "$VIDEO_FPS" \
        --device "$DEVICE" \
        --output-dir "$MLP_OUT"

echo
echo "============================================================"
echo "  OUTPUT FILES"
echo "  U-Net: $UNET_OUT/variance_summary.json"
echo "         $UNET_OUT/variance_plot.png"
echo "         $UNET_OUT/videos/ep*.mp4"
echo "  MLP  : $MLP_OUT/variance_summary.json"
echo "         $MLP_OUT/variance_plot.png"
echo "         $MLP_OUT/videos/ep*.mp4"
echo "============================================================"

# Quick side-by-side comparison from the two JSON summaries.
python - "$UNET_OUT/variance_summary.json" "$MLP_OUT/variance_summary.json" <<'PY'
import json, sys
paths = sys.argv[1:]
data  = [json.load(open(p)) for p in paths]
labels = [d["policy_type"] for d in data]
dims   = list(data[0]["per_dim"].keys())
print()
print("══════════════════ COMPARISON: mean_std per action dim ══════════════════")
h = f"  {'dim':<10}" + "".join(f"  {l:<22}" for l in labels)
print(h)
print("  " + "─" * (len(h) - 2))
for dim in dims:
    row = f"  {dim:<10}"
    for d in data:
        v = d["per_dim"][dim]["mean_std"]
        row += f"  {v:>10.5f}              "
    print(row)
print("  " + "─" * (len(h) - 2))
row = f"  {'overall':<10}"
for d in data:
    row += f"  {d['overall_mean_std']:>10.5f}              "
print(row)
print()
PY
