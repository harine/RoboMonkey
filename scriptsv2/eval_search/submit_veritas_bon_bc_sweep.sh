#!/usr/bin/env bash
# Best-of-N sweep of every per-split BC UNet of a task under the Veritas verifier, at each
# 10k checkpoint (the first saved step >= 10k, 20k, ..., 50k; saved steps carry a small
# epoch offset, e.g. step_0010005), N in NS, on the run's own split manifest (test
# episodes). Re-runnable: finished cells (eval_log.json) and queued jobs are skipped, and
# checkpoints that do not exist yet are reported, so re-run as training progresses.
#
#   SUBMIT=1 bash scriptsv2/eval_search/submit_veritas_bon_bc_sweep.sh eggplant
#
# Env: ARMS (default scale_null = upstream 1:1; scale_fit needs SCALE_FIT), NS
# (default "1 2 4 8 16 32 64"), STEPS_K (default "10 20 30 40 50"), PLAN_DIR, SBATCH_ARGS
# (default: ckpt scavenger with requeue; eval_search resumes per episode).
set -euo pipefail
TASK_SHORT="${1:?task: eggplant|stack}"
# LIVE must be explicit (2026-10-01): stale loops calling this with the old defaults
# (offline plans, both modes, every 10k step) must not resubmit the retired protocol.
: "${LIVE:?set LIVE=1 (Gemini plan at each reset, upstream) or LIVE=0 (offline plan cache)}"
export LIVE
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BC_ROOT="$REPO/diffusion_policy/data/train/bc_denoiser"
export NS="${NS:-1 2 4 8 16 32 64}"
export ARMS="${ARMS:-scale_null}"
export PLAN_DIR="${PLAN_DIR:-/gscratch/robotics/harine/data/veritas_plans/$TASK_SHORT/v1_upstream}"
export SBATCH_ARGS="${SBATCH_ARGS:---account=ckpt-robotics --partition=ckpt --qos=ckpt-gpu --gres=gpu:1 --constraint=a40|l40s|l40|a100 --requeue}"
STEPS_K="${STEPS_K:-10 20 30 40 50}"
SPLIT_DIR="$REPO/diffusion_policy/diffusion_policy/config/splits/simpler/$TASK_SHORT"

for run_dir in "$BC_ROOT"/bc_unet_h8_*_"${TASK_SHORT}"_*; do
  run="$(basename "$run_dir")"
  rule="${run#*_${TASK_SHORT}_}"                       # e.g. random_train400
  mf="$SPLIT_DIR/${rule}.json"
  [ -f "$mf" ] || { echo "no manifest for $run"; continue; }
  for k in $STEPS_K; do
    target=$((k * 1000))
    ckpt="$(ls "$run_dir"/checkpoints/step_*.ckpt 2>/dev/null | awk -v t="$target" -F'step_|.ckpt' '$2+0 >= t {print; exit}')"
    name=""
    if [ -z "$ckpt" ] && [ -f "$run_dir/checkpoints/latest.ckpt" ]; then
      # the final step is saved only as latest.ckpt (step_* carry an epoch offset, e.g.
      # step_0049006 is the last numbered one); use it when it has reached the target
      gs="$("$HOME/miniconda3/envs/dqc_jax/bin/python" - "$run_dir/checkpoints/latest.ckpt" "$REPO/data/eval/veritas_bon/_latest_steps.json" <<'PY'
import json, os, pickle, sys
path, cache_path = sys.argv[1], sys.argv[2]
key = f'{path}:{int(os.path.getmtime(path))}'
cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
if key not in cache:
    import torch
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cache[key] = int(pickle.loads(ck['pickles']['global_step']))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    json.dump(cache, open(cache_path, 'w'), indent=1)
print(cache[key])
PY
)"
      if [ "${gs:--1}" -ge "$target" ]; then ckpt="$run_dir/checkpoints/latest.ckpt"; name="$(printf 'step_%07d' "$gs")"; fi
    fi
    if [ -z "$ckpt" ]; then echo "pending: $run ${k}k (not saved yet)"; continue; fi
    STEP_NAME="$name" CKPT="$ckpt" SPLIT_FILE="$mf" bash "$REPO/scriptsv2/eval_search/submit_veritas_bon.sh" "$TASK_SHORT"
  done
done
