#!/bin/bash
#SBATCH --job-name=tr_resume
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/gscratch/robotics/harine/RoboMonkey_data/train/chunksum/slurm-%j.out
#SBATCH --error=/gscratch/robotics/harine/RoboMonkey_data/train/chunksum/slurm-%j.out
#
# RESUME a chunk-sum search-policy training on a (faster) GPU, writing
# checkpoints to GSCRATCH (not the 10GB home quota that corrupted the original
# saves). POLICY env selects: noise0.0 | noise0.4 | noise0.8 | tmrl.
# Account/partition/gres from the submit command.
#
# Bootstrap: on first run it copies the NEWEST LOADABLE checkpoint from the
# original home run dir into the gscratch run dir as latest.ckpt, so resume
# (training.resume=True) continues from valid weights. Later chain links find a
# valid gscratch latest.ckpt and just continue.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"

POLICY="${POLICY:?set POLICY=noise0.0|noise0.4|noise0.8|tmrl}"
GS_ROOT=/gscratch/robotics/harine/RoboMonkey_data/train/chunksum
GS_DIR="$GS_ROOT/$POLICY"
HOME_CKDIR="diffusion_policy/data/train/chunksum/$POLICY/checkpoints"
mkdir -p "$GS_DIR/checkpoints"

echo "[sbatch] host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-?} POLICY=$POLICY -> $GS_DIR"
nvidia-smi -L || true

export CONDA_ENV="${CONDA_ENV:-monkey-verifier}"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
export PYTHONPATH="$PWD/diffusion_policy:${PYTHONPATH:-}"

# --- Bootstrap gscratch latest.ckpt from the newest LOADABLE checkpoint ---
python - "$POLICY" "$HOME_CKDIR" "$GS_DIR/checkpoints" <<'PY'
import sys, os, glob, shutil, torch
pol, home, gs = sys.argv[1:4]
def ok(p):
    try: torch.load(p, map_location="cpu", weights_only=False); return True
    except Exception: return False
gs_latest = os.path.join(gs, "latest.ckpt")
if os.path.isfile(gs_latest) and ok(gs_latest):
    print(f"[bootstrap] gscratch latest.ckpt already valid -> resume in place"); sys.exit(0)
cands = sorted(glob.glob(f"{home}/*.ckpt"), key=os.path.getmtime, reverse=True)
for c in cands:
    if ok(c):
        shutil.copy(c, gs_latest); print(f"[bootstrap] copied {os.path.basename(c)} -> gscratch latest.ckpt"); sys.exit(0)
print("[bootstrap] WARNING: no loadable home checkpoint found; starting fresh");
PY

case "$POLICY" in
    noise0.0) LEVELS=0.0; RUN_COND=0;;
    noise0.4) LEVELS=0.4; RUN_COND=0;;
    noise0.8) LEVELS=0.8; RUN_COND=0;;
    tmrl)     LEVELS="";  RUN_COND=1;;
    *) echo "ERROR bad POLICY '$POLICY'"; exit 1;;
esac
MAX_GRADIENT_STEPS="${MAX_GRADIENT_STEPS:-7000}"
# MUST match the checkpoint being resumed: cond_pos_emb = n_obs_steps(2) +
# (max_actions-1). 16 = MAZE-consistent (cond_pos_emb=17). Old chunk-sum
# checkpoints trained at 8 (cond_pos_emb=9) are NOT resumable at 16 -> they must
# be retrained. Set MAX_ACTIONS=8 only to resume one of those legacy runs.
MAX_ACTIONS="${MAX_ACTIONS:-16}"

export ROBOMONKEY_KV_PREFIX="${ROBOMONKEY_KV_PREFIX:-0}"
export ROBOMONKEY_KV_CHUNK="${ROBOMONKEY_KV_CHUNK:-16}"
export ROBOMONKEY_IMAGE_FEAT_CACHE_SIZE="${ROBOMONKEY_IMAGE_FEAT_CACHE_SIZE:-512}"
export ROBOMONKEY_PROC_IMAGE_CACHE_SIZE="${ROBOMONKEY_PROC_IMAGE_CACHE_SIZE:-512}"
export ROBOMONKEY_PAIRED_CHUNK_SIZE="${ROBOMONKEY_PAIRED_CHUNK_SIZE:-8}"

export EXTRA_OVERRIDES="+training.max_gradient_steps=${MAX_GRADIENT_STEPS} \
task.dataset.use_cache=true \
task.dataset.dataset_dir=/gscratch/robotics/harine/data/eggplant_in_basket_complete \
+policy.verifier.score_executed_chunk=true \
+policy.verifier.score_agg=sum \
policy.max_actions=${MAX_ACTIONS} \
hydra.run.dir=${GS_DIR}"

DEVICE=cuda:0 LEVELS="$LEVELS" RUN_COND="$RUN_COND" \
    bash scriptsv2/train_search/train_noise_sweep.sh

echo "[sbatch] resume ($POLICY) done -> ${GS_DIR}/checkpoints/latest.ckpt"
