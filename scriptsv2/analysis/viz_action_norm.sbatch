#!/bin/bash
#SBATCH --job-name=viznorm
#SBATCH --cpus-per-task=6
#SBATCH --mem=56G
#SBATCH --time=8:00:00
#SBATCH --output=/mmfs1/home/harine/RoboMonkey/data/eval/action_norm_aug3/_logs/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# One policy: offline candidate sampling (demo states -> N candidates -> Q -> a*)
# THEN render_q overlay videos (candidates colored by Q-rank + chosen + a* expert
# dashed white + Q text), THEN organize into <task>/videos/ with a labeled name.
# Driven by --export vars:
#   CKPT, ZARR, TASK, INSTRUCTION, TEXT_EMB, N_SAMPLES, NUM_EPISODES,
#   TASKDIR (eggplant|spoon_on_towel|stack_cube), DEMOS (sub100|sub50), OUT_ROOT
# -----------------------------------------------------------------------------
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
REPO="$PWD"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-dqc_jax}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"; export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"; export DISPLAY=""
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/${CONDA_ENV:-dqc_jax}/lib:${LD_LIBRARY_PATH:-}"
export DIFFUSION_POLICY_ROOT="$REPO/diffusion_policy"; export PYTHONPATH="$DIFFUSION_POLICY_ROOT:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}"
export DQC_CKPT_DIR="${DQC_CKPT_DIR:-/gscratch/robotics/harine/dqc_checkpoint/sd000s_137913.0.20260605_163017}"

N_SAMPLES="${N_SAMPLES:-16}"; NUM_EPISODES="${NUM_EPISODES:-10}"; FPS="${FPS:-3}"
WORK="$OUT_ROOT/$TASKDIR/_work/$DEMOS"
VIDDIR="$OUT_ROOT/$TASKDIR/videos"
mkdir -p "$WORK" "$VIDDIR" "$OUT_ROOT/_logs"

echo "[viznorm] $TASKDIR/$DEMOS  N=$N_SAMPLES eps=$NUM_EPISODES  ckpt=$CKPT"

# (1) offline candidate sampling -> search_q/*.npz (candidates + values + a* + frames)
python scriptsv2/q_fn_eval/sample_offline_candidates.py \
  --ckpt "$CKPT" --zarr "$ZARR" --policy-kind search --split train \
  --config-label "search_train_N${N_SAMPLES}" --out-dir "$WORK" \
  --n-samples "$N_SAMPLES" --n-action-steps 4 --stride 4 \
  --num-episodes "$NUM_EPISODES" --task "$TASK" --instruction "$INSTRUCTION" \
  --dqc-text-emb "$TEXT_EMB" || { echo "[viznorm] dump FAILED"; exit 1; }

# (2) render Q-overlay videos (candidates by Q-rank + chosen + a* expert + Q text)
python scriptsv2/eval_search/render_q.py "$WORK/search_q" --out-dir "$WORK/videos" --fps "$FPS" \
  || { echo "[viznorm] render FAILED"; exit 1; }

# (3) organize into <task>/videos/ with a labeled name:
#     train_<demos>_seed<seed>_N<N>_<status>.mp4   (render_q names ep###_seed<seed>__N<N>_<status>.mp4)
n=0
for f in "$WORK"/videos/*.mp4; do
  [ -f "$f" ] || continue
  b="$(basename "$f")"
  seed="$(echo "$b" | sed -nE 's/.*_seed([0-9]+)__.*/\1/p')"
  rest="$(echo "$b" | sed -nE 's/.*_seed[0-9]+__(N[0-9]+_.*)/\1/p')"
  cp -f "$f" "$VIDDIR/train_${DEMOS}_seed${seed}_${rest}"
  n=$((n+1))
done
echo "[viznorm] done -> $VIDDIR ($n videos labeled train_${DEMOS}_*)"
