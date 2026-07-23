#!/bin/bash
#SBATCH --job-name=vidcmp_llava
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=data/eval/_verifier_comparison/videos/slurm-%x-%j.out
#SBATCH --error=data/eval/_verifier_comparison/videos/slurm-%x-%j.out
#
# First-8-eval-seed rollout videos for the two LLaVA eval_search campaigns:
#   C3  BC Best-of-N            (DiffusionUnet BC + in-process LLaVA verifier)
#   C4  diffusion SEARCH policy (noise0.0, 7k ckpt, sum-of-chunk, LLaVA verifier)
# Both run in the monkey-verifier env on one l40s GPU, 8 episodes (seeds 1000-1007).
# The DQC counterparts run via eval_bc_dqc.sbatch (C3) and eval_dqc.sbatch (C2);
# C4 has no DQC verifier (native search policy is LLaVA-only).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"

N="${N:-8}"; EP="${EP:-8}"; SS="${SS:-1000}"
VID_ROOT="data/eval/_verifier_comparison/videos"
mkdir -p "$VID_ROOT"

BC_CKPT="${BC_CKPT:-$(ls -td diffusion_policy/data/outputs/*/*robomonkey_eggplant_bc_*/checkpoints/best.ckpt 2>/dev/null | head -1)}"
C4_CKPT="${C4_CKPT:-/gscratch/robotics/harine/RoboMonkey_data/train/chunksum/noise0.0/checkpoints/step_0007000.ckpt}"

export DIFFUSION_POLICY_ROOT="$PWD/diffusion_policy"
export MONKEY_VERIFIER_SRC="$PWD/monkey-verifier/src"
export MODEL_DIR="$PWD/monkey-verifier/model_dir"
export ROBOMONKEY_KV_PREFIX=0

echo "############ C3 BC LLaVA (argmax, ddpm, N=$N, $EP ep) ############"
echo "[c3] bc ckpt = $BC_CKPT"
CHECKPOINT="$BC_CKPT" \
  N_LIST="$N" NUM_EPISODES="$EP" START_SEED="$SS" \
  MODE=argmax BC_SAMPLER=ddpm \
  VIZ_Q=1 VIZ_Q_EPISODES="$EP" SAVE_VIDEOS="$EP" VIDEO_FPS=10 \
  NUM_WORKERS=1 CONDA_ENV=monkey-verifier \
  BC_LABEL=vidcmp_c3_llava FORCE="${FORCE:-1}" \
  bash scriptsv2/eval_search/eval_search_bc.sh || echo "[c3] FAILED (see log above)"

echo "############ C4 diffusion search LLaVA sum-of-chunk (noise0.0 7k, nas8, N=$N, $EP ep) ############"
echo "[c4] ckpt = $C4_CKPT"
C4_OUT="$VID_ROOT/c4_diffusion_llava_sum_n${N}"
CONDA_ENV=monkey-verifier DEVICE=cuda:0 CUDA_VISIBLE_DEVICES=0 \
  MODE=softmax SOFTMAX_TEMP=1.0 \
  SCORE_AGG=sum SCORE_WINDOW=executed \
  N_SAMPLES="$N" N_ACTION_STEPS=8 \
  START_SEED="$SS" MAX_STEPS=120 USE_EMA=1 \
  TASK=widowx_put_eggplant_in_basket \
  VIZ_Q=1 VIZ_Q_EPISODES="$EP" SAVE_VIDEOS="$EP" VIDEO_FPS=10 \
  bash scriptsv2/eval_search/eval_search.sh "$C4_CKPT" "$EP" "$C4_OUT" \
  || echo "[c4] FAILED (see log above)"

echo "[vidcmp_llava] done"
