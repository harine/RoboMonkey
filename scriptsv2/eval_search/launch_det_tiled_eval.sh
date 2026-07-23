#!/bin/bash
# -----------------------------------------------------------------------------
# DETERMINISTIC, TILED true-horizon BoN eval on the sub50 search policies:
#   spoon 90k, stack 60k, eggplant 90k.   N=1..64, 50 ep, seeds 1000-1049.
# Fixes baked in: env-TimeLimit override (MAX_STEPS is honored), deterministic
# cuDNN/JAX + seeded softmax, object->goal distance metric, and TILED_ENVS=8
# (8 episodes run concurrently per cell). Fast SR+distance path (no videos).
#
#   STEPS_MODE=native  -> spoon/stack 60, eggplant 120 (task-native horizon)
#   STEPS_MODE=500     -> all three at 500 steps (true long horizon)
# Output -> data/eval/<task_folder>/<ck>_st<steps>_n<N>_seed1000_ep50/eval_log.json
# Idempotent. Max parallelism: all 21 cells spread round-robin over free GPUs.
#   STEPS_MODE=native bash scriptsv2/eval_search/launch_det_tiled_eval.sh   (DRY=1 to preview)
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
SB=scriptsv2/eval_search/eval_search_dqc_chunk.sbatch
MT=/gscratch/robotics/harine/RoboMonkey_data/train/rmse_multitask
FL=/gscratch/robotics/harine/RoboMonkey_data/train/rmse_filtered
N_LIST="1 2 4 8 16 32 64"
STEPS_MODE="${STEPS_MODE:-native}"
TILED="${TILED_ENVS:-8}"
DRY="${DRY:-0}"
# Route to robotics + cse + the cluster-wide ckpt (checkpoint) partition — NOT weirdlab.
# ckpt runs on idle a40/a100s across the cluster (preemptible) -> max parallelism.
ROUTES=( "robotics|gpu-a100|gpu:a100:1" "cse|gpu-a100|gpu:a100:1" "cse|gpu-l40s|gpu:l40s:1" "robotics|ckpt|gpu:a40:1" "robotics|ckpt|gpu:a100:1" )
# task_folder|TASK_ENV|ckpt_dir|milestone|native_steps
POLS=(
  "spoon_on_towel|widowx_spoon_on_towel|$MT/diff_search_rmse_spoon_sub50_diverse|90|60"
  "stack_cube|widowx_stack_cube|$MT/diff_search_rmse_stack_sub50_diverse|60|60"
  "eggplant|widowx_put_eggplant_in_basket|$FL/diff_search_rmse_sub50_diverse_data|90|120"
)
find_ckpt(){ ls "$1"/checkpoints/step_*.ckpt 2>/dev/null | awk -v M=$(($2*1000)) \
  '{n=$0;sub(/.*step_0*/,"",n);sub(/\.ckpt/,"",n);n+=0; if(n>=M-200&&n<=M+1200){d=(n>M)?n-M:M-n; if(b==""||d<bd){b=$0;bd=d}}} END{if(b)print b}'; }
QN="$(squeue -u "$USER" -h -o '%j')"; i=0; nsub=0; sk=0
for spec in "${POLS[@]}"; do
  IFS='|' read -r folder tenv dir mk native <<< "$spec"
  ck=$(find_ckpt "$dir" "$mk"); [ -z "$ck" ] && { echo "NO CKPT $folder ${mk}k"; continue; }
  steps=$([ "$STEPS_MODE" = 500 ] && echo 500 || echo "$native")
  for N in $N_LIST; do
    out="data/eval/$folder/${folder}_sub50_${mk}k_st${steps}_n${N}_seed1000_ep50"
    jn="det_${folder:0:2}${mk}_${steps}_n${N}"
    [ -s "$out/eval_log.json" ] && { sk=$((sk+1)); continue; }
    grep -qxF "$jn" <<<"$QN" && { sk=$((sk+1)); continue; }
    IFS='|' read -r a p g <<< "${ROUTES[$((i%${#ROUTES[@]}))]}"; i=$((i+1))
    if [ "$DRY" = 1 ]; then echo "WOULD $jn ($folder ${mk}k steps=$steps N=$N tiled=$TILED -> $a/$p)"; nsub=$((nsub+1)); continue; fi
    jid=$(sbatch --parsable --job-name="$jn" --account="$a" --partition="$p" --gres="$g" \
      --cpus-per-task=6 --mem=56G --time=8:00:00 \
      --export=ALL,CHECKPOINT="$ck",N_SAMPLES=$N,MODE=softmax,SOFTMAX_TEMP=1.0,NUM_EPISODES=50,START_SEED=1000,DQC_RESTORE_EPOCH=100000,TASK=$tenv,MAX_STEPS=$steps,TILED_ENVS=$TILED,SAVE_VIDEOS=0,VIZ_Q=0,RENDER_Q=0,OUT_DIR="$out" "$SB")
    echo "sub $jn -> $jid ($a/$p steps=$steps)"; nsub=$((nsub+1))
  done
done
echo "=== det tiled eval ($STEPS_MODE, tiled=$TILED): submitted=$nsub skipped=$sk ==="
