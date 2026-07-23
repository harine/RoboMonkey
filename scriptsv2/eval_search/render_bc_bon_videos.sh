#!/bin/bash
# -----------------------------------------------------------------------------
# Render BC-policy BoN Q-overlay videos (5 episodes, N=1/8/64) for every
# task+sub combo, at each BC policy's BEST n=1 checkpoint. The prior bon_* evals
# ran with VIZ_Q=0/RENDER_Q=0 (no videos), so this re-runs a SHORT 5-episode eval
# with VIZ_Q=1 VIZ_Q_EPISODES=5 RENDER_Q=1 -> search_q/videos/ep<ep>_seed<seed>__N<N>_<succ/trunc>.mp4.
# Correct per-task SimplerEnv TASK + DQC verifier embedding/instruction. Softmax BoN, ddpm sampler.
# organize_bc_bon_videos.sh then copies them into the search video folders' bc/ subfolder.
#   bash scriptsv2/eval_search/render_bc_bon_videos.sh          # submit
#   DRY=1 bash .../render_bc_bon_videos.sh                      # preview
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
SB=scriptsv2/eval_search/eval_bc_dqc_chunk.sbatch
BCD=/mmfs1/home/harine/RoboMonkey/diffusion_policy/data/train/bc_denoiser
Q=/mmfs1/home/harine/RoboMonkey/scriptsv2/q_fn_eval
N_LIST="1 8 64"
DRY="${DRY:-0}"
# free GPUs (hyakalloc 2026-07-06): weirdlab l40 x8, weirdlab a40 x5, robotics a40 x3
ROUTES=( "weirdlab|gpu-l40|gpu:l40:1" "weirdlab|gpu-a40|gpu:a40:1" "robotics|gpu-a40|gpu:a40:1" )

# task | sub | checkpoint (best n=1)
POLICIES=(
  "eggplant|sub100|$BCD/eggplant_bc_unet_h8_50k_sub100_diverse_nobs1/checkpoints/step_0030001.ckpt"
  "eggplant|sub50|$BCD/eggplant_bc_unet_h8_50k_sub50_diverse_nobs1/checkpoints/latest.ckpt"
  "eggplant|sub25|$BCD/eggplant_bc_unet_h8_50k_sub25_diverse_nobs1/checkpoints/step_0030001.ckpt"
  "stack|sub100|$BCD/bc_unet_h8_50k_stack_sub100_nobs1/checkpoints/step_0040005.ckpt"
  "stack|sub50|$BCD/bc_unet_h8_50k_stack_sub50_nobs1/checkpoints/step_0030002.ckpt"
  "stack|sub25|$BCD/bc_unet_h8_50k_stack_sub25_nobs1/checkpoints/step_0040020.ckpt"
  "spoon|sub100|$BCD/bc_unet_h8_50k_spoon_sub100_nobs1/checkpoints/step_0020002.ckpt"
  "spoon|sub50|$BCD/bc_unet_h8_50k_spoon_sub50_nobs1/checkpoints/step_0010001.ckpt"
  "spoon|sub25|$BCD/bc_unet_h8_50k_spoon_sub25_nobs1/checkpoints/step_0010002.ckpt"
)
task_env () {  # sets TASK EMB INSTR for $1
  case "$1" in
    eggplant) TASK=widowx_put_eggplant_in_basket; EMB=$Q/task_emb_use3.npy;       INSTR="put the eggplant in the basket";;
    stack)    TASK=widowx_stack_cube;             EMB=$Q/task_emb_stack_use3.npy; INSTR="stack the green block on the yellow block";;
    spoon)    TASK=widowx_spoon_on_towel;         EMB=$Q/task_emb_spoon_use3.npy; INSTR="put the spoon on the towel";;
  esac
}
QN="$(squeue -u "$USER" -h -o '%j')"; i=0; sub=0; sk=0
for spec in "${POLICIES[@]}"; do
  IFS='|' read -r task s ckpt <<< "$spec"
  [ -f "$ckpt" ] || { echo "MISSING ckpt: $ckpt -- skip $task $s"; continue; }
  task_env "$task"
  for N in $N_LIST; do
    jn="vidbc_${task:0:2}${s#sub}_n${N}"
    out="data/eval/bc_dqc_chunk/vidbc_${task}_${s}_n${N}_seed1000_ep5"
    if [ -s "$out/search_q/videos" ] 2>/dev/null || ls "$out"/search_q/videos/*.mp4 >/dev/null 2>&1; then sk=$((sk+1)); continue; fi
    grep -qxF "$jn" <<< "$QN" && { sk=$((sk+1)); continue; }
    IFS='|' read -r a p g <<< "${ROUTES[$((i % ${#ROUTES[@]}))]}"; i=$((i+1))
    if [ "$DRY" = 1 ]; then echo "WOULD $jn ($task $s N=$N -> $a/$p)"; sub=$((sub+1)); continue; fi
    jid=$(sbatch --parsable --job-name="$jn" --account="$a" --partition="$p" --gres="$g" \
          --cpus-per-task=2 --mem=16G \
          --export=ALL,CHECKPOINT="$ckpt",N_SAMPLES="$N",NUM_EPISODES=5,START_SEED=1000,BC_SAMPLER=ddpm,MODE=softmax,SOFTMAX_TEMP=1.0,VIZ_Q=1,VIZ_Q_EPISODES=5,RENDER_Q=1,SAVE_VIDEOS=0,TASK="$TASK",DQC_TEXT_EMB="$EMB",VERIFIER_INSTRUCTION="$INSTR",OUT_DIR="$out" \
          "$SB")
    echo "submitted $jn -> $jid ($task $s N=$N, $a/$p)"; sub=$((sub+1))
  done
done
echo "=== BC video renders: submitted=$sub skipped=$sk ==="
