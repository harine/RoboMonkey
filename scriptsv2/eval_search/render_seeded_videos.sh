#!/bin/bash
# -----------------------------------------------------------------------------
# Render action+Q videos for the SEEDED-selection runs (reproducible softmax), on
# the SAME seeds as the SR eval (START_SEED=1000). BOTH episode lengths (500 & 120)
# and both sub50/sub100, all three tasks, search + BC. Step length is a TAG in the
# filename (not a folder). SEARCH renders ALL N (1..64); BC renders N=1/8/64.
# 10 videos/cell (seeds 1000-1009), NUM_EPISODES == VIZ_Q_EPISODES (only run what we
# render). Seeded selection is the DEFAULT.
#
# Search checkpoints resolved by MILESTONE (nearest step): stack 20k, spoon 30k,
# eggplant 70k. BC uses each policy's best-n=1 ckpt. Staged ->
# data/eval/_seeded_videos/<kind>_<task>_<sub>_<ck>_st<steps>_n<N>/ ;
# collect_seeded_videos.sh organizes into search_<task>/<sub>/ with the step tag.
#   bash scriptsv2/eval_search/render_seeded_videos.sh    (DRY=1 to preview)
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
SBS=scriptsv2/eval_search/eval_search_dqc_chunk.sbatch
SBB=scriptsv2/eval_search/eval_bc_dqc_chunk.sbatch
MT=/gscratch/robotics/harine/RoboMonkey_data/train/rmse_multitask
FL=/gscratch/robotics/harine/RoboMonkey_data/train/rmse_filtered
BCD=/mmfs1/home/harine/RoboMonkey/diffusion_policy/data/train/bc_denoiser
Q=/mmfs1/home/harine/RoboMonkey/scriptsv2/q_fn_eval
STAGE=data/eval/_seeded_videos
STEPS="500 120"
N_SEARCH="1 2 4 8 16 32 64"       # all N for search
N_BC="1 8 64"
VIZ=10                            # videos (=episodes=seeds 1000-1009) per cell
DRY="${DRY:-0}"
mkdir -p "$STAGE"
ROUTES=( "weirdlab|gpu-l40|gpu:l40:1|4" "weirdlab|gpu-l40s|gpu:l40s:1|4" "cse|gpu-l40s|gpu:l40s:1|4" "weirdlab|gpu-a40|gpu:a40:1|4" "cse|gpu-a100|gpu:a100:1|4" )

find_ckpt () {  # $1=dir $2=milestone(k)
  local d="$1/checkpoints" M=$(( $2 * 1000 ))
  ls "$d"/step_*.ckpt 2>/dev/null | awk -v M="$M" '
    { n=$0; sub(/.*step_0*/,"",n); sub(/\.ckpt/,"",n); n+=0;
      if (n>=M-200 && n<=M+1200){dd=(n>M)?n-M:M-n; if(best==""||dd<bd){best=$0;bd=dd}} }
    END{ if(best!="") print best }'
}
SEARCH=(  # task|sub|Mk|dir|TASK_ENV
  "stack_cube|sub50|20|$MT/diff_search_rmse_stack_sub50_diverse|widowx_stack_cube"
  "stack_cube|sub100|20|$MT/diff_search_rmse_stack_sub100_diverse|widowx_stack_cube"
  "spoon_on_towel|sub50|30|$MT/diff_search_rmse_spoon_sub50_diverse|widowx_spoon_on_towel"
  "spoon_on_towel|sub100|30|$MT/diff_search_rmse_spoon_sub100_diverse|widowx_spoon_on_towel"
  "eggplant|sub50|70|$FL/diff_search_rmse_sub50_diverse_data|widowx_put_eggplant_in_basket"
  "eggplant|sub100|70|$FL/diff_search_rmse_sub100_diverse_data|widowx_put_eggplant_in_basket"
)
BC=(  # task|sub|cklabel|ckpt|TASK_ENV|EMB|INSTR
  "stack_cube|sub50|30k|$BCD/bc_unet_h8_50k_stack_sub50_nobs1/checkpoints/step_0030002.ckpt|widowx_stack_cube|$Q/task_emb_stack_use3.npy|stack the green block on the yellow block"
  "stack_cube|sub100|40k|$BCD/bc_unet_h8_50k_stack_sub100_nobs1/checkpoints/step_0040005.ckpt|widowx_stack_cube|$Q/task_emb_stack_use3.npy|stack the green block on the yellow block"
  "spoon_on_towel|sub50|10k|$BCD/bc_unet_h8_50k_spoon_sub50_nobs1/checkpoints/step_0010001.ckpt|widowx_spoon_on_towel|$Q/task_emb_spoon_use3.npy|put the spoon on the towel"
  "spoon_on_towel|sub100|20k|$BCD/bc_unet_h8_50k_spoon_sub100_nobs1/checkpoints/step_0020002.ckpt|widowx_spoon_on_towel|$Q/task_emb_spoon_use3.npy|put the spoon on the towel"
  "eggplant|sub50|50k|$BCD/eggplant_bc_unet_h8_50k_sub50_diverse_nobs1/checkpoints/latest.ckpt|widowx_put_eggplant_in_basket|$Q/task_emb_use3.npy|put the eggplant in the basket"
  "eggplant|sub100|30k|$BCD/eggplant_bc_unet_h8_50k_sub100_diverse_nobs1/checkpoints/step_0030001.ckpt|widowx_put_eggplant_in_basket|$Q/task_emb_use3.npy|put the eggplant in the basket"
)
QN="$(squeue -u "$USER" -h -o '%j')"; i=0; nsub=0; sk=0
submit () {  # $1=jn $2=out $3=sbatch $4=exp
  ls "$2"/search_q/videos/*.mp4 >/dev/null 2>&1 && { sk=$((sk+1)); return; }
  grep -qxF "$1" <<< "$QN" && { sk=$((sk+1)); return; }
  IFS='|' read -r a p g c <<< "${ROUTES[$((i % ${#ROUTES[@]}))]}"; i=$((i+1))
  [ "$DRY" = 1 ] && { echo "WOULD $1 -> $a/$p"; nsub=$((nsub+1)); return; }
  jid=$(sbatch --parsable --job-name="$1" --account="$a" --partition="$p" --gres="$g" \
        --cpus-per-task="$c" --mem=32G --time=20:00:00 --export=ALL,"$4" "$3") \
    && { echo "sub $1 -> $jid"; nsub=$((nsub+1)); }
}
for spec in "${SEARCH[@]}"; do
  IFS='|' read -r task sub Mk dir tenv <<< "$spec"
  ck="$(find_ckpt "$dir" "$Mk")"; [ -z "$ck" ] && { echo "NO CKPT search $task $sub ${Mk}k"; continue; }
  for st in $STEPS; do for N in $N_SEARCH; do
    out="$STAGE/search_${task}_${sub}_${Mk}k_st${st}_n${N}"
    submit "vs_s${task:0:2}${sub#sub}_${st}_n${N}" "$out" "$SBS" \
      "CHECKPOINT=$ck,N_SAMPLES=$N,MODE=softmax,SOFTMAX_TEMP=1.0,NUM_EPISODES=$VIZ,START_SEED=1000,DQC_RESTORE_EPOCH=100000,TASK=$tenv,MAX_STEPS=$st,SAVE_VIDEOS=0,VIDEO_FPS=10,VIZ_Q=1,VIZ_Q_EPISODES=$VIZ,RENDER_Q=1,OUT_DIR=$out"
  done; done
done
for spec in "${BC[@]}"; do
  IFS='|' read -r task sub cklabel ck tenv emb instr <<< "$spec"
  [ -f "$ck" ] || { echo "MISSING BC $task $sub ($ck)"; continue; }
  for st in $STEPS; do for N in $N_BC; do
    out="$STAGE/bc_${task}_${sub}_${cklabel}_st${st}_n${N}"
    submit "vs_b${task:0:2}${sub#sub}_${st}_n${N}" "$out" "$SBB" \
      "CHECKPOINT=$ck,N_SAMPLES=$N,NUM_EPISODES=$VIZ,START_SEED=1000,BC_SAMPLER=ddpm,MODE=softmax,SOFTMAX_TEMP=1.0,MAX_STEPS=$st,SAVE_VIDEOS=0,VIDEO_FPS=10,VIZ_Q=1,VIZ_Q_EPISODES=$VIZ,RENDER_Q=1,TASK=$tenv,DQC_TEXT_EMB=$emb,VERIFIER_INSTRUCTION=$instr,OUT_DIR=$out"
  done; done
done
echo "=== seeded video renders: submitted=$nsub skipped=$sk ==="
