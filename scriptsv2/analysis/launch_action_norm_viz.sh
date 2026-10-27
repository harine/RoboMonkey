#!/bin/bash
# -----------------------------------------------------------------------------
# 10 Q-overlay videos per rmse_v2 policy (expert a* + candidates + chosen + Q),
# organized one folder per TASK:  data/eval/action_norm_aug3/<task>/videos/
#   train_<demos>_seed<seed>_N16_<status>.mp4
# 6 policies = {eggplant, spoon_on_towel, stack_cube} x {sub100, sub50}, final
# (latest.ckpt) checkpoint, offline replay of the TRAIN demos, N=16 candidates.
#   DRY=1 bash scriptsv2/analysis/launch_action_norm_viz.sh    # preview
#         bash scriptsv2/analysis/launch_action_norm_viz.sh    # submit
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
SB=scriptsv2/analysis/viz_action_norm.sbatch
TR=/gscratch/robotics/harine/RoboMonkey_data/train/rmse_v2
OUT_ROOT="${OUT_ROOT:-data/eval/action_norm_aug3}"
N_SAMPLES="${N_SAMPLES:-16}"; NUM_EPISODES="${NUM_EPISODES:-10}"
DRY="${DRY:-0}"
mkdir -p "$OUT_ROOT/_logs"

# task -> (taskdir, simplerenv task, instruction, text_emb, train-data prefix)
declare -A TDIR TENV INSTR EMB DPFX
TDIR[eggplant]=eggplant; TENV[eggplant]=widowx_put_eggplant_in_basket
INSTR[eggplant]="put the eggplant in the basket"; EMB[eggplant]=scriptsv2/q_fn_eval/task_emb_use3.npy
DPFX[eggplant]=strict_distance_success
TDIR[stack]=stack_cube; TENV[stack]=widowx_stack_cube
INSTR[stack]="stack the green block on the yellow block"; EMB[stack]=scriptsv2/q_fn_eval/task_emb_stack_use3.npy
DPFX[stack]=stack_cube_strict
TDIR[spoon]=spoon_on_towel; TENV[spoon]=widowx_spoon_on_towel
INSTR[spoon]="put the spoon on the towel"; EMB[spoon]=scriptsv2/q_fn_eval/task_emb_spoon_use3.npy
DPFX[spoon]=spoon_on_towel_strict

# route: idle checkpoint partition + robotics a100 (GPU for policy+verifier).
# cse a100/l40s are capped; ckpt is idle with plenty of GPUs.
ROUTES=( "robotics|ckpt|gpu:a40:1" "robotics|gpu-a100|gpu:a100:1" "robotics|ckpt|gpu:a100:1" )
i=0; nsub=0
for task in eggplant stack spoon; do
  for demos in sub100 sub50; do
    run="rmse_v2_${task}_${demos}"
    ckpt="$TR/$run/checkpoints/latest.ckpt"
    zarr="/gscratch/robotics/harine/data/${DPFX[$task]}_${demos}_diverse_start/${DPFX[$task]}_${demos}_diverse_start.zarr"
    [ -f "$ckpt" ] || { echo "NO CKPT $run"; continue; }
    [ -d "$zarr" ] || { echo "NO ZARR $zarr"; continue; }
    IFS='|' read -r a p g <<< "${ROUTES[$((i%${#ROUTES[@]}))]}"; i=$((i+1))
    exp="CKPT=$ckpt,ZARR=$zarr,TASK=${TENV[$task]},INSTRUCTION=${INSTR[$task]},TEXT_EMB=${EMB[$task]},N_SAMPLES=$N_SAMPLES,NUM_EPISODES=$NUM_EPISODES,TASKDIR=${TDIR[$task]},DEMOS=$demos,OUT_ROOT=$OUT_ROOT"
    jn="viznorm_${task}_${demos}"
    if [ "$DRY" = 1 ]; then echo "WOULD $jn -> $a/$p  (${TDIR[$task]}/videos, $NUM_EPISODES vids)  ck=latest zarr=${zarr##*/}"; nsub=$((nsub+1)); continue; fi
    jid=$(sbatch --parsable --job-name="$jn" --account="$a" --partition="$p" --gres="$g" \
          --export=ALL,"$exp" "$SB")
    echo "sub $jn -> $jid ($a/$p)"; nsub=$((nsub+1))
  done
done
echo "=== action_norm viz: submitted=$nsub -> $OUT_ROOT/<task>/videos/ ==="
