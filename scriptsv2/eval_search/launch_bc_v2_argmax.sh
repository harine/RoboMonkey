#!/bin/bash
# -----------------------------------------------------------------------------
# ARGMAX BC baseline for the rmse_v2 comparison. The archived BC+DQC evals are
# SOFTMAX (not comparable to the argmax rmse_v2 search table for N>1), so re-run
# the 6 data-matched n_obs=1 BC denoisers at their best-by-N=1 checkpoint through
# the SAME DQC-chunk verifier with MODE=argmax, ddpm candidates, N=1..64, 50 ep
# seeds 1000-1049 -> apples-to-apples with rmse_v2_det.
#   DRY=1 bash scriptsv2/eval_search/launch_bc_v2_argmax.sh   # preview
#         bash scriptsv2/eval_search/launch_bc_v2_argmax.sh   # submit
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
SB=scriptsv2/eval_search/eval_bc_dqc_chunk.sbatch
BC=diffusion_policy/data/train/bc_denoiser
OUT_ROOT="${OUT_ROOT:-data/eval/bc_v2_det}"
N_LIST="${N_LIST:-1 2 4 8 16 32 64}"
MODE="${MODE:-argmax}"; NUM_EPISODES="${NUM_EPISODES:-50}"; DRY="${DRY:-0}"

# policy -> (bc run dir, best-by-N=1 target step, simplerenv task, emb, max_steps)
declare -A RUN STEP TENV EMB MST
RUN[eggplant_sub100]=eggplant_bc_unet_h8_50k_sub100_diverse_nobs1; STEP[eggplant_sub100]=30000
RUN[eggplant_sub50]=eggplant_bc_unet_h8_50k_sub50_diverse_nobs1;   STEP[eggplant_sub50]=50000
RUN[stack_sub100]=bc_unet_h8_50k_stack_sub100_nobs1;               STEP[stack_sub100]=40000
RUN[stack_sub50]=bc_unet_h8_50k_stack_sub50_nobs1;                 STEP[stack_sub50]=30000
RUN[spoon_sub100]=bc_unet_h8_50k_spoon_sub100_nobs1;               STEP[spoon_sub100]=20000
RUN[spoon_sub50]=bc_unet_h8_50k_spoon_sub50_nobs1;                 STEP[spoon_sub50]=10000
for k in eggplant_sub100 eggplant_sub50; do TENV[$k]=widowx_put_eggplant_in_basket; EMB[$k]=scriptsv2/q_fn_eval/task_emb_use3.npy; MST[$k]=120; done
for k in stack_sub100 stack_sub50; do TENV[$k]=widowx_stack_cube; EMB[$k]=scriptsv2/q_fn_eval/task_emb_stack_use3.npy; MST[$k]=60; done
for k in spoon_sub100 spoon_sub50; do TENV[$k]=widowx_spoon_on_towel; EMB[$k]=scriptsv2/q_fn_eval/task_emb_spoon_use3.npy; MST[$k]=60; done

resolve() { local d="$1" t="$2"
  [ "$t" -ge 50000 ] && { echo "$d/latest.ckpt"; return; }
  ls "$d"/step_*.ckpt 2>/dev/null | awk -F'step_|\\.ckpt' -v tg="$t" \
    '{dd=$2-tg;if(dd<0)dd=-dd;if(NR==1||dd<b){b=dd;f=$0}}END{if(b<=100)print f}'
}
ROUTES=( "robotics|ckpt|gpu:a40:1" "robotics|gpu-a100|gpu:a100:1" "robotics|ckpt|gpu:a100:1" "cse|gpu-l40s|gpu:l40s:1" )
QN="$(squeue -u "$USER" -h -o '%j')"
i=0; nsub=0; sk=0
for k in eggplant_sub100 eggplant_sub50 stack_sub100 stack_sub50 spoon_sub100 spoon_sub50; do
  ckpt="$(resolve "$BC/${RUN[$k]}/checkpoints" "${STEP[$k]}")"
  [ -n "$ckpt" ] && [ -f "$ckpt" ] || { echo "NO CKPT $k"; continue; }
  ckstep=$(echo "${ckpt##*/}" | grep -oE '[0-9]+' | head -1); [ -z "$ckstep" ] && ckstep=latest
  for N in $N_LIST; do
    out="$OUT_ROOT/$k/${MODE}_n${N}"
    jn="bcv2_${k}_${MODE}_n${N}"; jn="${jn//_sub/_s}"
    [ -s "$out/eval_log.json" ] && { sk=$((sk+1)); continue; }
    grep -qxF "$jn" <<<"$QN" && { sk=$((sk+1)); continue; }
    IFS='|' read -r a p g <<< "${ROUTES[$((i%${#ROUTES[@]}))]}"; i=$((i+1))
    exp="CHECKPOINT=$PWD/$ckpt,N_SAMPLES=$N,MODE=$MODE,SOFTMAX_TEMP=1.0,BC_SAMPLER=ddpm,NUM_EPISODES=$NUM_EPISODES,SPLIT=test,TASK=${TENV[$k]},MAX_STEPS=${MST[$k]},DQC_TEXT_EMB=$PWD/${EMB[$k]},OUT_DIR=$out"
    [ "$DRY" = 1 ] && { echo "WOULD $jn ($k ckstep=$ckstep N=$N -> $a/$p)"; nsub=$((nsub+1)); continue; }
    jid=$(sbatch --parsable --job-name="$jn" --account="$a" --partition="$p" --gres="$g" \
          --cpus-per-task=6 --mem=56G --time=12:00:00 --requeue --export=ALL,"$exp" "$SB")
    echo "sub $jn -> $jid ($a/$p) ck=${ckpt##*/}"; nsub=$((nsub+1))
  done
done
echo "=== bc_v2 argmax eval [N='$N_LIST']: submitted=$nsub skipped=$sk -> $OUT_ROOT/ ==="
