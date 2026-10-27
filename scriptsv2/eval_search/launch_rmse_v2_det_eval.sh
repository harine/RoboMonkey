#!/bin/bash
# -----------------------------------------------------------------------------
# Deterministic SR eval of the retrained rmse_v2 search policies (sub100/sub50,
# val=testset, trained by scriptsv2/train_search/launch_retrain_sub100_50.sh) at
# EACH 10k checkpoint: 10k / 20k / 30k / 40k / 50k.
#   SR cells: 50 ep, tiled(8), MODE=argmax, full N=1..64 -> eval_log.json (SR + dist)
# 6 runs x 5 ckpts x 7 N = 210 cells. Idempotent: skips cells whose eval_log.json
# exists or whose job is already queued, and skips checkpoints not yet written
# (re-run as training progresses). Reuses eval_search_dqc_chunk.sbatch (TASK ->
# baked per-task DQC-chunk verifier + text-emb; PG params inert at inference).
# Routes off weirdlab (trainings live there) and off the CPU-capped robotics-a40:
# cse a100/l40s + robotics a100 + ckpt.
#   DRY=1 bash scriptsv2/eval_search/launch_rmse_v2_det_eval.sh                 # preview
#   bash scriptsv2/eval_search/launch_rmse_v2_det_eval.sh                       # submit
#   RUNS_ONLY=rmse_v2_eggplant_sub50 N_LIST=16 STEPS_K=10 bash ...             # one cell
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
SB=scriptsv2/eval_search/eval_search_dqc_chunk.sbatch
TR=/gscratch/robotics/harine/RoboMonkey_data/train/rmse_v2
OUT_ROOT="${OUT_ROOT:-data/eval/rmse_v2_det}"
N_LIST="${N_LIST:-1 2 4 8 16 32 64}"
STEPS_K="${STEPS_K:-10 20 30 40 50}"
MODE="${MODE:-argmax}"
NUM_EPISODES="${NUM_EPISODES:-50}"
DRY="${DRY:-0}"
RUNS_ONLY="${RUNS_ONLY:-}"

declare -A TENV NSTEP
TENV[eggplant]=widowx_put_eggplant_in_basket; NSTEP[eggplant]=120
TENV[stack]=widowx_stack_cube;                NSTEP[stack]=60
TENV[spoon]=widowx_spoon_on_towel;            NSTEP[spoon]=60

RUNS=(
  "rmse_v2_eggplant_sub100:eggplant" "rmse_v2_eggplant_sub50:eggplant"
  "rmse_v2_stack_sub100:stack"       "rmse_v2_stack_sub50:stack"
  "rmse_v2_spoon_sub100:spoon"       "rmse_v2_spoon_sub50:spoon"
)
ROUTES=( "cse|gpu-a100|gpu:a100:1" "cse|gpu-l40s|gpu:l40s:1" "robotics|gpu-a100|gpu:a100:1" "robotics|ckpt|gpu:a40:1" "robotics|ckpt|gpu:a100:1" )

# resolve the checkpoint file for target step k*1000 (handles step_%07d and the +1 variant)
resolve_ck() { local dir="$1" base="$2" c
  for c in "$(printf '%s/checkpoints/step_%07d.ckpt' "$dir" "$base")" \
           "$(printf '%s/checkpoints/step_%07d.ckpt' "$dir" "$((base+1))")"; do
    [ -f "$c" ] && { echo "$c"; return 0; }
  done
  ls "$dir"/checkpoints/step_*.ckpt >/dev/null 2>&1 || return 1
  # The 10k-mark checkpoint drifts (+1 step per 10k: 10001, 20002, 30003, 40004, ...).
  # Accept the nearest checkpoint within 100 steps of the target -- checkpoints are
  # 1000 apart, so this uniquely identifies the intended 10k-mark ckpt.
  ls "$dir"/checkpoints/step_*.ckpt | awk -v t="$base" -F'step_|\\.ckpt' \
    '{d=$2-t; if(d<0)d=-d; if(NR==1||d<b){b=d;f=$0}} END{if(b<=100)print f}'
}

QN="$(squeue -u "$USER" -h -o '%j')"
TRAIN_Q="$(squeue -u "$USER" -h -o '%j' 2>/dev/null | grep '^rmse_v2_')"  # runs still training
i=0; nsub=0; sk=0; nock=0
for spec in "${RUNS[@]}"; do
  run="${spec%%:*}"; task="${spec##*:}"
  [ -n "$RUNS_ONLY" ] && [ "$run" != "$RUNS_ONLY" ] && continue
  tenv=${TENV[$task]}; steps=${NSTEP[$task]}
  for k in $STEPS_K; do
    ck="$(resolve_ck "$TR/$run" "$((k*1000))")"
    # 50k = FINAL model. Training caps at max_gradient_steps=50000 without saving a
    # 50k-boundary step-ckpt (highest is ~49006), so the true 50k model is
    # latest.ckpt. Use it only once the run has COMPLETED (not in the training queue),
    # otherwise latest.ckpt is a mid-training snapshot.
    if [ -z "$ck" ] && [ "$k" -ge 50 ] && [ -f "$TR/$run/checkpoints/latest.ckpt" ] \
       && ! grep -qxF "$run" <<<"$TRAIN_Q"; then
      ck="$TR/$run/checkpoints/latest.ckpt"
    fi
    [ -n "$ck" ] && [ -f "$ck" ] || { echo "NO CKPT $run ${k}k"; nock=$((nock+1)); continue; }
    for N in $N_LIST; do
      out="$OUT_ROOT/$run/${k}k/sr_n${N}"
      jn="rv2_${run#rmse_v2_}_${k}k_n${N}"; jn="${jn//_sub/_s}"
      [ -s "$out/eval_log.json" ] && { sk=$((sk+1)); continue; }
      grep -qxF "$jn" <<<"$QN" && { sk=$((sk+1)); continue; }
      IFS='|' read -r a p g <<< "${ROUTES[$((i%${#ROUTES[@]}))]}"; i=$((i+1))
      exp="CHECKPOINT=$ck,N_SAMPLES=$N,MODE=$MODE,SOFTMAX_TEMP=1.0,NUM_EPISODES=$NUM_EPISODES,SPLIT=test,DQC_RESTORE_EPOCH=100000,TASK=$tenv,MAX_STEPS=$steps,TILED_ENVS=8,VIZ_Q=0,RENDER_Q=0,SAVE_VIDEOS=0,VIDEO_FPS=10,OUT_DIR=$out"
      [ "$DRY" = 1 ] && { echo "WOULD SR $jn ($run ${k}k N=$N -> $a/$p)  ck=${ck##*/}"; nsub=$((nsub+1)); continue; }
      jid=$(sbatch --parsable --job-name="$jn" --account="$a" --partition="$p" --gres="$g" \
            --cpus-per-task=6 --mem=56G --time=12:00:00 --export=ALL,"$exp" "$SB")
      echo "sub $jn -> $jid ($a/$p) ck=${ck##*/}"; nsub=$((nsub+1))
    done
  done
done
echo "=== rmse_v2 det eval [mode=$MODE N='$N_LIST' steps='${STEPS_K}k']: submitted=$nsub skipped=$sk no_ckpt=$nock ==="
