#!/bin/bash
# -----------------------------------------------------------------------------
# Deterministic eval of the 8 PG-loss sub50 search policies at the 10k & 20k
# checkpoints, to compare SR against the non-PG (rmse-loss) sub50 baseline.
#   SR  cells: 50 ep, tiled(8), MODE=argmax, N in N_LIST -> eval_log.json (SR + dist)
#   VIZ cells: 10 ep, non-tiled, VIZ_Q=1 RENDER_Q=1 at N=VIZ_N -> 10 videos + npz
# PG params (pg_lambda/pg_mode) are training-only -> inert at inference; the ckpt
# just loads its model weights + baked per-task DQC verifier config. Reuses
# eval_search_dqc_chunk.sbatch verbatim (same interface as launch_det_eval2.sh).
# Routes: robotics/cse/ckpt for SR, weirdlab/robotics a40 for VIZ (NOT weirdlab for SR).
#   DRY=1 bash scriptsv2/eval_search/launch_pg_det_eval.sh                 # preview
#   bash scriptsv2/eval_search/launch_pg_det_eval.sh                       # full
#   RUNS_ONLY=pg_awbc_l0.1_eggplant_sub50 N_LIST=16 DO_VIZ=0 bash ...      # one cell
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
SB=scriptsv2/eval_search/eval_search_dqc_chunk.sbatch
PG=/gscratch/robotics/harine/RoboMonkey_data/train/pg_sub50
N_LIST="${N_LIST:-1 2 4 8 16 32 64}"
STEPS_K="${STEPS_K:-10 20}"
MODE="${MODE:-argmax}"
VIZ_N="${VIZ_N:-16}"
DRY="${DRY:-0}"; DO_SR="${DO_SR:-1}"; DO_VIZ="${DO_VIZ:-1}"
RUNS_ONLY="${RUNS_ONLY:-}"          # optional: restrict to one run name

declare -A TENV NSTEP
TENV[eggplant]=widowx_put_eggplant_in_basket; NSTEP[eggplant]=120
TENV[stack]=widowx_stack_cube;                NSTEP[stack]=60
TENV[spoon]=widowx_spoon_on_towel;            NSTEP[spoon]=60

RUNS=(
  "pg_awbc_l0.1_eggplant_sub50:eggplant"
  "pg_awbc_l0.1_stack_sub50:stack"
  "pg_awbc_l0.1_spoon_sub50:spoon"
  "pg_awbc_l0.3_spoon_sub50:spoon"
  "pg_reinf_l0.1_eggplant_sub50:eggplant"
  "pg_reinf_l0.1_stack_sub50:stack"
  "pg_reinf_l0.1_spoon_sub50:spoon"
  "pg_reinf_l0.3_spoon_sub50:spoon"
)
SR_ROUTES=( "robotics|gpu-a100|gpu:a100:1" "cse|gpu-a100|gpu:a100:1" "cse|gpu-l40s|gpu:l40s:1" "robotics|ckpt|gpu:a40:1" "robotics|ckpt|gpu:a100:1" )
VIZ_ROUTES=( "weirdlab|gpu-a40|gpu:a40:1" "robotics|gpu-a40|gpu:a40:1" )

QN="$(squeue -u "$USER" -h -o '%j')"
i=0; vi=0; nsub=0; sk=0; nock=0
for spec in "${RUNS[@]}"; do
  run="${spec%%:*}"; task="${spec##*:}"
  [ -n "$RUNS_ONLY" ] && [ "$run" != "$RUNS_ONLY" ] && continue
  tenv=${TENV[$task]}; steps=${NSTEP[$task]}
  for k in $STEPS_K; do
    ck="$PG/$run/checkpoints/$(printf 'step_%07d.ckpt' $((k*1000+1)))"
    [ -f "$ck" ] || { echo "NO CKPT $run ${k}k -> $ck"; nock=$((nock+1)); continue; }
    if [ "$DO_SR" = 1 ]; then
      for N in $N_LIST; do
        out="data/eval/pg_det/$run/${k}k/sr_n${N}"
        jn="pgsr_${run#pg_}_${k}k_n${N}"; jn="${jn//_sub50/}"
        [ -s "$out/eval_log.json" ] && { sk=$((sk+1)); continue; }
        grep -qxF "$jn" <<<"$QN" && { sk=$((sk+1)); continue; }
        IFS='|' read -r a p g <<< "${SR_ROUTES[$((i%${#SR_ROUTES[@]}))]}"; i=$((i+1))
        exp="CHECKPOINT=$ck,N_SAMPLES=$N,MODE=$MODE,SOFTMAX_TEMP=1.0,NUM_EPISODES=50,START_SEED=1000,DQC_RESTORE_EPOCH=100000,TASK=$tenv,MAX_STEPS=$steps,TILED_ENVS=8,TILED_SEED_PER_ENV=1,VIZ_Q=0,RENDER_Q=0,SAVE_VIDEOS=0,VIDEO_FPS=10,OUT_DIR=$out"
        [ "$DRY" = 1 ] && { echo "WOULD SR  $jn ($run ${k}k N=$N -> $a/$p)"; nsub=$((nsub+1)); continue; }
        jid=$(sbatch --parsable --job-name="$jn" --account="$a" --partition="$p" --gres="$g" \
              --cpus-per-task=6 --mem=56G --time=12:00:00 --export=ALL,"$exp" "$SB")
        echo "sub SR  $jn -> $jid ($a/$p)"; nsub=$((nsub+1))
      done
    fi
    if [ "$DO_VIZ" = 1 ]; then
      out="data/eval/pg_det/$run/${k}k/viz_n${VIZ_N}"
      jn="pgvz_${run#pg_}_${k}k"; jn="${jn//_sub50/}"
      ls "$out"/search_q/videos/*.mp4 >/dev/null 2>&1 && { sk=$((sk+1)); continue; }
      grep -qxF "$jn" <<<"$QN" && { sk=$((sk+1)); continue; }
      IFS='|' read -r a p g <<< "${VIZ_ROUTES[$((vi%${#VIZ_ROUTES[@]}))]}"; vi=$((vi+1))
      exp="CHECKPOINT=$ck,N_SAMPLES=$VIZ_N,MODE=$MODE,SOFTMAX_TEMP=1.0,NUM_EPISODES=10,START_SEED=1000,DQC_RESTORE_EPOCH=100000,TASK=$tenv,MAX_STEPS=$steps,TILED_ENVS=1,VIZ_Q=1,VIZ_Q_EPISODES=10,RENDER_Q=1,SAVE_VIDEOS=0,VIDEO_FPS=10,OUT_DIR=$out"
      [ "$DRY" = 1 ] && { echo "WOULD VIZ $jn ($run ${k}k N=$VIZ_N -> $a/$p)"; nsub=$((nsub+1)); continue; }
      jid=$(sbatch --parsable --job-name="$jn" --account="$a" --partition="$p" --gres="$g" \
            --cpus-per-task=6 --mem=56G --time=12:00:00 --export=ALL,"$exp" "$SB")
      echo "sub VIZ $jn -> $jid ($a/$p)"; nsub=$((nsub+1))
    fi
  done
done
echo "=== pg det eval [mode=$MODE N='$N_LIST' steps='${STEPS_K}k']: submitted=$nsub skipped=$sk no_ckpt=$nock ==="
