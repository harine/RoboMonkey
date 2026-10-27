#!/bin/bash
# -----------------------------------------------------------------------------
# Retrain the sub100 / sub50 diffusion SEARCH policies (rmse-loss, NO PG term)
# with the latest diffusion-search code changes, per the user's request:
#   "retrain the sub 100, 50 policies - 100 and 50 demos used in train dataset.
#    dont split this into val - use the additional test dataset for that purpose.
#    for all three tasks with the new changes for 50k steps. to wandb add the
#    normalized rmse curves. for each 10k checkpoint eval the success rate."
#
# Key differences vs the old rmse-loss runs:
#   * val = the held-out testset (*_testset_not_in_sub100), NOT a val_ratio split
#     -> VAL_DATASET_DIR set; Sim2RealImageMultiDataset forces val_ratio=0 so ALL
#        100/50 demos are used for training. (dataset code change.)
#   * "new changes" already committed (dropout-restore fix fb3172e + normalized
#     RMSE logging in train_mlp_image_workspace.py: {train,val}_action_mse_error_
#     {min,avg}_{normalized,unnormalized}).
#   * 50k gradient steps, plain rmse loss (PG_LAMBDA=0 -> PG block skipped entirely,
#     line 849 of search_policy_robomonkey.py; verifier loaded but unused).
#
# 6 runs = {eggplant, stack, spoon} x {sub100, sub50}. checkpoint_every=1000 saves
# step_00N0000.ckpt (NOT topk-pruned) -> 10k/20k/30k/40k/50k all persist for SR eval.
#
# Smoke first: SMOKE=1 bash scriptsv2/train_search/launch_retrain_sub100_50.sh
# Full:               bash scriptsv2/train_search/launch_retrain_sub100_50.sh
# -----------------------------------------------------------------------------
set -euo pipefail
cd /mmfs1/home/harine/RoboMonkey
SB=scriptsv2/train_search/train_search_pg.sbatch
SMOKE="${SMOKE:-0}"

# ---- common (frozen DQC chunk critic loaded for construction; unused at pg=0) ----
export CONFIG=robomonkey_eggplant_search_dqc
export CRITIC_DIR=/gscratch/robotics/harine/dqc_checkpoint/sd000s_137913.0.20260605_163017
export RESTORE_EPOCH=100000
export MAX_ACTIONS=16
export CONDA_ENV=dqc_jax
export NUM_WORKERS=6
export PG_LAMBDA=0.0          # plain rmse loss (PG term guarded off)
export PG_MODE=awbc          # irrelevant at pg_lambda=0
# Validate less often than the default every-10-epochs: the val action-RMSE uses
# N=16 diffusion+verifier search (~1-2 min/eval), and epochs are tiny (~9 steps for
# sub100). 100 epochs -> a val/RMSE point every ~450-900 steps (~55-110 pts/run).
export VAL_EVERY=100
export SAMPLE_EVERY=100

if [ "$SMOKE" = "1" ]; then
  export MAX_GRADIENT_STEPS="${SMOKE_STEPS:-5}"; export NUM_EPOCHS=1
  LOG_PROJECT_BASE=dqc_search_policy_smoke
  OUT_ROOT=/gscratch/robotics/harine/RoboMonkey_data/train/rmse_v2_smoke
  SUF=_smoke
else
  export MAX_GRADIENT_STEPS=50000; export NUM_EPOCHS=60000
  LOG_PROJECT_BASE=dqc_search_policy
  OUT_ROOT=/gscratch/robotics/harine/RoboMonkey_data/train/rmse_v2
  SUF=
fi
export LOG_PROJECT="$LOG_PROJECT_BASE"

# ---- per-task train dir / testset (val) dir / verifier text-emb / instruction ----
declare -A DS100 DS50 VAL EMB INSTR
DS100[eggplant]=/gscratch/robotics/harine/data/strict_distance_success_sub100_diverse_start
DS100[stack]=/gscratch/robotics/harine/data/stack_cube_strict_sub100_diverse_start
DS100[spoon]=/gscratch/robotics/harine/data/spoon_on_towel_strict_sub100_diverse_start
DS50[eggplant]=/gscratch/robotics/harine/data/strict_distance_success_sub50_diverse_start
DS50[stack]=/gscratch/robotics/harine/data/stack_cube_strict_sub50_diverse_start
DS50[spoon]=/gscratch/robotics/harine/data/spoon_on_towel_strict_sub50_diverse_start
VAL[eggplant]=/gscratch/robotics/harine/data/strict_distance_success_testset_not_in_sub100
VAL[stack]=/gscratch/robotics/harine/data/stack_cube_strict_testset_not_in_sub100
VAL[spoon]=/gscratch/robotics/harine/data/spoon_on_towel_strict_testset_not_in_sub100
EMB[eggplant]=/mmfs1/home/harine/RoboMonkey/scriptsv2/q_fn_eval/task_emb_use3.npy
EMB[stack]=/mmfs1/home/harine/RoboMonkey/scriptsv2/q_fn_eval/task_emb_stack_use3.npy
EMB[spoon]=/mmfs1/home/harine/RoboMonkey/scriptsv2/q_fn_eval/task_emb_spoon_use3.npy
INSTR[eggplant]="put the eggplant in the basket"
INSTR[stack]="stack the green block on the yellow block"
INSTR[spoon]="put the spoon on the towel"

# ---- run table: task demos acct part gres cpus ----
# robotics gpu-a40 assoc is CPU-capped by an unrelated pusht campaign, so route to
# robotics gpu-a100 (free, separate CPU pool) + weirdlab gpu-l40/l40s (fast Ada,
# free). All three have NO per-job wall limit. Per user: OK to use 4 weirdlab GPUs +
# the robotics node for faster training. --requeue + training.resume=True continue
# from latest.ckpt if interrupted.
if [ "$SMOKE" = "1" ]; then
  RUNS=(
    "eggplant sub50  weirdlab gpu-l40 gpu:l40:1 12"
    "eggplant sub100 weirdlab gpu-l40 gpu:l40:1 12"
  )
else
  RUNS=(
    "eggplant sub100 weirdlab gpu-l40 gpu:l40:1 12"
    "eggplant sub50  weirdlab gpu-l40 gpu:l40:1 12"
    "stack    sub100 weirdlab gpu-l40 gpu:l40:1 12"
    "stack    sub50  weirdlab gpu-l40 gpu:l40:1 12"
    "spoon    sub100 weirdlab gpu-l40 gpu:l40:1 12"
    "spoon    sub50  weirdlab gpu-l40 gpu:l40:1 12"
  )
fi

mkdir -p data/train/rmse_v2
for r in "${RUNS[@]}"; do
  read -r task demos acct part gres cpus <<< "$r"
  if [ "$demos" = "sub100" ]; then export DATASET_DIR="${DS100[$task]}"; else export DATASET_DIR="${DS50[$task]}"; fi
  export VAL_DATASET_DIR="${VAL[$task]}"
  export TEXT_EMB="${EMB[$task]}"
  export INSTRUCTION="${INSTR[$task]}"
  export RUN_NAME="rmse_v2_${task}_${demos}${SUF}"
  export TAGS="[${task},rmse,${demos},valtest,v2]"
  export OUTDIR="${OUT_ROOT}/${RUN_NAME}"
  jid=$(sbatch --parsable --job-name="$RUN_NAME" \
        --account="$acct" --partition="$part" --gres="$gres" \
        --cpus-per-task="$cpus" --mem=96G --time="${WALL:-3-00:00:00}" --requeue \
        --output="data/train/rmse_v2/slurm-%x-%j.out" \
        --error="data/train/rmse_v2/slurm-%x-%j.out" \
        --export=ALL "$SB")
  echo "submitted $RUN_NAME -> job $jid  [$acct/$part $gres cpus=$cpus demos=$demos val=${VAL[$task]##*/}]"
done
