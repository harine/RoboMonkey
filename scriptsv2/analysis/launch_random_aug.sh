#!/bin/bash
# -----------------------------------------------------------------------------
# Random-action-diversity augmentation campaign (spoon_on_towel).
#
# The search transformer's N candidates are near-collapsed, so the DQC verifier has
# nothing to rank. RAND_AUG_K>0 adds a stratified-random pool covering the whole
# translation-direction sphere (K sectors x gripper open/closed = 2K candidates),
# scored by the SAME verifier; selection is argmax over the union.
#   (scriptsv2/eval_search/random_action_aug.py)
#
# Three groups, all writing under data/eval/random_a_diverse/spoon_on_towel:
#   det      50 ep, seeds 1000-1049, argmax, 60 steps, tiled(8)  <- SAME protocol as the
#            existing rmse_det / pg_det cells, so the SR table is apples-to-apples.
#            ckpts 10k + 20k (the milestones where rmse / pg-awbc / reinforce all exist).
#   rollout  the 20 train_test_analysis DEMO seeds (10 train + 10 test) at the 90k ckpt,
#            VIZ_Q=1 non-tiled -> npz for the expert-overlay videos.
#   offline  demo states -> candidate pools + Q + a*, with the random pool flagged.
#
# Idempotent + cron-safe: a cell with a non-empty eval_log.json (or npz, offline) or a
# matching job in squeue is skipped. Routes round-robin over the partitions that had
# headroom on 2026-07-21 (robotics a100, weirdlab l40s, ckpt-all); re-check `hyakalloc`
# and repoint ROUTES if cells stall PD.
#
#   DRY=1 bash scriptsv2/analysis/launch_random_aug.sh          # preview everything
#   bash scriptsv2/analysis/launch_random_aug.sh                # submit gaps
#   CAMPAIGN=rollout bash scriptsv2/analysis/launch_random_aug.sh # one group
#   GRIPPER=copy OUT_ROOT=data/eval/random_a_with_grasp/spoon_on_towel bash ...  # keep the
#     search transformer's grasp; random candidates then vary only the translation direction
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
REPO="$PWD"

OUT_ROOT="${OUT_ROOT:-data/eval/random_a_diverse/spoon_on_towel}"
# both  = K sectors x gripper{open,closed} -> M=2K   (the gripper is part of the search)
# copy  = K sectors, gripper COPIED from the search transformer's argmax candidate -> M=K
GRIPPER="${GRIPPER:-both}"
JOB_TAG="${JOB_TAG:-}"          # suffix for slurm job names, so parallel campaigns are distinguishable
SB_EVAL=scriptsv2/eval_search/eval_search_dqc_chunk.sbatch
SB_ROLL=scriptsv2/analysis/rollout_demo_seeds.sbatch
SB_OFF=scriptsv2/analysis/offline_candidates.sbatch

CK=/gscratch/robotics/harine/RoboMonkey_data/train/rmse_multitask/diff_search_rmse_spoon_sub50_diverse/checkpoints
TASK=widowx_spoon_on_towel
INSTR="put the spoon on the towel"
TEMB="$REPO/scriptsv2/q_fn_eval/task_emb_spoon_use3.npy"
DATA=/gscratch/robotics/harine/data
TRAIN_DS=spoon_on_towel_strict_sub50_diverse_start
TEST_DS=spoon_on_towel_strict_testset_not_in_sub100
SEEDS_DIR="$REPO/scriptsv2/eval_search/demoseeds"

CAMPAIGN="${CAMPAIGN:-det rollout offline}"   # GROUPS is a bash builtin -- do not reuse that name
K_LIST="${K_LIST:-16 50}"
N_LIST="${N_LIST:-1 4 16 64}"
STEPS_K="${STEPS_K:-10 20}"
MAX_STEPS="${MAX_STEPS:-60}"
NUM_VIDEOS="${NUM_VIDEOS:-10}"
VIZ_CKPT_K="${VIZ_CKPT_K:-90}"          # rollout/offline use the analysis ckpt (90k)
DRY="${DRY:-0}"

# Override with ROUTES_STR="acct|part|gres acct|part|gres ..." when `hyakalloc` shows
# the defaults are full (cells otherwise sit PD on AssocGrpGRES).
ROUTES_STR="${ROUTES_STR:-robotics|gpu-a100|gpu:a100:1 weirdlab|gpu-l40s|gpu:l40s:1 robotics|ckpt-all|gpu:a40:1 cse|gpu-l40s|gpu:l40s:1 robotics|ckpt-all|gpu:a100:1}"
read -r -a ROUTES <<< "$ROUTES_STR"

mkdir -p "$OUT_ROOT/_logs"
QN="$(squeue -u "$USER" -h -o '%j')"
i=0; nsub=0; sk=0

ckpt_for () {  # $1 = milestone in k -> exact ckpt path (10k/20k are step_00X0001)
  case "$1" in
    10) echo "$CK/step_0010001.ckpt";;
    20) echo "$CK/step_0020001.ckpt";;
    90) echo "$CK/step_0090003.ckpt";;
    *)  ls "$CK"/step_0*"$1"000?.ckpt 2>/dev/null | head -1;;
  esac
}

route () { IFS='|' read -r ACCT PART GRES <<< "${ROUTES[$((i % ${#ROUTES[@]}))]}"; i=$((i+1)); }

submit () {  # $1=jobname $2=sbatch script  (env already exported)
  local jn="$1" sb="$2"
  grep -qxF "$jn" <<<"$QN" && { echo "skip(queued) $jn"; sk=$((sk+1)); return; }
  route
  if [ "$DRY" = 1 ]; then
    echo "WOULD $jn -> $ACCT/$PART ($GRES)  out=$OUT_DIR"; nsub=$((nsub+1)); return
  fi
  local jid
  jid=$(sbatch --parsable --job-name="$jn" --account="$ACCT" --partition="$PART" \
        --gres="$GRES" --export=ALL "$sb")
  echo "sub $jn -> $jid ($ACCT/$PART)"; nsub=$((nsub+1))
}

# ---------------------------------------------------------------- det (SR table)
if grep -qw det <<<"$CAMPAIGN"; then
  for k in $STEPS_K; do
    ck="$(ckpt_for "$k")"
    [ -f "$ck" ] || { echo "NO ckpt ${k}k -> $ck"; continue; }
    for K in $K_LIST; do
      for N in $N_LIST; do
        out="$OUT_ROOT/det/${k}k/sr_n${N}_k${K}"
        [ -s "$out/eval_log.json" ] && { sk=$((sk+1)); continue; }
        export CHECKPOINT="$ck" N_SAMPLES="$N" MODE=argmax SOFTMAX_TEMP=1.0 \
               NUM_EPISODES=50 START_SEED=1000 DQC_RESTORE_EPOCH=100000 \
               TASK="$TASK" MAX_STEPS="$MAX_STEPS" TILED_ENVS=8 TILED_SEED_PER_ENV=1 \
               VIZ_Q=0 RENDER_Q=0 SAVE_VIDEOS=0 OUT_DIR="$out" \
               RAND_AUG_K="$K" RAND_AUG_GRIPPER="$GRIPPER" RAND_AUG_ONLY=0
        submit "ra${JOB_TAG}_${k}k_n${N}_k${K}" "$SB_EVAL"
      done
    done
    # verifier-only control: drop the policy candidates entirely
    K=50; out="$OUT_ROOT/det/${k}k/randonly_k${K}"
    if [ ! -s "$out/eval_log.json" ]; then
      export CHECKPOINT="$ck" N_SAMPLES=16 MODE=argmax SOFTMAX_TEMP=1.0 \
             NUM_EPISODES=50 START_SEED=1000 DQC_RESTORE_EPOCH=100000 \
             TASK="$TASK" MAX_STEPS="$MAX_STEPS" TILED_ENVS=8 TILED_SEED_PER_ENV=1 \
             VIZ_Q=0 RENDER_Q=0 SAVE_VIDEOS=0 OUT_DIR="$out" \
             RAND_AUG_K="$K" RAND_AUG_GRIPPER="$GRIPPER" RAND_AUG_ONLY=1
      submit "ra${JOB_TAG}_${k}k_only_k${K}" "$SB_EVAL"
    else sk=$((sk+1)); fi
  done
fi

# ------------------------------------------------------------- rollout (videos)
if grep -qw rollout <<<"$CAMPAIGN"; then
  ck="$(ckpt_for "$VIZ_CKPT_K")"
  for split in train test; do
    sf="$SEEDS_DIR/spoon_${split}.txt"
    [ -f "$sf" ] || { echo "NO seeds file $sf"; continue; }
    for K in $K_LIST; do
      out="$OUT_ROOT/rollout/${split}_k${K}"
      ls "$out"/search_q/*.npz >/dev/null 2>&1 && { sk=$((sk+1)); continue; }
      export CKPT="$ck" TASK="$TASK" SEEDS_FILE="$sf" OUT_DIR="$out" \
             N_SAMPLES=16 MAX_STEPS="$MAX_STEPS" NUM_VIDEOS="$NUM_VIDEOS" \
             RAND_AUG_K="$K" RAND_AUG_GRIPPER="$GRIPPER" RAND_AUG_ONLY=0
      submit "raroll${JOB_TAG}_${split}_k${K}" "$SB_ROLL"
    done
  done
fi

# ------------------------------------------------------- offline candidate pools
if grep -qw offline <<<"$CAMPAIGN"; then
  ck="$(ckpt_for "$VIZ_CKPT_K")"
  K="${OFFLINE_K:-50}"
  for split in train test; do
    case "$split" in train) ds="$TRAIN_DS";; test) ds="$TEST_DS";; esac
    zarr="$DATA/$ds/$ds.zarr"
    [ -d "$zarr" ] || { echo "NO zarr $zarr"; continue; }
    out="$OUT_ROOT/offline/search_aug_${split}_N16"
    ls "$out"/search_q/*.npz >/dev/null 2>&1 && { sk=$((sk+1)); continue; }
    export CKPT="$ck" ZARR="$zarr" POLICY_KIND=search SPLIT="$split" \
           TASK="$TASK" INSTRUCTION="$INSTR" TEXT_EMB="$TEMB" \
           N_SAMPLES=16 OUT_DIR="$out" NUM_EPISODES=50 STRIDE=4 N_ACTION_STEPS=4 \
           RAND_AUG_K="$K" RAND_AUG_GRIPPER="$GRIPPER"
    submit "raoff${JOB_TAG}_${split}_k${K}" "$SB_OFF"
  done
fi

echo "=== random-aug campaign: submitted=$nsub skipped=$sk  (out=$OUT_ROOT) ==="
