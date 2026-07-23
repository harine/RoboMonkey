#!/bin/bash
# -----------------------------------------------------------------------------
# Launch the sub50 diffusion search-policy trainings WITH the frozen-DQC-verifier
# advantage term added to BC:  loss = loss_BC + PG_LAMBDA * loss_PG.
#
# Two loss forms (PG_MODE, both use the frozen DQC chunk critic as a stop-grad
# scalar; a* = expert, a_pi = policy sample):
#   awbc      : advantage-weighted imitation of the EXPERT,
#               grad = E[ grad_logpi(a*) * (Q(a*) - Q(a_pi)) ], weight clamped >=0.
#               (This is the loss the user wrote: ∇logπ · (Q(a_expert) - Q(a_pred)).)
#   reinforce : REINFORCE on the policy's own sample a_pi (the pre-existing
#               pg_lambda code), advantage Q(a_pi) - Q(a*).
#
# 8 runs total = both modes x {eggplant λ0.1, stack λ0.1, spoon λ0.1, spoon λ0.3},
# 100k gradient steps each, project dqc_search_policy, val action-MSE logged by the
# workspace (val_action_mse_error_min/avg). Routed to cse gpu-l40s + robotics
# gpu-a40 (non-preemptible; NOT weirdlab). training.resume=True -> a timed-out job
# resumes from latest.ckpt when resubmitted to the same OUTDIR.
#
# Smoke first:   SMOKE=1 bash scriptsv2/train_search/launch_pg_sub50.sh   (2 jobs, 5 steps)
# Full:                 bash scriptsv2/train_search/launch_pg_sub50.sh
# -----------------------------------------------------------------------------
set -euo pipefail
cd /mmfs1/home/harine/RoboMonkey
SB=scriptsv2/train_search/train_search_pg.sbatch
SMOKE="${SMOKE:-0}"

# ---- common (frozen DQC chunk critic, same as the BC sub50 search runs) ----
export CONFIG=robomonkey_eggplant_search_dqc
export CRITIC_DIR=/gscratch/robotics/harine/dqc_checkpoint/sd000s_137913.0.20260605_163017
export RESTORE_EPOCH=100000
export MAX_ACTIONS=16
export CONDA_ENV=dqc_jax
export NUM_WORKERS=2

if [ "$SMOKE" = "1" ]; then
  export MAX_GRADIENT_STEPS="${SMOKE_STEPS:-5}"; export NUM_EPOCHS=1
  LOG_PROJECT_BASE=dqc_search_policy_smoke
  OUT_ROOT_BASE=/gscratch/robotics/harine/RoboMonkey_data/train/pg_sub50_smoke
  SUF=_smoke
else
  export MAX_GRADIENT_STEPS=100000; export NUM_EPOCHS=60000
  LOG_PROJECT_BASE=dqc_search_policy
  OUT_ROOT_BASE=/gscratch/robotics/harine/RoboMonkey_data/train/pg_sub50
  SUF=
fi
export LOG_PROJECT="$LOG_PROJECT_BASE"
export OUT_ROOT="$OUT_ROOT_BASE"

# ---- per-task dataset dir / verifier text-emb / instruction ----
declare -A DS EMB INSTR
DS[eggplant]=/gscratch/robotics/harine/data/strict_distance_success_sub50_diverse_start
DS[stack]=/gscratch/robotics/harine/data/stack_cube_strict_sub50_diverse_start
DS[spoon]=/gscratch/robotics/harine/data/spoon_on_towel_strict_sub50_diverse_start
EMB[eggplant]=/mmfs1/home/harine/RoboMonkey/scriptsv2/q_fn_eval/task_emb_use3.npy
EMB[stack]=/mmfs1/home/harine/RoboMonkey/scriptsv2/q_fn_eval/task_emb_stack_use3.npy
EMB[spoon]=/mmfs1/home/harine/RoboMonkey/scriptsv2/q_fn_eval/task_emb_spoon_use3.npy
INSTR[eggplant]="put the eggplant in the basket"
INSTR[stack]="stack the green block on the yellow block"
INSTR[spoon]="put the spoon on the towel"

# ---- run table: mode task lambda acct part gres cpus ----
# Routing (per user): AWBC on weirdlab gpu-l40 (Ada L40 -> faster training, 4 GPUs),
# REINFORCE on robotics gpu-a40 (all robotics nodes). Both partitions have NO per-job
# wall limit and no assoc GPU cap; cse partitions cap wall at 1 day (unusable here).
# --time below is sized to fit before the Jul 14 09:00 maintenance reservation so jobs
# START NOW; --requeue + training.resume=True continue from latest.ckpt afterward.
if [ "$SMOKE" = "1" ]; then
  RUNS=(
    "awbc eggplant 0.1 weirdlab gpu-l40 gpu:l40:1 12"
    "reinforce eggplant 0.1 robotics gpu-a40 gpu:a40:1 6"
  )
else
  RUNS=(
    "awbc eggplant 0.1 weirdlab gpu-l40 gpu:l40:1 12"
    "awbc stack 0.1 weirdlab gpu-l40 gpu:l40:1 12"
    "awbc spoon 0.1 weirdlab gpu-l40 gpu:l40:1 12"
    "awbc spoon 0.3 weirdlab gpu-l40 gpu:l40:1 12"
    "reinforce eggplant 0.1 robotics gpu-a40 gpu:a40:1 6"
    "reinforce stack 0.1 robotics gpu-a40 gpu:a40:1 6"
    "reinforce spoon 0.1 robotics gpu-a40 gpu:a40:1 6"
    "reinforce spoon 0.3 robotics gpu-a40 gpu:a40:1 6"
  )
fi

mkdir -p data/train/pg_sub50
for r in "${RUNS[@]}"; do
  read -r mode task lam acct part gres cpus <<< "$r"
  short=reinf; [ "$mode" = awbc ] && short=awbc
  export PG_MODE="$mode"; export PG_LAMBDA="$lam"
  export DATASET_DIR="${DS[$task]}"
  export TEXT_EMB="${EMB[$task]}"
  export INSTRUCTION="${INSTR[$task]}"
  export RUN_NAME="pg_${short}_l${lam}_${task}_sub50${SUF}"
  export TAGS="[${task},pg_loss,${short},l${lam}]"
  export OUTDIR="${OUT_ROOT}/${RUN_NAME}"
  jid=$(sbatch --parsable --job-name="$RUN_NAME" \
        --account="$acct" --partition="$part" --gres="$gres" \
        --cpus-per-task="$cpus" --mem=96G --time="${WALL:-3-00:00:00}" --requeue \
        --output="data/train/pg_sub50/slurm-%x-%j.out" \
        --error="data/train/pg_sub50/slurm-%x-%j.out" \
        --export=ALL "$SB")
  echo "submitted $RUN_NAME -> job $jid  [$acct/$part $gres cpus=$cpus pg_lambda=$lam mode=$mode]"
done
