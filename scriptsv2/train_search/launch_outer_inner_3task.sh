#!/bin/bash
# -----------------------------------------------------------------------------
# Launch the 3 OUTER/INNER search-policy trainings: eggplant, stack, spoon.
#
# Same policy / data / loss as the offline arms (SearchPolicyRoboMonkeyDiffusion,
# BC on the expert action, base DQC critic baked as the search verifier) -- only
# the training LOOP differs: the search context is generated once per outer step
# for OUTER_BATCH_SIZE windows and reused for INNER_EPOCHS passes, so each update
# costs INNER_EPOCHS-times fewer verifier calls. See train_search_outer_inner.sbatch.
#
# Per-task text embedding is REQUIRED, not cosmetic: generate_search_context calls
# the language-conditioned DQC critic every outer step to build the context the
# policy conditions on, so the eggplant embedding on a stack run poisons the context.
#
# CAVEAT, carried over from launch_multitask_search_trainings.sh and worth restating:
# all three arms bake the SAME base eggplant DQC critic. That critic does not
# transfer zero-shot to stack/spoon -- so on those two tasks the search context is
# built from a verifier whose ranking is not trustworthy on that task. The BC loss
# target is still the dataset expert action, so training is well-posed either way;
# what is compromised is how informative the context is. Do not read the stack/spoon
# arms as "search works on stack/spoon" without a task-appropriate critic.
#
# Routed to the `robotics` account, which currently has the free GPUs (2x a100,
# 1x a40) -- no MaxWall there, so the 7-day request is accepted.
# Idempotent: a run already in squeue (by job-name) is skipped.
#   bash scriptsv2/train_search/launch_outer_inner_3task.sh          # submit
#   DRY=1 bash scriptsv2/train_search/launch_outer_inner_3task.sh    # preview
#   DEBUG=True TASKS=eggplant bash ...                               # smoke one task
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
SB=scriptsv2/train_search/train_search_outer_inner.sbatch
OUT_ROOT="${OUT_ROOT:-/gscratch/robotics/harine/RoboMonkey_data/train/outer_inner}"
CRITIC="${CRITIC:-/gscratch/robotics/harine/dqc_checkpoint/sd000s_137913.0.20260605_163017}"
EMBROOT=/mmfs1/home/harine/RoboMonkey/scriptsv2/q_fn_eval
DRY="${DRY:-0}"
DEBUG="${DEBUG:-False}"
TASKS="${TASKS:-eggplant stack spoon}"
INNER_EPOCHS="${INNER_EPOCHS:-4}"
MAX_GRADIENT_STEPS="${MAX_GRADIENT_STEPS:-20000}"
# Walltime must FIT BEFORE the next cluster maintenance reservation, or Slurm refuses to
# schedule at all -- the job sits PD on "(ReqNodeNotAvail, Reserved for maintenance)"
# forever rather than starting and being requeued. Check with `scontrol show reservation`;
# as of 2026-08-05 the next window is August11_Maintenance, Aug 11 09:00 -> Aug 12 09:00,
# covering essentially every node.
#
# A short walltime is cheap here: checkpoint_every=1000 steps and `resume: True` against a
# FIXED hydra.run.dir, so a run that hits the wall is continued by simply re-running this
# launcher -- it picks up from latest.ckpt. Prefer fitting the window over asking for 7d.
WALLTIME="${WALLTIME:-5-00:00:00}"
# Memory per job. MEASURED, not guessed: peak RSS is ~35G on eggplant (the largest
# dataset) and ~13G on the sub100 stack/spoon sets. The old 96G request was ~3x that and
# it does not merely waste quota -- `robotics` has a 188G cap on gpu-a100, so two 96G jobs
# do not fit and the second sits PD forever on (AssocGrpMemLimit). 64G keeps ~1.8x headroom
# over the observed peak and lets two arms share the account.
MEM="${MEM:-64G}"

declare -A DATASET=(
  [eggplant]=/gscratch/robotics/harine/data/eggplant_in_basket_complete
  [stack]=/gscratch/robotics/harine/data/stack_cube_strict_sub100_diverse_start
  [spoon]=/gscratch/robotics/harine/data/spoon_on_towel_strict_sub100_diverse_start
)
declare -A TEXTEMB=(
  [eggplant]="$EMBROOT/task_emb_use3.npy"
  [stack]="$EMBROOT/task_emb_stack_use3.npy"
  [spoon]="$EMBROOT/task_emb_spoon_use3.npy"
)
declare -A INSTR=(
  [eggplant]="put the eggplant in the basket"
  [stack]="stack the green block on the yellow block"
  [spoon]="put the spoon on the towel"
)
# acct|partition|gres -- one per task, routed at what is actually FREE.
# robotics/gpu-a100 has 2 GPUs + 13 CPUs free, which fits exactly two 6-CPU jobs.
# robotics/gpu-a40 has a free GPU but only 2 free CPUs, and six cand64_* jobs are
# already PD against that account's CPU quota -- a third job there would queue behind
# them rather than start -- so spoon goes to weirdlab/gpu-a40 (6 GPUs, 40 CPUs free,
# no MaxWall). Re-check `hyakalloc` and repoint this array if the arms sit PD.
declare -A ROUTE=(
  [eggplant]="robotics|gpu-a100|gpu:a100:1"
  [stack]="robotics|gpu-a100|gpu:a100:1"
  [spoon]="weirdlab|gpu-a40|gpu:a40:1"
)

QNAMES="$(squeue -u "$USER" -h -o '%j')"
sub=0; sk=0
for fam in $TASKS; do
  ds="${DATASET[$fam]}"; emb="${TEXTEMB[$fam]}"; instr="${INSTR[$fam]}"
  run="oi_${fam}_ie${INNER_EPOCHS}"
  [ "$DEBUG" = "True" ] && run="smoke_${run}"
  IFS='|' read -r acct part gres <<< "${ROUTE[$fam]}"

  if [ ! -d "$ds" ];  then echo "MISSING dataset:  $ds -- skip $run";  continue; fi
  if [ ! -f "$emb" ]; then echo "MISSING task emb: $emb -- skip $run"; continue; fi
  if grep -qxF "$run" <<< "$QNAMES"; then echo "skip (already queued): $run"; sk=$((sk+1)); continue; fi

  # A debug run must not land in (and resume from) the real run's output dir.
  outdir="${OUT_ROOT}/${run}"
  exp="ALL,CONFIG=robomonkey_eggplant_search_dqc_outer_inner,CRITIC_DIR=$CRITIC,RESTORE_EPOCH=100000"
  exp="$exp,DATASET_DIR=$ds,TEXT_EMB=$emb,INSTRUCTION=$instr"
  exp="$exp,INNER_EPOCHS=$INNER_EPOCHS,MAX_GRADIENT_STEPS=$MAX_GRADIENT_STEPS,DEBUG=$DEBUG"
  exp="$exp,LOG_PROJECT=dqc_search_outer_inner,RUN_NAME=$run,OUTDIR=$outdir"
  [ "$DEBUG" = "True" ] && exp="$exp,NUM_WORKERS=0"

  if [ "$DRY" = "1" ]; then
    echo "WOULD submit $run  [$acct/$part]  <- $(basename "$ds")  emb=$(basename "$emb")"
    sub=$((sub+1)); continue
  fi
  wall="$WALLTIME"
  [ "$DEBUG" = "True" ] && wall=00:30:00
  jid=$(sbatch --parsable --job-name="$run" \
        --account="$acct" --partition="$part" --gres="$gres" \
        --cpus-per-task=6 --mem="$MEM" --time="$wall" \
        --export="$exp" "$SB")
  echo "submitted $run -> $jid  [$acct/$part]  <- $(basename "$ds")  emb=$(basename "$emb")"
  sub=$((sub+1))
done
echo "=== outer/inner 3-task: submitted=$sub skipped=$sk  (debug=$DEBUG) -> $OUT_ROOT ==="
