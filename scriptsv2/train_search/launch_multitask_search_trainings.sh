#!/bin/bash
# -----------------------------------------------------------------------------
# Launch the 6 NEW-TASK diffusion SEARCH-policy trainings (stack_cube +
# spoon_on_towel, sub100/50/25 diverse-start), mirroring the eggplant sub-runs
# EXACTLY: SearchPolicyRoboMonkeyDiffusion, action-rmse / BC-only (PG_LAMBDA=0),
# base DQC critic baked, 50k grad-step cap, polynomial LR (warmup 500, decay
# 10000 -> 1e-5), checkpoints every 1k. Verified recipe = eggplant sub100's
# .hydra/overrides.yaml. num_epochs=30000: the 25-ep subsets yield only ~2
# steps/epoch, so num_epochs=20000 (first attempt) capped them at ~40k steps
# BEFORE the 50k step cap (spoon_sub25 finished at 39k). 30000 x 2 = 60k >= 50k,
# so MAX_GRADIENT_STEPS=50000 binds for every dataset size (step-based LR =>
# identical behaviour; extra epochs harmless, the step cap stops it first).
#
# NOTE: PG_LAMBDA=0 means the baked DQC verifier is NOT in the training loss (pure
# BC on the dataset), so the eggplant task_emb placeholder is harmless for TRAINING.
# EVAL-time BoN on these tasks will need a task-correct text embedding (+ ideally a
# task-appropriate critic) -- a separate step, not required to train.
#
# Routed to weirdlab gpu-l40s (7-day wall OK; cse l40s rejects 7-day via QOS).
# 1 GPU, 6 CPU (<= l40s 16 CPU/GPU ratio), 96G each. resume=True -> wall/preempt safe.
# Idempotent: a run already in squeue (by job-name) is skipped.
#   bash scriptsv2/train_search/launch_multitask_search_trainings.sh      # submit
#   DRY=1 bash .../launch_multitask_search_trainings.sh                   # preview
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
SB=scriptsv2/train_search/train_search_pg.sbatch
OUT_ROOT=/gscratch/robotics/harine/RoboMonkey_data/train/rmse_multitask
CRITIC=/gscratch/robotics/harine/dqc_checkpoint/sd000s_137913.0.20260605_163017
DRY="${DRY:-0}"
QNAMES="$(squeue -u "$USER" -h -o '%j')"

# task_family : dataset_prefix
declare -A DPREFIX=(
  [stack]=/gscratch/robotics/harine/data/stack_cube_strict
  [spoon]=/gscratch/robotics/harine/data/spoon_on_towel_strict
)
# Per-task MUSE (USE-3) instruction embedding for the language-conditioned DQC
# verifier. REQUIRED for training (not just eval): get_value runs every step in
# _run_search to build the Q-value search context the policy conditions on, so a
# wrong embedding poisons training. Embeddings from scriptsv2/q_fn_eval/make_task_embs.py.
EMBROOT=/mmfs1/home/harine/RoboMonkey/scriptsv2/q_fn_eval
declare -A TEXTEMB=(
  [stack]="$EMBROOT/task_emb_stack_use3.npy"
  [spoon]="$EMBROOT/task_emb_spoon_use3.npy"
)
declare -A INSTR=(
  [stack]="stack the green block on the yellow block"
  [spoon]="put the spoon on the towel"
)
sub=0; sk=0
for fam in stack spoon; do
  for s in sub100 sub50 sub25; do
    ds="${DPREFIX[$fam]}_${s}_diverse_start"
    run="diff_search_rmse_${fam}_${s}_diverse"
    emb="${TEXTEMB[$fam]}"; instr="${INSTR[$fam]}"
    if [ ! -d "$ds" ]; then echo "MISSING dataset: $ds -- skip $run"; continue; fi
    if [ ! -f "$emb" ]; then echo "MISSING task emb: $emb -- run make_task_embs.py first; skip $run"; continue; fi
    if grep -qxF "$run" <<< "$QNAMES"; then echo "skip (already queued): $run"; sk=$((sk+1)); continue; fi
    if [ "$DRY" = "1" ]; then echo "WOULD submit $run  <- $ds  emb=$(basename "$emb")"; sub=$((sub+1)); continue; fi
    jid=$(sbatch --parsable --job-name="$run" \
          --account=weirdlab --partition=gpu-l40s --gres=gpu:l40s:1 \
          --cpus-per-task=6 --mem=96G \
          --export=ALL,CONFIG=robomonkey_eggplant_search_dqc,PG_LAMBDA=0,CRITIC_DIR="$CRITIC",RESTORE_EPOCH=100000,DATASET_DIR="$ds",TEXT_EMB="$emb",INSTRUCTION="$instr",MAX_GRADIENT_STEPS=100000,NUM_EPOCHS=60000,NUM_WORKERS=2,LOG_PROJECT=dqc_search_policy,RUN_NAME="$run",OUT_ROOT="$OUT_ROOT" \
          "$SB")
    echo "submitted $run -> $jid  (<- $ds, emb=$(basename "$emb"))"; sub=$((sub+1))
  done
done
echo "=== multitask search trainings: submitted=$sub skipped=$sk ==="
