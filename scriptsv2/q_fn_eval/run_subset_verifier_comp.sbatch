#!/bin/bash
#SBATCH --job-name=subset_vcomp
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/mmfs1/home/harine/RoboMonkey/data/eval/subset_verifier_comp/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# Zero-shot: how does the FROZEN eggplant-trained DQC chunk-8 verifier separate
# strict-filtered (sub100/50/25) successes from messy successes and failures on
# UNSEEN tasks (stack, spoon)? Mirrors the eggplant filtered-verifier-comp, but
# DQC-only (no LLaVA) since the question is about the base DQC chunk-8 verifier.
#
# Per task, all in the dqc_jax env (single GPU):
#   1) subset_sample_pairs.py  -> pairs.npz   (3-group tagged, nested sub_level)
#   2) filtered_score_dqc.py    -> dqc_scores_filtered.csv  (chunk-8/chunk-1/action)
#   3) subset_plot_comp.py      -> verifier_comp.{png,md,csv} + auc_vs_subsize.{png,csv}
#
#   sbatch scriptsv2/q_fn_eval/run_subset_verifier_comp.sbatch
# Knobs (env): TASKS ("stack spoon"), MAX_EPS_OTHER_FAIL (120), STRIDE (4), SEED (42).
# -----------------------------------------------------------------------------
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
REPO="$PWD"
DATA_ROOT="/gscratch/robotics/harine/data"
QFN="$REPO/scriptsv2/q_fn_eval"

TASKS="${TASKS:-stack spoon}"
MAX_EPS_OTHER_FAIL="${MAX_EPS_OTHER_FAIL:-120}"
STRIDE="${STRIDE:-4}"
SEED="${SEED:-42}"
DQC_CKPT="${DQC_CKPT:-/gscratch/robotics/harine/dqc_checkpoint/sd000s_137913.0.20260605_163017}"
mkdir -p "$REPO/data/eval/subset_verifier_comp"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate dqc_jax
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/dqc_jax/lib:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.6}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"

run_task () {
  local task="$1" collect="$2" strict="$3" instr="$4" emb="$5" label="$6"
  local out="$REPO/data/eval/subset_verifier_comp/$task"
  mkdir -p "$out"
  echo "============================================================"
  echo " TASK $task | out=$out"
  echo "   collect=$collect  instr='$instr'  emb=$emb"
  echo "============================================================"

  echo "#### [$task] STAGE 1: sample 3-group pairs ####"
  python "$QFN/subset_sample_pairs.py" \
    --data-dir "$DATA_ROOT/$collect" \
    --shards state0.zarr state1.zarr \
    --sub-manifests \
      100:"$DATA_ROOT/${strict}_sub100_diverse_start/filter_manifest.json" \
      50:"$DATA_ROOT/${strict}_sub50_diverse_start/filter_manifest.json" \
      25:"$DATA_ROOT/${strict}_sub25_diverse_start/filter_manifest.json" \
    --strict-manifest "$DATA_ROOT/${strict}/filter_manifest.json" \
    --out-dir "$out" --instruction "$instr" \
    --max-eps-other-fail "$MAX_EPS_OTHER_FAIL" --stride "$STRIDE" --seed "$SEED" \
    || { echo "[$task] STAGE 1 FAILED"; return 1; }

  echo "#### [$task] STAGE 2: DQC critics (chunk-8 / chunk-1 / action) ####"
  python "$QFN/filtered_score_dqc.py" \
    --out-dir "$out" --ckpt-dir "$DQC_CKPT" \
    --instruction "$instr" --text-emb "$QFN/$emb" \
    || { echo "[$task] STAGE 2 FAILED"; return 1; }

  echo "#### [$task] STAGE 3: 3-group plot + AUC-vs-subsize ####"
  python "$QFN/subset_plot_comp.py" --out-dir "$out" --task-label "$label" \
    || { echo "[$task] STAGE 3 FAILED"; return 1; }

  echo "[$task] done -> $out"; ls -la "$out"
}

# task | complete-collection dir | strict dir stem | instruction | task_emb file | plot label
declare -A COLLECT=( [stack]="stack_cube"      [spoon]="spoon_on_towel" )
declare -A STRICT=(  [stack]="stack_cube_strict" [spoon]="spoon_on_towel_strict" )
declare -A INSTR=(   [stack]="stack the green block on the yellow block" [spoon]="put the spoon on the towel" )
declare -A EMB=(     [stack]="task_emb_stack_use3.npy" [spoon]="task_emb_spoon_use3.npy" )
declare -A LABEL=(   [stack]="stack (green block on yellow block)" [spoon]="spoon (spoon on towel)" )

rc=0
for t in $TASKS; do
  run_task "$t" "${COLLECT[$t]}" "${STRICT[$t]}" "${INSTR[$t]}" "${EMB[$t]}" "${LABEL[$t]}" || rc=1
done
echo "[subset_vcomp] all done (rc=$rc)"
exit $rc
