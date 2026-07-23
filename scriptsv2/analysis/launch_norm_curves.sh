#!/bin/bash
# -----------------------------------------------------------------------------
# Normalized (and raw) best-of-N / Q-pick action-RMSE vs training-step curves for
# ALL sub50 search policies: the 3 rmse-loss (non-PG) runs + the 8 PG runs.
# ckpt_rmse_curve.py now emits best_of_N_norm / qpick_N_norm (normalize candidates
# + a* by each ckpt's [-1,1] limits normalizer). N=16 (= the training val best-of-16),
# 3 shards/split, routed to ckpt-preemptible. Out: data/eval/norm_curves/_curves/.
#   DRY=1 bash scriptsv2/analysis/launch_norm_curves.sh     # preview
#   bash scriptsv2/analysis/launch_norm_curves.sh
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
SB=scriptsv2/analysis/ckpt_curve.sbatch
DAT=/gscratch/robotics/harine/data
RF=/gscratch/robotics/harine/RoboMonkey_data/train/rmse_filtered
MT=/gscratch/robotics/harine/RoboMonkey_data/train/rmse_multitask
PG=/gscratch/robotics/harine/RoboMonkey_data/train/pg_sub50
OUT=data/eval/norm_curves/_curves
N_SAMPLES="${N_SAMPLES:-16}"; NSHARDS="${NSHARDS:-3}"; DRY="${DRY:-0}"

# task -> env | instruction | train_zarr | test_zarr
declare -A ENV INSTR ZTR ZTE
ENV[eggplant]=widowx_put_eggplant_in_basket;  INSTR[eggplant]="put the eggplant in the basket"
ENV[spoon]=widowx_spoon_on_towel;             INSTR[spoon]="put the spoon on the towel"
ENV[stack]=widowx_stack_cube;                 INSTR[stack]="stack the green block on the yellow block"
ZTR[eggplant]=$DAT/strict_distance_success_sub50_diverse_start/strict_distance_success_sub50_diverse_start.zarr
ZTE[eggplant]=$DAT/strict_distance_success_testset_not_in_sub100/strict_distance_success_testset_not_in_sub100.zarr
ZTR[spoon]=$DAT/spoon_on_towel_strict_sub50_diverse_start/spoon_on_towel_strict_sub50_diverse_start.zarr
ZTE[spoon]=$DAT/spoon_on_towel_strict_testset_not_in_sub100/spoon_on_towel_strict_testset_not_in_sub100.zarr
ZTR[stack]=$DAT/stack_cube_strict_sub50_diverse_start/stack_cube_strict_sub50_diverse_start.zarr
ZTE[stack]=$DAT/stack_cube_strict_testset_not_in_sub100/stack_cube_strict_testset_not_in_sub100.zarr

# policy | task | ckpt_dir | min_step | ckpt_stride
POLS=(
  "rmse_eggplant|eggplant|$RF/diff_search_rmse_sub50_diverse_data/checkpoints|5000|5"
  "rmse_spoon|spoon|$MT/diff_search_rmse_spoon_sub50_diverse/checkpoints|5000|5"
  "rmse_stack|stack|$MT/diff_search_rmse_stack_sub50_diverse/checkpoints|5000|5"
  "pg_awbc_l0.1_eggplant_sub50|eggplant|$PG/pg_awbc_l0.1_eggplant_sub50/checkpoints|1000|2"
  "pg_awbc_l0.1_stack_sub50|stack|$PG/pg_awbc_l0.1_stack_sub50/checkpoints|1000|2"
  "pg_awbc_l0.1_spoon_sub50|spoon|$PG/pg_awbc_l0.1_spoon_sub50/checkpoints|1000|2"
  "pg_awbc_l0.3_spoon_sub50|spoon|$PG/pg_awbc_l0.3_spoon_sub50/checkpoints|1000|2"
  "pg_reinf_l0.1_eggplant_sub50|eggplant|$PG/pg_reinf_l0.1_eggplant_sub50/checkpoints|1000|2"
  "pg_reinf_l0.1_stack_sub50|stack|$PG/pg_reinf_l0.1_stack_sub50/checkpoints|1000|2"
  "pg_reinf_l0.1_spoon_sub50|spoon|$PG/pg_reinf_l0.1_spoon_sub50/checkpoints|1000|2"
  "pg_reinf_l0.3_spoon_sub50|spoon|$PG/pg_reinf_l0.3_spoon_sub50/checkpoints|1000|2"
)
ROUTES=( "robotics|ckpt|gpu:a40:1" "robotics|ckpt|gpu:a100:1" )
mkdir -p "$OUT" data/eval/train_test_analysis/_logs
QN="$(squeue -u "$USER" -h -o '%j')"; i=0; nsub=0; sk=0
for spec in "${POLS[@]}"; do
  IFS='|' read -r pol task cdir mstep cstr <<< "$spec"
  [ -d "$cdir" ] || { echo "NO CKPT_DIR $pol -> $cdir"; continue; }
  for split in train test; do
    zarr="${ZTR[$task]}"; [ "$split" = test ] && zarr="${ZTE[$task]}"
    for sh in $(seq 0 $((NSHARDS-1))); do
      out="$OUT/${pol}_${split}_s${sh}.csv"
      jn="nc_${pol#pg_}"; jn="${jn//_sub50/}_${split:0:2}s${sh}"
      [ -s "$out" ] && { sk=$((sk+1)); continue; }
      grep -qxF "$jn" <<<"$QN" && { sk=$((sk+1)); continue; }
      IFS='|' read -r a p g <<< "${ROUTES[$((i%${#ROUTES[@]}))]}"; i=$((i+1))
      exp="CKPT_DIR=$cdir,ZARR=$zarr,TASK=${ENV[$task]},INSTRUCTION=${INSTR[$task]},SPLIT=$split,OUT=$out,SHARD=$sh,NSHARDS=$NSHARDS,N_SAMPLES=$N_SAMPLES,NUM_EPISODES=50,MIN_STEP=$mstep,CKPT_STRIDE=$cstr,CONDA_ENV=dqc_jax"
      [ "$DRY" = 1 ] && { echo "WOULD $jn ($pol $split shard $sh/$NSHARDS min=$mstep str=$cstr -> $a/$p)"; nsub=$((nsub+1)); continue; }
      jid=$(sbatch --parsable --job-name="$jn" --account="$a" --partition="$p" --gres="$g" \
            --cpus-per-task=6 --mem=56G --time=10:00:00 --requeue --export=ALL,"$exp" "$SB")
      echo "sub $jn -> $jid ($a/$p)"; nsub=$((nsub+1))
    done
  done
done
echo "=== norm curves [N=$N_SAMPLES nshards=$NSHARDS]: submitted=$nsub skipped=$sk ==="
