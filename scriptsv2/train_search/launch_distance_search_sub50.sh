#!/bin/bash
# -----------------------------------------------------------------------------
# Launch the distance-verifier search-policy training on the 50-demo eggplant set.
#
# Calibrates the verifier against the dataset FIRST, then submits training with the
# measured constants. Calibration is the point of this wrapper: action_pos_scale and
# value_mean/value_std are dataset properties, and a wrong action_pos_scale does not
# crash — it makes every candidate's predicted displacement collapse toward zero, so
# all 16 candidates score alike, the search context carries no ranking signal, and the
# run silently degenerates into plain BC that merely LOOKS like a search arm.
#
# Comparison arm: oi_eggplant_ie4 (same policy, same loop, same geometry, DQC critic).
# The eggplant DQC critic is exploitable off-manifold — it prefers off-manifold actions
# to the expert, which drove n>1 success to 0% (see the dqc-q-exploitable-offmanifold
# note). A closed-form geometric verifier has no parameters to exploit, so this run
# tests whether that failure was the CRITIC or the search itself.
#
#   bash scriptsv2/train_search/launch_distance_search_sub50.sh          # calibrate + submit
#   DRY=1 bash scriptsv2/train_search/launch_distance_search_sub50.sh    # calibrate + preview
#   CALIBRATE_ONLY=1 bash ...                                           # just the numbers
#   DEBUG=True bash ...                                                 # 6-step smoke
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
SB=scriptsv2/train_search/train_search_distance.sbatch

DATASET_DIR="${DATASET_DIR:-/gscratch/robotics/harine/data/strict_distance_success_sub50_diverse_start}"
HORIZON="${HORIZON:-8}"
INNER_EPOCHS="${INNER_EPOCHS:-4}"
MAX_GRADIENT_STEPS="${MAX_GRADIENT_STEPS:-20000}"
MAX_ACTIONS="${MAX_ACTIONS:-16}"
GRASP_KEY="${GRASP_KEY:-last_gripper_action}"
OUT_ROOT="${OUT_ROOT:-/gscratch/robotics/harine/RoboMonkey_data/train/distance_search}"
DRY="${DRY:-0}"
DEBUG="${DEBUG:-False}"
CALIBRATE_ONLY="${CALIBRATE_ONLY:-0}"
# Walltime must fit before the next maintenance reservation or slurm never schedules
# the job at all (it sits PD on ReqNodeNotAvail forever). Check `scontrol show
# reservation`. 1 day is ample: the DQC arm did 20k in ~8h and this verifier is cheaper.
WALLTIME="${WALLTIME:-1-00:00:00}"
MEM="${MEM:-32G}"
ACCT="${ACCT:-robotics}"
PART="${PART:-gpu-a100}"
GRES="${GRES:-gpu:a100:1}"

RUN_NAME="${RUN_NAME:-dist_eggplant_sub50_ie${INNER_EPOCHS}}"
[ "$DEBUG" = "True" ] && RUN_NAME="smoke_${RUN_NAME}"
OUTDIR="${OUTDIR:-${OUT_ROOT}/${RUN_NAME}}"

if [ ! -d "$DATASET_DIR" ]; then echo "MISSING dataset: $DATASET_DIR"; exit 1; fi

# ---- calibrate --------------------------------------------------------------
mkdir -p data/train/distance_search
CAL_JSON="data/train/distance_search/${RUN_NAME}_calibration.json"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "${CONDA_ENV:-dqc_jax}" 2>/dev/null || true

echo "=== calibrating verifier against $(basename "$DATASET_DIR") ==="
python scriptsv2/train_search/calibrate_distance_verifier.py \
    --dataset-dir "$DATASET_DIR" --horizon "$HORIZON" \
    --grasp-key "$GRASP_KEY" --json-out "$CAL_JSON" || {
        echo "calibration FAILED — refusing to submit with uncalibrated constants"; exit 1; }

read -r ACTION_POS_SCALE VALUE_MEAN VALUE_STD R2 < <(python3 -c "
import json; d=json.load(open('$CAL_JSON'))
print(d['action_pos_scale'], d['value_mean'], d['value_std'], d['action_pos_scale_r2'])")

echo
echo "  action_pos_scale = $ACTION_POS_SCALE   (linear-model R^2 = $R2)"
echo "  value_mean       = $VALUE_MEAN"
echo "  value_std        = $VALUE_STD"
[ "$CALIBRATE_ONLY" = "1" ] && { echo "(CALIBRATE_ONLY=1 -> stopping)"; exit 0; }

# ---- submit -----------------------------------------------------------------
if grep -qxF "$RUN_NAME" <<< "$(squeue -u "$USER" -h -o '%j' 2>/dev/null)"; then
  echo "skip (already queued): $RUN_NAME"; exit 0
fi

exp="ALL,CONFIG=robomonkey_eggplant_search_distance_outer_inner,DATASET_DIR=$DATASET_DIR"
exp="$exp,ACTION_POS_SCALE=$ACTION_POS_SCALE,VALUE_MEAN=$VALUE_MEAN,VALUE_STD=$VALUE_STD"
exp="$exp,GRASP_KEY=$GRASP_KEY,MAX_ACTIONS=$MAX_ACTIONS"
exp="$exp,INNER_EPOCHS=$INNER_EPOCHS,MAX_GRADIENT_STEPS=$MAX_GRADIENT_STEPS,DEBUG=$DEBUG"
exp="$exp,RUN_NAME=$RUN_NAME,OUTDIR=$OUTDIR"
[ "$DEBUG" = "True" ] && exp="$exp,NUM_WORKERS=0"

wall="$WALLTIME"; [ "$DEBUG" = "True" ] && wall=00:30:00

if [ "$DRY" = "1" ]; then
  echo "WOULD submit $RUN_NAME [$ACCT/$PART] <- $(basename "$DATASET_DIR") -> $OUTDIR"; exit 0
fi

jid=$(sbatch --parsable --job-name="$RUN_NAME" \
      --account="$ACCT" --partition="$PART" --gres="$GRES" \
      --cpus-per-task=6 --mem="$MEM" --time="$wall" \
      --export="$exp" "$SB")
echo "submitted $RUN_NAME -> $jid  [$ACCT/$PART]  -> $OUTDIR"
echo
echo "next: once step_0005000.ckpt lands, start the eval ticker:"
echo "  bash scriptsv2/eval_search/tick_distance_sub50_evals.sh"
