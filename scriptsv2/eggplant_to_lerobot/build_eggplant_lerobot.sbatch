#!/bin/bash
#SBATCH --job-name=egg2lerobot
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/mmfs1/home/harine/RoboMonkey/data/viz_q/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# Build the 3 eggplant LeRobot datasets for DQC finetuning (CPU only; video encode).
# Submit on a CPU/scavenger partition, e.g.:
#   sbatch --account=weirdlab-ckpt --partition=ckpt --qos=ckpt \
#          scriptsv2/eggplant_to_lerobot/build_eggplant_lerobot.sbatch
# v1 success-ramp + v2 all + v3 success-distance, run in parallel.
# -----------------------------------------------------------------------------
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
mkdir -p data/viz_q

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-dqc_jax}"

S0=/gscratch/robotics/harine/data/eggplant_in_basket/state0.zarr
S1=/gscratch/robotics/harine/data/eggplant_in_basket/state1.zarr
DATAROOT=/gscratch/robotics/harine/data
TASK="put the eggplant in the basket"
CONV=scriptsv2/eggplant_to_lerobot/zarr_to_lerobot.py

echo "[build] v1 success-ramp, v2 all, v3 success-distance (parallel)"

python "$CONV" --zarr "$S0" --also "$S1" --filter success --reward label \
  --task "$TASK" --out "$DATAROOT/eggplant_success_ramp_lerobot" \
  > data/viz_q/build_v1.log 2>&1 &
P1=$!
python "$CONV" --zarr "$S0" --also "$S1" --filter all --reward label \
  --task "$TASK" --out "$DATAROOT/eggplant_all_lerobot" \
  > data/viz_q/build_v2.log 2>&1 &
P2=$!
python "$CONV" --zarr "$S0" --also "$S1" --filter success --reward distance \
  --task "$TASK" --out "$DATAROOT/eggplant_success_dist_lerobot" \
  > data/viz_q/build_v3.log 2>&1 &
P3=$!

rc=0
wait $P1 || { echo "[build] v1 FAILED"; rc=1; }
wait $P2 || { echo "[build] v2 FAILED"; rc=1; }
wait $P3 || { echo "[build] v3 FAILED"; rc=1; }
echo "==== tails ===="
for v in v1 v2 v3; do echo "## $v"; tail -3 "data/viz_q/build_${v}.log"; done
echo "[build] done (rc=$rc)"
exit $rc
