#!/bin/bash
# -----------------------------------------------------------------------------
# Watch the stack_cube / spoon_on_towel strict sub<N>_diverse_start datasets as
# they finish generating; for each, VERIFY #trajectories == sub<N>, then kick off
# an H8 / n_obs=1 / 50k BC training on a free GPU (weirdlab/gpu-l40, lean 4 CPU).
# Same BC config as eggplant (robomonkey_eggplant_bc — generic 40-dim widowx
# state; stack/spoon share the schema), just DATASET_DIR swapped.
# Idempotent: skips a dataset whose gen job is still running, whose #traj != N
# (logs a WARN, does not train), or whose training is already launched.
# Self-removes its crontab line once all 6 trainings are launched.
#
# Install (every 30 min), preserving existing entries:
#   ( crontab -l 2>/dev/null; echo "*/30 * * * * /bin/bash /mmfs1/home/harine/RoboMonkey/scriptsv2/train_search/watch_train_subsets.sh # WATCHSUBSETS" ) | crontab -
# Remove:  crontab -l | grep -v WATCHSUBSETS | crontab -
# Log: data/train/bc_denoiser/_logs/_watch_subsets.log
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
LOG=data/train/bc_denoiser/_logs/_watch_subsets.log
mkdir -p data/train/bc_denoiser/_logs
ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(ts)] === watch tick ===" >> "$LOG"

QNAMES="$(squeue -u "$USER" -h -o '%j' 2>/dev/null)"
DATA=/gscratch/robotics/harine/data
all_launched=1

count_traj() {  # $1=dataset dir -> #episodes or 0
  conda run -n monkey-verifier python3 -c "
import zarr, glob, sys
zs = glob.glob('$1/*.zarr')
try: print(len(zarr.open(zs[0], 'r')['meta']['episode_ends'][:]) if zs else 0)
except Exception: print(0)" 2>/dev/null || echo 0
}

# task : dataset-basename : target-N : gen-job-name
for spec in \
  "stack:stack_cube_strict:100:sub_stack_100" \
  "stack:stack_cube_strict:50:sub_stack_50" \
  "stack:stack_cube_strict:25:sub_stack_25" \
  "spoon:spoon_on_towel_strict:100:sub_spoon_100" \
  "spoon:spoon_on_towel_strict:50:sub_spoon_50" \
  "spoon:spoon_on_towel_strict:25:sub_spoon_25"; do
  IFS=: read -r task base N genjob <<< "$spec"
  DS="$DATA/${base}_sub${N}_diverse_start"
  RN="bc_unet_h8_50k_${task}_sub${N}_nobs1"
  jn="bc_${task}_sub${N}_nobs1"
  RUN_DIR="diffusion_policy/data/train/bc_denoiser/$RN"

  # already launched?
  if [[ -d "$RUN_DIR" ]] || grep -qxF "$jn" <<< "$QNAMES"; then continue; fi
  all_launched=0
  # dataset gen still running -> wait
  if grep -qxF "$genjob" <<< "$QNAMES"; then
    echo "[$(ts)] $task sub$N: gen job '$genjob' still running" >> "$LOG"; continue
  fi
  # verify trajectory count == target
  n=$(count_traj "$DS")
  if [[ "$n" != "$N" ]]; then
    echo "[$(ts)] $task sub$N: #traj=$n != $N (gen done? not launching)" >> "$LOG"; continue
  fi
  # kick off training on a free GPU, lean CPU
  jid=$(sbatch --parsable --job-name="$jn" \
        --account=weirdlab --partition=gpu-l40 --gres=gpu:l40:1 --cpus-per-task=4 --mem=48G --time=24:00:00 \
        --export=ALL,DENOISER=unet,HORIZON=8,N_OBS_STEPS=1,MAX_GRADIENT_STEPS=50000,LR_DECAY_STEPS=50000,DATASET_DIR="$DS",RUN_NAME="$RN",WANDB_NAME="unet_denoiser_pred8_roll4_50k_${task}_sub${N}_nobs1",WANDB_PROJECT="BC_${task}" \
        scriptsv2/train_search/train_bc_denoiser.sbatch 2>>"$LOG")
  echo "[$(ts)] LAUNCHED $jn -> $jid  (#traj=$n, DS=$DS)" >> "$LOG"
done

if [[ "$all_launched" == "1" ]]; then
  echo "[$(ts)] all 6 subset trainings launched -> removing watcher cron" >> "$LOG"
  crontab -l 2>/dev/null | grep -v 'WATCHSUBSETS' | crontab - 2>/dev/null || true
fi
echo "[$(ts)] === watch tick end ===" >> "$LOG"
