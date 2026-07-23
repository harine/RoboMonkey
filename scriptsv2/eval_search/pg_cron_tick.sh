#!/bin/bash
# -----------------------------------------------------------------------------
# Autonomous 30-min tick, run by the klone login-node crontab (NOT a Claude
# session) -> the PG BoN eval pipeline keeps progressing even when no session is
# open. A login-node cron is the only mechanism that is both durable across
# session close AND has the Slurm + /gscratch access the work needs (an Anthropic
# cloud agent has neither).
#
# Each tick: (1) launch any ready N=1..64 BoN eval cells whose PG checkpoint has
# reached a 10k/20k/30k/40k/50k milestone (idempotent: skips done/queued), then
# (2) refresh the results doc. flock prevents overlapping ticks; all output is
# appended to data/train/pg/pg_cron.log.
#
# Install (every 30 min, on the offset so it never races a manual run):
#   (crontab -l 2>/dev/null; echo '*/30 * * * * /bin/bash /mmfs1/home/harine/RoboMonkey/scriptsv2/eval_search/pg_cron_tick.sh') | crontab -
# Inspect:  crontab -l        Disable:  crontab -r   (or remove just this line)
# -----------------------------------------------------------------------------
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"   # cron's PATH is minimal; need sbatch+squeue
export USER="${USER:-harine}"
cd /mmfs1/home/harine/RoboMonkey || exit 1
LOG=data/train/pg/pg_cron.log
LOCK=data/train/pg/.pg_cron.lock
mkdir -p data/train/pg

exec 9>"$LOCK"
flock -n 9 || { echo "$(date '+%F %T') tick skipped (previous still running)" >>"$LOG"; exit 0; }

{
  echo "===== tick $(date '+%F %T')  host=$(hostname) ====="
  # -------------------------------------------------------------------------
  # ENTIRE pre-fix eval + video pipeline DISABLED 2026-07-08. All of it (PG /
  # rmse / multitask milestone BoN evals, seeded reruns, and every video
  # render/organize step) was env-capped at the task-native horizon (stack/spoon
  # 60, eggplant 120, not MAX_STEPS) AND nondeterministic (cuDNN/JAX). The data +
  # videos are archived in data/archive_gpu_stochastic/. It kept re-spawning
  # hundreds of superseded jobs, so it is fully off. The new DETERMINISTIC,
  # true-horizon, tiled eval (sub50: spoon 90k / stack 60k / eggplant 90k) is
  # launched + tracked manually, not from this cron.
  #   bash scriptsv2/eval_search/launch_pg_ckpt_evals.sh
  #   bash scriptsv2/eval_search/launch_rmse_ckpt_evals.sh
  #   bash scriptsv2/eval_search/launch_multitask_ckpt_evals.sh
  #   bash scriptsv2/eval_search/launch_seeded_ckpt_evals.sh
  #   bash scriptsv2/eval_search/stage_filtered_videos.sh
  #   bash scriptsv2/eval_search/organize_eggplant_search_videos.sh
  #   bash scriptsv2/eval_search/organize_multitask_search_videos.sh
  #   bash scriptsv2/eval_search/organize_bc_bon_videos.sh
  #   bash scriptsv2/eval_search/render_seeded_videos.sh
  #   bash scriptsv2/eval_search/collect_seeded_videos.sh
  #   bash scriptsv2/eval_search/render_pg_milestone_videos.sh
  # -------------------------------------------------------------------------
  :
  # doc refresh needs the monkey-verifier env (torch/matplotlib/zarr)
  if source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null && conda activate monkey-verifier 2>/dev/null; then
    python scriptsv2/analysis/aggregate_verifier_results.py >/dev/null 2>&1 \
      && echo "doc refreshed" || echo "doc refresh FAILED"
    # traj plots DISABLED 2026-07-08 (seeded video dumps archived; rebuilding for det. eval)
    # python scriptsv2/analysis/plot_seeded_trajectories.py >/dev/null 2>&1 \
    #   && echo "traj plots refreshed" || echo "traj plots FAILED"
  else
    echo "doc refresh skipped (env activate failed)"
  fi
  echo "----- tick done $(date '+%F %T') -----"
} >>"$LOG" 2>&1
