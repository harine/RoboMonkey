#!/bin/bash
# Fire the 20k rmse-filtered BoN+Q-video evals the instant each 20k checkpoint
# lands (pre-empts the up-to-30-min crontab granularity). Runs the idempotent
# launcher under the crontab's flock so it never races/double-submits. Exits once
# both diff & gauss 20k cells are queued (or ~8h timeout). Crontab is the backstop.
cd /mmfs1/home/harine/RoboMonkey
LOCK=data/train/pg/.pg_cron.lock
cnt () { echo $(( $(squeue -u harine -h -o '%j'|grep -cE "rmse_$1_20k_n") + $(ls data/eval/search_dqc_chunk/rmse_$1_20k_*_n*/eval_log.json 2>/dev/null|wc -l) )); }
for i in $(seq 1 170); do
  ( flock -w 90 9 && bash scriptsv2/eval_search/launch_rmse_ckpt_evals.sh >/dev/null 2>&1 ) 9>"$LOCK"
  d=$(cnt diff); g=$(cnt gauss)
  echo "$(date '+%m-%d %H:%M') 20k cells: diff=$d/7 gauss=$g/7"
  [ "$d" -ge 7 ] && [ "$g" -ge 7 ] && { echo "both 20k evals queued -> watcher done"; break; }
  sleep 180
done
