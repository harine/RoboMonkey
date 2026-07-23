#!/bin/bash
# Watch progress of ALL active RoboMonkey runs (training + eval + repro).
# Auto-discovers each job's slurm log by job id under data/ and prints a
# type-appropriate progress line.
#
#   bash scriptsv2/watch_all.sh            # one snapshot
#   bash scriptsv2/watch_all.sh --loop     # auto-refresh (30s)
#   bash scriptsv2/watch_all.sh --loop 10  # auto-refresh (10s)
cd "$(dirname "$0")/.." || exit 1   # -> repo root

# Dirs that hold slurm-*.out / *.log files for any run family.
LOGDIRS=(
  data/train/noise_sweep
  data/train/bc
  data/train/chunksum
  data/eval/search_bc_sweep/_summaries
  data/eval/search_noise_sweep/_summaries
  data/eval/search_nas_sweep/_summaries
  data/eval/robomonkey_repro
  data/eval/_smoke
  /gscratch/robotics/harine/data/_collect_logs
)

find_log() {  # $1=jobid -> newest matching log path (or empty)
  local id="$1" f
  for d in "${LOGDIRS[@]}"; do
    f=$(ls -t "$d"/*"$id"*.out "$d"/*"$id"*.log 2>/dev/null | head -1)
    [ -n "$f" ] && { echo "$f"; return; }
  done
}

progress() {  # $1=logfile -> single progress line
  local f="$1" t prog
  [ -z "$f" ] || [ ! -f "$f" ] && { echo "(no log yet)"; return; }
  t=$(tr '\r' '\n' < "$f" 2>/dev/null)
  # 1) training: tqdm epoch bar
  prog=$(echo "$t" | grep -aoE 'Training epoch [0-9]+: +[0-9]+%[^]]*]' | tail -1)
  [ -n "$prog" ] && { echo "$prog"; return; }
  # 2) eval sweep: count results + last result/start line
  if echo "$t" | grep -aqE '\[result\]|\[gpu=.* (start|done)'; then
    local n; n=$(echo "$t" | grep -acE '^\[result\]')
    prog=$(echo "$t" | grep -aE '\[result\]|\[gpu=.* start' | tail -1 | sed 's/  */ /g')
    echo "done=$n | $prog"; return
  fi
  # 3) data collection: rollout tqdm with episode count + success_rate
  prog=$(echo "$t" | grep -aoE '\[[0-9]+/[0-9]+\] episodes=[0-9]+ steps=[0-9]+ success_rate=[0-9.]+' | tail -1)
  [ -n "$prog" ] && { echo "collect $prog"; return; }
  # 4) robomonkey repro: server bring-up / eval status
  prog=$(echo "$t" | grep -aE '\[wait\]|\[verifier\]|\[vla\]|succ=|trial|seed' | tail -1)
  [ -n "$prog" ] && { echo "$prog"; return; }
  # 5) fallback: last non-empty line
  echo "$t" | grep -av '^$' | tail -1 | cut -c1-90
}

render() {
  printf '%-16s %-9s %-9s %-8s %s\n' JOB STATE ELAPSED ID PROGRESS
  printf '%-16s %-9s %-9s %-8s %s\n' --- ----- ------- -- --------
  squeue -u "$USER" -h -o "%i|%j|%T|%M" 2>/dev/null \
    | sort -t'|' -k2 | while IFS='|' read -r id name state elapsed; do
      local f reason=""
      if [ "$state" = "PENDING" ]; then
        reason=$(squeue -j "$id" -h -o "%r" 2>/dev/null)
        printf '%-16s %-9s %-9s %-8s %s\n' "${name:0:16}" "$state" "$elapsed" "$id" "($reason)"
        continue
      fi
      f=$(find_log "$id")
      printf '%-16s %-9s %-9s %-8s %s\n' "${name:0:16}" "$state" "$elapsed" "$id" "$(progress "$f")"
    done
}

if [ "$1" = "--loop" ] || [ "$1" = "-w" ]; then
  interval="${2:-30}"
  trap 'echo; exit 0' INT
  while true; do
    printf '\033[2J\033[H'
    echo "RoboMonkey — all runs — $(date '+%Y-%m-%d %H:%M:%S')  (refresh ${interval}s, Ctrl-C to stop)"
    render
    sleep "$interval"
  done
else
  render
fi
