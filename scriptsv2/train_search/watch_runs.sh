#!/bin/bash
# Show progress of all running noise-sweep search-policy runs (auto-discovers by
# job id -> log).
#   bash scriptsv2/train_search/watch_runs.sh            # one snapshot
#   bash scriptsv2/train_search/watch_runs.sh --loop     # auto-refresh (30s)
#   bash scriptsv2/train_search/watch_runs.sh --loop 10  # auto-refresh (10s)
cd "$(dirname "$0")/../.." || exit 1
LOGDIR=data/train/noise_sweep

render() {
  printf '%-12s %-9s %-8s %-9s %s\n' RUN STATE ELAPSED ACCT PROGRESS
  squeue -u "$USER" -h -o "%i|%T|%M|%a" 2>/dev/null | while IFS='|' read -r id state elapsed acct; do
    f=$(ls -t "$LOGDIR"/*job${id}.log 2>/dev/null | head -1)
    [ -z "$f" ] && continue
    tag=$(basename "$f" | grep -oE 'tmrl|tcorrupt[0-9.]+'); tag=${tag:-job$id}
    prog=$(tr '\r' '\n' < "$f" 2>/dev/null | grep -aoE 'Training epoch [0-9]+: +[0-9]+%[^]]*]' | tail -1)
    printf '%-12s %-9s %-8s %-9s %s\n' "$tag" "$state" "$elapsed" "$acct" "${prog:-loading...}"
  done
}

if [ "$1" = "--loop" ] || [ "$1" = "-w" ]; then
  interval="${2:-30}"
  trap 'echo; exit 0' INT
  while true; do
    printf '\033[2J\033[H'   # clear screen (no TERM dependency)
    echo "RoboMonkey noise-sweep runs — $(date '+%Y-%m-%d %H:%M:%S')  (refresh ${interval}s, Ctrl-C to stop)"
    render
    sleep "$interval"
  done
else
  render
fi
