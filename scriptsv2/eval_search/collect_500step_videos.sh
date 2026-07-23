#!/bin/bash
# -----------------------------------------------------------------------------
# Collect the action+Q videos rendered by launch_500step_sub50_evals.sh (staged
# under data/eval/_500step_runs/<kind>_<task>_<ck>_n<N>/search_q/videos/) into the
# same category folders as the 120-step videos, in a <ck>_500steps/ subfolder:
#
#   SEARCH  spoon @30k -> data/eval/spoon_towel_search/sub50/30k_500steps/
#           stack @20k -> data/eval/stack_cube_search/sub50/20k_500steps/
#           eggpl @70k -> data/eval/filtered_sub50_diverse_eggplant/70k_500steps/
#   BC      stack @30k -> data/eval/stack_cube_search/sub50/bc/30k_500steps/
#           spoon @10k -> data/eval/spoon_towel_search/sub50/bc/10k_500steps/
#           eggpl @50k -> data/eval/filtered_sub50_diverse_eggplant/bc/50k_500steps/
#
# Video filenames already carry __N<N>_<succ/trunc/fail>, so all N pool cleanly.
# Idempotent (cp -n). Also prints the 500-step success rate per cell from eval_log.json.
#   bash scriptsv2/eval_search/collect_500step_videos.sh
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
STAGE=data/eval/_500step_runs; E=data/eval; N_LIST="1 2 4 8 16 32 64 128 256 512"; copied=0
# task -> category root for the <ck>_500steps folder
cat_of () { case "$1" in stack) echo "$E/stack_cube_search/sub50";; spoon) echo "$E/spoon_towel_search/sub50";; eggplant) echo "$E/filtered_sub50_diverse_eggplant";; esac; }
# kind|task|cktag
CELLS=( "search|spoon|30k" "search|stack|20k" "search|eggplant|70k"
        "bc|stack|30k" "bc|spoon|10k" "bc|eggplant|50k" )
echo "=== 500-step SR + video collect ==="
for c in "${CELLS[@]}"; do
  IFS='|' read -r kind task ck <<< "$c"
  root=$(cat_of "$task")
  if [ "$kind" = search ]; then dest="$root/${ck}_500steps"; else dest="$root/bc/${ck}_500steps"; fi
  mkdir -p "$dest"; any=0
  for N in $N_LIST; do
    cell="$STAGE/${kind}_${task}_${ck}_n${N}"
    # success rate line
    sr=$(python3 - "$cell/eval_log.json" 2>/dev/null <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1])); n=int(d.get("num_episodes",0)); s=int(d.get("num_successes",d.get("num_success",d.get("successes",0))))
    print(f"{s}/{n} ({100*s/n:.0f}%)" if n else "no-data")
except Exception: print("pending")
PY
)
    printf "  %-7s %-9s %-3s N=%-2s : SR %s\n" "$kind" "$task" "$ck" "$N" "$sr"
    vd="$cell/search_q/videos"
    ls "$vd"/*.mp4 >/dev/null 2>&1 || continue
    any=1
    for f in "$vd"/*.mp4; do bn=$(basename "$f"); [ -f "$dest/$bn" ] && continue; cp -p "$f" "$dest/$bn" && copied=$((copied+1)); done
  done
  [ "$any" = 1 ] && echo "    -> $dest ($(ls "$dest"/*.mp4 2>/dev/null|wc -l) videos)"
done
echo "=== collected: $copied videos ==="
