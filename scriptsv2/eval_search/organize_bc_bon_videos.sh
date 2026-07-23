#!/bin/bash
# -----------------------------------------------------------------------------
# Copy the BC-policy BoN Q-videos rendered by render_bc_bon_videos.sh
# (data/eval/bc_dqc_chunk/vidbc_<task>_<sub>_n<N>_seed1000_ep5/search_q/videos/)
# into a bc/ subfolder of each category's search-video folder:
#   eggplant sub100 -> filtered_s100_diverse_eggplant/bc/
#   eggplant sub50  -> filtered_sub50_diverse_eggplant/bc/
#   eggplant sub25  -> filtered_sub25_diverse_eggplant/bc/
#   stack <sub>     -> stack_cube_search/<sub>/bc/
#   spoon <sub>     -> spoon_towel_search/<sub>/bc/
# Videos already carry __N<N>_<succ/trunc> tags. Idempotent (cp -n). Re-run any time.
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
E=data/eval; NS="1 8 64"; copied=0
dest_of () {  # $1=task $2=sub -> prints the bc/ dest folder
  case "$1" in
    eggplant) case "$2" in sub100) echo "$E/filtered_s100_diverse_eggplant/bc";;
                           sub50)  echo "$E/filtered_sub50_diverse_eggplant/bc";;
                           sub25)  echo "$E/filtered_sub25_diverse_eggplant/bc";; esac;;
    stack)    echo "$E/stack_cube_search/$2/bc";;
    spoon)    echo "$E/spoon_towel_search/$2/bc";;
  esac
}
for task in eggplant stack spoon; do for s in sub100 sub50 sub25; do
  dest=$(dest_of "$task" "$s"); any=0
  for N in $NS; do
    vid="$E/bc_dqc_chunk/vidbc_${task}_${s}_n${N}_seed1000_ep5/search_q/videos"
    ls "$vid"/*.mp4 >/dev/null 2>&1 || continue
    [ "$any" = 0 ] && { mkdir -p "$dest"; any=1; }
    for f in "$vid"/*.mp4; do bn=$(basename "$f"); [ -f "$dest/$bn" ] && continue; cp -p "$f" "$dest/$bn" && copied=$((copied+1)); done
  done
  [ "$any" = 1 ] && echo "  $task $s -> $dest ($(ls "$dest"/*.mp4 2>/dev/null|wc -l) BC videos)"
done; done
echo "=== BC videos organized: $copied copied ==="
