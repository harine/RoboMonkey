#!/bin/bash
# -----------------------------------------------------------------------------
# Organize the SEEDED action+Q videos (render_seeded_videos.sh, staged in
# data/eval/_seeded_videos/<kind>_<task>_<sub>_<ck>_st<steps>_n<N>/search_q/videos/)
# into a FLAT per-(task,sub) folder, step length as a filename TAG (not a folder):
#
#   data/eval/search_<task>/<sub>/seed<seed>_n<N>_<steps>steps_<search|bc>_<status>.mp4
#
# tasks: stack_cube, spoon_on_towel, eggplant.  search N=1..64, bc N=1/8/64, steps 500 & 120.
# Idempotent (skips existing). Re-run any time.
#   bash scriptsv2/eval_search/collect_seeded_videos.sh
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
# step tags: 60/120 = task-native env cap (pre-env-cap-fix data, relabeled to true
# num_steps); 500 = true long-horizon runs made AFTER the env-cap fix.
STAGE=data/eval/_seeded_videos; E=data/eval; STEPS="60 120 500"; copied=0
N_SEARCH="1 2 4 8 16 32 64"; N_BC="1 8 64"
# kind|task|sub|ck  (ck must match the render staging dirs)
CELLS=(
  "search|stack_cube|sub50|20k"     "search|stack_cube|sub100|20k"
  "search|spoon_on_towel|sub50|30k" "search|spoon_on_towel|sub100|30k"
  "search|eggplant|sub50|70k"       "search|eggplant|sub100|70k"
  "bc|stack_cube|sub50|30k"         "bc|stack_cube|sub100|40k"
  "bc|spoon_on_towel|sub50|10k"     "bc|spoon_on_towel|sub100|20k"
  "bc|eggplant|sub50|50k"           "bc|eggplant|sub100|30k"
)
echo "=== seeded video collect -> search_<task>/<sub>/ (step tag in filename) ==="
for c in "${CELLS[@]}"; do
  IFS='|' read -r kind task sub ck <<< "$c"
  dest="$E/search_${task}/${sub}"; mkdir -p "$dest"; any=0
  nlist="$N_SEARCH"; [ "$kind" = bc ] && nlist="$N_BC"
  for st in $STEPS; do for N in $nlist; do
    vd="$STAGE/${kind}_${task}_${sub}_${ck}_st${st}_n${N}/search_q/videos"
    ls "$vd"/*.mp4 >/dev/null 2>&1 || continue
    any=1
    for f in "$vd"/*.mp4; do
      bn=$(basename "$f")
      seed=$(sed -E 's/.*_seed([0-9]+)__.*/\1/' <<< "$bn")
      Nn=$(sed -E 's/.*__N([0-9]+)_.*/\1/' <<< "$bn")
      status=$(sed -E 's/.*_N[0-9]+_([a-z]+)\.mp4/\1/' <<< "$bn")
      new="seed${seed}_n${Nn}_${st}steps_${kind}_${status}.mp4"
      [ -f "$dest/$new" ] && continue
      cp -p "$f" "$dest/$new" && copied=$((copied+1))
    done
  done; done
  [ "$any" = 1 ] && echo "  $kind $task $sub -> $dest ($(ls "$dest"/*.mp4 2>/dev/null|wc -l) videos)"
done
echo "=== collected: $copied videos ==="
