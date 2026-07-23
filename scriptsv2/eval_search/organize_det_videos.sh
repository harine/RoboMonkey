#!/bin/bash
# -----------------------------------------------------------------------------
# Organize the DETERMINISTIC-campaign viz videos into per-task, per-selection folders:
#   softmax: data/eval/<task>/softmax_sub50_<search_ck>k/seed<seed>_n<N>_<steps>steps_<kind>_<status>.mp4
#   argmax : data/eval/<task>/argmax_sub50_<search_ck>k/seed<seed>_n<N>_<steps>steps_<kind>_<status>.mp4
# (The softmax folder was named plain `sub50_<ck>k` before 2026-07-21; renamed so both
#  folders state their selection mode.)
# Sources:
#   softmax: data/eval/<task>/_viz/<task>_<mk>_st<steps>_n<N>/search_q/videos/*.mp4
#   argmax : data/eval/<task>/_viz_argmax/<task>_<mk>_st<steps>_n<N>/search_q/videos/*.mp4
#   (mk = search milestone 90/60, or "bc"; steps = 60/120/500). search + bc land in the
# same folder, distinguished by the _search_/_bc_ tag. The raw _viz*/ cells are LEFT IN
# PLACE (they also hold the search_q npz used by the trajectory plots).
# Idempotent. Re-run any time.
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
E=data/eval; copied=0
declare -A SMK=( [eggplant]=90 [stack_cube]=60 [spoon_on_towel]=90 )
# "<viz_subdir>|<dest_prefix>"
VARIANTS=( "_viz|softmax_sub50" "_viz_argmax|argmax_sub50" )
for task in eggplant stack_cube spoon_on_towel; do
  smk=${SMK[$task]}
  for spec in "${VARIANTS[@]}"; do
    IFS='|' read -r vsub dprefix <<< "$spec"
    [ -d "$E/$task/$vsub" ] || continue
    dest="$E/$task/${dprefix}_${smk}k"; mkdir -p "$dest"
    for cell in "$E/$task/$vsub/${task}"_*_st*_n*; do
      [ -d "$cell" ] || continue
      bn=$(basename "$cell")
      mk=$(sed -E "s/^${task}_([^_]+)_st.*/\1/" <<< "$bn")
      steps=$(sed -E 's/.*_st([0-9]+)_n.*/\1/' <<< "$bn")
      N=$(sed -E 's/.*_n([0-9]+)$/\1/' <<< "$bn")
      kind=$([ "$mk" = bc ] && echo bc || echo search)
      vd="$cell/search_q/videos"; ls "$vd"/*.mp4 >/dev/null 2>&1 || continue
      for f in "$vd"/*.mp4; do
        fb=$(basename "$f")
        seed=$(sed -E 's/.*_seed([0-9]+)__.*/\1/' <<< "$fb")
        status=$(sed -E 's/.*_N[0-9]+_([a-z]+)\.mp4/\1/' <<< "$fb")
        new="seed${seed}_n${N}_${steps}steps_${kind}_${status}.mp4"
        [ -f "$dest/$new" ] && continue
        cp -p "$f" "$dest/$new" && copied=$((copied+1))
      done
    done
    echo "  $task [$dprefix] -> $dest ($(ls "$dest"/*.mp4 2>/dev/null|wc -l) videos)"
  done
done
echo "=== organized: $copied new det videos ==="
