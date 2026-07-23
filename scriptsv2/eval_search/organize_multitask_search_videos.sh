#!/bin/bash
# -----------------------------------------------------------------------------
# Organize the stack_cube + spoon_on_towel SEARCH-policy Q-overlay videos into
# per-(sub,checkpoint) subfolders, mirroring organize_eggplant_search_videos.sh.
# Each eval cell rendered 10 videos/episode under search_q/videos/, named
# ep<ep>_seed<seed>__N<N>_<succ|trunc|fail>.mp4 (N + status tags already baked in,
# and render_q draws them in-frame). This COPIES all N's videos for each
# (sub, checkpoint) into <task_dir>/<sub>/<ck>/.
#
#   stack -> data/eval/stack_cube_search/{sub100,sub50,sub25}/<ck>/
#   spoon -> data/eval/spoon_towel_search/{sub100,sub50,sub25}/<ck>/
#
# Idempotent (cp -n). Re-run to pick up newly-rendered checkpoints/N.
#   bash scriptsv2/eval_search/organize_multitask_search_videos.sh
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
CKPTS="5k 10k 20k 30k 40k 50k 60k 70k 80k 90k 100k"
NS="1 2 4 8 16 32 64 128 256 512"
E=data/eval
copied=0; made=0

organize () {  # $1=label(e.g. stack_sub100)  $2=src_root  $3=dest_root(<task>/<sub>)
  local label="$1" src="$2" dest="$3"
  for ck in $CKPTS; do
    local ckdest="$dest/$ck" any=0
    for N in $NS; do
      local vid="$src/${label}_${ck}_chunk8_predh8_roll4_softmax_n${N}_seed1000_ep50/search_q/videos"
      ls "$vid"/*.mp4 >/dev/null 2>&1 || continue
      if [ "$any" = 0 ]; then mkdir -p "$ckdest" && made=$((made+1)); any=1; fi
      for f in "$vid"/*.mp4; do
        bn=$(basename "$f"); [ -f "$ckdest/$bn" ] && continue
        cp -p "$f" "$ckdest/$bn" && copied=$((copied+1))
      done
    done
    [ "$any" = 1 ] && echo "  $label $ck -> $ckdest ($(ls "$ckdest"/*.mp4 2>/dev/null|wc -l) videos)"
  done
}

for sub in sub100 sub50 sub25; do
  organize "stack_${sub}" "$E/stack_cube_search"  "$E/stack_cube_search/$sub"
done
for sub in sub100 sub50 sub25; do
  organize "spoon_${sub}" "$E/spoon_towel_search" "$E/spoon_towel_search/$sub"
done

echo "=== organized: $copied videos copied into $made (sub,checkpoint) folders ==="
