#!/bin/bash
# -----------------------------------------------------------------------------
# Organize the eggplant SEARCH-policy Q-overlay videos into per-checkpoint
# subfolders. Each eval cell already rendered 10 videos/episode (seeds 1000-1009)
# under search_q/videos/, named ep<ep>_seed<seed>__N<N>_<succ|trunc|fail>.mp4 --
# i.e. the N tag AND the success/truncated tag are already in the filename (and
# render_q.py also draws them in-frame, colored by status). This just COPIES all
# N's videos for each checkpoint into <dest>/<ck>/ so every checkpoint has its 10
# videos per N in one place.
#
# Runs (label | src cell root | dest folder):
#   rmse_sub100  search_dqc_chunk                 -> filtered_s100_diverse_eggplant
#   rmse_sub50   filtered_sub50_diverse_eggplant  -> (same)
#   rmse_sub25   filtered_sub25_diverse_eggplant  -> (same)
#
# Idempotent (cp -n). Re-run any time to pick up newly-rendered checkpoints/N.
#   bash scriptsv2/eval_search/organize_eggplant_search_videos.sh
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
CKPTS="5k 10k 20k 30k 40k 50k 60k 70k 80k 90k 100k"
NS="1 2 4 8 16 32 64 128 256 512"
E=data/eval
copied=0; made=0

organize () {  # $1=label  $2=src_root  $3=dest_root
  local label="$1" src="$2" dest="$3"
  for ck in $CKPTS; do
    local ckdest="$dest/$ck" any=0
    for N in $NS; do
      local vid="$src/${label}_${ck}_chunk8_predh8_roll4_softmax_n${N}_seed1000_ep50/search_q/videos"
      ls "$vid"/*.mp4 >/dev/null 2>&1 || continue
      if [ "$any" = 0 ]; then mkdir -p "$ckdest" && made=$((made+1)); any=1; fi
      for f in "$vid"/*.mp4; do
        bn=$(basename "$f")            # ep000_seed1000__N8_succ.mp4 (N + status already tagged)
        [ -f "$ckdest/$bn" ] && continue
        cp -p "$f" "$ckdest/$bn" && copied=$((copied+1))
      done
    done
    [ "$any" = 1 ] && echo "  $label $ck -> $ckdest ($(ls "$ckdest"/*.mp4 2>/dev/null|wc -l) videos)"
  done
}

organize rmse_sub100 "$E/search_dqc_chunk"                "$E/filtered_s100_diverse_eggplant"
organize rmse_sub50  "$E/filtered_sub50_diverse_eggplant" "$E/filtered_sub50_diverse_eggplant"
organize rmse_sub25  "$E/filtered_sub25_diverse_eggplant" "$E/filtered_sub25_diverse_eggplant"

echo "=== organized: $copied videos copied into $made checkpoint folders ==="
