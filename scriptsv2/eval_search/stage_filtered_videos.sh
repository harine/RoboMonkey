#!/bin/bash
# -----------------------------------------------------------------------------
# Collect the rendered Q-overlay videos (candidate action trajectories colored by
# verifier-Q rank + Q-values as text) from each rmse eval cell's search_q/videos/
# into organized <run>_<tag>k/n<N>/ trees:
#   filtered / sub100  (cells in search_dqc_chunk/)  -> filtered_success_eggplant/
#   sub50              (cells in filtered_sub50.../)  -> filtered_sub50.../videos/
# Idempotent (skips already-staged files). Run standalone or from the crontab tick.
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
n=0
stage () {  # $1=glob of cell dirs  $2=dest root
  for d in $1; do
    ls "$d"/search_q/videos/*.mp4 >/dev/null 2>&1 || continue
    key=$(basename "$d" | sed -E 's/_chunk8_predh8_roll4_softmax//; s/_seed1000_ep50//')  # rmse_diff_10k_n8
    runtag=$(echo "$key" | sed -E 's/_n[0-9]+$//')      # rmse_diff_10k
    nn=$(echo "$key" | grep -oE 'n[0-9]+$')             # n8
    tgt="$2/$runtag/$nn"; mkdir -p "$tgt"
    for f in "$d"/search_q/videos/*.mp4; do
      bn=$(basename "$f"); [ -f "$tgt/$bn" ] && continue
      cp -p "$f" "$tgt/$bn" && n=$((n+1))
    done
  done
}
stage "data/eval/search_dqc_chunk/rmse_*k_chunk8_predh8_roll4_softmax_n*_seed1000_ep50" \
      data/eval/filtered_success_eggplant
stage "data/eval/filtered_sub50_diverse_eggplant/rmse_sub50_*k_chunk8_predh8_roll4_softmax_n*_seed1000_ep50" \
      data/eval/filtered_sub50_diverse_eggplant/videos
stage "data/eval/filtered_sub25_diverse_eggplant/rmse_sub25_*k_chunk8_predh8_roll4_softmax_n*_seed1000_ep50" \
      data/eval/filtered_sub25_diverse_eggplant/videos
echo "[stage_filtered_videos] staged $n new mp4"
