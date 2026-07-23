#!/bin/bash
# -----------------------------------------------------------------------------
# Render the expert-overlay videos for the random-augmentation rollouts.
#
# For each rollout cell ({train,test} x K) produced by launch_random_aug.sh:
#   1. augment_npz_with_expert.py  -> attaches the demo's expert EE path + demo_split
#   2. render_rollout_vs_expert.py -> mp4 per seed into <root>/videos, tagged k<K>
#      (policy candidates solid + colored by Q rank, RANDOM candidates dashed,
#       chosen chunk thick red, expert dashed white, executed path cyan)
# Then symlinks the EXISTING rmse@90k baseline videos (same seeds, no augmentation)
# into <root>/videos/baseline/ for side-by-side comparison -- they are not re-rendered.
#
#   conda activate monkey-verifier
#   bash scriptsv2/analysis/render_random_aug_videos.sh
#   K_LIST=50 SPLITS=train bash scriptsv2/analysis/render_random_aug_videos.sh
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey

ROOT="${ROOT:-data/eval/random_a_diverse/spoon_on_towel}"
BASE_VIDEOS="${BASE_VIDEOS:-data/eval/train_test_analysis/spoon/videos}"
SPLITS="${SPLITS:-train test}"
K_LIST="${K_LIST:-16 50}"
FPS="${FPS:-3}"

made=0
for split in $SPLITS; do
  for K in $K_LIST; do
    d="$ROOT/rollout/${split}_k${K}"
    ls "$d"/search_q/*.npz >/dev/null 2>&1 || { echo "skip (no npz): $d"; continue; }
    echo "=== $split k=$K"
    python scriptsv2/analysis/augment_npz_with_expert.py \
      --task spoon --split "$split" --dir "$d" || continue
    python scriptsv2/analysis/render_rollout_vs_expert.py \
      --dir "$d" --out "$ROOT/videos" --fps "$FPS" --tag "k${K}" || continue
    made=$((made+1))
  done
done

# baseline (rmse @90k, no augmentation) on the SAME seeds -- link, never re-render
if [ -d "$BASE_VIDEOS" ]; then
  mkdir -p "$ROOT/videos/baseline"
  n=0
  for f in "$BASE_VIDEOS"/*.mp4; do
    [ -e "$f" ] || continue
    ln -sf "$(realpath "$f")" "$ROOT/videos/baseline/$(basename "$f")" && n=$((n+1))
  done
  echo "[render] linked $n baseline videos -> $ROOT/videos/baseline"
fi

echo "[render] rendered $made cells -> $ROOT/videos"
ls "$ROOT/videos"/*.mp4 2>/dev/null | wc -l | xargs echo "[render] total mp4s:"
