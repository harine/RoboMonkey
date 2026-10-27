#!/usr/bin/env bash
# Move the non-strict SimplerEnv datasets out of the working data root.
#
# Only strict-filtered datasets are used from now on (strict_distance_success,
# stack_cube_strict, spoon_on_towel_strict). Archived: the raw collections those were
# filtered from, and the diverse-start sub-samples / complement test sets that the split
# manifests under diffusion_policy/config/splits/simpler/ replace.
#
# NOT touched: carrot_on_plate (no strict derivative yet), *_lerobot (DQC finetune data),
# eggplant_in_basket_val*, eggplant_reset_core_*, _collect_logs.
#
# Dry run by default; set DO_IT=1 to move. Run only AFTER the eggplant seed map is complete
# (scriptsv2/analysis/fix_eggplant_seeds.py reads the raw eggplant_in_basket shards).
set -euo pipefail
DATA="${DATA:-/gscratch/robotics/harine/data}"
ARCHIVE="${ARCHIVE:-$DATA/_archive}"
DO_IT="${DO_IT:-0}"

TO_ARCHIVE=(
  eggplant_in_basket
  eggplant_in_basket_complete
  stack_cube
  spoon_on_towel
  strict_distance_success_sub25_diverse_start
  strict_distance_success_sub50_diverse_start
  strict_distance_success_sub100_diverse_start
  strict_distance_success_testset_not_in_sub100
  stack_cube_strict_sub25_diverse_start
  stack_cube_strict_sub50_diverse_start
  stack_cube_strict_sub100_diverse_start
  stack_cube_strict_testset_not_in_sub100
  spoon_on_towel_strict_sub25_diverse_start
  spoon_on_towel_strict_sub50_diverse_start
  spoon_on_towel_strict_sub100_diverse_start
  spoon_on_towel_strict_testset_not_in_sub100
)

# Explicit dataset names on the command line replace the default list, e.g. the strict
# datasets once their split-only copies (scriptsv2/data/make_split_dataset.py) exist:
#   DO_IT=1 bash scriptsv2/data/archive_datasets.sh strict_distance_success stack_cube_strict
if [ "$#" -gt 0 ]; then TO_ARCHIVE=("$@"); fi

mkdir -p "$ARCHIVE"
MANIFEST="$ARCHIVE/manifest.jsonl"
for d in "${TO_ARCHIVE[@]}"; do
  src="$DATA/$d"
  [ -e "$src" ] || { echo "skip (absent): $d"; continue; }
  size=$(du -sh "$src" 2>/dev/null | cut -f1)
  md5=$( [ -f "$src/filter_manifest.json" ] && md5sum "$src/filter_manifest.json" | cut -d' ' -f1 || echo null )
  echo "archive: $d ($size) -> $ARCHIVE/$d"
  if [ "$DO_IT" = "1" ]; then
    mv "$src" "$ARCHIVE/$d"
    printf '{"name":"%s","orig":"%s","archived":"%s","size":"%s","filter_manifest_md5":"%s","date":"%s"}\n' \
      "$d" "$src" "$ARCHIVE/$d" "$size" "$md5" "$(date -Is)" >> "$MANIFEST"
  fi
done
[ "$DO_IT" = "1" ] || echo "(dry run; DO_IT=1 to move)"
