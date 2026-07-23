#!/bin/bash
# Consolidate the verifier-comparison results under data/eval/_verifier_comparison.
# Idempotent; copies (does not move) so in-flight jobs are safe.
#
# The critic is judged ONLY by BoN SUCCESS RATE in sim (build_bon_table.py). The old
# a*-match results (advantage Q(a*)-Q(a_pred), Q-vs-RMSE-to-a*, ranking, progression) are
# the WRONG eval and are ARCHIVED under
#   data/eval/_verifier_comparison/verifier_match_a_expert_eval/
# (scripts: scriptsv2/q_fn_eval/verifier_match_a_expert_eval/). This gather no longer
# collects them; it consolidates only the success-referenced + video artifacts + the doc.
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
ROOT=data/eval/_verifier_comparison
FT=$ROOT/_finetuned_dqc
mkdir -p "$FT/success_fail" "$FT/videos"

# ---- finetuned: success-vs-fail Q plots (success-referenced, NOT a*-match) ----
cp -f data/viz_q/_sf/*.png "$FT/success_fail/" 2>/dev/null

# ---- finetuned: re-rendered Q-overlay videos (preserve <tag>/<run>/<split>) ----
if [ -d data/viz_q/rescore ]; then
  while IFS= read -r f; do
    rel=${f#data/viz_q/rescore/}                 # <tag>/<run>/<split>/search_q/videos/x.mp4
    rel=${rel/\/search_q\/videos\//\/}            # -> <tag>/<run>/<split>/x.mp4
    mkdir -p "$FT/videos/$(dirname "$rel")"
    cp -f "$f" "$FT/videos/$rel"
  done < <(find data/viz_q/rescore -path '*/search_q/videos/*.mp4' 2>/dev/null)
fi

# ---- the results doc ----
cp -f scriptsv2/q_fn_eval/finetuned_dqc_june24.md "$ROOT/" 2>/dev/null

echo "==== consolidated under $ROOT (success-rate eval only) ===="
echo "ft success_fail: $(ls "$FT/success_fail"/*.png 2>/dev/null | wc -l) png"
echo "ft videos      : $(find "$FT/videos" -name '*.mp4' 2>/dev/null | wc -l) mp4"
echo "BoN success    : data/eval/{search_dqc_chunk,bc_dqc_chunk}/*_{v2all30k,v3dist30k}_*  (build_bon_table.py)"
echo "a*-match (ARCHIVED, do not use): $ROOT/verifier_match_a_expert_eval/"
echo "llava videos   : $(find "$ROOT/videos" -name '*.mp4' 2>/dev/null | wc -l) mp4 (pre-existing)"
