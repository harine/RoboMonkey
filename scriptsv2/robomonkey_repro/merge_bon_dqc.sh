#!/bin/bash
# Merge the parallel DQC BoN-sweep cells (par/g*/progress.json) into one curve
# and overlay the LLaVA sweep for comparison.
#   bash scriptsv2/robomonkey_repro/merge_bon_dqc.sh
set -u
REPO=/mmfs1/home/harine/RoboMonkey
PAR="$REPO/data/eval/robomonkey_repro/bon_sweep_dqc/par"
LLAVA="$REPO/data/eval/robomonkey_repro/bon_sweep/sweep_summary.txt"
OUT="$REPO/data/eval/robomonkey_repro/bon_sweep_dqc/SWEEP_COMPARISON.md"

python3 - "$PAR" "$LLAVA" "$OUT" <<'PY'
import json, sys, glob, os, re
par, llava_path, out = sys.argv[1:4]

# Aggregate DQC cells.
dqc = {}
for pj in glob.glob(os.path.join(par, "*", "progress.json")):
    try:
        d = json.load(open(pj))
    except Exception:
        continue
    for n, v in d.items():
        a = dqc.setdefault(int(n), {"episodes": 0, "success": 0})
        # cells are disjoint by N, but be safe: take the row with more episodes
        if v.get("episodes", 0) > a["episodes"]:
            a["episodes"] = v["episodes"]; a["success"] = v["success"]

# Parse the LLaVA sweep summary (lines like "N   54.0%   27/50").
llava = {}
if os.path.exists(llava_path):
    for line in open(llava_path):
        m = re.match(r"\s*(\d+)\s+([\d.]+)%", line)
        if m:
            llava[int(m.group(1))] = float(m.group(2))

lines = []
lines.append("# OpenVLA RoboMonkey BoN scaling — DQC verifier vs LLaVA verifier")
lines.append("task=eggplant  seed=1  init_samples=9  (augmented_samples = N)")
lines.append("")
lines.append("| N | DQC SR | DQC n | LLaVA SR | delta |")
lines.append("|---|--------|-------|----------|-------|")
for n in sorted(dqc):
    e = dqc[n]["episodes"]; s = dqc[n]["success"]
    dsr = 100.0 * s / e if e else 0.0
    lv = llava.get(n)
    lvs = f"{lv:.1f}%" if lv is not None else "-"
    dl = f"{dsr-lv:+.1f}" if lv is not None else "-"
    lines.append(f"| {n} | {dsr:.1f}% | {s}/{e} | {lvs} | {dl} |")

txt = "\n".join(lines) + "\n"
open(out, "w").write(txt)
print(txt)
print(f"[wrote] {out}")
PY
