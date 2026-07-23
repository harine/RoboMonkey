#!/usr/bin/env python
"""BoN success-rate of the FILTERED diff-10k policy across v4_filt_ramp finetune
checkpoints — to pick the best verifier checkpoint. Rows = verifier (base, then
v4_filt_ramp @ 10k..50k); cols = N. Reads eval_log.json cells; '·' = not finished.
Pure stdlib. Regenerate as cells complete: python scriptsv2/eval_search/ftckpt_bon_progression.py
"""
import json
from pathlib import Path

EVAL = Path("/mmfs1/home/harine/RoboMonkey/data/eval/search_dqc_chunk")
NS = [1, 2, 4, 8, 16, 32, 64]
PBASE = "difffilt10k"
# (label, cell tag) — base = baked (no tag), then each finetune epoch.
VERIFIERS = [("base", None), ("v4_filt@10k", "v4filt10k"), ("v4_filt@20k", "v4filt20k"),
             ("v4_filt@30k", "v4filt30k"), ("v4_filt@40k", "v4filt40k"), ("v4_filt@50k", "v4filt50k")]


def sr(tag, n):
    pfx = f"{PBASE}_chunk8_predh8_roll4_softmax" if tag is None \
        else f"{PBASE}_{tag}_chunk8_predh8_roll4_softmax"
    p = EVAL / f"{pfx}_n{n}_seed1000_ep50" / "eval_log.json"
    try:
        d = json.load(open(p))
        if int(d.get("num_episodes", 0)) >= 50:
            return int(round(100 * float(d["success_rate"])))
    except Exception:
        pass
    return None


rows = []
lines = ["# FILTERED diff-10k policy — BoN success % vs v4_filt_ramp finetune checkpoint", "",
         "| verifier | " + " | ".join(f"N{n}" for n in NS) + " | mean | best |",
         "|---|" + "---|" * (len(NS) + 2)]
for label, tag in VERIFIERS:
    vals = [sr(tag, n) for n in NS]
    got = [v for v in vals if v is not None]
    mean = f"{sum(got)/len(got):.0f}" if got else "·"
    best = f"{max(got)}" if got else "·"
    cells = " | ".join("·" if v is None else str(v) for v in vals)
    bold = "**" if tag is None else ""
    lines.append(f"| {bold}{label}{bold} | {cells} | {mean} | {best} |")
    rows.append((label, vals, got))

# headline: which checkpoint has the best mean over the fully-scored N (excl. N1 = verifier-irrelevant).
scored = [(lab, [v for n, v in zip(NS, vals) if n != 1 and v is not None])
          for lab, vals, _ in [(r[0], r[1], r[2]) for r in rows]]
ranked = sorted([(lab, sum(vs)/len(vs)) for lab, vs in scored if vs], key=lambda x: -x[1])
lines += ["", "**Ranking by mean success over N≥2 (verifier matters):** "
          + ", ".join(f"{lab} {m:.0f}" for lab, m in ranked) + "."]
print("\n".join(lines))
