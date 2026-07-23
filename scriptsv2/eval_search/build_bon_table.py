#!/usr/bin/env python
"""Emit the BoN sim-eval success-rate table (markdown) comparing the finetuned
DQC verifiers (v2_all@30k, v3_dist@30k) against the june19 base verifier, per
policy, over N. Reads eval_log.json cells; '·' = not finished (>=50 ep). Pure
stdlib so it runs anywhere; re-run to refresh as cells complete.
"""
import json
from pathlib import Path

EVAL = Path("/mmfs1/home/harine/RoboMonkey/data/eval")
NS = [1, 2, 4, 8, 16, 32, 64]
# policy -> (sub, base_prefix, finetuned_prefix_template{vt})
POLICIES = [
    ("diffusion-search 10k", "search_dqc_chunk",
     "diff10k_chunk8_predh8_roll4_softmax", "diff10k_{vt}_chunk8_predh8_roll4_softmax"),
    ("diffusion-search 10k FILTERED", "search_dqc_chunk",
     "difffilt10k_chunk8_predh8_roll4_softmax", "difffilt10k_{vt}_chunk8_predh8_roll4_softmax"),
    ("diffusion-search 33k", "search_dqc_chunk",
     "diffusion_chunk8_predh8_roll4_softmax", "diff33k_{vt}_chunk8_predh8_roll4_softmax"),
    ("gaussian-search 20k", "search_dqc_chunk",
     "gauss20k_chunk8_predh8_roll4_softmax", "gauss20k_{vt}_chunk8_predh8_roll4_softmax"),
    ("BC-H8 (ddpm)", "bc_dqc_chunk",
     "bcH8_chunk8_predh8_roll4_softmax_ddpm", "bcH8_{vt}_chunk8_predh8_roll4_softmax_ddpm"),
]
VERIFIERS = [("base", None), ("v2_all@30k", "v2all30k"), ("v3_dist@30k", "v3dist30k"),
             ("v4_filt_ramp@50k", "v4filt50k")]


def sr(sub, prefix, n):
    p = EVAL / sub / f"{prefix}_n{n}_seed1000_ep50" / "eval_log.json"
    try:
        d = json.load(open(p))
        if int(d.get("num_episodes", 0)) >= 50:
            return int(round(100 * float(d["success_rate"])))
    except Exception:
        pass
    return None


def fmt(v):
    return "·" if v is None else str(v)


lines = ["| policy | verifier | " + " | ".join(f"N{n}" for n in NS) + " |",
         "|" + "---|" * (len(NS) + 2)]
for label, sub, base_pfx, ft_tmpl in POLICIES:
    for i, (vlabel, vt) in enumerate(VERIFIERS):
        pfx = base_pfx if vt is None else ft_tmpl.format(vt=vt)
        cells = " | ".join(fmt(sr(sub, pfx, n)) for n in NS)
        pol_cell = label if i == 0 else ""
        bold = "**" if vt is None else ""
        lines.append(f"| {pol_cell} | {bold}{vlabel}{bold} | " + cells + " |")
    lines.append("| | | " + " | ".join("" for _ in NS) + " |")  # spacer
print("\n".join(lines))
