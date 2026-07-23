#!/usr/bin/env python3
"""
Aggregate the DETERMINISTIC sub50 eval campaign -> data/eval/det_results.md.
  Section 1: Best-of-N success-rate table (search + BC, native + 500 steps, N=1..64)
             with mean object->goal distance.
  Section 2: episode length for SUCCESS vs TRUNCATED (mean num_steps), per cell.
Cells: data/eval/<task>/<task>_<mk>_st<steps>_n<N>_seed1000_ep50/{eval_log.json, episodes.jsonl}
  mk = search milestone (90/60) or "bc"; steps = 60/120/500.
  conda activate monkey-verifier ; python scriptsv2/analysis/aggregate_det_results.py
"""
import glob, json, os, re, collections
E = "data/eval"; N_GRID = [1, 2, 4, 8, 16, 32, 64]
TASKS = ["eggplant", "stack_cube", "spoon_on_towel"]
# search SR cells: <task>_sub50_<mk>k_st<steps>_n<N>_...  ; BC SR cells: <task>_bc_st<steps>_n<N>_...
# argmax-selection cells carry a trailing "_argmax" suffix (default softmax).
CELL = re.compile(r"^(eggplant|stack_cube|spoon_on_towel)_(?:sub50_)?([0-9]+k?|bc)_st([0-9]+)_n([0-9]+)_seed1000_ep50(_argmax)?$")

# {(task, mk, steps, sel): {N: dict}}
rows = collections.defaultdict(dict)
for task in TASKS:
    for d in sorted(glob.glob(f"{E}/{task}/*_seed1000_ep50") + glob.glob(f"{E}/{task}/*_seed1000_ep50_argmax")):
        m = CELL.match(os.path.basename(d))
        if not m:
            continue
        try:
            j = json.load(open(f"{d}/eval_log.json"))
        except Exception:
            continue
        if int(j.get("num_episodes", 0)) < 50:
            continue
        t, mk, steps, N = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        sel = "argmax" if m.group(5) else "softmax"
        # per-episode success/trunc num_steps
        s_steps, t_steps = [], []
        try:
            for ln in open(f"{d}/episodes.jsonl"):
                e = json.loads(ln)
                (s_steps if e.get("success") else (t_steps if e.get("truncated") else [])).append(int(e.get("num_steps", 0)))
        except Exception:
            pass
        rows[(t, mk, steps, sel)][N] = dict(
            sr=100.0 * j["success_rate"], dist=j.get("mean_obj_goal_dist_min"),
            succ_len=(sum(s_steps) / len(s_steps) if s_steps else None),
            trunc_len=(sum(t_steps) / len(t_steps) if t_steps else None))

def kind(mk):
    return "BC" if mk == "bc" else "search"

L = ["# Deterministic sub50 eval — search vs BC (env-cap fixed, deterministic, tiled)\n",
     "Seeds 1000-1049, softmax BoN, DQC-chunk verifier, object->goal distance logged. "
     "search = {spoon 90k, stack 60k, eggplant 90k}; BC = each task's best n=1 ckpt. "
     "`—` = not done yet.\n",
     "## 1. Best-of-N success rate  (%, mean obj->goal dist in parens at peak N)\n",
     "argmax = deterministic top-Q selection; softmax = temp-1.0 multinomial BoN. "
     "argmax removes the softmax pool-size resampling noise (and the argmax sweep is "
     "pinned to one node to also remove cross-GPU FP variance).\n",
     "| task | policy | sel | steps | " + " | ".join(f"N{n}" for n in N_GRID) + " | peak |",
     "|" + "---|" * (len(N_GRID) + 5)]
for (t, mk, steps, sel) in sorted(rows):
    r = rows[(t, mk, steps, sel)]
    cells = [(f"{r[n]['sr']:.0f}" if n in r else "—") for n in N_GRID]
    pk = max((r[n]["sr"], n) for n in r) if r else (0, 0)
    pkd = r.get(pk[1], {}).get("dist")
    L.append(f"| {t} | {kind(mk)} | {sel} | {steps} | " + " | ".join(cells)
             + f" | {pk[0]:.0f}% @N{pk[1]}" + (f" (d={pkd:.2f})" if pkd else "") + " |")

L += ["\n## 2. Episode length — success vs truncated (mean num_steps)\n",
      "How long successes take vs how long truncations run (= the horizon). Big gap on "
      "truncation means the policy runs out the clock.\n",
      "| task | policy | sel | steps | N | succ_len | trunc_len |",
      "|---|---|---|---|---|---|---|"]
for (t, mk, steps, sel) in sorted(rows):
    for n in (1, 8, 64):
        if n not in rows[(t, mk, steps, sel)]:
            continue
        c = rows[(t, mk, steps, sel)][n]
        sl = f"{c['succ_len']:.0f}" if c["succ_len"] else "—"
        tl = f"{c['trunc_len']:.0f}" if c["trunc_len"] else "—"
        L.append(f"| {t} | {kind(mk)} | {sel} | {steps} | {n} | {sl} | {tl} |")

open(f"{E}/det_results.md", "w").write("\n".join(L) + "\n")
print(f"wrote {E}/det_results.md  ({len(rows)} policy-cells)")
