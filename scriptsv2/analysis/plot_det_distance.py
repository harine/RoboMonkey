#!/usr/bin/env python3
"""Object->goal distance: BC-fail vs search. Per task, box the per-episode min obj->goal
distance for search-success / search-fail / BC-fail (does search get the object closer,
even when it doesn't fully succeed?). Reads SR cells' episodes.jsonl.
  python scriptsv2/analysis/plot_det_distance.py   (DIST_N=8 to pick the N)
Out: data/eval/det_distance_plots/dist_search_vs_bcfail.png"""
import glob, json, os, re, collections, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
E = "data/eval"; OUT = f"{E}/raw_data/det_distance_plots"; os.makedirs(OUT, exist_ok=True)
NPICK = int(os.environ.get("DIST_N", "8"))
CELL = re.compile(r"^(eggplant|stack_cube|spoon_on_towel)_(?:sub50_)?([0-9]+k?|bc)_st([0-9]+)_n([0-9]+)_seed1000_ep50$")
data = collections.defaultdict(lambda: collections.defaultdict(list))
for d in glob.glob(f"{E}/*/*_seed1000_ep50"):
    m = CELL.match(os.path.basename(d))
    if not m:
        continue
    task, mk, steps, N = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
    if N != NPICK:
        continue
    kind = "bc" if mk == "bc" else "search"
    try:
        for ln in open(f"{d}/episodes.jsonl"):
            e = json.loads(ln); dist = e.get("obj_goal_dist_min")
            if dist is None:
                continue
            data[task][f"{kind}_{'succ' if e.get('success') else 'fail'}"].append(float(dist))
    except Exception:
        pass
tasks = [t for t in ("eggplant", "stack_cube", "spoon_on_towel") if data.get(t)]
if not tasks:
    print("no distance data yet"); raise SystemExit
cats = ["search_succ", "search_fail", "bc_fail"]; cols = ["#2a9d8f", "#e76f51", "#7b7bff"]
fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 5), squeeze=False)
for ax, task in zip(axes[0], tasks):
    vals = [data[task].get(c, []) for c in cats]
    bp = ax.boxplot([v if v else [np.nan] for v in vals], labels=["srch\nsucc", "srch\nfail", "bc\nfail"],
                    patch_artist=True, showmeans=True)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(.6)
    ax.set_title(f"{task}  (N={NPICK})"); ax.set_ylabel("min object→goal dist"); ax.grid(alpha=.3, axis="y")
    yt = ax.get_ylim()[1]
    for i, v in enumerate(vals):
        ax.text(i + 1, yt * 0.96, f"n={len(v)}", ha="center", fontsize=8, color="#444")
fig.suptitle(f"Object→goal distance (min over episode): search-succ vs search-fail vs BC-fail  [N={NPICK}]", fontsize=12)
p = f"{OUT}/dist_search_vs_bcfail.png"; fig.savefig(p, dpi=110, bbox_inches="tight")
print(f"wrote {p}")
