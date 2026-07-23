#!/usr/bin/env python3
"""Why do 'identical' sub50-90k spoon_on_towel rollouts (same seed, N=4) diverge across
max-step budgets? Overlay EE + spoon(object) x-y trajectories for st60/st120/st500, seeds
1003 & 1006. st60/st500 are the SEARCH policy (mk=90); st120 is the BC-BoN policy. Marks the
replan where the two search runs' seeded softmax pick first differs (the divergence point).
  conda activate monkey-verifier
  python scriptsv2/analysis/plot_stochasticity_spoon.py
Out: data/eval/raw_data/stochasticity_plots/spoon_stochasticity_seed<seed>.png"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

E = "data/eval/spoon_on_towel/_viz"
OUT = "data/eval/raw_data/stochasticity_plots"; os.makedirs(OUT, exist_ok=True)
# (step budget) -> (cell, policy-kind, color)
CFG = {
    60:  (f"{E}/spoon_on_towel_90_st60_n4/search_q",  "search", "#1f77b4"),
    120: (f"{E}/spoon_on_towel_bc_st120_n4/search_q", "BC",     "#2ca02c"),
    500: (f"{E}/spoon_on_towel_90_st500_n4/search_q", "search", "#d62728"),
}

def load(st, ep, seed):
    p = f"{CFG[st][0]}/{ep}_seed{seed}.npz"
    if not os.path.exists(p): return None
    d = np.load(p, allow_pickle=True)
    return dict(ee=np.asarray(d["ee_xyz"]), obj=np.asarray(d["src_xyz"]),
                tgt=np.asarray(d["tgt_xyz"]), sel=np.asarray(d["selected_index"]),
                succ=int(d["success"]), trunc=int(d["truncated"]), nsteps=int(d["num_steps"]))

def first_sel_divergence(a, b):
    n = min(len(a["sel"]), len(b["sel"]))
    for i in range(n):
        if a["sel"][i] != b["sel"][i]:
            return i
    return None

for seed, ep in [(1003, "ep003"), (1006, "ep006")]:
    runs = {st: load(st, ep, seed) for st in CFG}
    fig, (axg, axo) = plt.subplots(1, 2, figsize=(15, 7))
    for st, r in runs.items():
        if r is None: continue
        kind, col = CFG[st][1], CFG[st][2]
        ls = "-" if r["succ"] else "--"
        lab = f"st{st} [{kind}] {'SUCC@'+str(r['nsteps']) if r['succ'] else 'TRUNC@'+str(r['nsteps'])}"
        for ax, key in ((axg, "ee"), (axo, "obj")):
            xy = r[key]
            ax.plot(xy[:, 0], xy[:, 1], ls, color=col, lw=2.2, label=(lab if ax is axg else None), zorder=3)
            ax.scatter(*xy[0, :2], marker="o", s=55, facecolors="white", edgecolors=col, lw=2, zorder=5)
            ax.scatter(*xy[-1, :2], marker=("*" if r["succ"] else "X"),
                       s=200 if r["succ"] else 90, color=col, edgecolors="k", lw=0.6, zorder=6)
    # object start + target on the object panel
    r0 = runs[60] or runs[500]
    axo.scatter(*r0["obj"][0, :2], marker="o", s=120, facecolors="none", edgecolors="green", lw=2.5, zorder=7, label="spoon start")
    axo.scatter(*r0["tgt"][0, :2], marker="P", s=170, color="purple", edgecolors="k", zorder=7, label="towel target")
    # mark the seeded-softmax divergence replan on the two SEARCH runs (60 vs 500)
    div = first_sel_divergence(runs[60], runs[500]) if (runs[60] and runs[500]) else None
    if div is not None:
        for st in (60, 500):
            xy = runs[st]["ee"]
            if div < len(xy):
                axg.scatter(*xy[div, :2], marker="D", s=110, facecolors="yellow", edgecolors="k", lw=1.3, zorder=8)
                axg.annotate(f"pick flips\n@replan {div}", xy[div, :2], fontsize=8,
                             xytext=(8, 8), textcoords="offset points",
                             bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.7))
    for ax, ttl in ((axg, "gripper EE  x-y"), (axo, "spoon (object)  x-y")):
        ax.set_title(ttl); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_aspect("equal", "datalim"); ax.grid(alpha=.3)
    h, l = axg.get_legend_handles_labels()
    h += [Line2D([0], [0], marker="o", color="gray", lw=0, mfc="white", label="start"),
          Line2D([0], [0], marker="*", color="gray", lw=0, ms=13, label="success end"),
          Line2D([0], [0], marker="X", color="gray", lw=0, label="truncated end"),
          Line2D([0], [0], marker="D", color="k", lw=0, mfc="yellow", label="search pick diverges")]
    axg.legend(handles=h, fontsize=8, loc="best")
    axo.legend(fontsize=8, loc="best")
    fig.suptitle(f"spoon_on_towel  sub50-90k  N=4  seed {seed}   — same seed, 3 step budgets\n"
                 f"st60/st500 = SEARCH (same policy!), st120 = BC-BoN;  solid=success, dashed=truncated",
                 fontsize=11)
    p = f"{OUT}/spoon_stochasticity_seed{seed}.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {p}  (search pick first diverges @ replan {div})")
