#!/usr/bin/env python3
"""
Per-seed trajectory plots from the SEEDED video-render dumps (data/eval/_seeded_videos/).
For each (task, sub, seed): ONE figure with two panels — gripper (EE) x-y and object x-y —
overlaying the executed/chosen trajectory for ALL N (search n1/8/64 + bc n1/8/64), each line
COLORED BY the chosen-candidate verifier Q (high Q = red), SOLID = success / DASHED = truncated.
Lets you see how trajectories diverge across N and search-vs-bc, and why some fail.

  conda activate monkey-verifier
  python scriptsv2/analysis/plot_seeded_trajectories.py           # all task/sub/seed
Out: data/eval/trajectory_plots/<task>_<sub>_seed<seed>.png
"""
import glob, os, re, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import collections

ROOT = "data/eval/_seeded_videos"
OUT = "data/eval/trajectory_plots"; os.makedirs(OUT, exist_ok=True)
CMAP = "coolwarm"                      # low Q = blue, high Q = red
# dirs now carry a step tag: <kind>_<task>_<sub>_<ck>_st<steps>_n<N>. Plot the 500-step
# rollouts (STEP_PLOT); set STEP_PLOT=120 to plot the short-horizon ones instead.
STEP_PLOT = os.environ.get("STEP_PLOT", "500")
CELL_RE = re.compile(rf"^(search|bc)_(.+)_(sub\d+)_([0-9]+k)_st{STEP_PLOT}_n(\d+)$")

# gather: {(task,sub,seed): [ {kind,N,ee,obj,tgt,q,success,trunc}, ... ]}
groups = collections.defaultdict(list)
for cell in sorted(glob.glob(f"{ROOT}/*")):
    m = CELL_RE.match(os.path.basename(cell))
    if not m:
        continue
    kind, task, sub, ck, N = m.group(1), m.group(2), m.group(3), m.group(4), int(m.group(5))
    for npz in sorted(glob.glob(f"{cell}/search_q/ep*_seed*.npz")):
        try:
            d = np.load(npz, allow_pickle=True)
        except Exception:
            continue
        if "src_xyz" not in d.files:           # old dump w/o object logging -> skip
            continue
        ee, obj, tgt = d["ee_xyz"], d["src_xyz"], d["tgt_xyz"]
        vals, sel = d["values"], d["selected_index"]
        q = np.array([vals[i, sel[i]] if 0 <= sel[i] < vals.shape[1] else np.nan
                      for i in range(len(sel))], dtype=float)
        groups[(task, sub, int(d["seed"]))].append(dict(
            kind=kind, N=N, ee=ee, obj=obj, tgt=tgt, q=q,
            success=int(d["success"]), trunc=int(d["truncated"]), steps=int(d["num_steps"])))

def qcolored(ax, xy, q, norm, ls, lw):
    pts = xy[:, :2].reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=CMAP, norm=norm, linestyles=ls, linewidths=lw)
    lc.set_array(q[:-1]); ax.add_collection(lc); return lc

made = 0
for (task, sub, seed), runs in sorted(groups.items()):
    runs.sort(key=lambda r: (r["kind"], r["N"]))
    qall = np.concatenate([r["q"][~np.isnan(r["q"])] for r in runs if np.isfinite(r["q"]).any()]) if runs else np.array([0, 1])
    norm = Normalize(float(np.min(qall)), float(np.max(qall)))
    fig, (axg, axo) = plt.subplots(1, 2, figsize=(15, 6.5))
    for r in runs:
        ls = "-" if r["success"] else "--"                 # solid=success, dashed=trunc
        lw = 1.0 + 0.9 * np.log2(max(r["N"], 1))           # thicker = larger N
        for ax, key in ((axg, "ee"), (axo, "obj")):
            xy = r[key]
            if len(xy) < 2:
                continue
            qcolored(ax, xy, r["q"], norm, ls, lw)
            end = xy[-1]
            ax.scatter(*end[:2], marker=("*" if r["success"] else "x"),
                       s=90 if r["success"] else 60, c="k", zorder=5)
            ax.annotate(f"{r['kind'][0]}-n{r['N']}", end[:2], fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
    # object start + target markers on the object panel
    if runs:
        o0 = runs[0]["obj"][0]; tg = runs[0]["tgt"][0]
        axo.scatter(*o0[:2], marker="o", s=70, facecolors="none", edgecolors="green", zorder=6, label="obj start")
        axo.scatter(*tg[:2], marker="P", s=90, c="purple", zorder=6, label="target")
        axo.legend(fontsize=8, loc="best")
    for ax, ttl in ((axg, "gripper (EE) x-y"), (axo, "object x-y")):
        ax.set_title(ttl); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.autoscale(); ax.set_aspect("equal", "datalim"); ax.grid(alpha=.3)
    sm = ScalarMappable(norm=norm, cmap=CMAP); sm.set_array([])
    fig.colorbar(sm, ax=[axg, axo], label="chosen-candidate Q (red = high)", fraction=0.04)
    ns = ", ".join(f"{r['kind'][0]}n{r['N']}={'succ' if r['success'] else 'trunc'}" for r in runs)
    fig.suptitle(f"{task} {sub}  seed {seed}   (solid=success, dashed=trunc; thicker=larger N)\n{ns}", fontsize=11)
    # legend for line style
    axg.legend(handles=[Line2D([0], [0], color="gray", ls="-", label="success"),
                        Line2D([0], [0], color="gray", ls="--", label="truncated")], fontsize=8, loc="best")
    p = f"{OUT}/{task}_{sub}_seed{seed}.png"
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig); made += 1

print(f"wrote {made} trajectory figures -> {OUT}/  (groups found: {len(groups)})")
