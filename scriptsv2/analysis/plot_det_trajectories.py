#!/usr/bin/env python3
"""EE + object trajectory plots for the deterministic campaign: ALL N (search + bc) for
ONE seed, per (task, steps) -> see how the paths diverge. Q-colored (red=high), solid=succ
dashed=trunc, thicker=larger N. Reads data/eval/<task>/_viz/<cell>/search_q/*.npz.
  TRAJ_SEED=1000 python scriptsv2/analysis/plot_det_trajectories.py
Out: data/eval/det_trajectory_plots/<task>_st<steps>_seed<seed>.png"""
import glob, os, re, collections, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
E = "data/eval"; OUT = f"{E}/raw_data/det_trajectory_plots"; os.makedirs(OUT, exist_ok=True)
SEED = int(os.environ.get("TRAJ_SEED", "1000")); CMAP = "coolwarm"
CELL = re.compile(r"^(eggplant|stack_cube|spoon_on_towel)_([0-9]+|bc)_st([0-9]+)_n([0-9]+)$")
groups = collections.defaultdict(list)
for cell in glob.glob(f"{E}/*/_viz/*_st*_n*"):
    m = CELL.match(os.path.basename(cell))
    if not m:
        continue
    task, mk, steps, N = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
    fs = glob.glob(f"{cell}/search_q/*seed{SEED}*.npz")
    if not fs:
        continue
    try:
        d = np.load(fs[0], allow_pickle=True)
    except Exception:
        continue
    if "src_xyz" not in d.files:
        continue
    vals, sel = d["values"], d["selected_index"]
    q = np.array([vals[i, sel[i]] if 0 <= sel[i] < vals.shape[1] else np.nan for i in range(len(sel))])
    groups[(task, steps)].append(dict(N=N, kind=("bc" if mk == "bc" else "search"),
        ee=d["ee_xyz"], obj=d["src_xyz"], tgt=d["tgt_xyz"], q=q,
        succ=int(d["success"]), trunc=int(d["truncated"])))

def qcol(ax, xy, q, norm, ls, lw):
    p = xy[:, :2].reshape(-1, 1, 2); s = np.concatenate([p[:-1], p[1:]], 1)
    lc = LineCollection(s, cmap=CMAP, norm=norm, linestyles=ls, linewidths=lw)
    lc.set_array(q[:-1]); ax.add_collection(lc)

made = 0
for (task, steps), runs in sorted(groups.items()):
    runs.sort(key=lambda r: (r["kind"], r["N"]))
    qa = np.concatenate([r["q"][~np.isnan(r["q"])] for r in runs if np.isfinite(r["q"]).any()]) if runs else np.array([0, 1])
    norm = Normalize(float(qa.min()), float(qa.max()))
    fig, (axg, axo) = plt.subplots(1, 2, figsize=(15, 6.5))
    for r in runs:
        ls = "-" if r["succ"] else "--"; lw = 1.0 + 0.8 * np.log2(max(r["N"], 1))
        for ax, key in ((axg, "ee"), (axo, "obj")):
            xy = r[key]
            if len(xy) < 2:
                continue
            qcol(ax, xy, r["q"], norm, ls, lw)
            ax.annotate(f"{r['kind'][0]}n{r['N']}", xy[-1, :2], fontsize=7, xytext=(3, 3), textcoords="offset points")
            ax.scatter(*xy[-1, :2], marker=("*" if r["succ"] else "x"), s=80, c="k", zorder=5)
    if runs:
        o0 = runs[0]["obj"][0]; tg = runs[0]["tgt"][0]
        axo.scatter(*o0[:2], marker="o", s=70, facecolors="none", edgecolors="green", zorder=6, label="obj start")
        axo.scatter(*tg[:2], marker="P", s=90, c="purple", zorder=6, label="target"); axo.legend(fontsize=8)
    for ax, ttl in ((axg, "gripper EE x-y"), (axo, "object x-y")):
        ax.set_title(ttl); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal", "datalim"); ax.grid(alpha=.3)
    sm = ScalarMappable(norm=norm, cmap=CMAP); sm.set_array([])
    fig.colorbar(sm, ax=[axg, axo], label="chosen Q (red=high)", fraction=.04)
    fig.suptitle(f"{task} st{steps} seed{SEED} — all N (search+bc); solid=succ dashed=trunc, thicker=larger N", fontsize=11)
    axg.legend(handles=[Line2D([0], [0], color="gray", ls="-", label="succ"),
                        Line2D([0], [0], color="gray", ls="--", label="trunc")], fontsize=8)
    fig.savefig(f"{OUT}/{task}_st{steps}_seed{SEED}.png", dpi=110, bbox_inches="tight"); plt.close(fig); made += 1
print(f"wrote {made} trajectory figs -> {OUT} (groups {len(groups)})")
