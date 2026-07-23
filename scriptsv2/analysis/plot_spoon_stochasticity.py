#!/usr/bin/env python3
"""Spoon-on-towel determinism probe: for seeds 1003 & 1006, N=4, overlay the three
viz runs -- search st60 (native), search st500, BC st120 -- and show (a) object x-y
path, (b) chosen-Q per step, (c) object->goal distance per step, (d) the st60-vs-st500
SHARED-PREFIX end-effector divergence (log-y). The point: st60 and st500 are the SAME
search policy + same seed, differing only in max_steps; they still diverge because
residual GPU FP noise (cuDNN convs + JAX reductions) is amplified by contact dynamics
until the selected BoN candidate flips. BC st120 is a DIFFERENT policy (not comparable
as a "horizon" -- there is no st120 *search* cell for spoon).
  conda activate monkey-verifier ; python scriptsv2/analysis/plot_spoon_stochasticity.py
Out: data/eval/raw_data/spoon_stochasticity/spoon_stoch_seed{1003,1006}.png
"""
import glob, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

E = "data/eval/spoon_on_towel/_viz"
OUT = "data/eval/raw_data/spoon_stochasticity"; os.makedirs(OUT, exist_ok=True)
SEEDS = [1003, 1006]
# label -> (cell, color)
RUNS = [
    ("search st60",  "spoon_on_towel_90_st60_n4",  "#e76f51"),
    ("search st500", "spoon_on_towel_90_st500_n4", "#2a9d8f"),
    ("BC st120",     "spoon_on_towel_bc_st120_n4", "#5b6cff"),
]

def load(cell, seed):
    fs = glob.glob(f"{E}/{cell}/search_q/*seed{seed}*.npz")
    return np.load(fs[0], allow_pickle=True) if fs else None

def qsel(d):
    v, s = d["values"], d["selected_index"]
    return np.array([v[i, s[i]] if 0 <= s[i] < v.shape[1] else np.nan for i in range(len(s))])

def objdist(d):
    return np.linalg.norm(d["src_xyz"][:, :3] - d["tgt_xyz"][:, :3], axis=1)

for seed in SEEDS:
    ds = [(lbl, load(cell, seed), c) for lbl, cell, c in RUNS]
    ds = [(lbl, d, c) for lbl, d, c in ds if d is not None]
    fig, ax = plt.subplots(1, 4, figsize=(22, 5.2))

    # (a) object x-y path
    for lbl, d, c in ds:
        obj = d["src_xyz"]; ls = "-" if int(d["success"]) else "--"
        ax[0].plot(obj[:, 0], obj[:, 1], ls, color=c, lw=1.8,
                   label=f"{lbl} ({'succ' if int(d['success']) else 'trunc'}, {len(obj)} st)")
        ax[0].scatter(obj[0, 0], obj[0, 1], marker="o", s=60, facecolors="none", edgecolors=c, zorder=5)
        ax[0].scatter(obj[-1, 0], obj[-1, 1], marker=("*" if int(d["success"]) else "x"), s=120, c=c, zorder=6)
    tg = ds[0][1]["tgt_xyz"][0]
    ax[0].scatter(tg[0], tg[1], marker="P", s=160, c="purple", zorder=7, label="towel (target)")
    ax[0].set_title(f"seed {seed}: object (spoon) x-y path"); ax[0].set_xlabel("x"); ax[0].set_ylabel("y")
    ax[0].set_aspect("equal", "datalim"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=7)

    # (b) chosen Q per step
    for lbl, d, c in ds:
        q = qsel(d); ls = "-" if int(d["success"]) else "--"
        ax[1].plot(np.arange(len(q)), q, ls, color=c, lw=1.6, label=lbl)
    ax[1].set_title("chosen candidate Q vs policy step"); ax[1].set_xlabel("policy (replan) step")
    ax[1].set_ylabel("Q(selected)"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)

    # (c) object->goal distance per step
    for lbl, d, c in ds:
        dist = objdist(d); ls = "-" if int(d["success"]) else "--"
        ax[2].plot(np.arange(len(dist)), dist, ls, color=c, lw=1.6, label=lbl)
        mi = int(dist.argmin()); ax[2].scatter(mi, dist[mi], marker="v", s=55, c=c, zorder=5)
    ax[2].set_title("object -> goal distance vs step  (v = min)"); ax[2].set_xlabel("policy (replan) step")
    ax[2].set_ylabel("||spoon - towel||"); ax[2].grid(alpha=.3); ax[2].legend(fontsize=8)

    # (d) st60-vs-st500 shared-prefix EE divergence (log-y) -- the amplification curve
    a = load("spoon_on_towel_90_st60_n4", seed); b = load("spoon_on_towel_90_st500_n4", seed)
    if a is not None and b is not None:
        K = min(len(a["ee_xyz"]), len(b["ee_xyz"]))
        dee = np.linalg.norm(a["ee_xyz"][:K, :3] - b["ee_xyz"][:K, :3], axis=1) * 1000
        dob = np.linalg.norm(a["src_xyz"][:K, :3] - b["src_xyz"][:K, :3], axis=1) * 1000
        sa, sb = a["selected_index"][:K], b["selected_index"][:K]
        flip = next((t for t in range(K) if int(sa[t]) != int(sb[t])), None)
        ax[3].semilogy(np.arange(K), np.maximum(dee, 1e-3), "-o", ms=3, color="#e76f51", label="Δ end-effector")
        ax[3].semilogy(np.arange(K), np.maximum(dob, 1e-3), "-s", ms=3, color="#264653", label="Δ object")
        if flip is not None:
            ax[3].axvline(flip, color="k", ls=":", lw=1.5, label=f"BoN pick flips @{flip}")
        ax[3].set_title("st60 vs st500 shared-prefix divergence\n(same seed+policy, differ only in max_steps)")
        ax[3].set_xlabel("policy (replan) step"); ax[3].set_ylabel("|Δ| (mm, log)")
        ax[3].grid(alpha=.3, which="both"); ax[3].legend(fontsize=8)

    fig.suptitle(f"Spoon-on-towel N=4 seed {seed}: search st60 vs st500 (same policy) vs BC st120 (different policy). "
                 f"Residual FP noise amplified by contact dynamics -> divergent outcomes.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = f"{OUT}/spoon_stoch_seed{seed}.png"; fig.savefig(p, dpi=115, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {p}")
