#!/usr/bin/env python3
"""SEARCH-500 spoon: Q-spread(t) across the 4 candidates + EE-spoon distance(t) for two
truncated seeds (high spread) vs one successful seed (low spread). Shows spread spikes
coincide with off-task wandering / failed-grasp hovering, NOT with useful discrimination.
  conda activate monkey-verifier; python scriptsv2/analysis/plot_search_qspread_time.py"""
import numpy as np, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="data/eval/raw_data/stochasticity_plots"; os.makedirs(OUT,exist_ok=True)
D="data/eval/spoon_on_towel/_viz/spoon_on_towel_90_st500_n4/search_q"
seeds=[(1004,"TRUNC (mean spread 3.6)","#d62728"),(1001,"TRUNC (mean spread 2.4)","#ff7f0e"),(1007,"SUCC (mean spread 0.3)","#1f77b4")]
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(13,8),sharex=True)
for seed,lab,col in seeds:
    d=np.load(f"{D}/ep{seed-1000:03d}_seed{seed}.npz",allow_pickle=True)
    v=np.asarray(d["values"]); ee=np.asarray(d["ee_xyz"]); sp=np.asarray(d["src_xyz"]); bt=np.asarray(d["branch_t"])
    spread=v.max(1)-v.min(1); dist=np.linalg.norm(ee-sp,axis=1)
    succ=int(d["success"]); ls="-" if not succ else "-"
    ax1.plot(bt,spread,ls,color=col,lw=1.8,label=f"seed{seed} {lab}")
    if succ: ax1.scatter(bt[-1],spread[-1],marker="*",s=180,color=col,edgecolors="k",zorder=5)
    ax2.plot(bt,dist,ls,color=col,lw=1.8,label=f"seed{seed}")
    if succ: ax2.scatter(bt[-1],dist[-1],marker="*",s=180,color=col,edgecolors="k",zorder=5)
ax1.set_ylabel("Q spread across 4 candidates\n(max - min)"); ax1.grid(alpha=.3); ax1.legend(fontsize=9)
ax1.set_title("SEARCH-500 spoon: failing trajectories carry HIGH Q spread; the success stays tight (★=success end @ step 27)")
ax2.axhline(0.05,color="green",ls=":",alpha=.6); ax2.text(5,0.055,"~grasp range",color="green",fontsize=8)
ax2.set_ylabel("EE–spoon distance (m)"); ax2.set_xlabel("sim step"); ax2.grid(alpha=.3); ax2.legend(fontsize=9)
ax2.set_title("Spread spikes line up with the arm being FAR from the spoon (wandering) or hovering over a failed grasp")
fig.tight_layout()
p=f"{OUT}/search_qspread_time.png"; fig.savefig(p,dpi=120,bbox_inches="tight"); plt.close(fig)
print("wrote",p)
