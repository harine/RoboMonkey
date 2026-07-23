#!/usr/bin/env python3
"""Q over the course of the episode for spoon SEARCH-500, seeds 1001-1003, faceted by N.
Per replan: the CHOSEN candidate's Q (line) + a shaded min-max band across the N candidates.
Shows the verifier's level and how tight/flat the candidate spread is over the rollout.
  conda activate monkey-verifier; python scriptsv2/analysis/plot_q_over_episode_spoon.py
Out: data/eval/raw_data/stochasticity_plots/q_over_episode_spoon_seed<seed>.png"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="data/eval/raw_data/stochasticity_plots"; os.makedirs(OUT, exist_ok=True)
BASE="data/eval/spoon_on_towel/_viz/spoon_on_towel_90_st500_n{N}/search_q/ep{ep}_seed{seed}.npz"
NS=[1,2,4,8,16,32,64]
for seed in (1001,1002,1003):
    ep=f"{seed-1000:03d}"
    fig,axes=plt.subplots(len(NS),1,figsize=(13,2.05*len(NS)),sharex=True)
    for ax,N in zip(axes,NS):
        p=BASE.format(N=N,ep=ep,seed=seed)
        d=np.load(p,allow_pickle=True)
        v=np.asarray(d["values"],dtype=float)               # (R, N)
        bt=np.asarray(d["branch_t"]); sel=np.asarray(d["selected_index"])
        R=len(v)
        qsel=np.array([v[i,sel[i]] if 0<=sel[i]<v.shape[1] else np.nan for i in range(R)])
        qmin=v.min(1); qmax=v.max(1)
        ax.fill_between(bt,qmin,qmax,color="#d62728",alpha=.18,label="candidate min–max")
        ax.plot(bt,qsel,"-",color="#d62728",lw=1.6,label="chosen Q")
        succ=int(d["success"]); ns=int(d["num_steps"])
        tag=f"{'SUCC' if succ else 'TRUNC'}@{ns}"
        if succ: ax.scatter(bt[-1],qsel[-1],marker="*",s=140,color="k",zorder=5)
        ax.set_ylabel(f"N={N}\nQ",fontsize=8)
        ax.grid(alpha=.3)
        ax.text(0.995,0.90,f"{tag}  meanSpread={np.mean(qmax-qmin):.2f}",transform=ax.transAxes,
                ha="right",va="top",fontsize=8,bbox=dict(boxstyle="round,pad=.2",fc="white",alpha=.7))
        if N==NS[0]: ax.legend(fontsize=8,loc="upper left")
    axes[-1].set_xlabel("sim step (branch_t)")
    fig.suptitle(f"spoon SEARCH-500  seed {seed}: chosen Q + candidate min–max band over the episode (faceted by N)\n"
                 f"★ = success end;  band width = verifier spread across candidates",fontsize=11)
    fig.tight_layout(rect=[0,0,1,0.98])
    pp=f"{OUT}/q_over_episode_spoon_seed{seed}.png"
    fig.savefig(pp,dpi=120,bbox_inches="tight"); plt.close(fig); print("wrote",pp)
