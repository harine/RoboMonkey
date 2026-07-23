#!/usr/bin/env python3
"""stack_cube n=32: st60 vs st500 are the SAME search policy + same node (g3102), so replan-0
is bit-identical. Yet some seeds diverge. Overlay EE + cube(object) x-y for st60 (blue) vs
st500 (red); mark the replan where the seeded softmax pick first differs. Shows the divergence
originates from sim ULP noise amplified by a near-flat verifier + temp=1 softmax.
  conda activate monkey-verifier; python scriptsv2/analysis/plot_stochasticity_stack.py
Out: data/eval/raw_data/stochasticity_plots/stack_n32_seed<seed>.png"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
E="data/eval/stack_cube/_viz"
OUT="data/eval/raw_data/stochasticity_plots"; os.makedirs(OUT,exist_ok=True)
A=f"{E}/stack_cube_60_st60_n32/search_q"; B=f"{E}/stack_cube_60_st500_n32/search_q"
def ld(d,ep,seed):
    p=f"{d}/{ep}_seed{seed}.npz"
    if not os.path.exists(p): return None
    x=np.load(p,allow_pickle=True)
    return dict(ee=np.asarray(x["ee_xyz"]),obj=np.asarray(x["src_xyz"]),tgt=np.asarray(x["tgt_xyz"]),
                sel=np.asarray(x["selected_index"]),succ=int(x["success"]),trunc=int(x["truncated"]),ns=int(x["num_steps"]))
for seed in [1000,1002,1003,1005]:
    ep=f"ep{seed-1000:03d}"; a=ld(A,ep,seed); b=ld(B,ep,seed)
    if a is None or b is None: continue
    n=min(len(a["sel"]),len(b["sel"]))
    div=next((i for i in range(n) if a["sel"][i]!=b["sel"][i]),None)
    fig,(axg,axo)=plt.subplots(1,2,figsize=(15,7))
    for r,col,tag in [(a,"#1f77b4","st60"),(b,"#d62728","st500")]:
        ls="-" if r["succ"] else "--"
        lab=f"{tag} {'SUCC@'+str(r['ns']) if r['succ'] else 'TRUNC@'+str(r['ns'])}"
        for ax,key in ((axg,"ee"),(axo,"obj")):
            xy=r[key]
            ax.plot(xy[:,0],xy[:,1],ls,color=col,lw=2.2,label=(lab if ax is axg else None),zorder=3)
            ax.scatter(*xy[0,:2],marker="o",s=55,facecolors="white",edgecolors=col,lw=2,zorder=5)
            ax.scatter(*xy[-1,:2],marker=("*" if r["succ"] else "X"),s=200 if r["succ"] else 90,color=col,edgecolors="k",lw=.6,zorder=6)
    axo.scatter(*a["obj"][0,:2],marker="o",s=120,facecolors="none",edgecolors="green",lw=2.5,zorder=7,label="cube start")
    axo.scatter(*a["tgt"][0,:2],marker="P",s=170,color="purple",edgecolors="k",zorder=7,label="target block")
    if div is not None:
        for r in (a,b):
            if div<len(r["ee"]):
                axg.scatter(*r["ee"][div,:2],marker="D",s=110,facecolors="yellow",edgecolors="k",lw=1.3,zorder=8)
        axg.annotate(f"softmax pick flips @replan {div}",a["ee"][min(div,len(a['ee'])-1),:2],fontsize=8,
                     xytext=(8,8),textcoords="offset points",bbox=dict(boxstyle="round,pad=0.2",fc="yellow",alpha=.7))
    title_div = f"pick flips @replan {div}" if div is not None else "picks identical over overlap (state drift only)"
    for ax,ttl in ((axg,"gripper EE  x-y"),(axo,"cube (object)  x-y")):
        ax.set_title(ttl); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal","datalim"); ax.grid(alpha=.3)
    h,l=axg.get_legend_handles_labels()
    h+=[Line2D([0],[0],marker="D",color="k",lw=0,mfc="yellow",label="pick diverges")]
    axg.legend(handles=h,fontsize=8); axo.legend(fontsize=8)
    fig.suptitle(f"stack_cube  sub50-60k  N=32  seed {seed}   st60 vs st500 (SAME policy, SAME node g3102)\n"
                 f"replan-0 bit-identical; {title_div};  solid=success dashed=trunc",fontsize=11)
    p=f"{OUT}/stack_n32_seed{seed}.png"; fig.savefig(p,dpi=120,bbox_inches="tight"); plt.close(fig)
    print(f"wrote {p}  (pick diverges @ replan {div})")
