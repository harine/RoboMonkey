#!/usr/bin/env python3
"""Diagnose the BC-120 seed1001 spoon truncation: grasp attempt at t=12 fails, at t=16 all 4
candidates retreat. Shows WHY BoN can't recover: candidates are near-identical (no re-grasp
proposed) and verifier Q is flat across them. Panels: (A) EE+spoon x-y with candidate arrows
at t=16; (B) time series of EE-spoon dist, gripper, Q-spread, candidate diversity.
  conda activate monkey-verifier; python scriptsv2/analysis/plot_grasp_failure_seed1001.py"""
import numpy as np, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="data/eval/raw_data/stochasticity_plots"; os.makedirs(OUT,exist_ok=True)
p="data/eval/spoon_on_towel/_viz/spoon_on_towel_bc_st120_n4/search_q/ep001_seed1001.npz"
d=np.load(p,allow_pickle=True)
ee=np.asarray(d["ee_xyz"]); sp=np.asarray(d["src_xyz"]); bt=np.asarray(d["branch_t"])
ca=np.asarray(d["candidate_actions"]); v=np.asarray(d["values"]); sel=np.asarray(d["selected_index"])
R=len(ee)
# fit action->world map from executed candidate
X=np.array([ca[i,int(sel[i]),:,:3].sum(0) for i in range(R-1)]); Y=np.diff(ee,axis=0)
M,_,_,_=np.linalg.lstsq(X,Y,rcond=None)
dist=np.linalg.norm(ee-sp,axis=1)
grip=ca[:,:,:,6].mean((1,2))
qspread=v.max(1)-v.min(1)
def div(i):
    mv=ca[i,:,:,:3].sum(1)@M; u=mv/(np.linalg.norm(mv,axis=1,keepdims=True)+1e-9)
    return np.mean([u[a].dot(u[b]) for a in range(len(u)) for b in range(a+1,len(u))])
divs=np.array([div(i) for i in range(R)])

fig=plt.figure(figsize=(16,6.8))
axA=fig.add_subplot(1,2,1)
# EE path colored by time
sc=axA.scatter(ee[:,0],ee[:,1],c=bt,cmap="viridis",s=28,zorder=3)
axA.plot(ee[:,0],ee[:,1],"-",color="gray",lw=1,alpha=.6,zorder=2)
axA.scatter(*sp[0,:2],marker="*",s=320,c="orange",edgecolors="k",zorder=6,label="spoon (never moves)")
# candidate arrows at t=16 (replan 4) in world frame
r=4; base=ee[r,:2]
for k in range(ca.shape[1]):
    mv=(ca[r,k,:,:3].sum(0)@M)[:2]; mv=mv/ (np.linalg.norm(mv)+1e-9)*0.04
    axA.annotate("",xy=base+mv,xytext=base,arrowprops=dict(arrowstyle="->",color="red",lw=2))
axA.annotate("t=16: all 4 candidates\nretreat from spoon",base,fontsize=9,xytext=(12,-28),
             textcoords="offset points",color="red",bbox=dict(boxstyle="round,pad=.2",fc="mistyrose"))
# toward-spoon reference arrow
ts=(sp[r,:2]-ee[r,:2]); ts=ts/np.linalg.norm(ts)*0.04
axA.annotate("",xy=base+ts,xytext=base,arrowprops=dict(arrowstyle="->",color="green",lw=2,ls="--"))
axA.annotate("toward spoon",base+ts,fontsize=8,color="green")
axA.scatter(*ee[3,:2],marker="P",s=140,c="blue",edgecolors="k",zorder=6)
axA.annotate("t=12 grasp attempt\n(grip closes, fails)",ee[3,:2],fontsize=9,xytext=(-120,10),textcoords="offset points",
             color="blue",bbox=dict(boxstyle="round,pad=.2",fc="lightcyan"))
axA.set_title("BC-120 seed1001: gripper EE x-y (color=step)"); axA.set_xlabel("x (m)"); axA.set_ylabel("y (m)")
axA.set_aspect("equal","datalim"); axA.grid(alpha=.3); axA.legend(fontsize=8,loc="lower right")
fig.colorbar(sc,ax=axA,label="sim step",fraction=.045)

axB=fig.add_subplot(1,2,2)
t=bt
axB.plot(t,dist,"-o",ms=3,label="EE–spoon dist (m)",color="black")
axB.plot(t,grip,"-s",ms=3,label="gripper cmd (1=open,0=closed)",color="purple")
axB.plot(t,divs,"-^",ms=3,label="candidate diversity (pairwise cos; 1=identical)",color="teal")
axB.plot(t,qspread/ (np.abs(v.mean(1))+1e-9),"-d",ms=3,label="Q spread / |Q|  (across 4 cands)",color="darkorange")
for x in (12,16): axB.axvline(x,color="red",ls=":",alpha=.7)
axB.text(14,0.02,"t=12 / t=16",color="red",fontsize=8,ha="center")
axB.set_ylim(-0.05,1.05); axB.set_xlabel("sim step"); axB.set_title("Why BoN can't recover: no candidate diversity, flat Q")
axB.grid(alpha=.3); axB.legend(fontsize=8,loc="upper right")
fig.suptitle("spoon_on_towel BC-120 N=4 seed1001 (TRUNCATED): failed grasp, then all candidates retreat; "
             "candidates near-identical (cos≈1) and Q nearly flat → BoN is a no-op",fontsize=10)
pp=f"{OUT}/grasp_failure_seed1001.png"; fig.savefig(pp,dpi=120,bbox_inches="tight"); plt.close(fig)
print("wrote",pp)
