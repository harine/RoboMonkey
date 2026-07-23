#!/usr/bin/env python3
"""Stack-cube BoN N=32 vs N=64, seed 1000 (viz, non-tiled): why N=64 fails to grasp when
N=32 succeeds. N=64's candidate pool shares N=32's first 32 draws (nested by seeding), but
with SOFTMAX selection a larger pool resamples a different action even when the shared
candidates + their Q are identical (step 0: Δcand=3e-7, yet different pick). That
perturbation compounds (contact dynamics) and DESYNCS THE GRASP: N32 closes the gripper at
step 4 and grasps; N64 keeps it open, closes ~3 steps late, misses the cube, lifts empty.
  conda activate monkey-verifier ; python scriptsv2/analysis/plot_stack_bon_grasp.py
Out: data/eval/raw_data/stack_bon_grasp/stack_seed1000_n32_vs_n64.png
"""
import glob, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = "data/eval/raw_data/stack_bon_grasp"; os.makedirs(OUT, exist_ok=True)
def load(cell):
    fs = glob.glob(f"data/eval/stack_cube/_viz/{cell}/search_q/*seed1000*.npz")
    return np.load(fs[0], allow_pickle=True)
A = load("stack_cube_60_st60_n32")   # N=32 -> SUCCESS
B = load("stack_cube_60_st60_n64")   # N=64 -> FAIL

def chosen_grip(d):   # mean gripper (dim6) of chosen 4-substep chunk, per step
    ca, s = d["candidate_actions"], d["selected_index"]
    return np.array([ca[t, s[t], :, 6].mean() if 0 <= s[t] < ca.shape[1] else np.nan for t in range(len(s))])
def chosen_q(d):
    v, s = d["values"], d["selected_index"]
    return np.array([v[t, s[t]] if 0 <= s[t] < v.shape[1] else np.nan for t in range(len(s))])

gA, gB = chosen_grip(A), chosen_grip(B)
qA, qB = chosen_q(A), chosen_q(B)
tA, tB = np.arange(len(gA)), np.arange(len(gB))
def grasp_step(g):    # first step gripper closes below 0.5
    idx = np.where(g < 0.5)[0]; return int(idx[0]) if len(idx) else None
grA, grB = grasp_step(gA), grasp_step(gB)

fig, ax = plt.subplots(1, 4, figsize=(22, 5.2))
CA, CB = "#2a9d8f", "#e76f51"   # N32 green (succ), N64 orange (fail)

# (1) chosen gripper vs step -- the grasp-timing desync
ax[0].plot(tA, gA, "-o", ms=4, color=CA, label="N=32 (succ)")
ax[0].plot(tB, gB, "-o", ms=4, color=CB, label="N=64 (fail)")
ax[0].axhline(0.5, color="gray", ls=":", lw=1)
if grA is not None: ax[0].axvline(grA, color=CA, ls="--", lw=1.3, label=f"N32 grasps @{grA}")
if grB is not None: ax[0].axvline(grB, color=CB, ls="--", lw=1.3, label=f"N64 grasps @{grB}")
ax[0].set_title("chosen action gripper (1=open, 0=close)\n-> grasp TIMING desync"); ax[0].set_xlabel("policy step")
ax[0].set_ylabel("mean gripper cmd"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)

# (2) chosen Q + pool Q range vs step
for d, t, c, lb in ((A, tA, CA, "N=32"), (B, tB, CB, "N=64")):
    v = d["values"]
    ax[1].fill_between(t, v.min(1), v.max(1), color=c, alpha=.15)
    ax[1].plot(t, chosen_q(d), "-o", ms=4, color=c, label=f"{lb} chosen Q")
ax[1].set_title("pool Q range (band) + chosen Q (line)"); ax[1].set_xlabel("policy step")
ax[1].set_ylabel("Q"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)

# (3) EE z vs step -- grasp & lift vs empty drift
ax[2].plot(tA, A["ee_xyz"][:, 2], "-o", ms=4, color=CA, label="N=32 EE z")
ax[2].plot(tB, B["ee_xyz"][:, 2], "-o", ms=4, color=CB, label="N=64 EE z")
ax[2].set_title("end-effector height z\n(N64 drifts UP empty, never grasps)"); ax[2].set_xlabel("policy step")
ax[2].set_ylabel("EE z"); ax[2].grid(alpha=.3); ax[2].legend(fontsize=8)

# (4) chosen index vs step; marker = shared(<32) vs extra(>=32) pool
sB = B["selected_index"]; sA = A["selected_index"]
ax[3].scatter(tA, sA, marker="o", s=45, c=CA, label="N32 pick (of 32)")
shared = sB < 32
ax[3].scatter(tB[shared],  sB[shared],  marker="o", s=45, c=CB, label="N64 pick (shared <32)")
ax[3].scatter(tB[~shared], sB[~shared], marker="^", s=70, c=CB, edgecolors="k",
              label="N64 pick from EXTRA pool (>=32)")
ax[3].axhline(32, color="gray", ls=":", lw=1)
ax[3].set_title("selected candidate index\n(softmax picks non-argmax; N64 dips into extra pool)")
ax[3].set_xlabel("policy step"); ax[3].set_ylabel("selected index"); ax[3].grid(alpha=.3); ax[3].legend(fontsize=7)

fig.suptitle("Stack-cube seed 1000, N=32 (succeeds) vs N=64 (fails). Larger softmax pool resamples a different "
             "action even from identical shared candidates -> grasp desyncs -> N64 closes gripper ~3 steps late & misses.",
             fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
p = f"{OUT}/stack_seed1000_n32_vs_n64.png"; fig.savefig(p, dpi=115, bbox_inches="tight"); plt.close(fig)
print(f"wrote {p}  (N32 grasp@{grA} succ, N64 grasp@{grB} fail)")
