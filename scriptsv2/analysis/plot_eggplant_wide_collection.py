#!/usr/bin/env python3
"""Visualize the eggplant_wide ("wider reset") collection: eggplant start positions
colored by collection success/fail, + a binned success-rate heatmap over the reset
region. Reads long0.zarr (collect_trajectories.py output).

Success is derived from episode length: a SUCCESS episode terminates early (env
returned done=task-complete); a FAIL runs to max_steps (truncation). We infer
max_steps as the modal/longest length and cross-check the success fraction against
the collector's reported rate (~0.327).
"""
import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ZARR = "/gscratch/robotics/harine/data/eggplant_reset_core_x-0.22_-0.06_y0.02_0.26/long0.zarr"
OUT = "data/eval/_verifier_comparison/eggplant_wide_collection.png"

g = zarr.open(ZARR, "r")
ends = np.asarray(g["meta"]["episode_ends"][:])              # (N,) cumulative
src = np.asarray(g["data"]["obs"]["insertive_asset_pose"][:, :2])   # (T,2) eggplant xy
tgt = np.asarray(g["data"]["obs"]["receptive_asset_pose"][:, :2])   # (T,2) basket xy
N = len(ends)
starts = np.empty((N, 2), np.float32)
basket = np.empty((N, 2), np.float32)
lengths = np.empty(N, np.int64)
prev = 0
for i, e in enumerate(ends):
    starts[i] = src[prev]          # eggplant pos at episode's first recorded step
    basket[i] = tgt[prev]
    lengths[i] = e - prev
    prev = e

# success = terminated before max_steps (done=task complete). max_steps = modal long length.
maxlen = int(np.bincount(lengths).argmax())   # the truncation length (most common = fails)
# fails run to ~maxlen; successes are strictly shorter
succ = lengths < maxlen
sr = succ.mean()
print(f"[plot] episodes={N}  maxlen(trunc)={maxlen}  success_frac={sr:.3f} (collector reported ~0.327)")
print(f"[plot] eggplant start x range [{starts[:,0].min():.3f},{starts[:,0].max():.3f}]  "
      f"y [{starts[:,1].min():.3f},{starts[:,1].max():.3f}]")

# Clip the view to the dense reset region (p1-p99); ~1-5% physics-glitch outliers
# stretch the raw range (x up to 0.73). Bin/plot over the bulk.
xlo, xhi = np.percentile(starts[:, 0], [1, 99])
ylo, yhi = np.percentile(starts[:, 1], [1, 99])
mx = 0.02 * (xhi - xlo); my = 0.02 * (yhi - ylo)
xlim = (xlo - mx, xhi + mx); ylim = (ylo - my, yhi + my)
n_out = int((~((starts[:, 0] >= xlim[0]) & (starts[:, 0] <= xlim[1]) &
               (starts[:, 1] >= ylim[0]) & (starts[:, 1] <= ylim[1]))).sum())

fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))

# --- Panel 1: scatter, success vs fail ---
ax = axes[0]
ax.scatter(starts[~succ, 0], starts[~succ, 1], s=9, c="#d62728", alpha=0.45, label=f"fail ({(~succ).sum()})", linewidths=0)
ax.scatter(starts[succ, 0],  starts[succ, 1],  s=9, c="#2ca02c", alpha=0.55, label=f"success ({succ.sum()})", linewidths=0)
ax.scatter(basket[:, 0].mean(), basket[:, 1].mean(), marker="*", s=320, c="gold",
           edgecolors="k", linewidths=0.8, label="basket (mean)", zorder=5)
ax.set_xlabel("eggplant start x (m)"); ax.set_ylabel("eggplant start y (m)")
ax.set_title(f"eggplant_wide start positions by outcome\n{N} episodes, SR={100*sr:.1f}%  ({n_out} outliers off-plot)")
ax.legend(loc="upper right", fontsize=9); ax.set_aspect("equal"); ax.grid(alpha=0.25)
ax.set_xlim(*xlim); ax.set_ylim(*ylim)

# --- Panel 2: binned success-rate heatmap ---
ax = axes[1]
nb = 12
xe = np.linspace(xlim[0], xlim[1], nb + 1)
ye = np.linspace(ylim[0], ylim[1], nb + 1)
tot, _, _ = np.histogram2d(starts[:, 0], starts[:, 1], bins=[xe, ye])
win, _, _ = np.histogram2d(starts[succ, 0], starts[succ, 1], bins=[xe, ye])
with np.errstate(invalid="ignore", divide="ignore"):
    rate = np.where(tot >= 3, win / tot, np.nan)   # need >=3 samples/bin
im = ax.pcolormesh(xe, ye, rate.T, cmap="RdYlGn", vmin=0, vmax=1, shading="flat")
ax.scatter(basket[:, 0].mean(), basket[:, 1].mean(), marker="*", s=320, c="gold",
           edgecolors="k", linewidths=0.8, zorder=5)
ax.set_xlabel("eggplant start x (m)"); ax.set_ylabel("eggplant start y (m)")
ax.set_title(f"success rate by start cell ({nb}x{nb}, ≥3 samples/bin)")
ax.set_aspect("equal")
fig.colorbar(im, ax=ax, label="success rate", fraction=0.046, pad=0.04)

fig.suptitle("eggplant_wide ('wider reset') data collection — OpenVLA expert rollouts", fontsize=13)
fig.tight_layout()
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"[plot] wrote {OUT}")
