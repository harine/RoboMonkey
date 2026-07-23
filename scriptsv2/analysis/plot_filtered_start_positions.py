#!/usr/bin/env python3
"""Colorblind-accessible start-position map for the strict_distance_success filter.

The eggplant reset positions are gridded (~28 discrete start cells, 8-109 episodes
each), so a raw scatter just stacks all categories on top of each other. Instead we
aggregate per cell and show:

  Panel A  per-cell composition, the three outcomes DODGED (offset) around each cell
           so none occlude, marker area ~ episode count:
             * FAILURE               (grey  x)  never reached basket (max reward<=0)
             * SUCCESS, filtered OUT (vermillion ^)  removed by the 3-stage filter
             * SUCCESS, kept         (blue  o)  survived -> training set (699)
  Panel B  kept-fraction among successes per cell (viridis) -> is filtering spatially
           biased? size ~ #successes in the cell.
  Panel C  success rate per cell (viridis) -> which start positions are hard. size ~
           #episodes.

Accessibility: Okabe-Ito palette + distinct marker shapes (A) and the perceptually
uniform, CVD-safe viridis map (B, C); every encoding also works in greyscale.

Basket is fixed at ~(-0.148, 0.022), ~0.19 m below the eggplant grid (off-panel);
noted in the title rather than plotted so the grid stays readable.

Extends scriptsv2/analysis/plot_eggplant_wide_collection.py (same start extraction).
"""
import json
from collections import defaultdict

import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "/gscratch/robotics/harine/data/eggplant_in_basket"
SHARDS = ["state0.zarr", "state1.zarr"]
KEPT_MANIFEST = "/gscratch/robotics/harine/data/strict_distance_success/filter_manifest.json"
OUTDIR = "/mmfs1/home/harine/RoboMonkey/data/eval/filtered_success_eggplant"

C_FAIL = "#999999"   # grey        (Okabe-Ito neutral)
C_OUT  = "#D55E00"   # vermillion
C_KEEP = "#0072B2"   # blue


def load():
    kept = {(e["src_shard"], e["src_episode"])
            for e in json.loads(open(KEPT_MANIFEST).read())["episodes"]}
    X, Y, CAT, B = [], [], [], []
    for sh in SHARDS:
        g = zarr.open(f"{SRC}/{sh}", "r")
        ends = np.asarray(g["meta"]["episode_ends"][:])
        st = np.concatenate([[0], ends[:-1]]).astype(np.int64)
        rew = np.asarray(g["data"]["rewards"][:])
        egg = np.asarray(g["data"]["obs"]["insertive_asset_pose"][:, :2])
        bsk = np.asarray(g["data"]["obs"]["receptive_asset_pose"][:, :2])
        for ep, (s, e) in enumerate(zip(st, ends)):
            succ = rew[s:e].max() > 0
            CAT.append("fail" if not succ else ("keep" if (sh, ep) in kept else "out"))
            X.append(float(egg[s, 0])); Y.append(float(egg[s, 1])); B.append(bsk[s])
    return np.array(X), np.array(Y), np.array(CAT), np.array(B, float)


def main():
    X, Y, CAT, B = load()
    n = len(CAT)
    idx = {"fail": 0, "out": 1, "keep": 2}
    key = np.round(np.c_[X, Y], 3)
    cells = defaultdict(lambda: [0, 0, 0])          # per cell: [fail, out, keep]
    for (x, y), c in zip(key, CAT):
        cells[(x, y)][idx[c]] += 1
    cxy = np.array(list(cells.keys()))
    comp = np.array(list(cells.values()), float)     # (Ncell, 3)
    fail, out, keep = comp[:, 0], comp[:, 1], comp[:, 2]
    tot = comp.sum(1); succ = out + keep
    kept_frac = np.where(succ > 0, keep / np.maximum(succ, 1), np.nan)
    succ_rate = succ / tot
    print(f"episodes={n}  cells={len(cxy)}  fail={int(fail.sum())} "
          f"out={int(out.sum())} keep={int(keep.sum())}")

    # dodge offsets (triangle) sized to the grid spacing
    ux, uy = np.unique(cxy[:, 0]), np.unique(cxy[:, 1])
    dx = np.min(np.diff(ux)) if len(ux) > 1 else 0.01
    dy = np.min(np.diff(uy)) if len(uy) > 1 else 0.01
    r = 0.22 * min(dx, dy)
    ang = {"fail": np.deg2rad(90), "out": np.deg2rad(210), "keep": np.deg2rad(330)}
    off = {k: (r * np.cos(a), r * np.sin(a)) for k, a in ang.items()}
    area = lambda cnt: 10 + 2.0 * cnt                 # marker area ~ count

    xlim = (ux.min() - 1.5 * dx, ux.max() + 1.5 * dx)
    ylim = (uy.min() - 1.5 * dy, uy.max() + 1.5 * dy)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.3))

    # --- Panel A: dodged composition -----------------------------------------
    ax = axes[0]
    specs = [("fail", fail, C_FAIL, "x", "failure"),
             ("out",  out,  C_OUT,  "^", "success, filtered out"),
             ("keep", keep, C_KEEP, "o", "success, kept")]
    for kkey, cnt, col, mk, lab in specs:
        ox, oy = off[kkey]
        m = cnt > 0
        ax.scatter(cxy[m, 0] + ox, cxy[m, 1] + oy, s=area(cnt[m]), c=col, marker=mk,
                   alpha=0.8, linewidths=0.6,
                   edgecolors="k" if mk != "x" else col,
                   label=f"{lab} ({int(cnt.sum())})")
    ax.set_title(f"Per-cell outcome composition (dodged)\n{len(cxy)} start cells, "
                 f"marker area ∝ #episodes")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95, scatterpoints=1)

    # --- Panel B: kept fraction among successes ------------------------------
    ax = axes[1]
    sc = ax.scatter(cxy[:, 0], cxy[:, 1], s=20 + 2.2 * succ, c=kept_frac,
                    cmap="viridis", vmin=0, vmax=1, edgecolors="k", linewidths=0.5)
    ax.set_title(f"Fraction of SUCCESSES kept per cell\n(mean {np.nanmean(kept_frac):.2f}, "
                 f"range {np.nanmin(kept_frac):.2f}–{np.nanmax(kept_frac):.2f})")
    fig.colorbar(sc, ax=ax, label="kept / successes", fraction=0.046, pad=0.04)

    # --- Panel C: success rate -----------------------------------------------
    ax = axes[2]
    sc = ax.scatter(cxy[:, 0], cxy[:, 1], s=20 + 1.4 * tot, c=succ_rate,
                    cmap="viridis", vmin=0, vmax=1, edgecolors="k", linewidths=0.5)
    ax.set_title(f"Success rate per cell\n(mean {succ_rate.mean():.2f}, "
                 f"range {succ_rate.min():.2f}–{succ_rate.max():.2f})")
    fig.colorbar(sc, ax=ax, label="successes / episodes", fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("eggplant start x (m)"); ax.set_ylabel("eggplant start y (m)")
        ax.set_aspect("equal"); ax.grid(alpha=0.25)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)

    fig.suptitle("strict_distance_success filter — start-position coverage   "
                 f"(699 kept / 1042 successes / {n} episodes;  basket fixed at "
                 f"(-0.148, 0.022), ~0.19 m below grid)", fontsize=12.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = f"{OUTDIR}/start_positions_filtered.{ext}"
        fig.savefig(p, dpi=140, bbox_inches="tight")
        print(f"[plot] wrote {p}")


if __name__ == "__main__":
    main()
