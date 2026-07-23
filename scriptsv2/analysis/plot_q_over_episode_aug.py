#!/usr/bin/env python3
"""Q over the episode: sampled candidates + chosen, random-AUGMENTED vs NOT augmented.

(Sibling of plot_q_over_episode.py, which does the same for the PG-loss variants.)

One figure per (split, seed, K), both runs on the SAME axes -- they start from the same demo
seed, so env time is directly comparable and the Q axis is the same frozen DQC chunk critic:

  unaugmented (rmse search, 16 candidates)  teal   dots = every sampled candidate's Q
                                            teal   line = the CHOSEN (argmax-Q) candidate
  augmented   (+2K random candidates)       olive  dots = its 16 POLICY candidates
                                            salmon dots = its 2K RANDOM candidates
                                            salmon line = the CHOSEN candidate

Sources (nothing is re-run):
  unaugmented  data/eval/train_test_analysis/spoon/rollout/{split}/search_q/*.npz
  augmented    data/eval/random_a_diverse/spoon_on_towel/rollout/{split}_k{K}/search_q/*.npz

Out: data/eval/random_a_diverse/spoon_on_towel/q_over_episode/
     <split>_seed<seed>_k<K>_<base status>_vs_<aug status>.png   (mirrors the video naming)

  conda activate monkey-verifier
  python scriptsv2/analysis/plot_q_over_episode_aug.py
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

E = "data/eval"
AUG = f"{E}/random_a_diverse/spoon_on_towel"
AUG_DEFAULT = AUG
BASE = f"{E}/train_test_analysis/spoon"
C_BASE, C_POL, C_RAND = "#2a9d8f", "#6a994e", "#e76f51"


def load_ep(npz: str):
    d = np.load(npz, allow_pickle=True)
    v = np.asarray(d["values"], float)                       # (R, K)
    t = (np.asarray(d["branch_t"], int) if "branch_t" in d.files
         else np.arange(v.shape[0]))
    isr = (np.asarray(d["candidate_is_random"], bool) if "candidate_is_random" in d.files
           else np.zeros(v.shape, bool))
    return dict(
        seed=int(d["seed"]), t=t, values=v, is_random=isr,
        chosen=np.asarray(d["selected_value"], float),
        status=("succ" if int(d["success"]) else
                ("trunc" if int(d["truncated"]) else "fail")),
    )


def index_by_seed(cell: str):
    out = {}
    for f in sorted(glob.glob(f"{cell}/search_q/*.npz")):
        try:
            e = load_ep(f)
        except Exception:
            continue
        out[e["seed"]] = e
    return out


def scatter_pool(ax, t, vals, color, label, size=7, alpha=.30):
    ax.scatter(np.repeat(t, vals.shape[1]), vals.ravel(), s=size, color=color,
               alpha=alpha, edgecolors="none", label=label, zorder=2)


def plot_seed(split, seed, K, base, aug, out_dir):
    fig, ax = plt.subplots(figsize=(8.4, 4.8))

    # draw the big diffuse random cloud FIRST so the two tight policy clouds stay visible
    m = aug["is_random"]
    npol = int((~m[0]).sum())
    scatter_pool(ax, aug["t"], aug["values"][:, npol:], C_RAND,
                 f"augmented: {aug['values'].shape[1] - npol} random candidates",
                 size=6, alpha=.18)
    scatter_pool(ax, aug["t"], aug["values"][:, :npol], C_POL,
                 f"augmented: {npol} policy candidates", size=13, alpha=.55)
    scatter_pool(ax, base["t"], base["values"], C_BASE,
                 f"unaugmented: {base['values'].shape[1]} candidates", size=13, alpha=.55)

    ax.plot(base["t"], base["chosen"], "-o", ms=4, lw=2.4, color=C_BASE, zorder=5,
            label=f"unaugmented CHOSEN  ({base['status']})")
    ax.plot(aug["t"], aug["chosen"], "-o", ms=4, lw=2.4, color=C_RAND, zorder=5,
            label=f"augmented CHOSEN  ({aug['status']})")

    won = float(np.mean([bool(m[r, int(np.argmax(aug["values"][r]))])
                         for r in range(m.shape[0])]))
    ax.set_xlabel("env step of the replan")
    ax.set_ylabel("DQC chunk Q")
    ax.set_title(f"{split}  seed {seed}   K={K}\n"
                 f"unaugmented {base['status']} ({len(base['t'])} replans)  vs  "
                 f"augmented {aug['status']} ({len(aug['t'])} replans)   —   "
                 f"a random candidate won {100 * won:.0f}% of replans", fontsize=10)
    ax.grid(alpha=.3)
    leg = ax.legend(fontsize=7, loc="best", framealpha=.95, markerscale=2.2)
    for h in leg.legend_handles:            # opaque swatches even though the clouds are faint
        h.set_alpha(1.0)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    p = f"{out_dir}/{split}_seed{seed}_k{K}_{base['status']}_vs_{aug['status']}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=f"{AUG}/q_over_episode")
    ap.add_argument("--splits", nargs="*", default=["train", "test"])
    ap.add_argument("--k-list", nargs="*", type=int, default=None,
                    help="default: every rollout/<split>_k<K> cell present")
    ap.add_argument("--aug-root", default=AUG,
                    help="campaign root (e.g. data/eval/random_a_with_grasp/spoon_on_towel)")
    a = ap.parse_args()
    globals()["AUG"] = a.aug_root
    if a.out_dir.startswith(AUG_DEFAULT):
        a.out_dir = a.out_dir.replace(AUG_DEFAULT, a.aug_root, 1)

    n = 0
    for split in a.splits:
        base = index_by_seed(f"{BASE}/rollout/{split}")
        if not base:
            print(f"  !! no unaugmented rollout for {split}"); continue
        ks = a.k_list or sorted(
            int(re.search(r"_k(\d+)$", d).group(1))
            for d in glob.glob(f"{AUG}/rollout/{split}_k*")
            if re.search(r"_k(\d+)$", d))
        for K in ks:
            augd = index_by_seed(f"{AUG}/rollout/{split}_k{K}")
            common = sorted(set(base) & set(augd))
            if not common:
                print(f"  !! no shared seeds for {split} k{K}"); continue
            for seed in common:
                print(f"  -> {plot_seed(split, seed, K, base[seed], augd[seed], a.out_dir)}")
                n += 1
    print(f"[q-over-episode] {n} figures -> {a.out_dir}")


if __name__ == "__main__":
    main()
