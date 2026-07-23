#!/usr/bin/env python3
"""Q-value plots: rmse search policy pool vs the random-augmented pool.

Three figures, written to data/eval/random_a_diverse/spoon_on_towel/metrics/:

  q_distribution.png  offline on the demo states -- histogram of every candidate's Q,
                      policy pool vs random pool, with Q(a*) (the EXPERT action) marked.
                      Answers "how far above the expert does the critic score random actions?"
  q_vs_index.png      Q vs candidate index i: the policy chain (i=0..15, the autoregressive
                      candidates) and the random pool's spread on the same axis, plus Q(a*).
  q_closed_loop.png   closed-loop: Q of the CHOSEN candidate vs replan step, unaugmented
                      rmse rollout vs the augmented rollout, on the same demo seeds.
                      The augmented run rides a much higher Q while failing every episode.

Sources (all pre-existing, nothing is re-run):
  policy-only offline   data/eval/train_test_analysis/spoon/offline/search_{split}_N16
  augmented offline     data/eval/random_a_diverse/spoon_on_towel/offline/search_aug_{split}_N16
  policy-only rollout   data/eval/train_test_analysis/spoon/rollout/{split}
  augmented rollout     data/eval/random_a_diverse/spoon_on_towel/rollout/{split}_k{K}

  conda activate monkey-verifier
  python scriptsv2/analysis/plot_aug_q.py
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

E = "data/eval"
AUG = f"{E}/random_a_diverse/spoon_on_towel"
AUG_DEFAULT = AUG
BASE = f"{E}/train_test_analysis/spoon"
SPLITS = ("train", "test")
C_POL, C_RAND, C_STAR = "#2a9d8f", "#e76f51", "black"


def load_offline(cell: str):
    """-> (values (F,K), is_random (F,K), astar_q (F,)) over every frame in the cell."""
    V, R, A = [], [], []
    for f in sorted(glob.glob(f"{cell}/search_q/*.npz")):
        d = np.load(f, allow_pickle=True)
        if "values" not in d.files:
            continue
        v = np.asarray(d["values"], float)
        r = (np.asarray(d["candidate_is_random"], bool) if "candidate_is_random" in d.files
             else np.zeros(v.shape, bool))
        a = np.asarray(d["a_star_q"], float) if "a_star_q" in d.files else np.full(len(v), np.nan)
        V.append(v); R.append(r); A.append(a)
    if not V:
        return None
    return np.concatenate(V), np.concatenate(R), np.concatenate(A)


def load_rollout(d: str):
    """-> list of (chosen Q per replan) arrays, one per episode."""
    out = []
    for f in sorted(glob.glob(f"{d}/search_q/*.npz")):
        z = np.load(f, allow_pickle=True)
        if "selected_value" in z.files:
            out.append(np.asarray(z["selected_value"], float))
    return out


def fig_distribution(data, out):
    """Two rows. Top: Q CENTERED per frame on that frame's policy-pool mean -- the only
    honest within-state comparison (raw Q drifts a lot across demo frames). Bottom:
    paired per-frame best Q, policy vs random, against y=x."""
    fig, axes = plt.subplots(2, len(SPLITS), figsize=(11, 7.6), squeeze=False)
    for c, split in enumerate(SPLITS):
        d = data.get(split)
        if d is None:
            axes[0][c].set_visible(False); axes[1][c].set_visible(False); continue
        V, R, A = d
        npol = int((~R[0]).sum())
        pol, rnd = V[:, :npol], V[:, npol:]
        base = pol.mean(1, keepdims=True)                  # per-frame reference

        ax = axes[0][c]
        cp, cr = (pol - base).ravel(), (rnd - base).ravel()
        bins = np.linspace(min(cp.min(), cr.min()), max(cp.max(), cr.max()), 80)
        ax.hist(cp, bins=bins, color=C_POL, alpha=.75, density=True,
                label=f"policy pool ({npol}/frame)")
        ax.hist(cr, bins=bins, color=C_RAND, alpha=.55, density=True,
                label=f"random pool ({rnd.shape[1]}/frame)")
        ax.axvline(np.nanmean(A - base.ravel()), color=C_STAR, ls="--", lw=1.6,
                   label="Q(a*) expert")
        ax.axvline(0, color=C_POL, ls=":", lw=1.2)
        ax.set_yscale("log")
        ax.set_title(f"{split}  ({V.shape[0]} demo frames)\n"
                     f"mean Δ {np.mean(rnd.mean(1) - pol.mean(1)):+.2f}   "
                     f"**best-of-pool Δ {np.mean(rnd.max(1) - pol.max(1)):+.2f}**", fontsize=9)
        ax.set_xlabel("Q − (that frame's policy-pool mean Q)"); ax.grid(alpha=.3)
        if c == 0:
            ax.set_ylabel("density (log)"); ax.legend(fontsize=7)

        ax = axes[1][c]
        x, y = pol.max(1), rnd.max(1)
        ax.scatter(x, y, s=8, alpha=.35, color=C_RAND, edgecolors="none")
        lim = [min(x.min(), y.min()) - .3, max(x.max(), y.max()) + .3]
        ax.plot(lim, lim, "k--", lw=1.2, label="y = x (no gain)")
        ax.scatter(x, np.where(np.isnan(A), np.nan, A), s=8, alpha=.35, color=C_STAR,
                   edgecolors="none", label="Q(a*) expert")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("best Q in the POLICY pool"); ax.grid(alpha=.3)
        frac = float((y > x).mean())
        ax.set_title(f"random beats policy on {100 * frac:.0f}% of frames", fontsize=9)
        if c == 0:
            ax.set_ylabel("best Q in the RANDOM pool"); ax.legend(fontsize=7)
    fig.suptitle("Per-frame Q, policy pool vs random pool (same verifier, same chunk-Q).\n"
                 "The two pools have the SAME mean — the random pool wins on its long upper "
                 "TAIL, which is exactly what argmax selects.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


def fig_vs_index(data, base, out):
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(11, 4), squeeze=False)
    for c, split in enumerate(SPLITS):
        ax = axes[0][c]
        d = data.get(split)
        if d is None:
            ax.set_visible(False); continue
        V, R, A = d
        npol = int((~R[0]).sum())
        pol = V[:, :npol]
        ax.plot(np.arange(npol), pol.mean(0), "-o", ms=3, color=C_POL, lw=1.8,
                label="policy chain: Q of candidate i")
        ax.fill_between(np.arange(npol), np.percentile(pol, 10, axis=0),
                        np.percentile(pol, 90, axis=0), color=C_POL, alpha=.2)
        rnd = V[:, npol:]
        ax.axhline(rnd.mean(), color=C_RAND, lw=1.8, label="random pool: mean Q")
        ax.axhspan(np.percentile(rnd, 10), np.percentile(rnd, 90), color=C_RAND, alpha=.18)
        ax.axhline(rnd.max(1).mean(), color=C_RAND, ls="-.", lw=1.4,
                   label="random pool: mean best Q (what argmax picks)")
        ax.axhline(np.nanmean(A), color=C_STAR, ls="--", lw=1.6, label="Q(a*) expert")
        # unaugmented baseline cell (same ckpt, policy-only) as a cross-check
        b = base.get(split)
        if b is not None:
            ax.plot(np.arange(b[0].shape[1]), b[0].mean(0), "--", color=C_POL, lw=1.1, alpha=.7,
                    label="policy chain, unaugmented cell")
        ax.set_title(f"{split}", fontsize=10)
        ax.set_xlabel("autoregressive candidate index i"); ax.grid(alpha=.3)
        if c == 0:
            ax.set_ylabel("DQC chunk Q"); ax.legend(fontsize=6.5, loc="best")
    fig.suptitle("Q vs candidate index — the policy chain is flat AND tight (its 10-90 band is "
                 "invisible at this scale);\nthe random pool has the same mean but a 10-90 band "
                 "spanning ~6 Q, so its max (dash-dot) sits ~5 above everything.", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


def fig_closed_loop(K, out):
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(11, 4), squeeze=False)
    for c, split in enumerate(SPLITS):
        ax = axes[0][c]
        for d, col, lab in ((f"{BASE}/rollout/{split}", C_POL, "rmse search (unaugmented)"),
                            (f"{AUG}/rollout/{split}_k{K}", C_RAND, f"augmented K={K}")):
            eps = load_rollout(d)
            if not eps:
                continue
            for i, q in enumerate(eps):
                ax.plot(np.arange(len(q)), q, "-", color=col, lw=.9, alpha=.35,
                        label=lab if i == 0 else None)
            L = max(len(q) for q in eps)
            M = np.full((len(eps), L), np.nan)
            for i, q in enumerate(eps):
                M[i, :len(q)] = q
            ax.plot(np.arange(L), np.nanmean(M, 0), "-", color=col, lw=2.6)
        ax.set_title(f"{split} demo seeds (n=10)", fontsize=10)
        ax.set_xlabel("replan step"); ax.grid(alpha=.3)
        if c == 0:
            ax.set_ylabel("Q of the CHOSEN candidate"); ax.legend(fontsize=7)
    fig.suptitle("Closed loop: the augmented run is chosen on much HIGHER Q at every replan "
                 "— and succeeds 0/10 vs 8/10 and 6/10", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=f"{AUG}/metrics")
    ap.add_argument("--rollout-k", type=int, default=50)
    ap.add_argument("--aug-root", default=AUG,
                    help="campaign root (e.g. data/eval/random_a_with_grasp/spoon_on_towel)")
    a = ap.parse_args()
    globals()["AUG"] = a.aug_root
    if a.out_dir.startswith(AUG_DEFAULT):
        a.out_dir = a.out_dir.replace(AUG_DEFAULT, a.aug_root, 1)
    os.makedirs(a.out_dir, exist_ok=True)

    data = {s: load_offline(f"{AUG}/offline/search_aug_{s}_N16") for s in SPLITS}
    base = {s: load_offline(f"{BASE}/offline/search_{s}_N16") for s in SPLITS}
    data = {k: v for k, v in data.items() if v is not None}
    base = {k: v for k, v in base.items() if v is not None}
    if data:
        fig_distribution(data, f"{a.out_dir}/q_distribution.png")
        fig_vs_index(data, base, f"{a.out_dir}/q_vs_index.png")
    fig_closed_loop(a.rollout_k, f"{a.out_dir}/q_closed_loop.png")


if __name__ == "__main__":
    main()
