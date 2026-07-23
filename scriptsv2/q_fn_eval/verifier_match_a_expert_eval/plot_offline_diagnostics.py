#!/usr/bin/env python
"""plot_offline_diagnostics.py — dataset-level diagnostics from the offline
candidate dumps written by sample_offline_candidates.py. Pure numpy/matplotlib
(jax-free), so it runs in any env with the search_q/*.npz files.

Three figures per split (one or more policy runs, faceted for comparison):

  q_vs_rmse_<split>.png        the key plot. Every candidate's verifier Q (y) vs
                               its RMSE-to-a* (x). Colored by success/failure
                               frame; a* drawn at RMSE=0 on the Q scale; Spearman
                               (Q, RMSE) annotated (negative = good verifier:
                               higher Q for lower action error). One subplot/run.
  astar_rank_quality_<split>.png  per run: a*'s rank among candidates by Q
                               (histogram) + a*'s absolute Q in success vs
                               failure frames (with AUC). Tests "is a* itself
                               scored low / not a good target".
  ranking_correctness_<split>.png  per run: per-frame concordance of Q vs
                               -RMSE (C-index, 0.5=chance), fraction of frames
                               where top-1 by Q == lowest-RMSE candidate, and
                               where the lowest-RMSE candidate is in Q's top-3.

Usage:
  python plot_offline_diagnostics.py --split train --out data/viz/_diag \
     --run "BC-H8:data/viz/bc_h8/train" \
     --run "diff10k:data/viz/diff10k/train" ... (one --run per policy)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARM = slice(0, 6)
GRIP = slice(6, 7)


# --------------------------------------------------------------------------- #
def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(random pos ranks above random neg). 0.5 = chance."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x[m])).astype(float)
    ry = np.argsort(np.argsort(y[m])).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def _concordance(q: np.ndarray, neg_rmse: np.ndarray) -> float:
    """C-index: over comparable candidate pairs, fraction where higher Q also
    has higher -RMSE (i.e. lower error). 0.5 = chance, 1.0 = Q perfectly orders
    candidates by closeness to a*."""
    q, r = np.asarray(q, float), np.asarray(neg_rmse, float)
    m = np.isfinite(q) & np.isfinite(r)
    q, r = q[m], r[m]
    if len(q) < 2:
        return float("nan")
    dq = q[:, None] - q[None, :]
    dr = r[:, None] - r[None, :]
    comp = dr != 0
    if comp.sum() == 0:
        return float("nan")
    conc = ((dq > 0) & (dr > 0)) | ((dq < 0) & (dr < 0))
    ties = (dq == 0) & comp
    return float((conc.sum() + 0.5 * ties.sum()) / comp.sum())


def _rmse(cand: np.ndarray, a_star: np.ndarray, dims) -> np.ndarray:
    """Per-candidate RMSE over `dims` of the chunk vs a* (NaN pad ignored)."""
    diff = cand[..., dims].astype(np.float64) - a_star[None, ..., dims].astype(np.float64)
    mask = ~np.isnan(a_star[None, ..., dims])
    se = np.where(mask, diff ** 2, np.nan)
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.nanmean(se.reshape(se.shape[0], -1), axis=1))


# --------------------------------------------------------------------------- #
def load_run(label: str, path: str) -> Dict:
    d = Path(path)
    sq = d if d.name == "search_q" else d / "search_q"
    frames = []   # list of dicts per frame
    for p in sorted(sq.glob("ep*.npz")):
        z = np.load(p, allow_pickle=False)
        cand = z["candidate_actions"]          # (R,K,chunk_h,7)
        vals = z["values"]                     # (R,K)
        a_star = z["a_star"]                    # (R,chunk_h,7)
        a_q = z["a_star_q"] if "a_star_q" in z.files else np.full(len(vals), np.nan)
        a_rk = z["a_star_rank"] if "a_star_rank" in z.files else np.full(len(vals), -1)
        succ = bool(int(z["success"])) if "success" in z.files else False
        for r in range(cand.shape[0]):
            frames.append(dict(
                cand=cand[r], vals=vals[r], a_star=a_star[r],
                a_q=float(a_q[r]), a_rank=int(a_rk[r]), success=succ,
            ))
    print(f"[load] {label}: {len(frames)} frames from {sq}")
    return dict(label=label, frames=frames)


def flatten_candidates(run: Dict) -> Dict[str, np.ndarray]:
    Q, Rall, Rarm, Rgrip, succ = [], [], [], [], []
    for f in run["frames"]:
        Q.append(f["vals"])
        Rall.append(_rmse(f["cand"], f["a_star"], slice(0, 7)))
        Rarm.append(_rmse(f["cand"], f["a_star"], ARM))
        Rgrip.append(_rmse(f["cand"], f["a_star"], GRIP))
        succ.append(np.full(len(f["vals"]), f["success"]))
    cat = lambda L: np.concatenate(L) if L else np.array([])
    return dict(Q=cat(Q), rmse_all=cat(Rall), rmse_arm=cat(Rarm),
                rmse_grip=cat(Rgrip), success=cat(succ).astype(bool))


# --------------------------------------------------------------------------- #
def fig_q_vs_rmse(runs: List[Dict], split: str, out: Path):
    ncol = len(runs)
    fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 4.8), squeeze=False)
    for j, run in enumerate(runs):
        flat = flatten_candidates(run)
        ax = axes[0][j]
        for succ, color, name in [(True, "steelblue", "success"),
                                  (False, "tomato", "failure")]:
            m = flat["success"] == succ
            if m.sum():
                ax.scatter(flat["rmse_all"][m], flat["Q"][m], s=8, alpha=0.25,
                           color=color, label=f"{name} frames", linewidths=0)
        # a* at RMSE=0 (its Q on the candidates' scale)
        a_qs = np.array([f["a_q"] for f in run["frames"] if np.isfinite(f["a_q"])])
        if len(a_qs):
            ax.scatter(np.zeros_like(a_qs), a_qs, marker="*", s=70, color="gold",
                       edgecolor="k", linewidth=0.4, zorder=5, label="a* (RMSE=0)")
        rho = _spearman(flat["Q"], flat["rmse_all"])
        ax.set_title(f"{run['label']}\nSpearman(Q,RMSE)={rho:+.3f}")
        ax.set_xlabel("RMSE to a* (7-D action chunk)")
        ax.set_ylabel("verifier Q (DQC chunk)")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"Q vs RMSE-to-a*  —  {split}  (negative Spearman = verifier "
                 f"rewards low action error)", fontsize=12)
    fig.tight_layout()
    p = out / f"q_vs_rmse_{split}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[out] {p}")


def fig_astar_rank_quality(runs: List[Dict], split: str, out: Path):
    ncol = len(runs)
    fig, axes = plt.subplots(2, ncol, figsize=(5.0 * ncol, 8.0), squeeze=False)
    for j, run in enumerate(runs):
        N = int(run["frames"][0]["vals"].shape[0]) if run["frames"] else 16
        ranks = np.array([f["a_rank"] for f in run["frames"] if f["a_rank"] >= 0])
        ax = axes[0][j]
        if len(ranks):
            ax.hist(ranks, bins=np.arange(-0.5, N + 1.5, 1), color="slateblue", alpha=0.8)
            ax.axvline(np.median(ranks), color="k", ls="--",
                       label=f"median={np.median(ranks):.0f}")
            ax.legend(fontsize=8)
        ax.set_title(f"{run['label']}\na* rank among N={N} (0=better than all)")
        ax.set_xlabel("# candidates with Q > a*'s Q"); ax.set_ylabel("frames")
        ax.grid(alpha=0.3, axis="y")

        ax = axes[1][j]
        pos = np.array([f["a_q"] for f in run["frames"] if f["success"] and np.isfinite(f["a_q"])])
        neg = np.array([f["a_q"] for f in run["frames"] if not f["success"] and np.isfinite(f["a_q"])])
        auc = _auc(pos, neg)
        for x, vals, color, name in [(0, pos, "steelblue", "success"),
                                     (1, neg, "tomato", "failure")]:
            if len(vals):
                jit = np.linspace(-0.16, 0.16, len(vals)) if len(vals) > 1 else [0.0]
                ax.scatter(x + np.asarray(jit), vals, s=20, alpha=0.6, color=color,
                           edgecolor="k", linewidth=0.3)
                ax.hlines(vals.mean(), x - 0.25, x + 0.25, color=color, linewidth=3)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["success", "failure"])
        ax.set_ylabel("a* Q")
        ax.set_title(f"a* Q by frame outcome   AUC={auc:.3f}")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"a* quality & rank  —  {split}", fontsize=12)
    fig.tight_layout()
    p = out / f"astar_rank_quality_{split}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[out] {p}")


def ranking_stats(run: Dict) -> Dict[str, float]:
    conc, top1, top3 = [], [], []
    for f in run["frames"]:
        rmse = _rmse(f["cand"], f["a_star"], slice(0, 7))
        q = f["vals"]
        if not np.isfinite(rmse).any():
            continue
        conc.append(_concordance(q, -rmse))
        closest = int(np.nanargmin(rmse))
        order = np.argsort(q)[::-1]
        top1.append(int(order[0]) == closest)
        top3.append(closest in set(int(k) for k in order[:3]))
    return dict(
        conc=np.array(conc, float),
        mean_conc=float(np.nanmean(conc)) if conc else float("nan"),
        top1_frac=float(np.mean(top1)) if top1 else float("nan"),
        top3_frac=float(np.mean(top3)) if top3 else float("nan"),
    )


def fig_ranking_correctness(runs: List[Dict], split: str, out: Path) -> List[Dict]:
    stats = [dict(label=r["label"], **ranking_stats(r)) for r in runs]
    labels = [s["label"] for s in stats]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(6.5 + 1.2 * len(labels), 4.6))

    ax = axes[0]
    w = 0.26
    ax.bar(x - w, [s["mean_conc"] for s in stats], w, label="C-index(Q,-RMSE)", color="seagreen")
    ax.bar(x, [s["top1_frac"] for s in stats], w, label="top1==minRMSE", color="darkorange")
    ax.bar(x + w, [s["top3_frac"] for s in stats], w, label="minRMSE in top3", color="slateblue")
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("fraction / index")
    ax.set_title("Per-frame ranking correctness"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    for s in stats:
        if len(s["conc"]):
            ax.hist(s["conc"], bins=np.linspace(0, 1, 21), histtype="step",
                    linewidth=1.6, label=s["label"])
    ax.axvline(0.5, color="grey", ls="--", lw=1)
    ax.set_xlabel("per-frame C-index(Q, -RMSE)"); ax.set_ylabel("frames")
    ax.set_title("Distribution of per-frame concordance"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"Ranking correctness  —  {split}  (0.5=chance)", fontsize=12)
    fig.tight_layout()
    p = out / f"ranking_correctness_{split}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[out] {p}")
    return stats


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True,
                    help="LABEL:DIR (DIR contains search_q/); repeatable, one per policy")
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    runs = []
    for spec in args.run:
        label, _, path = spec.partition(":")
        run = load_run(label, path)
        if run["frames"]:
            runs.append(run)
    if not runs:
        raise SystemExit("[diag] no frames loaded from any --run")

    fig_q_vs_rmse(runs, args.split, out)
    fig_astar_rank_quality(runs, args.split, out)
    stats = fig_ranking_correctness(runs, args.split, out)

    # CSV summary across runs (Spearman + ranking stats).
    csv_path = out / f"summary_{args.split}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "split", "n_frames", "spearman_Q_rmse",
                    "mean_concordance", "top1_match_frac", "top3_match_frac",
                    "astar_q_auc_succ_vs_fail"])
        for run, st in zip(runs, stats):
            flat = flatten_candidates(run)
            pos = np.array([f["a_q"] for f in run["frames"] if f["success"] and np.isfinite(f["a_q"])])
            neg = np.array([f["a_q"] for f in run["frames"] if not f["success"] and np.isfinite(f["a_q"])])
            w.writerow([run["label"], args.split, len(run["frames"]),
                        f"{_spearman(flat['Q'], flat['rmse_all']):.4f}",
                        f"{st['mean_conc']:.4f}", f"{st['top1_frac']:.4f}",
                        f"{st['top3_frac']:.4f}", f"{_auc(pos, neg):.4f}"])
    print(f"[out] {csv_path}")


if __name__ == "__main__":
    main()
