#!/usr/bin/env python
"""plot_advantage_investigation.py — investigate the expert-vs-predicted advantage
A(s) = Q(s, a_expert) - Q(s, a_pred) on the cached offline logs (search_q/*.npz),
for the EXISTING DQC Q (default) or a re-scored / finetuned Q (via --q-key/--aq-key).

a_expert = a*  (the recorded demo chunk; Q is a_star_q).
a_pred   = the policy's deployed pick = candidate `selected_index` (Q is values[sel]).
We also report the advantage vs the mean candidate and vs the verifier's top candidate.

The verifier ranks well iff a* scores above the policy's prediction: A > 0. The base
DQC ranks near chance, so we expect A <= 0 often (the verifier prefers its own pick over
the demo). Re-running with a finetuned Q (rescore_offline_candidates.py writes
values_<tag>/a_star_q_<tag>) shows whether finetuning pushes A positive.

Pure numpy/matplotlib (no GPU). Reuses the loading/metric conventions of
plot_offline_diagnostics.py.

Usage (existing DQC, base Q already in the NPZs):
  python plot_advantage_investigation.py --split train --out data/viz_q/_adv \
     --run "bc_h8:data/viz/bc_h8/train" --run "diff33k:data/viz/diff33k/train" ...
Finetuned Q (after rescore writes values_ft):
  ... --q-key values_ft --aq-key a_star_q_ft --tag ft
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_run(label: str, path: str, q_key: str, aq_key: str) -> Dict:
    d = Path(path)
    sq = d if d.name == "search_q" else d / "search_q"
    frames = []
    for p in sorted(sq.glob("ep*.npz")):
        z = np.load(p, allow_pickle=False)
        if q_key not in z.files or aq_key not in z.files:
            continue
        cand_q = z[q_key]                       # (R,K)
        aq = z[aq_key]                           # (R,)
        sel = z["selected_index"] if "selected_index" in z.files else cand_q.argmax(1)
        succ = bool(int(z["success"])) if "success" in z.files else False
        for r in range(cand_q.shape[0]):
            frames.append(dict(cand_q=cand_q[r], aq=float(aq[r]),
                               sel=int(sel[r]), success=succ))
    return dict(label=label, frames=frames)


def advantages(run: Dict) -> Dict[str, np.ndarray]:
    A_sel, A_mean, A_top, rank, succ = [], [], [], [], []
    for f in run["frames"]:
        q, aq, sel = f["cand_q"], f["aq"], f["sel"]
        if not np.isfinite(aq):
            continue
        A_sel.append(aq - q[sel])               # expert vs deployed pick
        A_mean.append(aq - float(np.mean(q)))   # expert vs average candidate
        A_top.append(aq - float(np.max(q)))     # expert vs verifier's best candidate
        rank.append(int((q > aq).sum()))        # #candidates beating a*  (0=a* best)
        succ.append(f["success"])
    arr = lambda L: np.asarray(L, float)
    return dict(A_sel=arr(A_sel), A_mean=arr(A_mean), A_top=arr(A_top),
                rank=np.asarray(rank, int), success=np.asarray(succ, bool))


def plot_runs(runs: List[Dict], split: str, out: Path, tag: str):
    data = [(r["label"], advantages(r)) for r in runs]
    data = [(lbl, a) for lbl, a in data if a["A_sel"].size]
    if not data:
        print("[adv] no frames"); return
    ncol = len(data)
    fig, axes = plt.subplots(2, ncol, figsize=(4.6 * ncol, 8), squeeze=False)
    for j, (lbl, a) in enumerate(data):
        ax = axes[0][j]
        ax.hist(a["A_sel"], bins=30, color="slateblue", alpha=0.8)
        ax.axvline(0, color="k", ls="--", lw=1)
        frac = float((a["A_sel"] > 0).mean())
        ax.set_title(f"{lbl}\nA=Q(a*)-Q(a_pred)  [{tag}]\n"
                     f"mean={a['A_sel'].mean():+.3f}  frac(A>0)={frac:.2f}")
        ax.set_xlabel("advantage (a_pred = deployed pick)"); ax.set_ylabel("frames")
        ax.grid(alpha=0.3)

        ax = axes[1][j]
        N = int(a["rank"].max()) if a["rank"].size else 16
        ax.hist(a["rank"], bins=np.arange(-0.5, N + 1.5, 1), color="seagreen", alpha=0.8)
        ax.axvline(np.median(a["rank"]), color="k", ls="--",
                   label=f"median={np.median(a['rank']):.0f}")
        ax.set_title(f"a* rank among candidates (0 = a* best)")
        ax.set_xlabel("# candidates with Q > Q(a*)"); ax.set_ylabel("frames")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Expert-vs-predicted advantage  —  {split}  ({tag} DQC Q)", fontsize=13)
    fig.tight_layout()
    p = out / f"advantage_{tag}_{split}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[out] {p}")

    csv_path = out / f"advantage_{tag}_{split}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "split", "tag", "n_frames", "mean_A_sel", "frac_A_sel>0",
                    "mean_A_meanpool", "mean_A_vs_top", "mean_astar_rank", "median_astar_rank"])
        for lbl, a in data:
            w.writerow([lbl, split, tag, a["A_sel"].size,
                        f"{a['A_sel'].mean():.4f}", f"{(a['A_sel']>0).mean():.4f}",
                        f"{a['A_mean'].mean():.4f}", f"{a['A_top'].mean():.4f}",
                        f"{a['rank'].mean():.3f}", f"{np.median(a['rank']):.1f}"])
    print(f"[out] {csv_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="LABEL:DIR (repeatable)")
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--q-key", default="values", help="NPZ key for candidate Q")
    ap.add_argument("--aq-key", default="a_star_q", help="NPZ key for Q(a*)")
    ap.add_argument("--tag", default="base", help="label for this Q source (base|ft|...)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    runs = []
    for spec in args.run:
        lbl, _, path = spec.partition(":")
        r = load_run(lbl, path, args.q_key, args.aq_key)
        print(f"[load] {lbl}: {len(r['frames'])} frames ({args.q_key}/{args.aq_key})")
        if r["frames"]:
            runs.append(r)
    if not runs:
        raise SystemExit("[adv] no runs loaded (missing q-key in NPZs?)")
    plot_runs(runs, args.split, out, args.tag)


if __name__ == "__main__":
    main()
