#!/usr/bin/env python3
"""Pool-split metrics for the random-augmented offline candidate cells.

Reads the npz written by ``sample_offline_candidates.py --rand-aug-k K`` (which
carries the extra ``candidate_is_random`` (R,K) mask) and answers, per frame and
aggregated per cell:

  * best-Q and mean-Q of the POLICY pool vs the RANDOM pool, and Q(a*)
  * frac_argmax_random -- how often the augmented pool wins the argmax
  * diversity (mean pairwise cosine of net dxyz; 1 = collapsed) per pool
  * oracle RMSE to a* (min over the pool) and Q-pick RMSE, per pool and on the union
    -> whether the extra coverage actually contains better actions, and whether Q finds them

Reuses ``offline_candidate_metrics._rmse`` so the RMSE convention (NaN-aware, whole
8-step chunk) is identical to the main analysis.

  conda activate monkey-verifier
  python scriptsv2/analysis/aug_candidate_metrics.py \
      --root data/eval/random_a_diverse/spoon_on_towel/offline
Out: <root>/../metrics/aug_pool_metrics.csv  (+ printed table)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from offline_candidate_metrics import _rmse, ALL  # noqa: E402

COLS = ["cell", "n_frames", "n_policy", "n_random",
        "q_pol_best", "q_rand_best", "q_astar", "frac_argmax_random",
        "div_pol", "div_rand", "div_union",
        "oracle_pol", "oracle_rand", "oracle_union",
        "qpick_pol", "qpick_union"]


def _div_cos(cand: np.ndarray) -> float:
    """Mean pairwise cosine of the candidates' net dxyz (1.0 = collapsed)."""
    if cand.shape[0] < 2:
        return float("nan")
    net = np.nansum(cand[:, :, :3], axis=1)
    u = net / (np.linalg.norm(net, axis=1, keepdims=True) + 1e-9)
    C = u @ u.T
    n = C.shape[0]
    return float((C.sum() - np.trace(C)) / (n * (n - 1)))


def cell_metrics(cell_dir: str) -> dict | None:
    rows = []
    for f in sorted(glob.glob(f"{cell_dir}/search_q/*.npz")):
        d = np.load(f, allow_pickle=True)
        if "candidate_actions" not in d.files or "candidate_is_random" not in d.files:
            continue
        ca, vv, ast = d["candidate_actions"], d["values"], d["a_star"]
        isr = np.asarray(d["candidate_is_random"], bool)
        aq = d["a_star_q"] if "a_star_q" in d.files else np.full(len(vv), np.nan)
        for r in range(ca.shape[0]):
            m = isr[r]
            if not m.any() or m.all():
                continue                       # not an augmented cell / no policy pool
            cand, vals = ca[r], vv[r]
            rmse = _rmse(cand, ast[r], ALL)
            pol, rnd = ~m, m
            rows.append(dict(
                q_pol_best=float(np.nanmax(vals[pol])),
                q_rand_best=float(np.nanmax(vals[rnd])),
                q_astar=float(aq[r]),
                argmax_random=float(bool(m[int(np.argmax(vals))])),
                div_pol=_div_cos(cand[pol]), div_rand=_div_cos(cand[rnd]),
                div_union=_div_cos(cand),
                oracle_pol=float(np.nanmin(rmse[pol])),
                oracle_rand=float(np.nanmin(rmse[rnd])),
                oracle_union=float(np.nanmin(rmse)),
                qpick_pol=float(rmse[pol][int(np.argmax(vals[pol]))]),
                qpick_union=float(rmse[int(np.argmax(vals))]),
                n_policy=int(pol.sum()), n_random=int(rnd.sum()),
            ))
    if not rows:
        return None
    agg = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    agg["n_frames"] = len(rows)
    agg["cell"] = os.path.basename(cell_dir)
    agg["frac_argmax_random"] = agg.pop("argmax_random")
    agg["n_policy"] = int(round(agg["n_policy"]))
    agg["n_random"] = int(round(agg["n_random"]))
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir holding the <cell>/search_q trees")
    ap.add_argument("--out", default=None, help="CSV path (default <root>/../metrics/aug_pool_metrics.csv)")
    a = ap.parse_args()

    cells = [c for c in sorted(glob.glob(f"{a.root}/*")) if os.path.isdir(c)]
    aggs = [m for m in (cell_metrics(c) for c in cells) if m]
    if not aggs:
        print(f"[aug-metrics] no augmented cells under {a.root}")
        return

    out = Path(a.out) if a.out else Path(a.root).parent / "metrics" / "aug_pool_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.write(",".join(COLS) + "\n")
        for m in aggs:
            fh.write(",".join(
                (f"{m[c]:.5f}" if isinstance(m[c], float) else str(m[c])) for c in COLS) + "\n")

    w = max(len(m["cell"]) for m in aggs)
    print(f"{'cell':{w}}  frames  Qpol_best  Qrand_best   Q(a*)  argmaxRand  "
          f"div_pol  div_rand  oracle_pol  oracle_union")
    for m in aggs:
        print(f"{m['cell']:{w}}  {m['n_frames']:6d}  {m['q_pol_best']:9.3f}  "
              f"{m['q_rand_best']:10.3f}  {m['q_astar']:6.3f}  {m['frac_argmax_random']:10.3f}  "
              f"{m['div_pol']:7.3f}  {m['div_rand']:8.3f}  {m['oracle_pol']:10.4f}  "
              f"{m['oracle_union']:12.4f}")
    print(f"[aug-metrics] -> {out}")


if __name__ == "__main__":
    main()
