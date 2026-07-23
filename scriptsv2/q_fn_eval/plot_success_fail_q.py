#!/usr/bin/env python
"""plot_success_fail_q.py — success-vs-failure verifier-Q discrimination over the
episode, for a (re)scored search_q dump. Same two-panel layout as the previous
eval (scriptsv2/q_fn_eval/plot_utils.make_plot, used by eval_dqc/eval_robomonkey):

  top    — per-replan verifier Q over the episode timeline (env timestep at
           replan), each episode a faint line, success (blue) vs failure (red),
           with mean±std bands.
  bottom — per-EPISODE mean Q, success vs failure distributions, with the AUC
           = P(random success-episode Q > random failure-episode Q); 0.5 = chance.

The Q plotted is the verifier's score of the DEPLOYED action chunk per replan
(``selected_value``; falls back to argmax of ``values``) — i.e. the quantity the
policy actually acted on, so a good verifier should score higher across a
successful rollout than a failed one. Point --run at a finetuned rescore dir
(values/selected_value = finetuned Q) to get the finetuned version; at the raw
data/viz/<cfg>/<split> for the base DQC.

Usage:
  python plot_success_fail_q.py --tag v2_all_s30000 --out data/viz_q/_sf \
     --run "bc_h8:data/viz_q/rescore/v2_all_s30000/bc_h8/train" \
     --run "diff33k:data/viz_q/rescore/v2_all_s30000/diff33k/train" ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from plot_utils import make_plot, _auc


def load_run_df(path: str, metric_key: str) -> pd.DataFrame:
    d = Path(path)
    sq = d if d.name == "search_q" else d / "search_q"
    rows = []
    for ep_i, p in enumerate(sorted(sq.glob("ep*.npz"))):
        z = np.load(p, allow_pickle=False)
        vals = z["values"]                                   # (R,K)
        R = vals.shape[0]
        if metric_key in z.files:
            q = np.asarray(z[metric_key], np.float32).reshape(-1)
        else:                                                # fall back to argmax-Q
            q = vals.max(1).astype(np.float32)
        bt = np.asarray(z["branch_t"], np.float32).reshape(-1) if "branch_t" in z.files else np.arange(R, dtype=np.float32)
        succ = bool(int(z["success"])) if "success" in z.files else False
        for r in range(R):
            rows.append(dict(ep_idx=ep_i, branch_t=float(bt[r]),
                             success=succ, q_chunk=float(q[r])))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="LABEL:DIR (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="ft")
    ap.add_argument("--metric-key", default="selected_value",
                    help="NPZ key for per-replan deployed-action Q")
    ap.add_argument("--source", default="raw", choices=["raw", "zarr"],
                    help="raw=env-timestep x-axis; zarr=normalized episode progress")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    combined = []
    for spec in args.run:
        label, path = spec.split(":", 1)
        df = load_run_df(path, args.metric_key)
        if df.empty:
            print(f"[sf] {label}: no frames, skip"); continue
        n_succ = df.groupby("ep_idx")["success"].first().sum()
        n_ep = df["ep_idx"].nunique()
        per_ep = df.groupby("ep_idx").agg(success=("success", "first"), val=("q_chunk", "mean"))
        auc = _auc(per_ep[per_ep.success].val.values, per_ep[~per_ep.success].val.values)
        print(f"[sf] {label}: {n_ep} eps ({n_succ} succ / {n_ep-n_succ} fail)  "
              f"per-ep success/fail AUC={auc:.3f}")
        p = out / f"success_fail_q_{args.tag}_{label}.png"
        make_plot(df, p, f"{label}  ({args.tag})  succ/fail verifier-Q", args.source)
        print(f"[sf]   -> {p}")
        df["label"] = label
        combined.append(df)

    if combined:
        allp = out / f"success_fail_q_{args.tag}_ALL.png"
        make_plot(pd.concat(combined, ignore_index=True), allp,
                  f"all runs  ({args.tag})  succ/fail verifier-Q", args.source)
        print(f"[sf] combined -> {allp}")


if __name__ == "__main__":
    main()
