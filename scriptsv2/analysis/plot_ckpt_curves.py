#!/usr/bin/env python3
"""Plot best-of-N action-RMSE learning curves (train vs test) vs training step.

Reads the shard CSVs written by ckpt_rmse_curve.py:
    data/eval/train_test_analysis/_curves/<task>_<split>_s<shard>.csv
    columns: step,split,N,best_of_N,qpick_N,q_best_N,q_mean_i0,n_frames

Per task, writes <task>/metrics/ckpt_rmse_curve.png with two panels:
  left  : best-of-N RMSE vs step, one line per N (1,4,16,64), train solid / test dashed
          -> the oracle curve: how close the candidate SET gets to a*
  right : Q-pick RMSE vs step (what argmax-Q actually selects) with the N=64 oracle
          underneath -> the gap is the cost of the critic's ranking, over training

Also writes _curves/ckpt_curves.csv (merged) and prints the best step per (task, split, N).

  conda activate monkey-verifier
  python scriptsv2/analysis/plot_ckpt_curves.py
"""
from __future__ import annotations

import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "data/eval/train_test_analysis"
CURVES = f"{ROOT}/_curves"
TASKS = ["eggplant", "spoon", "stack_cube"]
NS = [1, 4, 16, 64]
COL = {1: "#264653", 4: "#2a9d8f", 16: "#e9c46a", 64: "#e76f51"}
FN = re.compile(r"^(eggplant|spoon|stack_cube)_(train|test)_s\d+\.csv$")


def load():
    """-> data[task][split][N] = (steps sorted, best, qpick, qbest)"""
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for p in sorted(glob.glob(f"{CURVES}/*.csv")):
        m = FN.match(os.path.basename(p))
        if not m:
            continue
        task = m.group(1)
        for r in csv.DictReader(open(p)):
            N, step = int(r["N"]), int(r["step"])
            raw[task][r["split"]][N][step] = (
                float(r["best_of_N"]), float(r["qpick_N"]), float(r["q_best_N"]))
    out = {}
    for task, bysplit in raw.items():
        out[task] = {}
        for split, byN in bysplit.items():
            out[task][split] = {}
            for N, bystep in byN.items():
                steps = sorted(bystep)
                arr = np.array([bystep[s] for s in steps])
                out[task][split][N] = (np.array(steps), arr[:, 0], arr[:, 1], arr[:, 2])
    return out


def main() -> None:
    data = load()
    if not data:
        print("no curve CSVs yet"); return

    # merged csv
    os.makedirs(CURVES, exist_ok=True)
    with open(f"{CURVES}/ckpt_curves.csv", "w") as fh:
        fh.write("task,split,N,step,best_of_N,qpick_N,q_best_N\n")
        for task in sorted(data):
            for split in sorted(data[task]):
                for N in sorted(data[task][split]):
                    st, b, q, qb = data[task][split][N]
                    for i in range(len(st)):
                        fh.write(f"{task},{split},{N},{st[i]},{b[i]:.6f},{q[i]:.6f},{qb[i]:.5f}\n")
    print(f"[curves] merged -> {CURVES}/ckpt_curves.csv")

    for task in TASKS:
        if task not in data:
            continue
        d = data[task]
        out = f"{ROOT}/{task}/metrics"
        os.makedirs(out, exist_ok=True)
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 4.6))

        for split, ls in (("train", "-"), ("test", "--")):
            if split not in d:
                continue
            for N in NS:
                if N not in d[split]:
                    continue
                st, b, _, _ = d[split][N]
                axL.plot(st, b, ls, color=COL[N], lw=1.7,
                         label=f"N={N} {split}" if N in (1, 64) else None)
        axL.set_xlabel("training step"); axL.set_ylabel("best-of-N RMSE to a*")
        axL.set_title(f"{task} — oracle best-of-N RMSE\n(solid=train, dashed=test)", fontsize=10)
        axL.grid(alpha=.3); axL.legend(fontsize=7)

        for split, ls in (("train", "-"), ("test", "--")):
            if split not in d or 64 not in d[split]:
                continue
            st, b, q, _ = d[split][64]
            axR.plot(st, q, ls, color="#e76f51", lw=1.9, label=f"Q-pick N=64 {split}")
            axR.plot(st, b, ls, color="#2a9d8f", lw=1.3, alpha=.7, label=f"oracle N=64 {split}")
        axR.set_xlabel("training step"); axR.set_ylabel("RMSE to a*")
        axR.set_title(f"{task} — what Q selects vs the best available\n"
                      f"(gap = cost of the critic's ranking)", fontsize=10)
        axR.grid(alpha=.3); axR.legend(fontsize=7)

        fig.tight_layout()
        p = f"{out}/ckpt_rmse_curve.png"
        fig.savefig(p, dpi=115, bbox_inches="tight"); plt.close(fig)
        print(f"[curves] {task} -> {p}")

        for split in ("train", "test"):
            if split not in d:
                continue
            for N in NS:
                if N not in d[split]:
                    continue
                st, b, _, _ = d[split][N]
                i = int(np.argmin(b))
                print(f"    {task:11s} {split:5s} N={N:<2d}: best step={st[i]:6d} "
                      f"RMSE={b[i]:.4f}   (last={b[-1]:.4f}, first={b[0]:.4f}, "
                      f"{len(st)} ckpts)")


if __name__ == "__main__":
    main()
