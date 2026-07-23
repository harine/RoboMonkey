#!/usr/bin/env python3
"""Overlay normalized (and raw) best-of-16 / Q-pick action-RMSE vs training step for
the rmse-loss (non-PG) vs PG (awbc / reinforce) sub50 search policies, per task.

Reads the merged shards under data/eval/norm_curves/_curves/<pol>_<split>_s<sh>.csv
(cols incl. best_of_N / best_of_N_norm / qpick_N / qpick_N_norm at N=1..16), writes
data/eval/norm_curves/<task>_norm_rmse.png  (rows: bo16 | Q-pick ; cols: train | test)
and _summary.csv with the final-step values.
"""
import csv, glob, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "data/eval/norm_curves"
CUR = f"{BASE}/_curves"
NPLOT = 16
TASKS = {
    "eggplant": ["rmse_eggplant", "pg_awbc_l0.1_eggplant_sub50", "pg_reinf_l0.1_eggplant_sub50"],
    "stack":    ["rmse_stack", "pg_awbc_l0.1_stack_sub50", "pg_reinf_l0.1_stack_sub50"],
    "spoon":    ["rmse_spoon", "pg_awbc_l0.1_spoon_sub50", "pg_reinf_l0.1_spoon_sub50",
                 "pg_awbc_l0.3_spoon_sub50", "pg_reinf_l0.3_spoon_sub50"],
}
LABEL = {"rmse_eggplant": "rmse-loss", "rmse_stack": "rmse-loss", "rmse_spoon": "rmse-loss",
         "pg_awbc_l0.1_eggplant_sub50": "awbc λ0.1", "pg_reinf_l0.1_eggplant_sub50": "reinf λ0.1",
         "pg_awbc_l0.1_stack_sub50": "awbc λ0.1", "pg_reinf_l0.1_stack_sub50": "reinf λ0.1",
         "pg_awbc_l0.1_spoon_sub50": "awbc λ0.1", "pg_reinf_l0.1_spoon_sub50": "reinf λ0.1",
         "pg_awbc_l0.3_spoon_sub50": "awbc λ0.3", "pg_reinf_l0.3_spoon_sub50": "reinf λ0.3"}
COLOR = {"rmse-loss": "#444444", "awbc λ0.1": "#2e86ab", "reinf λ0.1": "#d1495b",
         "awbc λ0.3": "#118ab2", "reinf λ0.3": "#e07a1f"}


def load():
    d = defaultdict(lambda: defaultdict(dict))   # pol -> split -> step -> row(N=16)
    for f in glob.glob(f"{CUR}/*_s*.csv"):
        stem = os.path.basename(f).rsplit("_s", 1)[0]
        pol, split = stem.rsplit("_", 1)
        for r in csv.DictReader(open(f)):
            if int(r["N"]) != NPLOT:
                continue
            d[pol][split][int(r["step"])] = r
    return d


def series(d, pol, split, col):
    rows = d[pol][split]
    steps = sorted(rows)
    return np.array(steps), np.array([float(rows[s][col]) for s in steps])


def main():
    d = load()
    summ = []
    for task, pols in TASKS.items():
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for ci, split in enumerate(("train", "test")):
            for ri, (col, ylab) in enumerate([("best_of_N_norm", "normalized best-of-16 RMSE"),
                                               ("qpick_N_norm", "normalized Q-pick RMSE")]):
                ax = axes[ri][ci]
                for pol in pols:
                    if pol not in d or split not in d[pol]:
                        continue
                    st, y = series(d, pol, split, col)
                    lab = LABEL[pol]
                    ax.plot(st / 1000, y, "-o", ms=3, lw=1.6, color=COLOR[lab], label=lab)
                    if ri == 0:
                        summ.append(dict(task=task, policy=lab, split=split,
                                         last_step=int(st[-1]), bo16_norm=float(y[-1])))
                ax.set_title(f"{task} · {split} · {ylab}", fontsize=10)
                ax.set_ylabel(ylab, fontsize=8); ax.grid(alpha=.3)
                if ri == 1:
                    ax.set_xlabel("training step (k)")
                if ci == 0 and ri == 0:
                    ax.legend(fontsize=8)
        fig.suptitle(f"{task}: normalized action-RMSE vs step — rmse-loss vs PG (awbc/reinf)", fontsize=12)
        fig.tight_layout()
        out = f"{BASE}/{task}_norm_rmse.png"
        fig.savefig(out, dpi=120); plt.close(fig)
        print(f"[norm-curves] wrote {out}")
    with open(f"{BASE}/_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["task", "policy", "split", "last_step", "bo16_norm"])
        w.writeheader(); [w.writerow(r) for r in summ]
    print(f"[norm-curves] summary -> {BASE}/_summary.csv ({len(summ)} rows)")


if __name__ == "__main__":
    main()
