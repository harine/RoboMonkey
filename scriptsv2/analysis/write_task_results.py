#!/usr/bin/env python3
"""Generate a per-task RESULTS.md from that task's metrics CSVs.

Reads  data/eval/train_test_analysis/<task>/metrics/{metrics.csv,summary.csv}
Writes data/eval/train_test_analysis/<task>/RESULTS.md

Answers, for one task, the three questions:
  1. Q of the candidates across the autoregressive index i -- do they rise? vs Q(a*)?
  2. Action RMSE of the candidates vs a* -- do they fall? diversity?
  3. search vs BC on both.
Plus the oracle-vs-Q-pick gap (what ranking by Q costs) and the video inventory.

  conda activate monkey-verifier
  python scriptsv2/analysis/write_task_results.py            # all tasks
  python scriptsv2/analysis/write_task_results.py --tasks spoon
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict

ROOT = "data/eval/train_test_analysis"
TASKS = ["eggplant", "spoon", "stack_cube"]
WANDB = {
    "eggplant": ("dqc_search_policy / diff_search_rmse_sub50_diverse_data (l1x3fvwe)",
                 "BC_eggplant / unet_denoiser_pred8_roll4_50k_sub50_diverse_nobs1"),
    "spoon": ("dqc_search_policy / diff_search_rmse_spoon_sub50_diverse (uyxvs4f4)",
              "BC_spoon / unet_denoiser_pred8_roll4_50k_spoon_sub50_nobs1"),
    "stack_cube": ("dqc_search_policy / diff_search_rmse_stack_sub50_diverse (ns0n3xv0)",
                   "BC_stack / unet_denoiser_pred8_roll4_50k_stack_sub50_nobs1"),
}


def load(task):
    md = f"{ROOT}/{task}/metrics"
    if not os.path.exists(f"{md}/summary.csv"):
        return None, None
    per_i = defaultdict(dict)
    for r in csv.DictReader(open(f"{md}/metrics.csv")):
        per_i[(r["kind"], r["split"], int(r["N"]))][int(r["i"])] = r
    summ = {(r["kind"], r["split"], int(r["N"])): r
            for r in csv.DictReader(open(f"{md}/summary.csv"))}
    return per_i, summ


def fmt(v, p=4):
    try:
        return f"{float(v):.{p}f}"
    except Exception:
        return "-"


def section_q(per_i, summ, Ns):
    L = ["| N | policy | split | Q: i0 -> iL | dQ | Q(a*) | a* rank | a* top-1 |",
         "|---|---|---|---|---|---|---|---|"]
    for N in Ns:
        for split in ("train", "test"):
            for kind in ("search", "bc"):
                k = (kind, split, N)
                if k not in summ:
                    continue
                s = summ[k]
                q0, qL = float(s["q_i0_mean"]), float(s["q_ilast_mean"])
                L.append(f"| {N} | {'**search**' if kind=='search' else 'BC'} | {split} | "
                         f"{q0:.2f} -> {qL:.2f} | **{qL-q0:+.2f}** | {fmt(s['astar_q_mean'],2)} | "
                         f"{fmt(s['astar_rank_mean'],2)} | {fmt(s['frac_astar_top1'],2)} |")
    return L


def section_rmse(summ, Ns):
    L = ["| N | policy | split | RMSE: i0 -> iL | dRMSE |",
         "|---|---|---|---|---|"]
    for N in Ns:
        for split in ("train", "test"):
            for kind in ("search", "bc"):
                k = (kind, split, N)
                if k not in summ:
                    continue
                s = summ[k]
                r0, rL = float(s["rmse_i0_mean"]), float(s["rmse_ilast_mean"])
                L.append(f"| {N} | {'**search**' if kind=='search' else 'BC'} | {split} | "
                         f"{r0:.4f} -> {rL:.4f} | **{rL-r0:+.4f}** |")
    return L


def section_div(per_i, Ns):
    L = ["| N | policy | split | div @ i=1 | div @ i=last | oracle RMSE | Q-pick RMSE | cost of Q |",
         "|---|---|---|---|---|---|---|---|"]
    for N in Ns:
        if N < 2:
            continue
        for split in ("train", "test"):
            for kind in ("search", "bc"):
                d = per_i.get((kind, split, N))
                if not d:
                    continue
                last = max(d)
                d1 = float(d[1]["div_cos_mean"]) if 1 in d else float("nan")
                dl = float(d[last]["div_cos_mean"])
                orc, qp = float(d[last]["oracle_rmse_mean"]), float(d[last]["qpick_rmse_mean"])
                L.append(f"| {N} | {'**search**' if kind=='search' else 'BC'} | {split} | "
                         f"{d1:.3f} | {dl:.3f} | {orc:.4f} | {qp:.4f} | **{qp-orc:+.4f}** |")
    return L


def videos(task):
    """All videos for a task live in ONE folder, <task>/videos/, with the split in the
    filename: <task>/videos/<split>_seed<seed>_N<N>_<status>.mp4"""
    rows = []
    for split in ("train", "test"):
        vids = sorted(glob.glob(f"{ROOT}/{task}/videos/{split}_seed*.mp4"))
        if vids:
            s = sum("succ" in os.path.basename(v) for v in vids)
            rows.append(f"| {split} | {len(vids)} | {s} succ / {len(vids)-s} trunc | "
                        f"`{task}/videos/{split}_seed*_N*_*.mp4` |")
        else:
            rows.append(f"| {split} | 0 | - | *(pending)* |")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    a = ap.parse_args()

    for task in a.tasks:
        per_i, summ = load(task)
        if summ is None:
            print(f"[results] {task}: no metrics yet, skipped")
            continue
        Ns = sorted({k[2] for k in summ})
        cells = len(summ)
        sw, bw = WANDB.get(task, ("?", "?"))

        L = [f"# {task} — search vs BC on train vs test demos", "",
             f"Auto-generated by `scriptsv2/analysis/write_task_results.py`. "
             f"Methodology + caveats: `../NOTES.md`.", "",
             f"- **Cells present:** {cells}/16 (search|bc x train|test x N in 1,4,16,64)",
             f"- **Search policy (wandb):** `{sw}`",
             f"- **BC policy (wandb):** `{bw}`",
             f"- **Figures:** `{task}/metrics/*.png`   **Raw:** `{task}/metrics/{{metrics,summary}}.csv`",
             "",
             "> `train` = the 50 demos this policy trained on. `test` = strict-success demos not in "
             "sub100 (⊇ sub50) — unseen, same distribution and same filter.",
             "> `i` is the **autoregressive candidate index**. Search conditions `a_i` on "
             "`(a_<i, v_<i)`; **BC's draws are i.i.d., so its curves must be flat** (that is the "
             "control, not a finding). For `N=64`, context saturates at 15 and then slides — past "
             "`i=16` the axis means *chain depth*, not more context.",
             "",
             "## 1. Q of the candidates (does Q rise? how vs Q(a*)?)", ""]
        L += section_q(per_i, summ, Ns)
        L += ["", "## 2. Action RMSE to the expert a* (do candidates converge?)", ""]
        L += section_rmse(summ, Ns)
        L += ["", "## 3. Diversity, and what ranking by Q costs", "",
              "`div` = mean pairwise cosine of the candidates' net Δxyz (**1.0 = collapsed**). "
              "`oracle` = min RMSE among the candidates; `Q-pick` = RMSE of `argmax Q`. "
              "Their gap is the cost of ranking by Q rather than by distance to `a*`.", ""]
        L += section_div(per_i, Ns)
        L += ["", "## 4. Closed-loop videos (chosen action vs expert trajectory)", "",
              "| split | videos | outcomes | path |", "|---|---|---|---|"]
        L += videos(task)
        L += ["", "Each frame: N=16 candidate fan colored by Q rank, **chosen (argmax-Q) in red**, "
              "**demo expert path dashed white**, policy path cyan, running divergence in cm. "
              "Rollouts are reset at each demo's own seed.", ""]

        out = f"{ROOT}/{task}/RESULTS.md"
        open(out, "w").write("\n".join(L) + "\n")
        print(f"[results] wrote {out}  ({cells}/16 cells)")


if __name__ == "__main__":
    main()
