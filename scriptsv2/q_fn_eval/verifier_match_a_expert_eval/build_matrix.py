#!/usr/bin/env python
"""Emit the finetuned_dqc_june24 results matrix (markdown) from the rescore
diagnostic/advantage CSVs + the progression CSVs. Pure stdlib (csv), so it runs
anywhere. Prints 'pending' for variant/step tags whose CSVs aren't there yet, so
it can be re-run as jobs land.
"""
import csv, os
from pathlib import Path

REPO = Path("/mmfs1/home/harine/RoboMonkey")
ADV_FT = REPO / "data/viz_q/_adv_ft"
DIAG_FT = REPO / "data/viz_q/_diag_ft"
ADV_BASE = REPO / "data/viz_q/_adv/advantage_base_val.csv"
DIAG_BASE = REPO / "data/viz/_diag/summary_val.csv"
PROG = Path("/gscratch/robotics/harine/RoboMonkey_data/viz_q/_prog")

VARIANTS = ["v1_ramp", "v2_all", "v3_dist"]
STEPS = [5000, 15000, 30000]
GROUPS = [("BC", ["bc_h8"]), ("Diffusion", ["diff10k", "diff33k"]), ("Gaussian", ["gauss20k", "gauss50k"])]


def _read(path):
    if not Path(path).exists():
        return None
    with open(path) as f:
        return {r["label"]: r for r in csv.DictReader(f)}


def _avg(rows, runs, key):
    vals = [float(rows[r][key]) for r in runs if r in rows and rows[r].get(key) not in (None, "", "nan")]
    return sum(vals) / len(vals) if vals else None


def fmt(x, nd=2):
    return "—" if x is None else f"{x:+.{nd}f}" if nd == 2 else f"{x:.{nd}f}"


def adv_csv(tag):   return _read(ADV_FT / f"advantage_{tag}_val.csv")
def diag_csv(tag):  return _read(DIAG_FT / tag / "summary_val.csv")


def group_cells(rows, fn):
    return " | ".join(fn(rows, runs) for _, runs in GROUPS)


print("### 1a. Per-step validation progression (aggregate over the 5 policy runs)\n")
print("`A`=mean Q(a\\*)−Q(deployed); `fracA`=frac A>0; `rmse`=BoN RMSE-to-a\\*. (from `progression_*.csv`)\n")
hdr = "| step | " + " | ".join(f"{v} A / fracA / rmse" for v in VARIANTS) + " |"
print(hdr); print("|" + "---|" * (len(VARIANTS) + 1))
prog = {}
for v in VARIANTS:
    p = PROG / f"progression_{v}.csv"
    if p.exists():
        with open(p) as f:
            prog[v] = {int(r["step"]): r for r in csv.DictReader(f)}
allsteps = sorted({s for v in prog for s in prog[v]})
for s in allsteps:
    if s < 0:
        label = "base"
    elif s == 0:
        continue
    else:
        label = f"{s//1000}k"
    cells = []
    for v in VARIANTS:
        r = prog.get(v, {}).get(s if s >= 0 else -1)
        if r:
            cells.append(f"{float(r['val_A_mean']):+.2f} / {float(r['val_fracA']):.2f} / {float(r['val_bon_rmse']):.3f}")
        else:
            cells.append("—")
    print(f"| {label} | " + " | ".join(cells) + " |")

print("\n### 1b. Advantage by policy group  (mean_A_sel / fracA), validation\n")
print("| variant @ step | " + " | ".join(f"{g} A / fracA" for g, _ in GROUPS) + " |")
print("|" + "---|" * (len(GROUPS) + 1))
b = adv_csv  # base
base = _read(ADV_BASE)
if base:
    print("| **base** | " + group_cells(base, lambda rs, ru: f"{_avg(rs,ru,'mean_A_sel'):+.2f} / {_avg(rs,ru,'frac_A_sel>0'):.2f}") + " |")
for v in VARIANTS:
    for s in STEPS:
        rows = adv_csv(f"{v}_s{s}")
        if rows:
            cell = group_cells(rows, lambda rs, ru: (f"{_avg(rs,ru,'mean_A_sel'):+.2f} / {_avg(rs,ru,'frac_A_sel>0'):.2f}"
                                                     if _avg(rs,ru,'mean_A_sel') is not None else "—"))
        else:
            cell = " | ".join(["_pending_"] * len(GROUPS))
        print(f"| {v} @ {s//1000}k | " + cell + " |")

print("\n### 1c. Candidate ranking by policy group  (spearman / concordance / top1), validation\n")
print("spearman(Q,RMSE) **negative=good**; concordance 0.5=chance; top1 chance≈0.06\n")
print("| variant @ step | " + " | ".join(f"{g} sp / conc / top1" for g, _ in GROUPS) + " |")
print("|" + "---|" * (len(GROUPS) + 1))
baseD = _read(DIAG_BASE)
def rank_cell(rs, ru):
    sp, co, t1 = _avg(rs,ru,'spearman_Q_rmse'), _avg(rs,ru,'mean_concordance'), _avg(rs,ru,'top1_match_frac')
    return "—" if sp is None else f"{sp:+.2f} / {co:.3f} / {t1:.2f}"
if baseD:
    print("| **base** | " + group_cells(baseD, rank_cell) + " |")
for v in VARIANTS:
    for s in STEPS:
        rows = diag_csv(f"{v}_s{s}")
        cell = group_cells(rows, rank_cell) if rows else " | ".join(["_pending_"] * len(GROUPS))
        print(f"| {v} @ {s//1000}k | " + cell + " |")
