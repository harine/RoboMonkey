"""Plot BoN success rate vs k for the improved-verifier sweep.

Reads `data/eval/bon/<run_name>/replan*_k*_score*_startseed*/eval_log.json`,
groups by seed, and produces a single line chart:

  - one line per start seed
  - one bolded line for the mean over seeds (95% CI via ±1.96 * SEM)
  - x-axis: log2-scale k
  - colorblind-friendly palette + distinct markers + distinct linestyles, so
    the plot is legible in grayscale and to dichromats

The chart is saved under data/eval/bon/_summaries/ with the current date,
checkpoint short name, and k range encoded in the filename so it doesn't get
overwritten by re-runs.

Usage:
  python scriptsv2/bon/plot_bon.py [--run-glob '<run_name_glob>'] \
                                    [--exclude-k 64] \
                                    [--out <png_path>]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Wong (2011) colorblind-safe 8-color palette. Sourced from
# https://www.nature.com/articles/nmeth.1618 — used widely for CB-safe plots.
WONG_PALETTE = [
    "#000000",  # black     (mean line)
    "#E69F00",  # orange    (seed 17)
    "#56B4E9",  # sky blue  (seed 38)
    "#009E73",  # bluish green (seed 99)
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#CC79A7",  # reddish purple
]
SEED_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
SEED_LINESTYLES = [(0, (3, 1)), (0, (1, 1)), (0, (5, 1, 1, 1)),
                   (0, (4, 2, 1, 2)), "dashdot", "dashed", "dotted"]


def _collect(run_glob: str) -> Tuple[str, Dict[str, Dict[int, dict]]]:
    """Return (run_name, {seed -> {k -> log_dict}}). Excludes 'none' seed."""
    matches = sorted(glob.glob(f"data/eval/bon/{run_glob}/"))
    if not matches:
        sys.exit(f"no run dirs match data/eval/bon/{run_glob}/")
    run_path = matches[0].rstrip("/")
    run_name = os.path.basename(run_path)
    out: Dict[str, Dict[int, dict]] = defaultdict(dict)
    cell_re = re.compile(r"replan(\d+)_k(\d+)_score(\d+)(?:_startseed(\d+))?")
    for f in glob.glob(f"{run_path}/*/eval_log.json"):
        m = cell_re.match(os.path.basename(os.path.dirname(f)))
        if not m:
            continue
        k = int(m.group(2))
        seed = m.group(4)
        if seed is None:  # legacy "no-seed-tag" cells; skip
            continue
        with open(f) as fh:
            out[seed][k] = json.load(fh)
    return run_name, out


def _bootstrap_ci(values: List[float], n_boot: int = 2000) -> Tuple[float, float]:
    """95% bootstrap CI of the mean. Used for the across-seed band."""
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    arr = np.asarray(values, dtype=float)
    boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-glob", default="16.45.01_train_diffusion_unet*",
                   help="glob matching the run dir under data/eval/bon/")
    p.add_argument("--exclude-k", type=int, nargs="*", default=[],
                   help="k values to drop (e.g. 64 if it OOM'd)")
    p.add_argument("--out", type=Path, default=None,
                   help="output PNG path (default: auto under data/eval/bon/_summaries/)")
    p.add_argument("--title-suffix", default="",
                   help="optional extra text appended to the plot title")
    args = p.parse_args(argv)

    run_name, data = _collect(args.run_glob)
    if not data:
        sys.exit(f"no cells found under {args.run_glob}")

    seeds = sorted(data.keys(), key=int)
    all_ks = sorted({k for s in data.values() for k in s})
    ks = [k for k in all_ks if k not in args.exclude_k]
    if not ks:
        sys.exit("no k values left after --exclude-k")

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=140)

    # Per-seed lines
    for i, seed in enumerate(seeds):
        xs, ys = [], []
        for k in ks:
            d = data[seed].get(k)
            if d is None:
                continue
            xs.append(k)
            ys.append(100.0 * d["success_rate"])
        if not xs:
            continue
        color = WONG_PALETTE[1 + i % (len(WONG_PALETTE) - 1)]
        marker = SEED_MARKERS[i % len(SEED_MARKERS)]
        ls = SEED_LINESTYLES[i % len(SEED_LINESTYLES)]
        ax.plot(xs, ys, color=color, linestyle=ls, marker=marker,
                ms=7, mfc=color, mec="black", mew=0.6, lw=1.4,
                label=f"seed {seed}")

    # Mean over seeds + 95% bootstrap CI
    mean_xs, mean_ys, ci_lo, ci_hi = [], [], [], []
    for k in ks:
        vals = [100.0 * data[s][k]["success_rate"]
                for s in seeds if k in data[s]]
        if len(vals) >= 1:
            mean_xs.append(k)
            mean_ys.append(np.mean(vals))
            lo, hi = _bootstrap_ci(vals) if len(vals) > 1 else (np.nan, np.nan)
            ci_lo.append(lo)
            ci_hi.append(hi)

    ax.plot(mean_xs, mean_ys, color=WONG_PALETTE[0], lw=2.4, marker="o",
            ms=8, mfc="white", mec="black", mew=1.4, label="mean")
    valid = ~np.isnan(ci_lo)
    if valid.any():
        ax.fill_between(np.asarray(mean_xs)[valid],
                        np.asarray(ci_lo)[valid],
                        np.asarray(ci_hi)[valid],
                        color=WONG_PALETTE[0], alpha=0.10,
                        label="95% CI (bootstrap)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("BoN k (log scale)")
    ax.set_ylabel("episode success rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
    ax.axhline(50, color="gray", lw=0.6, ls=":", zorder=0)

    short = run_name.split("_")[0]
    n_eps_set = {d["num_episodes"] for s in data.values() for d in s.values()}
    n_eps_str = (str(next(iter(n_eps_set))) if len(n_eps_set) == 1
                 else f"varies {sorted(n_eps_set)}")
    title = (f"BoN success rate vs k  —  improved verifier\n"
             f"run {short}   seeds={','.join(seeds)}   "
             f"episodes/cell={n_eps_str}")
    if args.title_suffix:
        title += f"\n{args.title_suffix}"
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.92)

    fig.tight_layout()

    if args.out is None:
        date = _dt.date.today().strftime("%Y%m%d")
        ks_tag = "k" + "-".join(str(k) for k in ks)
        seeds_tag = "seeds" + "-".join(seeds)
        out_dir = Path("data/eval/bon/_summaries")
        out_dir.mkdir(parents=True, exist_ok=True)
        args.out = out_dir / f"bon_sr_vs_k_{date}_{short}_{seeds_tag}_{ks_tag}.png"
    fig.savefig(args.out)
    plt.close(fig)

    print(f"[plot_bon] saved -> {args.out}")
    # Also drop a small text summary next to the PNG.
    txt = args.out.with_suffix(".txt")
    lines = [
        f"run        : {run_name}",
        f"seeds      : {','.join(seeds)}",
        f"ks         : {ks}",
        f"excluded_k : {args.exclude_k}",
        f"num_eps    : {n_eps_str}",
        "",
        "per-seed success rate (%):",
    ]
    header = "k  | " + "  ".join(f"seed_{s:>4}" for s in seeds) + "  | mean"
    lines.append(header)
    lines.append("-" * len(header))
    for k, mu in zip(mean_xs, mean_ys):
        row = f"{k:>3} | " + "  ".join(
            (f"{100*data[s][k]['success_rate']:>8.1f}%" if k in data[s] else f"{'-':>9}")
            for s in seeds
        ) + f"  | {mu:>6.1f}%"
        lines.append(row)
    txt.write_text("\n".join(lines) + "\n")
    print(f"[plot_bon] table -> {txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
