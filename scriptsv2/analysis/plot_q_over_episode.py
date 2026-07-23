#!/usr/bin/env python3
"""Q over the episode: sampled candidates + chosen, augmented (PG) vs not (rmse-loss).

One figure per (task, seed) with four panels:
  [rmse-loss] [awbc] [reinforce]  — each: all N=16 candidate Q as points + min–max band +
                                    mean (dashed) + the CHOSEN (argmax-Q) trace in bold
  [overlay]                       — the three chosen-Q traces on one axis for direct comparison

All three are the same eval seeds (1000–1009), N=16, MODE=argmax, same frozen DQC chunk critic,
so the Q axis is directly comparable.

  augmented (PG) : data/eval/pg_det/pg_{awbc,reinf}_l0.1_<task>_sub50/<CK>/viz_n16/search_q/*.npz
  not augmented  : data/eval/<taskdir>/_viz_argmax/<prefix>_n16/search_q/*.npz

Checkpoint caveat: the rmse-loss viz is at its LATE checkpoint (eggplant/spoon 90k, stack 60k);
the PG runs are at 10k/20k (still training). Stated on every figure.

Out: data/eval/q_over_episode/<task>_seed<seed>_N16_pg<CK>.png
Usage: python scriptsv2/analysis/plot_q_over_episode.py [--ck 20k]
"""
import argparse, glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "data/eval"
OUT = f"{ROOT}/q_over_episode"
# task -> (rmse viz dir glob, pg task token, rmse ckpt label)
TASKS = {
    "eggplant": (f"{ROOT}/eggplant/_viz_argmax/eggplant_90_st120_n16", "eggplant", "90k"),
    "spoon":    (f"{ROOT}/spoon_on_towel/_viz_argmax/spoon_on_towel_90_st60_n16", "spoon", "90k"),
    "stack":    (f"{ROOT}/stack_cube/_viz_argmax/stack_cube_60_st60_n16", "stack", "60k"),
}
COL = {"rmse-loss": "#444444", "awbc": "#2e86ab", "reinforce": "#d1495b"}


def load(npz):
    d = np.load(npz, allow_pickle=True)
    bt = np.asarray(d["branch_t"], float)
    vals = np.asarray(d["values"], float)          # (R, N)
    sel = np.asarray(d["selected_value"], float)   # (R,)
    succ, trunc = int(d["success"]), int(d["truncated"])
    status = "SUCCESS" if succ else ("TRUNC" if trunc else "FAIL")
    return bt, vals, sel, status, int(d["num_steps"])


def find_seed(dirpath, seed):
    hits = glob.glob(f"{dirpath}/search_q/*_seed{seed}.npz")
    return hits[0] if hits else None


def panel(ax, bt, vals, sel, label, status, nsteps, color):
    # every sampled candidate as a faint point at its replan timestep
    for r in range(vals.shape[0]):
        ax.plot(np.full(vals.shape[1], bt[r]), vals[r], ".", ms=3,
                color=color, alpha=.28, zorder=1)
    ax.fill_between(bt, vals.min(1), vals.max(1), color=color, alpha=.12, zorder=0,
                    label="candidate min–max")
    ax.plot(bt, vals.mean(1), "--", lw=1.1, color=color, alpha=.75, label="candidate mean")
    ax.plot(bt, sel, "-o", ms=4, lw=2.0, color=color, label="chosen (argmax-Q)", zorder=3)
    ax.set_title(f"{label}  ·  {status} ({nsteps} steps)", fontsize=9)
    ax.grid(alpha=.3); ax.set_xlabel("env timestep")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", default="20k", help="PG checkpoint (10k|20k)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1000, 1010)))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    n = 0
    for task, (rmse_dir, pgtok, rck) in TASKS.items():
        pg = {"awbc": f"{ROOT}/pg_det/pg_awbc_l0.1_{pgtok}_sub50/{a.ck}/viz_n16",
              "reinforce": f"{ROOT}/pg_det/pg_reinf_l0.1_{pgtok}_sub50/{a.ck}/viz_n16"}
        for seed in a.seeds:
            series = {}
            f = find_seed(rmse_dir, seed)
            if f: series["rmse-loss"] = load(f)
            for k, d in pg.items():
                f = find_seed(d, seed)
                if f: series[k] = load(f)
            if len(series) < 2:
                continue
            fig, axes = plt.subplots(1, 4, figsize=(19, 4.1))
            for i, key in enumerate(["rmse-loss", "awbc", "reinforce"]):
                ax = axes[i]
                if key not in series:
                    ax.set_title(f"{key} — n/a", fontsize=9); ax.grid(alpha=.3); continue
                bt, vals, sel, st, ns = series[key]
                lab = f"{key} ({rck})" if key == "rmse-loss" else f"{key} ({a.ck})"
                panel(ax, bt, vals, sel, lab, st, ns, COL[key])
                if i == 0:
                    ax.set_ylabel("verifier Q"); ax.legend(fontsize=7, loc="lower right")
            ax = axes[3]
            for key, (bt, vals, sel, st, ns) in series.items():
                ax.plot(bt, sel, "-o", ms=4, lw=2.0, color=COL[key], label=f"{key} · {st}")
            ax.set_title("chosen-Q overlay", fontsize=9); ax.grid(alpha=.3)
            ax.set_xlabel("env timestep"); ax.legend(fontsize=7, loc="lower right")
            fig.suptitle(
                f"{task} · seed {seed} · N=16 argmax · Q over episode — augmented (PG) vs not "
                f"(rmse-loss @{rck}; PG @{a.ck})", fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            out = f"{OUT}/{task}_seed{seed}_N16_pg{a.ck}.png"
            fig.savefig(out, dpi=115); plt.close(fig)
            print(f"  wrote {out}"); n += 1
    print(f"[q_over_episode] {n} figures -> {OUT}/")


if __name__ == "__main__":
    main()
