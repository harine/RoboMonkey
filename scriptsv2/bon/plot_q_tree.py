"""Visualize BoN candidate Q-values from one or more bon_q/*.npz files.

Each .npz (one per episode) stores, at every replan point t:
  - candidate_actions       [n_replans, k, horizon, action_dim]
  - per_candidate_rewards   [n_replans, k, score_num_actions]
  - per_candidate_mean_reward [n_replans, k]      <- the score the policy uses
  - selected_index          [n_replans]
  - selected_reward         [n_replans]
  - frames                  [n_replans, H, W, 3]  (for context, unused here)

The plot is a "branching tree" over replan steps:
  x = replan index (or env step t)
  y = verifier reward (per_candidate_mean_reward)
  - light gray dots: every candidate
  - red line       : the chosen candidate's reward at each replan
  - shaded band    : per-replan min..max range
  - secondary axis : spike score = max - min across candidates per replan

Usage:
  python scriptsv2/bon/plot_q_tree.py <npz_path>                 # one episode
  python scriptsv2/bon/plot_q_tree.py <cell_dir>                 # all eps in bon_q/
  python scriptsv2/bon/plot_q_tree.py <npz_or_dir> --out PATH    # custom save loc
  python scriptsv2/bon/plot_q_tree.py <cell_dir> --top N         # only top-N spike eps
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(npz_path: Path) -> dict:
    with np.load(npz_path, allow_pickle=False) as z:
        return {
            "seed": int(z["seed"]),
            "ep_idx": int(z["ep_idx"]),
            "success": bool(z["success"]),
            "truncated": bool(z["truncated"]),
            "num_steps": int(z["num_steps"]),
            "bon_k": int(z["bon_k"]),
            "branch_t": z["branch_t"].astype(np.int32),
            "Q": z["per_candidate_mean_reward"].astype(np.float32),
            "Q_per_action": z["per_candidate_rewards"].astype(np.float32),
            "selected_index": z["selected_index"].astype(np.int32),
            "selected_reward": z["selected_reward"].astype(np.float32),
        }


def _spike_score(Q: np.ndarray) -> np.ndarray:
    """max - min across the k candidates at each replan (range = spike-iness)."""
    return Q.max(axis=1) - Q.min(axis=1)


def plot_one(npz_path: Path, out_path: Path) -> dict:
    d = _load(npz_path)
    Q = d["Q"]                       # [R, k]
    if Q.size == 0:
        print(f"[skip] {npz_path.name}: no branches")
        return {}
    R, k = Q.shape
    sel = d["selected_index"]        # [R]
    sel_reward = d["selected_reward"]  # [R]
    ts = d["branch_t"]               # [R] env-step indices at each replan

    fig, ax = plt.subplots(figsize=(max(7, R * 0.18), 4.5), dpi=130)

    # all candidates as scatter, color-coded by rank within each replan
    for r in range(R):
        x = np.full(k, ts[r])
        ax.scatter(x, Q[r], s=10, alpha=0.35, color="gray", linewidths=0)
    # min..max envelope
    ax.fill_between(ts, Q.min(axis=1), Q.max(axis=1),
                    alpha=0.10, color="steelblue", label="min..max")
    # mean across candidates
    ax.plot(ts, Q.mean(axis=1), color="steelblue", lw=1.2, alpha=0.7,
            label="mean(Q)")
    # chosen candidate
    ax.plot(ts, sel_reward, color="crimson", lw=1.5, marker="o",
            ms=4, label="chosen")

    ax.set_xlabel("env step t (replan points)")
    ax.set_ylabel("verifier reward (mean over first score-actions)")
    status = ("success" if d["success"]
              else ("truncated" if d["truncated"] else "fail"))
    title = (f"ep{d['ep_idx']:03d}  seed={d['seed']}  k={d['bon_k']}  "
             f"steps={d['num_steps']}  -> {status}")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, framealpha=0.85)

    # right axis: spike score
    spike = _spike_score(Q)
    ax2 = ax.twinx()
    ax2.plot(ts, spike, color="goldenrod", lw=1.0, alpha=0.7,
             ls="--", label="spike (max-min)")
    ax2.set_ylabel("spike (max-min)", color="goldenrod")
    ax2.tick_params(axis="y", colors="goldenrod")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    print(f"[ok]  {npz_path.name} -> {out_path}  "
          f"R={R} k={k} spike_max={spike.max():.3f} "
          f"chosen_was_top1={np.mean(Q.argmax(axis=1) == sel):.2f}")
    return {
        "path": str(npz_path),
        "out": str(out_path),
        "R": R, "k": k,
        "success": d["success"],
        "spike_max": float(spike.max()),
        "spike_mean": float(spike.mean()),
        "chosen_top1_rate": float(np.mean(Q.argmax(axis=1) == sel)),
        "Q_global_max": float(Q.max()),
        "Q_global_min": float(Q.min()),
    }


def iter_npz(target: Path) -> Iterable[Path]:
    if target.is_file() and target.suffix == ".npz":
        yield target
        return
    if target.is_dir():
        # accept either a cell dir (has bon_q/) or the bon_q dir itself
        cand = target / "bon_q"
        root = cand if cand.is_dir() else target
        yield from sorted(root.glob("ep*_seed*.npz"))


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("target", type=Path,
                   help="single .npz, a cell dir, or a bon_q/ dir")
    p.add_argument("--out", type=Path, default=None,
                   help="output PNG (single) or output dir (multi). "
                        "Default: <bon_q>/_plots/")
    p.add_argument("--top", type=int, default=0,
                   help="when plotting a dir, only the top-N eps by spike_max")
    args = p.parse_args(argv)

    files = list(iter_npz(args.target))
    if not files:
        print(f"no .npz files found under {args.target}", file=sys.stderr)
        return 2

    if len(files) == 1:
        f = files[0]
        out = args.out if (args.out and args.out.suffix == ".png") \
            else (args.out or f.with_suffix(".png").parent / "_plots" / f.with_suffix(".png").name)
        if args.out and args.out.is_dir():
            out = args.out / f.with_suffix(".png").name
        plot_one(f, out)
        return 0

    out_dir = args.out if args.out else (files[0].parent / "_plots")
    summaries = []
    for f in files:
        try:
            s = plot_one(f, out_dir / f.with_suffix(".png").name)
            if s:
                summaries.append(s)
        except Exception as e:
            print(f"[err] {f.name}: {e!r}", file=sys.stderr)

    if args.top and summaries:
        summaries.sort(key=lambda s: s["spike_max"], reverse=True)
        keep = {s["path"] for s in summaries[: args.top]}
        for s in summaries[args.top:]:
            png = Path(s["out"])
            if png.exists():
                png.unlink()
        print(f"[top] kept {min(args.top, len(summaries))} highest-spike plots")

    if summaries:
        avg_top1 = np.mean([s["chosen_top1_rate"] for s in summaries])
        sr = np.mean([1.0 if s["success"] else 0.0 for s in summaries])
        print(f"\n[summary] eps={len(summaries)}  success_rate={sr:.2f}  "
              f"chosen=argmax(Q) rate={avg_top1:.2f}  "
              f"mean_spike_max={np.mean([s['spike_max'] for s in summaries]):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
