#!/usr/bin/env python3
"""Offline candidate metrics vs the autoregressive candidate index i.

Reads the npz written by scriptsv2/q_fn_eval/sample_offline_candidates.py:
  candidate_actions (R,K,8,7)  -- IN GENERATION ORDER (i = 0..K-1)
  values            (R,K)      -- DQC chunk Q of each candidate
  a_star (R,8,7), a_star_q (R,), a_star_rank (R,)

For each (task, kind, split, N) group, aggregated over every frame of every demo:
  * Q_i                  -- does the verifier value rise with iterations?
  * Q_i - Q(a*)          -- does the search out-score the EXPERT action? (>0 = verifier
                            prefers a candidate over a*, i.e. the Q is exploitable)
  * RMSE_i to a*         -- arm (dims 0:6), gripper (6:7), all -- do candidates converge?
  * diversity_i          -- mean pairwise cosine of the candidates' net dxyz over the
                            first i+1 candidates (1 = collapsed), and per-dim std
  * oracle_rmse_i        -- min RMSE among candidates 0..i   (best achievable BoN)
  * qpick_rmse_i         -- RMSE of argmax_Q among candidates 0..i (what BoN actually picks)
                            The gap qpick - oracle is how much the verifier's ranking costs.

SEARCH candidates are an autoregressive chain (a_i conditioned on a_<i, v_<i), so Q_i should
rise / RMSE_i fall. BC candidates are i.i.d. draws, so its curves MUST be flat -- that is both
the headline contrast and a check that generation order was preserved.

N>16 caveat: search's context saturates at max_actions-1=15 and then slides a fixed window
(search_policy_robomonkey.py:710-711,723-724), so past i=16 the x-axis means CHAIN DEPTH, not
more context. BC has no window.

  conda activate monkey-verifier
  python scriptsv2/analysis/offline_candidate_metrics.py
Out: data/eval/train_test_analysis/_metrics/{metrics.csv, summary.csv, *.png}
"""
from __future__ import annotations

import glob
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "data/eval/train_test_analysis"
ARM, GRIP, ALL = slice(0, 6), slice(6, 7), slice(0, 7)
CELL = re.compile(r"^(search|bc)_(train|test)_N(\d+)$")
ALL_TASKS = ["eggplant", "spoon", "stack_cube"]


def out_dir_for(task: str) -> str:
    """Per-task outputs live under the task's own folder."""
    return f"{ROOT}/{task}/metrics"


def _rmse(cand: np.ndarray, a_star: np.ndarray, dims) -> np.ndarray:
    """Per-candidate RMSE over `dims` of the chunk vs a* (NaN pad ignored).
    Reused verbatim from q_fn_eval/verifier_match_a_expert_eval/plot_offline_diagnostics.py."""
    diff = cand[..., dims].astype(np.float64) - a_star[None, ..., dims].astype(np.float64)
    mask = ~np.isnan(a_star[None, ..., dims])
    se = np.where(mask, diff ** 2, np.nan)
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.nanmean(se.reshape(se.shape[0], -1), axis=1))


def _frame_metrics(cand: np.ndarray, vals: np.ndarray, a_star: np.ndarray):
    """cand (K,8,7), vals (K,), a_star (8,7) -> dict of (K,) arrays indexed by i."""
    K = cand.shape[0]
    r_all = _rmse(cand, a_star, ALL)
    r_arm = _rmse(cand, a_star, ARM)
    r_grip = _rmse(cand, a_star, GRIP)

    # diversity: mean pairwise cosine of net dxyz direction among candidates 0..i
    net = np.nansum(cand[:, :, :3], axis=1)                      # (K,3)
    u = net / (np.linalg.norm(net, axis=1, keepdims=True) + 1e-9)
    C = u @ u.T
    div_cos = np.full(K, np.nan)
    for i in range(1, K):
        sub = C[:i + 1, :i + 1]
        div_cos[i] = (sub.sum() - np.trace(sub)) / (sub.shape[0] * (sub.shape[0] - 1))
    # per-dim std across candidates 0..i (mean over the 8x7 chunk entries)
    div_std = np.full(K, np.nan)
    flat = cand.reshape(K, -1)
    for i in range(1, K):
        div_std[i] = float(np.nanmean(np.nanstd(flat[:i + 1], axis=0)))

    # running best-of-first-(i+1)
    oracle = np.minimum.accumulate(r_all)
    qbest_idx = np.array([int(np.argmax(vals[:i + 1])) for i in range(K)])
    qpick = r_all[qbest_idx]
    qmax = np.maximum.accumulate(vals)
    return dict(q=vals, r_all=r_all, r_arm=r_arm, r_grip=r_grip,
                div_cos=div_cos, div_std=div_std,
                oracle_rmse=oracle, qpick_rmse=qpick, q_running_max=qmax)


def load_group(cell_dir: str):
    """Aggregate every frame of every episode npz in a cell -> per-i stacks."""
    acc = defaultdict(list)
    astar_q, astar_rank = [], []
    for f in sorted(glob.glob(f"{cell_dir}/search_q/*.npz")):
        try:
            d = np.load(f, allow_pickle=True)
        except Exception:
            continue
        if "candidate_actions" not in d.files:
            continue
        ca, vv, ast = d["candidate_actions"], d["values"], d["a_star"]
        aq = d["a_star_q"] if "a_star_q" in d.files else np.full(len(vv), np.nan)
        ar = d["a_star_rank"] if "a_star_rank" in d.files else np.full(len(vv), -1)
        # sanity: selected_index must be argmax(values)
        if "selected_index" in d.files:
            si = np.asarray(d["selected_index"])
            if not np.array_equal(si, np.argmax(vv, axis=1)):
                print(f"  !! {os.path.basename(f)}: selected_index != argmax(values)")
        for r in range(ca.shape[0]):
            m = _frame_metrics(ca[r], vv[r], ast[r])
            m["q_minus_astar"] = m["q"] - float(aq[r])
            for k, v in m.items():
                acc[k].append(v)
            astar_q.append(float(aq[r])); astar_rank.append(int(ar[r]))
    if not acc:
        return None
    out = {k: np.stack(v) for k, v in acc.items()}          # each (F, K)
    out["_astar_q"] = np.asarray(astar_q)
    out["_astar_rank"] = np.asarray(astar_rank)
    return out


STY = {("search", "train"): ("#2a9d8f", "-"), ("search", "test"): ("#2a9d8f", "--"),
       ("bc", "train"): ("#e76f51", "-"), ("bc", "test"): ("#e76f51", "--")}
PANELS = [("q", "Q of candidate i", "q_vs_index"),
          ("q_minus_astar", "Q_i - Q(a*)   (>0: verifier prefers candidate over expert)",
           "q_minus_astar_vs_index"),
          ("r_all", "RMSE(candidate i, a*)", "rmse_vs_index"),
          ("div_cos", "diversity: mean pairwise cos of net dxyz (1 = collapsed)",
           "diversity_vs_index")]

MET_HDR = ("kind,split,N,i,n_frames,q_mean,q_sd,q_minus_astar_mean,"
           "rmse_all_mean,rmse_arm_mean,rmse_grip_mean,div_cos_mean,div_std_mean,"
           "oracle_rmse_mean,qpick_rmse_mean\n")
SUM_HDR = ("kind,split,N,n_frames,astar_q_mean,astar_rank_mean,"
           "frac_astar_top1,frac_astar_top3,q_i0_mean,q_ilast_mean,q_gain,"
           "rmse_i0_mean,rmse_ilast_mean,rmse_drop\n")


def _met_row(g, i):
    F = g["q"].shape[0]
    return (f"{F},{np.nanmean(g['q'][:, i]):.5f},{np.nanstd(g['q'][:, i]):.5f},"
            f"{np.nanmean(g['q_minus_astar'][:, i]):.5f},"
            f"{np.nanmean(g['r_all'][:, i]):.6f},{np.nanmean(g['r_arm'][:, i]):.6f},"
            f"{np.nanmean(g['r_grip'][:, i]):.6f},"
            f"{np.nanmean(g['div_cos'][:, i]):.5f},{np.nanmean(g['div_std'][:, i]):.6f},"
            f"{np.nanmean(g['oracle_rmse'][:, i]):.6f},"
            f"{np.nanmean(g['qpick_rmse'][:, i]):.6f}")


def _sum_row(g):
    F = g["q"].shape[0]
    rk = g["_astar_rank"]
    q0, qL = np.nanmean(g["q"][:, 0]), np.nanmean(g["q"][:, -1])
    r0, rL = np.nanmean(g["r_all"][:, 0]), np.nanmean(g["r_all"][:, -1])
    return (f"{F},{np.nanmean(g['_astar_q']):.5f},{rk.mean():.3f},"
            f"{(rk == 0).mean():.4f},{(rk < 3).mean():.4f},"
            f"{q0:.5f},{qL:.5f},{qL - q0:+.5f},{r0:.6f},{rL:.6f},{rL - r0:+.6f}")


def analyze_task(task: str):
    """Load every cell for one task and write its CSVs + figures under <task>/metrics/."""
    groups = {}
    for cd in sorted(glob.glob(f"{ROOT}/{task}/offline/*")):
        m = CELL.match(os.path.basename(cd))
        if not m:
            continue
        kind, split, N = m.group(1), m.group(2), int(m.group(3))
        g = load_group(cd)
        if g is None:
            continue
        groups[(kind, split, N)] = g
        print(f"[{task}] {kind:6s} {split:5s} N={N:<3d} frames={g['q'].shape[0]:4d} "
              f"K={g['q'].shape[1]}")
    if not groups:
        print(f"[{task}] no cells yet"); return None

    out = out_dir_for(task)
    os.makedirs(out, exist_ok=True)

    with open(f"{out}/metrics.csv", "w") as fh:
        fh.write(MET_HDR)
        for (kind, split, N), g in sorted(groups.items()):
            for i in range(g["q"].shape[1]):
                fh.write(f"{kind},{split},{N},{i},{_met_row(g, i)}\n")
    with open(f"{out}/summary.csv", "w") as fh:
        fh.write(SUM_HDR)
        for (kind, split, N), g in sorted(groups.items()):
            fh.write(f"{kind},{split},{N},{_sum_row(g)}\n")

    NS = sorted({k[2] for k in groups})
    SPLITS = [s for s in ("train", "test") if any(k[1] == s for k in groups)]

    for key, ylab, fname in PANELS:
        fig, axes = plt.subplots(len(SPLITS), len(NS),
                                 figsize=(4.3 * len(NS), 3.4 * len(SPLITS)), squeeze=False)
        for r, split in enumerate(SPLITS):
            for c, N in enumerate(NS):
                ax = axes[r][c]
                for kind in ("search", "bc"):
                    g = groups.get((kind, split, N))
                    if g is None:
                        continue
                    col, ls = STY[(kind, split)]
                    y = np.nanmean(g[key], axis=0)
                    ax.plot(np.arange(len(y)), y, "-", color=col, lw=1.7, label=kind)
                if key == "q":
                    g = groups.get(("search", split, N)) or groups.get(("bc", split, N))
                    if g is not None:
                        ax.axhline(np.nanmean(g["_astar_q"]), color="k", ls=":", lw=1.2,
                                   label="Q(a*)")
                if N > 16:
                    ax.axvline(15.5, color="gray", ls=":", lw=1)
                    ax.text(16, ax.get_ylim()[1], " window slides", fontsize=6,
                            va="top", color="gray")
                ax.set_title(f"{split}  N={N}", fontsize=9)
                ax.grid(alpha=.3)
                if r == len(SPLITS) - 1:
                    ax.set_xlabel("candidate index i")
                if c == 0:
                    ax.set_ylabel(ylab, fontsize=7)
                if r == 0 and c == 0:
                    ax.legend(fontsize=6)
        fig.suptitle(f"{task} — {ylab}\nsearch = autoregressive chain, BC = i.i.d. draws "
                     f"(BC must be FLAT)", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(f"{out}/{fname}.png", dpi=115, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(len(SPLITS), len(NS),
                             figsize=(4.3 * len(NS), 3.4 * len(SPLITS)), squeeze=False)
    for r, split in enumerate(SPLITS):
        for c, N in enumerate(NS):
            ax = axes[r][c]
            for kind in ("search", "bc"):
                g = groups.get((kind, split, N))
                if g is None:
                    continue
                col, _ = STY[(kind, split)]
                ax.plot(np.nanmean(g["oracle_rmse"], axis=0), "--", color=col, lw=1.3,
                        alpha=.6, label=f"{kind} oracle")
                ax.plot(np.nanmean(g["qpick_rmse"], axis=0), "-", color=col, lw=1.8,
                        label=f"{kind} Q-pick")
            ax.set_title(f"{split}  N={N}", fontsize=9); ax.grid(alpha=.3)
            if r == len(SPLITS) - 1:
                ax.set_xlabel("candidates considered (i+1)")
            if c == 0:
                ax.set_ylabel("RMSE to a*", fontsize=7)
            if r == 0 and c == 0:
                ax.legend(fontsize=6)
    fig.suptitle(f"{task} — Best-of-(i+1): oracle (min RMSE, dashed) vs what Q picks (solid).\n"
                 f"Gap = cost of ranking by Q instead of by distance to a*.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{out}/oracle_vs_qpick.png", dpi=115, bbox_inches="tight"); plt.close(fig)

    print(f"[{task}] -> {out}/ (metrics.csv, summary.csv, 5 figs)")
    return groups


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=ALL_TASKS)
    a = ap.parse_args()

    all_groups = {}
    for task in a.tasks:
        g = analyze_task(task)
        if g:
            for k, v in g.items():
                all_groups[(task, *k)] = v

    if not all_groups:
        print("no groups found yet"); return

    # cross-task index (one row per group) for quick comparison
    idx = f"{ROOT}/_metrics"
    os.makedirs(idx, exist_ok=True)
    with open(f"{idx}/all_summary.csv", "w") as fh:
        fh.write("task," + SUM_HDR)
        for (task, kind, split, N), g in sorted(all_groups.items()):
            fh.write(f"{task},{kind},{split},{N},{_sum_row(g)}\n")
    print(f"[metrics] cross-task index -> {idx}/all_summary.csv")


if __name__ == "__main__":
    main()
