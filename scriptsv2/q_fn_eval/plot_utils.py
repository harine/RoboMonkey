"""plot_utils.py — jax-free plotting shared by eval_q / eval_dqc / eval_robomonkey.

Kept separate from eval_q.py (which imports JAX) so the monkey-verifier env —
which has no jax — can still produce the combined comparison plot.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability a random success ranks above a random failure (0.5=chance)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def make_plot(df, out_path: Path, title: str, source: str) -> None:
    """Two rows:
      top    — per-step metric over the episode (mean±std time-series)
      bottom — per-EPISODE mean of the metric, success vs failure, with AUC.
               This bottom panel is what reveals discrimination: a small mean
               shift swamped by per-step std in the time-series shows up here
               as a separation between two distributions (AUC > 0.5).
    Any metric column that is entirely NaN is skipped, so the same function
    renders DQC-only, verifier-only, or the combined comparison.
    """
    import pandas as pd

    df = df.copy()
    metrics = []
    if "q_value" in df and not df["q_value"].isna().all():
        metrics.append(("q_value", "DQC Q value (min ensemble)"))
    if "verifier_score" in df and not df["verifier_score"].isna().all():
        metrics.append(("verifier_score", "RoboMonkey verifier score"))

    ncol = len(metrics)
    fig, axes = plt.subplots(2, ncol, figsize=(7 * ncol, 9), squeeze=False)

    if source == "zarr":
        ep_lens = df.groupby("ep_idx")["branch_t"].transform("max").clip(lower=1)
        df["x"] = df["branch_t"] / ep_lens
        x_label = "Episode progress (0=start, 1=end)"
        bins = np.linspace(0.0, 1.0, 21)
        mids = (bins[:-1] + bins[1:]) / 2
        df["x_bin"] = pd.cut(df["x"], bins=bins, labels=mids, include_lowest=True).astype(float)
        grp_col = "x_bin"
    else:
        df["x"] = df["branch_t"]
        x_label = "Env timestep at replan"
        grp_col = "branch_t"

    classes = [(True, "steelblue", "success"), (False, "tomato", "failure")]

    for j, (metric, ylabel) in enumerate(metrics):
        # ── top: time-series ──
        ax = axes[0][j]
        for success, color, name in classes:
            sub = df[df["success"] == success]
            if sub.empty:
                continue
            for ep_id in sub["ep_idx"].unique():
                er = sub[sub["ep_idx"] == ep_id].sort_values("x")
                ax.plot(er["x"], er[metric], color=color, alpha=0.12, linewidth=0.6)
            g = sub.groupby(grp_col)[metric]
            means = g.mean().dropna(); stds = g.std().reindex(means.index).fillna(0)
            ax.plot(means.index, means.values, color=color, linewidth=2.5, label=name)
            ax.fill_between(means.index, means - stds, means + stds, color=color, alpha=0.2)
        ax.set_xlabel(x_label); ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}\nper-step over episode (±std)")
        ax.legend(); ax.grid(alpha=0.3)

        # ── bottom: per-episode distribution + AUC ──
        ax = axes[1][j]
        per_ep = df.groupby("ep_idx").agg(success=("success", "first"),
                                          val=(metric, "mean"))
        pos = per_ep[per_ep["success"]]["val"].values
        neg = per_ep[~per_ep["success"]]["val"].values
        auc = _auc(pos, neg)
        for xpos, vals, color, name in [(0, pos, "steelblue", "success"),
                                        (1, neg, "tomato", "failure")]:
            if len(vals) == 0:
                continue
            jitter = (np.linspace(-0.18, 0.18, len(vals)) if len(vals) > 1 else [0.0])
            ax.scatter(xpos + np.asarray(jitter), vals, color=color, alpha=0.6,
                       s=28, edgecolor="k", linewidth=0.3, zorder=3)
            ax.hlines(vals.mean(), xpos - 0.28, xpos + 0.28,
                      color=color, linewidth=3, zorder=4)
            ax.text(xpos, vals.mean(), f"  μ={vals.mean():.3f}", va="center", fontsize=9)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["success", "failure"])
        ax.set_ylabel(f"per-episode mean {metric}")
        ax.set_title(f"per-episode separation   AUC={auc:.3f}\n(0.5=chance, 1.0=perfect)")
        ax.grid(alpha=0.3, axis="y")
        print(f"[auc] {metric}: AUC={auc:.3f}  "
              f"(success μ={pos.mean():.3f}, failure μ={neg.mean():.3f})")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[out] plot → {out_path}")
