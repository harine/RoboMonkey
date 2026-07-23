#!/usr/bin/env python
"""filtered_plot_comp.py — verifier discrimination across the 3 episode groups.

Reads ``dqc_scores_filtered.csv`` (filtered_score_dqc.py) + ``verifier_scores.npz``
(eval_robomonkey.py, row-aligned to pairs.npz) and asks, for each verifier:

  how well does it separate
    filtered_success (group 0)  vs  other_success (1)  vs  failure (2) ?

Scores are aggregated to a per-EPISODE mean (the deployed quantity), then we
report, per verifier, four ranking AUCs (P(left-group ep ranks above right)):

  filt_succ vs failure     does the verifier rank the cleanest successes > failures?
  other_succ vs failure    ... and the messy successes > failures?
  filt_succ vs other_succ  does the verifier think filtered > messy successes?
  all_succ  vs failure     the plain success-vs-failure baseline (== before filtering)

Outputs (in --out-dir): verifier_comp.png, verifier_comp_auc.md, verifier_comp.csv.
Runs in any env with pandas + matplotlib (no jax). Usage: run_filtered_verifier_comp.sbatch.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GROUP_NAMES = {0: "filtered_success", 1: "other_success", 2: "failure"}
GROUP_COLOR = {0: "seagreen", 1: "goldenrod", 2: "tomato"}
# Headline 3 first (chunk-8 vs chunk-1 vs LLaVA), action critic as supplement.
METRICS = [
    ("q_chunk8", "DQC chunk-8 Q\n(chunk critic, 8-step)"),
    ("q_chunk1", "DQC chunk-1 Q\n(chunk critic, 1-step)"),
    ("verifier_score", "LLaVA (RoboMonkey)\nverifier"),
    ("q_action", "DQC single-step Q\n(action critic)"),
]


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)

    df = pd.read_csv(out / "dqc_scores_filtered.csv")     # row, ep_uid, group, success, ...
    vfile = out / "verifier_scores.npz"
    if vfile.exists():
        vs = np.load(vfile, allow_pickle=True)["verifier_score"]
        if len(vs) != len(df):
            raise SystemExit(f"verifier_scores ({len(vs)}) != dqc rows ({len(df)}) — re-run on same pairs.npz")
        df["verifier_score"] = np.asarray(vs)[df["row"].values]
        print(f"[plot] merged LLaVA verifier_score ({len(vs)} rows)")
    else:
        print(f"[plot] {vfile} missing — DQC-only (run eval_robomonkey.py for LLaVA).")

    df.to_csv(out / "verifier_comp.csv", index=False)
    metrics = [(c, l) for c, l in METRICS if c in df and not df[c].isna().all()]

    # ---- per-episode means -------------------------------------------------
    agg = {c: (c, "mean") for c, _ in metrics}
    per_ep = df.groupby("ep_uid").agg(group=("group", "first"), **agg).reset_index()
    g0 = per_ep[per_ep.group == 0]; g1 = per_ep[per_ep.group == 1]; g2 = per_ep[per_ep.group == 2]
    n0, n1, n2 = len(g0), len(g1), len(g2)
    print(f"[plot] per-episode: filtered_success={n0}, other_success={n1}, failure={n2}")

    # ---- AUC table ---------------------------------------------------------
    PAIRS = [
        ("filt_succ_vs_failure", 0, 2),
        ("other_succ_vs_failure", 1, 2),
        ("filt_succ_vs_other_succ", 0, 1),
    ]
    rows = []
    for col, label in metrics:
        rec = {"verifier": label.replace("\n", " ")}
        for name, ga, gb in PAIRS:
            pos = per_ep[per_ep.group == ga][col].values
            neg = per_ep[per_ep.group == gb][col].values
            rec[name] = auc(pos, neg)
        # all_succ vs failure (before-filtering baseline)
        allsucc = per_ep[per_ep.group.isin([0, 1])][col].values
        fail = per_ep[per_ep.group == 2][col].values
        rec["all_succ_vs_failure"] = auc(allsucc, fail)
        rows.append(rec)
    auc_df = pd.DataFrame(rows)

    md = ["# Verifier discrimination on filtered / other / failure eggplant episodes",
          "",
          f"Per-episode mean scores; ranking AUC = P(left-group episode ranks above right). "
          f"0.5 = no separation, 1.0 = perfect.",
          "",
          f"Groups (episodes): **filtered_success={n0}**, **other_success={n1}**, **failure={n2}**.",
          "",
          "| verifier | filt_succ vs failure | other_succ vs failure | filt_succ vs other_succ | all_succ vs failure (baseline) |",
          "|---|---|---|---|---|"]
    for r in rows:
        md.append("| {verifier} | {filt_succ_vs_failure:.3f} | {other_succ_vs_failure:.3f} | "
                  "{filt_succ_vs_other_succ:.3f} | {all_succ_vs_failure:.3f} |".format(**r))
    md += ["",
           "- **filt_succ vs failure**: does the verifier rank the cleanest successes above failures?",
           "- **filt_succ vs other_succ**: >0.5 means the verifier scores strictly-filtered successes "
           "higher than messy (reward-only) successes — i.e. it captures the filter's notion of quality. "
           "~0.5 means it can't tell filtered from other successes.",
           "- **all_succ vs failure**: the plain success-vs-failure AUC (what the pre-filtering "
           "comparison measured).",
           ""]
    (out / "verifier_comp_auc.md").write_text("\n".join(md))
    print("\n" + "\n".join(md))

    # ---- figure: per-episode score strip per group, per verifier -----------
    ncol = len(metrics)
    fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 5.6), squeeze=False)
    for j, (col, label) in enumerate(metrics):
        ax = axes[0][j]
        for g in (0, 1, 2):
            vals = per_ep[per_ep.group == g][col].values
            if len(vals) == 0:
                continue
            jit = (np.linspace(-0.2, 0.2, len(vals)) if len(vals) > 1 else [0.0])
            ax.scatter(g + np.asarray(jit), vals, color=GROUP_COLOR[g], alpha=0.55,
                       s=24, edgecolor="k", linewidth=0.25, zorder=3)
            ax.hlines(vals.mean(), g - 0.3, g + 0.3, color=GROUP_COLOR[g], linewidth=3, zorder=4)
            ax.text(g, vals.mean(), f" mu={vals.mean():.2f}", va="center", fontsize=8)
        r = rows[j]
        ax.set_title(f"{label}\n"
                     f"filt/fail AUC={r['filt_succ_vs_failure']:.2f}  "
                     f"other/fail={r['other_succ_vs_failure']:.2f}\n"
                     f"filt/other={r['filt_succ_vs_other_succ']:.2f}  "
                     f"allsucc/fail={r['all_succ_vs_failure']:.2f}", fontsize=9)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([f"filtered\nsuccess\n(n={n0})", f"other\nsuccess\n(n={n1})",
                            f"failure\n(n={n2})"], fontsize=8)
        ax.set_ylabel("per-episode mean score")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Verifier scores by episode group — eggplant (strict-filter / other-success / failure)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "verifier_comp.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out/'verifier_comp.png'}, {out/'verifier_comp_auc.md'}, {out/'verifier_comp.csv'}")


if __name__ == "__main__":
    main()
