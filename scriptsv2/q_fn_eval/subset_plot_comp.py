#!/usr/bin/env python
"""subset_plot_comp.py — DQC (chunk-8 / chunk-1 / action) discrimination on a
NEW task (stack / spoon), zero-shot from the frozen eggplant-trained checkpoint.

Reads ``dqc_scores_filtered.csv`` (filtered_score_dqc.py) + ``pairs.npz``
(subset_sample_pairs.py, for the per-episode ``sub_level``). Scores are aggregated
to a per-EPISODE mean (the deployed quantity). Two outputs:

  1. verifier_comp.png / _auc.md / .csv — the eggplant-style 3-group strip plot
     (filtered_success vs other_success vs failure) with ranking AUCs, using the
     FULL filtered subsample (sub100) as the positive group.

  2. auc_vs_subsize.png / .csv — how the headline AUCs move as the filtered group
     shrinks to the nested sub50 / sub25 subsamples (sub25 subset sub50 subset sub100).

Ranking AUC = P(left-group episode ranks above right); 0.5 = no separation.
LLaVA is intentionally omitted (question is about the base DQC chunk-8 verifier).
Runs in any env with pandas + matplotlib. See run_subset_verifier_comp.sbatch.
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
METRICS = [
    ("q_chunk8", "DQC chunk-8 Q\n(chunk critic, 8-step)"),
    ("q_chunk1", "DQC chunk-1 Q\n(chunk critic, 1-step)"),
    ("q_action", "DQC single-step Q\n(action critic)"),
]
SUB_COLOR = {"q_chunk8": "seagreen", "q_chunk1": "steelblue", "q_action": "gray"}


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
    ap.add_argument("--task-label", default="", help="e.g. 'stack (stack green on yellow)'")
    args = ap.parse_args()
    out = Path(args.out_dir)
    tlab = args.task_label or out.name

    df = pd.read_csv(out / "dqc_scores_filtered.csv")     # row, ep_uid, group, success, q_*
    # per-episode sub_level from pairs.npz (25/50/100 for filtered eps, 0 else)
    d = np.load(out / "pairs.npz", allow_pickle=True)
    sl = pd.DataFrame({"ep_uid": d["ep_uid"], "sub_level": d["sub_level"]}) \
        .groupby("ep_uid")["sub_level"].first()
    df.to_csv(out / "verifier_comp.csv", index=False)
    metrics = [(c, l) for c, l in METRICS if c in df and not df[c].isna().all()]

    # ---- per-episode means -------------------------------------------------
    agg = {c: (c, "mean") for c, _ in metrics}
    per_ep = df.groupby("ep_uid").agg(group=("group", "first"), **agg).reset_index()
    per_ep = per_ep.merge(sl, on="ep_uid", how="left")
    g0, g1, g2 = (per_ep[per_ep.group == g] for g in (0, 1, 2))
    n0, n1, n2 = len(g0), len(g1), len(g2)
    print(f"[plot] per-episode: filtered_success={n0}, other_success={n1}, failure={n2}")

    # ---- AUC table (full sub100 filtered group) ----------------------------
    PAIRS = [("filt_succ_vs_failure", 0, 2), ("other_succ_vs_failure", 1, 2),
             ("filt_succ_vs_other_succ", 0, 1)]
    rows = []
    for col, label in metrics:
        rec = {"verifier": label.replace("\n", " ")}
        for name, ga, gb in PAIRS:
            rec[name] = auc(per_ep[per_ep.group == ga][col].values,
                            per_ep[per_ep.group == gb][col].values)
        rec["all_succ_vs_failure"] = auc(per_ep[per_ep.group.isin([0, 1])][col].values,
                                         per_ep[per_ep.group == 2][col].values)
        rows.append(rec)

    md = [f"# DQC zero-shot verifier discrimination — {tlab}",
          "",
          "Frozen eggplant-trained DQC checkpoint scoring an unseen task. Per-episode "
          "mean scores; ranking AUC = P(left-group episode ranks above right). "
          "0.5 = no separation, 1.0 = perfect.",
          "",
          f"Groups (episodes): **filtered_success={n0}** (sub100), **other_success={n1}** "
          f"(messy reward-only), **failure={n2}**.",
          "",
          "| verifier | filt_succ vs failure | other_succ vs failure | filt_succ vs other_succ | all_succ vs failure |",
          "|---|---|---|---|---|"]
    for r in rows:
        md.append("| {verifier} | {filt_succ_vs_failure:.3f} | {other_succ_vs_failure:.3f} | "
                  "{filt_succ_vs_other_succ:.3f} | {all_succ_vs_failure:.3f} |".format(**r))
    md += ["",
           "- **filt/other_succ vs failure**: can the frozen verifier rank clean / messy "
           "successes above failures on this unseen task?",
           "- **filt_succ vs other_succ**: >0.5 means it scores strictly-filtered > messy successes.",
           ""]
    (out / "verifier_comp_auc.md").write_text("\n".join(md))
    print("\n" + "\n".join(md))

    # ---- figure 1: 3-group strip plot per DQC variant ----------------------
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
        ax.set_ylabel("per-episode mean DQC Q")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Frozen DQC verifier by episode group — {tlab} (zero-shot)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "verifier_comp.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- figure 2: AUC vs subsample size (nested filtered group) -----------
    sizes = sorted(int(x) for x in per_ep.loc[per_ep.group == 0, "sub_level"].dropna().unique())
    fail = per_ep[per_ep.group == 2]
    other = per_ep[per_ep.group == 1]
    sub_rows = []
    for L in sizes:
        filt = per_ep[(per_ep.group == 0) & (per_ep.sub_level <= L)]
        rec = {"sub_size": L, "n_filt": len(filt)}
        for col, _ in metrics:
            rec[f"{col}_filt_fail"] = auc(filt[col].values, fail[col].values)
            rec[f"{col}_filt_other"] = auc(filt[col].values, other[col].values)
        sub_rows.append(rec)
    sub_df = pd.DataFrame(sub_rows)
    sub_df.to_csv(out / "auc_vs_subsize.csv", index=False)

    fig2, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.6), squeeze=True)
    xs = sub_df["sub_size"].values
    for col, label in metrics:
        axa.plot(xs, sub_df[f"{col}_filt_fail"], "o-", color=SUB_COLOR[col],
                 label=label.split("\n")[0])
        axb.plot(xs, sub_df[f"{col}_filt_other"], "o-", color=SUB_COLOR[col],
                 label=label.split("\n")[0])
    for ax, ttl in [(axa, "filtered_success vs failure"),
                    (axb, "filtered_success vs other_success")]:
        ax.axhline(0.5, color="k", lw=0.8, ls=":")
        ax.set_xticks(xs)
        ax.set_xlabel("filtered subsample size (episodes)")
        ax.set_ylabel("per-episode ranking AUC")
        ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig2.suptitle(f"DQC zero-shot AUC vs filtered subsample size — {tlab}", fontsize=12)
    fig2.tight_layout()
    fig2.savefig(out / "auc_vs_subsize.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[plot] wrote verifier_comp.png, auc_vs_subsize.png, "
          f"verifier_comp_auc.md, verifier_comp.csv, auc_vs_subsize.csv in {out}")
    print("[auc-vs-size]\n" + sub_df.to_string(index=False))


if __name__ == "__main__":
    main()
