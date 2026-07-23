#!/usr/bin/env python
"""filtered_ft_progression.py — 3-group verifier discrimination vs FINETUNE STEP.

Companion to filtered_plot_comp.py. That script judges ONE DQC checkpoint on the
3 episode groups (filtered_success / other_success / failure). This one sweeps the
per-checkpoint scores produced while FINETUNING the DQC chunk-Q on the filtered
eggplant data (v4_filt_ramp) and plots each ranking AUC as a function of the
finetune step, with the ORIGINAL base DQC (@100k) as the step-0 reference and the
(checkpoint-independent) LLaVA verifier as a flat reference line.

Question: does finetuning on the strict-filtered successes make the chunk-Q better
at (a) separating successes from failures and (b) telling the filtered "clean"
successes from the messy reward-only successes?  (i.e. do filt/fail, allsucc/fail,
and especially filt/other AUC rise with the finetune step, or regress?)

Reads, per finetune step S:
  <out-dir>/<ft-subdir>/step_<S>/dqc_scores_filtered.csv   (filtered_score_dqc.py)
plus the base reference:
  <out-dir>/dqc_scores_filtered.csv                        (base DQC @100k)
and the LLaVA reference (constant across DQC checkpoints):
  <out-dir>/verifier_scores.npz                            (eval_robomonkey.py)

Writes (in <out-dir>):
  ft_progression.csv   long form: step, verifier, pair, auc
  ft_progression.md    one table per AUC pair (rows = step)
  ft_progression.png   2x2 grid, one panel per AUC pair, lines per DQC variant

Runs in any env with pandas + numpy + matplotlib (no jax).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# DQC variants scored per checkpoint (from dqc_scores_filtered.csv) + LLaVA ref.
DQC_METRICS = [
    ("q_chunk8", "DQC chunk-8"),
    ("q_chunk1", "DQC chunk-1"),
    ("q_action", "DQC action(1-step)"),
]
# The 4 ranking AUCs (group_a vs group_b); groups: 0 filtered, 1 other, 2 failure.
PAIRS = [
    ("filt_succ_vs_failure", 0, 2),
    ("other_succ_vs_failure", 1, 2),
    ("filt_succ_vs_other_succ", 0, 1),   # the filter-quality question
    ("all_succ_vs_failure", (0, 1), 2),  # before-filtering baseline
]


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def _per_ep_auc(df: pd.DataFrame, col: str, ga, gb) -> float:
    """Per-episode-mean ranking AUC of `col` for group(s) `ga` vs `gb`."""
    per = df.groupby("ep_uid").agg(group=("group", "first"), val=(col, "mean")).reset_index()
    ga_set = ga if isinstance(ga, tuple) else (ga,)
    gb_set = gb if isinstance(gb, tuple) else (gb,)
    pos = per[per.group.isin(ga_set)].val.values
    neg = per[per.group.isin(gb_set)].val.values
    return _auc(pos, neg)


def _load_llava_col(base_csv: Path, llava_npz: Path) -> pd.DataFrame | None:
    """Base scores CSV with an added `verifier_score` (LLaVA) column, row-aligned."""
    if not llava_npz.exists():
        return None
    df = pd.read_csv(base_csv)
    vs = np.load(llava_npz, allow_pickle=True)["verifier_score"]
    if len(vs) != len(df):
        print(f"[prog] LLaVA rows ({len(vs)}) != base rows ({len(df)}) — skipping LLaVA ref")
        return None
    df = df.copy()
    df["verifier_score"] = np.asarray(vs)[df["row"].values]
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ft-subdir", default="ft")
    ap.add_argument("--base-label", default="base@100k")
    args = ap.parse_args()
    out = Path(args.out_dir)
    ft_root = out / args.ft_subdir
    base_csv = out / "dqc_scores_filtered.csv"
    llava_npz = out / "verifier_scores.npz"

    # ---- collect (step, csv) points: base at step 0, then each finetune step ----
    points: list[tuple[int, str, Path]] = []
    if base_csv.exists():
        points.append((0, args.base_label, base_csv))
    for d in sorted(ft_root.glob("step_*")):
        m = re.match(r"step_(\d+)$", d.name)
        c = d / "dqc_scores_filtered.csv"
        if m and c.exists():
            points.append((int(m.group(1)), f"ft@{int(m.group(1))//1000}k", c))
    points.sort(key=lambda p: p[0])
    if not points:
        raise SystemExit(f"[prog] no dqc_scores_filtered.csv found under {out} / {ft_root}/step_*")
    print(f"[prog] {len(points)} checkpoints: " + ", ".join(f"{lbl}(s={s})" for s, lbl, _ in points))

    # ---- LLaVA reference (constant across DQC steps) ----
    llava_df = _load_llava_col(base_csv, llava_npz)
    llava_auc = {}
    if llava_df is not None:
        for name, ga, gb in PAIRS:
            llava_auc[name] = _per_ep_auc(llava_df, "verifier_score", ga, gb)
        print("[prog] LLaVA ref AUCs: " + ", ".join(f"{k}={v:.3f}" for k, v in llava_auc.items()))

    # ---- compute AUCs per checkpoint ----
    long_rows = []
    for step, label, csv in points:
        df = pd.read_csv(csv)
        for col, vlabel in DQC_METRICS:
            if col not in df:
                continue
            for name, ga, gb in PAIRS:
                long_rows.append({"step": step, "label": label, "verifier": vlabel,
                                  "pair": name, "auc": _per_ep_auc(df, col, ga, gb)})
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(out / "ft_progression.csv", index=False)

    # ---- markdown: one table per AUC pair (rows = step) ----
    steps = [s for s, _, _ in points]
    labels = {s: l for s, l, _ in points}
    md = ["# DQC finetune (v4_filt_ramp) — 3-group discrimination vs finetune step", "",
          "Per-episode-mean ranking AUC (P(left-group episode ranks above right); 0.5 = no "
          "separation). Base = original DQC @100k (step 0). LLaVA is checkpoint-independent "
          "(flat reference).", ""]
    for name, _, _ in PAIRS:
        md += [f"## {name}", "",
               "| verifier | " + " | ".join(labels[s] for s in steps) + " |",
               "|---|" + "---|" * len(steps)]
        for col, vlabel in DQC_METRICS:
            cells = []
            for s in steps:
                v = long_df[(long_df.step == s) & (long_df.verifier == vlabel) & (long_df.pair == name)]
                cells.append(f"{v.auc.iloc[0]:.3f}" if len(v) else "—")
            md.append(f"| {vlabel} | " + " | ".join(cells) + " |")
        if name in llava_auc:
            md.append(f"| LLaVA (ref) | " + " | ".join(f"{llava_auc[name]:.3f}" for _ in steps) + " |")
        md.append("")
    md += ["**filt_succ_vs_other_succ > 0.5** means the verifier scores the strict-filtered "
           "successes above the messy reward-only successes — i.e. it captures the filter's "
           "notion of quality. Watch whether finetuning on the filtered set raises it.", "",
           "> Caveat: this is OFFLINE discrimination on the recorded DEMO action, NOT best-of-N "
           "candidate ranking in sim. A higher AUC here does not guarantee a better BoN verifier "
           "(see finetuned_dqc_june24.md — advantage rose while BoN success fell). Confirm with "
           "the sim BoN sweep before trusting a checkpoint.", ""]
    (out / "ft_progression.md").write_text("\n".join(md))
    print("\n" + "\n".join(md))

    # ---- figure: 2x2, one panel per AUC pair, x=step, line per DQC variant ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), squeeze=False)
    xs = np.array(steps, float)
    for k, (name, _, _) in enumerate(PAIRS):
        ax = axes[k // 2][k % 2]
        for col, vlabel in DQC_METRICS:
            ys = [long_df[(long_df.step == s) & (long_df.verifier == vlabel)
                          & (long_df.pair == name)].auc.iloc[0] for s in steps]
            lw = 2.6 if col == "q_chunk8" else 1.6
            ax.plot(xs, ys, marker="o", linewidth=lw, label=vlabel)
        if name in llava_auc:
            ax.axhline(llava_auc[name], ls="--", color="gray", lw=1.3, label="LLaVA (ref)")
        ax.axhline(0.5, ls=":", color="k", lw=1.0, alpha=0.6)
        ax.axvline(0.0, ls=":", color="silver", lw=1.0)  # base marker
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("finetune step (0 = base @100k)")
        ax.set_ylabel("per-episode ranking AUC")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8, loc="best")
    fig.suptitle("DQC v4_filt_ramp finetune — verifier discrimination of filtered / other / "
                 "failure eggplant episodes vs step", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "ft_progression.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[prog] wrote {out/'ft_progression.csv'}, {out/'ft_progression.md'}, {out/'ft_progression.png'}")


if __name__ == "__main__":
    main()
