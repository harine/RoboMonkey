#!/usr/bin/env python3
"""Generate data/eval/random_a_diverse/spoon_on_towel/RESULTS.md.

The deliverable table: method x success rate for the spoon search policies.

Reads ONLY existing eval_log.json files -- the baselines are never re-run:
  rmse            data/eval/rmse_det/rmse_spoon/{10k,20k}/sr_n<N>/eval_log.json
  pg (awbc)       data/eval/pg_det/pg_awbc_l0.*_spoon_sub50/{10k,20k}/sr_n<N>/...
  reinforce       data/eval/pg_det/pg_reinf_l0.*_spoon_sub50/{10k,20k}/sr_n<N>/...
  rmse+augmented  data/eval/random_a_diverse/spoon_on_towel/det/{10k,20k}/sr_n<N>_k<K>/...
All of these are the SAME protocol: 50 episodes, seeds 1000-1049, MODE=argmax,
MAX_STEPS=60, tiled(8), DQC chunk verifier -> directly comparable.

Plus the demo-seed table (the seeds the train/test analysis videos use) and the
augmentation diagnostics (how often a random candidate won, offline pool metrics).

  conda activate monkey-verifier
  python scriptsv2/analysis/write_random_aug_results.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re

E = "data/eval"
ROOT = f"{E}/random_a_diverse/spoon_on_towel"
ROOT_DEFAULT = ROOT
N_GRID = [1, 2, 4, 8, 16, 32, 64]
STEPS_K = ["10k", "20k"]


def load(path):
    try:
        with open(path) as fh:
            d = json.load(fh)
        return d if d.get("num_episodes") else None
    except Exception:
        return None


def sr_row(pattern):
    """pattern has a {N} slot -> {N: eval_log dict}."""
    out = {}
    for N in N_GRID:
        d = load(pattern.format(N=N))
        if d:
            out[N] = d
    return out


def fmt_row(label, cells):
    if not cells:
        return None
    vals = []
    for N in N_GRID:
        d = cells.get(N)
        vals.append(f"{100 * d['success_rate']:.0f}" if d else "—")
    bestN = max(cells, key=lambda n: cells[n]["success_rate"])
    b = cells[bestN]
    peak = f"**{100 * b['success_rate']:.0f}%** @N{bestN}"
    dist = b.get("mean_obj_goal_dist_min")
    if dist is not None:
        peak += f" (d={dist:.2f})"
    return f"| {label} | " + " | ".join(vals) + f" | {peak} |"


def section_table(rows):
    hdr = ["| method | " + " | ".join(f"N{n}" for n in N_GRID) + " | peak |",
           "|---" * (len(N_GRID) + 2) + "|"]
    return hdr + [r for r in rows if r]


def methods_for(k):
    """(label, eval_log-path pattern) for one checkpoint milestone."""
    m = [(f"rmse (baseline)", f"{E}/rmse_det/rmse_spoon/{k}/sr_n{{N}}/eval_log.json")]
    for run, lab in (("pg_awbc_l0.1_spoon_sub50", "pg / awbc λ0.1"),
                     ("pg_awbc_l0.3_spoon_sub50", "pg / awbc λ0.3"),
                     ("pg_reinf_l0.1_spoon_sub50", "reinforce λ0.1"),
                     ("pg_reinf_l0.3_spoon_sub50", "reinforce λ0.3")):
        m.append((lab, f"{E}/pg_det/{run}/{k}/sr_n{{N}}/eval_log.json"))
    for K in sorted({int(x) for x in
                     (re.search(r"_k(\d+)$", p).group(1)
                      for p in glob.glob(f"{ROOT}/det/{k}/sr_n*_k*")
                      if re.search(r"_k(\d+)$", p))}):
        m.append((f"**rmse + augmented (K={K})**",
                  f"{ROOT}/det/{k}/sr_n{{N}}_k{K}/eval_log.json"))
    return m


def diag_lines(k):
    """frac of replans an augmented candidate won, per augmented cell."""
    out = []
    for p in sorted(glob.glob(f"{ROOT}/det/{k}/*/eval_log.json")):
        d = load(p)
        if not d or d.get("rand_aug_k") is None:
            continue
        cell = os.path.basename(os.path.dirname(p))
        f = d.get("frac_replans_random_won")
        out.append(f"| {k} | {cell} | {d['rand_aug_k']} | {d['rand_aug_m']} | "
                   f"{d.get('rand_aug_gripper', '?')} | "
                   f"{'yes' if d.get('rand_aug_only') else 'no'} | "
                   f"{100 * d['success_rate']:.0f} | "
                   f"{'—' if f is None else f'{100 * f:.0f}%'} |")
    return out


def rollout_rows():
    rows = []
    for split in ("train", "test"):
        d = load(f"{E}/train_test_analysis/spoon/rollout/{split}/eval_log.json")
        if d:
            rows.append(f"| rmse @90k (baseline) | {split} | {d['num_episodes']} | "
                        f"{100 * d['success_rate']:.0f} | — |")
    for p in sorted(glob.glob(f"{ROOT}/rollout/*/eval_log.json")):
        d = load(p)
        if not d:
            continue
        cell = os.path.basename(os.path.dirname(p))
        split = cell.split("_")[0]
        f = d.get("frac_replans_random_won")
        rows.append(f"| rmse @90k + augmented ({cell.split('_', 1)[1]}) | {split} | "
                    f"{d['num_episodes']} | {100 * d['success_rate']:.0f} | "
                    f"{'—' if f is None else f'{100 * f:.0f}%'} |")
    return rows


def offline_table():
    p = f"{ROOT}/metrics/aug_pool_metrics.csv"
    if not os.path.exists(p):
        return []
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return []
    L = ["| cell | frames | best Q policy | best Q random | Q(a*) | argmax random | "
         "div policy | div random | oracle RMSE policy | oracle RMSE union |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['cell']} | {r['n_frames']} | {float(r['q_pol_best']):.2f} | "
                 f"{float(r['q_rand_best']):.2f} | {float(r['q_astar']):.2f} | "
                 f"{100 * float(r['frac_argmax_random']):.0f}% | "
                 f"{float(r['div_pol']):.3f} | {float(r['div_rand']):.3f} | "
                 f"{float(r['oracle_pol']):.4f} | {float(r['oracle_union']):.4f} |")
    return L


def gripper_note():
    """Describe how the random candidates got their gripper, read off the cells."""
    modes = {d.get("rand_aug_gripper") for d in
             (load(p) for p in glob.glob(f"{ROOT}/det/*/*/eval_log.json")) if d}
    modes.discard(None)
    if modes == {"copy"}:
        return ("**Gripper: `copy`** — every random candidate KEEPS the grasp (dim 6) of the "
                "search transformer's argmax candidate, so the random pool varies only the "
                "translation direction. M = K candidates per replan.")
    if modes == {"both"}:
        return ("**Gripper: `both`** — each sector emits two candidates, grasp forced open and "
                "closed, so the verifier also picks the grasp state. M = 2K per replan.")
    return f"Gripper modes present: {', '.join(sorted(modes))}." if modes else ""


def headline():
    """One-paragraph verdict computed from the augmented cells themselves."""
    aug, won = [], []
    for p in glob.glob(f"{ROOT}/det/*/*/eval_log.json"):
        d = load(p)
        if d and d.get("rand_aug_k") is not None:
            aug.append(d["success_rate"])
            if d.get("frac_replans_random_won") is not None:
                won.append(d["frac_replans_random_won"])
    if not aug:
        return []
    base = [d["success_rate"] for d in
            (load(f"{E}/rmse_det/rmse_spoon/{k}/sr_n{N}/eval_log.json")
             for k in STEPS_K for N in N_GRID) if d]
    o = offline_row()
    txt = [f"## Headline", "",
           f"Augmenting the pool destroys the policy: **{100 * max(aug):.0f}% peak SR** "
           f"across {len(aug)} augmented cells (vs **{100 * max(base):.0f}%** peak for the "
           f"unaugmented rmse baseline on the same seeds), while an augmented candidate wins "
           f"the argmax in **{100 * sum(won) / len(won):.0f}%** of replans."]
    if o:
        txt[-1] += (
            f" Offline on the demo states the picture is unambiguous: the random pool's best "
            f"Q is **{o['q_rand_best']:.1f}** vs **{o['q_pol_best']:.1f}** for the policy pool "
            f"and **{o['q_astar']:.1f}** for the *expert* action `a*` — the verifier scores "
            f"off-manifold actions far above the ground truth. Candidate diversity confirms "
            f"the augmentation works as intended (mean pairwise cos of net Δxyz: "
            f"**{o['div_pol']:.3f}** policy = collapsed, **{o['div_rand']:.3f}** random = spread), "
            f"and the union's oracle RMSE to `a*` ({o['oracle_union']:.4f}) is *better* than the "
            f"policy pool's ({o['oracle_pol']:.4f}) — so better actions ARE in the pool; the "
            f"DQC chunk-Q simply cannot find them.")
    txt += ["", "**Conclusion:** the collapsed candidate pool was not the binding constraint. "
            "The DQC chunk critic is only meaningful *within* the narrow action distribution "
            "it was trained to score; ranking a wider set is a reward-hacking regime, so more "
            "action diversity strictly hurts until the verifier is trained on off-policy "
            "(negative) actions."]
    return txt


def offline_row():
    p = f"{ROOT}/metrics/aug_pool_metrics.csv"
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return None
    keys = ["q_pol_best", "q_rand_best", "q_astar", "div_pol", "div_rand",
            "oracle_pol", "oracle_union"]
    return {k: sum(float(r[k]) for r in rows) / len(rows) for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT,
                    help="campaign root (e.g. data/eval/random_a_with_grasp/spoon_on_towel)")
    ap.add_argument("--out", default=None, help="default: <root>/RESULTS.md")
    a = ap.parse_args()
    globals()["ROOT"] = a.root
    a.out = a.out or f"{a.root}/RESULTS.md"

    L = ["# spoon_on_towel — action-diversity augmentation vs the loss variants", "",
         "Auto-generated by `scriptsv2/analysis/write_random_aug_results.py`. "
         "Augmentation: `scriptsv2/eval_search/random_action_aug.py` — the candidate pool "
         "gains stratified-random chunks (K equal-area direction sectors, constant push over "
         "the 8-step horizon, rotation copied from the search transformer's argmax candidate), "
         "scored by the SAME DQC chunk verifier; selection is `argmax Q` over the union. "
         + gripper_note(), "",
         "> **All Table-1 cells share one protocol:** 50 episodes, seeds 1000–1049, "
         "`MODE=argmax`, `MAX_STEPS=60`, tiled(8), DQC chunk verifier baked into the ckpt. "
         "The rmse / pg / reinforce numbers are the pre-existing runs "
         "(`data/eval/{rmse_det,pg_det}`) — they were **not** re-run.", ""]

    figs = sorted(glob.glob(f"{ROOT}/metrics/*.png"))
    if figs:
        L += ["**Q figures** (`scriptsv2/analysis/plot_aug_q.py`): "
              + "  ".join(f"[`{os.path.basename(f)}`](metrics/{os.path.basename(f)})"
                          for f in figs), ""]
    qoe = glob.glob(f"{ROOT}/q_over_episode/*.png")
    if qoe:
        L += [f"**Per-seed Q over the episode** ({len(qoe)} figures, augmented vs unaugmented "
              f"on the same axes, `scriptsv2/analysis/plot_q_over_episode_aug.py`): "
              f"[`q_over_episode/`](q_over_episode/) — "
              f"`<split>_seed<seed>_k<K>_<base status>_vs_<aug status>.png`", ""]

    hl = headline()
    if hl:
        L += hl + [""]

    L += ["## 1. Success rate (%), 50 unseen episodes"]
    for k in STEPS_K:
        rows = [fmt_row(lab, sr_row(pat)) for lab, pat in methods_for(k)]
        if not any(rows):
            continue
        L += ["", f"### checkpoint {k}", ""] + section_table(rows)

    diag = sum((diag_lines(k) for k in STEPS_K), [])
    if diag:
        L += ["", "## 2. Did the random pool actually win?", "",
              "`random won` = fraction of replans whose argmax-Q candidate came from the "
              "augmented pool. Near 0 ⇒ the verifier never prefers the diverse actions; "
              "near 1 ⇒ it prefers them over everything the policy proposed.", "",
              "| ckpt | cell | K | M=2K | gripper | random-only | SR % | random won |",
              "|---|---|---|---|---|---|---|---|"] + diag

    rr = rollout_rows()
    if rr:
        L += ["", "## 3. Demo seeds (the train/test-analysis video seeds)", "",
              "The 10 train + 10 test demo seeds from "
              "`scriptsv2/eval_search/demoseeds/spoon_{train,test}.txt`, search ckpt 90k — "
              "the same rollouts the expert-overlay videos are rendered from "
              "(`videos/`, baseline linked under `videos/baseline/`). n=10 per split, so "
              "SR resolution is 10 points.", "",
              "| method | split | episodes | SR % | random won |",
              "|---|---|---|---|---|"] + rr

    ot = offline_table()
    if ot:
        L += ["", "## 4. Offline candidate pools on the demo states", "",
              "`div` = mean pairwise cosine of the candidates' net Δxyz (**1.0 = collapsed**). "
              "`oracle RMSE` = min RMSE to the expert chunk `a*` within the pool: if "
              "`oracle union` ≪ `oracle policy`, the random pool genuinely contains better "
              "actions and only the verifier's ranking stands between us and them.", ""] + ot

    L += ["", "## Caveats", "",
          "1. **Checkpoint matching.** pg / reinforce runs only reached ~60–65k, so the "
          "comparison is pinned at 10k and 20k where every method has data. The demo-seed "
          "table uses the 90k rmse ckpt (what the existing analysis videos used).",
          "2. **n=10** on the demo-seed table — treat it as qualitative; Table 1 (n=50) is "
          "the quantitative comparison.",
          "3. Both demo splits are success-filtered, so absolute numbers there are not a "
          "neutral baseline (see `data/eval/train_test_analysis/NOTES.md` §5).",
          "4. The random pool is scored by the SAME verifier and the SAME chunk-Q "
          "convention as the policy pool, so a random candidate winning is a verifier "
          "preference, not a scale artifact.", ""]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[results] -> {a.out}  ({len(L)} lines)")


if __name__ == "__main__":
    main()
