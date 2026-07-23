#!/usr/bin/env python
"""Episode / trajectory length analysis for the sub50 BC + search policies (seeded 500-step
runs at the peak checkpoints).

Per-episode records (scriptsv2/eval_search/eval_search.py) carry only two terminal flags:
  success   = env returned done  (task completed)
  truncated = env returned trunc (hit the env's native TimeLimit)
There is NO separate "failure" flag: an episode is success XOR truncated, so failure == truncated
(ran to the env horizon without succeeding). num_steps = ENV steps (replans = num_steps / 4).

Figures -> data/eval/jul8_notes/:
  1. episode_lengths_by_policy.png  — length distribution (all episodes) per policy, per task.
  2. episode_lengths_by_outcome.png — length split by success vs truncated, per policy.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = "/mmfs1/home/harine/RoboMonkey/data/eval/_seeded"
OUT = "/mmfs1/home/harine/RoboMonkey/data/eval/jul8_notes"
C_SUCC, C_TRUNC = "#55A868", "#C44E52"
C = {"search": "#4C72B0", "bc": "#DD8452"}

# task -> env native truncation cap (env steps) and the peak-N seeded 500-step cell per kind.
TASKS = {
    "eggplant": {"cap": 120,
                 "bc": "eggplant_sub50_bc_st500_n8_seed1000_ep50",
                 "search": "eggplant_sub50_search_90k_st500_n8_seed1000_ep50",
                 "N": {"bc": 8, "search": 8}},
    "stack":    {"cap": 60,
                 "bc": "stack_sub50_bc_st500_n8_seed1000_ep50",
                 "search": "stack_sub50_search_60k_st500_n8_seed1000_ep50",
                 "N": {"bc": 8, "search": 8}},
    "spoon":    {"cap": 60,
                 "bc": "spoon_sub50_bc_st500_n1_seed1000_ep50",
                 "search": "spoon_sub50_search_10k_st500_n2_seed1000_ep50",
                 "N": {"bc": 1, "search": 2}},
}


def load(cell):
    succ, trunc = [], []
    p = os.path.join(SEED, cell, "episodes.jsonl")
    if not os.path.isfile(p):
        return np.array([]), np.array([])
    for l in open(p):
        d = json.loads(l)
        if d.get("num_steps", 0) == 0:      # skip phantom/instant-error episodes
            continue
        (succ if d["success"] else trunc).append(d["num_steps"])
    return np.array(succ), np.array(trunc)


def jitter(x, n, w=0.16):
    return np.full(n, x) + (np.random.RandomState(0).rand(n) - 0.5) * 2 * w


def fig_by_policy(data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), squeeze=False)
    for ti, (task, cfg) in enumerate(TASKS.items()):
        ax = axes[0][ti]
        for xi, kind in enumerate(["bc", "search"]):
            s, t = data[(task, kind)]
            allv = np.concatenate([s, t]) if len(s) + len(t) else np.array([])
            if len(allv):
                parts = ax.violinplot([allv], positions=[xi], widths=0.7, showmedians=False,
                                      showextrema=False)
                for b in parts["bodies"]:
                    b.set_facecolor(C[kind]); b.set_alpha(0.28)
            ax.scatter(jitter(xi, len(s)), s, s=16, color=C_SUCC, alpha=0.7, label="success" if xi == 0 else None, zorder=3)
            ax.scatter(jitter(xi, len(t)), t, s=16, color=C_TRUNC, alpha=0.7, label="truncated" if xi == 0 else None, zorder=3)
            n = len(s) + len(t)
            if n:
                ax.annotate(f"SR {len(s)/n*100:.0f}%\nn={n}", (xi, cfg["cap"] * 1.02),
                            ha="center", va="bottom", fontsize=8, color="dimgray")
        ax.axhline(cfg["cap"], color="gray", ls="--", lw=1.0, alpha=0.6)
        ax.text(1.5, cfg["cap"], f"env cap {cfg['cap']}", ha="right", va="bottom", fontsize=7, color="gray")
        ax.set_xticks([0, 1]); ax.set_xticklabels([f"BC (N={cfg['N']['bc']})", f"search (N={cfg['N']['search']})"])
        ax.set_title(f"{task}"); ax.set_ylabel("episode length (env steps)")
        ax.set_ylim(0, cfg["cap"] * 1.18); ax.grid(alpha=0.2, axis="y")
        if ti == 0:
            ax.legend(fontsize=8, loc="center left")
    fig.suptitle("Episode length per policy (sub50, seeded 500-step, peak N) — green=success, red=truncated\n"
                 "successes are short & similar across policies; failures run to the env cap", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(OUT, "episode_lengths_by_policy.png"); fig.savefig(p, dpi=130); plt.close(fig)
    print("[fig] " + p)


def fig_by_outcome(data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), squeeze=False)
    for ti, (task, cfg) in enumerate(TASKS.items()):
        ax = axes[0][ti]
        pos, labels, box_succ = [], [], []
        x = 0
        for kind in ["bc", "search"]:
            s, t = data[(task, kind)]
            for out, arr, col in [("success", s, C_SUCC), ("trunc", t, C_TRUNC)]:
                if len(arr):
                    bp = ax.boxplot([arr], positions=[x], widths=0.5, patch_artist=True,
                                    showfliers=False, medianprops=dict(color="black"))
                    bp["boxes"][0].set_facecolor(col); bp["boxes"][0].set_alpha(0.55)
                    ax.scatter(jitter(x, len(arr), 0.10), arr, s=12, color=col, alpha=0.6, zorder=3)
                    ax.annotate(f"n={len(arr)}", (x, 2), ha="center", va="bottom", fontsize=7, color="dimgray")
                labels.append(f"{kind}\n{out}"); pos.append(x); x += 1
            x += 0.4
        ax.axhline(cfg["cap"], color="gray", ls="--", lw=1.0, alpha=0.6)
        ax.set_xticks(pos); ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"{task}  (env cap {cfg['cap']})"); ax.set_ylabel("episode length (env steps)")
        ax.set_ylim(0, cfg["cap"] * 1.1); ax.grid(alpha=0.2, axis="y")
    fig.suptitle("Trajectory length by outcome (sub50, seeded 500-step, peak N)\n"
                 "success = time-to-complete (short); truncated = ran to the env horizon", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    p = os.path.join(OUT, "episode_lengths_by_outcome.png"); fig.savefig(p, dpi=130); plt.close(fig)
    print("[fig] " + p)


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {}
    table = {}
    for task, cfg in TASKS.items():
        for kind in ["bc", "search"]:
            s, t = load(cfg[kind])
            data[(task, kind)] = (s, t)
            n = len(s) + len(t)
            table[f"{task}/{kind}"] = {
                "N": cfg["N"][kind], "n_ep": n, "SR_pct": round(len(s) / n * 100, 1) if n else None,
                "n_success": len(s), "n_truncated": len(t), "env_cap": cfg["cap"],
                "succ_len_median": int(np.median(s)) if len(s) else None,
                "succ_len_p10_p90": [int(np.percentile(s, 10)), int(np.percentile(s, 90))] if len(s) else None,
                "trunc_len_median": int(np.median(t)) if len(t) else None,
            }
    fig_by_policy(data)
    fig_by_outcome(data)
    json.dump(table, open(os.path.join(OUT, "episode_length_table.json"), "w"), indent=2)
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
