#!/usr/bin/env python3
"""Verifier comparison — OpenVLA: LLaVA vs single-step DQC (eggplant).

Two sections, both on task `widowx_put_eggplant_in_basket` (init=9, 50 ep/cell):

  1. RoboMonkey reproduction (per-seed, single operating point):
       LLaVA  robomonkey_repro/inprocess/summary.txt   (older, good; aug=32, 3 seeds)
       DQC    single-step, aug=C1_AUG: seed 1 = the aug=C1_AUG cell of
              bon_sweep_dqc_real; seeds 2/3 = bon_sweep_dqc_c1_seed{2,3} (re-runs)

  2. OpenVLA Best-of-N sweep (seed=1, across N):
       LLaVA  robomonkey_repro/bon_sweep/progress.json           (older, good)
       DQC    robomonkey_repro/bon_sweep_dqc_real/progress.json  (single-step, corrected)

The pre-2026-06-18 DQC results (`robomonkey_repro/dqc`, `robomonkey_repro/bon_sweep_dqc`)
are WRONG and were relabeled `*__WRONG_DO_NOT_USE`; this script never reads them.

data/eval -> /gscratch/robotics/harine/RoboMonkey_relocated/data/eval

Run:
  python scriptsv2/analysis/aggregate_verifier_results.py
Outputs under data/eval/_verifier_comparison/:
  search_bon_results.md, comparison.json, openvla_llava_vs_dqc.png
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "data" / "eval"
OUT = EVAL / "raw_data" / "_verifier_comparison"   # kept out of eval/ top-level (organized)

N_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# --- section 2: OpenVLA BoN sweep (seed=1) --------------------------------- #
LLAVA_PROGRESS = EVAL / "robomonkey_repro/bon_sweep/progress.json"
DQC_PROGRESS = EVAL / "robomonkey_repro/bon_sweep_dqc_real/progress.json"

# --- section 1: RoboMonkey reproduction (per-seed) ------------------------- #
# LLaVA (older, good) reproduction operating point is aug=32; the single-step
# DQC reproduction is at aug=C1_AUG. seed=1 IS the aug=C1_AUG cell of the
# corrected full sweep; seeds 2/3 are dedicated single-cell re-runs.
C1_LLAVA_AUG = 32
C1_LLAVA_SUMMARY = EVAL / "robomonkey_repro/inprocess/summary.txt"
C1_DQC_AUG = 32
C1_DQC_SEED_PROGRESS = {
    1: EVAL / "robomonkey_repro/bon_sweep_dqc_real/progress.json",
    2: EVAL / "robomonkey_repro/bon_sweep_dqc_c1_seed2/progress.json",
    3: EVAL / "robomonkey_repro/bon_sweep_dqc_c1_seed3/progress.json",
}

# Okabe-Ito colorblind-safe palette
CB = {"blue": "#0072B2", "vermillion": "#D55E00"}


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def load_json(p: Path):
    try:
        with open(p) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_progress(p: Path) -> dict[int, tuple[int, int]]:
    """progress.json {N: {episodes, success}} -> {N: (success, episodes)}."""
    out: dict[int, tuple[int, int]] = {}
    d = load_json(p)
    if not d:
        return out
    for k, v in d.items():
        out[int(k)] = (int(v.get("success", 0)), int(v.get("episodes", 0)))
    return out


def parse_repro_summary(p: Path) -> dict[int, tuple[int, int] | None]:
    """robomonkey_repro/*/summary.txt -> {seed: (n_success, n_episodes) | None}.

    None marks a seed whose run FAILED (e.g. http seed 1)."""
    out: dict[int, tuple[int, int] | None] = {}
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        m = re.match(r"^\s*(\d+)\s+(.+)$", line)
        if not m or line.lower().startswith("seed"):
            continue
        seed, rest = int(m.group(1)), m.group(2)
        if "FAILED" in rest:
            out[seed] = None
            continue
        m2 = re.search(r"(\d+)\s*\((\d+\.\d+)%\)", rest)
        if m2:
            n_succ, pct_v = int(m2.group(1)), float(m2.group(2))
            n_ep = round(n_succ / (pct_v / 100.0)) if pct_v > 0 else 50
            out[seed] = (n_succ, n_ep)
    return out


def load_c1_dqc_seeds() -> dict[int, tuple[int, int] | None]:
    """Per-seed (n_success, n_episodes) for the C1 single-step-DQC condition.

    Reads the aug=C1_DQC_AUG cell from each seed's progress.json (seed=1 from the
    full sweep, seeds 2/3 from the dedicated re-runs). None for a seed whose run
    hasn't produced that cell yet (reported as pending, not silently dropped)."""
    out: dict[int, tuple[int, int] | None] = {}
    for seed, p in sorted(C1_DQC_SEED_PROGRESS.items()):
        out[seed] = load_progress(p).get(C1_DQC_AUG)
    return out


def rate(t: tuple[int, int] | None) -> str:
    if not t or t[1] == 0:
        return "—"
    sc, ep = t
    flag = "" if ep >= 50 else "*"  # mark short (<50 ep) cells
    return f"{100.0 * sc / ep:.1f}% ({sc}/{ep}){flag}"


def pct(t: tuple[int, int] | None) -> float | None:
    return 100.0 * t[0] / t[1] if (t and t[1]) else None


def peak(dd: dict[int, tuple[int, int]]) -> tuple[int, float] | None:
    """(N, success%) of the best full-50-ep cell."""
    best = None
    for n, t in dd.items():
        if t[1] >= 50:
            p = 100.0 * t[0] / t[1]
            if best is None or p > best[1]:
                best = (n, p)
    return best


def mean_str(ts: list[tuple[int, int] | None]) -> str:
    ps = [pct(t) for t in ts if t and t[1] >= 50]
    if not ps:
        return "—"
    m = statistics.mean(ps)
    if len(ps) > 1:
        return f"{m:.1f}% (n={len(ps)})"
    return f"{m:.1f}% (n=1)"


# --------------------------------------------------------------------------- #
# render markdown
# --------------------------------------------------------------------------- #
def render_header() -> str:
    L = []
    L.append("# Verifier comparison — OpenVLA: LLaVA vs single-step DQC (eggplant)\n")
    L.append("Task = `widowx_put_eggplant_in_basket`, init=9, 50 episodes/cell. "
             "Auto-generated by `scriptsv2/analysis/aggregate_verifier_results.py` "
             "from on-disk results.\n")
    L.append("- **LLaVA** = OpenVLA candidates re-ranked by the RoboMonkey LLaVA reward "
             "model (older runs — good).")
    L.append("- **single-step DQC** = OpenVLA candidates re-ranked by the corrected "
             "single-step DQC Q-critic (`bon_sweep_dqc_real` + per-seed re-runs).\n")
    L.append("> ⚠️ The pre-2026-06-18 DQC results (`robomonkey_repro/dqc`, "
             "`robomonkey_repro/bon_sweep_dqc`) are **wrong** and were relabeled "
             "`*__WRONG_DO_NOT_USE`. This doc uses only the corrected single-step DQC runs.\n")
    L.append("`*` marks cells with <50 episodes (partial). `—` = not run.\n")
    return "\n".join(L)


def render_reproduction(llava_seeds, dqc_seeds) -> str:
    L = []
    L.append("## 1. RoboMonkey reproduction — LLaVA vs single-step DQC (per-seed)\n")
    L.append(f"Single operating point, 3 VLA seeds, 50 ep/seed. "
             f"**LLaVA** at aug={C1_LLAVA_AUG} (in-process reward server, older/good); "
             f"**single-step DQC** at aug={C1_DQC_AUG}. seed 1 DQC is the aug={C1_DQC_AUG} "
             "cell of the full sweep (§2); seeds 2/3 are dedicated re-runs. "
             "A seed is only filled in once its full 50 episodes complete (partial → pending).\n")
    if C1_LLAVA_AUG != C1_DQC_AUG:
        L.append(f"> ℹ️ Note the different sample budgets: LLaVA aug={C1_LLAVA_AUG} vs "
                 f"DQC aug={C1_DQC_AUG}.\n")
    L.append(f"| seed | LLaVA (aug={C1_LLAVA_AUG}) | single-step DQC (aug={C1_DQC_AUG}) |")
    L.append("|---|---|---|")

    def repro_cell(t):  # only fill at full 50 ep; partial/not-started -> pending
        if t is None:
            return "_(pending)_"
        if t[1] < 50:
            return f"_(pending — {t[1]}/50 ep)_"
        return rate(t)
    for s in (1, 2, 3):
        lv = llava_seeds.get(s)
        dq = dqc_seeds.get(s)
        lv_cell = "_(failed)_" if (lv is None and s in llava_seeds) else repro_cell(lv)
        dq_cell = repro_cell(dq)
        L.append(f"| {s} | {lv_cell} | {dq_cell} |")
    L.append(f"| **mean** | **{mean_str([llava_seeds.get(s) for s in (1, 2, 3)])}** | "
             f"**{mean_str([dqc_seeds.get(s) for s in (1, 2, 3)])}** |")
    L.append("")
    pending = sorted(s for s in (1, 2, 3)
                     if dqc_seeds.get(s) is None or (dqc_seeds.get(s) or (0, 0))[1] < 50)
    if pending:
        L.append(f"DQC seeds pending/partial: {pending} (will fill in as re-runs land).\n")
    return "\n".join(L)


def render_sweep(llava, dqc) -> str:
    L = []
    L.append("## 2. OpenVLA Best-of-N sweep — LLaVA vs single-step DQC (seed=1)\n")
    L.append("seed=1, init=9, augmented_samples = N, 50 ep/cell. "
             "LLaVA = `bon_sweep`; single-step DQC = `bon_sweep_dqc_real`.\n")
    L.append("| N | LLaVA | single-step DQC | Δ(DQC−LLaVA) |")
    L.append("|---|---|---|---|")
    for n in N_GRID:
        lv, dq = llava.get(n), dqc.get(n)
        lv_p, dq_p = pct(lv), pct(dq)
        delta = f"{dq_p - lv_p:+.1f}" if (lv_p is not None and dq_p is not None) else ""
        L.append(f"| {n} | {rate(lv)} | {rate(dq)} | {delta} |")
    L.append("")

    pk_l, pk_d = peak(llava), peak(dqc)
    if pk_l and pk_d:
        L.append(f"**Peaks:** LLaVA {pk_l[1]:.1f}% @ N={pk_l[0]} · "
                 f"single-step DQC {pk_d[1]:.1f}% @ N={pk_d[0]}. "
                 "single-step DQC ≥ LLaVA for N≤32 (notably +24 @ N=2, +20 @ N=4); "
                 "LLaVA edges ahead by a few points for N≥64.\n")

    # gaps (no silent truncation)
    L.append("### Gaps & caveats\n")
    gaps = []
    miss_l = [n for n in N_GRID if n not in llava]
    short_l = sorted(n for n, t in llava.items() if t[1] < 50)
    miss_d = [n for n in N_GRID if n not in dqc]
    short_d = sorted(n for n, t in dqc.items() if t[1] < 50)
    if miss_l: gaps.append(f"§2 LLaVA missing N: {miss_l}.")
    if short_l: gaps.append(f"§2 LLaVA partial (<50 ep) N: {short_l}.")
    if miss_d: gaps.append(f"§2 single-step DQC missing N: {miss_d}.")
    if short_d: gaps.append(f"§2 single-step DQC partial (<50 ep) N: {short_d}.")
    for g in gaps:
        L.append(f"- {g}")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# plot (section 2 scaling curve)
# --------------------------------------------------------------------------- #
def render_plot(llava, dqc, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[plot] matplotlib unavailable ({e}); skipping figure.")
        return False

    def series(dd):
        xs = [n for n in N_GRID if n in dd and dd[n][1]]
        ys = [100.0 * dd[n][0] / dd[n][1] for n in xs]
        return xs, ys

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    pk_l, pk_d = peak(llava), peak(dqc)
    lbl_l = f"LLaVA (peak {pk_l[1]:.0f}% @ N={pk_l[0]})" if pk_l else "LLaVA"
    lbl_d = f"single-step DQC (peak {pk_d[1]:.0f}% @ N={pk_d[0]})" if pk_d else "single-step DQC"

    xs, ys = series(llava)
    ax.plot(xs, ys, marker="o", color=CB["blue"], label=lbl_l)
    xs, ys = series(dqc)
    ax.plot(xs, ys, marker="s", color=CB["vermillion"], label=lbl_d)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("N (Best-of-N samples)")
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("OpenVLA Best-of-N — LLaVA vs single-step DQC verifier\n"
                 "(eggplant, seed=1, init=9, 50 ep/cell)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"[plot] wrote {path}")
    return True


# --------------------------------------------------------------------------- #
# §3-5: search-policy & BC Best-of-N sweeps
# (DQC chunk verifier unless noted; predict-8/rollout-4; 50 ep/cell, seeds 1000-1049)
# Cell results: EVAL/<sub>/<prefix>_n{N}_seed1000_ep50/eval_log.json
# --------------------------------------------------------------------------- #
BON_N = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# (key, label, sub, prefix). Search sweeps are split into the KEPT set — the filtered
# rmse-diff runs for the three tasks, shown at the top — and the ARCHIVED set (older /
# superseded variants), rendered in §6. All DQC-chunk verifier; predict-8/rollout-4.

# --- ARCHIVED search-policy sweeps: original diffusion-search (unfiltered data,
#     argmax + softmax), gaussian-search, the PG base-critic runs, and the gaussian
#     rmse variant. Superseded by the filtered rmse-diff runs kept at the top.
ARCHIVED_SEARCH_SWEEPS = [
    ("diff10k", "diffusion-search 10k · argmax", "search_dqc_chunk", "diff10k_chunk8_predh8_roll4"),
    ("diff20k", "diffusion-search 20k · argmax", "search_dqc_chunk", "diff20k_chunk8_predh8_roll4"),
    ("diff33k", "diffusion-search 33k · argmax", "search_dqc_chunk", "diffusion_chunk8_predh8_roll4"),
    ("diff1k_sm", "diffusion-search 1k · softmax", "search_dqc_chunk", "diff1k_chunk8_predh8_roll4_softmax"),
    ("diff2k_sm", "diffusion-search 2k · softmax", "search_dqc_chunk", "diff2k_chunk8_predh8_roll4_softmax"),
    ("diff5k_sm", "diffusion-search 5k · softmax", "search_dqc_chunk", "diff5k_chunk8_predh8_roll4_softmax"),
    ("diff10k_sm", "diffusion-search 10k · softmax", "search_dqc_chunk", "diff10k_chunk8_predh8_roll4_softmax"),
    ("diff20k_sm", "diffusion-search 20k · softmax", "search_dqc_chunk", "diff20k_chunk8_predh8_roll4_softmax"),
    ("diff33k_sm", "diffusion-search 33k · softmax", "search_dqc_chunk", "diffusion_chunk8_predh8_roll4_softmax"),
    ("gauss10k_sm", "gaussian-search 10k · softmax", "search_dqc_chunk", "gauss10k_chunk8_predh8_roll4_softmax"),
    ("gauss20k_sm", "gaussian-search 20k · softmax", "search_dqc_chunk", "gauss20k_chunk8_predh8_roll4_softmax"),
    ("gauss30k_sm", "gaussian-search 30k · softmax", "search_dqc_chunk", "gauss30k_chunk8_predh8_roll4_softmax"),
    ("gauss40k_sm", "gaussian-search 40k · softmax", "search_dqc_chunk", "gauss40k_chunk8_predh8_roll4_softmax"),
    ("gauss50k_sm", "gaussian-search 50k · softmax", "search_dqc_chunk", "gauss50k_chunk8_predh8_roll4_softmax"),
]


def _pg_sweep(run, m):
    return (f"{run}_{m}k", f"PG {run[3:].replace('_', '+')} {m}k",
            "search_dqc_chunk", f"{run}_{m}k_chunk8_predh8_roll4_softmax")


# PG base-critic runs (BC + advantage policy-gradient, lambda=0.1), softmax, 10-50k.
for _pg_run in ("pg_gauss_base", "pg_diff_base"):
    for _pg_m in (10, 20, 30, 40, 50):
        ARCHIVED_SEARCH_SWEEPS.append(_pg_sweep(_pg_run, _pg_m))
# rmse GAUSS on the filtered dataset — archived (the kept set is rmse DIFF only).
for _rg_m in (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
    ARCHIVED_SEARCH_SWEEPS.append((
        f"rmse_gauss_{_rg_m}k", f"rmse gauss (filtered) {_rg_m}k",
        "search_dqc_chunk", f"rmse_gauss_{_rg_m}k_chunk8_predh8_roll4_softmax"))
# v2-critic PG runs — abandoned (broken BoN verifier); rendered in §6c.
ARCHIVED_V2_SWEEPS = [_pg_sweep(r, m)
                      for r in ("pg_gauss_v2", "pg_diff_v2")
                      for m in (10, 20, 30, 40, 50)]

# --- KEPT search-policy sweeps: the FILTERED rmse-diff results for the three tasks.
#     eggplant — full 699-ep filtered set + its diverse subsamples ("eggplant" added to
#     the display name, previously just "rmse diff …"); stack_cube / spoon_on_towel —
#     diverse subsamples. base critic; softmax; milestones 5-100k (empty rows dropped).
KEPT_SEARCH_SWEEPS = []
_KEPT_RM = [("rmse_diff",   "rmse diff eggplant (filtered)",       "search_dqc_chunk"),
            ("rmse_sub100", "rmse diff eggplant (sub100-diverse)", "search_dqc_chunk"),
            ("rmse_sub50",  "rmse diff eggplant (sub50-diverse)",  "filtered_sub50_diverse_eggplant"),
            ("rmse_sub25",  "rmse diff eggplant (sub25-diverse)",  "filtered_sub25_diverse_eggplant")]
for _rm_lbl, _rm_disp, _rm_sub in _KEPT_RM:
    for _rm_m in (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
        KEPT_SEARCH_SWEEPS.append((
            f"{_rm_lbl}_{_rm_m}k",
            f"{_rm_disp} {_rm_m}k",
            _rm_sub,
            f"{_rm_lbl}_{_rm_m}k_chunk8_predh8_roll4_softmax",
        ))
# NEW-TASK diffusion search policies (stack_cube + spoon_on_towel), sub100/50/25
# diverse-start, BC-only, TASK-CORRECT baked DQC verifier. Milestones 5-100k.
_MT = [("stack_sub100", "stack diff (sub100-div)", "stack_cube_search"),
       ("stack_sub50",  "stack diff (sub50-div)",  "stack_cube_search"),
       ("stack_sub25",  "stack diff (sub25-div)",  "stack_cube_search"),
       ("spoon_sub100", "spoon diff (sub100-div)", "spoon_towel_search"),
       ("spoon_sub50",  "spoon diff (sub50-div)",  "spoon_towel_search"),
       ("spoon_sub25",  "spoon diff (sub25-div)",  "spoon_towel_search")]
for _mt_lbl, _mt_disp, _mt_sub in _MT:
    for _mt_m in (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
        KEPT_SEARCH_SWEEPS.append((
            f"{_mt_lbl}_{_mt_m}k",
            f"{_mt_disp} {_mt_m}k",
            _mt_sub,
            f"{_mt_lbl}_{_mt_m}k_chunk8_predh8_roll4_softmax",
        ))
# (label, sub, prefix, verifier, sampler, selection, ckpt)
# KEPT BC: the search-matched BC-H8 n_obs=1 policies + the strong BC-H8 filtered-699.
# eggplant sub100/50/25 + filtered-699, then the NEW-TASK stack_cube / spoon_on_towel
# n_obs=1 BC policies (each on its best n=1 ckpt; SR not comparable to eggplant).
KEPT_BC_SWEEPS = [
    ("BC-H8 eggplant filtered-699 · DQC-chunk · ddpm · argmax",        "bc_dqc_chunk", "bcH8filt50k_chunk8_predh8_roll4_ddpm",                 "DQC-chunk", "ddpm", "argmax",  "eggpl filt50k"),
    ("BC-H8 eggplant sub100-div n_obs=1 · DQC-chunk · ddpm · softmax", "bc_dqc_chunk", "bcH8sub100nobs1_30k_chunk8_predh8_roll4_ddpm_softmax", "DQC-chunk", "ddpm", "softmax", "eggpl sub100 30k"),
    ("BC-H8 eggplant sub50-div n_obs=1 · DQC-chunk · ddpm · softmax",  "bc_dqc_chunk", "bcH8sub50nobs1_50k_chunk8_predh8_roll4_ddpm_softmax",  "DQC-chunk", "ddpm", "softmax", "eggpl sub50 50k"),
    ("BC-H8 eggplant sub25-div n_obs=1 · DQC-chunk · ddpm · softmax",  "bc_dqc_chunk", "bcH8sub25nobs1_30k_chunk8_predh8_roll4_ddpm_softmax",  "DQC-chunk", "ddpm", "softmax", "eggpl sub25 30k"),
    ("BC-H8 stack sub100-div n_obs=1 · DQC-chunk · ddpm · softmax", "bc_dqc_chunk", "bcH8stack_sub100nobs1_40k_chunk8_predh8_roll4_ddpm_softmax", "DQC-chunk", "ddpm", "softmax", "stack sub100_nobs1 40k"),
    ("BC-H8 stack sub50-div n_obs=1 · DQC-chunk · ddpm · softmax",  "bc_dqc_chunk", "bcH8stack_sub50nobs1_30k_chunk8_predh8_roll4_ddpm_softmax",  "DQC-chunk", "ddpm", "softmax", "stack sub50_nobs1 30k"),
    ("BC-H8 stack sub25-div n_obs=1 · DQC-chunk · ddpm · softmax",  "bc_dqc_chunk", "bcH8stack_sub25nobs1_40k_chunk8_predh8_roll4_ddpm_softmax",  "DQC-chunk", "ddpm", "softmax", "stack sub25_nobs1 40k"),
    ("BC-H8 spoon sub100-div n_obs=1 · DQC-chunk · ddpm · softmax", "bc_dqc_chunk", "bcH8spoon_sub100nobs1_20k_chunk8_predh8_roll4_ddpm_softmax", "DQC-chunk", "ddpm", "softmax", "spoon sub100_nobs1 20k"),
    ("BC-H8 spoon sub50-div n_obs=1 · DQC-chunk · ddpm · softmax",  "bc_dqc_chunk", "bcH8spoon_sub50nobs1_10k_chunk8_predh8_roll4_ddpm_softmax",  "DQC-chunk", "ddpm", "softmax", "spoon sub50_nobs1 10k"),
    ("BC-H8 spoon sub25-div n_obs=1 · DQC-chunk · ddpm · softmax",  "bc_dqc_chunk", "bcH8spoon_sub25nobs1_10k_chunk8_predh8_roll4_ddpm_softmax",  "DQC-chunk", "ddpm", "softmax", "spoon sub25_nobs1 10k"),
]
# ARCHIVED BC: H16 (all), non-n_obs=1 H8 (jun12 ddim/ddpm), MLP/Transformer denoisers.
ARCHIVED_BC_SWEEPS = [
    ("BC-H16 · DQC-chunk · ddpm · softmax", "search_bc_sweep", "bc_dqcchunk_h16_na8/softmaxT1.0_ddpm", "DQC-chunk", "ddpm", "softmax", "jun11"),
    ("BC-H16 · LLaVA · ddpm · softmax",     "search_bc_sweep", "bc_llava_h16_na1/softmaxT1.0_ddpm",   "LLaVA",     "ddpm", "softmax", "jun11"),
    ("BC-H16 · LLaVA · ddim · argmax",      "search_bc_sweep", "bc/argmax_ddim",                      "LLaVA",     "ddim", "argmax",  "jun05b (OLD ckpt)"),
    ("BC-H16 · LLaVA · ddpm · argmax",      "search_bc_sweep", "bc/argmax_ddpm",                      "LLaVA",     "ddpm", "argmax",  "jun05b (OLD ckpt)"),
    ("BC-H8 · DQC-chunk · ddim · argmax (cancelled)",   "bc_dqc_chunk", "bcH8_chunk8_predh8_roll4_ddim",          "DQC-chunk", "ddim", "argmax",  "jun12"),
    ("BC-H8 · DQC-chunk · ddpm · argmax (cancelled)",   "bc_dqc_chunk", "bcH8_chunk8_predh8_roll4_ddpm",          "DQC-chunk", "ddpm", "argmax",  "jun12"),
    ("BC-H8 · DQC-chunk · ddim · softmax",  "bc_dqc_chunk", "bcH8_chunk8_predh8_roll4_softmax_ddim",  "DQC-chunk", "ddim", "softmax", "jun12"),
    ("BC-H8 · DQC-chunk · ddpm · softmax",  "bc_dqc_chunk", "bcH8_chunk8_predh8_roll4_softmax_ddpm",  "DQC-chunk", "ddpm", "softmax", "jun12 (unet)"),
    ("BC-H8 MLP-denoiser · DQC · ddpm · softmax",         "bc_dqc_chunk", "bc_mlp_h8_chunk8_predh8_roll4_softmax_ddpm",         "DQC-chunk", "ddpm", "softmax", "mlp_h8"),
    ("BC-H8 Transformer-denoiser · DQC · ddpm · softmax", "bc_dqc_chunk", "bc_transformer_h8_chunk8_predh8_roll4_softmax_ddpm", "DQC-chunk", "ddpm", "softmax", "transformer_h8"),
]

# BC n=1 base SR by training checkpoint (raw policy, ddim, no verifier). Cells:
# search_bc_sweep/<prefix>_<tag>/argmax_ddim_n1_seed1000_ep50. (display, prefix)
# KEPT n=1 base SR: BC-H8 filtered-699 + the n_obs=1 (search-matched) rows.
KEPT_BC_N1_BY_CKPT = [
    ("BC-H8 filtered-699",             "unet50k_h8_filt"),
    ("BC-H8 sub100-diverse n_obs=1",   "unet50k_h8_sub100_nobs1"),
    ("BC-H8 sub50-diverse n_obs=1",    "unet50k_h8_sub50_nobs1"),
    ("BC-H8 sub25-diverse n_obs=1",    "unet50k_h8_sub25_nobs1"),
    # NEW TASKS (different SimplerEnv task — SR NOT comparable to eggplant above)
    ("stack_cube sub100 n_obs=1",      "unet50k_h8_stack_sub100_nobs1"),
    ("stack_cube sub50 n_obs=1",       "unet50k_h8_stack_sub50_nobs1"),
    ("stack_cube sub25 n_obs=1",       "unet50k_h8_stack_sub25_nobs1"),
    ("spoon_on_towel sub100 n_obs=1",  "unet50k_h8_spoon_sub100_nobs1"),
    ("spoon_on_towel sub50 n_obs=1",   "unet50k_h8_spoon_sub50_nobs1"),
    ("spoon_on_towel sub25 n_obs=1",   "unet50k_h8_spoon_sub25_nobs1"),
]
# ARCHIVED n=1 base SR: H16-filtered + the n_obs=2 diverse subsets (superseded by n_obs=1).
ARCHIVED_BC_N1_BY_CKPT = [
    ("BC-H16 filtered-699",            "unet50k_filt"),
    ("BC-H8 sub100-diverse (n_obs=2)", "unet50k_h8_sub100"),
    ("BC-H8 sub50-diverse (n_obs=2)",  "unet50k_h8_sub50"),
    ("BC-H8 sub25-diverse (n_obs=2)",  "unet50k_h8_sub25"),
]
BC_N1_TAGS = ["10k", "20k", "30k", "40k", "50k"]


def load_bon_sweep(sub: str, prefix: str) -> dict[int, tuple[int, int]]:
    """{N: (n_success, n_episodes)} from each cell's eval_log.json."""
    out: dict[int, tuple[int, int]] = {}
    for n in BON_N:
        d = load_json(EVAL / sub / f"{prefix}_n{n}_seed1000_ep50" / "eval_log.json")
        if not d:
            continue
        succ, ep = d.get("num_successes"), d.get("num_episodes")
        if succ is None or not ep:
            continue
        out[n] = (int(succ), int(ep))
    return out


def _cell(t: tuple[int, int] | None) -> str:
    if not t or not t[1]:
        return "—"
    return f"{100.0 * t[0] / t[1]:.0f}{'' if t[1] >= 50 else '*'}"


# BC argmax sweeps were cancelled (softmax is now the default selection). Their
# unfinished cells render "x" (stopped), not "—" (pending), so the table is honest
# that no more data is coming for them.
CANCELLED_BC_PREFIXES = {
    "bcH8_chunk8_predh8_roll4_ddim",
    "bcH8_chunk8_predh8_roll4_ddpm",
}


def _cell_x(t: tuple[int, int] | None, cancelled: bool) -> str:
    if cancelled and (not t or not t[1]):
        return "x"
    return _cell(t)


def _nonempty(pairs):
    """(meta, data) pairs that have at least one populated cell — drops empty rows."""
    return [(m, d) for m, d in pairs if d]


def render_matrix(search_pairs, bc_pairs) -> str:
    L = ["## 3. Full Best-of-N matrix — filtered / curated runs\n",
         "Success rate % per cell (50 ep/cell, seeds 1000–1049, predict-8/rollout-4). Shows only "
         "the **filtered** search-policy runs (rmse-diff on filtered/diverse data — eggplant, "
         "stack_cube, spoon_on_towel) and the curated BC baselines (**BC-H8 n_obs=1** search-matched "
         "+ **BC-H8 filtered-699**). Everything else — original diffusion/gaussian search, PG, rmse "
         "gauss, and the non-curated BC policies — is in the **§6 Archive**. Empty (never-run) rows "
         "are omitted. `*` = partial (<50 ep); `—` = not run; `x` = cancelled.\n",
         "| sweep | " + " | ".join(f"N{n}" for n in BON_N) + " |",
         "|" + "---|" * (len(BON_N) + 1)]
    for (_, label, sub, prefix), data in search_pairs:
        L.append("| " + label + " | " + " | ".join(_cell(data.get(n)) for n in BON_N) + " |")
    for meta, data in bc_pairs:
        canc = meta[2] in CANCELLED_BC_PREFIXES
        L.append("| " + meta[0] + " | " + " | ".join(_cell_x(data.get(n), canc) for n in BON_N) + " |")
    L.append("")
    return "\n".join(L)


def render_bc(bc_pairs) -> str:
    L = ["## 4. BC policies — Best-of-N (curated: H8 n_obs=1 + filtered)\n",
         "`DiffusionUnetImagePolicy`, DQC-chunk verifier (predict-8/rollout-4). Kept here: the "
         "**search-matched BC-H8 n_obs=1** policies (eggplant sub100/50/25) and the strong "
         "**BC-H8 filtered-699** baseline. Other BC variants (H16, non-n_obs=1 H8, MLP/Transformer "
         "denoisers) are in the **§6 Archive**. `*` = partial; `x` = cancelled.\n",
         "| policy · verifier · sampler · selection | ckpt | " + " | ".join(f"N{n}" for n in BON_N) + " |",
         "|" + "---|" * (len(BON_N) + 2)]
    for meta, data in bc_pairs:
        label, _, prefix, _, _, _, ckpt = meta
        canc = prefix in CANCELLED_BC_PREFIXES
        L.append(f"| {label} | {ckpt} | " + " | ".join(_cell_x(data.get(n), canc) for n in BON_N) + " |")
    L.append("")
    L.append("> ℹ️ **BC-H8 sweeps capped at N=64.** The ddpm N≥128 long-tail cells (17–80h each) "
             "were not run — no BoN benefit was seen at any N for H8 BC. `—` at N≥128 = intentionally "
             "not run.\n")
    L.append("> ℹ️ **stack / spoon rows are NEW tasks** (SimplerEnv `widowx_stack_cube` / "
             "`widowx_spoon_on_towel`, task-correct baked DQC verifier, each on its best n=1 ckpt) — "
             "their SR is **not** comparable to the eggplant rows above. As with the search policies, "
             "DQC-chunk BoN gives **no lift** over n=1 on these harder tasks.\n")
    return "\n".join(L)


def load_bc_n1(prefix: str) -> dict[str, tuple[int, int]]:
    """{tag: (n_success, n_episodes)} for the raw n=1 (ddim) cells per checkpoint."""
    out: dict[str, tuple[int, int]] = {}
    for t in BC_N1_TAGS:
        d = load_json(EVAL / "search_bc_sweep" / f"{prefix}_{t}"
                      / "argmax_ddim_n1_seed1000_ep50" / "eval_log.json")
        if d and d.get("num_episodes"):
            out[t] = (int(d.get("num_successes", 0)), int(d["num_episodes"]))
    return out


def render_bc_n1(pairs) -> str:
    L = ["## 4b. Filtered / n_obs=1 BC — n=1 base SR by checkpoint\n",
         "Raw n=1 policy (ddim, no verifier) per training checkpoint. **Filtered** = triple-filtered "
         "`strict_distance_success` (699 ep) or its diverse-start subsamples (sub100/50/25 — "
         "round-robin over 28 eggplant start cells). **n_obs=1** rows are the *search-matched "
         "baseline* (BC conditions on 1 obs like the search policy — the baseline the search policy "
         "must beat, not a BoN payoff). The H16-filtered and n_obs=2 subsets are in the §6 Archive. "
         "`*` = partial; `—` = not run/pending.\n",
         "| policy | " + " | ".join(BC_N1_TAGS) + " |",
         "|" + "---|" * (len(BC_N1_TAGS) + 1)]
    for (label, _), data in pairs:
        L.append(f"| {label} | " + " | ".join(_cell(data.get(t)) for t in BC_N1_TAGS) + " |")
    L.append("")
    L.append("> ℹ️ **stack_cube / spoon_on_towel** rows are NEW tasks (SimplerEnv `widowx_stack_cube` "
             "/ `widowx_spoon_on_towel`, n_obs=1, same generic BC config) — their SR is **not** "
             "comparable to eggplant. stack is a harder task (low n=1 SR); spoon is moderate.\n")
    return "\n".join(L)


def render_diffusion(pairs) -> str:
    L = ["## 5. DQC diffusion-search policy — Best-of-N by training checkpoint (filtered runs)\n",
         "`SearchPolicyRoboMonkeyDiffusion` + baked DQC chunk verifier (softmax, "
         "predict-8/rollout-4). Only the kept filtered rmse-diff runs (eggplant / stack_cube / "
         "spoon_on_towel); empty rows dropped. `*` = partial.\n",
         "| checkpoint | " + " | ".join(f"N{n}" for n in BON_N) + " | peak |",
         "|" + "---|" * (len(BON_N) + 2)]
    for (_, label, sub, prefix), data in pairs:
        pk = peak(data)
        L.append(f"| {label} | " + " | ".join(_cell(data.get(n)) for n in BON_N)
                 + f" | {(f'{pk[1]:.0f}% @ N={pk[0]}' if pk else '—')} |")
    L.append("")
    L.append("**Finding:** on the filtered/diverse data the eggplant rmse-diff search policy reaches "
             "~88–92% by 10–40k (full-699 and sub100 both peak 92%; sub50 ~84%, tiny sub25 ~58%). "
             "stack_cube stays low (~40–52%, a hard task) and spoon_on_towel is moderate (~44–56%).\n")
    return "\n".join(L)


def render_archive(arch_search, arch_bc, arch_bc_n1, v2_data) -> str:
    L = ["## 6. Archive\n",
         "Runs kept for the record but not part of the headline comparison: older/superseded "
         "search-policy variants, non-curated BC policies, and the abandoned v2-critic PG runs. "
         "Empty (never-run) milestone rows are omitted except in the v2 table (§6c), where `—` "
         "is itself the record.\n"]

    # 6a — archived search-policy sweeps
    L.append("### 6a. Archived search-policy BoN — original diffusion/gaussian search, PG (base critic), rmse gauss\n")
    L.append("Preceded the filtered rmse-diff runs: original diffusion-search on the unfiltered data "
             "(10k/20k/33k, argmax + softmax), gaussian-search, the PG base-critic runs, and the "
             "gaussian rmse variant. **Finding (diffusion-search):** test-time search helps a *weak* "
             "base policy (10k/20k climb to ~84–86%) but not the *saturated* 33k (N=1≈94%; search "
             "drifts noisily 70–90). All eggplant, DQC-chunk verifier.\n")
    L.append("| sweep | " + " | ".join(f"N{n}" for n in BON_N) + " |")
    L.append("|" + "---|" * (len(BON_N) + 1))
    for (_, label, sub, prefix), data in arch_search:
        L.append("| " + label + " | " + " | ".join(_cell(data.get(n)) for n in BON_N) + " |")
    L.append("")

    # 6b — archived BC policies (BoN + n=1 base)
    L.append("### 6b. Archived BC policies — H16, non-n_obs=1 H8, denoiser variants\n")
    L.append("BC variants outside the curated set: **BC-H16** (all samplers/verifiers), the "
             "**non-n_obs=1 BC-H8** jun12 runs (ddim/ddpm, argmax/softmax), and the **MLP / "
             "Transformer denoiser** ablations. `x` = cancelled (argmax stopped — softmax is the "
             "default selection).\n")
    L.append("| policy · verifier · sampler · selection | ckpt | " + " | ".join(f"N{n}" for n in BON_N) + " |")
    L.append("|" + "---|" * (len(BON_N) + 2))
    for meta, data in arch_bc:
        label, _, prefix, _, _, _, ckpt = meta
        canc = prefix in CANCELLED_BC_PREFIXES
        L.append(f"| {label} | {ckpt} | " + " | ".join(_cell_x(data.get(n), canc) for n in BON_N) + " |")
    L.append("")
    L.append("> ⚠️ The two `argmax` BC-H16·LLaVA rows use the **older jun05b checkpoint** (sampler "
             "controls); the other BC-H16 rows use the canonical **jun11** ckpt, BC-H8 uses **jun12**. "
             "Compare verifier/sampler/selection within a ckpt, not across.\n")
    if arch_bc_n1:
        L.append("Archived n=1 base SR (ddim, no verifier) — H16-filtered and the n_obs=2 diverse "
                 "subsets (superseded by n_obs=1 in §4b):\n")
        L.append("| policy | " + " | ".join(BC_N1_TAGS) + " |")
        L.append("|" + "---|" * (len(BC_N1_TAGS) + 1))
        for (label, _), data in arch_bc_n1:
            L.append(f"| {label} | " + " | ".join(_cell(data.get(t)) for t in BC_N1_TAGS) + " |")
        L.append("")
        L.append("> ℹ️ Eggplant diverse subsets ~match the full 699-ep set at 7–28× less data (n_obs=2 "
                 "sub50 hits 0.98 @ 30k). **n_obs=1 weakens the raw policy vs n_obs=2** (less obs "
                 "context; worst on the tiny sub25). sub100 (n_obs=2) stopped at 44k, sub50 at 30k.\n")

    # 6c — abandoned v2-critic PG runs
    L.append("### 6c. Abandoned v2-critic PG runs\n")
    L.append("**What the v2 critic was.** The DQC chunk Q-critic was *finetuned* on additional "
             "eggplant data (run `v2_all`, `params_50000.pkl`) to try to sharpen its action ranking, "
             "vs. the original **base** critic (`sd000s_137913…`, `params_100000.pkl`). Two of the "
             "four PG search-policy runs used this v2 critic as **both** the policy-gradient training "
             "signal **and** the baked Best-of-N verifier: `PG gauss+v2` and `PG diff+v2`.\n")
    L.append("**Why they were abandoned.** The v2 critic is a **broken BoN verifier** — selecting "
             "candidates by its score actively picks *failing* actions. Both v2 runs collapse to "
             "~0% success by N≥2–8, and this held from 10k to 20k (more training did not fix it), "
             "while the base-critic runs show no such pathology. So the two v2 trainings and all v2 "
             "checkpoint evals were **stopped**; the partial data collected before stopping is kept "
             "below as the record (`—` = never run). Conclusion: **use the base critic, not v2.**\n")
    L.append("| sweep | " + " | ".join(f"N{n}" for n in BON_N) + " |")
    L.append("|" + "---|" * (len(BON_N) + 1))
    for (_, label, sub, prefix), data in zip(ARCHIVED_V2_SWEEPS, v2_data):
        L.append("| " + label + " | " + " | ".join(_cell(data.get(n)) for n in BON_N) + " |")
    L.append("")
    return "\n".join(L)


def plot_bon(series_list, title: str, path: Path, ylim=(0, 100)) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[plot] matplotlib unavailable ({e}); skipping {path.name}")
        return False
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    markers = "os^Dv><pP*hX"
    plotted = 0
    for i, (label, data) in enumerate(series_list):
        xs = [n for n in BON_N if n in data and data[n][1]]
        ys = [100.0 * data[n][0] / data[n][1] for n in xs]
        if xs:
            ax.plot(xs, ys, marker=markers[i % len(markers)], label=label, alpha=0.85)
            plotted += 1
    if not plotted:
        plt.close(fig)
        return False
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N (Best-of-N samples)")
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"[plot] wrote {path}")
    return True


# -----------------------------------------------------------------------------
# 500-step (max_steps=500) sub50 re-runs, spliced INLINE right after the matching
# 120-step checkpoint row (same policy + ckpt) so the tables show longer-episode SR
# next to the default-120 SR. Cells stage under
# data/eval/_500step_runs/<kind>_<task>_<ck>_n<N>/ (search matches its before-grid
# N=1..64 stack/spoon, 1..512 eggplant; BC 1,8,64). Tagged `· 500st`.
_500_TASKDIR = {"stack_cube": "stack", "spoon_on_towel": "spoon", "put_eggplant_in_basket": "eggplant"}
# kept_search key -> (500-step display label, task, ckpt-tag)
_500_AFTER_SEARCH = {
    "stack_sub50_20k": ("stack diff (sub50-div) 20k · 500st",             "stack_cube",             "20k"),
    "spoon_sub50_30k": ("spoon diff (sub50-div) 30k · 500st",             "spoon_on_towel",         "30k"),
    "rmse_sub50_70k":  ("rmse diff eggplant (sub50-diverse) 70k · 500st", "put_eggplant_in_basket", "70k"),
}
# kept_bc prefix -> (500-step label, ckpt label, task, ckpt-tag)
_500_AFTER_BC = {
    "bcH8stack_sub50nobs1_30k_chunk8_predh8_roll4_ddpm_softmax":
        ("BC-H8 stack sub50-div n_obs=1 · DQC-chunk · ddpm · softmax · 500st", "stack sub50_nobs1 30k · 500st", "stack_cube",             "30k"),
    "bcH8spoon_sub50nobs1_10k_chunk8_predh8_roll4_ddpm_softmax":
        ("BC-H8 spoon sub50-div n_obs=1 · DQC-chunk · ddpm · softmax · 500st", "spoon sub50_nobs1 10k · 500st", "spoon_on_towel",         "10k"),
    "bcH8sub50nobs1_50k_chunk8_predh8_roll4_ddpm_softmax":
        ("BC-H8 eggplant sub50-div n_obs=1 · DQC-chunk · ddpm · softmax · 500st", "eggpl sub50 50k · 500st",    "put_eggplant_in_basket", "50k"),
}


def _load_500(task: str, kind: str, ck: str) -> dict[int, tuple[int, int]]:
    """{N: (n_success, n_episodes)} for the 500-step cells (all BON_N present on disk)."""
    td = _500_TASKDIR[task]
    kd = "search" if kind == "search" else "bc"
    out: dict[int, tuple[int, int]] = {}
    for n in BON_N:
        d = load_json(EVAL / "_500step_runs" / f"{kd}_{td}_{ck}_n{n}" / "eval_log.json")
        if not d:
            continue
        succ, ep = d.get("num_successes"), d.get("num_episodes")
        if succ is None or not ep:
            continue
        out[n] = (int(succ), int(ep))
    return out


def splice_500_search(pairs):
    """Insert each 500-step search row right after its matching 120-step (meta,data)."""
    out = []
    for meta, data in pairs:
        out.append((meta, data))
        m = _500_AFTER_SEARCH.get(meta[0])
        if m:
            lbl, task, ck = m
            out.append(((f"{meta[0]}_500st", lbl, "_500step_runs", ""), _load_500(task, "search", ck)))
    return out


def splice_500_bc(pairs):
    """Insert each 500-step BC row right after its matching 120-step (meta,data)."""
    out = []
    for meta, data in pairs:
        out.append((meta, data))
        m = _500_AFTER_BC.get(meta[2])
        if m:
            lbl, cklbl, task, ck = m
            m2 = (lbl, meta[1], f"{meta[2]}__500st", meta[3], meta[4], meta[5], cklbl)
            out.append((m2, _load_500(task, "bc", ck)))
    return out


# -----------------------------------------------------------------------------
# SEEDED-selection reruns (the up-to-date, reproducible numbers). Softmax selection
# is now seeded from replan_seed. Cells: data/eval/_seeded/<task>_<sub>_search_<Mk>k
# _st{500,120}_n{N}_seed1000_ep50/ and _seeded/<task>_<sub>_bc_st500_n{N}_.../. N=1
# @120 reuses the OLD deterministic milestone cell (N=1 has no selection).
SEEDED_N = [1, 2, 4, 8, 16, 32, 64]
SEEDED_MILES = [10, 20, 30, 50, 60, 70, 90, 100]
SEEDED_POL = [("stack", "sub50"), ("stack", "sub100"), ("spoon", "sub50"),
              ("spoon", "sub100"), ("eggplant", "sub50"), ("eggplant", "sub100")]
# (task,sub) -> (odir,label) of the OLD unseeded 120-step milestone cells (N=1 reuse)
_OLD_N1 = {
    ("stack", "sub50"): ("stack_cube_search", "stack_sub50"),
    ("stack", "sub100"): ("stack_cube_search", "stack_sub100"),
    ("spoon", "sub50"): ("spoon_towel_search", "spoon_sub50"),
    ("spoon", "sub100"): ("spoon_towel_search", "spoon_sub100"),
    ("eggplant", "sub50"): ("filtered_sub50_diverse_eggplant", "rmse_sub50"),
    ("eggplant", "sub100"): ("search_dqc_chunk", "rmse_sub100"),
}


def _seeded_cell(task, sub, kind, Mk, sl, N):
    if kind == "search":
        p = EVAL / "_seeded" / f"{task}_{sub}_search_{Mk}k_st{sl}_n{N}_seed1000_ep50" / "eval_log.json"
    else:
        p = EVAL / "_seeded" / f"{task}_{sub}_bc_st{sl}_n{N}_seed1000_ep50" / "eval_log.json"
    d = load_json(p)
    if (not d or not d.get("num_episodes")) and kind == "search" and sl == 120 and N == 1:
        odir, label = _OLD_N1[(task, sub)]                       # reuse deterministic old N=1
        d = load_json(EVAL / odir / f"{label}_{Mk}k_chunk8_predh8_roll4_softmax_n1_seed1000_ep50" / "eval_log.json")
    if not d or not d.get("num_episodes"):
        return None
    return (int(d.get("num_successes", 0)), int(d["num_episodes"]))


def render_orig500() -> str:
    """Bottom section: the ORIGINAL 500-step sub50 runs (unseeded selection, superseded)."""
    N7 = BON_N[:7]  # 1..64
    L = ["## 7. Original 500-step runs — sub50, UNSEEDED selection (superseded, stopped)\n",
         "> ⚠️ The FIRST 500-step episode-length runs on the three sub50 policies, done BEFORE the "
         "softmax-selection seeding fix (selection reseeded from OS entropy → not reproducible). These "
         "runs have been **stopped**; kept here for the record only. The reproducible 500-vs-120 "
         "numbers are in the **⭐ seeded table at the top**. Same seeds (1000-1049), predict-8/"
         "rollout-4, DQC-chunk verifier, MAX_STEPS=500. `—` = never run.\n",
         "| policy | ckpt | " + " | ".join(f"N{n}" for n in N7) + " | peak |",
         "|" + "---|" * (len(N7) + 3)]   # 3 non-N cols: policy, ckpt, peak
    for kind, rows in (("search", [("stack_cube", "20k"), ("spoon_on_towel", "30k"), ("put_eggplant_in_basket", "70k")]),
                       ("BC",     [("stack_cube", "30k"), ("spoon_on_towel", "10k"), ("put_eggplant_in_basket", "50k")])):
        for task, ck in rows:
            data = _load_500(task, "search" if kind == "search" else "bc", ck)
            pk = peak(data)
            L.append(f"| {task} {kind} | {ck} | " + " | ".join(_cell(data.get(n)) for n in N7)
                     + f" | {(f'{pk[1]:.0f}% @N={pk[0]}' if pk else '—')} |")
    L.append("")
    return "\n".join(L)


def render_seeded() -> str:
    # native env-TimeLimit cap per task (the true horizon these pre-fix runs hit).
    NATIVE = {"stack": 60, "spoon": 60, "eggplant": 120}
    L = ["## Archive — SEEDED reruns that expose the NON-DETERMINISM issue (env-capped, pre-fix)\n",
         "> ⚠️ These runs used the SAME eval seeds (1000-1049) with the seeded softmax selection, but "
         "predate the **cuDNN/JAX determinism fix** — so a rerun of the same (seed, ckpt, N) still "
         "diverges from residual GPU-kernel nondeterminism (near-tie selection flips). They also predate "
         "the **env-TimeLimit fix**: the SAPIEN env truncates at the task-native horizon (stack/spoon "
         "**60**, eggplant **120**) regardless of MAX_STEPS, so the earlier '500'/'120' labels were both "
         "really that native cap — **relabeled here to the true step count**. Kept only to document the "
         "non-determinism; NOT a final result. Deterministic + true-horizon reruns pending.\n",
         "| policy · steps | ckpt | " + " | ".join(f"N{n}" for n in SEEDED_N) + " | peak |",
         "|" + "---|" * (len(SEEDED_N) + 3)]   # 3 non-N cols: policy, ckpt, peak
    n_rows = 0
    for task, sub in SEEDED_POL:
        cap = NATIVE[task]; disp = f"{task} {sub}"
        for Mk in SEEDED_MILES:
            # st500 & st120 are the SAME native-cap condition -> take whichever has data
            row = {N: (_seeded_cell(task, sub, "search", Mk, 120, N)
                       or _seeded_cell(task, sub, "search", Mk, 500, N)) for N in SEEDED_N}
            if not any(row.values()):
                continue
            pk = peak({n: v for n, v in row.items() if v})
            L.append(f"| {disp} {cap}st | {Mk}k | " + " | ".join(_cell(row[n]) for n in SEEDED_N)
                     + f" | {(f'{pk[1]:.0f}% @N={pk[0]}' if pk else '—')} |")
            n_rows += 1
        row = {N: _seeded_cell(task, sub, "bc", None, 500, N) for N in SEEDED_N}
        if any(row.values()):
            pk = peak({n: v for n, v in row.items() if v})
            L.append(f"| {disp} bc·{cap}st | — | " + " | ".join(_cell(row[n]) for n in SEEDED_N)
                     + f" | {(f'{pk[1]:.0f}% @N={pk[0]}' if pk else '—')} |")
            n_rows += 1
    if n_rows == 0:
        L.append("| _(no seeded cells)_ |  | " + " | ".join("—" for _ in SEEDED_N) + " | — |")
    return "\n".join(L)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # section 1 (reproduction, per-seed)
    c1_llava = parse_repro_summary(C1_LLAVA_SUMMARY)
    c1_dqc = load_c1_dqc_seeds()
    # section 2 (BoN sweep, seed=1)
    llava = load_progress(LLAVA_PROGRESS)
    dqc = load_progress(DQC_PROGRESS)
    if not c1_llava:
        print(f"[warn] no C1 LLaVA results at {C1_LLAVA_SUMMARY}")
    if not llava:
        print(f"[warn] no §2 LLaVA results at {LLAVA_PROGRESS}")
    if not dqc:
        print(f"[warn] no §2 single-step DQC results at {DQC_PROGRESS}")

    # §3-6 (search-policy & BC BoN sweeps), split into KEPT (filtered/curated) and
    # ARCHIVED. Empty (never-run) rows are dropped from every table except the v2 one.
    kept_search = _nonempty((m, load_bon_sweep(m[2], m[3])) for m in KEPT_SEARCH_SWEEPS)
    arch_search = _nonempty((m, load_bon_sweep(m[2], m[3])) for m in ARCHIVED_SEARCH_SWEEPS)
    kept_bc = _nonempty((m, load_bon_sweep(m[1], m[2])) for m in KEPT_BC_SWEEPS)
    arch_bc = _nonempty((m, load_bon_sweep(m[1], m[2])) for m in ARCHIVED_BC_SWEEPS)
    kept_bc_n1 = _nonempty((m, load_bc_n1(m[1])) for m in KEPT_BC_N1_BY_CKPT)
    arch_bc_n1 = _nonempty((m, load_bc_n1(m[1])) for m in ARCHIVED_BC_N1_BY_CKPT)
    v2_data = [load_bon_sweep(sub, prefix) for (_, _, sub, prefix) in ARCHIVED_V2_SWEEPS]

    # combined lists (kept + archived) for the JSON dump and the scaling plots
    all_search = KEPT_SEARCH_SWEEPS + ARCHIVED_SEARCH_SWEEPS
    all_search_data = [load_bon_sweep(m[2], m[3]) for m in all_search]
    all_bc = KEPT_BC_SWEEPS + ARCHIVED_BC_SWEEPS
    all_bc_data = [load_bon_sweep(m[1], m[2]) for m in all_bc]

    _ARCHIVE_BANNER = (
        "---\n\n## Archived tables — RNG-randomized softmax selection (pre-seed-fix)\n\n"
        "> ⚠️ Everything below (§1–§6) was produced **before** the softmax-selection seeding fix. "
        "The selection was reseeded from OS entropy on every replan, so a rerun of the same "
        "(seed, ckpt, N) picked different candidates and gave a different rollout — run-to-run "
        "success rates carry selection noise (see the 120-vs-500 divergence analysis; that noise, "
        "not episode length, drove most matched-N differences). Kept for reference. The reproducible, "
        "up-to-date numbers are in the **seeded** table at the top."
    )
    md = "\n\n".join([
        render_header().rstrip(),
        render_reproduction(c1_llava, c1_dqc).rstrip(),
        render_sweep(llava, dqc).rstrip(),
        # --- PRUNED 2026-07-08 ---------------------------------------------------
        # The search-policy + BC BoN tables (former §3-§7 + the seeded archive) were
        # all PRE-fix: env-capped at the task-native horizon (stack/spoon 60, eggplant
        # 120, not the requested MAX_STEPS) AND run with nondeterministic cuDNN/JAX
        # kernels. Their data folders were moved to data/eval/archive_gpu_stochastic/
        # and the full pre-fix doc is preserved at
        # archive_gpu_stochastic/search_bon_results_prefix.md. Deterministic,
        # true-horizon reruns (with the object->goal distance metric) are pending;
        # their results will repopulate this doc.
        "---\n\n## Search / BC BoN results — ARCHIVED (pre-determinism-fix, env-capped)\n\n"
        "> The stack/spoon/eggplant search-policy and BC Best-of-N tables were archived to "
        "`data/eval/archive_gpu_stochastic/` (data) + `search_bon_results_prefix.md` (full doc). "
        "They were env-capped (native horizon, not MAX_STEPS) and nondeterministic. "
        "Deterministic, true-horizon reruns on sub50 {spoon 90k, stack 60k, eggplant 90k} are in "
        "progress and will repopulate this section.",
    ]) + "\n"
    (OUT / "search_bon_results.md").write_text(md)
    print(f"[md] wrote {OUT / 'search_bon_results.md'}")
    # Mirror to data/eval/search_bon_results.md (top-level results doc) so it
    # stays in sync with the canonical _verifier_comparison/ copy on every tick.
    (EVAL / "search_bon_results.md").write_text(md)
    print(f"[md] mirrored {EVAL / 'search_bon_results.md'}")

    def jsonable(dd):
        return {str(k): {"success": v[0], "episodes": v[1]} for k, v in sorted(dd.items())}

    def seedable(dd):
        return {str(s): ({"success": t[0], "episodes": t[1]} if t else None)
                for s, t in sorted(dd.items())}
    js = {
        "task": "widowx_put_eggplant_in_basket", "init": 9,
        "reproduction": {
            "llava_aug": C1_LLAVA_AUG, "dqc_aug": C1_DQC_AUG,
            "llava_seeds": seedable(c1_llava),
            "single_step_dqc_seeds": seedable(c1_dqc),
        },
        "bon_sweep_seed1": {
            "llava": jsonable(llava),
            "single_step_dqc": jsonable(dqc),
        },
        "diffusion_search_bon": {
            m[0]: jsonable(data)
            for m, data in zip(all_search, all_search_data)
        },
        "bc_bon": {
            m[0]: {"ckpt": m[6], "verifier": m[3], "sampler": m[4],
                   "selection": m[5], "by_N": jsonable(data)}
            for m, data in zip(all_bc, all_bc_data)
        },
    }
    (OUT / "comparison.json").write_text(json.dumps(js, indent=2))
    print(f"[json] wrote {OUT / 'comparison.json'}")

    render_plot(llava, dqc, OUT / "openvla_llava_vs_dqc.png")
    plot_bon([(all_search[i][1], all_search_data[i]) for i in range(len(all_search))],
             "DQC diffusion-search — Best-of-N by checkpoint",
             OUT / "diffusion_search_bon.png")
    plot_bon([(all_bc[i][0], all_bc_data[i]) for i in range(len(all_bc))],
             "BC policies — Best-of-N (DQC-chunk & LLaVA)",
             OUT / "bc_bon.png")
    print("\n" + md)


if __name__ == "__main__":
    main()
