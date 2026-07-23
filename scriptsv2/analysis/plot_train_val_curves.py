#!/usr/bin/env python
"""Analysis 1 (jul8 notes): train-vs-val curves for the sub50 BC + search policies on the
three tasks, parsed straight from each run's local logs.json.txt (NO wandb API).

Keys (verified): train_loss, val_loss  (both = diffusion DENOISING MSE); action-space MSE on a
TRAIN batch -> search: train_action_mse_error_min / _avg ; BC: train_action_mse_error. There is
no logged val ACTION mse (only val_loss denoising) -> that gap is filled offline in Step C.

Output: data/eval/jul8_notes/train_val_curves.png  (+ a small summary json).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SR = "/gscratch/robotics/harine/RoboMonkey_data/train"
BC = "/mmfs1/home/harine/RoboMonkey/diffusion_policy/data/train/bc_denoiser"
OUT = "/mmfs1/home/harine/RoboMonkey/data/eval/jul8_notes"

# task -> (search run dir, search peak step, bc run dir, bc peak step)
RUNS = {
    "eggplant": (f"{SR}/rmse_filtered/diff_search_rmse_sub50_diverse_data", 90000,
                 f"{BC}/eggplant_bc_unet_h8_50k_sub50_diverse_nobs1", 50000),
    "stack":    (f"{SR}/rmse_multitask/diff_search_rmse_stack_sub50_diverse", 60000,
                 f"{BC}/bc_unet_h8_50k_stack_sub50_nobs1", 30000),
    "spoon":    (f"{SR}/rmse_multitask/diff_search_rmse_spoon_sub50_diverse", 10000,
                 f"{BC}/bc_unet_h8_50k_spoon_sub50_nobs1", 10000),
}
C_SEARCH, C_BC = "#4C72B0", "#DD8452"   # blue = search, orange = BC


def load_logs(run_dir):
    steps, tl, vstep, vl = [], [], [], []
    amse = {}     # key -> (steps, vals)
    with open(os.path.join(run_dir, "logs.json.txt")) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            g = d.get("global_step")
            if g is None:
                continue
            if "train_loss" in d:
                steps.append(g); tl.append(d["train_loss"])
            if "val_loss" in d:
                vstep.append(g); vl.append(d["val_loss"])
            for k in ("train_action_mse_error", "train_action_mse_error_min",
                      "train_action_mse_error_avg"):
                if k in d:
                    amse.setdefault(k, ([], []))
                    amse[k][0].append(g); amse[k][1].append(d[k])
    return (np.array(steps), np.array(tl), np.array(vstep), np.array(vl),
            {k: (np.array(a), np.array(b)) for k, (a, b) in amse.items()})


def bin_mean(x, y, nbins=250):
    """Bin-average over global step -> kills the per-batch diffusion-timestep noise so the
    train-vs-val trend is legible."""
    if len(x) == 0:
        return x, y
    x = np.asarray(x, float); y = np.asarray(y, float)
    nbins = min(nbins, len(x))
    edges = np.linspace(x.min(), x.max(), nbins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nbins - 1)
    xs, ys = [], []
    for b in range(nbins):
        m = idx == b
        if m.any():
            xs.append(x[m].mean()); ys.append(y[m].mean())
    return np.array(xs), np.array(ys)


def main():
    os.makedirs(OUT, exist_ok=True)
    tasks = list(RUNS.keys())
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), squeeze=False)
    summary = {}

    for j, task in enumerate(tasks):
        s_dir, s_peak, b_dir, b_peak = RUNS[task]
        s = load_logs(s_dir); b = load_logs(b_dir)
        summary[task] = {}

        # ---- Row 0: denoising train vs val loss ----
        ax = axes[0][j]
        for (data, color, name, peak) in [(s, C_SEARCH, "search", s_peak),
                                          (b, C_BC, "BC", b_peak)]:
            steps, tl, vstep, vl, _ = data
            xs, ys = bin_mean(steps, tl)
            ax.plot(xs, ys, color=color, lw=1.8, label=f"{name} train")
            if len(vstep):
                xv, yv = bin_mean(vstep, vl, 120)
                ax.plot(xv, yv, color=color, lw=1.6, ls="--", alpha=0.95, label=f"{name} val")
            ax.axvline(peak, color=color, ls=":", lw=1.0, alpha=0.5)
            if len(vstep):
                # val loss at (nearest logged) peak step
                k = int(np.argmin(np.abs(vstep - peak)))
                summary[task][f"{name}_val_loss_at_peak"] = float(vl[k])
        ax.set_title(f"{task} — denoising loss (train solid, val dashed)")
        ax.set_xlabel("global step"); ax.set_ylabel("epsilon-MSE loss")
        ax.grid(alpha=0.25); ax.legend(fontsize=8, ncol=2)

        # ---- Row 1: train action MSE (sampled) ----
        ax = axes[1][j]
        amse_s = s[4]; amse_b = b[4]
        for key, color, ls, lab in [
                ("train_action_mse_error_min", C_SEARCH, "-", "search min-cand"),
                ("train_action_mse_error_avg", C_SEARCH, "--", "search avg-cand"),
                ("train_action_mse_error", C_BC, "-", "BC (single)")]:
            src = amse_s if color == C_SEARCH else amse_b
            if key in src:
                xs, ys = src[key]
                xs2, ys2 = bin_mean(xs, ys, 200)
                ax.plot(xs2, ys2, color=color, ls=ls, lw=1.6, label=lab)
        ax.axvline(s_peak, color=C_SEARCH, ls=":", lw=1.0, alpha=0.5)
        ax.axvline(b_peak, color=C_BC, ls=":", lw=1.0, alpha=0.5)
        ax.set_yscale("log")
        ax.set_title(f"{task} — train action MSE (on train batch)")
        ax.set_xlabel("global step"); ax.set_ylabel("action MSE (log)")
        ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=8)

    fig.suptitle("Analysis 1 — sub50 BC vs search: train/val denoising loss + train action MSE "
                 "(dotted verticals = selected peak ckpt)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = os.path.join(OUT, "train_val_curves.png")
    fig.savefig(p, dpi=130)
    json.dump(summary, open(os.path.join(OUT, "train_val_summary.json"), "w"), indent=2)
    print(f"[done] wrote {p}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
