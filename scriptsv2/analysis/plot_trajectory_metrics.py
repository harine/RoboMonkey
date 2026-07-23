#!/usr/bin/env python3
"""Per-trajectory diagnostics + data dump for each visualized (video) seed.

For every closed-loop rollout npz under
  data/eval/train_test_analysis/<task>/rollout/<split>/search_q/*.npz
this writes, into a folder that shares the NAME of that seed's video
(data/eval/train_test_analysis/<task>/videos/<video-stem>/):
    action_mse.png          all three MSEs over time (see names below)
    action_mse_no_grip.png  same, gripper excluded (no all-dims line)
    ee_divergence.png       world-EE divergence in cm (matches the video legend)
    action_delta.png        per-dim (chosen - expert) raw action delta over time
    q_value.png             chosen-candidate Q over time (+ candidate min/mean/max band)
    trajectory.csv          per replan step (all scalars below)
    action_steps.csv        per env step: chosen_* / expert_* / delta_* (7 action dims)

and consolidates every seed into
    <task>/metrics/trajectory_metrics.csv   (all seeds, per replan step)
    <task>/metrics/trajectory_steps.csv     (all seeds, per env step)

MSE quantities (chosen = the executed argmax-Q candidate at each replan; a_pred = the
16 predicted candidate chunks; reference = expert_actions aligned at branch_t):
    max_q_action_all_dims_mse   chosen candidate MSE over all 7 dims
    max_q_no_gripper_mse        chosen candidate MSE over arm dims 0:6 (gripper dropped)
    max_q_gripper_only_mse      chosen candidate MSE over gripper dim only
    lowest_a_pred_mse           best-of-16 (min MSE to expert) over all 7 dims
    lowest_a_pred_no_grip_mse   best-of-16 (min MSE to expert) over arm dims 0:6
    qpick_gap                   max_q_action_all_dims_mse - lowest_a_pred_mse
    ee_div_cm                   ||policy EE - expert EE|| at this env step, in cm
                                (identical to the video's on-screen "divergence")

NOTE on spaces: action_delta / the *_mse columns are in raw ACTION space (per-step,
instantaneous). ee_div_cm is in world POSITION space (integrated) and is what the video
draws — a small per-step action delta still integrates into cm-scale EE drift.

Usage:
  python scriptsv2/analysis/plot_trajectory_metrics.py --task eggplant
  python scriptsv2/analysis/plot_trajectory_metrics.py --task eggplant --splits train
"""
import argparse, csv, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/mmfs1/home/harine/RoboMonkey/data/eval/train_test_analysis"
ARM = slice(0, 6); GRIP = slice(6, 7)
DIMS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip"]


def flat_chosen_expert(d):
    """Executed (chosen) and expert action chunks flattened to per-env-step arrays."""
    bt = d["branch_t"]; ca = d["candidate_actions"]; sel = d["selected_index"]
    exa = d["expert_actions"]; T, _, H, A = ca.shape
    ts, ch, ea = [], [], []
    for t in range(T):
        s = int(bt[t]); e = exa[s:s + H]; h = min(H, e.shape[0])
        if h == 0:
            continue
        ch.append(ca[t, int(sel[t])][:h]); ea.append(e[:h]); ts.extend(range(s, s + h))
    if not ch:
        return np.array([]), np.zeros((0, 7)), np.zeros((0, 7))
    return np.asarray(ts), np.concatenate(ch), np.concatenate(ea)


def per_replan(d):
    """Per replan step: env_t, the six MSE quantities, Q stats, world-EE divergence."""
    bt = d["branch_t"]; ca = d["candidate_actions"]; vals = d["values"]; sel = d["selected_index"]
    exa = d["expert_actions"]; ee = d["ee_xyz"]; exp_ee = d["expert_ee_world"]
    T, K, H, A = ca.shape
    R = []
    for t in range(T):
        s = int(bt[t]); e = exa[s:s + H]; h = min(H, e.shape[0])
        if h == 0:
            continue
        ch = ca[t, int(sel[t])][:h]; e = e[:h]
        cand = ca[t][:, :h, :]                                   # (K,h,7)
        cand_mse_all = ((cand - e[None]) ** 2).mean(axis=(1, 2))          # (K,)
        cand_mse_arm = ((cand[:, :, ARM] - e[None][:, :, ARM]) ** 2).mean(axis=(1, 2))
        max_q_all = float(((ch - e) ** 2).mean())
        low_all = float(cand_mse_all.min())
        j = int(min(s, exp_ee.shape[0] - 1))
        R.append(dict(
            replan=t, env_t=s, selected_index=int(sel[t]),
            max_q_action_all_dims_mse=max_q_all,
            max_q_no_gripper_mse=float(((ch[:, ARM] - e[:, ARM]) ** 2).mean()),
            max_q_gripper_only_mse=float(((ch[:, GRIP] - e[:, GRIP]) ** 2).mean()),
            lowest_a_pred_mse=low_all,
            lowest_a_pred_no_grip_mse=float(cand_mse_arm.min()),
            qpick_gap=max_q_all - low_all,
            ee_div_cm=float(np.linalg.norm(ee[t] - exp_ee[j]) * 100.0),
            q_sel=float(vals[t, int(sel[t])]),
            q_min=float(vals[t].min()), q_mean=float(vals[t].mean()), q_max=float(vals[t].max()),
        ))
    return R


def find_video_stem(task, split, seed):
    hits = sorted(glob.glob(f"{BASE}/{task}/videos/{split}_seed{seed}_N*_*.mp4"))
    return os.path.splitext(os.path.basename(hits[0]))[0] if hits else None


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def plot_seed(task, npz_path, split):
    d = np.load(npz_path, allow_pickle=True)
    seed = int(d["seed"]); succ = int(d["success"]); trunc = int(d["truncated"])
    n_steps = int(d["num_steps"]); demo_ep = int(d["demo_ep_idx"])
    stem = find_video_stem(task, split, seed)
    if stem is None:
        return None
    outdir = f"{BASE}/{task}/videos/{stem}"; os.makedirs(outdir, exist_ok=True)
    status = "SUCCESS" if succ else ("TRUNCATED" if trunc else "FAIL")
    hdr = f"{task} · {split} · seed {seed} · {status}"

    R = per_replan(d); et = [r["env_t"] for r in R]
    ts, ch, ea = flat_chosen_expert(d)

    # 1) action MSE over time — all three quantities
    fig, ax = plt.subplots(figsize=(7.2, 4))
    ax.plot(et, [r["max_q_action_all_dims_mse"] for r in R], "-o", color="#d1495b", lw=2, label="max_q_action_all_dims_mse")
    ax.plot(et, [r["lowest_a_pred_mse"] for r in R], "--s", color="#2e86ab", lw=1.6, label="lowest_a_pred_mse")
    ax.plot(et, [r["max_q_no_gripper_mse"] for r in R], ":", color="#6a4c93", lw=1.4, label="max_q_no_gripper_mse")
    ax.set_xlabel("env timestep (replan branch)"); ax.set_ylabel("action MSE vs expert a*")
    ax.set_title(f"Action MSE over trajectory\n{hdr}"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{outdir}/action_mse.png", dpi=120); plt.close(fig)

    # 1b) no-gripper variant — drops the all-dims line
    fig, ax = plt.subplots(figsize=(7.2, 4))
    ax.plot(et, [r["max_q_no_gripper_mse"] for r in R], "-o", color="#6a4c93", lw=2, label="max_q_no_gripper_mse")
    ax.plot(et, [r["lowest_a_pred_no_grip_mse"] for r in R], "--s", color="#2e86ab", lw=1.6, label="lowest_a_pred_no_grip_mse")
    ax.set_xlabel("env timestep (replan branch)"); ax.set_ylabel("arm-only action MSE vs expert a*")
    ax.set_title(f"Action MSE over trajectory — gripper excluded\n{hdr}"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{outdir}/action_mse_no_grip.png", dpi=120); plt.close(fig)

    # 2) world-EE divergence (cm) — matches the video's on-screen "divergence"
    fig, ax = plt.subplots(figsize=(7.2, 4))
    ax.plot(et, [r["ee_div_cm"] for r in R], "-o", color="#118ab2", lw=2, label="||policy EE − expert EE||")
    ax.set_xlabel("env timestep (replan branch)"); ax.set_ylabel("EE divergence (cm)")
    ax.set_title(f"World-EE divergence over trajectory (matches video)\n{hdr}")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{outdir}/ee_divergence.png", dpi=120); plt.close(fig)

    # 3) per-dimension (chosen - expert) raw action delta over time
    fig, ax = plt.subplots(figsize=(8, 4.2))
    if ts.size:
        delta = ch - ea
        for j, name in enumerate(DIMS):
            ax.plot(ts, delta[:, j], "-", lw=1.4, label=name)
        ax.axhline(0, color="k", lw=.6, alpha=.5)
    ax.set_xlabel("env timestep"); ax.set_ylabel("chosen − expert  (raw action units)")
    ax.set_title(f"Per-dimension action delta (raw action space; cf. ee_divergence for world space)\n{hdr}")
    ax.grid(alpha=.3); ax.legend(ncol=7, fontsize=7, loc="upper center")
    fig.tight_layout(); fig.savefig(f"{outdir}/action_delta.png", dpi=120); plt.close(fig)

    # 4) Q value over time
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(et, [r["q_min"] for r in R], [r["q_max"] for r in R],
                    color="#cccccc", alpha=.6, label="candidate min–max")
    ax.plot(et, [r["q_mean"] for r in R], "--", color="#888888", lw=1.2, label="candidate mean")
    ax.plot(et, [r["q_sel"] for r in R], "-o", color="#e07a1f", lw=2, label="chosen (argmax-Q)")
    ax.set_xlabel("env timestep (replan branch)"); ax.set_ylabel("verifier Q")
    ax.set_title(f"Q value over trajectory\n{hdr}"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{outdir}/q_value.png", dpi=120); plt.close(fig)

    # ---- data dump ----
    meta = dict(task=task, split=split, seed=seed, status=status, success=succ,
                truncated=trunc, num_steps=n_steps, demo_ep_idx=demo_ep, video=stem)
    replan_rows = [{**meta, **r} for r in R]
    step_rows = []
    for i in range(ts.shape[0]):
        row = {**meta, "env_t": int(ts[i])}
        for j, name in enumerate(DIMS):
            row[f"chosen_{name}"] = float(ch[i, j]); row[f"expert_{name}"] = float(ea[i, j])
            row[f"delta_{name}"] = float(ch[i, j] - ea[i, j])
        step_rows.append(row)

    rp_fields = ["task", "split", "seed", "status", "video", "replan", "env_t", "selected_index",
                 "max_q_action_all_dims_mse", "max_q_no_gripper_mse", "max_q_gripper_only_mse",
                 "lowest_a_pred_mse", "lowest_a_pred_no_grip_mse", "qpick_gap", "ee_div_cm",
                 "q_sel", "q_min", "q_mean", "q_max"]
    st_fields = ["task", "split", "seed", "status", "video", "env_t"] + \
                [f"{p}_{n}" for p in ("chosen", "expert", "delta") for n in DIMS]
    write_csv(f"{outdir}/trajectory.csv", replan_rows, rp_fields)
    write_csv(f"{outdir}/action_steps.csv", step_rows, st_fields)
    return stem, replan_rows, step_rows, rp_fields, st_fields


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    args = ap.parse_args()
    all_rp, all_st, rp_fields, st_fields = [], [], None, None
    n = 0
    for split in args.splits:
        for f in sorted(glob.glob(f"{BASE}/{args.task}/rollout/{split}/search_q/*.npz")):
            res = plot_seed(args.task, f, split)
            if not res:
                continue
            stem, rp, st, rp_fields, st_fields = res
            all_rp += rp; all_st += st; n += 1
            print(f"  [{args.task}/{split}] {os.path.basename(f)} -> videos/{stem}/  ({len(rp)} replans, {len(st)} steps)")
    if all_rp:
        md = f"{BASE}/{args.task}/metrics"; os.makedirs(md, exist_ok=True)
        write_csv(f"{md}/trajectory_metrics.csv", all_rp, rp_fields)
        write_csv(f"{md}/trajectory_steps.csv", all_st, st_fields)
        print(f"[plot_trajectory_metrics] {args.task}: {n} seeds -> "
              f"metrics/trajectory_metrics.csv ({len(all_rp)} rows), trajectory_steps.csv ({len(all_st)} rows)")


if __name__ == "__main__":
    main()
