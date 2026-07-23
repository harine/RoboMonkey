#!/usr/bin/env python3
"""SEED GATE: prove the reconstructed demo seeds really reproduce the demos' start states.

For every (task, split, demo) in scriptsv2/eval_search/demoseeds/<task>_<split>.json:
  * env.reset(seed=<reconstructed seed>)  -> read the start pose via build_obs_step
  * compare to the DATASET's stored pose at that episode's first step
    (zarr data/obs/<key>[out_start])

An exact match proves `seed = 100000 + START_INDEX(src_shard) + src_episode` is right --
in particular eggplant's state1 START_INDEX=1472. A mismatch means we would silently roll
out the WRONG start states, so this gates the closed-loop videos.

Pass iff max|delta| < --tol for every episode and every compared key.

Run in dqc_jax / simpler_env (MUJOCO_GL=osmesa). See validate_demo_seeds.sbatch.
  python scriptsv2/analysis/validate_demo_seeds.py [--tol 1e-4] [--tasks spoon ...]
Exit code 0 = all pass, 1 = any mismatch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scriptsv2" / "eval_diffusion"))
SEED_DIR = REPO / "scriptsv2" / "eval_search" / "demoseeds"

# env obs key -> zarr obs array name (identical names here, but be explicit)
COMPARE_KEYS = ["insertive_asset_pose", "receptive_asset_pose", "end_effector_pose"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-4, help="xyz tolerance (metres)")
    ap.add_argument("--quat-tol", type=float, default=1e-4,
                    help="quaternion elementwise tolerance. Eggplant is an irregular object "
                         "that SETTLES on reset, and settling depends on the sim's prior "
                         "history -- replaying only the demo seeds reproduces xyz to <1mm but "
                         "can leave a different orientation. Stack/spoon are exact.")
    ap.add_argument("--tasks", nargs="*", default=["eggplant", "spoon", "stack_cube"])
    ap.add_argument("--splits", nargs="*", default=["train", "test"])
    ap.add_argument("--out", default="data/eval/train_test_analysis/_seed_gate")
    a = ap.parse_args()

    import zarr
    from eval_diffusion import build_obs_step  # type: ignore
    import simpler_env  # noqa: F401
    from simpler_env import make as make_env

    report, n_fail = {}, 0
    for task in a.tasks:
        env = None
        for split in a.splits:
            meta = json.loads((SEED_DIR / f"{task}_{split}.json").read_text())
            if env is None:
                print(f"\n[gate] building env {meta['env_task']} for {task} ...", flush=True)
                env = make_env(meta["env_task"])
            z = zarr.open(meta["zarr"], "r")
            zobs = z["data"]["obs"]
            ep_ends = np.asarray(z["meta"]["episode_ends"][:])

            eps, n_ok = [], 0
            for row in meta["episodes"]:
                seed, ep_idx, out_start = row["seed"], row["ep_idx"], row["out_start"]
                # cross-check out_start against episode_ends (guards manifest drift)
                expect_start = 0 if ep_idx == 0 else int(ep_ends[ep_idx - 1])
                if expect_start != out_start:
                    print(f"  !! {task}/{split} ep{ep_idx}: out_start={out_start} "
                          f"but episode_ends implies {expect_start}")
                    n_fail += 1
                    continue

                obs, _ = env.reset(seed=int(seed))
                eb = getattr(env, "unwrapped", env)
                step_obs = build_obs_step(
                    env=eb, obs=obs,
                    prev_arm_action=np.zeros(6, np.float32),
                    prev_gripper_action=np.zeros(1, np.float32),
                    source_obj=getattr(eb, "episode_source_obj", None),
                    target_obj=getattr(eb, "episode_target_obj", None),
                    tcp_link=getattr(eb, "tcp", None),
                )
                # split xyz (position) from quat (orientation): eggplant settles on reset,
                # so its orientation can differ while position matches to sub-mm.
                dxyz, dquat = 0.0, 0.0
                for k in COMPARE_KEYS:
                    got = step_obs.get(k)
                    if got is None or k not in zobs:
                        continue
                    got = np.asarray(got, np.float32).ravel()
                    ref = np.asarray(zobs[k][out_start], np.float32).ravel()
                    n = min(got.size, ref.size)
                    dxyz = max(dxyz, float(np.max(np.abs(got[:3] - ref[:3]))))
                    if n >= 7:
                        dquat = max(dquat, float(np.max(np.abs(got[3:7] - ref[3:7]))))
                ok_ep = (dxyz <= a.tol) and (dquat <= a.quat_tol)
                eps.append({"ep_idx": ep_idx, "seed": seed, "d_xyz": dxyz,
                            "d_quat": dquat, "pass": ok_ep})
                n_ok += int(ok_ep)
                if not ok_ep:
                    n_fail += 1

            X = np.array([e["d_xyz"] for e in eps]) if eps else np.zeros(1)
            Qd = np.array([e["d_quat"] for e in eps]) if eps else np.zeros(1)
            ok = n_ok == len(eps)
            report[f"{task}/{split}"] = {
                "n": len(eps), "n_ok": n_ok, "pass": ok,
                "xyz": {"median": float(np.median(X)), "p90": float(np.quantile(X, .9)),
                        "max": float(X.max())},
                "quat": {"median": float(np.median(Qd)), "p90": float(np.quantile(Qd, .9)),
                         "max": float(Qd.max())},
                "episodes": eps,   # per-episode deltas -> pick the best demos for videos
            }
            print(f"[gate] {task:11s} {split:5s}: {n_ok}/{len(eps)} "
                  f"{'PASS' if ok else 'FAIL'}   "
                  f"xyz(med={np.median(X):.2e} max={X.max():.2e})  "
                  f"quat(med={np.median(Qd):.2e} max={Qd.max():.2e})", flush=True)
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "seed_gate_report.json").write_text(json.dumps(
        {"tol": a.tol, "n_fail": n_fail, "groups": report}, indent=2, default=str))
    total = sum(v["n"] for v in report.values())
    n_ok = sum(v["n_ok"] for v in report.values())
    print(f"\n[gate] TOTAL {n_ok}/{total} episodes match (tol={a.tol:g}) -> "
          f"{out}/seed_gate_report.json")
    if n_ok != total:
        print("[gate] ***  GATE FAILED — do NOT run the closed-loop videos  ***")
        sys.exit(1)
    print("[gate] *** GATE PASSED — seed reconstruction is exact ***")


if __name__ == "__main__":
    main()
