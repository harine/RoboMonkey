#!/usr/bin/env python
"""GO/NO-GO for the sim-rollout distance verifier: can a usable sim state be
RECONSTRUCTED from the recorded obs?

The sub50 zarr stores observations, not simulator state. A sim-rollout verifier must
`set_state` into a live env before stepping a candidate chunk, so the whole approach
rests on rebuilding a state vector out of what the dataset actually kept. This script
answers whether that reconstruction is faithful enough to trust, BEFORE any of the
training machinery is built against it.

LAYOUT (mani_skill2_real2sim/envs/sapien_env.py:653-680)
    state = for each actor        [pos3, quat4, lin_vel3, ang_vel3]          (13)
          + for each articulation [pos3, quat4, lin_vel3, ang_vel3, qpos, qvel]
A template is taken from a real `reset()`, then the slices we know are overwritten:
the eggplant and basket actor poses, and the robot's arm qpos/qvel.

WHAT IS NECESSARILY APPROXIMATE (and why this script exists):
  * object linear/angular velocities are NOT in the zarr -> zeroed. Harmless if the
    eggplant is resting or rigidly held, wrong if it is mid-fall or slipping. This is
    the single largest source of divergence and the main thing measured here.
  * gripper finger joints are not stored -> inferred from last_gripper_action.
  * quaternion convention of the recorded poses is ASSUMED to match sapien (wxyz);
    if it does not, orientation error will be large and obvious below.

THE TEST: reconstruct the state at window t, replay the DEMO's OWN next H actions,
and compare the resulting poses against the recorded obs at t+H. Replaying the demo's
actions is what makes this a clean measurement — any divergence is reconstruction
error, not policy error. If the eggplant lands within a few mm of where the dataset
says it did, the verifier's ranking can be trusted; if it is centimetres off, the
rollout is scoring a different world than the one the data came from.

Usage:
    python scriptsv2/train_search/validate_sim_state_recon.py --n-windows 40
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import zarr

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")


def _open(dataset_dir: str):
    roots = sorted(glob.glob(os.path.join(dataset_dir, "*.zarr")))
    if not roots:
        raise SystemExit(f"no *.zarr under {dataset_dir}")
    return zarr.open(roots[0], mode="r")


def describe(base) -> dict:
    """Print the state layout and return {name: (start, ndim)} offsets."""
    off, start = {}, 0
    print("--- actors (13 each: pos3 quat4 linvel3 angvel3) ---")
    for a in base._actors:
        off[f"actor:{a.get_name()}"] = (start, 13)
        print(f"  [{start:4d}:{start+13:4d}]  {a.get_name()}")
        start += 13
    print("--- articulations (13 + 2*dof) ---")
    for art in base._articulations:
        dof = art.dof
        off[f"art:{art.get_name()}"] = (start, 13 + 2 * dof)
        print(f"  [{start:4d}:{start+13+2*dof:4d}]  {art.get_name()}  dof={dof}"
              f"   qpos@{start+13}  qvel@{start+13+dof}")
        start += 13 + 2 * dof
    print(f"total state dim = {start}")
    return off


def find_key(off: dict, *needles) -> str | None:
    for k in off:
        low = k.lower()
        if all(n in low for n in needles):
            return k
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir",
                    default="/gscratch/robotics/harine/data/"
                            "strict_distance_success_sub50_diverse_start")
    ap.add_argument("--task", default="widowx_put_eggplant_in_basket")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--n-windows", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import simpler_env
    env = simpler_env.make(args.task)
    obs_env, _ = env.reset(seed=args.seed)
    base = env.unwrapped

    print(f"env = {args.task}  action_space = {env.action_space.shape}")
    template = np.asarray(base.get_state(), dtype=np.float64).copy()
    print(f"state dim from get_state() = {template.shape}\n")
    off = describe(base)

    egg_k = find_key(off, "actor", "eggplant") or find_key(off, "actor", "insert")
    bskt_k = find_key(off, "actor", "basket") or find_key(off, "actor", "recept")
    art_k = next((k for k in off if k.startswith("art:")), None)
    print(f"\nresolved: eggplant={egg_k}  basket={bskt_k}  robot={art_k}")
    if egg_k is None or art_k is None:
        raise SystemExit(
            "Could not identify the eggplant actor / robot articulation by name. "
            "Reconstruction cannot proceed — inspect the printed layout above and "
            "map the slices explicitly.")

    root = _open(args.dataset_dir)
    o = root["data/obs"]
    actions = np.asarray(root["data/actions"])
    ends = np.asarray(root["meta/episode_ends"])
    ee = np.asarray(o["end_effector_pose"])
    egg = np.asarray(o["insertive_asset_pose"])
    bskt = np.asarray(o["receptive_asset_pose"])
    qpos = np.asarray(o["arm_joint_pos"])
    qvel = np.asarray(o["joint_vel"])
    N = len(actions)

    # windows whose full horizon stays inside one episode
    ep_of = np.searchsorted(ends, np.arange(N), side="right")
    H = args.horizon
    ok = np.flatnonzero((np.arange(N) + H < N)
                        & (ep_of == ep_of[np.clip(np.arange(N) + H, 0, N - 1)]))
    rng = np.random.default_rng(args.seed)
    sel = rng.choice(ok, size=min(args.n_windows, len(ok)), replace=False)

    def reconstruct(t: int) -> np.ndarray:
        s = template.copy()
        st, _ = off[egg_k]; s[st:st + 7] = egg[t]; s[st + 7:st + 13] = 0.0
        if bskt_k is not None:
            st, _ = off[bskt_k]; s[st:st + 7] = bskt[t]; s[st + 7:st + 13] = 0.0
        st, nd = off[art_k]
        dof = (nd - 13) // 2
        na = min(qpos.shape[1], dof)
        s[st + 13:st + 13 + na] = qpos[t, :na]
        s[st + 13 + dof:st + 13 + dof + na] = qvel[t, :na]
        return s

    ee_err, egg_err, set_fail = [], [], 0
    for t in sel:
        try:
            base.set_state(reconstruct(int(t)))
        except Exception as exc:                      # noqa: BLE001
            set_fail += 1
            if set_fail == 1:
                print(f"\nset_state FAILED: {exc}")
            continue
        for k in range(H):
            env.step(np.asarray(actions[t + k], dtype=np.float32))
        st, _ = off[egg_k]
        got = np.asarray(base.get_state())
        egg_err.append(np.linalg.norm(got[st:st + 3] - egg[t + H, :3]))
        st_a, nd_a = off[art_k]
        # EE pose is not in the state vector; read it from the agent instead
        try:
            tcp = base.agent.robot.get_links()
            ee_pos = None
            for lk in tcp:
                if "gripper" in lk.get_name().lower() or "tcp" in lk.get_name().lower():
                    ee_pos = lk.get_pose().p
            if ee_pos is not None:
                ee_err.append(np.linalg.norm(ee_pos - ee[t + H, :3]))
        except Exception:                              # noqa: BLE001
            pass

    print(f"\n=== reconstruction fidelity over {len(egg_err)} windows (H={H}) ===")
    if set_fail:
        print(f"  set_state failures : {set_fail}/{len(sel)}")
    if egg_err:
        e = np.array(egg_err)
        print(f"  eggplant pos error : mean {e.mean()*1000:7.2f} mm   "
              f"median {np.median(e)*1000:7.2f} mm   p90 {np.percentile(e,90)*1000:7.2f} mm")
    if ee_err:
        e = np.array(ee_err)
        print(f"  ee pos error       : mean {e.mean()*1000:7.2f} mm   "
              f"median {np.median(e)*1000:7.2f} mm")

    # The verifier ranks candidates by a distance whose spread across candidates is a
    # few cm at most. Reconstruction error must be small RELATIVE to that spread, or
    # the ranking is dominated by state-rebuild noise rather than by the actions.
    if egg_err:
        m = float(np.median(egg_err))
        print(f"\n  VERDICT: median eggplant error {m*1000:.1f} mm")
        if m < 0.005:
            print("  GO — well below the inter-candidate spread; rollout ranking is sound.")
        elif m < 0.02:
            print("  MARGINAL — usable, but comparable to the spread between similar "
                  "candidates. Expect noisy rankings among near-ties.")
        else:
            print("  NO-GO — reconstruction error exceeds the differences the verifier "
                  "is supposed to rank. Zeroed object velocities and/or a quaternion "
                  "convention mismatch are the likely causes. Fix before training.")
    env.close()


if __name__ == "__main__":
    main()
