#!/usr/bin/env python3
"""Why doesn't eggplant seed k map to global_idx k?

k=0 -> g=0 matched exactly, so start_seed=100000 and the obs/pose alignment are correct.
Yet k=1 matched only at g=15. The raw shards hold BOTH successes and failures
(success_frac~0.49), so `only_successful=False` -> no skipped episodes. That leaves one
prime suspect: SimplerEnv's `reset(seed=...)` for eggplant is ORDER-DEPENDENT (the episode
model / config is drawn from an internal counter that advances per reset, not purely from
the seed). Spoon/stack matched exactly for arbitrary seeds, so they are order-free.

This prints, for raw episodes k=0..2:
  (A) delta vs a SEQUENTIAL scan  reset(100000+g), g=0..GMAX   (mirrors collection order)
  (B) delta vs a FRESH env per g  (env rebuilt, single reset)  (order-free)
If (B) matches at g==k but (A) does not, reset() is order-dependent and the seed alone is
NOT sufficient to reproduce a start -- the episode's position in the reset sequence matters.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scriptsv2" / "eval_diffusion"))
RAW = Path("/gscratch/robotics/harine/data/eggplant_in_basket")
TASK = "widowx_put_eggplant_in_basket"
START_SEED = 100_000
KEYS = ["insertive_asset_pose", "receptive_asset_pose"]
GMAX = 20


def pose_from(env, seed):
    from eval_diffusion import build_obs_step  # type: ignore
    obs, _ = env.reset(seed=int(seed))
    eb = getattr(env, "unwrapped", env)
    so = build_obs_step(env=eb, obs=obs,
                        prev_arm_action=np.zeros(6, np.float32),
                        prev_gripper_action=np.zeros(1, np.float32),
                        source_obj=getattr(eb, "episode_source_obj", None),
                        target_obj=getattr(eb, "episode_target_obj", None),
                        tcp_link=getattr(eb, "tcp", None))
    return np.concatenate([np.asarray(so[k], np.float32).ravel()[:3] for k in KEYS])


def main() -> None:
    import zarr
    from simpler_env import make as make_env

    z = zarr.open(str(RAW / "state0.zarr"), "r")
    zobs = z["data"]["obs"]
    ep_ends = np.asarray(z["meta"]["episode_ends"][:])
    starts = np.concatenate([[0], ep_ends[:-1]])
    ref = np.stack([np.concatenate([np.asarray(zobs[k][int(s)], np.float32).ravel()[:3]
                                    for k in KEYS]) for s in starts[:3]])   # (3,6)

    # (A) sequential scan in one env  (mirrors the collection reset order)
    env = make_env(TASK)
    seqA = np.stack([pose_from(env, START_SEED + g) for g in range(GMAX + 1)])
    try: env.close()
    except Exception: pass

    # (B) fresh env per g (order-free: each seed is the FIRST reset of its env)
    seqB = []
    for g in range(GMAX + 1):
        e = make_env(TASK)
        seqB.append(pose_from(e, START_SEED + g))
        try: e.close()
        except Exception: pass
    seqB = np.stack(seqB)

    print(f"\n{'k':>3} | {'best g (A: sequential)':>24} | {'best g (B: fresh env)':>22}")
    print("-" * 58)
    for k in range(3):
        dA = np.max(np.abs(seqA - ref[k]), axis=1)
        dB = np.max(np.abs(seqB - ref[k]), axis=1)
        gA, gB = int(dA.argmin()), int(dB.argmin())
        print(f"{k:>3} | g={gA:<3d} delta={dA[gA]:.2e}      | g={gB:<3d} delta={dB[gB]:.2e}")
        print(f"    |   delta at g=k: {dA[k]:.2e}        |   delta at g=k: {dB[k]:.2e}")

    same = np.max(np.abs(seqA - seqB), axis=1)
    print(f"\nmax |A - B| per g (0 => reset is order-free): "
          f"{np.array2string(same, precision=2, max_line_width=200)}")
    if same.max() < 1e-6:
        print("=> reset() is ORDER-FREE; the seed alone determines the start. "
              "The k->g drift must come from skipped/reordered episodes.")
    else:
        print("=> reset() is ORDER-DEPENDENT for eggplant: the same seed gives different "
              "starts depending on how many resets preceded it. Seed alone CANNOT "
              "reproduce a demo start; the reset sequence must be replayed.")


if __name__ == "__main__":
    main()
