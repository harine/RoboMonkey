#!/usr/bin/env python3
"""Recover eggplant's true write_index -> global_idx map (and hence env seeds).

WHY: `seed = 100000 + START_INDEX(shard) + src_episode` assumes the k-th episode WRITTEN
into a raw shard is the k-th episode ATTEMPTED. That holds for spoon/stack (state0/state1
each wrote exactly 1000 = num_trajectories), but NOT for eggplant: the collector skips
episodes (`collect_trajectories.py` lines 419, 478, 480-481) and eggplant's state0 wrote
only 1472. So write index k drifts behind global_idx g, and the naive seed is wrong.
The seed gate caught this (eggplant 13/100, spoon+stack 200/200 exact).

HOW: the map is monotone (g strictly increases with k). Reset the env at
seed=100000+g for g = 0,1,2,... and compare the start pose to the raw shard's episode k
start pose. On a match, record map[k]=g and advance k. Skipped g's simply don't match.

  python scriptsv2/analysis/fix_eggplant_seeds.py --shard state0.zarr --max-k 30 --gmax 200   # smoke
  python scriptsv2/analysis/fix_eggplant_seeds.py --all                                       # full

Writes scriptsv2/eval_search/demoseeds/_eggplant_shard_map.json  {shard: {k: g}}.
Run in dqc_jax (MUJOCO_GL=osmesa).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scriptsv2" / "eval_diffusion"))
RAW = Path("/gscratch/robotics/harine/data/eggplant_in_basket")
SEED_DIR = REPO / "scriptsv2" / "eval_search" / "demoseeds"
TASK = "widowx_put_eggplant_in_basket"
START_SEED = 100_000
KEYS = ["insertive_asset_pose", "receptive_asset_pose"]


def raw_episode_starts(shard: str):
    """Return (K, ref[K, 6]) — per-episode start (insertive_xyz, receptive_xyz)."""
    import zarr
    z = zarr.open(str(RAW / shard), "r")
    zobs = z["data"]["obs"]
    ep_ends = np.asarray(z["meta"]["episode_ends"][:])
    starts = np.concatenate([[0], ep_ends[:-1]])
    ref = np.stack([
        np.concatenate([np.asarray(zobs[k][int(s)], np.float32).ravel()[:3] for k in KEYS])
        for s in starts
    ])                                        # (K, 6)
    return len(starts), ref


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="state0.zarr")
    ap.add_argument("--max-k", type=int, default=0, help="0 = all episodes in the shard")
    ap.add_argument("--gmax", type=int, default=12000)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--all", action="store_true", help="do state0 then state1, full")
    a = ap.parse_args()

    from eval_diffusion import build_obs_step  # type: ignore
    import simpler_env  # noqa: F401
    from simpler_env import make as make_env

    env = make_env(TASK)

    def start_pose(seed: int) -> np.ndarray:
        obs, _ = env.reset(seed=int(seed))
        eb = getattr(env, "unwrapped", env)
        so = build_obs_step(env=eb, obs=obs,
                            prev_arm_action=np.zeros(6, np.float32),
                            prev_gripper_action=np.zeros(1, np.float32),
                            source_obj=getattr(eb, "episode_source_obj", None),
                            target_obj=getattr(eb, "episode_target_obj", None),
                            tcp_link=getattr(eb, "tcp", None))
        return np.concatenate([np.asarray(so[k], np.float32).ravel()[:3] for k in KEYS])

    shards = ["state0.zarr", "state1.zarr"] if a.all else [a.shard]
    out_path = SEED_DIR / "_eggplant_shard_map.json"
    full = json.loads(out_path.read_text()) if out_path.exists() else {}

    for shard in shards:
        K, ref = raw_episode_starts(shard)
        need = K if (a.all or a.max_k == 0) else min(a.max_k, K)
        print(f"\n[map] {shard}: {K} written episodes; solving map for k<{need}", flush=True)

        mapping, k, g, misses = {}, 0, 0, 0
        while k < need and g < a.gmax:
            if np.max(np.abs(start_pose(START_SEED + g) - ref[k])) < a.tol:
                mapping[k] = g
                if k < 8 or k % 200 == 0:
                    print(f"  k={k:5d} -> g={g:5d}  (skips so far={g - k})", flush=True)
                k += 1
            else:
                misses += 1
            g += 1

        if k < need:
            print(f"  !! only matched {k}/{need} before gmax={a.gmax}")
        else:
            print(f"  matched {k}/{need}; total skipped global_idx = {g - k}")
        full[shard] = {str(kk): int(vv) for kk, vv in mapping.items()}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(full, indent=2))
        print(f"  -> {out_path}")

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
