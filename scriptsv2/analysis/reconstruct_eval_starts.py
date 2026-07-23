#!/usr/bin/env python3
"""Reconstruct eval-episode START positions by resetting SimplerEnv at the eval
seeds. Start poses are seed-deterministic, so no policy rollout is needed — just
`env.reset(seed)` and read the asset/EE poses (same `init_state` schema
eval_search.py writes). Output episodes.jsonl is consumable by
plot_start_positions.py to overlay eval starts on the training cloud.

Run inside the simpler_env conda env (LD_LIBRARY_PATH/MUJOCO_GL set as in
eval_search.sh). Usage:
  python reconstruct_eval_starts.py --start-seed 1000 --num 50 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scriptsv2" / "eval_diffusion"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="widowx_put_eggplant_in_basket")
    ap.add_argument("--start-seed", type=int, default=1000)
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from eval_diffusion import build_obs_step  # type: ignore
    import simpler_env  # noqa: F401
    from simpler_env import make as make_env

    env = make_env(a.task)
    rows = []
    for i in range(a.num):
        seed = a.start_seed + i
        obs, _ = env.reset(seed=int(seed))
        env_base = getattr(env, "unwrapped", env)
        step_obs = build_obs_step(
            env=env_base, obs=obs,
            prev_arm_action=np.zeros(6, np.float32),
            prev_gripper_action=np.zeros(1, np.float32),
            source_obj=getattr(env_base, "episode_source_obj", None),
            target_obj=getattr(env_base, "episode_target_obj", None),
            tcp_link=getattr(env_base, "tcp", None),
        )
        init = {}
        for name, key in (("insertive_xyz", "insertive_asset_pose"),
                          ("receptive_xyz", "receptive_asset_pose"),
                          ("ee_xyz", "end_effector_pose")):
            v = step_obs.get(key)
            init[name] = np.asarray(v, np.float32)[:3].tolist() if v is not None else None
        rows.append({"ep_idx": i, "seed": int(seed), "success": False,
                     "init_state": init})

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "episodes.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[reconstruct] wrote {len(rows)} eval starts (seeds "
          f"{a.start_seed}..{a.start_seed + a.num - 1}) -> {out}/episodes.jsonl")


if __name__ == "__main__":
    main()
