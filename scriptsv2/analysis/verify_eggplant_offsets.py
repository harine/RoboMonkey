#!/usr/bin/env python3
"""Is eggplant's naive seed simply NOISY, or is it the WRONG seed?

Two effects were conflated by the seed gate:
  (a) eggplant's env reset has ~4e-4 of jitter in the start pose (stack: exactly 0.0,
      spoon: 5e-5), so the gate's tol=1e-4 was below eggplant's noise floor;
  (b) some demos are off by ~5e-3 .. 2e-1 -- far beyond jitter -- i.e. genuinely wrong seeds.

This separates them. For every eggplant train/test demo we scan a window of seeds around
the naive one and report the ARGMIN offset:
  offset == 0  -> the naive seed IS the best match; the residual is just jitter (effect a)
  offset != 0  -> the seed mapping is wrong for that demo (effect b)
Reported per src_shard, so a systematically wrong state1 START_INDEX shows up immediately.

  python scriptsv2/analysis/verify_eggplant_offsets.py [--window 24] [--split train test]
Run in dqc_jax (MUJOCO_GL=osmesa).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scriptsv2" / "eval_diffusion"))
SEED_DIR = REPO / "scriptsv2" / "eval_search" / "demoseeds"
TASK = "widowx_put_eggplant_in_basket"
KEYS = ["insertive_asset_pose", "receptive_asset_pose"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=24, help="scan naive-4 .. naive+window")
    ap.add_argument("--splits", nargs="*", default=["train", "test"])
    ap.add_argument("--max-demos", type=int, default=0, help="0 = all")
    a = ap.parse_args()

    import zarr
    from eval_diffusion import build_obs_step  # type: ignore
    from simpler_env import make as make_env

    env = make_env(TASK)

    def pose(seed: int) -> np.ndarray:
        obs, _ = env.reset(seed=int(seed))
        eb = getattr(env, "unwrapped", env)
        so = build_obs_step(env=eb, obs=obs,
                            prev_arm_action=np.zeros(6, np.float32),
                            prev_gripper_action=np.zeros(1, np.float32),
                            source_obj=getattr(eb, "episode_source_obj", None),
                            target_obj=getattr(eb, "episode_target_obj", None),
                            tcp_link=getattr(eb, "tcp", None))
        return np.concatenate([np.asarray(so[k], np.float32).ravel()[:3] for k in KEYS])

    report = {}
    for split in a.splits:
        meta = json.loads((SEED_DIR / f"eggplant_{split}.json").read_text())
        z = zarr.open(meta["zarr"], "r")
        zobs = z["data"]["obs"]
        rows = meta["episodes"][: a.max_demos] if a.max_demos else meta["episodes"]

        off_by_shard = defaultdict(Counter)
        d_naive, d_best = [], []
        for r in rows:
            ref = np.concatenate([np.asarray(zobs[k][r["out_start"]], np.float32).ravel()[:3]
                                  for k in KEYS])
            gs = list(range(r["seed"] - 4, r["seed"] + a.window + 1))
            ds = np.array([np.max(np.abs(pose(s) - ref)) for s in gs])
            j = int(ds.argmin())
            off = gs[j] - r["seed"]
            off_by_shard[r["src_shard"]][off] += 1
            d_naive.append(float(ds[4]))     # delta at the naive seed (index 4 => offset 0)
            d_best.append(float(ds[j]))

        d_naive, d_best = np.array(d_naive), np.array(d_best)
        print(f"\n=== eggplant / {split}  ({len(rows)} demos, window {-4}..+{a.window}) ===")
        for shard, c in sorted(off_by_shard.items()):
            tot = sum(c.values())
            top = c.most_common(4)
            print(f"  {shard:14s} n={tot:3d}  argmin-offset histogram: "
                  + ", ".join(f"{o:+d}:{n}" for o, n in top))
        print(f"  delta@naive : median={np.median(d_naive):.2e}  p90={np.quantile(d_naive,.9):.2e}  max={d_naive.max():.2e}")
        print(f"  delta@best  : median={np.median(d_best):.2e}  p90={np.quantile(d_best,.9):.2e}  max={d_best.max():.2e}")
        frac0 = float(np.mean([o == 0 for c in off_by_shard.values() for o, n in c.items() for _ in range(n)]))
        print(f"  fraction with argmin-offset == 0 (naive seed is best): {frac0:.2%}")
        report[split] = {
            "offsets": {s: dict(c) for s, c in off_by_shard.items()},
            "delta_naive_median": float(np.median(d_naive)),
            "delta_best_median": float(np.median(d_best)),
            "frac_offset_zero": frac0,
        }

    out = Path("data/eval/train_test_analysis/_seed_gate/eggplant_offsets.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n-> {out}")
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
