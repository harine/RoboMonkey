#!/usr/bin/env python3
"""Attach the DEMO's expert trajectory to each closed-loop rollout npz.

In a closed-loop rollout the per-replan expert action a* is undefined once the policy
diverges from the demo. What IS well defined -- because the rollout was reset at the demo's
own seed -- is the demo's full end-effector path. Overlaying it shows exactly where the
policy departs from the expert.

For every search_q/*.npz produced by rollout_demo_seeds.sbatch, look up the episode whose
seed matches, read that demo's EE path from the zarr, and add:
    expert_ee_world (T_demo, 3)   -- demo end-effector xyz, world frame
    expert_actions  (T_demo, 7)   -- demo actions (for reference)
    demo_ep_idx, demo_split

  python scriptsv2/analysis/augment_npz_with_expert.py --task spoon --split train --dir <out_dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SEED_DIR = REPO / "scriptsv2" / "eval_search" / "demoseeds"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["eggplant", "spoon", "stack_cube"])
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--dir", required=True, help="rollout out_dir (contains search_q/)")
    a = ap.parse_args()

    import zarr

    meta = json.loads((SEED_DIR / f"{a.task}_{a.split}.json").read_text())
    by_seed = {int(r["seed"]): r for r in meta["episodes"]}
    z = zarr.open(meta["zarr"], "r")
    zobs = z["data"]["obs"]
    zact = z["data"]["actions"]

    sq = Path(a.dir) / "search_q"
    npzs = sorted(sq.glob("*.npz"))
    if not npzs:
        raise SystemExit(f"no npz under {sq}")

    n_ok = 0
    for p in npzs:
        d = np.load(p, allow_pickle=True)
        seed = int(d["seed"])
        row = by_seed.get(seed)
        if row is None:
            print(f"  !! {p.name}: seed {seed} not in {a.task}_{a.split} demo list — skipped")
            continue
        s, e = row["out_start"], row["out_end"]
        ee = np.asarray(zobs["end_effector_pose"][s:e], np.float32)[:, :3]   # (T,3)
        act = np.asarray(zact[s:e], np.float32)                              # (T,7)

        payload = {k: d[k] for k in d.files}
        payload["expert_ee_world"] = ee
        payload["expert_actions"] = act
        payload["demo_ep_idx"] = np.int32(row["ep_idx"])
        payload["demo_split"] = str(a.split)
        np.savez_compressed(p, **payload)
        n_ok += 1
        print(f"  {p.name}: seed={seed} demo_ep={row['ep_idx']} expert_len={len(ee)}")

    print(f"[augment] {n_ok}/{len(npzs)} npz augmented in {sq}")


if __name__ == "__main__":
    main()
