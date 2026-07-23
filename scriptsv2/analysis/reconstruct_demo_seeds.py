#!/usr/bin/env python3
"""Reconstruct the SIMPLER env seed of every TRAIN (sub50) and TEST
(testset_not_in_sub100) demo, for all three tasks.

The seed is not stored in any dataset, but is exactly reconstructable. Collection
(`openvla-mini/experiments/robot/simpler/collect_trajectories.py:94,371-374`) assigns
    ep_seed = start_seed(=100000) + global_idx
and every filter/subsample stage propagates the raw-shard coordinates verbatim
(`src_shard`, `src_episode`) into its `filter_manifest.json`. Hence

    seed = 100000 + START_INDEX(src_shard) + src_episode

START_INDEX is the raw shard's first global_idx: state0.zarr -> 0, and state1.zarr ->
the episode count of state0.zarr (1000 for spoon/stack, 1472 for eggplant -- verified
against the live raw shards, and consistent with collect_eggplant_a100.sbatch chaining
state2 at 2094 = 1472 + 622).

TEST episodes are subsampled with the SAME rule `sample_offline_candidates.pick_episodes`
uses (np.linspace(0, n-1, num).round()), so the offline metrics and the closed-loop
videos cover identical demos.

Writes, per task and split:
  scriptsv2/eval_search/demoseeds/<task>_<split>.txt    one seed per line (for --seeds)
  scriptsv2/eval_search/demoseeds/<task>_<split>.json   provenance: ep_idx, seed, out_start, src_*

  python scriptsv2/analysis/reconstruct_demo_seeds.py [--num 50]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "scriptsv2" / "eval_search" / "demoseeds"
DATA = Path("/gscratch/robotics/harine/data")

START_SEED = 100_000
# raw shard -> first global_idx (== number of episodes in the preceding shard)
START_INDEX = {
    "eggplant": {"state0.zarr": 0, "state1.zarr": 1472},
    "spoon": {"state0.zarr": 0, "state1.zarr": 1000},
    "stack_cube": {"state0.zarr": 0, "state1.zarr": 1000},
}
# task -> (env task name, train dataset dir, test dataset dir)
TASKS = {
    "eggplant": (
        "widowx_put_eggplant_in_basket",
        DATA / "strict_distance_success_sub50_diverse_start",
        DATA / "strict_distance_success_testset_not_in_sub100",
    ),
    "spoon": (
        "widowx_spoon_on_towel",
        DATA / "spoon_on_towel_strict_sub50_diverse_start",
        DATA / "spoon_on_towel_strict_testset_not_in_sub100",
    ),
    "stack_cube": (
        "widowx_stack_cube",
        DATA / "stack_cube_strict_sub50_diverse_start",
        DATA / "stack_cube_strict_testset_not_in_sub100",
    ),
}


def pick_episodes(n: int, num: int) -> np.ndarray:
    """Identical to sample_offline_candidates.pick_episodes (lines 182-186)."""
    if n <= num:
        return np.arange(n)
    return np.linspace(0, n - 1, num).round().astype(int)


def seeds_for(task: str, ds_dir: Path, num: int):
    mf = json.loads((ds_dir / "filter_manifest.json").read_text())
    eps = mf["episodes"]                     # ordered by out_episode
    sel = pick_episodes(len(eps), num)
    rows = []
    for ep_idx in sel:
        e = eps[int(ep_idx)]
        shard = str(e["src_shard"]).split("/")[-1]
        base = START_INDEX[task].get(shard)
        if base is None:
            raise SystemExit(f"{task}: unknown src_shard {shard!r}")
        rows.append({
            "ep_idx": int(ep_idx),                 # index into the zarr's episodes
            "seed": START_SEED + base + int(e["src_episode"]),
            "out_start": int(e["out_start"]),      # global step of this ep in the zarr
            "out_end": int(e["out_end"]),
            "src_shard": shard,
            "src_episode": int(e["src_episode"]),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=50, help="demos per task per split")
    a = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_seeds = {}
    for task, (env_task, train_dir, test_dir) in TASKS.items():
        for split, ds in (("train", train_dir), ("test", test_dir)):
            rows = seeds_for(task, ds, a.num)
            seeds = [r["seed"] for r in rows]
            assert len(seeds) == len(set(seeds)), f"{task}/{split}: duplicate seeds"
            (OUT_DIR / f"{task}_{split}.txt").write_text(
                "\n".join(str(s) for s in seeds) + "\n")
            (OUT_DIR / f"{task}_{split}.json").write_text(json.dumps({
                "task": task, "env_task": env_task, "split": split,
                "dataset_dir": str(ds), "zarr": str(ds / f"{ds.name}.zarr"),
                "start_seed": START_SEED, "start_index": START_INDEX[task],
                "n": len(rows), "episodes": rows,
            }, indent=2))
            all_seeds[(task, split)] = set(seeds)
            print(f"[seeds] {task:11s} {split:5s}: n={len(seeds):3d} "
                  f"range [{min(seeds)}, {max(seeds)}] -> {task}_{split}.txt")

    # sanity: train and test must be disjoint, and both disjoint from eval seeds 1000-1049
    evalset = set(range(1000, 1050))
    for task in TASKS:
        tr, te = all_seeds[(task, "train")], all_seeds[(task, "test")]
        assert not (tr & te), f"{task}: train/test seed overlap {sorted(tr & te)[:5]}"
        assert not ((tr | te) & evalset), f"{task}: overlaps eval seeds 1000-1049"
        print(f"[check] {task:11s}: train∩test=0  ∩eval(1000-1049)=0  OK")
    print(f"\n[seeds] wrote {2 * len(TASKS)} seed lists -> {OUT_DIR}")


if __name__ == "__main__":
    main()
