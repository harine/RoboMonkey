#!/usr/bin/env python
"""Sub-sample N trajectories from a filtered pick-and-place dataset with MAXIMAL
start-position variety.

The source-object resets are gridded into discrete start cells (by the quantised
initial xy of insertive_asset_pose). To keep the subset representative of every
start position, we sample round-robin across cells: shuffle
episodes within each cell (seeded), then repeatedly take one episode from each cell
in turn until N are collected. This gives ~N/n_cells per cell and guarantees every
start cell that exists in the source is represented (as long as N >= n_cells).

Writes a NEW Zarr replay buffer (drop-in shard for Sim2RealImageMultiDataset) plus
filter_manifest.json (provenance + start cell) and README. Input read-only.

Usage:
  python scriptsv2/data_collect/subsample_diverse_start.py \
      --in-dir  /gscratch/robotics/harine/data/strict_distance_success \
      --n 100 --tag sub100_diverse_start
"""
import argparse
import json
import os
import shutil
from collections import defaultdict

import numpy as np
import zarr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default="/gscratch/robotics/harine/data/strict_distance_success")
    ap.add_argument("--in-zarr", default="strict_distance_success.zarr")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tag", default="sub100_diverse_start")
    ap.add_argument("--data-root", default="/gscratch/robotics/harine/data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--round-dp", type=int, default=3, help="decimals to quantise start xy into cells.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = os.path.join(args.in_dir, args.in_zarr)
    z = zarr.open(src, mode="r")
    data_keys = sorted(z["data"].array_keys())
    obs_keys = sorted(z["data"]["obs"].array_keys())
    in_manifest = json.loads(open(os.path.join(args.in_dir, "filter_manifest.json")).read())["episodes"]
    ee = np.asarray(z["meta"]["episode_ends"][:])
    starts = np.concatenate([[0], ee[:-1]]).astype(np.int64); ends = ee.astype(np.int64)
    n_in = len(ends)
    if len(in_manifest) != n_in:
        raise SystemExit(f"[fatal] manifest {len(in_manifest)} != zarr {n_in} episodes")
    if args.n > n_in:
        raise SystemExit(f"[fatal] --n {args.n} > available {n_in}")

    # start cell per episode (eggplant xy at first step, quantised)
    egg = np.asarray(z["data"]["obs"]["insertive_asset_pose"][:, :2])
    cell_of = {}
    by_cell = defaultdict(list)
    for i, s in enumerate(starts):
        c = (round(float(egg[s, 0]), args.round_dp), round(float(egg[s, 1]), args.round_dp))
        cell_of[i] = c; by_cell[c].append(i)

    # round-robin across cells (shuffle within cell, seeded)
    rng = np.random.RandomState(args.seed)
    cell_keys = sorted(by_cell.keys())
    queues = {c: list(rng.permutation(by_cell[c])) for c in cell_keys}
    chosen = []
    while len(chosen) < args.n:
        progressed = False
        for c in cell_keys:
            if queues[c]:
                chosen.append(int(queues[c].pop()))
                progressed = True
                if len(chosen) == args.n:
                    break
        if not progressed:
            break
    chosen.sort()  # keep temporal/episode order stable in output

    per_cell = defaultdict(int)
    for i in chosen:
        per_cell[cell_of[i]] += 1
    total_steps = sum(int(ends[i] - starts[i]) for i in chosen)
    print(f"[plan] input episodes : {n_in} across {len(cell_keys)} start cells")
    print(f"[plan] selected       : {len(chosen)} episodes, {total_steps} steps")
    print(f"[plan] cells covered  : {len(per_cell)}/{len(cell_keys)}  "
          f"(per-cell min {min(per_cell.values())} max {max(per_cell.values())})")
    if args.dry_run:
        print("[dry-run] no output written."); return

    out_dir = os.path.join(args.data_root, f"{os.path.basename(args.in_dir)}_{args.tag}")
    out_zarr = os.path.join(out_dir, f"{os.path.basename(args.in_dir)}_{args.tag}.zarr")
    if os.path.exists(out_zarr):
        print(f"[out] removing existing {out_zarr}"); shutil.rmtree(out_zarr)
    os.makedirs(out_dir, exist_ok=True)
    root = zarr.open(out_zarr, mode="w")
    for k, v in dict(z.attrs).items():
        root.attrs[k] = v
    root.attrs["subsample"] = f"{args.tag}: {len(chosen)} of {n_in}, round-robin over {len(cell_keys)} start cells, seed={args.seed}"
    dgrp = root.create_group("data"); ogrp = dgrp.create_group("obs")

    def make_out(parent, src_arr, name):
        shp = (total_steps,) + src_arr.shape[1:]
        chunks = (src_arr.chunks[0],) + src_arr.shape[1:]
        return parent.create_dataset(name, shape=shp, chunks=chunks,
                                     dtype=src_arr.dtype, compressor=src_arr.compressor)

    out_data = {k: make_out(dgrp, z["data"][k], k) for k in data_keys}
    out_obs = {k: make_out(ogrp, z["data"]["obs"][k], k) for k in obs_keys}

    manifest = []; cur = 0
    for i in chosen:
        s, e = int(starts[i]), int(ends[i]); n = e - s
        for k in data_keys:
            out_data[k][cur:cur + n] = z["data"][k][s:e]
        for k in obs_keys:
            out_obs[k][cur:cur + n] = z["data"]["obs"][k][s:e]
        prev = in_manifest[i]
        manifest.append({"out_episode": len(manifest), "in_episode": i,
                         "src_shard": prev.get("src_shard"), "src_episode": prev.get("src_episode"),
                         "start_cell": list(cell_of[i]), "n_steps": n,
                         "out_start": cur, "out_end": cur + n})
        cur += n
    assert cur == total_steps, (cur, total_steps)
    root.create_group("meta").create_dataset(
        "episode_ends", data=np.array([m["out_end"] for m in manifest], dtype=np.int64),
        chunks=(len(manifest),), dtype=np.int64)

    json.dump({"in_dir": args.in_dir, "out_zarr": out_zarr, "n_input": n_in,
               "n_episodes": len(manifest), "n_steps": total_steps, "seed": args.seed,
               "n_start_cells": len(cell_keys), "per_cell_counts": {str(k): v for k, v in per_cell.items()},
               "episodes": manifest},
              open(os.path.join(out_dir, "filter_manifest.json"), "w"), indent=2)
    open(os.path.join(out_dir, "README.md"), "w").write(
        f"""# {os.path.basename(out_dir)}

{len(chosen)}-trajectory sub-sample of `{args.in_dir}` ({n_in} episodes), chosen for
**start-position variety**: round-robin over the {len(cell_keys)} discrete source-object
(insertive_asset) start cells (shuffled within each cell, seed={args.seed}), so every
start position is represented (~{args.n // len(cell_keys)}-{args.n // len(cell_keys) + 1} per cell).
Drop-in Zarr replay-buffer shard for `Sim2RealImageMultiDataset`.

- cells covered : {len(per_cell)}/{len(cell_keys)}  (per-cell min {min(per_cell.values())}, max {max(per_cell.values())})
- steps         : {total_steps}
- source        : {args.in_dir} (see its README for the strict-filter criteria)

Point a training `dataset_dir`/`dataset_path` at this directory.
""")
    print(f"\n[done] wrote {len(manifest)} episodes / {total_steps} steps -> {out_zarr}")
    print(f"[done] dataset dir -> {out_dir}")


if __name__ == "__main__":
    main()
