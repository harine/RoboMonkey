#!/usr/bin/env python
"""Filter collected val shards down to SUCCESSFUL episodes only.

The val shards (val0/val1/val2.zarr ...) hold OpenVLA rollouts where only ~50%
of episodes reach the goal. For the action-error eval we want a clean reference
set of successful trajectories. This copies the success episodes from one or
more source shards into a single merged zarr, recomputing episode_ends.

NON-DESTRUCTIVE + RE-RUNNABLE: the source shards are opened read-only and never
modified, so this can be re-run later (e.g. after val3 finishes) by just adding
shards to --shards. A manifest.json records exactly which source shard+episode
each kept episode came from, so the selection is reproducible / auditable.

Success criterion: an episode is a success iff max(reward) > 0.5 within it
(reward is 1.0 on the terminal success step, 0 otherwise; verified that the
#(reward==1) steps == the collector's reported success count per shard).

Usage:
  python scriptsv2/data_collect/filter_val_success.py \
      --shards val0.zarr val1.zarr val2.zarr \
      --src-dir /gscratch/robotics/harine/data/eggplant_in_basket_val \
      --out /gscratch/robotics/harine/data/eggplant_in_basket_val_success/val_success.zarr
"""
import argparse, json, os, shutil, sys
import numpy as np
import zarr


def episode_bounds(episode_ends):
    ends = np.asarray(episode_ends)
    starts = np.concatenate([[0], ends[:-1]])
    return list(zip(starts.tolist(), ends.tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+",
                    default=["val0.zarr", "val1.zarr", "val2.zarr"])
    ap.add_argument("--src-dir",
                    default="/gscratch/robotics/harine/data/eggplant_in_basket_val")
    ap.add_argument("--out",
                    default="/gscratch/robotics/harine/data/eggplant_in_basket_val_success/val_success.zarr")
    ap.add_argument("--reward-thresh", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # --- pass 1: enumerate success episodes across all shards ---------------
    selection = []   # (shard, ep_idx, start, end, n_steps)
    schema = None    # (data_keys, obs_keys) fingerprint from first shard
    for shard in args.shards:
        p = os.path.join(args.src_dir, shard)
        if not os.path.isdir(p):
            print(f"[skip] {shard}: not found at {p}", file=sys.stderr)
            continue
        g = zarr.open(p, mode="r")
        data_keys = sorted(g["data"].array_keys())
        obs_keys = sorted(g["data"]["obs"].array_keys())
        if schema is None:
            schema = (data_keys, obs_keys, dict(g.attrs))
        elif (data_keys, obs_keys) != (schema[0], schema[1]):
            raise SystemExit(f"[fatal] {shard} schema differs from first shard; "
                             f"refusing to merge mismatched data.")
        rewards = g["data"]["rewards"][:]
        ee = g["meta"]["episode_ends"][:]
        for ep_idx, (s, e) in enumerate(episode_bounds(ee)):
            if rewards[s:e].max() > args.reward_thresh:
                selection.append((shard, ep_idx, int(s), int(e), int(e - s)))

    total_steps = sum(n for *_, n in selection)
    print(f"[plan] {len(selection)} success episodes, {total_steps} steps total "
          f"across {len(args.shards)} shard(s)")
    for shard in args.shards:
        n = sum(1 for sh, *_ in selection if sh == shard)
        st = sum(x[4] for x in selection if x[0] == shard)
        print(f"        {shard}: {n} episodes, {st} steps")
    if args.dry_run:
        return

    # --- create output zarr with matching dtypes/chunks/compressors ---------
    out = args.out
    if os.path.exists(out):
        print(f"[out] removing existing {out}")
        shutil.rmtree(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    root = zarr.open(out, mode="w")
    data_keys, obs_keys, src_attrs = schema
    for k, v in src_attrs.items():
        root.attrs[k] = v
    root.attrs["filtered_from"] = list(args.shards)
    root.attrs["filter"] = f"success: max(reward)>{args.reward_thresh}"
    dgrp = root.create_group("data")
    ogrp = dgrp.create_group("obs")

    # template arrays come from the first available shard
    first = zarr.open(os.path.join(args.src_dir, selection[0][0]), mode="r")

    def make_out_array(parent, src_arr, name):
        shp = (total_steps,) + src_arr.shape[1:]
        chunks = (src_arr.chunks[0],) + src_arr.shape[1:]
        return parent.create_dataset(name, shape=shp, chunks=chunks,
                                     dtype=src_arr.dtype,
                                     compressor=src_arr.compressor)

    out_data = {k: make_out_array(dgrp, first["data"][k], k)
                for k in data_keys}
    out_obs = {k: make_out_array(ogrp, first["data"]["obs"][k], k)
               for k in obs_keys}

    # --- copy episode by episode -------------------------------------------
    handles = {}  # cache open shards
    manifest = []
    cur = 0
    for shard, ep_idx, s, e, n in selection:
        if shard not in handles:
            handles[shard] = zarr.open(os.path.join(args.src_dir, shard), mode="r")
        g = handles[shard]
        for k in data_keys:
            out_data[k][cur:cur + n] = g["data"][k][s:e]
        for k in obs_keys:
            out_obs[k][cur:cur + n] = g["data"]["obs"][k][s:e]
        manifest.append({"out_episode": len(manifest), "src_shard": shard,
                         "src_episode": ep_idx, "src_start": s, "src_end": e,
                         "n_steps": n, "out_start": cur, "out_end": cur + n})
        cur += n

    assert cur == total_steps, (cur, total_steps)
    episode_ends = np.array([m["out_end"] for m in manifest], dtype=np.int64)
    mgrp = root.create_group("meta")
    mgrp.create_dataset("episode_ends", data=episode_ends,
                        chunks=(len(episode_ends),), dtype=np.int64)

    manifest_path = os.path.join(os.path.dirname(out), "filter_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"shards": args.shards, "src_dir": args.src_dir,
                   "n_episodes": len(manifest), "n_steps": total_steps,
                   "reward_thresh": args.reward_thresh,
                   "episodes": manifest}, f, indent=2)

    print(f"[done] wrote {len(manifest)} episodes / {total_steps} steps -> {out}")
    print(f"[done] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
