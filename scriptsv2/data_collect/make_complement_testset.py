#!/usr/bin/env python
"""Build a held-out TEST dataset = the strict-success episodes NOT used to train the
sub50/sub100 policies, i.e. the complement of the sub100 diverse-start subsample within
the source strict set.

Mirrors subsample_diverse_start.py (same Zarr replay-buffer layout + filter_manifest.json
provenance) but INVERTS the selection: instead of the round-robin subsample it keeps every
source episode whose stable `in_episode` index is absent from the sub100 manifest. The sub50
training data is a nested subset of sub100, so this complement is disjoint from what the
sub50 policy ever saw.

  complement = sorted(range(n_source)  -  {e["in_episode"] for e in sub100.episodes})

Counts (verified): eggplant 699-100 -> 599; stack 280-100 -> 180; spoon 510-100 -> 410.

Usage:
  python scriptsv2/data_collect/make_complement_testset.py \
      --in-dir  /gscratch/robotics/harine/data/stack_cube_strict \
      --sub-dir /gscratch/robotics/harine/data/stack_cube_strict_sub100_diverse_start
  (--out-dir defaults to  <in-dir>_testset_not_in_sub100)
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np
import zarr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True, help="source strict dataset dir (one *.zarr inside)")
    ap.add_argument("--sub-dir", required=True,
                    help="the sub100 diverse-start dir whose episodes are EXCLUDED from the test set")
    ap.add_argument("--out-dir", default=None,
                    help="default: <in-dir>_testset_not_in_sub100")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = glob.glob(os.path.join(args.in_dir, "*.zarr"))
    if len(src) != 1:
        raise SystemExit(f"[fatal] expected exactly one *.zarr in {args.in_dir}, found {src}")
    src = src[0]
    z = zarr.open(src, mode="r")
    data_keys = sorted(z["data"].array_keys())
    obs_keys = sorted(z["data"]["obs"].array_keys())
    src_manifest = json.load(open(os.path.join(args.in_dir, "filter_manifest.json")))["episodes"]
    ee = np.asarray(z["meta"]["episode_ends"][:])
    starts = np.concatenate([[0], ee[:-1]]).astype(np.int64)
    ends = ee.astype(np.int64)
    n_in = len(ends)
    if len(src_manifest) != n_in:
        raise SystemExit(f"[fatal] source manifest {len(src_manifest)} != zarr {n_in} episodes")

    sub_manifest = json.load(open(os.path.join(args.sub_dir, "filter_manifest.json")))
    if sub_manifest.get("n_input") not in (None, n_in):
        raise SystemExit(f"[fatal] sub n_input {sub_manifest.get('n_input')} != source episodes {n_in} "
                         f"(mismatched source/sub pair)")
    excluded = sorted({int(e["in_episode"]) for e in sub_manifest["episodes"]})
    if excluded and (min(excluded) < 0 or max(excluded) >= n_in):
        raise SystemExit(f"[fatal] sub in_episode out of range [0,{n_in})")
    chosen = sorted(set(range(n_in)) - set(excluded))

    total_steps = sum(int(ends[i] - starts[i]) for i in chosen)
    print(f"[plan] source episodes : {n_in}  ({os.path.basename(src)})")
    print(f"[plan] sub100 excluded  : {len(excluded)}")
    print(f"[plan] test (complement): {len(chosen)} episodes, {total_steps} steps")
    assert len(chosen) + len(excluded) == n_in, (len(chosen), len(excluded), n_in)
    assert not (set(chosen) & set(excluded)), "complement overlaps excluded set!"
    if args.dry_run:
        print("[dry-run] no output written.")
        return

    out_dir = args.out_dir or (os.path.normpath(args.in_dir) + "_testset_not_in_sub100")
    out_zarr = os.path.join(out_dir, os.path.basename(out_dir) + ".zarr")
    if os.path.exists(out_zarr):
        print(f"[out] removing existing {out_zarr}")
        shutil.rmtree(out_zarr)
    os.makedirs(out_dir, exist_ok=True)

    root = zarr.open(out_zarr, mode="w")
    for k, v in dict(z.attrs).items():
        root.attrs[k] = v
    root.attrs["testset"] = (f"held-out complement of {os.path.basename(args.sub_dir)}: "
                             f"{len(chosen)} of {n_in} strict episodes NOT in sub100")
    dgrp = root.create_group("data")
    ogrp = dgrp.create_group("obs")

    def make_out(parent, src_arr, name):
        shp = (total_steps,) + src_arr.shape[1:]
        chunks = (src_arr.chunks[0],) + src_arr.shape[1:]
        return parent.create_dataset(name, shape=shp, chunks=chunks,
                                     dtype=src_arr.dtype, compressor=src_arr.compressor)

    out_data = {k: make_out(dgrp, z["data"][k], k) for k in data_keys}
    out_obs = {k: make_out(ogrp, z["data"]["obs"][k], k) for k in obs_keys}

    manifest = []
    cur = 0
    for i in chosen:
        s, e = int(starts[i]), int(ends[i])
        n = e - s
        for k in data_keys:
            out_data[k][cur:cur + n] = z["data"][k][s:e]
        for k in obs_keys:
            out_obs[k][cur:cur + n] = z["data"]["obs"][k][s:e]
        prev = src_manifest[i]
        manifest.append({"out_episode": len(manifest), "in_episode": i,
                         "src_shard": prev.get("src_shard"), "src_episode": prev.get("src_episode"),
                         "n_steps": n, "out_start": cur, "out_end": cur + n})
        cur += n
    assert cur == total_steps, (cur, total_steps)
    root.create_group("meta").create_dataset(
        "episode_ends", data=np.array([m["out_end"] for m in manifest], dtype=np.int64),
        chunks=(len(manifest),), dtype=np.int64)

    json.dump({"in_dir": args.in_dir, "sub_dir": args.sub_dir, "out_zarr": out_zarr,
               "n_input": n_in, "n_excluded_sub100": len(excluded), "n_episodes": len(manifest),
               "n_steps": total_steps, "excluded_in_episodes": excluded,
               "episodes": manifest},
              open(os.path.join(out_dir, "filter_manifest.json"), "w"), indent=2)
    open(os.path.join(out_dir, "README.md"), "w").write(
        f"""# {os.path.basename(out_dir)}

Held-out **TEST** dataset for offline action-RMSE eval: the {len(chosen)} strict-success
episodes of `{args.in_dir}` ({n_in} episodes) that are **NOT** in
`{os.path.basename(args.sub_dir)}` (the sub100 diverse-start subsample). Since sub50 is a
nested subset of sub100, this complement is disjoint from the sub50 policy's training data.

- source        : {args.in_dir}
- excluded (sub100): {os.path.basename(args.sub_dir)}  ({len(excluded)} episodes)
- test episodes : {len(chosen)}
- steps         : {total_steps}

Drop-in Zarr replay-buffer shard for `Sim2RealImageMultiDataset` (same layout as the strict set).
""")
    print(f"\n[done] wrote {len(manifest)} episodes / {total_steps} steps -> {out_zarr}")
    print(f"[done] test dir -> {out_dir}")


if __name__ == "__main__":
    main()
