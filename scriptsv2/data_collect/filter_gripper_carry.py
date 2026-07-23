#!/usr/bin/env python
"""Stage 2 of the strict eggplant filter: keep only demos where the GRIPPER
actually holds the eggplant while carrying it to the basket.

Stage 1 (filter_strict_distance_success.py) keeps successes whose eggplant->basket
distance decreases monotonically. But that alone can pass "accident" episodes where
the eggplant reached the basket without being held — knocked / bounced / rolled in,
or released early. This stage removes those by requiring the gripper to stay close
to the eggplant throughout the carry.

Distance: d_ge_t = || end_effector_pose[:3] - insertive_asset_pose[:3] ||  (gripper
xyz minus eggplant xyz). The carry window for each episode is
[t_grasp .. t_place], where t_grasp = argmin_t d_ge (tightest grasp) and
t_place = argmin_t d_eb (eggplant closest to basket, d_eb = eggplant->basket dist).
An episode is KEPT iff  max_{t in window} d_ge_t <= --gripper-carry-max  (default
0.03 m): once grasped, the gripper never lets the eggplant drift more than ~3 cm
away before it is placed.

Reads the stage-1 Zarr + its filter_manifest.json (for provenance + eggplant
metrics), writes a NEW Zarr replay buffer (drop-in shard) with only the kept
episodes, plus a merged filter_manifest.json and a README documenting BOTH filter
stages.  NON-DESTRUCTIVE: input opened read-only; output written to --out-dir.

Usage:
  python scriptsv2/data_collect/filter_gripper_carry.py \
      --in-dir  /gscratch/robotics/harine/data/strict_distance_success \
      --out-dir /gscratch/robotics/harine/data/strict_distance_success__dbl \
      --gripper-carry-max 0.03 [--dry-run]
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import zarr


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default="/gscratch/robotics/harine/data/strict_distance_success")
    ap.add_argument("--in-zarr", default="strict_distance_success.zarr")
    ap.add_argument("--out-dir", default="/gscratch/robotics/harine/data/strict_distance_success__dbl")
    ap.add_argument("--out-name", default="strict_distance_success.zarr")
    ap.add_argument("--gripper-carry-max", type=float, default=0.03,
                    help="max gripper->eggplant distance (m) allowed during the "
                         "grasp->placement window (strict=0.03).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = os.path.join(args.in_dir, args.in_zarr)
    if not os.path.isdir(src):
        raise SystemExit(f"[fatal] input zarr not found: {src}")
    z = zarr.open(src, mode="r")
    data_keys = sorted(z["data"].array_keys())
    obs_keys = sorted(z["data"]["obs"].array_keys())
    for need in ("end_effector_pose", "insertive_asset_pose", "receptive_asset_pose"):
        if need not in obs_keys:
            raise SystemExit(f"[fatal] input missing obs/{need}")
    in_manifest = json.loads(
        open(os.path.join(args.in_dir, "filter_manifest.json")).read())["episodes"]

    ee = np.asarray(z["meta"]["episode_ends"][:])
    starts = np.concatenate([[0], ee[:-1]]).astype(np.int64)
    ends = ee.astype(np.int64)
    n_in = len(ends)
    if len(in_manifest) != n_in:
        raise SystemExit(f"[fatal] manifest has {len(in_manifest)} episodes but "
                         f"zarr has {n_in}; refusing to proceed.")
    eef = np.asarray(z["data"]["obs"]["end_effector_pose"][:, :3], dtype=np.float64)
    ins = np.asarray(z["data"]["obs"]["insertive_asset_pose"][:, :3], dtype=np.float64)
    rec = np.asarray(z["data"]["obs"]["receptive_asset_pose"][:, :3], dtype=np.float64)
    d_ge = np.linalg.norm(eef - ins, axis=1)
    d_eb = np.linalg.norm(ins - rec, axis=1)

    # --- pass 1: select episodes passing the gripper-carry test --------------
    selection = []   # (in_ep, start, end, carry_max)
    n_rej = 0
    for i, (s, e) in enumerate(zip(starts, ends)):
        dge = d_ge[s:e]; deb = d_eb[s:e]
        t_grasp = int(np.argmin(dge))
        t_place = int(np.argmin(deb))
        hi = max(t_place, t_grasp) + 1
        win = dge[t_grasp:hi] if hi > t_grasp else dge[t_grasp:t_grasp + 1]
        carry_max = float(win.max())
        if carry_max <= args.gripper_carry_max:
            selection.append((i, int(s), int(e), carry_max,
                              float(dge.min()), float(t_grasp), float(t_place)))
        else:
            n_rej += 1

    total_steps = sum(e - s for _, s, e, *_ in selection)
    print(f"[plan] input episodes        : {n_in}")
    print(f"[plan] kept (gripper-carry)  : {len(selection)} episodes, {total_steps} steps "
          f"({100*len(selection)/max(n_in,1):.1f}% of input)")
    print(f"[plan] removed (gripper far) : {n_rej}")
    print(f"[plan] criterion             : max d_ge over [grasp..place] <= "
          f"{args.gripper_carry_max} m")
    if args.dry_run:
        print("[dry-run] no output written.")
        return
    if not selection:
        raise SystemExit("[fatal] nothing selected; loosen --gripper-carry-max.")

    # --- create output zarr preserving dtypes/chunks/compressors -------------
    out = os.path.join(args.out_dir, args.out_name)
    if os.path.exists(out):
        print(f"[out] removing existing {out}")
        shutil.rmtree(out)
    os.makedirs(args.out_dir, exist_ok=True)
    root = zarr.open(out, mode="w")
    for k, v in dict(z.attrs).items():
        root.attrs[k] = v
    root.attrs["filter_stage2"] = (
        f"gripper_carry: max d_ge over [grasp..place] <= {args.gripper_carry_max}")
    dgrp = root.create_group("data")
    ogrp = dgrp.create_group("obs")

    def make_out_array(parent, src_arr, name):
        shp = (total_steps,) + src_arr.shape[1:]
        chunks = (src_arr.chunks[0],) + src_arr.shape[1:]
        return parent.create_dataset(name, shape=shp, chunks=chunks,
                                     dtype=src_arr.dtype, compressor=src_arr.compressor)

    out_data = {k: make_out_array(dgrp, z["data"][k], k) for k in data_keys}
    out_obs = {k: make_out_array(ogrp, z["data"]["obs"][k], k) for k in obs_keys}

    # --- copy episode by episode --------------------------------------------
    manifest = []
    cur = 0
    for in_ep, s, e, carry_max, grasp_min, t_grasp, t_place in selection:
        n = e - s
        for k in data_keys:
            out_data[k][cur:cur + n] = z["data"][k][s:e]
        for k in obs_keys:
            out_obs[k][cur:cur + n] = z["data"]["obs"][k][s:e]
        prev = in_manifest[in_ep]
        entry = {"out_episode": len(manifest),
                 "src_shard": prev.get("src_shard"),
                 "src_episode": prev.get("src_episode"),
                 "n_steps": n, "out_start": cur, "out_end": cur + n,
                 # stage-1 eggplant->basket metrics carried forward:
                 "d0": prev.get("d0"), "dmin": prev.get("dmin"),
                 "dend": prev.get("dend"), "backtrack": prev.get("backtrack"),
                 "end_over_d0": prev.get("end_over_d0"),
                 # stage-2 gripper->eggplant metrics:
                 "gripper_carry_max": round(carry_max, 6),
                 "gripper_grasp_min": round(grasp_min, 6)}
        manifest.append(entry)
        cur += n

    assert cur == total_steps, (cur, total_steps)
    episode_ends = np.array([m["out_end"] for m in manifest], dtype=np.int64)
    mgrp = root.create_group("meta")
    mgrp.create_dataset("episode_ends", data=episode_ends,
                        chunks=(len(episode_ends),), dtype=np.int64)

    manifest_path = os.path.join(args.out_dir, "filter_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"in_dir": args.in_dir, "in_zarr": args.in_zarr, "out_zarr": out,
                   "n_input": n_in, "n_episodes": len(manifest), "n_steps": total_steps,
                   "n_removed_stage2": n_rej,
                   "gripper_carry_max": args.gripper_carry_max,
                   "episodes": manifest}, f, indent=2)

    readme_path = os.path.join(args.out_dir, "README.md")
    open(readme_path, "w").write(_readme(args, n_in, len(manifest), total_steps, n_rej, out))

    print(f"\n[done] wrote {len(manifest)} episodes / {total_steps} steps -> {out}")
    print(f"[done] manifest -> {manifest_path}")
    print(f"[done] README   -> {readme_path}")


def _readme(args, n_in, n_kept, n_steps, n_rej, out):
    return f"""# strict_distance_success (doubly filtered)

Clean success-only `widowx_put_eggplant_in_basket` demonstrations for search-policy
and BC training. Drop-in Zarr replay-buffer shard for `Sim2RealImageMultiDataset`
(`discover_zarr_files` globs `*.zarr` in this dir).

Two filter stages are applied to the raw `state0.zarr + state1.zarr` successes:

## Stage 1 — eggplant approaches the basket (no drops)
`d_eb_t = || insertive_asset_pose[:3] - receptive_asset_pose[:3] ||` (eggplant->basket).
Kept iff: success (`max reward>0`) AND finite AND net decrease (`d_end <= 0.9*d_0`)
AND bounded backtrack (`max_t(d_eb_t - running_min) <= max(0.012, 0.06*d_0)`).
See `scriptsv2/data_collect/filter_strict_distance_success.py`.

## Stage 2 — the gripper actually holds the eggplant (no knock-ins / early release)
`d_ge_t = || end_effector_pose[:3] - insertive_asset_pose[:3] ||` (gripper->eggplant).
Let `t_grasp = argmin d_ge` (tightest grasp) and `t_place = argmin d_eb` (placement).
Kept iff `max_{{t in [t_grasp .. t_place]}} d_ge_t <= {args.gripper_carry_max}` m —
once grasped, the gripper never lets the eggplant drift more than ~{args.gripper_carry_max*100:.0f} cm
away before it is placed. This drops episodes where the eggplant reached the basket
without being carried (knocked / bounced / rolled in, or released early).
See `scriptsv2/data_collect/filter_gripper_carry.py`.

## Result
- stage-1 episodes : {n_in}
- **kept after stage 2** : {n_kept} episodes, {n_steps} steps ({100*n_kept/max(n_in,1):.1f}% of stage 1)
- removed by stage 2 : {n_rej} (gripper too far from eggplant during carry)

## Output
- `{os.path.basename(out)}` — merged Zarr replay buffer (all `data/` + `data/obs/`
  keys preserved; `meta/episode_ends` recomputed). Point a training
  `dataset_dir`/`dataset_path` at this directory.
- `filter_manifest.json` — per-episode source provenance + both stages' distance
  metrics (`d0/dmin/dend/backtrack/end_over_d0`, `gripper_carry_max/gripper_grasp_min`).
"""


if __name__ == "__main__":
    main()
