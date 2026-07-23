#!/usr/bin/env python
"""Build ``strict_distance_success`` — a clean, success-only eggplant-in-basket
dataset whose demonstrations approach the basket *monotonically* (no drops).

Motivation
----------
The raw eggplant shards (``state0.zarr`` + ``state1.zarr``) contain both failures
and successes. Even among the successes, some episodes are "messy": the eggplant
is grasped, dropped or knocked away (the eggplant->basket distance jumps back up),
then re-grasped before finally landing in the basket. Those re-grasp / accident
trajectories are bad supervision for BC and for the search-policy / verifier
training. This script keeps only the successes whose eggplant->basket distance
*decreases over time* (small jitter allowed), discarding the drop/accident ones.

Selection criteria (see also the README written into the output dir)
--------------------------------------------------------------------
Let ``d_t = || eggplant_xyz - basket_xyz ||`` per step, where eggplant = the first
3 dims of ``obs/insertive_asset_pose`` and basket = ``obs/receptive_asset_pose``.
An episode is kept iff ALL of:

  1. SUCCESS         : max reward in the episode > 0 (eggplant placed in basket).
  2. FINITE          : every d_t is finite (guards physics-sim blow-ups).
  3. NET DECREASE    : d_end <= ``end_over_d0_max`` * d_0  (ends meaningfully
                       closer than it started).
  4. BOUNDED BACKTRACK (the "no drop / no accident" test, strict by default):
                       max_t ( d_t - min_{s<=t} d_s )  <=  tol,
                       tol = max(``backtrack_abs``, ``backtrack_rel`` * d_0).
                       I.e. after any progress toward the basket, the eggplant
                       never moves back *away* from the basket by more than ~tol
                       (~1.2 cm with the strict defaults). A drop / re-grasp /
                       knock-away produces a large upward excursion and is cut.

Defaults are "strict": backtrack_abs=0.012 m, backtrack_rel=0.06 -> keeps ~86%
of the 1042 successes (~895 episodes), removing ~147 drop/accident trajectories.
Loosen with --backtrack-abs / --backtrack-rel for "some variation".

Output (a drop-in training shard)
---------------------------------
Writes a single merged Zarr replay buffer that the diffusion_policy loader
(``Sim2RealImageMultiDataset`` / ``discover_zarr_files`` globs ``*.zarr`` in a
dir) consumes directly — all ``data/`` and ``data/obs/`` keys are preserved with
matching dtype/chunks/compressor, episode_ends recomputed. Alongside it:
  filter_manifest.json  per-episode provenance + distance metrics (auditable)
  README.md             human-readable description of the filter criteria

NON-DESTRUCTIVE + RE-RUNNABLE: sources opened read-only; output dir overwritten.

Usage
-----
  python scriptsv2/data_collect/filter_strict_distance_success.py \
      --shards state0.zarr state1.zarr \
      --src-dir /gscratch/robotics/harine/data/eggplant_in_basket \
      --out-dir /gscratch/robotics/harine/data/strict_distance_success \
      [--dry-run]
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import zarr


def episode_bounds(episode_ends):
    ends = np.asarray(episode_ends)
    starts = np.concatenate([[0], ends[:-1]])
    return list(zip(starts.tolist(), ends.tolist()))


def episode_distance(g, s, e):
    """Per-step eggplant->basket distance over [s, e) from the asset poses."""
    ins = np.asarray(g["data"]["obs"]["insertive_asset_pose"][s:e], dtype=np.float64)[:, :3]
    rec = np.asarray(g["data"]["obs"]["receptive_asset_pose"][s:e], dtype=np.float64)[:, :3]
    return np.linalg.norm(ins - rec, axis=1)


def distance_metrics(d):
    """Metrics used for the monotone-approach filter (d is a 1-D distance array)."""
    runmin = np.minimum.accumulate(d)
    diffs = np.diff(d)
    up = float(diffs[diffs > 0].sum()) if diffs.size else 0.0
    descent = max(float(d[0] - d.min()), 1e-6)
    return {
        "d0": float(d[0]),
        "dmin": float(d.min()),
        "dend": float(d[-1]),
        "backtrack": float(np.max(d - runmin)),       # max rise above running min
        "end_over_d0": float(d[-1] / max(d[0], 1e-6)),
        "frac_up": up / descent,                       # total upward / net descent
    }


def keep_episode(d, args):
    """Return (kept: bool, reason: str|None, metrics: dict)."""
    if d.size < 2 or not np.all(np.isfinite(d)):
        return False, "nonfinite", {}
    m = distance_metrics(d)
    if m["end_over_d0"] > args.end_over_d0_max:
        return False, "net_increase", m
    tol = max(args.backtrack_abs, args.backtrack_rel * m["d0"])
    if m["backtrack"] > tol:
        return False, "backtrack", m
    return True, None, m


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", nargs="+", default=["state0.zarr", "state1.zarr"])
    ap.add_argument("--src-dir", default="/gscratch/robotics/harine/data/eggplant_in_basket")
    ap.add_argument("--out-dir", default="/gscratch/robotics/harine/data/strict_distance_success")
    ap.add_argument("--out-name", default="strict_distance_success.zarr",
                    help="zarr filename written inside --out-dir.")
    ap.add_argument("--reward-thresh", type=float, default=0.0,
                    help="episode is SUCCESS iff max(reward) > this (default >0).")
    ap.add_argument("--backtrack-abs", type=float, default=0.012,
                    help="absolute backtrack tolerance in meters (strict=0.012).")
    ap.add_argument("--backtrack-rel", type=float, default=0.06,
                    help="relative backtrack tolerance as a fraction of d0.")
    ap.add_argument("--end-over-d0-max", type=float, default=0.90,
                    help="require d_end <= this * d_0 (net-decrease guard).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # --- pass 1: enumerate episodes that pass success + strict-distance -------
    selection = []   # dicts: shard, ep_idx, start, end, n_steps, metrics
    n_success_total = 0
    rej = {"nonfinite": 0, "net_increase": 0, "backtrack": 0}
    schema = None    # (data_keys, obs_keys, attrs) fingerprint from first shard
    for shard in args.shards:
        p = os.path.join(args.src_dir, shard)
        if not os.path.isdir(p):
            print(f"[skip] {shard}: not found at {p}", file=sys.stderr)
            continue
        g = zarr.open(p, mode="r")
        data_keys = sorted(g["data"].array_keys())
        obs_keys = sorted(g["data"]["obs"].array_keys())
        for need in ("insertive_asset_pose", "receptive_asset_pose"):
            if need not in obs_keys:
                raise SystemExit(f"[fatal] {shard} missing obs/{need}; cannot "
                                 f"compute eggplant->basket distance.")
        if schema is None:
            schema = (data_keys, obs_keys, dict(g.attrs))
        elif (data_keys, obs_keys) != (schema[0], schema[1]):
            raise SystemExit(f"[fatal] {shard} schema differs from first shard; "
                             f"refusing to merge mismatched data.")
        rewards = g["data"]["rewards"][:]
        ee = g["meta"]["episode_ends"][:]
        n_succ_shard = n_keep_shard = 0
        for ep_idx, (s, e) in enumerate(episode_bounds(ee)):
            if not (rewards[s:e].max() > args.reward_thresh):
                continue   # not a success
            n_succ_shard += 1
            n_success_total += 1
            d = episode_distance(g, s, e)
            kept, reason, metrics = keep_episode(d, args)
            if not kept:
                rej[reason] += 1
                continue
            selection.append({"shard": shard, "ep_idx": ep_idx,
                              "start": int(s), "end": int(e), "n_steps": int(e - s),
                              "metrics": metrics})
            n_keep_shard += 1
        print(f"[scan] {shard}: success {n_succ_shard}  ->  kept {n_keep_shard}")

    total_steps = sum(x["n_steps"] for x in selection)
    print(f"\n[plan] success episodes : {n_success_total}")
    print(f"[plan] kept (strict)    : {len(selection)} episodes, {total_steps} steps "
          f"({100*len(selection)/max(n_success_total,1):.1f}% of successes)")
    print(f"[plan] removed          : {n_success_total - len(selection)} "
          f"(backtrack={rej['backtrack']} net_increase={rej['net_increase']} "
          f"nonfinite={rej['nonfinite']})")
    print(f"[plan] criteria         : success max(reward)>{args.reward_thresh}; "
          f"d_end<= {args.end_over_d0_max}*d0; "
          f"backtrack<= max({args.backtrack_abs}, {args.backtrack_rel}*d0)")
    if args.dry_run:
        print("[dry-run] no output written.")
        return
    if not selection:
        raise SystemExit("[fatal] nothing selected; loosen the tolerances.")

    # --- create output zarr with matching dtypes/chunks/compressors ----------
    out = os.path.join(args.out_dir, args.out_name)
    if os.path.exists(out):
        print(f"[out] removing existing {out}")
        shutil.rmtree(out)
    os.makedirs(args.out_dir, exist_ok=True)
    root = zarr.open(out, mode="w")
    data_keys, obs_keys, src_attrs = schema
    for k, v in src_attrs.items():
        root.attrs[k] = v
    root.attrs["filtered_from"] = list(args.shards)
    root.attrs["filter"] = (
        f"strict_distance_success: success(max reward>{args.reward_thresh}) AND "
        f"d_end<= {args.end_over_d0_max}*d0 AND "
        f"backtrack<= max({args.backtrack_abs},{args.backtrack_rel}*d0)")
    dgrp = root.create_group("data")
    ogrp = dgrp.create_group("obs")

    first = zarr.open(os.path.join(args.src_dir, selection[0]["shard"]), mode="r")

    def make_out_array(parent, src_arr, name):
        shp = (total_steps,) + src_arr.shape[1:]
        chunks = (src_arr.chunks[0],) + src_arr.shape[1:]
        return parent.create_dataset(name, shape=shp, chunks=chunks,
                                     dtype=src_arr.dtype, compressor=src_arr.compressor)

    out_data = {k: make_out_array(dgrp, first["data"][k], k) for k in data_keys}
    out_obs = {k: make_out_array(ogrp, first["data"]["obs"][k], k) for k in obs_keys}

    # --- copy episode by episode --------------------------------------------
    handles = {}
    manifest = []
    cur = 0
    for sel in selection:
        shard, s, e, n = sel["shard"], sel["start"], sel["end"], sel["n_steps"]
        if shard not in handles:
            handles[shard] = zarr.open(os.path.join(args.src_dir, shard), mode="r")
        g = handles[shard]
        for k in data_keys:
            out_data[k][cur:cur + n] = g["data"][k][s:e]
        for k in obs_keys:
            out_obs[k][cur:cur + n] = g["data"]["obs"][k][s:e]
        entry = {"out_episode": len(manifest), "src_shard": shard,
                 "src_episode": sel["ep_idx"], "src_start": s, "src_end": e,
                 "n_steps": n, "out_start": cur, "out_end": cur + n}
        entry.update({k: round(v, 6) for k, v in sel["metrics"].items()})
        manifest.append(entry)
        cur += n

    assert cur == total_steps, (cur, total_steps)
    episode_ends = np.array([m["out_end"] for m in manifest], dtype=np.int64)
    mgrp = root.create_group("meta")
    mgrp.create_dataset("episode_ends", data=episode_ends,
                        chunks=(len(episode_ends),), dtype=np.int64)

    # --- manifest + README ---------------------------------------------------
    criteria = {
        "success": f"max(reward) > {args.reward_thresh}",
        "finite_distance": True,
        "net_decrease": f"d_end <= {args.end_over_d0_max} * d_0",
        "bounded_backtrack": (f"max_t(d_t - running_min) <= "
                              f"max({args.backtrack_abs}, {args.backtrack_rel} * d_0)"),
        "distance_def": ("|| insertive_asset_pose[:3] - receptive_asset_pose[:3] || "
                         "(eggplant xyz minus basket xyz)"),
    }
    manifest_path = os.path.join(args.out_dir, "filter_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"shards": args.shards, "src_dir": args.src_dir,
                   "out_zarr": out, "n_success_total": n_success_total,
                   "n_episodes": len(manifest), "n_steps": total_steps,
                   "n_removed": n_success_total - len(manifest), "rejections": rej,
                   "criteria": criteria, "episodes": manifest}, f, indent=2)

    readme_path = os.path.join(args.out_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(_readme_text(args, n_success_total, len(manifest), total_steps, rej, out))

    print(f"\n[done] wrote {len(manifest)} episodes / {total_steps} steps -> {out}")
    print(f"[done] manifest -> {manifest_path}")
    print(f"[done] README   -> {readme_path}")


def _readme_text(args, n_success, n_kept, n_steps, rej, out):
    return f"""# strict_distance_success

Success-only `widowx_put_eggplant_in_basket` demonstrations whose eggplant->basket
distance **decreases monotonically over time** (small jitter allowed) — i.e. clean
approaches with **no drops, re-grasps, or knock-aways**. Built for search-policy
and BC training (drop-in Zarr replay-buffer shard for `Sim2RealImageMultiDataset`).

## Source
`{args.src_dir}` shards: {", ".join(args.shards)}
(the same shards as `eggplant_in_basket_complete`).

## Selection criteria
Per step, define the eggplant->basket distance
`d_t = || insertive_asset_pose[:3] - receptive_asset_pose[:3] ||`
(insertive = eggplant, receptive = basket). An episode is kept iff ALL of:

1. **Success** — `max(reward) > {args.reward_thresh}` (eggplant placed in basket).
2. **Finite** — every `d_t` is finite (guards physics-sim blow-ups).
3. **Net decrease** — `d_end <= {args.end_over_d0_max} * d_0` (ends meaningfully
   closer to the basket than it started).
4. **Bounded backtrack** (the "no drop / no accident" test) —
   `max_t( d_t - min_{{s<=t}} d_s ) <= max({args.backtrack_abs}, {args.backtrack_rel} * d_0)`.
   After any progress toward the basket the eggplant never moves back *away* by
   more than ~{args.backtrack_abs*100:.1f} cm; a drop / re-grasp / knock-away
   produces a large upward excursion and is rejected.

These thresholds are the **strict** setting. Loosen `--backtrack-abs` /
`--backtrack-rel` to admit more variation.

## Result
- successes considered : {n_success}
- **kept** : {n_kept} episodes, {n_steps} steps ({100*n_kept/max(n_success,1):.1f}% of successes)
- removed : {n_success - n_kept}  (backtrack={rej['backtrack']}, net_increase={rej['net_increase']}, nonfinite={rej['nonfinite']})

## Output
- `{os.path.basename(out)}` — merged Zarr replay buffer (all `data/` + `data/obs/`
  keys preserved; `meta/episode_ends` recomputed). Point a training
  `dataset_dir`/`dataset_path` at this directory / file.
- `filter_manifest.json` — per-episode provenance (source shard + episode) and the
  distance metrics (`d0`, `dmin`, `dend`, `backtrack`, `end_over_d0`, `frac_up`).

## Reproduce
```
python scriptsv2/data_collect/filter_strict_distance_success.py \\
    --shards {" ".join(args.shards)} \\
    --src-dir {args.src_dir} \\
    --out-dir {args.out_dir} \\
    --backtrack-abs {args.backtrack_abs} --backtrack-rel {args.backtrack_rel} \\
    --end-over-d0-max {args.end_over_d0_max}
```
"""


if __name__ == "__main__":
    main()
