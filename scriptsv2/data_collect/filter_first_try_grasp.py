#!/usr/bin/env python
"""Stage 3 of the strict eggplant filter: keep only demos where the eggplant is
grasped on the FIRST try and never dropped afterwards.

Stages 1-2 (filter_strict_distance_success.py, filter_gripper_carry.py) keep
successes whose eggplant->basket distance decreases and whose gripper stays near
the eggplant over the grasp->placement window. But that window is anchored at the
*tightest* grasp, so an episode that fumbles the eggplant several times before
finally grabbing it (gripper closes, fails, retreats, retries) can still slip
through. This stage removes those using the contact + gripper-action signals.

Signals (per step):
  binary_contact        scalar 0/1, 1 = gripper touching the eggplant.
  last_gripper_action   scalar, 0 = CLOSED (grasp), 1 = OPEN.  (closed := <0.5)
  d_ge = || end_effector_pose[:3] - insertive_asset_pose[:3] ||  gripper->eggplant
  t_fc    = first step with contact==1                 (first grasp moment)
  t_place = argmin_t d_eb  (eggplant->basket)           (placement moment)

Kept iff ALL of:
  A. FIRST-TRY GRASP
     - closes_to_contact <= --max-closes-to-contact (default 1): the gripper makes
       at most one open->closed transition up to first contact (it grabs on the
       first close, not after several failed attempts).
     - reopen_runs(>=--reopen-min-len) over [t_fc..t_place] == --max-reopen-runs
       (default 0): after grasping, the gripper never re-opens for >=2 consecutive
       steps (no release-and-retry).
  B. HELD, NOT DROPPED
     - max_contact_gap over [t_fc..t_place] <= --max-contact-gap (default 1):
       contact is never lost for more than one step (1-step flicker tolerated).
     - dge_rise = max(d_ge[t_fc..t_place]) - d_ge[t_fc] <= --max-dge-rise
       (default 0.015 m): after grasping, the gripper->eggplant distance never
       grows by more than ~1.5 cm (the eggplant is not dropped / does not slip).

Reads the stage-2 Zarr + filter_manifest.json, writes a NEW Zarr replay buffer
(drop-in shard) with only the kept episodes, a merged filter_manifest.json, and a
README documenting all three stages. NON-DESTRUCTIVE: input read-only.

Usage:
  python scriptsv2/data_collect/filter_first_try_grasp.py \
      --in-dir  /gscratch/robotics/harine/data/strict_distance_success \
      --out-dir /gscratch/robotics/harine/data/strict_distance_success__s3 [--dry-run]
"""
import argparse
import json
import os
import shutil

import numpy as np
import zarr


def runs_true(mask):
    """Lengths of maximal True runs in a 1-D boolean array."""
    out = []; c = 0
    for m in mask:
        if m:
            c += 1
        elif c:
            out.append(c); c = 0
    if c:
        out.append(c)
    return out


def grasp_metrics(grip, contact, dge, deb, closed_thresh=0.5):
    """Return dict of first-try-grasp / drop metrics for one episode, or None if
    the eggplant is never contacted."""
    closed = grip < closed_thresh           # True = gripper closed (grasping)
    idx = np.flatnonzero(contact == 1)
    if idx.size == 0:
        return None
    t_fc = int(idx[0])
    t_place = int(np.argmin(deb))
    hi = max(t_place, t_fc) + 1
    # open->closed transitions up to (and incl. the step after) first contact
    becomes_closed = np.flatnonzero((~closed[:-1]) & closed[1:]) + 1
    closes_to_contact = int(np.sum(becomes_closed <= t_fc + 1))
    # re-open runs of length >= 2 after first contact, before placement
    seg_open = ~closed[t_fc:hi]
    reopen_runs2 = int(sum(1 for r in runs_true(seg_open) if r >= 2))
    cw = contact[t_fc:hi]
    gaps = runs_true(cw == 0)
    max_contact_gap = int(max(gaps) if gaps else 0)
    win = dge[t_fc:hi]
    dge_rise = float(win.max() - dge[t_fc])
    return {"t_fc": t_fc, "t_place": t_place,
            "closes_to_contact": closes_to_contact,
            "reopen_runs2": reopen_runs2,
            "max_contact_gap": max_contact_gap,
            "dge_rise": round(dge_rise, 6),
            "dge_at_grasp": round(float(dge[t_fc]), 6)}


def keeps(m, args):
    if m is None:
        return False
    A = (m["closes_to_contact"] <= args.max_closes_to_contact
         and m["reopen_runs2"] <= args.max_reopen_runs)
    B = (m["max_contact_gap"] <= args.max_contact_gap
         and m["dge_rise"] <= args.max_dge_rise)
    return A and B


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default="/gscratch/robotics/harine/data/strict_distance_success")
    ap.add_argument("--in-zarr", default="strict_distance_success.zarr")
    ap.add_argument("--out-dir", default="/gscratch/robotics/harine/data/strict_distance_success__s3")
    ap.add_argument("--out-name", default="strict_distance_success.zarr")
    ap.add_argument("--max-closes-to-contact", type=int, default=1)
    ap.add_argument("--max-reopen-runs", type=int, default=0)
    ap.add_argument("--reopen-min-len", type=int, default=2)   # documented; runs>=2 hard-coded in metric
    ap.add_argument("--max-contact-gap", type=int, default=1)
    ap.add_argument("--max-dge-rise", type=float, default=0.015)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = os.path.join(args.in_dir, args.in_zarr)
    z = zarr.open(src, mode="r")
    data_keys = sorted(z["data"].array_keys())
    obs_keys = sorted(z["data"]["obs"].array_keys())
    for need in ("binary_contact", "last_gripper_action", "end_effector_pose",
                 "insertive_asset_pose", "receptive_asset_pose"):
        if need not in obs_keys:
            raise SystemExit(f"[fatal] input missing obs/{need}")
    in_manifest = json.loads(
        open(os.path.join(args.in_dir, "filter_manifest.json")).read())["episodes"]

    ee = np.asarray(z["meta"]["episode_ends"][:])
    starts = np.concatenate([[0], ee[:-1]]).astype(np.int64); ends = ee.astype(np.int64)
    n_in = len(ends)
    if len(in_manifest) != n_in:
        raise SystemExit(f"[fatal] manifest {len(in_manifest)} != zarr {n_in} episodes")

    grip = np.asarray(z["data"]["obs"]["last_gripper_action"][:]).reshape(len(z["data"]["rewards"]), -1)[:, 0]
    contact = np.asarray(z["data"]["obs"]["binary_contact"][:]).astype(int).reshape(-1)
    eef = np.asarray(z["data"]["obs"]["end_effector_pose"][:, :3], dtype=np.float64)
    ins = np.asarray(z["data"]["obs"]["insertive_asset_pose"][:, :3], dtype=np.float64)
    rec = np.asarray(z["data"]["obs"]["receptive_asset_pose"][:, :3], dtype=np.float64)
    d_ge = np.linalg.norm(eef - ins, axis=1)
    d_eb = np.linalg.norm(ins - rec, axis=1)

    # --- pass 1: select ------------------------------------------------------
    selection = []
    rej = {"no_contact": 0, "first_try": 0, "dropped": 0}
    for i, (s, e) in enumerate(zip(starts, ends)):
        m = grasp_metrics(grip[s:e], contact[s:e], d_ge[s:e], d_eb[s:e])
        if m is None:
            rej["no_contact"] += 1; continue
        A = (m["closes_to_contact"] <= args.max_closes_to_contact
             and m["reopen_runs2"] <= args.max_reopen_runs)
        B = (m["max_contact_gap"] <= args.max_contact_gap
             and m["dge_rise"] <= args.max_dge_rise)
        if not A:
            rej["first_try"] += 1; continue
        if not B:
            rej["dropped"] += 1; continue
        selection.append((i, int(s), int(e), m))

    total_steps = sum(e - s for _, s, e, _ in selection)
    print(f"[plan] input episodes          : {n_in}")
    print(f"[plan] kept (first-try + held) : {len(selection)} episodes, {total_steps} steps "
          f"({100*len(selection)/max(n_in,1):.1f}% of input)")
    print(f"[plan] removed                 : not-first-try={rej['first_try']} "
          f"dropped={rej['dropped']} no_contact={rej['no_contact']}")
    print(f"[plan] criteria                : closes_to_contact<={args.max_closes_to_contact}, "
          f"reopen_runs(>=2)<={args.max_reopen_runs}, contact_gap<={args.max_contact_gap}, "
          f"dge_rise<={args.max_dge_rise}")
    if args.dry_run:
        print("[dry-run] no output written."); return
    if not selection:
        raise SystemExit("[fatal] nothing selected; loosen thresholds.")

    # --- write output zarr ---------------------------------------------------
    out = os.path.join(args.out_dir, args.out_name)
    if os.path.exists(out):
        print(f"[out] removing existing {out}"); shutil.rmtree(out)
    os.makedirs(args.out_dir, exist_ok=True)
    root = zarr.open(out, mode="w")
    for k, v in dict(z.attrs).items():
        root.attrs[k] = v
    root.attrs["filter_stage3"] = (
        f"first_try_grasp: closes_to_contact<={args.max_closes_to_contact}, "
        f"reopen_runs(>=2)<={args.max_reopen_runs}, contact_gap<={args.max_contact_gap}, "
        f"dge_rise<={args.max_dge_rise}")
    dgrp = root.create_group("data"); ogrp = dgrp.create_group("obs")

    def make_out_array(parent, src_arr, name):
        shp = (total_steps,) + src_arr.shape[1:]
        chunks = (src_arr.chunks[0],) + src_arr.shape[1:]
        return parent.create_dataset(name, shape=shp, chunks=chunks,
                                     dtype=src_arr.dtype, compressor=src_arr.compressor)

    out_data = {k: make_out_array(dgrp, z["data"][k], k) for k in data_keys}
    out_obs = {k: make_out_array(ogrp, z["data"]["obs"][k], k) for k in obs_keys}

    manifest = []
    cur = 0
    for in_ep, s, e, m in selection:
        n = e - s
        for k in data_keys:
            out_data[k][cur:cur + n] = z["data"][k][s:e]
        for k in obs_keys:
            out_obs[k][cur:cur + n] = z["data"]["obs"][k][s:e]
        prev = in_manifest[in_ep]
        entry = {"out_episode": len(manifest), "src_shard": prev.get("src_shard"),
                 "src_episode": prev.get("src_episode"), "n_steps": n,
                 "out_start": cur, "out_end": cur + n,
                 "backtrack": prev.get("backtrack"), "end_over_d0": prev.get("end_over_d0"),
                 "gripper_carry_max": prev.get("gripper_carry_max"),
                 "closes_to_contact": m["closes_to_contact"],
                 "reopen_runs2": m["reopen_runs2"],
                 "max_contact_gap": m["max_contact_gap"], "dge_rise": m["dge_rise"]}
        manifest.append(entry)
        cur += n

    assert cur == total_steps, (cur, total_steps)
    mgrp = root.create_group("meta")
    mgrp.create_dataset("episode_ends",
                        data=np.array([m["out_end"] for m in manifest], dtype=np.int64),
                        chunks=(len(manifest),), dtype=np.int64)

    json.dump({"in_dir": args.in_dir, "in_zarr": args.in_zarr, "out_zarr": out,
               "n_input": n_in, "n_episodes": len(manifest), "n_steps": total_steps,
               "rejections_stage3": rej,
               "criteria": {"max_closes_to_contact": args.max_closes_to_contact,
                            "max_reopen_runs_ge2": args.max_reopen_runs,
                            "max_contact_gap": args.max_contact_gap,
                            "max_dge_rise": args.max_dge_rise},
               "episodes": manifest},
              open(os.path.join(args.out_dir, "filter_manifest.json"), "w"), indent=2)
    open(os.path.join(args.out_dir, "README.md"), "w").write(
        _readme(args, n_in, len(manifest), total_steps, rej, out))

    print(f"\n[done] wrote {len(manifest)} episodes / {total_steps} steps -> {out}")


def _readme(args, n_in, n_kept, n_steps, rej, out):
    return f"""# strict_distance_success (triple filtered)

Clean success-only `widowx_put_eggplant_in_basket` demos for search-policy and BC
training. Drop-in Zarr replay-buffer shard for `Sim2RealImageMultiDataset`.

Three filter stages on the raw `state0.zarr + state1.zarr` successes:

## Stage 1 — eggplant approaches the basket (no drops of the object)
`d_eb = ||eggplant_xyz - basket_xyz||` decreases monotonically: success AND
`d_end<=0.9*d_0` AND `max backtrack <= max(0.012, 0.06*d_0)`.
(`filter_strict_distance_success.py`)

## Stage 2 — gripper stays near the eggplant while carrying
`d_ge = ||gripper_xyz - eggplant_xyz||`; `max d_ge over [grasp..place] <= 0.03 m`.
(`filter_gripper_carry.py`)

## Stage 3 — grasped on the FIRST try and never dropped  (this file)
Using `binary_contact` (1=touching) and `last_gripper_action` (0=closed, 1=open):
- **First-try grasp**: `closes_to_contact <= {args.max_closes_to_contact}` (gripper
  grabs on its first close, not after failed attempts) AND `reopen_runs(>=2) <= {args.max_reopen_runs}`
  over grasp->placement (never releases and retries).
- **Held, not dropped**: `max_contact_gap <= {args.max_contact_gap}` (contact lost
  for at most one step) AND `dge_rise = max(d_ge[t_fc:t_place]) - d_ge[t_fc] <= {args.max_dge_rise}` m
  (the gripper->eggplant distance never grows after grasping).
(`filter_first_try_grasp.py`)

## Result
- input (stage-2) episodes : {n_in}
- **kept after stage 3** : {n_kept} episodes, {n_steps} steps ({100*n_kept/max(n_in,1):.1f}% of input)
- removed by stage 3 : not-first-try={rej['first_try']}, dropped={rej['dropped']}, no_contact={rej['no_contact']}

## Output
- `{os.path.basename(out)}` — merged Zarr replay buffer (all keys; episode_ends
  recomputed). Point a training `dataset_dir`/`dataset_path` here.
- `filter_manifest.json` — per-episode provenance + all stages' metrics.
"""


if __name__ == "__main__":
    main()
