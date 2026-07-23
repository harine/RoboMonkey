#!/usr/bin/env python
"""Scene-parameterized STRICT success filter for SIMPLER WidowX pick-and-place
demos (stack cube, spoon on towel, carrot on plate, eggplant in basket).

Keeps ONLY the "clean, one-try, no-failure" demonstrations, i.e. episodes where:

  1. SUCCESS / NO FAILURE  — the episode succeeded (max reward > 0). Failures are
     dropped outright (the raw shards store successes AND failures).

  2. DISTANCE TO GOAL ALWAYS DECREASES — the source asset's distance to the goal
     (receptive asset) decreases monotonically (small sim jitter allowed):
       * finite every step,
       * net decrease  d_end <= end_over_d0_max * d_0,
       * bounded backtrack  max_t( d_t - min_{s<=t} d_s ) <= tol
         (tol = max(backtrack_abs, backtrack_rel * d_0)); any drop / knock-away /
         re-approach makes d rise above its running min and is cut.

  3. GRIPPER CLOSES ON THE OBJECT ONCE AND DOESN'T LET GO / DRIFT AFTER —
       * the gripper contacts the object (a real grasp happens),
       * it makes at most `max_closes_to_contact` (=1) open->closed transitions up
         to first contact  (grabs on the FIRST close, not after failed attempts),
       * no re-open run of >=2 steps between grasp and placement  (no release+retry
         => "just one try"),
       * contact is never lost for more than `max_contact_gap` (=1) steps (held),
       * after grasping, the gripper->object distance never rises more than
         `max_dge_rise` m  ("does not increase distance after closing" — the object
         is not dropped / does not slip out), AND
       * after grasping, the object->goal distance never backtracks above tol
         ("does not increase distance after closing" for the object->goal metric).

DISTANCE MODE (the one real per-scene difference)
-------------------------------------------------
For STACKING the two cubes start close together, so the vertical pick-up dominates
the 3-D center distance and would look like a huge backtrack. Stacking success is
defined by the env as horizontal (xy) alignment of the green cube over the yellow
cube, so we measure the asset->goal distance in **xy** for `stack`. For spoon /
carrot / eggplant the objects start separated horizontally and we use the full
**xyz** center distance (matches the validated eggplant filter).

Output: a single merged Zarr replay buffer (drop-in for the diffusion_policy
`Sim2RealImageMultiDataset` loader, which globs `*.zarr` in a dir) with all
`data/` + `data/obs/` keys preserved (dtype/chunks/compressor) and
`meta/episode_ends` recomputed, plus `filter_manifest.json` (per-episode
provenance + metrics) and `README.md`. NON-DESTRUCTIVE + RE-RUNNABLE.

Usage
-----
  # by scene (uses the collected shards + sensible defaults):
  python scriptsv2/data_collect/filter_strict_success.py --scene stack   [--dry-run]
  python scriptsv2/data_collect/filter_strict_success.py --scene spoon   [--dry-run]

  # or fully explicit:
  python scriptsv2/data_collect/filter_strict_success.py \
      --src-dir /gscratch/robotics/harine/data/stack_cube \
      --shards state0.zarr state1.zarr \
      --out-dir /gscratch/robotics/harine/data/stack_cube_strict \
      --dist-mode xy
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import zarr

DATA_ROOT = "/gscratch/robotics/harine/data"

# Per-scene defaults. `dist_mode` is the only mechanistic difference (see docstring).
SCENES = {
    "stack": dict(
        src_dir=f"{DATA_ROOT}/stack_cube",
        shards=["state0.zarr", "state1.zarr"],
        out_dir=f"{DATA_ROOT}/stack_cube_strict",
        dist_mode="xy",
        instruction="stack the green cube on the yellow cube",
    ),
    "spoon": dict(
        src_dir=f"{DATA_ROOT}/spoon_on_towel",
        shards=["state0.zarr", "state1.zarr"],
        out_dir=f"{DATA_ROOT}/spoon_on_towel_strict",
        dist_mode="xyz",
        instruction="put the spoon on the towel",
    ),
    # provided for convenience (new carrot shards are state2/state3):
    "carrot": dict(
        src_dir=f"{DATA_ROOT}/carrot_on_plate",
        shards=["state2.zarr", "state3.zarr"],
        out_dir=f"{DATA_ROOT}/carrot_on_plate_strict",
        dist_mode="xyz",
        instruction="put carrot on plate",
    ),
    "eggplant": dict(
        src_dir=f"{DATA_ROOT}/eggplant_in_basket",
        shards=["state0.zarr", "state1.zarr"],
        out_dir=f"{DATA_ROOT}/eggplant_in_basket_strict",
        dist_mode="xyz",
        instruction="put eggplant into yellow basket",
    ),
}

REQUIRED_OBS = ("insertive_asset_pose", "receptive_asset_pose",
                "end_effector_pose", "binary_contact", "last_gripper_action")


def episode_bounds(episode_ends):
    ends = np.asarray(episode_ends, dtype=np.int64)
    starts = np.concatenate([[0], ends[:-1]])
    return list(zip(starts.tolist(), ends.tolist()))


def runs_true(mask):
    """Lengths of maximal True runs in a 1-D boolean array."""
    out, c = [], 0
    for m in mask:
        if m:
            c += 1
        elif c:
            out.append(c); c = 0
    if c:
        out.append(c)
    return out


def episode_metrics(asset, goal, eef, contact, grip, dist_mode, closed_thresh=0.5):
    """Compute all strict-filter metrics for one episode. `asset`,`goal`,`eef` are
    (T,3) xyz; `contact` (T,) 0/1; `grip` (T,) with <closed_thresh == closed."""
    dims = slice(0, 2) if dist_mode == "xy" else slice(0, 3)
    d_ag = np.linalg.norm(asset[:, dims] - goal[:, dims], axis=1)   # asset -> goal
    d_ge = np.linalg.norm(eef[:, :3] - asset[:, :3], axis=1)        # gripper -> asset

    m = {"d0": float(d_ag[0]), "dmin": float(d_ag.min()), "dend": float(d_ag[-1])}
    finite = bool(np.all(np.isfinite(d_ag)) and np.all(np.isfinite(d_ge)))
    m["finite"] = finite
    if not finite or d_ag.size < 2:
        return None, d_ag, d_ge

    runmin = np.minimum.accumulate(d_ag)
    m["backtrack"] = float(np.max(d_ag - runmin))
    m["end_over_d0"] = float(d_ag[-1] / max(d_ag[0], 1e-6))

    closed = grip < closed_thresh
    idx = np.flatnonzero(contact == 1)
    if idx.size == 0:                       # object never grasped
        m["contacted"] = False
        return m, d_ag, d_ge
    m["contacted"] = True
    t_fc = int(idx[0])                      # first-contact (grasp) step
    t_place = int(np.argmin(d_ag))          # closest-to-goal (placement) step
    hi = max(t_place, t_fc) + 1

    becomes_closed = np.flatnonzero((~closed[:-1]) & closed[1:]) + 1
    m["closes_to_contact"] = int(np.sum(becomes_closed <= t_fc + 1))
    m["reopen_runs2"] = int(sum(1 for r in runs_true(~closed[t_fc:hi]) if r >= 2))
    gaps = runs_true(contact[t_fc:hi] == 0)
    m["max_contact_gap"] = int(max(gaps) if gaps else 0)
    m["dge_rise"] = float(d_ge[t_fc:hi].max() - d_ge[t_fc])
    # object->goal backtrack strictly AFTER the grasp:
    seg = d_ag[t_fc:]
    m["post_grasp_backtrack"] = float(np.max(seg - np.minimum.accumulate(seg))) if seg.size else 0.0
    m["t_fc"] = t_fc
    m["t_place"] = t_place
    return m, d_ag, d_ge


def keep_episode(m, args):
    """Return (kept, reason). `m` is the metrics dict from episode_metrics (or None)."""
    if m is None or not m.get("finite", False):
        return False, "nonfinite"
    tol = max(args.backtrack_abs, args.backtrack_rel * m["d0"])
    # (2) distance-to-goal always decreases
    if m["end_over_d0"] > args.end_over_d0_max:
        return False, "net_increase"
    if m["backtrack"] > tol:
        return False, "backtrack"
    # (3) gripper closes once on the object and holds without drifting
    if not m.get("contacted", False):
        return False, "no_contact"
    if m["closes_to_contact"] > args.max_closes_to_contact:
        return False, "not_first_try"
    if m["reopen_runs2"] > args.max_reopen_runs:
        return False, "release_retry"
    if m["max_contact_gap"] > args.max_contact_gap:
        return False, "dropped_contact"
    if m["dge_rise"] > args.max_dge_rise:
        return False, "gripper_drift"
    if m["post_grasp_backtrack"] > tol:
        return False, "post_grasp_increase"
    return True, None


def load_full(g):
    """Load the arrays needed for filtering from a shard (read-only)."""
    n = int(g["meta"]["episode_ends"][-1]) if g["meta"]["episode_ends"].shape[0] else 0
    ins = np.asarray(g["data"]["obs"]["insertive_asset_pose"][:, :3], dtype=np.float64)
    rec = np.asarray(g["data"]["obs"]["receptive_asset_pose"][:, :3], dtype=np.float64)
    eef = np.asarray(g["data"]["obs"]["end_effector_pose"][:, :3], dtype=np.float64)
    contact = np.asarray(g["data"]["obs"]["binary_contact"][:]).reshape(n if n else -1, -1)[:, 0].astype(int)
    grip = np.asarray(g["data"]["obs"]["last_gripper_action"][:]).reshape(n if n else -1, -1)[:, 0].astype(np.float64)
    rewards = np.asarray(g["data"]["rewards"][:]).reshape(-1)
    return ins, rec, eef, contact, grip, rewards


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", choices=sorted(SCENES), default=None,
                    help="preset src-dir/shards/out-dir/dist-mode for a known scene.")
    ap.add_argument("--src-dir", default=None)
    ap.add_argument("--shards", nargs="+", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--out-name", default=None,
                    help="zarr filename inside --out-dir (default <scene>_strict.zarr).")
    ap.add_argument("--dist-mode", choices=["xy", "xyz"], default=None,
                    help="asset->goal distance dims (stack=xy, others=xyz).")
    ap.add_argument("--goal-offset-z", type=float, default=0.0,
                    help="add this z (m) to the receptive-asset pose to define the goal "
                         "(only used for dist-mode=xyz; e.g. a stacked cube height).")
    # (1) success
    ap.add_argument("--reward-thresh", type=float, default=0.0)
    # (2) distance-to-goal decreases
    ap.add_argument("--backtrack-abs", type=float, default=0.012)
    ap.add_argument("--backtrack-rel", type=float, default=0.06)
    ap.add_argument("--end-over-d0-max", type=float, default=0.90)
    # (3) grasp: close once, hold, don't drift
    ap.add_argument("--max-closes-to-contact", type=int, default=1)
    ap.add_argument("--max-reopen-runs", type=int, default=0)
    ap.add_argument("--max-contact-gap", type=int, default=1)
    ap.add_argument("--max-dge-rise", type=float, default=0.015)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # resolve scene presets (explicit flags win)
    preset = SCENES.get(args.scene, {})
    src_dir = args.src_dir or preset.get("src_dir")
    shards = args.shards or preset.get("shards")
    out_dir = args.out_dir or preset.get("out_dir")
    dist_mode = args.dist_mode or preset.get("dist_mode", "xyz")
    instruction = preset.get("instruction", args.scene or "")
    if not (src_dir and shards and out_dir):
        ap.error("need --scene OR all of --src-dir/--shards/--out-dir")
    out_name = args.out_name or (f"{args.scene}_strict.zarr" if args.scene
                                 else "strict_success.zarr")
    args.src_dir, args.shards, args.out_dir = src_dir, shards, out_dir
    args.dist_mode, args.out_name = dist_mode, out_name

    print(f"[cfg] scene={args.scene} src_dir={src_dir}")
    print(f"[cfg] shards={shards} dist_mode={dist_mode} out={os.path.join(out_dir, out_name)}")

    # --- pass 1: select --------------------------------------------------------
    selection = []
    n_success_total = 0
    rej = {k: 0 for k in ("nonfinite", "net_increase", "backtrack", "no_contact",
                          "not_first_try", "release_retry", "dropped_contact",
                          "gripper_drift", "post_grasp_increase")}
    schema = None
    for shard in shards:
        p = os.path.join(src_dir, shard)
        if not os.path.isdir(p):
            print(f"[skip] {shard}: not found at {p}", file=sys.stderr)
            continue
        g = zarr.open(p, mode="r")
        data_keys = sorted(g["data"].array_keys())
        obs_keys = sorted(g["data"]["obs"].array_keys())
        for need in REQUIRED_OBS:
            if need not in obs_keys:
                raise SystemExit(f"[fatal] {shard} missing obs/{need}")
        if schema is None:
            schema = (data_keys, obs_keys, dict(g.attrs))
        elif (data_keys, obs_keys) != (schema[0], schema[1]):
            raise SystemExit(f"[fatal] {shard} schema differs from first shard")

        ins, rec, eef, contact, grip, rewards = load_full(g)
        goal = rec.copy()
        if dist_mode == "xyz" and args.goal_offset_z:
            goal[:, 2] += args.goal_offset_z
        ee = g["meta"]["episode_ends"][:]
        n_succ = n_keep = 0
        for ep_idx, (s, e) in enumerate(episode_bounds(ee)):
            if not (rewards[s:e].max() > args.reward_thresh):
                continue
            n_succ += 1; n_success_total += 1
            m, _, _ = episode_metrics(ins[s:e], goal[s:e], eef[s:e],
                                      contact[s:e], grip[s:e], dist_mode)
            kept, reason = keep_episode(m, args)
            if not kept:
                rej[reason] += 1
                continue
            selection.append({"shard": shard, "ep_idx": ep_idx, "start": int(s),
                              "end": int(e), "n_steps": int(e - s), "metrics": m})
            n_keep += 1
        print(f"[scan] {shard}: success {n_succ} -> kept {n_keep}")

    total_steps = sum(x["n_steps"] for x in selection)
    print(f"\n[plan] success episodes : {n_success_total}")
    print(f"[plan] kept (strict)    : {len(selection)} episodes, {total_steps} steps "
          f"({100*len(selection)/max(n_success_total,1):.1f}% of successes)")
    print(f"[plan] removed          : {n_success_total - len(selection)}  {rej}")
    if args.dry_run:
        print("[dry-run] no output written.")
        return
    if not selection:
        raise SystemExit("[fatal] nothing selected; loosen the tolerances.")

    # --- write merged output zarr ---------------------------------------------
    out = os.path.join(out_dir, out_name)
    if os.path.exists(out):
        print(f"[out] removing existing {out}"); shutil.rmtree(out)
    os.makedirs(out_dir, exist_ok=True)
    root = zarr.open(out, mode="w")
    data_keys, obs_keys, src_attrs = schema
    for k, v in src_attrs.items():
        root.attrs[k] = v
    root.attrs["filtered_from"] = list(shards)
    root.attrs["filter"] = (f"strict_success scene={args.scene} dist_mode={dist_mode}: "
                            f"success(reward>{args.reward_thresh}) AND monotone approach "
                            f"AND single held grasp")
    dgrp = root.create_group("data"); ogrp = dgrp.create_group("obs")
    first = zarr.open(os.path.join(src_dir, selection[0]["shard"]), mode="r")

    def make_out_array(parent, src_arr, name):
        shp = (total_steps,) + src_arr.shape[1:]
        chunks = (src_arr.chunks[0],) + src_arr.shape[1:]
        return parent.create_dataset(name, shape=shp, chunks=chunks,
                                     dtype=src_arr.dtype, compressor=src_arr.compressor)

    out_data = {k: make_out_array(dgrp, first["data"][k], k) for k in data_keys}
    out_obs = {k: make_out_array(ogrp, first["data"]["obs"][k], k) for k in obs_keys}

    handles, manifest, cur = {}, [], 0
    for sel in selection:
        shard, s, e, n = sel["shard"], sel["start"], sel["end"], sel["n_steps"]
        g = handles.setdefault(shard, zarr.open(os.path.join(src_dir, shard), mode="r"))
        for k in data_keys:
            out_data[k][cur:cur + n] = g["data"][k][s:e]
        for k in obs_keys:
            out_obs[k][cur:cur + n] = g["data"]["obs"][k][s:e]
        entry = {"out_episode": len(manifest), "src_shard": shard,
                 "src_episode": sel["ep_idx"], "src_start": s, "src_end": e,
                 "n_steps": n, "out_start": cur, "out_end": cur + n}
        entry.update({k: (round(v, 6) if isinstance(v, float) else v)
                      for k, v in sel["metrics"].items()})
        manifest.append(entry); cur += n

    assert cur == total_steps, (cur, total_steps)
    mgrp = root.create_group("meta")
    mgrp.create_dataset("episode_ends",
                        data=np.array([m["out_end"] for m in manifest], dtype=np.int64),
                        chunks=(len(manifest),), dtype=np.int64)

    with open(os.path.join(out_dir, "filter_manifest.json"), "w") as f:
        json.dump({"scene": args.scene, "src_dir": src_dir, "shards": shards,
                   "dist_mode": dist_mode, "out_zarr": out,
                   "n_success_total": n_success_total, "n_episodes": len(manifest),
                   "n_steps": total_steps, "n_removed": n_success_total - len(manifest),
                   "rejections": rej, "criteria": vars(args), "episodes": manifest},
                  f, indent=2)
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(_readme(args, instruction, n_success_total, len(manifest), total_steps, rej, out))

    print(f"\n[done] wrote {len(manifest)} episodes / {total_steps} steps -> {out}")


def _readme(args, instruction, n_success, n_kept, n_steps, rej, out):
    return f"""# {args.scene or os.path.basename(args.out_dir)} — strict success filter

Clean, **one-try, no-failure** `{instruction}` demonstrations for search-policy /
BC training. Drop-in Zarr replay-buffer shard for `Sim2RealImageMultiDataset`
(`discover_zarr_files` globs `*.zarr` in this dir).

Source: `{args.src_dir}` shards {", ".join(args.shards)}.
Asset->goal distance measured in **{args.dist_mode}** (stack uses xy because success
is horizontal cube-over-cube alignment; the vertical pick-up would otherwise look
like a backtrack in 3-D).

## Kept iff ALL of
1. **Success / no failure** — `max(reward) > {args.reward_thresh}`.
2. **Distance to goal always decreases** — `d_t = ||asset - goal||_{args.dist_mode}`
   (asset=`insertive_asset_pose[:3]`, goal=`receptive_asset_pose[:3]`): finite, net
   decrease `d_end <= {args.end_over_d0_max}*d_0`, bounded backtrack
   `max_t(d_t - running_min) <= max({args.backtrack_abs}, {args.backtrack_rel}*d_0)`.
3. **Gripper closes on the object once and doesn't let go / drift after** — using
   `binary_contact` (1=touching) and `last_gripper_action` (0=closed,1=open):
   `closes_to_contact <= {args.max_closes_to_contact}` (grabs on first close),
   `reopen_runs(>=2) <= {args.max_reopen_runs}` (no release+retry => one try),
   `max_contact_gap <= {args.max_contact_gap}` (held), `dge_rise <= {args.max_dge_rise}` m
   (gripper->object distance never grows after grasping), and the object->goal
   distance never backtracks after the grasp.

## Result
- successes considered : {n_success}
- **kept** : {n_kept} episodes, {n_steps} steps ({100*n_kept/max(n_success,1):.1f}% of successes)
- removed : {n_success - n_kept}  {rej}

## Output
- `{os.path.basename(out)}` — merged Zarr replay buffer (all `data/`+`data/obs/`
  keys preserved; `meta/episode_ends` recomputed). Point a training
  `dataset_dir`/`dataset_path` here.
- `filter_manifest.json` — per-episode provenance + all metrics.

## Reproduce
```
python scriptsv2/data_collect/filter_strict_success.py --scene {args.scene}
```
"""


if __name__ == "__main__":
    main()
