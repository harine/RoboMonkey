#!/usr/bin/env python
"""filtered_sample_pairs.py — sample eggplant (s,a) pairs tagged into THREE groups,
so we can see how well each verifier separates the *strictly filtered* successes
from the messy successes and the failures.

  group 0  filtered_success  kept by the strict_distance_success triple filter
                             (eggplant monotonically approaches basket, gripper
                             carries cleanly, first-try grasp never dropped)
  group 1  other_success     terminal-reward success BUT rejected by the filter
  group 2  failure           terminal reward == 0

This mirrors ``eval_dqc.py::sample_pairs`` but:
  (a) reads BOTH ``state0.zarr`` + ``state1.zarr`` of the *complete* eggplant set
      (which holds successes AND failures — the *_complete shards, not the
      success-only training symlinks);
  (b) tags each episode's group from the filter's ``filter_manifest.json``
      provenance (``src_shard`` + ``src_episode`` index over ALL episodes in the
      shard — verified: manifest n_steps == episode length at that all-episodes
      index);
  (c) samples a fixed number of episodes PER GROUP (seeded).

For each sampled step it stores the frame, the single raw 7-D action, and the
next-``BACKUP_HORIZON`` raw actions (NaN past the episode end) — the exact inputs
the DQC critics and the LLaVA verifier score, byte-identical across scorers.

Writes ``pairs.npz``. Downstream (same out-dir):
  filtered_score_dqc.py  (dqc_jax env)        -> dqc_scores_filtered.csv
  eval_robomonkey.py     (monkey-verifier env)-> verifier_scores.npz
  filtered_plot_comp.py  (any mpl env)        -> verifier_comp.{png,md,csv}

Runs in any env with zarr + numpy (dqc_jax / simpler_env). Usage in
run_filtered_verifier_comp.sbatch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BACKUP_HORIZON = 8  # chunk critic action chunk length (= checkpoint backup_horizon)
GROUP_NAMES = {0: "filtered_success", 1: "other_success", 2: "failure"}
DEFAULT_DATA = "/gscratch/robotics/harine/data/eggplant_in_basket_complete"
DEFAULT_MANIFEST = "/gscratch/robotics/harine/data/strict_distance_success/filter_manifest.json"


def load_filtered_set(manifest_path: str) -> set:
    """{(src_shard, src_episode)} of the 699 strict-filter-kept episodes."""
    m = json.load(open(manifest_path))
    s = {(str(ep["src_shard"]), int(ep["src_episode"])) for ep in m["episodes"]}
    print(f"[sample] manifest: {len(s)} filtered-success episodes "
          f"(n_input={m.get('n_input')}, n_episodes={m.get('n_episodes')})")
    return s


def shard_episodes(zarr_path: str, shard_name: str, filtered_set: set):
    """Return (z, rows) where rows = list of dicts per episode for one shard."""
    import zarr
    z = zarr.open(zarr_path, "r")
    obs = z["data"]["obs"]
    if "agentview_image" not in list(obs.keys()):
        raise ValueError(f"{zarr_path} has no agentview_image — needs images.")
    ee = np.asarray(z["meta"]["episode_ends"][:])
    rew = np.asarray(z["data"]["rewards"][:])
    starts = np.concatenate([[0], ee[:-1]])
    rows = []
    for ep, (s, e) in enumerate(zip(starts, ee)):
        success = bool(rew[int(e) - 1] > 0.5)
        if (shard_name, ep) in filtered_set:
            g = 0
        elif success:
            g = 1
        else:
            g = 2
        rows.append(dict(ep=ep, start=int(s), end=int(e), group=g, success=success))
    return z, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--shards", nargs="+", default=["state0.zarr", "state1.zarr"])
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--instruction", default="put the eggplant in the basket")
    ap.add_argument("--max-eps-per-group", type=int, default=80)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max-frames-per-ep", type=int, default=0, help="0 = all strided")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    filtered_set = load_filtered_set(args.manifest)

    # ---- build the master episode table across shards -------------------------
    z_handles, master = {}, []
    for si, shard in enumerate(args.shards):
        zpath = str(Path(args.data_dir) / shard)
        z, rows = shard_episodes(zpath, shard, filtered_set)
        z_handles[si] = z
        for r in rows:
            r["shard_idx"], r["shard"] = si, shard
            master.append(r)
    counts = {g: sum(r["group"] == g for r in master) for g in (0, 1, 2)}
    n_filt_succ = sum(r["group"] == 0 and r["success"] for r in master)
    print(f"[sample] total {len(master)} episodes across {len(args.shards)} shards — "
          + ", ".join(f"{GROUP_NAMES[g]}={counts[g]}" for g in (0, 1, 2)))
    assert n_filt_succ == counts[0], \
        f"filtered group has non-success episodes ({n_filt_succ} vs {counts[0]})"

    # ---- sample episodes per group -------------------------------------------
    rng = np.random.default_rng(args.seed)
    sel = []
    for g in (0, 1, 2):
        pool = [i for i, r in enumerate(master) if r["group"] == g]
        take = min(args.max_eps_per_group, len(pool))
        sel.extend(rng.choice(pool, take, replace=False).tolist())
        print(f"[sample]   group {g} ({GROUP_NAMES[g]}): {take}/{len(pool)} episodes")
    sel = sorted(sel)

    # ---- read frames + single actions + 8-step chunks ------------------------
    g_index, ep_uid, branch_t, group, succ, chunks = [], [], [], [], [], []
    shard_of = []
    for i in sel:
        r = master[i]
        z = z_handles[r["shard_idx"]]
        acts = z["data"]["actions"]
        A = int(acts.shape[-1])
        s, en = r["start"], r["end"]            # [s, en)
        uid = r["shard_idx"] * 1_000_000 + r["ep"]
        gs = list(range(s, en, args.stride))
        if args.max_frames_per_ep > 0:
            gs = gs[: args.max_frames_per_ep]
        for g in gs:
            n = min(BACKUP_HORIZON, en - g)
            chunk = np.full((BACKUP_HORIZON, A), np.nan, np.float32)
            chunk[:n] = np.asarray(acts[g:g + n], np.float32)
            chunks.append(chunk)
            g_index.append((r["shard_idx"], g))
            ep_uid.append(uid); branch_t.append(g - s)
            group.append(r["group"]); succ.append(r["success"])
            shard_of.append(r["shard_idx"])

    print(f"[sample] reading {len(g_index)} frames + single actions ...")
    images = np.stack([np.asarray(z_handles[si]["data"]["obs"]["agentview_image"][g])
                       for si, g in g_index]).astype(np.uint8)
    actions = np.stack([np.asarray(z_handles[si]["data"]["actions"][g])
                        for si, g in g_index]).astype(np.float32)

    pairs_file = out / "pairs.npz"
    np.savez_compressed(
        pairs_file,
        instruction=np.asarray(args.instruction),
        images=images,                                  # (G,H,W,3) uint8
        actions=actions,                                # (G,7) raw
        action_chunks=np.stack(chunks),                 # (G,8,7) raw, NaN pad
        # eval_robomonkey.py expects ep_idx / t_real / success -> alias to ours.
        ep_idx=np.asarray(ep_uid, np.int64),
        t_real=np.asarray(branch_t, np.int32),
        ep_uid=np.asarray(ep_uid, np.int64),
        branch_t=np.asarray(branch_t, np.int32),
        group=np.asarray(group, np.int32),
        success=np.asarray(succ, bool),
        shard_idx=np.asarray(shard_of, np.int32),
        group_names=np.asarray([GROUP_NAMES[g] for g in (0, 1, 2)]),
    )
    # per-group frame counts
    grp = np.asarray(group)
    fc = {g: int((grp == g).sum()) for g in (0, 1, 2)}
    print(f"[pairs] wrote {pairs_file} ({pairs_file.stat().st_size/1e6:.0f} MB)  "
          f"frames/group: " + ", ".join(f"{GROUP_NAMES[g]}={fc[g]}" for g in (0, 1, 2)))


if __name__ == "__main__":
    main()
