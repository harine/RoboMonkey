#!/usr/bin/env python
"""subset_sample_pairs.py — sample (s,a) pairs from a NEW-task complete collection
(stack / spoon) tagged into THREE groups, exactly like the eggplant
``filtered_sample_pairs.py`` — so we can ask how the frozen eggplant-trained
**DQC chunk-8 verifier** discriminates on a task it never saw (zero-shot).

  group 0  filtered_success  the strict-filter *diverse-start subsample* keepers
                             (the sub100/50/25 datasets). Each episode also gets a
                             ``sub_level`` = the SMALLEST subsample it belongs to
                             (25 / 50 / 100), since sub25 subset sub50 subset sub100 —
                             so one scoring pass covers all three subsample sizes.
  group 1  other_success     terminal-reward success BUT NOT in the FULL strict set
                             (the messy, reward-only successes — the strict-kept
                             episodes that were dropped only by diverse-start
                             subsampling are EXCLUDED entirely, keeping filt-vs-other
                             a clean "filter quality" contrast).
  group 2  failure           terminal reward == 0.

Group membership is resolved from the ``filter_manifest.json`` provenance
``(src_shard, src_episode)`` — ``src_episode`` indexes over ALL episodes in the
complete shard (same convention verified for eggplant; re-verified here via the
per-episode ``n_steps``).

For each sampled step it stores the frame, the single raw 7-D action, and the
next-``BACKUP_HORIZON`` raw actions (NaN past the episode end) — byte-identical to
what ``filtered_score_dqc.py`` (DQC critics) reads. Writes ``pairs.npz``.

Downstream (same out-dir):
  filtered_score_dqc.py  (dqc_jax env)  -> dqc_scores_filtered.csv
  subset_plot_comp.py    (mpl env)      -> verifier_comp.{png,md,csv} + auc_vs_subsize.png

Runs in any env with zarr + numpy (dqc_jax / simpler_env). See
run_subset_verifier_comp.sbatch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BACKUP_HORIZON = 8  # chunk critic action chunk length (= checkpoint backup_horizon)
GROUP_NAMES = {0: "filtered_success", 1: "other_success", 2: "failure"}


def load_manifest_keys(manifest_path: str) -> set:
    """{(src_shard, src_episode)} of the episodes in a filter/subsample manifest."""
    m = json.load(open(manifest_path))
    eps = m["episodes"]
    return {(str(e["src_shard"]), int(e["src_episode"])) for e in eps}


def shard_episodes(zarr_path, shard_name, sub_levels: dict, strict_keys: set):
    """Return (z, rows). Each row tags an episode's group + (for group 0) sub_level.

    sub_levels: {(shard, ep): smallest_subsample_size} for the filtered subsample.
    strict_keys: {(shard, ep)} of the FULL strict set (its non-subsampled members
                 are excluded from every group).
    """
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
        key = (shard_name, ep)
        success = bool(rew[int(e) - 1] > 0.5)
        if key in sub_levels:
            g, lvl = 0, sub_levels[key]                 # filtered subsample keeper
        elif key in strict_keys:
            g, lvl = -1, 0                              # strict-kept but not subsampled -> DROP
        elif success:
            g, lvl = 1, 0                              # messy (reward-only) success
        else:
            g, lvl = 2, 0                              # failure
        rows.append(dict(ep=ep, start=int(s), end=int(e), group=g,
                         success=success, sub_level=lvl))
    return z, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True,
                    help="complete collection dir (has state0.zarr/state1.zarr with successes+failures)")
    ap.add_argument("--shards", nargs="+", default=["state0.zarr", "state1.zarr"])
    ap.add_argument("--sub-manifests", nargs="+", required=True,
                    help="LEVEL:PATH pairs, e.g. 100:.../sub100/filter_manifest.json 50:... 25:...")
    ap.add_argument("--strict-manifest", required=True,
                    help="FULL strict filter_manifest.json (its non-subsampled members are dropped)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--max-eps-other-fail", type=int, default=120,
                    help="cap on episodes sampled for other_success / failure groups")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- build filtered subsample -> smallest-level map -----------------------
    level_keys = {}
    for spec in args.sub_manifests:
        lvl_s, path = spec.split(":", 1)
        level_keys[int(lvl_s)] = load_manifest_keys(path)
    sub_levels = {}
    for lvl in sorted(level_keys):                     # ascending: 25, 50, 100
        for k in level_keys[lvl]:
            sub_levels.setdefault(k, lvl)              # first (smallest) level wins
    strict_keys = load_manifest_keys(args.strict_manifest)
    # sanity: every subsample key must be in strict
    assert set(sub_levels).issubset(strict_keys), "subsample not subset of strict set"
    lvl_counts = {lvl: sum(v == lvl for v in sub_levels.values()) for lvl in sorted(level_keys)}
    print(f"[sample] filtered subsample: {len(sub_levels)} eps, by smallest-level {lvl_counts} "
          f"| strict set {len(strict_keys)} eps")

    # ---- master episode table across shards -----------------------------------
    z_handles, master = {}, []
    for si, shard in enumerate(args.shards):
        zpath = str(Path(args.data_dir) / shard)
        z, rows = shard_episodes(zpath, shard, sub_levels, strict_keys)
        z_handles[si] = z
        for r in rows:
            r["shard_idx"], r["shard"] = si, shard
            master.append(r)
    counts = {g: sum(r["group"] == g for r in master) for g in (0, 1, 2)}
    dropped = sum(r["group"] == -1 for r in master)
    print(f"[sample] {len(master)} episodes total — "
          + ", ".join(f"{GROUP_NAMES[g]}={counts[g]}" for g in (0, 1, 2))
          + f", strict-not-subsampled(dropped)={dropped}")
    assert counts[0] == len(sub_levels), (counts[0], len(sub_levels))

    # ---- select episodes: ALL of group 0; cap group 1/2 -----------------------
    rng = np.random.default_rng(args.seed)
    sel = [i for i, r in enumerate(master) if r["group"] == 0]
    for g in (1, 2):
        pool = [i for i, r in enumerate(master) if r["group"] == g]
        take = min(args.max_eps_other_fail, len(pool))
        sel.extend(rng.choice(pool, take, replace=False).tolist())
        print(f"[sample]   group {g} ({GROUP_NAMES[g]}): {take}/{len(pool)} episodes")
    print(f"[sample]   group 0 (filtered_success): {counts[0]}/{counts[0]} episodes (all)")
    sel = sorted(sel)

    # ---- read frames + single actions + 8-step chunks -------------------------
    g_index, ep_uid, branch_t, group, succ, chunks, sublvl, shard_of = [], [], [], [], [], [], [], []
    for i in sel:
        r = master[i]
        z = z_handles[r["shard_idx"]]
        acts = z["data"]["actions"]
        A = int(acts.shape[-1])
        s, en = r["start"], r["end"]                    # [s, en)
        uid = r["shard_idx"] * 1_000_000 + r["ep"]
        for g in range(s, en, args.stride):
            n = min(BACKUP_HORIZON, en - g)
            chunk = np.full((BACKUP_HORIZON, A), np.nan, np.float32)
            chunk[:n] = np.asarray(acts[g:g + n], np.float32)
            chunks.append(chunk)
            g_index.append((r["shard_idx"], g))
            ep_uid.append(uid); branch_t.append(g - s)
            group.append(r["group"]); succ.append(r["success"])
            sublvl.append(r["sub_level"]); shard_of.append(r["shard_idx"])

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
        # filtered_score_dqc.py reads ep_uid / branch_t / group / success.
        ep_idx=np.asarray(ep_uid, np.int64),
        t_real=np.asarray(branch_t, np.int32),
        ep_uid=np.asarray(ep_uid, np.int64),
        branch_t=np.asarray(branch_t, np.int32),
        group=np.asarray(group, np.int32),
        success=np.asarray(succ, bool),
        sub_level=np.asarray(sublvl, np.int32),         # 25/50/100 for group 0, 0 else
        shard_idx=np.asarray(shard_of, np.int32),
        group_names=np.asarray([GROUP_NAMES[g] for g in (0, 1, 2)]),
    )
    grp = np.asarray(group)
    fc = {g: int((grp == g).sum()) for g in (0, 1, 2)}
    print(f"[pairs] wrote {pairs_file} ({pairs_file.stat().st_size/1e6:.0f} MB)  frames/group: "
          + ", ".join(f"{GROUP_NAMES[g]}={fc[g]}" for g in (0, 1, 2)))


if __name__ == "__main__":
    main()
