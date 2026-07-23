#!/usr/bin/env python
"""Render MP4 videos of the KEPT (strict-filtered) episodes for a scene.

The strict filter (`filter_strict_success.py`) writes a merged zarr whose episodes
already contain the stored `data/obs/agentview_image` frames, so we can render
videos directly from disk — no simulator re-run. Samples N episodes (evenly spread
across the kept set by default) and writes one MP4 each into --out-dir.

Usage (run AFTER filter_strict_success.py for the scene):
  python scriptsv2/data_collect/save_kept_videos.py --scene stack --n 20
  python scriptsv2/data_collect/save_kept_videos.py --scene spoon --n 20 \
      --out-dir /mmfs1/home/harine/RoboMonkey/data/eval
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filter_strict_success import SCENES  # noqa: E402


def episode_bounds(episode_ends):
    ends = np.asarray(episode_ends, dtype=np.int64)
    starts = np.concatenate([[0], ends[:-1]])
    return list(zip(starts.tolist(), ends.tolist()))


def pick_indices(n_have, n_want, how, seed):
    if n_want >= n_have:
        return list(range(n_have))
    if how == "first":
        return list(range(n_want))
    if how == "random":
        rng = np.random.default_rng(seed)
        return sorted(rng.choice(n_have, size=n_want, replace=False).tolist())
    # even: evenly spread across the kept set
    return np.unique(np.linspace(0, n_have - 1, n_want).round().astype(int)).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", choices=sorted(SCENES), default=None)
    ap.add_argument("--in-zarr", default=None,
                    help="explicit path to the strict zarr (overrides --scene).")
    ap.add_argument("--n", type=int, default=20, help="number of videos to save.")
    ap.add_argument("--out-dir", default="/mmfs1/home/harine/RoboMonkey/data/eval",
                    help="videos go into <out-dir>/filtered_<task_dir>/ (unless --flat).")
    ap.add_argument("--flat", action="store_true",
                    help="write directly to --out-dir instead of a filtered_<task> subfolder.")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--select", choices=["even", "first", "random"], default="even")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import zarr
    import imageio.v2 as imageio

    if args.in_zarr:
        zpath = args.in_zarr
    elif args.scene:
        p = SCENES[args.scene]
        zpath = os.path.join(p["out_dir"], f"{args.scene}_strict.zarr")
    else:
        ap.error("need --scene or --in-zarr")

    if not os.path.isdir(zpath):
        raise SystemExit(f"[fatal] strict zarr not found: {zpath}\n"
                         f"        run filter_strict_success.py --scene {args.scene} first.")
    z = zarr.open(zpath, mode="r")
    obs_keys = list(z["data"]["obs"].array_keys())
    if "agentview_image" not in obs_keys:
        raise SystemExit(f"[fatal] {zpath} has no data/obs/agentview_image "
                         f"(collect with SAVE_IMAGES=True). obs keys: {obs_keys}")

    ee = z["meta"]["episode_ends"][:]
    bounds = episode_bounds(ee)
    n_have = len(bounds)
    if n_have == 0:
        raise SystemExit(f"[fatal] {zpath} has 0 episodes.")

    # optional provenance from the filter manifest (for readable filenames)
    prov = {}
    mpath = os.path.join(os.path.dirname(zpath), "filter_manifest.json")
    if os.path.isfile(mpath):
        try:
            for m in json.load(open(mpath)).get("episodes", []):
                prov[m["out_episode"]] = m
        except Exception:
            pass

    idxs = pick_indices(n_have, args.n, args.select, args.seed)
    tag = args.scene or os.path.splitext(os.path.basename(zpath))[0]
    # organize videos under a per-task subfolder: <out-dir>/filtered_<task_dir>/
    if args.scene:
        taskdir = os.path.basename(SCENES[args.scene]["src_dir"].rstrip("/"))
    else:
        taskdir = os.path.basename(os.path.dirname(zpath)).replace("_strict", "")
    out_dir = args.out_dir if args.flat else os.path.join(args.out_dir, f"filtered_{taskdir}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[cfg] zarr={zpath}")
    print(f"[cfg] kept_episodes={n_have}  saving {len(idxs)} ({args.select}) @ {args.fps}fps -> {out_dir}")

    img_arr = z["data"]["obs"]["agentview_image"]
    saved = 0
    for rank, ep in enumerate(idxs):
        s, e = bounds[ep]
        frames = np.asarray(img_arr[s:e], dtype=np.uint8)
        if frames.ndim != 4 or frames.shape[0] == 0:
            print(f"[skip] ep {ep}: bad frames shape {frames.shape}")
            continue
        p = prov.get(ep, {})
        src = f"_src{p.get('src_shard','?').split('.')[0]}-{p.get('src_episode','?')}" if p else ""
        name = f"{tag}_{rank:02d}_ep{ep:04d}{src}.mp4"
        out_path = os.path.join(out_dir, name)
        w = imageio.get_writer(out_path, fps=args.fps)
        for fr in frames:
            w.append_data(fr)
        w.close()
        saved += 1
        print(f"[save] {name}  ({frames.shape[0]} frames)")

    print(f"\n[done] wrote {saved} videos to {out_dir}")


if __name__ == "__main__":
    main()
