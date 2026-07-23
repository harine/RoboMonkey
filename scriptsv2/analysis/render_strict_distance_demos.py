#!/usr/bin/env python3
"""Render demo videos from the `strict_distance_success` filtered dataset.

These are the success-only eggplant demos whose eggplant->basket distance
decreases monotonically over time (no drops / re-grasps) — see
scriptsv2/data_collect/filter_strict_distance_success.py and the dataset README.

Reads the merged Zarr replay buffer (agentview_image frames) and the companion
filter_manifest.json (per-episode source provenance + distance metrics), then
writes one MP4 per selected episode plus a MANIFEST.md. Episodes are spread
evenly across the kept set for start-position variety.

Usage:
  python scriptsv2/analysis/render_strict_distance_demos.py \
      --dataset /gscratch/robotics/harine/data/strict_distance_success \
      --out /mmfs1/home/harine/RoboMonkey/data/eval/june_19-24/videos/openVLA_demos_strict_distance_success \
      --n 25 --fps 10
"""
import argparse
import json
from pathlib import Path

import imageio
import numpy as np
import zarr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/gscratch/robotics/harine/data/strict_distance_success",
                    help="dir containing <name>.zarr + filter_manifest.json")
    ap.add_argument("--zarr-name", default="strict_distance_success.zarr")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    ds = Path(args.dataset)
    z = zarr.open(str(ds / args.zarr_name), mode="r")
    manifest = json.loads((ds / "filter_manifest.json").read_text())["episodes"]
    ee = np.asarray(z["meta/episode_ends"][:])
    starts = np.concatenate([[0], ee[:-1]]).astype(np.int64)
    imgs = z["data/obs/agentview_image"]
    n_ep = len(ee)
    print(f"Filtered dataset has {n_ep} episodes; rendering {args.n}.")

    sel = sorted(set(int(s) for s in np.linspace(0, n_ep - 1, args.n, dtype=np.int64)))
    args.out.mkdir(parents=True, exist_ok=True)
    for stale in args.out.glob("strict_demo_*.mp4"):   # clear prior render
        stale.unlink()

    lines = [
        "# strict_distance_success demos — filtered (monotone-approach) eggplant trajectories",
        f"# source dataset: {ds}",
        "# kept = success AND eggplant->basket distance decreases over time (no drops/re-grasps);",
        "# backtrack = max rise of distance above its running min (smaller = cleaner approach);",
        "# end/d0 = final distance as a fraction of the start distance (lower = closer landing).",
        f"# {len(sel)} of {n_ep} kept episodes, spread evenly; fps={args.fps}.",
        "",
        "# gripper_carry = max gripper->eggplant distance while carrying (stage-2 filter; <=0.03).",
        "| file | src_shard | src_ep | length | seconds | backtrack | end/d0 | gripper_carry |",
        "|------|-----------|--------|--------|---------|-----------|--------|---------------|",
    ]
    for n, ei in enumerate(sel):
        s, e = int(starts[ei]), int(ee[ei])
        m = manifest[ei]
        frames = np.asarray(imgs[s:e])  # (L, 224, 224, 3) uint8
        L = frames.shape[0]
        sh = m["src_shard"].replace(".zarr", "")
        name = (f"strict_demo_{n:02d}__{sh}_ep{m['src_episode']:04d}"
                f"_len{L:03d}_bt{m['backtrack']:.3f}.mp4")
        w = imageio.get_writer(str(args.out / name), fps=args.fps, codec="libx264",
                               quality=8, macro_block_size=1)
        for f in frames:
            w.append_data(f)
        w.close()
        secs = L / args.fps
        print(f"[{n:02d}] {name}  ({L} frames, {secs:.1f}s)")
        gc = m.get("gripper_carry_max")
        gc_str = f"{gc:.4f}" if gc is not None else "-"
        lines.append(f"| {name} | {m['src_shard']} | {m['src_episode']} | {L} | "
                     f"{secs:.1f} | {m['backtrack']:.4f} | {m['end_over_d0']:.3f} | {gc_str} |")

    (args.out / "MANIFEST.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {len(sel)} videos + MANIFEST.md to {args.out}")


if __name__ == "__main__":
    main()
