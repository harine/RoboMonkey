#!/usr/bin/env python3
"""Render success-trajectory demo videos from the OpenVLA-collected eggplant
training dataset (the exact zarr shards that train the BC + search policies).

Dataset:  /gscratch/robotics/harine/data/eggplant_in_basket_complete
          = state0.zarr (726 success) + state1.zarr (316 success) = 1042.
Success:  terminal `rewards` step > 0  (matches Sim2RealImageDataset success_only).
Frames:   data/obs/agentview_image  (T, 224, 224, 3) uint8.

Picks N success episodes spread evenly across the combined success set (for
start-position variety) and writes one MP4 per episode at --fps.
"""
import argparse
from pathlib import Path

import imageio
import numpy as np
import zarr

BASE = "/gscratch/robotics/harine/data/eggplant_in_basket"
SHARDS = ["state0.zarr", "state1.zarr"]


def collect_successes(base):
    """Return list of (shard, ep_idx, start, end) for every success episode."""
    out = []
    for sh in SHARDS:
        z = zarr.open(f"{base}/{sh}", mode="r")
        ee = np.asarray(z["meta/episode_ends"][:])
        rew = np.asarray(z["data/rewards"][:])
        starts = np.concatenate([[0], ee[:-1]]).astype(np.int64)
        ends = ee.astype(np.int64)
        succ = rew[ee - 1] > 0
        for i in np.flatnonzero(succ):
            out.append((sh, int(i), int(starts[i]), int(ends[i])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--base", default=BASE)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    succ = collect_successes(args.base)
    print(f"Total success episodes: {len(succ)}")

    # Evenly spaced selection across the combined success set -> variety.
    sel = np.linspace(0, len(succ) - 1, args.n, dtype=np.int64)
    sel = sorted(set(int(s) for s in sel))

    # Cache opened arrays per shard.
    imgs = {sh: zarr.open(f"{args.base}/{sh}", mode="r")["data/obs/agentview_image"]
            for sh in SHARDS}

    manifest = [
        "# OpenVLA demos — success trajectories from the BC/search training dataset",
        f"# source: {args.base} (state0.zarr + state1.zarr)",
        "# success = terminal rewards>0; frames = data/obs/agentview_image (224x224);",
        f"# {args.n} episodes spread evenly across the 1042-success set; fps={args.fps}.",
        "",
        "| file | shard | episode | length | seconds |",
        "|------|-------|---------|--------|---------|",
    ]
    for n, idx in enumerate(sel):
        sh, ep, s, e = succ[idx]
        frames = np.asarray(imgs[sh][s:e])  # (L, 224, 224, 3) uint8
        L = frames.shape[0]
        name = f"openvla_demo_{n:02d}__{sh.replace('.zarr','')}_ep{ep:04d}_len{L:03d}_succ.mp4"
        path = args.out / name
        w = imageio.get_writer(str(path), fps=args.fps, codec="libx264",
                               quality=8, macro_block_size=1)
        for f in frames:
            w.append_data(f)
        w.close()
        secs = L / args.fps
        print(f"[{n:02d}] {name}  ({L} frames, {secs:.1f}s)")
        manifest.append(f"| {name} | {sh} | {ep} | {L} | {secs:.1f} |")

    (args.out / "MANIFEST.md").write_text("\n".join(manifest) + "\n")
    print(f"\nWrote {len(sel)} videos + MANIFEST.md to {args.out}")


if __name__ == "__main__":
    main()
