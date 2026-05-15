"""Stitch a folder of PNGs into a single MP4 using imageio's bundled ffmpeg.

Usage:
  python scriptsv2/bon/png_to_video.py <png_dir> [--fps 2] [--out path.mp4]

By default writes <png_dir>/_video.mp4 at 2 fps (slow enough to read the
panels). Pattern: any *.png in <png_dir>, sorted lexicographically; that
naturally orders ep<...>_b000_t000.png, b001_t004.png, etc.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("png_dir", type=Path)
    p.add_argument("--fps", type=float, default=2.0,
                   help="frames per second (default 2 — slow read of branches)")
    p.add_argument("--out", type=Path, default=None,
                   help="output MP4 (default: <png_dir>/_video.mp4)")
    p.add_argument("--glob", default="*.png",
                   help="filename glob inside png_dir (default *.png)")
    args = p.parse_args(argv)

    files = sorted(args.png_dir.glob(args.glob))
    if not files:
        sys.exit(f"no PNGs match {args.png_dir}/{args.glob}")
    out = args.out or (args.png_dir / "_video.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Read first frame, pad subsequent frames to the same canvas (in case
    # matplotlib produced slightly different sizes when text wrapped).
    first = imageio.imread(files[0])
    H, W = first.shape[:2]

    with imageio.get_writer(out, fps=args.fps, codec="libx264",
                            quality=8, macro_block_size=1) as wr:
        for f in files:
            img = imageio.imread(f)
            if img.shape[:2] != (H, W):
                # pad to max canvas
                h, w = img.shape[:2]
                Hn, Wn = max(H, h), max(W, w)
                if Hn != H or Wn != W:
                    H, W = Hn, Wn
                padded = np.full((H, W, img.shape[2] if img.ndim == 3 else 1),
                                 255, dtype=img.dtype)
                padded[:img.shape[0], :img.shape[1], ...] = img
                img = padded
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            if img.shape[-1] == 4:
                img = img[..., :3]
            wr.append_data(img)
    print(f"[png_to_video] {len(files)} frames @ {args.fps} fps -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
