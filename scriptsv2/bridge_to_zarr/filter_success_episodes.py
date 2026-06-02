"""Filter a Sim2Real zarr shard down to its successful episodes.

A "successful" episode is one whose terminal `rewards` step is strictly
positive (matches `Sim2RealImageDataset._compute_success_mask` in
diffusion_policy). Episode boundaries are determined by
``meta/episode_ends``.

The output zarr mirrors the input layout: every dataset under
``data/`` (including nested groups like ``data/obs/*``) is rewritten
with only the kept episodes' steps, and ``meta/episode_ends`` is
re-cumulated. Per-array chunk sizes / compressors are preserved.

Episode-contiguous range slicing is used (no fancy indexing) so large
chunked image arrays stay sequential.

Usage:
    python scriptsv2/bridge_to_zarr/filter_success_episodes.py \
        --src ~/data/eggplant_in_basket/all_data/state0.zarr \
        --dst ~/data/eggplant_in_basket/state0.zarr
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr


def filter_zarr(src_path: Path, dst_path: Path, reward_key: str = "rewards") -> None:
    src = zarr.open(str(src_path), mode="r")
    episode_ends = src["meta"]["episode_ends"][:]
    if reward_key not in src["data"]:
        raise KeyError(
            f"{reward_key!r} not found under {src_path}/data. "
            f"Available: {list(src['data'].keys())}"
        )
    rewards = src["data"][reward_key][:]

    starts = np.concatenate([[0], episode_ends[:-1]]).astype(np.int64)
    terminal_rewards = rewards[episode_ends - 1]
    success_mask = terminal_rewards > 0

    keep_ranges = [
        (int(s), int(e))
        for s, e, ok in zip(starts, episode_ends, success_mask)
        if ok
    ]
    new_total = sum(e - s for s, e in keep_ranges)
    new_episode_ends = np.cumsum(
        [e - s for s, e in keep_ranges], dtype=np.int64
    )

    print(
        f"[{src_path.name}] kept {len(keep_ranges)}/{len(episode_ends)} "
        f"episodes, {new_total}/{len(rewards)} steps"
    )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst = zarr.open(str(dst_path), mode="w")
    data_group = dst.create_group("data")

    def copy_array(src_arr, parent_group, name):
        new_shape = (new_total,) + tuple(src_arr.shape[1:])
        chunks = src_arr.chunks
        out = parent_group.create_dataset(
            name,
            shape=new_shape,
            chunks=chunks,
            dtype=src_arr.dtype,
            compressor=src_arr.compressor,
        )
        cursor = 0
        for s, e in keep_ranges:
            n = e - s
            out[cursor : cursor + n] = src_arr[s:e]
            cursor += n

    src_data = src["data"]
    for key in src_data.keys():
        item = src_data[key]
        if isinstance(item, zarr.hierarchy.Group):
            sub = data_group.create_group(key)
            for okey in item.keys():
                copy_array(item[okey], sub, okey)
                print(f"  copied data/{key}/{okey}")
        else:
            copy_array(item, data_group, key)
            print(f"  copied data/{key}")

    meta_group = dst.create_group("meta")
    meta_group.create_dataset("episode_ends", data=new_episode_ends)
    print(f"[{src_path.name}] wrote {dst_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--reward_key", default="rewards")
    args = parser.parse_args()
    filter_zarr(args.src.expanduser(), args.dst.expanduser(), args.reward_key)


if __name__ == "__main__":
    main()
