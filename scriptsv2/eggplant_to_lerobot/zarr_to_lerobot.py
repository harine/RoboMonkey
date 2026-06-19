"""Convert eggplant-in-basket Zarr replay buffers into the LeRobot layout that the
DQC verifier loader (`dqc/utils/bridge_lerobot_dataset.py`) consumes.

The DQC loader reads, per source directory:
  meta/episodes.jsonl    one JSON line per episode: episode_index, tasks, length
  meta/tasks.jsonl       one JSON line per task:    task_index, task
  data/chunk-XXX/episode_YYYYYY.parquet   columns: action [T,7] (raw), task_index [T]
  videos/chunk-XXX/<video_key>/episode_YYYYYY.mp4   the agentview image stream

Rewards are NOT written as a parquet column. Instead a `labels.json` is emitted
({"label_corrections": {ep: "successful"|"failure"}}); the loader then derives a
shaped 0->1 ramp reward for successes and an all-zero reward for failures, matching
how the Bridge/SOAR verifier was trained.

Actions are written RAW (un-normalized). The loader applies BridgeV2 min-max
normalization at read time; eggplant action ranges sit inside those bounds.

Example (phase-1 success-only build):
  python scriptsv2/eggplant_to_lerobot/zarr_to_lerobot.py \
      --zarr /home/harine/RoboMonkey/openvla-mini/data/eggplant_in_basket_with_images_01/state0.zarr \
      --out  /home/harine/RoboMonkey/data/eggplant_success_lerobot \
      --filter success
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr


def _episode_bounds(episode_ends: np.ndarray) -> list[tuple[int, int]]:
    starts = np.concatenate([[0], episode_ends[:-1]])
    return [(int(s), int(e)) for s, e in zip(starts, episode_ends)]


def _load_zarr_episodes(zarr_path: Path, image_obs_key: str):
    """Yield (actions[T,7], images[T,H,W,3] uint8, success bool) per episode."""
    root = zarr.open(str(zarr_path), mode="r")
    actions = root["data/actions"]
    rewards = root["data/rewards"][:]
    images = root[f"data/obs/{image_obs_key}"]
    episode_ends = root["meta/episode_ends"][:]
    task_description = dict(root.attrs).get("task_description")
    for start, end in _episode_bounds(episode_ends):
        if end - start <= 1:
            continue  # loader requires length > 1
        ep_actions = np.asarray(actions[start:end], dtype=np.float32)
        ep_images = np.asarray(images[start:end], dtype=np.uint8)
        success = bool(np.asarray(rewards[start:end]).max() > 0)
        yield ep_actions, ep_images, success, task_description


def _write_parquet(path: Path, actions: np.ndarray, task_index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = actions.shape[0]
    action_col = pa.array([row.tolist() for row in actions], type=pa.list_(pa.float32()))
    task_col = pa.array(np.full(n, task_index, dtype=np.int64))
    table = pa.table({"action": action_col, "task_index": task_col})
    pq.write_table(table, path)


def _write_video(path: Path, frames: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # libx264, even dims (224 % 16 == 0 -> no padding); keeps all T frames so the
    # loader's `frames.shape[0] >= length` check passes.
    with imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=16) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zarr", type=Path, required=True, help="Primary state*.zarr with images.")
    parser.add_argument("--also", type=Path, nargs="*", default=[], help="Extra zarrs to append (continuing indices).")
    parser.add_argument("--out", type=Path, required=True, help="Output LeRobot dataset root.")
    parser.add_argument("--filter", choices=["success", "failure", "all"], default="success")
    parser.add_argument("--task", default="put eggplant into yellow basket")
    parser.add_argument("--video-key", default="observation.images.image_0")
    parser.add_argument("--image-obs-key", default="agentview_image")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--task-index", type=int, default=0)
    args = parser.parse_args()

    out = args.out
    (out / "meta").mkdir(parents=True, exist_ok=True)

    episodes_meta: list[dict] = []
    labels: dict[str, str] = {}
    out_idx = 0
    kept_success = kept_failure = 0
    task_text = args.task

    for zarr_path in [args.zarr, *args.also]:
        for ep_actions, ep_images, success, zarr_task in _load_zarr_episodes(zarr_path, args.image_obs_key):
            if zarr_task and out_idx == 0 and args.task == parser.get_default("task"):
                task_text = zarr_task  # prefer the task text stored in the zarr attrs
            if args.filter == "success" and not success:
                continue
            if args.filter == "failure" and success:
                continue

            length = int(ep_actions.shape[0])
            if ep_images.shape[0] < length:
                raise ValueError(f"{zarr_path} ep {out_idx}: {ep_images.shape[0]} frames < {length} actions")

            chunk = f"chunk-{out_idx // 1000:03d}"
            _write_parquet(out / "data" / chunk / f"episode_{out_idx:06d}.parquet", ep_actions, args.task_index)
            _write_video(out / "videos" / chunk / args.video_key / f"episode_{out_idx:06d}.mp4", ep_images[:length], args.fps)

            episodes_meta.append({"episode_index": out_idx, "tasks": [task_text], "length": length})
            labels[str(out_idx)] = "successful" if success else "failure"
            kept_success += int(success)
            kept_failure += int(not success)
            out_idx += 1

    if out_idx == 0:
        raise RuntimeError(f"No episodes matched filter={args.filter!r}")

    with (out / "meta" / "tasks.jsonl").open("w") as f:
        f.write(json.dumps({"task_index": args.task_index, "task": task_text}) + "\n")
    with (out / "meta" / "episodes.jsonl").open("w") as f:
        for entry in episodes_meta:
            f.write(json.dumps(entry) + "\n")
    with (out / "labels.json").open("w") as f:
        json.dump({"label_corrections": labels}, f, indent=2)

    print(f"Wrote {out_idx} episodes to {out}")
    print(f"  task: {task_text!r}")
    print(f"  successful={kept_success}  failure={kept_failure}  (filter={args.filter})")
    print(f"  meta/episodes.jsonl, meta/tasks.jsonl, labels.json, data/, videos/{args.video_key}/")


if __name__ == "__main__":
    main()
