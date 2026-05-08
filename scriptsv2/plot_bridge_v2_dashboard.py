"""Plot dataset-level diagnostics for a filtered Bridge V2 LeRobot subset.

Mirrors the structure of ``scripts/zarr_dataset_dashboard.py`` (which targets
collected SIMPLER rollouts) but operates on the LeRobot v2.0 layout produced by
``scriptsv2/filter_bridge_v2.py``.

Two figures are produced:
  * ``<out_dir>/<tag>_start_positions.png`` -- end-effector start state per episode
    plus a thumbnail grid of first camera frames (Bridge V2 does not track
    per-asset poses, so the thumbnail grid plays the role of the asset panels in
    the SIMPLER dashboard).
  * ``<out_dir>/<tag>_dashboard.png`` -- episode lengths, action statistics,
    per-step end-effector trace overlays, and gripper distribution.

The ``observation.state`` schema used here matches Bridge V2 (widowx):
``[x, y, z, roll, pitch, yaw, gripper]``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import pyarrow.parquet as pq
except ImportError as e:
    sys.exit(f"pyarrow is required: {e}")

# imageio (with the imageio-ffmpeg bundled binary) reliably decodes AV1; the
# system OpenCV's FFmpeg often does not. We try imageio first and fall back to
# OpenCV as a courtesy.
try:
    import imageio.v3 as iio
except ImportError:
    iio = None

try:
    import cv2
except ImportError:
    cv2 = None

CHUNK_SIZE_DEFAULT = 1000
CAMERA_KEYS = ("image_0", "image_1", "image_2", "image_3")
STATE_LABELS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
ACTION_LABELS = ("dx", "dy", "dz", "dr", "dp", "dyaw", "grip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/bridge_v2_filtered"),
        help="Filtered LeRobot dataset root (default: %(default)s).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Directory for output PNGs (default: <data_dir>/plots).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="bridge_v2_carrot",
        help="Filename prefix for the saved PNGs (default: %(default)s).",
    )
    parser.add_argument(
        "--task_filter",
        type=str,
        default=None,
        help=(
            "Only consider episodes whose `tasks` list contains this exact string. "
            "Defaults to all episodes in the filtered dataset."
        ),
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Optional cap on episodes to read (deterministic, sorted by index).",
    )
    parser.add_argument(
        "--n_thumbnails",
        type=int,
        default=16,
        help="Number of episode first-frame thumbnails to render (default: %(default)s).",
    )
    parser.add_argument(
        "--thumbnail_camera",
        type=str,
        default="auto",
        choices=("auto", *CAMERA_KEYS),
        help=(
            "Which camera to use for thumbnails. 'auto' picks the first available "
            "camera per episode (default: %(default)s)."
        ),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[dict]:
    with path.open("r") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def select_episodes(
    episodes: Sequence[dict],
    task_filter: Optional[str],
    max_episodes: Optional[int],
) -> List[dict]:
    rows = list(episodes)
    if task_filter is not None:
        rows = [e for e in rows if task_filter in (e.get("tasks") or [])]
    rows.sort(key=lambda e: int(e["episode_index"]))
    if max_episodes is not None:
        rows = rows[:max_episodes]
    return rows


def episode_parquet_path(data_dir: Path, episode_index: int, chunk_size: int) -> Path:
    chunk = episode_index // chunk_size
    return (
        data_dir
        / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
    )


def episode_video_path(
    data_dir: Path, episode_index: int, chunk_size: int, camera: str
) -> Path:
    chunk = episode_index // chunk_size
    return (
        data_dir
        / f"videos/chunk-{chunk:03d}/observation.images.{camera}/episode_{episode_index:06d}.mp4"
    )


def _column_to_2d(arr: np.ndarray) -> np.ndarray:
    """Cast a (n,) object/list column to a (n, k) float array."""
    if arr.dtype == object:
        return np.asarray([np.asarray(row, dtype=np.float32) for row in arr])
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def read_episode_arrays(parquet_path: Path) -> dict:
    """Return ``observation.state``, ``action``, and frame count for an episode."""
    table = pq.read_table(parquet_path, columns=["observation.state", "action"])
    state = _column_to_2d(table.column("observation.state").to_numpy(zero_copy_only=False))
    action = _column_to_2d(table.column("action").to_numpy(zero_copy_only=False))
    return {"state": state, "action": action, "n_frames": int(state.shape[0])}


def read_first_frame(video_path: Path) -> Optional[np.ndarray]:
    """Decode and return the first frame of a video as RGB uint8, or None on failure.

    Tries imageio (uses imageio-ffmpeg's bundled ffmpeg, which supports AV1) before
    falling back to OpenCV. Bridge V2 videos are AV1-encoded; OpenCV's bundled
    FFmpeg often lacks libdav1d.
    """
    if not video_path.exists():
        return None
    if iio is not None:
        try:
            frame = iio.imread(str(video_path), index=0, plugin="FFMPEG")
            if frame is not None and frame.ndim == 3 and frame.shape[2] >= 3:
                return np.asarray(frame[..., :3], dtype=np.uint8)
        except Exception:
            pass
    if cv2 is not None:
        cap = cv2.VideoCapture(str(video_path))
        try:
            ok, frame = cap.read()
            if ok and frame is not None:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()
    return None


def first_available_thumbnail(
    data_dir: Path,
    episode_index: int,
    chunk_size: int,
    preferred_camera: str,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    cameras = (
        (preferred_camera,) if preferred_camera != "auto" else CAMERA_KEYS
    )
    for cam in cameras:
        path = episode_video_path(data_dir, episode_index, chunk_size, cam)
        frame = read_first_frame(path)
        if frame is not None:
            return frame, cam
    return None, None


def evenly_spaced_indices(n: int, k: int) -> np.ndarray:
    if n == 0 or k <= 0:
        return np.empty(0, dtype=np.int64)
    if k >= n:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, k).round().astype(np.int64)


def _scatter_xy(
    ax,
    xy: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    if xy.size:
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=14,
            alpha=0.7,
            color="#4c78a8",
            edgecolors="black",
            linewidths=0.2,
            label=f"n={len(xy)}",
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)


def _hist(
    ax,
    values: np.ndarray,
    title: str,
    xlabel: str,
    bins: int = 40,
    color: str = "#4c78a8",
) -> None:
    finite = values[np.isfinite(values)]
    if finite.size:
        ax.hist(finite, bins=bins, color=color, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("episodes")


def make_start_positions(
    out_path: Path,
    starts_state: np.ndarray,
    lengths: np.ndarray,
    thumbnails: List[Tuple[int, np.ndarray, Optional[str]]],
    title_lines: Sequence[str],
) -> None:
    """starts_state: (E, 7) array of episode-0 ``observation.state`` rows."""
    n_eps = int(starts_state.shape[0])
    n_thumbs = len(thumbnails)
    has_thumbs = n_thumbs > 0
    fig_h = 16 if has_thumbs else 11
    fig = plt.figure(figsize=(18, fig_h), constrained_layout=True)
    height_ratios = [1.0, 1.0, 1.0, 1.6] if has_thumbs else [1.0, 1.0, 1.0]
    n_rows = 4 if has_thumbs else 3
    gs = fig.add_gridspec(n_rows, 4, height_ratios=height_ratios)

    xy = starts_state[:, [0, 1]]
    xz = starts_state[:, [0, 2]]
    yz = starts_state[:, [1, 2]]
    z = starts_state[:, 2]
    rpy = starts_state[:, 3:6]
    gripper = starts_state[:, 6]

    _scatter_xy(fig.add_subplot(gs[0, 0]), xy, "EE start (x, y) [top-down]", "x [m]", "y [m]")
    _scatter_xy(fig.add_subplot(gs[0, 1]), xz, "EE start (x, z) [side]", "x [m]", "z [m]")
    _scatter_xy(fig.add_subplot(gs[0, 2]), yz, "EE start (y, z)", "y [m]", "z [m]")

    _hist(fig.add_subplot(gs[0, 3]), z, "EE start z [m]", "z")

    _hist(fig.add_subplot(gs[1, 0]), rpy[:, 0], "Initial roll [rad]", "roll", color="#54a24b")
    _hist(fig.add_subplot(gs[1, 1]), rpy[:, 1], "Initial pitch [rad]", "pitch", color="#54a24b")
    _hist(fig.add_subplot(gs[1, 2]), rpy[:, 2], "Initial yaw [rad]", "yaw", color="#54a24b")
    _hist(fig.add_subplot(gs[1, 3]), gripper, "Initial gripper", "gripper", color="#f58518")

    ax = fig.add_subplot(gs[2, 0])
    _hist(ax, lengths.astype(float), "Episode length [frames]", "frames", color="#e45756")

    ax = fig.add_subplot(gs[2, 1:])
    width = 0.6
    x = np.arange(starts_state.shape[1])
    means = np.nanmean(starts_state, axis=0)
    stds = np.nanstd(starts_state, axis=0)
    ax.bar(x, means, width=width, yerr=stds, capsize=3, color="#72b7b2")
    ax.set_xticks(x)
    ax.set_xticklabels(STATE_LABELS, rotation=15)
    ax.set_title("Initial observation.state (mean ± std)")
    ax.set_ylabel("value")
    ax.grid(True, axis="y", alpha=0.25)

    if has_thumbs:
        cols = max(1, min(8, n_thumbs))
        rows_needed = math.ceil(n_thumbs / cols)
        sub_gs = gs[3, :].subgridspec(rows_needed, cols, wspace=0.08, hspace=0.30)
        for i, (ep_idx, frame, cam) in enumerate(thumbnails):
            r, c = divmod(i, cols)
            sub_ax = fig.add_subplot(sub_gs[r, c])
            sub_ax.imshow(frame)
            sub_ax.set_xticks([])
            sub_ax.set_yticks([])
            sub_ax.set_title(f"ep {ep_idx} ({cam})", fontsize=8)

    fig.suptitle("\n".join(title_lines), fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _trace_overlay(
    ax,
    traces: List[np.ndarray],
    dim_x: int,
    dim_y: int,
    title: str,
    xlabel: str,
    ylabel: str,
    max_traces: int = 200,
) -> None:
    if max_traces < len(traces):
        idx = evenly_spaced_indices(len(traces), max_traces)
        traces_to_plot = [traces[i] for i in idx]
    else:
        traces_to_plot = traces
    for t in traces_to_plot:
        if t.shape[0] < 2:
            continue
        ax.plot(t[:, dim_x], t[:, dim_y], linewidth=0.7, alpha=0.35, color="#4c78a8")
    if traces_to_plot:
        starts = np.array([t[0, [dim_x, dim_y]] for t in traces_to_plot])
        ends = np.array([t[-1, [dim_x, dim_y]] for t in traces_to_plot])
        ax.scatter(starts[:, 0], starts[:, 1], s=12, color="#54a24b", label=f"start (n={len(starts)})", zorder=3)
        ax.scatter(ends[:, 0], ends[:, 1], s=12, color="#e45756", label=f"end (n={len(ends)})", zorder=3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)


def make_dashboard(
    out_path: Path,
    states: List[np.ndarray],
    actions: List[np.ndarray],
    lengths: np.ndarray,
    title_lines: Sequence[str],
) -> None:
    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)

    _hist(fig.add_subplot(gs[0, 0]), lengths.astype(float), "Episode lengths [frames]", "frames", color="#4c78a8")

    flat_actions = np.concatenate(actions, axis=0) if actions else np.zeros((0, len(ACTION_LABELS)))
    flat_states = np.concatenate(states, axis=0) if states else np.zeros((0, len(STATE_LABELS)))

    ax = fig.add_subplot(gs[0, 1])
    if flat_actions.size:
        x = np.arange(flat_actions.shape[1])
        ax.bar(x - 0.2, np.nanmean(flat_actions, axis=0), width=0.4, label="mean", color="#4c78a8")
        ax.bar(x + 0.2, np.nanstd(flat_actions, axis=0), width=0.4, label="std", color="#f58518")
        ax.set_xticks(x)
        ax.set_xticklabels(ACTION_LABELS[: flat_actions.shape[1]], rotation=15)
    ax.set_title("Action statistics (per-dim)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)

    ax = fig.add_subplot(gs[0, 2])
    if flat_states.size:
        ax.hist(flat_states[:, 6], bins=50, color="#54a24b", alpha=0.85)
    ax.set_title("Per-step gripper distribution")
    ax.set_xlabel("gripper")
    ax.set_ylabel("steps")

    _trace_overlay(
        fig.add_subplot(gs[1, 0]),
        states,
        dim_x=0,
        dim_y=1,
        title="EE trace (x, y) [top-down]",
        xlabel="x [m]",
        ylabel="y [m]",
    )
    _trace_overlay(
        fig.add_subplot(gs[1, 1]),
        states,
        dim_x=0,
        dim_y=2,
        title="EE trace (x, z) [side]",
        xlabel="x [m]",
        ylabel="z [m]",
    )
    _trace_overlay(
        fig.add_subplot(gs[1, 2]),
        states,
        dim_x=1,
        dim_y=2,
        title="EE trace (y, z)",
        xlabel="y [m]",
        ylabel="z [m]",
    )

    ax = fig.add_subplot(gs[2, :])
    if flat_states.size:
        x = np.arange(flat_states.shape[1])
        ax.bar(x - 0.2, np.nanmean(flat_states, axis=0), width=0.4, label="mean", color="#4c78a8")
        ax.bar(x + 0.2, np.nanstd(flat_states, axis=0), width=0.4, label="std", color="#f58518")
        ax.set_xticks(x)
        ax.set_xticklabels(STATE_LABELS[: flat_states.shape[1]], rotation=15)
    ax.set_title("Per-step observation.state statistics")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("\n".join(title_lines), fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    data_dir: Path = args.data_dir.resolve()
    if not data_dir.exists():
        sys.exit(f"data_dir does not exist: {data_dir}")
    info_path = data_dir / "meta/info.json"
    episodes_path = data_dir / "meta/episodes.jsonl"
    if not info_path.exists() or not episodes_path.exists():
        sys.exit(
            f"Missing meta files in {data_dir}/meta. "
            "Run scriptsv2/filter_bridge_v2.py first."
        )
    info = json.loads(info_path.read_text())
    chunk_size = int(info.get("chunks_size", CHUNK_SIZE_DEFAULT))
    episodes_all = load_jsonl(episodes_path)
    episodes = select_episodes(episodes_all, args.task_filter, args.max_episodes)
    if not episodes:
        sys.exit(
            f"No episodes match task_filter={args.task_filter!r} "
            f"in {episodes_path} (total upstream: {len(episodes_all)})."
        )

    out_dir = (args.out_dir or (data_dir / "plots")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[load] {len(episodes)} episode(s) from {data_dir} "
        f"(task_filter={args.task_filter!r})"
    )

    starts: List[np.ndarray] = []
    lengths: List[int] = []
    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    used_episode_indices: List[int] = []

    skipped_missing = 0
    for ep in episodes:
        ei = int(ep["episode_index"])
        path = episode_parquet_path(data_dir, ei, chunk_size)
        if not path.exists():
            skipped_missing += 1
            continue
        arrs = read_episode_arrays(path)
        if arrs["n_frames"] == 0:
            continue
        starts.append(arrs["state"][0])
        lengths.append(arrs["n_frames"])
        states.append(arrs["state"])
        actions.append(arrs["action"])
        used_episode_indices.append(ei)

    if not starts:
        sys.exit(
            f"No usable episodes found (missing parquet files: {skipped_missing}). "
            "Did the download finish?"
        )

    starts_state = np.asarray(starts, dtype=np.float32)
    lengths_arr = np.asarray(lengths, dtype=np.int64)
    print(
        f"[load] used episodes={len(used_episode_indices)}, "
        f"frames={int(lengths_arr.sum())}, skipped (missing parquet)={skipped_missing}"
    )

    print(f"[thumbs] extracting {min(args.n_thumbnails, len(used_episode_indices))} thumbnails...")
    thumb_idx_positions = evenly_spaced_indices(len(used_episode_indices), args.n_thumbnails)
    thumbnails: List[Tuple[int, np.ndarray, Optional[str]]] = []
    for pos in thumb_idx_positions:
        ei = used_episode_indices[int(pos)]
        frame, cam = first_available_thumbnail(
            data_dir, ei, chunk_size, args.thumbnail_camera
        )
        if frame is not None:
            thumbnails.append((ei, frame, cam))
    print(f"[thumbs] obtained {len(thumbnails)} thumbnail(s)")

    task_strings: List[str] = []
    for ep in episodes:
        for t in ep.get("tasks") or []:
            if t not in task_strings:
                task_strings.append(t)
    title_tag = (
        args.task_filter
        if args.task_filter
        else (task_strings[0] if len(task_strings) == 1 else f"{len(task_strings)} task(s)")
    )
    title_lines = [
        f"{data_dir}",
        f"task: {title_tag}",
        (
            f"episodes={len(used_episode_indices)}  "
            f"frames={int(lengths_arr.sum())}  "
            f"mean_len={lengths_arr.mean():.1f}"
        ),
    ]

    start_path = out_dir / f"{args.tag}_start_positions.png"
    dash_path = out_dir / f"{args.tag}_dashboard.png"

    print(f"[plot] writing {start_path}")
    make_start_positions(start_path, starts_state, lengths_arr, thumbnails, title_lines)
    print(f"[plot] writing {dash_path}")
    make_dashboard(dash_path, states, actions, lengths_arr, title_lines)

    print("Done.")
    print(f"  start positions: {start_path}")
    print(f"  dashboard:       {dash_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
