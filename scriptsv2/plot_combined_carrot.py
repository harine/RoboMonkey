"""Combined start-positions + summary plot for the 'put carrot on plate' task.

Overlays two data sources side-by-side / on shared histograms:
  * RoboMonkey SIMPLER trajectories collected via ``scripts/collect_carrot_on_plate.sh``
    (Zarr v2 shard).
  * Bridge V2 human demonstrations filtered via ``scriptsv2/filter_bridge_v2.py``
    (LeRobot v2.0 layout).

Note on coordinate frames:
  - SIMPLER ``obs/end_effector_pose`` is in the simulator world frame
    (carrot/plate/EE z is around 1.0).
  - Bridge V2 ``observation.state`` is in the arm-base frame (z is around 0..0.25).

These frames are NOT aligned, so EE start positions are shown in side-by-side panels
rather than overlaid on the same axes. Histograms (episode length, action stats,
gripper) and asset panels are still meaningful to compare.

Usage:
    python scriptsv2/plot_combined_carrot.py \
        --simpler_shard openvla-mini/data/carrot_on_plate/state0.zarr \
        --bridge_dir data/bridge_v2_filtered \
        --out_dir openvla-mini/data/carrot_on_plate \
        --tag carrot_combined
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

try:
    import zarr
except ImportError as e:
    sys.exit(f"zarr is required: {e}")

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
ACTION_LABELS = ("dx", "dy", "dz", "dr", "dp", "dyaw", "grip")

C_SIM_SUCCESS = "#54a24b"
C_SIM_FAIL = "#bab0ac"
C_BRIDGE = "#f58518"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--simpler_shard",
        type=Path,
        default=Path("openvla-mini/data/carrot_on_plate/state0.zarr"),
        help="Path to the collected SIMPLER Zarr shard (default: %(default)s).",
    )
    parser.add_argument(
        "--bridge_dir",
        type=Path,
        default=Path("data/bridge_v2_filtered"),
        help="Filtered Bridge V2 LeRobot dataset root (default: %(default)s).",
    )
    parser.add_argument(
        "--bridge_task_filter",
        type=str,
        default="put carrot on plate",
        help="Bridge task string to keep (default: %(default)r).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory (default: <bridge_dir>/plots).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="carrot_combined",
        help="Filename prefix (default: %(default)s).",
    )
    parser.add_argument(
        "--success_mode",
        choices=("reward_any", "reward_sum", "done_last"),
        default="reward_any",
        help="How to label SIMPLER episodes as successful (default: %(default)s).",
    )
    parser.add_argument(
        "--reward_threshold",
        type=float,
        default=0.0,
        help="Threshold used by reward_any / reward_sum (default: %(default)s).",
    )
    parser.add_argument(
        "--max_simpler_episodes",
        type=int,
        default=None,
        help="Optional cap on SIMPLER episodes loaded.",
    )
    parser.add_argument(
        "--max_bridge_episodes",
        type=int,
        default=None,
        help="Optional cap on Bridge V2 episodes loaded.",
    )
    parser.add_argument(
        "--n_thumbnails",
        type=int,
        default=8,
        help="Bridge V2 first-frame thumbnails in the bottom strip (default: %(default)s).",
    )
    parser.add_argument(
        "--simpler_x_flip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Negate the x coordinate of SIMPLER end-effector / carrot / plate "
            "positions to align with the Bridge V2 arm-base convention "
            "(SIMPLER world frame has +x pointing opposite to widowx +x). "
            "Default: enabled. Disable with --no-simpler_x_flip."
        ),
    )
    parser.add_argument(
        "--simpler_z_offset",
        type=str,
        default="auto",
        help=(
            "Subtract this constant from SIMPLER z (ee/carrot/plate) so its scale "
            "matches Bridge V2 (arm-base frame, table at z=0). Use a number, or "
            "'auto' (default) to align SIMPLER median z to Bridge median z, or "
            "'none' to disable."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("focused", "full"),
        default="focused",
        help=(
            "'focused' (default): only EE/carrot/plate/offset (x,y) panels with "
            "shared legend outside the plot. 'full': the original multi-row "
            "dashboard with episode lengths, action stats, gripper, thumbnails."
        ),
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# SIMPLER Zarr loaders
# --------------------------------------------------------------------------- #


def episode_bounds(episode_ends: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    starts = np.concatenate([[0], episode_ends[:-1]]).astype(np.int64)
    ends = episode_ends.astype(np.int64)
    return starts, ends


def classify_simpler_success(
    rewards: zarr.Array,
    dones: zarr.Array,
    starts: np.ndarray,
    ends: np.ndarray,
    mode: str,
    reward_threshold: float,
) -> np.ndarray:
    success = np.zeros(len(ends), dtype=bool)
    for i, (s, e) in enumerate(zip(starts, ends)):
        if e <= s:
            continue
        if mode == "done_last":
            success[i] = bool(dones[int(e - 1)])
            continue
        ep = np.asarray(rewards[int(s) : int(e)])
        if mode == "reward_any":
            success[i] = bool(np.any(ep > reward_threshold))
        elif mode == "reward_sum":
            success[i] = bool(np.sum(ep) > reward_threshold)
        else:
            raise ValueError(f"Unknown success mode: {mode}")
    return success


def collect_simpler_starts(
    root: zarr.Group,
    starts: np.ndarray,
    ends: np.ndarray,
    valid: np.ndarray,
) -> dict:
    obs = root["data/obs"]
    starts_v = starts[valid].astype(np.int64)
    ee_xyz = np.asarray(obs["end_effector_pose"].get_orthogonal_selection((starts_v, slice(0, 3))))
    src_xyz = np.asarray(obs["insertive_asset_pose"].get_orthogonal_selection((starts_v, slice(0, 3))))
    tgt_xyz = np.asarray(obs["receptive_asset_pose"].get_orthogonal_selection((starts_v, slice(0, 3))))
    grip = np.asarray(obs["last_gripper_action"].get_orthogonal_selection((starts_v, slice(None)))).reshape(-1)
    return {
        "ee_xyz": ee_xyz,
        "carrot_xyz": src_xyz,
        "plate_xyz": tgt_xyz,
        "gripper": grip,
    }


def load_simpler_summary(
    shard_path: Path,
    success_mode: str,
    reward_threshold: float,
    max_episodes: Optional[int],
    flip_x: bool = False,
    z_offset: float = 0.0,
) -> dict:
    if not shard_path.exists():
        sys.exit(f"SIMPLER shard not found: {shard_path}")
    root = zarr.open_group(str(shard_path), mode="r")
    episode_ends = np.asarray(root["meta/episode_ends"][:])
    starts, ends = episode_bounds(episode_ends)
    if max_episodes is not None and max_episodes < len(ends):
        starts = starts[:max_episodes]
        ends = ends[:max_episodes]

    success = classify_simpler_success(
        root["data/rewards"], root["data/dones"], starts, ends,
        success_mode, reward_threshold,
    )
    valid = ends > starts
    starts_data = collect_simpler_starts(root, starts, ends, valid)
    success_v = success[valid]
    lengths_all = (ends - starts).astype(np.int64)
    lengths_valid = lengths_all[valid]

    # SIMPLER's world frame has +x pointing opposite the widowx arm-base +x. To
    # overlay with Bridge V2 (arm-base frame) we negate the x coordinate of the
    # tracked positions only -- actions/gripper/lengths are unaffected.
    if flip_x:
        for key in ("ee_xyz", "carrot_xyz", "plate_xyz"):
            if starts_data[key].size:
                starts_data[key] = starts_data[key].copy()
                starts_data[key][:, 0] *= -1.0
    # SIMPLER table top sits at z ~= 0.87 in world frame while Bridge V2 puts
    # the table at z = 0 in arm-base frame. Subtracting a constant aligns the
    # vertical scales so EE/carrot/plate z values overlap.
    if z_offset:
        for key in ("ee_xyz", "carrot_xyz", "plate_xyz"):
            if starts_data[key].size:
                if not starts_data[key].flags.writeable:
                    starts_data[key] = starts_data[key].copy()
                starts_data[key][:, 2] -= z_offset

    actions_arr = root["data/actions"]
    n_steps = int(actions_arr.shape[0])
    sample_step = max(1, int(np.ceil(n_steps / 200_000)))
    actions_subsampled = np.asarray(actions_arr[::sample_step])

    return {
        "shard": shard_path,
        "attrs": dict(root.attrs),
        "starts": starts,
        "ends": ends,
        "valid": valid,
        "success_valid": success_v,
        "ee_xyz": starts_data["ee_xyz"],
        "carrot_xyz": starts_data["carrot_xyz"],
        "plate_xyz": starts_data["plate_xyz"],
        "gripper": starts_data["gripper"],
        "lengths": lengths_valid,
        "actions_sub": actions_subsampled,
        "n_episodes": int(len(ends)),
        "n_episodes_valid": int(valid.sum()),
        "n_success": int(success_v.sum()),
    }


# --------------------------------------------------------------------------- #
# Bridge V2 LeRobot loaders
# --------------------------------------------------------------------------- #


def load_jsonl(path: Path) -> List[dict]:
    with path.open("r") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def episode_parquet_path(data_dir: Path, episode_index: int, chunk_size: int) -> Path:
    chunk = episode_index // chunk_size
    return data_dir / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"


def episode_video_path(data_dir: Path, episode_index: int, chunk_size: int, camera: str) -> Path:
    chunk = episode_index // chunk_size
    return (
        data_dir
        / f"videos/chunk-{chunk:03d}/observation.images.{camera}/episode_{episode_index:06d}.mp4"
    )


def _column_to_2d(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == object:
        return np.asarray([np.asarray(row, dtype=np.float32) for row in arr])
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def read_first_frame(video_path: Path) -> Optional[np.ndarray]:
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
    data_dir: Path, episode_index: int, chunk_size: int
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    for cam in CAMERA_KEYS:
        frame = read_first_frame(episode_video_path(data_dir, episode_index, chunk_size, cam))
        if frame is not None:
            return frame, cam
    return None, None


def evenly_spaced_indices(n: int, k: int) -> np.ndarray:
    if n == 0 or k <= 0:
        return np.empty(0, dtype=np.int64)
    if k >= n:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, k).round().astype(np.int64)


def load_bridge_summary(
    bridge_dir: Path,
    task_filter: Optional[str],
    max_episodes: Optional[int],
    n_thumbnails: int,
) -> dict:
    info_path = bridge_dir / "meta/info.json"
    eps_path = bridge_dir / "meta/episodes.jsonl"
    if not info_path.exists() or not eps_path.exists():
        sys.exit(f"Bridge meta not found in {bridge_dir}/meta")
    info = json.loads(info_path.read_text())
    chunk_size = int(info.get("chunks_size", CHUNK_SIZE_DEFAULT))
    episodes = load_jsonl(eps_path)
    if task_filter is not None:
        episodes = [e for e in episodes if task_filter in (e.get("tasks") or [])]
    episodes.sort(key=lambda e: int(e["episode_index"]))
    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    starts: List[np.ndarray] = []
    lengths: List[int] = []
    states_concat: List[np.ndarray] = []
    actions_concat: List[np.ndarray] = []
    used_indices: List[int] = []

    for ep in episodes:
        ei = int(ep["episode_index"])
        path = episode_parquet_path(bridge_dir, ei, chunk_size)
        if not path.exists():
            continue
        table = pq.read_table(path, columns=["observation.state", "action"])
        state = _column_to_2d(table.column("observation.state").to_numpy(zero_copy_only=False))
        action = _column_to_2d(table.column("action").to_numpy(zero_copy_only=False))
        if state.shape[0] == 0:
            continue
        starts.append(state[0])
        lengths.append(state.shape[0])
        states_concat.append(state)
        actions_concat.append(action)
        used_indices.append(ei)

    if not starts:
        sys.exit("No Bridge episodes loaded.")

    starts_arr = np.asarray(starts, dtype=np.float32)
    lengths_arr = np.asarray(lengths, dtype=np.int64)
    flat_states = np.concatenate(states_concat, axis=0) if states_concat else np.zeros((0, 7))
    flat_actions = np.concatenate(actions_concat, axis=0) if actions_concat else np.zeros((0, 7))

    # Bridge V2 has no tracked asset poses, so we use gripper-event heuristics:
    #   carrot proxy: EE position at the step with the smallest gripper value
    #     (i.e. most closed = grasp moment).
    #   plate proxy:  EE position at the final step (placement / end of episode).
    # state columns: [x, y, z, roll, pitch, yaw, gripper] in arm-base frame.
    carrot_proxy: List[np.ndarray] = []
    plate_proxy: List[np.ndarray] = []
    for st in states_concat:
        if st.shape[0] == 0:
            continue
        grasp_idx = int(np.argmin(st[:, 6])) if st.shape[1] >= 7 else 0
        carrot_proxy.append(st[grasp_idx, :3])
        plate_proxy.append(st[-1, :3])
    carrot_proxy_arr = np.asarray(carrot_proxy, dtype=np.float32) if carrot_proxy else np.zeros((0, 3))
    plate_proxy_arr = np.asarray(plate_proxy, dtype=np.float32) if plate_proxy else np.zeros((0, 3))

    thumb_positions = evenly_spaced_indices(len(used_indices), n_thumbnails)
    thumbnails: List[Tuple[int, np.ndarray, Optional[str]]] = []
    for pos in thumb_positions:
        ei = used_indices[int(pos)]
        frame, cam = first_available_thumbnail(bridge_dir, ei, chunk_size)
        if frame is not None:
            thumbnails.append((ei, frame, cam))

    return {
        "bridge_dir": bridge_dir,
        "starts": starts_arr,
        "lengths": lengths_arr,
        "flat_states": flat_states,
        "flat_actions": flat_actions,
        "carrot_proxy_xyz": carrot_proxy_arr,
        "plate_proxy_xyz": plate_proxy_arr,
        "thumbnails": thumbnails,
        "n_episodes": int(starts_arr.shape[0]),
    }


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #


def _percentile_limits(
    arrays: Sequence[np.ndarray],
    axis: int,
    lo: float = 1.0,
    hi: float = 99.0,
    pad_frac: float = 0.05,
) -> Optional[Tuple[float, float]]:
    """Compute an axis range from a percentile of pooled values across arrays."""
    vals = []
    for a in arrays:
        if a is None or a.size == 0:
            continue
        col = np.asarray(a)[:, axis]
        col = col[np.isfinite(col)]
        if col.size:
            vals.append(col)
    if not vals:
        return None
    pooled = np.concatenate(vals)
    lo_v = float(np.percentile(pooled, lo))
    hi_v = float(np.percentile(pooled, hi))
    if hi_v == lo_v:
        return (lo_v - 1.0, lo_v + 1.0)
    pad = (hi_v - lo_v) * pad_frac
    return (lo_v - pad, hi_v + pad)


def _scatter_combined(
    ax,
    series: Sequence[Tuple[str, Optional[np.ndarray], str, float, float, float]],
    title: str,
    xlabel: str,
    ylabel: str,
    clip_percentile: bool = True,
) -> None:
    """Overlay multiple scatter populations on the same axes.

    Each series tuple is ``(label, xy[N,2], color, marker_size, alpha, edge_lw)``.
    The legend reports per-source counts; axes are clipped to the 1st-99th
    percentile of the pooled point cloud so SIMPLER physics-failure outliers
    don't squish the visible cluster.
    """
    xy_arrays = [s[1] for s in series if s[1] is not None and s[1].size]
    for label, xy, color, size, alpha, edge_lw in series:
        if xy is None or xy.size == 0:
            continue
        ax.scatter(
            xy[:, 0], xy[:, 1],
            s=size, alpha=alpha, color=color,
            edgecolors="black" if edge_lw > 0 else "none",
            linewidths=edge_lw,
            label=f"{label} (n={len(xy)})",
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=7)
    if clip_percentile and xy_arrays:
        x_lim = _percentile_limits(xy_arrays, axis=0)
        y_lim = _percentile_limits(xy_arrays, axis=1)
        if x_lim is not None:
            ax.set_xlim(*x_lim)
        if y_lim is not None:
            ax.set_ylim(*y_lim)


def _hist_overlay(
    ax,
    series: Sequence[Tuple[str, np.ndarray, str]],
    title: str,
    xlabel: str,
    bins: int = 40,
    clip_percentile: bool = True,
) -> None:
    finite = [s[1][np.isfinite(s[1])] if s[1] is not None else np.array([]) for s in series]
    nonempty = [v for v in finite if v.size]
    if nonempty:
        pooled = np.concatenate(nonempty)
        if clip_percentile and pooled.size > 50:
            rng = (float(np.percentile(pooled, 1)), float(np.percentile(pooled, 99)))
        else:
            rng = (float(pooled.min()), float(pooled.max()))
        if rng[0] == rng[1]:
            rng = (rng[0] - 1e-3, rng[1] + 1e-3)
    else:
        rng = None
    for (label, values, color), values_finite in zip(series, finite):
        if values_finite.size == 0:
            continue
        ax.hist(
            values_finite, bins=bins, range=rng,
            color=color, alpha=0.55, label=f"{label} (n={len(values_finite)})",
            density=True,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.25)


def _grouped_bar_actions(ax, simpler: np.ndarray, bridge: np.ndarray, stat: str) -> None:
    ndim = max(simpler.shape[1] if simpler.size else 0, bridge.shape[1] if bridge.size else 0, 7)
    x = np.arange(ndim)
    width = 0.4
    if stat == "mean":
        sim_v = np.nanmean(simpler, axis=0) if simpler.size else np.zeros(ndim)
        br_v = np.nanmean(bridge, axis=0) if bridge.size else np.zeros(ndim)
    else:
        sim_v = np.nanstd(simpler, axis=0) if simpler.size else np.zeros(ndim)
        br_v = np.nanstd(bridge, axis=0) if bridge.size else np.zeros(ndim)
    ax.bar(x - width / 2, sim_v[:ndim], width=width, label="SIMPLER", color=C_SIM_SUCCESS)
    ax.bar(x + width / 2, br_v[:ndim], width=width, label="Bridge V2", color=C_BRIDGE)
    ax.set_xticks(x)
    ax.set_xticklabels(ACTION_LABELS[:ndim], rotation=15)
    ax.set_title(f"Action {stat} per-dim")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)


# --------------------------------------------------------------------------- #
# Master figure
# --------------------------------------------------------------------------- #


def make_focused(
    out_path: Path,
    sim: dict,
    br: dict,
    title_lines: Sequence[str],
) -> None:
    """2x3 figure with EE/carrot/plate/offset panels (xy + EE xz/yz) and a
    single shared legend outside the plot area (right margin).

    Z values are expected to already be in a shared frame (table at z=0); see
    --simpler_z_offset in the CLI. Signs are NOT flipped on z."""
    succ = sim["success_valid"]
    fail = ~succ

    sim_ee_xy = sim["ee_xyz"][:, [0, 1]]
    sim_ee_xz = sim["ee_xyz"][:, [0, 2]]
    sim_ee_yz = sim["ee_xyz"][:, [1, 2]]
    sim_carrot_xy = sim["carrot_xyz"][:, [0, 1]]
    sim_plate_xy = sim["plate_xyz"][:, [0, 1]]
    sim_offset_xy = sim_carrot_xy - sim_plate_xy

    br_ee_xy = br["starts"][:, [0, 1]] if br["starts"].size else np.zeros((0, 2))
    br_ee_xz = br["starts"][:, [0, 2]] if br["starts"].size else np.zeros((0, 2))
    br_ee_yz = br["starts"][:, [1, 2]] if br["starts"].size else np.zeros((0, 2))
    br_carrot_xy = br["carrot_proxy_xyz"][:, [0, 1]] if br["carrot_proxy_xyz"].size else np.zeros((0, 2))
    br_plate_xy = br["plate_proxy_xyz"][:, [0, 1]] if br["plate_proxy_xyz"].size else np.zeros((0, 2))
    br_offset_xy = (
        br_carrot_xy - br_plate_xy
        if br_carrot_xy.size and br_plate_xy.size
        else np.zeros((0, 2))
    )

    # Reserve right margin for the shared legend (no constrained_layout because
    # we want manual control over the legend bbox).
    fig = plt.figure(figsize=(20, 11))
    fig.subplots_adjust(left=0.05, right=0.86, top=0.86, bottom=0.07,
                        wspace=0.30, hspace=0.40)
    axes = fig.subplots(2, 3)

    panel_specs = [
        (axes[0, 0], "EE start (x, y)",          sim_ee_xy,     br_ee_xy,     "x [m]", "y [m]"),
        (axes[0, 1], "EE start (x, z)",          sim_ee_xz,     br_ee_xz,     "x [m]", "z [m]"),
        (axes[0, 2], "EE start (y, z)",          sim_ee_yz,     br_ee_yz,     "y [m]", "z [m]"),
        (axes[1, 0], "Carrot start (x, y)\n[Bridge proxy = EE @ gripper-min]",
                                                  sim_carrot_xy, br_carrot_xy, "x [m]", "y [m]"),
        (axes[1, 1], "Plate start (x, y)\n[Bridge proxy = EE @ final step]",
                                                  sim_plate_xy,  br_plate_xy,  "x [m]", "y [m]"),
        (axes[1, 2], "Carrot - Plate offset (x, y)",
                                                  sim_offset_xy, br_offset_xy, "dx [m]", "dy [m]"),
    ]

    legend_handles = []
    for i, (ax, title, sim_xy, br_xy, xlabel, ylabel) in enumerate(panel_specs):
        clouds = []
        if sim_xy.size and fail.any():
            h = ax.scatter(sim_xy[fail, 0], sim_xy[fail, 1],
                           s=8, alpha=0.25, color=C_SIM_FAIL)
            clouds.append((h, f"SIMPLER fail (n={int(fail.sum())})", sim_xy[fail]))
        if sim_xy.size and succ.any():
            h = ax.scatter(sim_xy[succ, 0], sim_xy[succ, 1],
                           s=14, alpha=0.85, color=C_SIM_SUCCESS,
                           edgecolors="black", linewidths=0.3)
            clouds.append((h, f"SIMPLER success (n={int(succ.sum())})", sim_xy[succ]))
        if br_xy.size:
            label = (
                f"Bridge V2 (n={len(br_xy)})"
                if "proxy" not in title.lower()
                else f"Bridge V2 proxy (n={len(br_xy)})"
            )
            h = ax.scatter(br_xy[:, 0], br_xy[:, 1],
                           s=14, alpha=0.70, color=C_BRIDGE,
                           edgecolors="black", linewidths=0.2)
            clouds.append((h, label, br_xy))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        clouds_xy = [c[2] for c in clouds]
        if clouds_xy:
            x_lim = _percentile_limits(clouds_xy, axis=0)
            y_lim = _percentile_limits(clouds_xy, axis=1)
            if x_lim is not None:
                ax.set_xlim(*x_lim)
            if y_lim is not None:
                ax.set_ylim(*y_lim)
        if i == 0:
            for handle, lbl, _ in clouds:
                legend_handles.append((handle, lbl))

    if legend_handles:
        fig.legend(
            [h for h, _ in legend_handles],
            [lbl for _, lbl in legend_handles],
            loc="center right",
            bbox_to_anchor=(0.995, 0.5),
            frameon=True,
            fontsize=10,
            title="Source",
        )

    fig.suptitle("\n".join(title_lines), fontsize=12, y=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def make_combined(
    out_path: Path,
    sim: dict,
    br: dict,
    title_lines: Sequence[str],
) -> None:
    has_thumbs = bool(br["thumbnails"])
    n_rows = 4 if has_thumbs else 3
    height_ratios = [1.0, 1.0, 1.0, 1.6] if has_thumbs else [1.0, 1.0, 1.0]
    fig_h = 18 if has_thumbs else 13
    fig = plt.figure(figsize=(20, fig_h), constrained_layout=True)
    gs = fig.add_gridspec(n_rows, 4, height_ratios=height_ratios)

    succ = sim["success_valid"]
    fail = ~succ

    sim_ee_xy = sim["ee_xyz"][:, [0, 1]]
    sim_ee_xz = sim["ee_xyz"][:, [0, 2]]
    sim_ee_yz = sim["ee_xyz"][:, [1, 2]]
    sim_carrot_xy = sim["carrot_xyz"][:, [0, 1]]
    sim_plate_xy = sim["plate_xyz"][:, [0, 1]]
    sim_offset_xy = sim_carrot_xy - sim_plate_xy
    sim_offset_dist = np.linalg.norm(sim_offset_xy, axis=1)

    br_ee_xy = br["starts"][:, [0, 1]] if br["starts"].size else np.zeros((0, 2))
    br_ee_xz = br["starts"][:, [0, 2]] if br["starts"].size else np.zeros((0, 2))
    br_ee_yz = br["starts"][:, [1, 2]] if br["starts"].size else np.zeros((0, 2))
    br_carrot_xy = br["carrot_proxy_xyz"][:, [0, 1]] if br["carrot_proxy_xyz"].size else np.zeros((0, 2))
    br_plate_xy = br["plate_proxy_xyz"][:, [0, 1]] if br["plate_proxy_xyz"].size else np.zeros((0, 2))
    br_offset_xy = (
        br_carrot_xy - br_plate_xy
        if br_carrot_xy.size and br_plate_xy.size
        else np.zeros((0, 2))
    )
    br_offset_dist = np.linalg.norm(br_offset_xy, axis=1) if br_offset_xy.size else np.zeros((0,))

    def _ee_series(xy_arr_sim, xy_arr_br):
        return [
            ("SIMPLER fail",    xy_arr_sim[fail] if xy_arr_sim.size else None, C_SIM_FAIL,    8,  0.25, 0.0),
            ("SIMPLER success", xy_arr_sim[succ] if xy_arr_sim.size else None, C_SIM_SUCCESS, 14, 0.80, 0.3),
            ("Bridge V2",       xy_arr_br,                                     C_BRIDGE,      14, 0.70, 0.2),
        ]

    _scatter_combined(
        fig.add_subplot(gs[0, 0]),
        _ee_series(sim_ee_xy, br_ee_xy),
        "EE start (x, y)\n[SIMPLER world / Bridge arm-base frames superimposed]",
        "x [m]", "y [m]",
    )
    _scatter_combined(
        fig.add_subplot(gs[0, 1]),
        _ee_series(sim_ee_xz, br_ee_xz),
        "EE start (x, z)", "x [m]", "z [m]",
    )
    _scatter_combined(
        fig.add_subplot(gs[0, 2]),
        _ee_series(sim_ee_yz, br_ee_yz),
        "EE start (y, z)", "y [m]", "z [m]",
    )

    ax = fig.add_subplot(gs[0, 3])
    sim_z = sim["ee_xyz"][:, 2]
    br_z = br["starts"][:, 2] if br["starts"].size else np.array([])
    _hist_overlay(
        ax,
        [
            ("SIMPLER fail", sim_z[fail], C_SIM_FAIL),
            ("SIMPLER success", sim_z[succ], C_SIM_SUCCESS),
            ("Bridge V2", br_z, C_BRIDGE),
        ],
        title="EE start z [m]",
        xlabel="z",
    )

    _scatter_combined(
        fig.add_subplot(gs[1, 0]),
        [
            ("SIMPLER carrot fail",     sim_carrot_xy[fail], C_SIM_FAIL,    8,  0.25, 0.0),
            ("SIMPLER carrot success",  sim_carrot_xy[succ], C_SIM_SUCCESS, 14, 0.80, 0.3),
            ("Bridge carrot (proxy)",   br_carrot_xy,        C_BRIDGE,      14, 0.70, 0.2),
        ],
        "Carrot start (x, y)\n[Bridge proxy = EE @ gripper-min]",
        "x [m]", "y [m]",
    )
    _scatter_combined(
        fig.add_subplot(gs[1, 1]),
        [
            ("SIMPLER plate fail",     sim_plate_xy[fail], C_SIM_FAIL,    8,  0.25, 0.0),
            ("SIMPLER plate success",  sim_plate_xy[succ], C_SIM_SUCCESS, 14, 0.80, 0.3),
            ("Bridge plate (proxy)",   br_plate_xy,        C_BRIDGE,      14, 0.70, 0.2),
        ],
        "Plate start (x, y)\n[Bridge proxy = EE @ final step]",
        "x [m]", "y [m]",
    )
    _scatter_combined(
        fig.add_subplot(gs[1, 2]),
        [
            ("SIMPLER offset fail",    sim_offset_xy[fail], C_SIM_FAIL,    8,  0.25, 0.0),
            ("SIMPLER offset success", sim_offset_xy[succ], C_SIM_SUCCESS, 14, 0.80, 0.3),
            ("Bridge offset (proxy)",  br_offset_xy,        C_BRIDGE,      14, 0.70, 0.2),
        ],
        "Carrot - Plate offset (x, y)",
        "dx [m]", "dy [m]",
    )

    _hist_overlay(
        fig.add_subplot(gs[1, 3]),
        [
            ("SIMPLER fail", sim_offset_dist[fail], C_SIM_FAIL),
            ("SIMPLER success", sim_offset_dist[succ], C_SIM_SUCCESS),
            ("Bridge (proxy)", br_offset_dist, C_BRIDGE),
        ],
        title="|Carrot - Plate| distance",
        xlabel="distance [m]",
    )

    _hist_overlay(
        fig.add_subplot(gs[2, 0]),
        [
            ("SIMPLER fail", sim["lengths"][~succ], C_SIM_FAIL),
            ("SIMPLER success", sim["lengths"][succ], C_SIM_SUCCESS),
            ("Bridge V2", br["lengths"], C_BRIDGE),
        ],
        title="Episode length [frames]",
        xlabel="frames",
    )

    _grouped_bar_actions(fig.add_subplot(gs[2, 1]), sim["actions_sub"], br["flat_actions"], "mean")
    _grouped_bar_actions(fig.add_subplot(gs[2, 2]), sim["actions_sub"], br["flat_actions"], "std")

    br_grip_start = br["starts"][:, 6] if br["starts"].shape[1] >= 7 else np.array([])
    _hist_overlay(
        fig.add_subplot(gs[2, 3]),
        [
            ("SIMPLER", sim["gripper"], C_SIM_SUCCESS),
            ("Bridge V2", br_grip_start, C_BRIDGE),
        ],
        title="Initial gripper command",
        xlabel="gripper",
        bins=30,
    )

    if has_thumbs:
        thumbs = br["thumbnails"]
        cols = max(1, min(8, len(thumbs)))
        rows_needed = math.ceil(len(thumbs) / cols)
        sub_gs = gs[3, :].subgridspec(rows_needed, cols, wspace=0.08, hspace=0.30)
        for i, (ep_idx, frame, cam) in enumerate(thumbs):
            r, c = divmod(i, cols)
            sub_ax = fig.add_subplot(sub_gs[r, c])
            sub_ax.imshow(frame)
            sub_ax.set_xticks([])
            sub_ax.set_yticks([])
            sub_ax.set_title(f"Bridge V2 ep {ep_idx} ({cam})", fontsize=8)

    fig.suptitle("\n".join(title_lines), fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    args = parse_args()

    print(f"[load] Bridge V2 dir: {args.bridge_dir}  task_filter={args.bridge_task_filter!r}")
    n_thumbs = args.n_thumbnails if args.mode == "full" else 0
    br = load_bridge_summary(
        args.bridge_dir,
        args.bridge_task_filter,
        args.max_bridge_episodes,
        n_thumbs,
    )
    print(
        f"  episodes={br['n_episodes']}  total_frames={int(br['lengths'].sum())}  "
        f"thumbnails={len(br['thumbnails'])}"
    )

    # Resolve simpler_z_offset before loading SIMPLER. 'auto' uses the median z
    # of the SIMPLER carrot/plate (insertive/receptive) start poses as the
    # table-top reference -- subtracting it puts the table at z=0 in both
    # sources without flipping signs (Bridge V2 already has table at z=0 in
    # arm-base frame). This keeps the EE above the table and the assets at the
    # table in the resulting plot.
    z_off = 0.0
    z_choice = (args.simpler_z_offset or "").strip().lower()
    if z_choice in ("", "none", "0"):
        z_off = 0.0
    elif z_choice == "auto":
        sim_root = zarr.open_group(str(args.simpler_shard), mode="r")
        sim_ends = np.asarray(sim_root["meta/episode_ends"][:])
        sim_starts = np.concatenate([[0], sim_ends[:-1]]).astype(np.int64)
        sim_starts_v = sim_starts[sim_ends > sim_starts].astype(np.int64)
        carrot_z = np.asarray(
            sim_root["data/obs/insertive_asset_pose"].get_orthogonal_selection(
                (sim_starts_v, 2)
            )
        )
        plate_z = np.asarray(
            sim_root["data/obs/receptive_asset_pose"].get_orthogonal_selection(
                (sim_starts_v, 2)
            )
        )
        table_zs = np.concatenate([carrot_z, plate_z])
        z_off = float(np.median(table_zs)) if table_zs.size else 0.0
    else:
        try:
            z_off = float(z_choice)
        except ValueError:
            sys.exit(f"--simpler_z_offset must be a number, 'auto', or 'none' (got {z_choice!r})")

    print(
        f"[load] SIMPLER shard: {args.simpler_shard}  "
        f"(x_flip={'on' if args.simpler_x_flip else 'off'}, z_offset={z_off:+.4f})"
    )
    sim = load_simpler_summary(
        args.simpler_shard,
        args.success_mode,
        args.reward_threshold,
        args.max_simpler_episodes,
        flip_x=args.simpler_x_flip,
        z_offset=z_off,
    )
    print(
        f"  episodes={sim['n_episodes']}  valid={sim['n_episodes_valid']}  "
        f"success={sim['n_success']}/{sim['n_episodes_valid']} "
        f"({sim['n_success'] / max(1, sim['n_episodes_valid']):.3f})  "
        f"mode={args.success_mode}"
    )

    out_dir = (args.out_dir or (args.bridge_dir / "plots")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.tag}.png"

    title_lines = [
        f"SIMPLER: {args.simpler_shard}  |  Bridge V2: {args.bridge_dir}",
        (
            f"task: {sim['attrs'].get('task_description', 'n/a')}  "
            f"|  bridge filter: {args.bridge_task_filter!r}  "
            f"|  simpler x_flip={'on' if args.simpler_x_flip else 'off'} z_offset={z_off:+.3f}"
        ),
        (
            f"SIMPLER episodes={sim['n_episodes_valid']} "
            f"(success {sim['n_success']}/{sim['n_episodes_valid']})  "
            f"|  Bridge V2 episodes={br['n_episodes']} "
            f"(frames={int(br['lengths'].sum())})"
        ),
    ]

    print(f"[plot] writing ({args.mode}) {out_path}")
    if args.mode == "focused":
        make_focused(out_path, sim, br, title_lines)
    else:
        make_combined(out_path, sim, br, title_lines)
    print(f"Done.\n  combined plot: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
