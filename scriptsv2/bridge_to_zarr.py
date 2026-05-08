"""Convert a filtered Bridge V2 LeRobot subset into a SIMPLER-compatible Zarr shard.

The output shard mirrors the on-disk layout used by ``scripts/collect_carrot_on_plate.sh``
so it can be loaded with the same downstream code (e.g. ``scripts/zarr_dataset_dashboard.py``,
``scriptsv2/plot_combined_carrot.py``):

    <out>.zarr/
        attrs:                env_name, task_description, source, frame, notes
        data/actions          (N, 7) float32
        data/dones            (N,)   bool
        data/rewards          (N,)   float32
        data/obs/
            arm_joint_pos                          (N, 6) float32   <- zero-filled
            binary_contact                         (N,)   float32   <- zero-filled
            end_effector_pose                      (N, 7) float32   (x, y, z, qx, qy, qz, qw)
            end_effector_vel_lin_ang_b             (N, 6) float32   <- finite difference
            expert_action_mean                     (N, 7) float32   = recorded action
            expert_action_std                      (N, 7) float32   <- zero-filled
            insertive_asset_in_receptive_asset_frame (N, 7) float32 <- carrot-plate translation proxy
            insertive_asset_pose                   (N, 7) float32   <- carrot proxy (EE @ gripper-min)
            joint_vel                              (N, 6) float32   <- zero-filled
            last_arm_action                        (N, 6) float32   = previous action[:6]
            last_gripper_action                    (N, 1) float32   = observation.state[6]
            receptive_asset_pose                   (N, 7) float32   <- plate proxy (EE @ final step)
        meta/episode_ends     (E,)   int64

Bridge V2 ``observation.state`` is ``[x, y, z, roll, pitch, yaw, gripper]`` in arm-base
frame; rotations are converted to quaternions ``(qx, qy, qz, qw)`` via scipy's
extrinsic XYZ convention.

Usage:
    python scriptsv2/bridge_to_zarr.py
    python scriptsv2/bridge_to_zarr.py \
        --bridge_dir data/bridge_v2_filtered \
        --task_filter "put carrot on plate" \
        --out openvla-mini/data/carrot_on_plate/bridge_v2_carrot.zarr
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

try:
    import pyarrow.parquet as pq
except ImportError as e:
    sys.exit(f"pyarrow is required: {e}")

try:
    import zarr
    from numcodecs import Blosc
except ImportError as e:
    sys.exit(f"zarr / numcodecs are required: {e}")

try:
    from scipy.spatial.transform import Rotation as R
except ImportError as e:
    sys.exit(f"scipy is required: {e}")

CHUNK_SIZE_DEFAULT = 1000
DEFAULT_CHUNK_ROWS = 1024
DEFAULT_COMPRESSOR = Blosc(cname="zstd", clevel=3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bridge_dir",
        type=Path,
        default=Path("data/bridge_v2_filtered"),
        help="Filtered Bridge V2 LeRobot dataset root (default: %(default)s).",
    )
    parser.add_argument(
        "--task_filter",
        type=str,
        default="put carrot on plate",
        help="Bridge task string to include (default: %(default)r).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("openvla-mini/data/carrot_on_plate/bridge_v2_carrot.zarr"),
        help="Destination .zarr shard (default: %(default)s).",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Optional cap on episodes to convert (sorted by episode_index).",
    )
    parser.add_argument(
        "--euler_convention",
        type=str,
        default="xyz",
        choices=("xyz", "ZYX", "XYZ", "zyx"),
        help=(
            "Convention passed to scipy.spatial.transform.Rotation.from_euler() "
            "for converting Bridge V2 (roll, pitch, yaw) to quaternions "
            "(default: %(default)s = extrinsic xyz)."
        ),
    )
    parser.add_argument(
        "--success_reward",
        type=float,
        default=1.0,
        help=(
            "Reward to write at the final step of each episode (Bridge demos are "
            "successful by construction). Use 0 to disable. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace --out if it already exists.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[dict]:
    with path.open("r") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def episode_parquet_path(bridge_dir: Path, episode_index: int, chunk_size: int) -> Path:
    chunk = episode_index // chunk_size
    return bridge_dir / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"


def _column_to_2d(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == object:
        return np.asarray([np.asarray(row, dtype=np.float32) for row in arr])
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def euler_to_quat_xyzw(rpy: np.ndarray, convention: str) -> np.ndarray:
    """(N, 3) roll-pitch-yaw -> (N, 4) (qx, qy, qz, qw). scipy returns (x,y,z,w)."""
    if rpy.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    quat = R.from_euler(convention, rpy).as_quat()  # (N, 4) in (x, y, z, w)
    return quat.astype(np.float32)


def finite_diff_se3(pose_xyz_quat: np.ndarray) -> np.ndarray:
    """Approximate (vx, vy, vz, wx, wy, wz) from consecutive (x,y,z,qx,qy,qz,qw).

    Linear velocity: simple finite difference of xyz (per-step, unit ``frame``).
    Angular velocity: per-step axis-angle of relative rotation between frames.
    Step ``0`` is zero-filled.
    """
    n = int(pose_xyz_quat.shape[0])
    out = np.zeros((n, 6), dtype=np.float32)
    if n < 2:
        return out
    xyz = pose_xyz_quat[:, :3]
    quat = pose_xyz_quat[:, 3:]
    out[1:, 0:3] = xyz[1:] - xyz[:-1]
    rots = R.from_quat(quat)
    rel = rots[1:] * rots[:-1].inv()
    out[1:, 3:6] = rel.as_rotvec().astype(np.float32)
    return out


def select_episodes(bridge_dir: Path, task_filter: Optional[str], max_episodes: Optional[int]) -> tuple:
    info_path = bridge_dir / "meta/info.json"
    eps_path = bridge_dir / "meta/episodes.jsonl"
    if not info_path.exists() or not eps_path.exists():
        sys.exit(f"Bridge meta missing in {bridge_dir}/meta")
    info = json.loads(info_path.read_text())
    chunk_size = int(info.get("chunks_size", CHUNK_SIZE_DEFAULT))
    eps = load_jsonl(eps_path)
    if task_filter is not None:
        eps = [e for e in eps if task_filter in (e.get("tasks") or [])]
    eps.sort(key=lambda e: int(e["episode_index"]))
    if max_episodes is not None:
        eps = eps[:max_episodes]
    return info, chunk_size, eps


def open_destination(out_path: Path, overwrite: bool) -> zarr.Group:
    if out_path.exists():
        if not overwrite:
            sys.exit(f"{out_path} already exists; pass --overwrite to replace it.")
        shutil.rmtree(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(out_path), mode="w")
    return root


def _appendable(group: zarr.Group, key: str, shape_tail: tuple, dtype) -> zarr.Array:
    return group.create_dataset(
        key,
        shape=(0,) + shape_tail,
        chunks=(DEFAULT_CHUNK_ROWS,) + shape_tail,
        dtype=dtype,
        compressor=DEFAULT_COMPRESSOR,
    )


def build_destination(root: zarr.Group) -> dict:
    data = root.require_group("data")
    obs = data.require_group("obs")
    meta = root.require_group("meta")

    arrays = {
        "actions":                   _appendable(data, "actions", (7,),  np.float32),
        "dones":                     _appendable(data, "dones",   (),    np.bool_),
        "rewards":                   _appendable(data, "rewards", (),    np.float32),
        "arm_joint_pos":             _appendable(obs,  "arm_joint_pos", (6,), np.float32),
        "binary_contact":            _appendable(obs,  "binary_contact", (), np.float32),
        "end_effector_pose":         _appendable(obs,  "end_effector_pose", (7,), np.float32),
        "end_effector_vel_lin_ang_b":_appendable(obs,  "end_effector_vel_lin_ang_b", (6,), np.float32),
        "expert_action_mean":        _appendable(obs,  "expert_action_mean", (7,), np.float32),
        "expert_action_std":         _appendable(obs,  "expert_action_std", (7,), np.float32),
        "insertive_asset_in_receptive_asset_frame":
                                     _appendable(obs,  "insertive_asset_in_receptive_asset_frame", (7,), np.float32),
        "insertive_asset_pose":      _appendable(obs,  "insertive_asset_pose", (7,), np.float32),
        "joint_vel":                 _appendable(obs,  "joint_vel", (6,), np.float32),
        "last_arm_action":           _appendable(obs,  "last_arm_action", (6,), np.float32),
        "last_gripper_action":       _appendable(obs,  "last_gripper_action", (1,), np.float32),
        "receptive_asset_pose":      _appendable(obs,  "receptive_asset_pose", (7,), np.float32),
    }
    episode_ends = meta.create_dataset(
        "episode_ends",
        shape=(0,),
        chunks=(DEFAULT_CHUNK_ROWS,),
        dtype=np.int64,
        compressor=DEFAULT_COMPRESSOR,
    )
    return {"arrays": arrays, "episode_ends": episode_ends}


def episode_to_arrays(
    state: np.ndarray,
    action: np.ndarray,
    convention: str,
    success_reward: float,
) -> dict:
    n = int(state.shape[0])
    if n == 0:
        return {}
    xyz = state[:, 0:3].astype(np.float32)
    rpy = state[:, 3:6].astype(np.float32)
    grip = state[:, 6:7].astype(np.float32)
    quat = euler_to_quat_xyzw(rpy, convention)
    ee_pose = np.concatenate([xyz, quat], axis=1).astype(np.float32)

    grasp_idx = int(np.argmin(grip[:, 0]))
    insertive_pose = np.tile(ee_pose[grasp_idx], (n, 1)).astype(np.float32)
    receptive_pose = np.tile(ee_pose[-1], (n, 1)).astype(np.float32)
    relative = np.zeros_like(insertive_pose)
    relative[:, 0:3] = insertive_pose[:, 0:3] - receptive_pose[:, 0:3]
    relative[:, 3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    last_arm_action = np.zeros((n, 6), dtype=np.float32)
    if n >= 2:
        last_arm_action[1:] = action[:-1, :6].astype(np.float32)

    dones = np.zeros((n,), dtype=np.bool_)
    dones[-1] = True

    rewards = np.zeros((n,), dtype=np.float32)
    if success_reward != 0.0:
        rewards[-1] = float(success_reward)

    out = {
        "actions": action.astype(np.float32),
        "dones": dones,
        "rewards": rewards,
        "arm_joint_pos": np.zeros((n, 6), dtype=np.float32),
        "binary_contact": np.zeros((n,), dtype=np.float32),
        "end_effector_pose": ee_pose,
        "end_effector_vel_lin_ang_b": finite_diff_se3(ee_pose),
        "expert_action_mean": action.astype(np.float32),
        "expert_action_std": np.zeros((n, 7), dtype=np.float32),
        "insertive_asset_in_receptive_asset_frame": relative,
        "insertive_asset_pose": insertive_pose,
        "joint_vel": np.zeros((n, 6), dtype=np.float32),
        "last_arm_action": last_arm_action,
        "last_gripper_action": grip,
        "receptive_asset_pose": receptive_pose,
    }
    return out


def main() -> int:
    args = parse_args()
    info, chunk_size, episodes = select_episodes(
        args.bridge_dir, args.task_filter, args.max_episodes,
    )
    if not episodes:
        sys.exit("No episodes match the filter; nothing to write.")

    print(
        f"[load] {len(episodes)} episode(s) from {args.bridge_dir} "
        f"(task_filter={args.task_filter!r})"
    )

    root = open_destination(args.out, args.overwrite)
    plan = build_destination(root)
    arrays = plan["arrays"]
    episode_ends = plan["episode_ends"]

    total_steps = 0
    used = 0
    skipped = 0
    for i, ep in enumerate(episodes):
        ei = int(ep["episode_index"])
        path = episode_parquet_path(args.bridge_dir, ei, chunk_size)
        if not path.exists():
            skipped += 1
            continue
        table = pq.read_table(path, columns=["observation.state", "action"])
        state = _column_to_2d(table.column("observation.state").to_numpy(zero_copy_only=False)).astype(np.float32)
        action = _column_to_2d(table.column("action").to_numpy(zero_copy_only=False)).astype(np.float32)
        if state.shape[0] == 0:
            continue
        if state.shape[1] < 7 or action.shape[1] < 7:
            print(f"  skip ep {ei}: unexpected widths state={state.shape}, action={action.shape}")
            skipped += 1
            continue
        ep_arrs = episode_to_arrays(state, action, args.euler_convention, args.success_reward)
        for key, arr in ep_arrs.items():
            arrays[key].append(arr)
        total_steps += state.shape[0]
        episode_ends.append(np.asarray([total_steps], dtype=np.int64))
        used += 1
        if used % 50 == 0 or used == len(episodes):
            print(f"  {used}/{len(episodes)} episodes converted, total_steps={total_steps}")

    if used == 0:
        sys.exit("No episodes were converted (all missing parquet?).")

    root.attrs.update({
        "env_name": "bridge_v2_widowx_carrot_on_plate",
        "task_description": args.task_filter,
        "source": "jesbu1/bridge_v2_lerobot (filtered subset)",
        "source_dir": str(args.bridge_dir),
        "frame": "arm_base",
        "euler_convention": args.euler_convention,
        "n_episodes": int(used),
        "n_steps": int(total_steps),
        "skipped": int(skipped),
        "notes": (
            "Real-world demonstrations imported from Bridge V2 LeRobot. "
            "insertive_asset_pose / receptive_asset_pose are GRIPPER-EVENT PROXIES "
            "(carrot ~= EE pose at gripper-min step; plate ~= EE pose at final step), "
            "broadcast across all timesteps in the episode. "
            "arm_joint_pos / joint_vel / binary_contact are zero-filled because they are "
            "not part of the upstream Bridge V2 LeRobot schema. "
            "expert_action_std is zero (single demo per episode). "
            "rewards are zero except at the final step (set to --success_reward, default 1.0) "
            "since Bridge V2 demos are successful by construction."
        ),
    })

    print()
    print(f"Wrote {args.out}")
    print(
        f"  attrs.env_name = {root.attrs['env_name']}\n"
        f"  attrs.task_description = {root.attrs['task_description']!r}\n"
        f"  episodes = {used}, steps = {total_steps}, skipped = {skipped}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
