#!/usr/bin/env python3
"""Create a dashboard for collected RoboMonkey Zarr shards.

The collector stores episode boundaries in ``data/dones`` by forcing the final
step of every saved episode to True, so successful trajectories should usually
be inferred from rewards unless the collection code is changed to store an
explicit success flag.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr


ACTION_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grip"]
JOINT_LABELS = ["waist", "shoulder", "elbow", "forearm", "wrist", "rotate"]

# Map from substrings found in `attrs["env_name"]` to ("insertive", "receptive")
# panel labels. Used so e.g. eggplant_in_basket renders 'Eggplant'/'Basket'
# instead of the hardcoded 'Carrot'/'Plate'.
ASSET_LABEL_MAP = [
    (("carrot", "plate"),         ("Carrot",   "Plate")),
    (("eggplant", "basket"),      ("Eggplant", "Basket")),
    (("spoon", "towel"),          ("Spoon",    "Towel")),
    (("stack_cube",),             ("Source cube", "Target cube")),
]


def infer_asset_labels(env_name: str) -> tuple[str, str]:
    name = (env_name or "").lower()
    for keys, labels in ASSET_LABEL_MAP:
        if all(k in name for k in keys):
            return labels
    return ("Insertive asset", "Receptive asset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot dataset-level diagnostics for a RoboMonkey Zarr shard."
    )
    parser.add_argument(
        "--shard",
        type=Path,
        default=Path("openvla-mini/data/carrot_on_plate/state0.zarr"),
        help="Path to a collected .zarr shard.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <shard parent>/<shard stem>_dashboard.png.",
    )
    parser.add_argument(
        "--success-mode",
        choices=["reward_any", "reward_sum", "done_last"],
        default="reward_any",
        help=(
            "How to classify successful episodes. Use reward_any for current "
            "collector output; done_last is an episode-boundary flag here."
        ),
    )
    parser.add_argument(
        "--reward-threshold",
        type=float,
        default=0.0,
        help="Reward threshold used by reward_any/reward_sum success modes.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=100,
        help="Episode window for success-rate smoothing.",
    )
    parser.add_argument(
        "--max-steps-for-stats",
        type=int,
        default=200_000,
        help="Maximum timesteps to load for global action/joint statistics.",
    )
    parser.add_argument(
        "--max-episodes-for-heatmap",
        type=int,
        default=2_000,
        help="Maximum episodes shown in the expert_action_std heatmap.",
    )
    parser.add_argument(
        "--success-indices-out",
        type=Path,
        default=None,
        help="Optional text file for successful episode indices.",
    )
    parser.add_argument(
        "--success-shard-out",
        type=Path,
        default=None,
        help="Optional output .zarr shard containing only successful episodes.",
    )
    parser.add_argument(
        "--start-positions-out",
        type=Path,
        default=None,
        help=(
            "Output PNG for the per-episode starting positions plot. "
            "Defaults to <shard parent>/<shard stem>_start_positions.png."
        ),
    )
    parser.add_argument(
        "--scatters-only-out",
        type=Path,
        default=None,
        help=(
            "Output PNG for a clean 2x2 figure with ONLY the four start scatter "
            "panels (EE / insertive / receptive / offset). "
            "Defaults to <shard parent>/<shard stem>_scatters.png."
        ),
    )
    parser.add_argument(
        "--insertive-label",
        type=str,
        default=None,
        help=(
            "Label for the insertive asset in the start-positions panels. "
            "If omitted, inferred from attrs['env_name'] (e.g. 'Carrot' / "
            "'Eggplant' / 'Spoon')."
        ),
    )
    parser.add_argument(
        "--receptive-label",
        type=str,
        default=None,
        help=(
            "Label for the receptive asset (e.g. 'Plate' / 'Basket' / 'Towel'). "
            "Inferred from env_name if omitted."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing --success-shard-out if it already exists.",
    )
    return parser.parse_args()


def episode_bounds(episode_ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.concatenate([[0], episode_ends[:-1]]).astype(np.int64)
    ends = episode_ends.astype(np.int64)
    return starts, ends


def classify_success(
    rewards: zarr.Array,
    dones: zarr.Array,
    starts: np.ndarray,
    ends: np.ndarray,
    mode: str,
    reward_threshold: float,
) -> np.ndarray:
    success = np.zeros(len(ends), dtype=bool)
    for i, (start, end) in enumerate(zip(starts, ends)):
        if end <= start:
            continue
        if mode == "done_last":
            success[i] = bool(dones[int(end - 1)])
        else:
            episode_rewards = np.asarray(rewards[int(start) : int(end)])
            if mode == "reward_any":
                success[i] = bool(np.any(episode_rewards > reward_threshold))
            elif mode == "reward_sum":
                success[i] = bool(np.sum(episode_rewards) > reward_threshold)
            else:
                raise ValueError(f"Unknown success mode: {mode}")
    return success


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values.astype(float)
    window = max(1, min(int(window), len(values)))
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values.astype(np.float64), kernel, mode="same")


def subsampled_array(arr: zarr.Array, max_items: int) -> np.ndarray:
    n = int(arr.shape[0])
    if n == 0:
        return np.asarray(arr[:])
    stride = max(1, int(np.ceil(n / max_items)))
    return np.asarray(arr[::stride])


def per_episode_stat(
    arr: zarr.Array,
    starts: np.ndarray,
    ends: np.ndarray,
    episode_indices: np.ndarray,
    reducer,
) -> np.ndarray:
    rows = []
    for ep in episode_indices:
        start, end = int(starts[ep]), int(ends[ep])
        if end <= start:
            rows.append(np.full(arr.shape[1:], np.nan, dtype=np.float32))
        else:
            rows.append(reducer(np.asarray(arr[start:end]), axis=0))
    return np.asarray(rows)


def collect_starting_positions(
    obs: zarr.Group, starts: np.ndarray, ends: np.ndarray
) -> dict[str, np.ndarray]:
    valid = ends > starts
    starts_valid = starts[valid].astype(np.int64)
    ee_xyz = np.asarray(obs["end_effector_pose"].get_orthogonal_selection((starts_valid, slice(0, 3))))
    src_xyz = np.asarray(obs["insertive_asset_pose"].get_orthogonal_selection((starts_valid, slice(0, 3))))
    tgt_xyz = np.asarray(obs["receptive_asset_pose"].get_orthogonal_selection((starts_valid, slice(0, 3))))
    arm_q0 = np.asarray(obs["arm_joint_pos"].get_orthogonal_selection((starts_valid, slice(None))))
    return {
        "valid": valid,
        "ee_xyz": ee_xyz,
        "src_xyz": src_xyz,
        "tgt_xyz": tgt_xyz,
        "arm_q0": arm_q0,
    }


FAIL_COLOR = "#404040"     # dark grey for clearer fail visibility
SUCCESS_COLOR = "#54a24b"  # green


def _scatter_xy(
    ax,
    xy_fail: np.ndarray,
    xy_success: np.ndarray,
    title: str,
) -> None:
    if xy_fail.size:
        ax.scatter(
            xy_fail[:, 0],
            xy_fail[:, 1],
            s=8,
            alpha=0.45,
            color=FAIL_COLOR,
            label=f"fail (n={len(xy_fail)})",
        )
    if xy_success.size:
        ax.scatter(
            xy_success[:, 0],
            xy_success[:, 1],
            s=14,
            alpha=0.85,
            color=SUCCESS_COLOR,
            edgecolors="black",
            linewidths=0.3,
            label=f"success (n={len(xy_success)})",
        )
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8)


def make_scatters_only(
    root: zarr.Group,
    out_path: Path,
    success: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    args: argparse.Namespace,
) -> None:
    """Render only the four start scatter panels (no histograms, no joint bars).

    Layout: 2x2 grid -- EE / insertive asset / receptive asset / offset (all xy).
    Asset labels follow attrs['env_name'] heuristics or --insertive-label /
    --receptive-label CLI overrides.
    """
    env_name = str(dict(root.attrs).get("env_name", ""))
    auto_ins, auto_rcp = infer_asset_labels(env_name)
    insertive_lbl = args.insertive_label or auto_ins
    receptive_lbl = args.receptive_label or auto_rcp

    obs = root["data/obs"]
    data = collect_starting_positions(obs, starts, ends)
    valid = data["valid"]
    success_valid = success[valid]
    fail = ~success_valid

    ee_xy = data["ee_xyz"][:, :2]
    src_xy = data["src_xyz"][:, :2]
    tgt_xy = data["tgt_xyz"][:, :2]
    rel_xy = src_xy - tgt_xy

    fig, axes = plt.subplots(2, 2, figsize=(12, 11), constrained_layout=True)
    _scatter_xy(axes[0, 0], ee_xy[fail],  ee_xy[success_valid],  "End-effector start (x,y)")
    _scatter_xy(axes[0, 1], src_xy[fail], src_xy[success_valid], f"{insertive_lbl} start (x,y)")
    _scatter_xy(axes[1, 0], tgt_xy[fail], tgt_xy[success_valid], f"{receptive_lbl} start (x,y)")
    _scatter_xy(axes[1, 1], rel_xy[fail], rel_xy[success_valid],
                f"{insertive_lbl} - {receptive_lbl} offset (x,y)")
    axes[1, 1].set_xlabel("dx [m]")
    axes[1, 1].set_ylabel("dy [m]")

    attrs = dict(root.attrs)
    fig.suptitle(
        "\n".join([
            f"{args.shard}",
            f"env={attrs.get('env_name', 'unknown')}  task={attrs.get('task_description', 'unknown')}",
            (
                f"episodes(valid)={int(valid.sum())}  "
                f"success={int(success_valid.sum())}/{int(valid.sum())} "
                f"({success_valid.mean() if valid.any() else 0:.3f})  "
                f"mode={args.success_mode}"
            ),
        ]),
        fontsize=12,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_starting_positions(
    root: zarr.Group,
    out_path: Path,
    success: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    args: argparse.Namespace,
) -> None:
    env_name = str(dict(root.attrs).get("env_name", ""))
    auto_ins, auto_rcp = infer_asset_labels(env_name)
    insertive_lbl = args.insertive_label or auto_ins
    receptive_lbl = args.receptive_label or auto_rcp
    obs = root["data/obs"]
    data = collect_starting_positions(obs, starts, ends)
    valid = data["valid"]
    success_valid = success[valid]
    fail = ~success_valid

    ee_xy = data["ee_xyz"][:, :2]
    src_xy = data["src_xyz"][:, :2]
    tgt_xy = data["tgt_xyz"][:, :2]
    rel_xy = src_xy - tgt_xy
    arm_q0 = data["arm_q0"]

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)

    _scatter_xy(
        fig.add_subplot(gs[0, 0]),
        ee_xy[fail],
        ee_xy[success_valid],
        "End-effector start (x,y)",
    )
    _scatter_xy(
        fig.add_subplot(gs[0, 1]),
        src_xy[fail],
        src_xy[success_valid],
        f"{insertive_lbl} start (x,y)",
    )
    _scatter_xy(
        fig.add_subplot(gs[0, 2]),
        tgt_xy[fail],
        tgt_xy[success_valid],
        f"{receptive_lbl} start (x,y)",
    )
    _scatter_xy(
        fig.add_subplot(gs[1, 0]),
        rel_xy[fail],
        rel_xy[success_valid],
        f"{insertive_lbl} - {receptive_lbl} offset (x,y)",
    )

    ax = fig.add_subplot(gs[1, 1])
    bins = 40
    ax.hist(
        data["ee_xyz"][fail, 2], bins=bins, alpha=0.55, color=FAIL_COLOR, label="fail"
    )
    ax.hist(
        data["ee_xyz"][success_valid, 2],
        bins=bins,
        alpha=0.75,
        color=SUCCESS_COLOR,
        label="success",
    )
    ax.set_title("End-effector start z [m]")
    ax.set_xlabel("z")
    ax.set_ylabel("episodes")
    ax.legend(loc="best", fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    rel_dist = np.linalg.norm(rel_xy, axis=1)
    ax.hist(rel_dist[fail], bins=bins, alpha=0.55, color=FAIL_COLOR, label="fail")
    ax.hist(
        rel_dist[success_valid],
        bins=bins,
        alpha=0.75,
        color=SUCCESS_COLOR,
        label="success",
    )
    ax.set_title(f"|{insertive_lbl} - {receptive_lbl}| start distance [m]")
    ax.set_xlabel("distance")
    ax.set_ylabel("episodes")
    ax.legend(loc="best", fontsize=8)

    ax = fig.add_subplot(gs[2, :])
    n_joints = arm_q0.shape[1]
    width = 0.35
    x = np.arange(n_joints)
    fail_mean = np.nanmean(arm_q0[fail], axis=0) if fail.any() else np.zeros(n_joints)
    fail_std = np.nanstd(arm_q0[fail], axis=0) if fail.any() else np.zeros(n_joints)
    succ_mean = (
        np.nanmean(arm_q0[success_valid], axis=0)
        if success_valid.any()
        else np.zeros(n_joints)
    )
    succ_std = (
        np.nanstd(arm_q0[success_valid], axis=0)
        if success_valid.any()
        else np.zeros(n_joints)
    )
    ax.bar(
        x - width / 2,
        fail_mean,
        width=width,
        yerr=fail_std,
        capsize=3,
        color=FAIL_COLOR,
        label="fail",
    )
    ax.bar(
        x + width / 2,
        succ_mean,
        width=width,
        yerr=succ_std,
        capsize=3,
        color=SUCCESS_COLOR,
        label="success",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(JOINT_LABELS[:n_joints], rotation=20)
    ax.set_title("Initial arm joint positions (mean ± std)")
    ax.set_ylabel("rad")
    ax.legend(loc="best", fontsize=9)

    attrs = dict(root.attrs)
    fig.suptitle(
        "\n".join(
            [
                f"{args.shard}",
                f"env={attrs.get('env_name', 'unknown')}  task={attrs.get('task_description', 'unknown')}",
                (
                    f"episodes(valid)={int(valid.sum())}  "
                    f"success={int(success_valid.sum())}/{int(valid.sum())} "
                    f"({success_valid.mean() if valid.any() else 0:.3f})  "
                    f"mode={args.success_mode}"
                ),
            ]
        ),
        fontsize=14,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_dashboard(
    root: zarr.Group,
    out_path: Path,
    success: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    args: argparse.Namespace,
) -> None:
    data = root["data"]
    obs = data["obs"]
    lengths = ends - starts
    rewards = np.asarray(data["rewards"][:])
    episode_returns = per_episode_stat(
        data["rewards"], starts, ends, np.arange(len(ends)), np.sum
    ).reshape(-1)
    actions = subsampled_array(data["actions"], args.max_steps_for_stats)
    joint_vel = subsampled_array(obs["joint_vel"], args.max_steps_for_stats)

    heatmap_eps = np.arange(len(ends))
    if len(heatmap_eps) > args.max_episodes_for_heatmap:
        heatmap_eps = np.linspace(
            0, len(ends) - 1, args.max_episodes_for_heatmap, dtype=np.int64
        )
    expert_std = per_episode_stat(
        obs["expert_action_std"], starts, ends, heatmap_eps, np.mean
    )

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)

    ax = fig.add_subplot(gs[0, 0])
    ax.hist(lengths, bins=40, color="#4c78a8", alpha=0.9)
    ax.set_title("Episode Lengths")
    ax.set_xlabel("steps")
    ax.set_ylabel("episodes")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(success.astype(float), ".", markersize=2, alpha=0.35, label="episode")
    ax.plot(
        rolling_mean(success, args.rolling_window),
        linewidth=2,
        label=f"rolling {args.rolling_window}",
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Success Over Time ({args.success_mode})")
    ax.set_xlabel("episode")
    ax.set_ylabel("success")
    ax.legend(loc="best")

    ax = fig.add_subplot(gs[0, 2])
    ax.hist(rewards, bins=60, color="#f58518", alpha=0.85)
    ax.set_title("Per-Step Reward Distribution")
    ax.set_xlabel("reward")
    ax.set_ylabel("steps")

    ax = fig.add_subplot(gs[1, 0])
    ax.hist(episode_returns, bins=60, color="#e45756", alpha=0.85)
    ax.set_title("Episode Return Distribution")
    ax.set_xlabel("sum reward")
    ax.set_ylabel("episodes")

    ax = fig.add_subplot(gs[1, 1])
    x = np.arange(actions.shape[1])
    ax.bar(x - 0.2, np.nanmean(actions, axis=0), width=0.4, label="mean")
    ax.bar(x + 0.2, np.nanstd(actions, axis=0), width=0.4, label="std")
    ax.set_xticks(x)
    ax.set_xticklabels(ACTION_LABELS[: actions.shape[1]], rotation=35)
    ax.set_title("Action Statistics")
    ax.legend(loc="best")

    ax = fig.add_subplot(gs[1, 2])
    x = np.arange(joint_vel.shape[1])
    ax.bar(x - 0.2, np.nanmean(joint_vel, axis=0), width=0.4, label="mean")
    ax.bar(x + 0.2, np.nanstd(joint_vel, axis=0), width=0.4, label="std")
    ax.set_xticks(x)
    ax.set_xticklabels(JOINT_LABELS[: joint_vel.shape[1]], rotation=35)
    ax.set_title("Joint Velocity Statistics")
    ax.legend(loc="best")

    ax = fig.add_subplot(gs[2, :])
    im = ax.imshow(expert_std.T, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_yticks(np.arange(min(expert_std.shape[1], len(ACTION_LABELS))))
    ax.set_yticklabels(ACTION_LABELS[: expert_std.shape[1]])
    ax.set_title("Mean expert_action_std Per Episode")
    ax.set_xlabel("episode sample")
    ax.set_ylabel("action dim")
    fig.colorbar(im, ax=ax, label="std")

    attrs = dict(root.attrs)
    fig.suptitle(
        "\n".join(
            [
                f"{args.shard}",
                f"env={attrs.get('env_name', 'unknown')}  task={attrs.get('task_description', 'unknown')}",
                (
                    f"episodes={len(ends)}  steps={int(ends[-1]) if len(ends) else 0}  "
                    f"success={int(success.sum())}/{len(success)} ({success.mean() if len(success) else 0:.3f})"
                ),
            ]
        ),
        fontsize=14,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def create_success_shard(
    src: zarr.Group,
    dst_path: Path,
    success_episode_indices: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    args: argparse.Namespace,
) -> None:
    if dst_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dst_path} exists; pass --overwrite to replace it")
        shutil.rmtree(dst_path)

    dst = zarr.open_group(str(dst_path), mode="w")
    dst.attrs.update(dict(src.attrs))
    dst.attrs["filtered_from"] = str(args.shard)
    dst.attrs["success_mode"] = args.success_mode
    dst.attrs["reward_threshold"] = args.reward_threshold

    dst_data = dst.require_group("data")
    dst_obs = dst_data.require_group("obs")
    dst_meta = dst.require_group("meta")

    def create_like(dst_group: zarr.Group, key: str, src_arr: zarr.Array) -> zarr.Array:
        return dst_group.create_dataset(
            key,
            shape=(0,) + src_arr.shape[1:],
            chunks=src_arr.chunks,
            dtype=src_arr.dtype,
            compressor=src_arr.compressor,
        )

    dst_arrays = {}
    for key, value in src["data"].items():
        if key == "obs":
            continue
        dst_arrays[("data", key)] = create_like(dst_data, key, value)
    for key, value in src["data/obs"].items():
        dst_arrays[("obs", key)] = create_like(dst_obs, key, value)

    episode_ends = dst_meta.create_dataset(
        "episode_ends",
        shape=(0,),
        chunks=src["meta/episode_ends"].chunks,
        dtype=src["meta/episode_ends"].dtype,
        compressor=src["meta/episode_ends"].compressor,
    )

    total_steps = 0
    for ep in success_episode_indices:
        start, end = int(starts[ep]), int(ends[ep])
        if end <= start:
            continue
        for key, value in src["data"].items():
            if key == "obs":
                continue
            dst_arrays[("data", key)].append(np.asarray(value[start:end]))
        for key, value in src["data/obs"].items():
            dst_arrays[("obs", key)].append(np.asarray(value[start:end]))
        total_steps += end - start
        episode_ends.append(np.asarray([total_steps], dtype=episode_ends.dtype))


def main() -> None:
    args = parse_args()
    root = zarr.open_group(str(args.shard), mode="r")
    starts, ends = episode_bounds(np.asarray(root["meta/episode_ends"][:]))
    success = classify_success(
        root["data/rewards"],
        root["data/dones"],
        starts,
        ends,
        args.success_mode,
        args.reward_threshold,
    )

    out_path = args.out
    if out_path is None:
        out_path = args.shard.parent / f"{args.shard.name.removesuffix('.zarr')}_dashboard.png"
    make_dashboard(root, out_path, success, starts, ends, args)

    start_path = args.start_positions_out
    if start_path is None:
        start_path = (
            args.shard.parent
            / f"{args.shard.name.removesuffix('.zarr')}_start_positions.png"
        )
    make_starting_positions(root, start_path, success, starts, ends, args)

    scatters_path = args.scatters_only_out
    if scatters_path is None:
        scatters_path = (
            args.shard.parent
            / f"{args.shard.name.removesuffix('.zarr')}_scatters.png"
        )
    make_scatters_only(root, scatters_path, success, starts, ends, args)

    success_episode_indices = np.flatnonzero(success)
    if args.success_indices_out is not None:
        args.success_indices_out.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(args.success_indices_out, success_episode_indices, fmt="%d")

    if args.success_shard_out is not None:
        create_success_shard(
            root, args.success_shard_out, success_episode_indices, starts, ends, args
        )

    print(f"dashboard: {out_path}")
    print(f"start positions: {start_path}")
    print(f"scatters only: {scatters_path}")
    print(
        f"episodes={len(ends)} steps={int(ends[-1]) if len(ends) else 0} "
        f"success={len(success_episode_indices)} ({success.mean() if len(success) else 0:.3f}) "
        f"mode={args.success_mode}"
    )
    if args.success_indices_out is not None:
        print(f"success indices: {args.success_indices_out}")
    if args.success_shard_out is not None:
        print(f"success-only shard: {args.success_shard_out}")


if __name__ == "__main__":
    main()
