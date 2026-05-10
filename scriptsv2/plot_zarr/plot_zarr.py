#!/usr/bin/env python3
"""Create a dashboard for collected RoboMonkey Zarr shards.

The collector stores episode boundaries in ``data/dones`` by forcing the final
step of every saved episode to True, so successful trajectories should usually
be inferred from rewards unless the collection code is changed to store an
explicit success flag.

Multiple shards can be passed to ``--shard`` and their data will be merged
before plotting so all episodes appear in a single combined figure.

Frame alignment
---------------
SIMPLER-collected shards live in the simulator world frame (x flipped vs.
WidowX arm-base, table-top z ~ 0.87). Bridge V2 shards produced by
``bridge_to_zarr.py`` live in the arm-base frame (table at z=0). Use
``--align-to-arm-base`` to negate the SIMPLER x and subtract its table-top z
median so SIMPLER + Bridge shards overlay correctly in one figure.

Usage:
    # single SIMPLER shard
    python scriptsv2/plot_zarr/plot_zarr.py --shard data/state0.zarr

    # merge multiple shards
    python scriptsv2/plot_zarr/plot_zarr.py \\
        --shard data/state0.zarr data/state1.zarr

    # overlay SIMPLER + Bridge V2 (auto-aligned)
    python scriptsv2/plot_zarr/plot_zarr.py \\
        --shard data/state0.zarr data/bridge_v2_carrot.zarr \\
        --align-to-arm-base
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List

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
        description="Plot dataset-level diagnostics for one or more RoboMonkey Zarr shards."
    )
    parser.add_argument(
        "--shard",
        type=Path,
        nargs="+",
        default=[Path("openvla-mini/data/carrot_on_plate/state0.zarr")],
        help="Path(s) to collected .zarr shard(s). Multiple shards are merged before plotting.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <first shard parent>/<first shard stem>_dashboard.png.",
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
        help="Optional text file for successful episode indices (single-shard only).",
    )
    parser.add_argument(
        "--success-shard-out",
        type=Path,
        default=None,
        help="Optional output .zarr shard containing only successful episodes (single-shard only).",
    )
    parser.add_argument(
        "--start-positions-out",
        type=Path,
        default=None,
        help=(
            "Output PNG for the per-episode starting positions plot. "
            "Defaults to <first shard parent>/<first shard stem>_start_positions.png."
        ),
    )
    parser.add_argument(
        "--scatters-only-out",
        type=Path,
        default=None,
        help=(
            "Output PNG for a clean 2x2 figure with ONLY the four start scatter "
            "panels (EE / insertive / receptive / offset). "
            "Defaults to <first shard parent>/<first shard stem>_scatters.png."
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
    parser.add_argument(
        "--align-to-arm-base",
        action="store_true",
        help=(
            "For each SIMPLER-frame shard (env_name not containing 'bridge_v2'): "
            "negate x and subtract the median asset-pose z so it overlays cleanly "
            "with Bridge V2 arm-base-frame shards. Bridge V2 shards are untouched."
        ),
    )
    parser.add_argument(
        "--frame-override",
        choices=["auto", "simpler", "arm_base", "none"],
        default="auto",
        help=(
            "Override frame detection for ALL shards when used with "
            "--align-to-arm-base. 'auto' (default) uses env_name heuristics; "
            "'simpler' forces SIMPLER (apply flip+offset); 'arm_base' forces "
            "Bridge-style (no change); 'none' disables alignment for all."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------


def episode_bounds(episode_ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.concatenate([[0], episode_ends[:-1]]).astype(np.int64)
    ends = episode_ends.astype(np.int64)
    return starts, ends


def classify_success(
    rewards: np.ndarray,
    dones: np.ndarray,
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


def per_episode_stat_np(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    episode_indices: np.ndarray,
    reducer,
) -> np.ndarray:
    """Like per_episode_stat but operates on a pre-loaded numpy array."""
    rows = []
    for ep in episode_indices:
        start, end = int(starts[ep]), int(ends[ep])
        if end <= start:
            rows.append(np.full(arr.shape[1:], np.nan, dtype=np.float32))
        else:
            rows.append(reducer(arr[start:end], axis=0))
    return np.asarray(rows)


# ---------------------------------------------------------------------------
# Single-shard loader  →  plain numpy dict
# ---------------------------------------------------------------------------


def _should_align(env_name: str, frame_override: str) -> bool:
    """Decide whether a shard needs the SIMPLER->arm-base alignment applied.

    SIMPLER shards have env_name like 'widowx_carrot_on_plate' (no 'bridge_v2'
    substring); Bridge V2 shards produced by bridge_to_zarr.py have env_name
    'bridge_v2_widowx_carrot_on_plate' and are already in arm-base frame.
    """
    if frame_override == "none":
        return False
    if frame_override == "simpler":
        return True
    if frame_override == "arm_base":
        return False
    return "bridge_v2" not in (env_name or "").lower()


def _simpler_z_offset(obs: zarr.Group, starts: np.ndarray, ends: np.ndarray) -> float:
    """Median z of insertive+receptive asset poses at episode starts. Used as a
    table-top reference so we can subtract it from EE/asset z to align with
    Bridge V2 (table at z=0)."""
    valid = ends > starts
    starts_v = starts[valid].astype(np.int64)
    if starts_v.size == 0:
        return 0.0
    src_z = np.asarray(obs["insertive_asset_pose"].get_orthogonal_selection((starts_v, 2)))
    tgt_z = np.asarray(obs["receptive_asset_pose"].get_orthogonal_selection((starts_v, 2)))
    z_all = np.concatenate([src_z, tgt_z])
    return float(np.median(z_all)) if z_all.size else 0.0


def load_shard_data(shard_path: Path, args: argparse.Namespace) -> dict:
    """Load a single zarr shard into a dict of numpy arrays."""
    root = zarr.open_group(str(shard_path), mode="r")
    data = root["data"]
    obs = data["obs"]

    starts, ends = episode_bounds(np.asarray(root["meta/episode_ends"][:]))

    rewards_full = np.asarray(data["rewards"][:])
    dones_full = np.asarray(data["dones"][:])

    success = classify_success(
        rewards_full, dones_full, starts, ends,
        args.success_mode, args.reward_threshold,
    )

    n_steps = int(rewards_full.shape[0])
    stride = max(1, int(np.ceil(n_steps / args.max_steps_for_stats)))
    actions_sub = np.asarray(data["actions"][::stride])
    joint_vel_sub = np.asarray(obs["joint_vel"][::stride])

    heatmap_eps = np.arange(len(ends))
    if len(heatmap_eps) > args.max_episodes_for_heatmap:
        heatmap_eps = np.linspace(
            0, len(ends) - 1, args.max_episodes_for_heatmap, dtype=np.int64
        )
    expert_std_full = np.asarray(obs["expert_action_std"][:])
    expert_std = per_episode_stat_np(
        expert_std_full, starts, ends, heatmap_eps, np.mean
    )

    episode_returns = per_episode_stat_np(
        rewards_full.reshape(-1, 1), starts, ends, np.arange(len(ends)), np.sum
    ).reshape(-1)

    valid = ends > starts
    starts_valid = starts[valid].astype(np.int64)

    ee_full = np.asarray(obs["end_effector_pose"][:])
    src_full = np.asarray(obs["insertive_asset_pose"][:])
    tgt_full = np.asarray(obs["receptive_asset_pose"][:])
    arm_full = np.asarray(obs["arm_joint_pos"][:])

    attrs = dict(root.attrs)
    align_applied = False
    z_off = 0.0
    if getattr(args, "align_to_arm_base", False) and _should_align(
        str(attrs.get("env_name", "")), getattr(args, "frame_override", "auto")
    ):
        z_off = _simpler_z_offset(obs, starts, ends)
        ee_full = ee_full.copy(); src_full = src_full.copy(); tgt_full = tgt_full.copy()
        for arr in (ee_full, src_full, tgt_full):
            arr[:, 0] *= -1.0
            arr[:, 2] -= z_off
        align_applied = True
        print(f"[align] {shard_path.name}: x-flip + z-offset={z_off:+.4f}")

    return {
        "attrs": attrs,
        "shard": shard_path,
        "align_applied": align_applied,
        "z_offset": z_off,
        "starts": starts,
        "ends": ends,
        "success": success,
        "valid": valid,
        "success_valid": success[valid],
        "rewards_full": rewards_full,
        "episode_returns": episode_returns,
        "lengths": ends - starts,
        "actions_sub": actions_sub,
        "joint_vel_sub": joint_vel_sub,
        "expert_std": expert_std,
        "ee_xyz": ee_full[starts_valid, :3],
        "src_xyz": src_full[starts_valid, :3],
        "tgt_xyz": tgt_full[starts_valid, :3],
        "arm_q0": arm_full[starts_valid],
    }


# ---------------------------------------------------------------------------
# Multi-shard merge
# ---------------------------------------------------------------------------


def merge_shard_data(shard_dicts: List[dict]) -> dict:
    """Concatenate multiple shard data dicts into one combined dict."""
    if len(shard_dicts) == 1:
        return shard_dicts[0]

    def cat(key, axis=0):
        arrays = [s[key] for s in shard_dicts if s[key].size]
        return np.concatenate(arrays, axis=axis) if arrays else shard_dicts[0][key]

    # Episode-level 1-D arrays
    success = cat("success")
    valid = cat("valid")
    success_valid = cat("success_valid")
    episode_returns = cat("episode_returns")
    lengths = cat("lengths")

    # Step-level arrays (subsampled independently per shard, then merged)
    rewards_full = cat("rewards_full")
    actions_sub = cat("actions_sub")
    joint_vel_sub = cat("joint_vel_sub")
    expert_std = cat("expert_std")

    # Starting-position arrays (one row per valid episode)
    ee_xyz = cat("ee_xyz")
    src_xyz = cat("src_xyz")
    tgt_xyz = cat("tgt_xyz")
    arm_q0 = cat("arm_q0")

    return {
        "attrs": shard_dicts[0]["attrs"],
        "shard": [s["shard"] for s in shard_dicts],
        "success": success,
        "valid": valid,
        "success_valid": success_valid,
        "rewards_full": rewards_full,
        "episode_returns": episode_returns,
        "lengths": lengths,
        "actions_sub": actions_sub,
        "joint_vel_sub": joint_vel_sub,
        "expert_std": expert_std,
        "ee_xyz": ee_xyz,
        "src_xyz": src_xyz,
        "tgt_xyz": tgt_xyz,
        "arm_q0": arm_q0,
    }


# ---------------------------------------------------------------------------
# Plot colours
# ---------------------------------------------------------------------------

FAIL_COLOR = "#404040"
SUCCESS_COLOR = "#54a24b"


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


def _shard_label(shard) -> str:
    if isinstance(shard, list):
        return " + ".join(str(Path(s).name) for s in shard)
    return str(Path(shard).name)


def _title_lines(data: dict, args: argparse.Namespace) -> list[str]:
    attrs = data["attrs"]
    success_valid = data["success_valid"]
    n_valid = int(len(success_valid))
    n_success = int(success_valid.sum())
    return [
        _shard_label(data["shard"]),
        f"env={attrs.get('env_name', 'unknown')}  task={attrs.get('task_description', 'unknown')}",
        (
            f"episodes(valid)={n_valid}  "
            f"success={n_success}/{n_valid} "
            f"({success_valid.mean() if n_valid else 0:.3f})  "
            f"mode={args.success_mode}"
        ),
    ]


# ---------------------------------------------------------------------------
# Plot functions  (operate on merged numpy data dict)
# ---------------------------------------------------------------------------


def make_scatters_only(
    data: dict,
    out_path: Path,
    args: argparse.Namespace,
) -> None:
    """2×2 scatter panels: EE / insertive / receptive / offset (all xy)."""
    attrs = data["attrs"]
    env_name = str(attrs.get("env_name", ""))
    auto_ins, auto_rcp = infer_asset_labels(env_name)
    insertive_lbl = args.insertive_label or auto_ins
    receptive_lbl = args.receptive_label or auto_rcp

    success_valid = data["success_valid"]
    fail = ~success_valid

    ee_xy  = data["ee_xyz"][:, :2]
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

    fig.suptitle("\n".join(_title_lines(data, args)), fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_starting_positions(
    data: dict,
    out_path: Path,
    args: argparse.Namespace,
) -> None:
    attrs = data["attrs"]
    env_name = str(attrs.get("env_name", ""))
    auto_ins, auto_rcp = infer_asset_labels(env_name)
    insertive_lbl = args.insertive_label or auto_ins
    receptive_lbl = args.receptive_label or auto_rcp

    success_valid = data["success_valid"]
    fail = ~success_valid

    ee_xy  = data["ee_xyz"][:, :2]
    src_xy = data["src_xyz"][:, :2]
    tgt_xy = data["tgt_xyz"][:, :2]
    rel_xy = src_xy - tgt_xy
    arm_q0 = data["arm_q0"]

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)

    _scatter_xy(
        fig.add_subplot(gs[0, 0]),
        ee_xy[fail], ee_xy[success_valid],
        "End-effector start (x,y)",
    )
    _scatter_xy(
        fig.add_subplot(gs[0, 1]),
        src_xy[fail], src_xy[success_valid],
        f"{insertive_lbl} start (x,y)",
    )
    _scatter_xy(
        fig.add_subplot(gs[0, 2]),
        tgt_xy[fail], tgt_xy[success_valid],
        f"{receptive_lbl} start (x,y)",
    )
    _scatter_xy(
        fig.add_subplot(gs[1, 0]),
        rel_xy[fail], rel_xy[success_valid],
        f"{insertive_lbl} - {receptive_lbl} offset (x,y)",
    )

    bins = 40
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(data["ee_xyz"][fail, 2], bins=bins, alpha=0.55, color=FAIL_COLOR, label="fail")
    ax.hist(data["ee_xyz"][success_valid, 2], bins=bins, alpha=0.75, color=SUCCESS_COLOR, label="success")
    ax.set_title("End-effector start z [m]")
    ax.set_xlabel("z")
    ax.set_ylabel("episodes")
    ax.legend(loc="best", fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    rel_dist = np.linalg.norm(rel_xy, axis=1)
    ax.hist(rel_dist[fail], bins=bins, alpha=0.55, color=FAIL_COLOR, label="fail")
    ax.hist(rel_dist[success_valid], bins=bins, alpha=0.75, color=SUCCESS_COLOR, label="success")
    ax.set_title(f"|{insertive_lbl} - {receptive_lbl}| start distance [m]")
    ax.set_xlabel("distance")
    ax.set_ylabel("episodes")
    ax.legend(loc="best", fontsize=8)

    ax = fig.add_subplot(gs[2, :])
    n_joints = arm_q0.shape[1]
    width = 0.35
    x = np.arange(n_joints)
    fail_mean = np.nanmean(arm_q0[fail], axis=0) if fail.any() else np.zeros(n_joints)
    fail_std  = np.nanstd(arm_q0[fail], axis=0)  if fail.any() else np.zeros(n_joints)
    succ_mean = np.nanmean(arm_q0[success_valid], axis=0) if success_valid.any() else np.zeros(n_joints)
    succ_std  = np.nanstd(arm_q0[success_valid], axis=0)  if success_valid.any() else np.zeros(n_joints)
    ax.bar(x - width / 2, fail_mean, width=width, yerr=fail_std,  capsize=3, color=FAIL_COLOR,    label="fail")
    ax.bar(x + width / 2, succ_mean, width=width, yerr=succ_std, capsize=3, color=SUCCESS_COLOR, label="success")
    ax.set_xticks(x)
    ax.set_xticklabels(JOINT_LABELS[:n_joints], rotation=20)
    ax.set_title("Initial arm joint positions (mean ± std)")
    ax.set_ylabel("rad")
    ax.legend(loc="best", fontsize=9)

    fig.suptitle("\n".join(_title_lines(data, args)), fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_dashboard(
    data: dict,
    out_path: Path,
    args: argparse.Namespace,
) -> None:
    success = data["success"]
    lengths = data["lengths"]
    rewards = data["rewards_full"]
    episode_returns = data["episode_returns"]
    actions = data["actions_sub"]
    joint_vel = data["joint_vel_sub"]
    expert_std = data["expert_std"]

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

    attrs = data["attrs"]
    n_eps = int(len(data["success"]))
    n_succ = int(data["success"].sum())
    fig.suptitle(
        "\n".join([
            _shard_label(data["shard"]),
            f"env={attrs.get('env_name', 'unknown')}  task={attrs.get('task_description', 'unknown')}",
            (
                f"episodes={n_eps}  "
                f"success={n_succ}/{n_eps} ({data['success'].mean() if n_eps else 0:.3f})"
            ),
        ]),
        fontsize=14,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Success-shard export  (single-shard only)
# ---------------------------------------------------------------------------


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
    dst.attrs["filtered_from"] = str(args.shard[0])
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    shard_paths: List[Path] = args.shard
    print(f"Loading {len(shard_paths)} shard(s)…")
    shard_dicts = []
    for sp in shard_paths:
        print(f"  {sp}")
        shard_dicts.append(load_shard_data(sp, args))

    data = merge_shard_data(shard_dicts)

    # ----- default output paths (keyed on first shard) -----
    first = shard_paths[0]
    stem = first.name.removesuffix(".zarr")

    out_path = args.out or (first.parent / f"{stem}_dashboard.png")
    start_path = args.start_positions_out or (first.parent / f"{stem}_start_positions.png")
    scatters_path = args.scatters_only_out or (first.parent / f"{stem}_scatters.png")

    make_dashboard(data, out_path, args)
    make_starting_positions(data, start_path, args)
    make_scatters_only(data, scatters_path, args)

    success_episode_indices = np.flatnonzero(data["success"])

    if args.success_indices_out is not None:
        if len(shard_paths) > 1:
            print("[warn] --success-indices-out ignored for multi-shard runs")
        else:
            args.success_indices_out.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(args.success_indices_out, success_episode_indices, fmt="%d")

    if args.success_shard_out is not None:
        if len(shard_paths) > 1:
            print("[warn] --success-shard-out ignored for multi-shard runs")
        else:
            root = zarr.open_group(str(shard_paths[0]), mode="r")
            sd = shard_dicts[0]
            create_success_shard(root, args.success_shard_out, success_episode_indices, sd["starts"], sd["ends"], args)

    n_succ = int(data["success"].sum())
    n_eps = int(len(data["success"]))
    print(f"dashboard:       {out_path}")
    print(f"start positions: {start_path}")
    print(f"scatters only:   {scatters_path}")
    print(
        f"episodes={n_eps}  success={n_succ} ({data['success'].mean() if n_eps else 0:.3f})  mode={args.success_mode}"
    )
    if args.success_indices_out is not None and len(shard_paths) == 1:
        print(f"success indices: {args.success_indices_out}")
    if args.success_shard_out is not None and len(shard_paths) == 1:
        print(f"success-only shard: {args.success_shard_out}")


if __name__ == "__main__":
    main()
