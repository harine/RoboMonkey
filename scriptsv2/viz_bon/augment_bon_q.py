"""Augment existing bon_q/*.npz files with per-branch EE pose + camera params.

For each (existing) bon_q episode npz file produced by eval_diffusion.py with
--viz-q, replay the env deterministically at the recorded seed -- executing the
already-chosen action chunk per branch (first ``bon_replan_every_n_steps``
actions of ``candidate_actions[r, selected_index[r]]``) -- and record at each
branch:

  * ``tcp_world_p``   (R, 3)  -- EE position in world frame
  * ``tcp_world_q``   (R, 4)  -- EE orientation quaternion (wxyz) in world frame
  * ``cam_intrinsic`` (3, 3)  -- pinhole intrinsics (3rd_view_camera)
  * ``cam_extrinsic`` (R, 3, 4)  -- world->cam (CV) at each branch (the camera
    is attached to base_link, which is fixed during the episode, so all rows
    are equal in practice -- but we record per-branch to be robust to any
    future env change)

The augmented npz is written next to the input as ``<name>_aug.npz`` (or
in-place if --inplace).

This does NOT require the verifier; it only steps the env. Requires the same
simpler_env / sapien runtime as eval_diffusion.py (use xvfb-run + the
simpler_env conda env).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

# Lazy-import sapien-using deps only inside main() so this module can be
# imported on machines without vulkan.


def _convert_maniskill(action: np.ndarray) -> np.ndarray:
    """Mirror of scriptsv2/eval_diffusion/eval_diffusion.py:convert_maniskill."""
    from transforms3d.euler import euler2axangle
    assert action.shape[0] == 7, f"expected 7D action, got {action.shape}"
    a = action.astype(np.float32, copy=True)
    roll, pitch, yaw = float(a[3]), float(a[4]), float(a[5])
    axis, angle = euler2axangle(roll, pitch, yaw)
    a[3:6] = np.asarray(axis, dtype=np.float32) * np.float32(angle)
    g = 2.0 * float(a[6]) - 1.0
    a[6] = 1.0 if g >= 0.0 else -1.0
    return a


def augment_one(npz_path: Path, task: str, out_path: Path) -> None:
    import simpler_env  # noqa: F401  registers envs
    from simpler_env import make as make_env

    with np.load(npz_path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}

    seed = int(data["seed"])
    replan_every_n = int(data["bon_replan_every_n_steps"])
    if replan_every_n <= 0:
        raise RuntimeError(
            f"{npz_path}: bon_replan_every_n_steps={replan_every_n}; "
            "augmenter assumes a fixed replan cadence."
        )

    branch_t: np.ndarray = data["branch_t"]                  # (R,)
    candidate_actions: np.ndarray = data["candidate_actions"]  # (R, K, T, 7)
    selected_index: np.ndarray = data["selected_index"]       # (R,)
    frames: np.ndarray = data["frames"]                       # (R, H, W, 3)
    R = int(branch_t.shape[0])

    print(f"[aug] {npz_path.name}  seed={seed} R={R} replan_every_n={replan_every_n}")

    env = make_env(task)
    obs, _info = env.reset(seed=seed)
    base = env.unwrapped if hasattr(env, "unwrapped") else env

    tcp_world_p = np.zeros((R, 3), dtype=np.float32)
    tcp_world_q = np.zeros((R, 4), dtype=np.float32)
    cam_extrinsic = np.zeros((R, 3, 4), dtype=np.float32)
    cam_intrinsic = None
    cam_name = "3rd_view_camera"

    t = 0
    for r in range(R):
        # Sanity check: env step counter should match the recorded branch_t[r].
        if int(branch_t[r]) != t:
            print(
                f"[aug] WARN {npz_path.name} r={r}: env t={t} != "
                f"recorded branch_t={int(branch_t[r])}; trusting env."
            )

        # Capture EE pose + camera params at this branch.
        tcp = base.tcp.pose
        tcp_world_p[r] = np.asarray(tcp.p, dtype=np.float32)
        tcp_world_q[r] = np.asarray(tcp.q, dtype=np.float32)
        cam_params = base.get_camera_params()
        if cam_name not in cam_params:
            # fall back to whatever the first camera is
            cam_name = next(iter(cam_params.keys()))
        cp = cam_params[cam_name]
        if cam_intrinsic is None:
            cam_intrinsic = np.asarray(cp["intrinsic_cv"], dtype=np.float32)
        ex = np.asarray(cp["extrinsic_cv"], dtype=np.float32)
        if ex.shape == (4, 4):
            ex = ex[:3, :]
        cam_extrinsic[r] = ex

        # Optional: quick frame sanity check vs the recorded frame.
        if r == 0 and frames.size:
            from simpler_env.utils.env.observation_utils import (
                get_image_from_maniskill2_obs_dict,
            )
            try:
                img = np.asarray(get_image_from_maniskill2_obs_dict(base, obs))
                if img.dtype != np.uint8:
                    img = np.clip(img * 255, 0, 255).astype(np.uint8)
                if img.shape == frames[0].shape:
                    diff = float(np.abs(img.astype(np.int32) - frames[0].astype(np.int32)).mean())
                    print(f"[aug]   frame[0] mean abs diff vs recorded: {diff:.2f}")
            except Exception as e:
                print(f"[aug]   frame sanity skipped: {e!r}")

        # Execute the chosen chunk's first `replan_every_n` actions.
        k_sel = int(selected_index[r])
        chunk = candidate_actions[r, k_sel, :replan_every_n, :]  # (n, 7)
        for raw in chunk:
            env_action = _convert_maniskill(np.asarray(raw, dtype=np.float32))
            obs, _rew, done, trunc, _info = env.step(env_action)
            t += 1
            if bool(done) or bool(trunc):
                break
        else:
            continue
        # break out of outer loop too if env ended
        if r + 1 < R:
            print(f"[aug]   env ended at t={t} before R={R}; truncating recorded fields.")
            tcp_world_p = tcp_world_p[: r + 1]
            tcp_world_q = tcp_world_q[: r + 1]
            cam_extrinsic = cam_extrinsic[: r + 1]
        break

    # Write augmented npz. We copy all original fields and add the new ones.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        tcp_world_p=tcp_world_p,
        tcp_world_q=tcp_world_q,
        cam_intrinsic=cam_intrinsic if cam_intrinsic is not None else np.zeros((3, 3), np.float32),
        cam_extrinsic=cam_extrinsic,
        cam_name=np.asarray(cam_name),
        **data,
    )
    print(f"[aug]   wrote -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", type=Path,
                   help="bon_q .npz file(s) or directory containing them")
    p.add_argument("--task", default="widowx_put_eggplant_in_basket")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="If set, write augmented files here (mirroring input "
                        "basenames). Default: alongside input as <stem>_aug.npz")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only process the first N inputs (sorted).")
    args = p.parse_args()

    files: List[Path] = []
    for inp in args.inputs:
        if inp.is_dir():
            files.extend(sorted(inp.glob("ep*_seed*.npz")))
        else:
            files.append(inp)
    files = [f for f in files if not f.stem.endswith("_aug")]
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        print("no input .npz files; nothing to do", file=sys.stderr)
        sys.exit(2)

    print(f"[aug] task={args.task}  inputs={len(files)}")
    for f in files:
        if args.out_dir is not None:
            out = args.out_dir / f.with_suffix(".npz").name
        else:
            out = f.with_name(f.stem + "_aug.npz")
        augment_one(f, args.task, out)


if __name__ == "__main__":
    main()
