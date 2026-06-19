"""Render an action+Q-overlay video from ``search_q/*.npz`` files written
by ``eval_search.py --viz-q``.

For each replan, project each of the K candidate action trajectories into
the scored image, draw a colored polyline (color = Q-rank, red = best,
blue = worst), and write the verifier Q-value as text at the trajectory
endpoint. One video per episode, one frame per replan. No sidebar.

Expected npz keys (written by the patched eval script):
    frames:           (R, H, W, 3) uint8
    candidate_actions:(R, K, T_a, 7) float32 — first 3 dims are Δxyz in the
                                                robot BASE frame (rotated to
                                                world via R_wb before drawing)
    values:           (R, K)        float32
    selected_index:   (R,)          int32
    selected_value:   (R,)          float32
    branch_t:         (R,)          int32
    ee_xyz:           (R, 3)        float32 — current EE world xyz per replan
    cam_K:            (3, 3)        float32 — intrinsic_cv
    cam_T_wc:         (4, 4)        float32 — extrinsic_cv (world -> cam)
    seed, ep_idx, success, truncated, num_steps, mode, max_actions: scalars
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np


def _color_for_rank(rank: int, k: int) -> tuple[int, int, int]:
    """rank 0 (worst Q) -> blue, rank k-1 (best Q) -> red. Returns BGR."""
    t = 0.0 if k <= 1 else rank / float(k - 1)
    r = int(255 * t)
    b = int(255 * (1 - t))
    g = int(160 * (1 - abs(2 * t - 1)))
    return (b, g, r)


def _project(world_xyz: np.ndarray, K: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
    """world_xyz: (..., 3) -> pixel (..., 2). Points behind cam (z<=0) -> NaN."""
    shape = world_xyz.shape[:-1]
    pts = world_xyz.reshape(-1, 3)
    R = T_wc[:3, :3]
    t = T_wc[:3, 3]
    cam = pts @ R.T + t
    z = cam[:, 2]
    safe = z > 1e-4
    uv = np.full((cam.shape[0], 2), np.nan, dtype=np.float32)
    uv[safe, 0] = (K[0, 0] * cam[safe, 0] + K[0, 2] * z[safe]) / z[safe]
    uv[safe, 1] = (K[1, 1] * cam[safe, 1] + K[1, 2] * z[safe]) / z[safe]
    return uv.reshape(*shape, 2)


def _quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    """sapien quat order [w, x, y, z] -> 3x3 rotation matrix."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


def _candidate_world_traj(
        ee_xyz: np.ndarray,
        actions: np.ndarray,
        R_wb: Optional[np.ndarray] = None,
    ) -> np.ndarray:
    """ee_xyz: (3,)   actions: (K, T_a, 7)   ->   (K, T_a + 1, 3) world xyz.

    Action xyz deltas are in the robot's BASE frame; ``R_wb`` rotates them
    into world frame. If R_wb is None, assumes the action frame == world
    frame (legacy / no robot-base info available).
    """
    deltas = actions[..., :3]                         # (K, T_a, 3)
    if R_wb is not None:
        # Apply R_wb to every delta vector.  einsum: ij, ...j -> ...i
        deltas = np.einsum("ij,...j->...i", R_wb, deltas)
    cum = np.cumsum(deltas, axis=1)                   # (K, T_a, 3)
    start = np.broadcast_to(ee_xyz, (cum.shape[0], 1, 3))
    traj = np.concatenate([start, start + cum], axis=1)  # (K, T_a+1, 3)
    return traj


def _draw_polyline(img, pts_xy: np.ndarray, color, thickness: int):
    """Draw a polyline through valid (non-NaN) pixel points."""
    import cv2
    H, W = img.shape[:2]
    pts = []
    for u, v in pts_xy:
        if np.isnan(u) or np.isnan(v):
            if len(pts) >= 2:
                cv2.polylines(img, [np.asarray(pts, dtype=np.int32)],
                              isClosed=False, color=color,
                              thickness=thickness, lineType=cv2.LINE_AA)
            pts = []
            continue
        ui, vi = int(round(u)), int(round(v))
        # clip so the line still gets drawn against the image edge
        ui = max(-10000, min(10000, ui))
        vi = max(-10000, min(10000, vi))
        pts.append((ui, vi))
    if len(pts) >= 2:
        cv2.polylines(img, [np.asarray(pts, dtype=np.int32)],
                      isClosed=False, color=color,
                      thickness=thickness, lineType=cv2.LINE_AA)


def _render_one_replan(
        frame: np.ndarray,
        cand: np.ndarray,           # (K, T_a, 7)
        values: np.ndarray,         # (K,)
        selected_index: int,
        ee_xyz: np.ndarray,         # (3,)
        K: np.ndarray,              # (3, 3)
        T_wc: np.ndarray,           # (4, 4)
        R_wb: Optional[np.ndarray] = None,  # (3, 3) world<-base rotation
        mode: str = "argmax",       # "argmax" -> chosen drawn in pure red
    ) -> np.ndarray:
    """Compose a single overlay frame."""
    import cv2

    img = frame.copy()
    Kc = int(values.shape[0])

    # rank order (ascending in value)
    order = np.argsort(values)
    rank_of = {int(k): int(r) for r, k in enumerate(order)}

    # project all candidate trajectories at once
    world_traj = _candidate_world_traj(ee_xyz, cand, R_wb=R_wb)   # (K, T_a+1, 3)
    px = _project(world_traj.reshape(-1, 3), K, T_wc).reshape(*world_traj.shape[:2], 2)

    # Draw order: every non-chosen candidate worst-to-best (so rank gradient
    # reads correctly), then the chosen one last on top.
    draw_order = [int(k) for k in order if int(k) != int(selected_index)]
    draw_order.append(int(selected_index))

    for k_idx in draw_order:
        is_chosen = (k_idx == int(selected_index))
        # Argmax mode: override the chosen-candidate color with pure red so
        # it stands out from the rank gradient (which already maps best -> red).
        if is_chosen and str(mode).lower() == "argmax":
            color = (0, 0, 255)   # pure BGR red
        else:
            color = _color_for_rank(rank_of[k_idx], Kc)
        thick = 2 if is_chosen else 1
        _draw_polyline(img, px[k_idx], color, thick)

        # End-point dot (small filled circle) at the last valid pixel
        last_valid = None
        for u, v in px[k_idx]:
            if not (np.isnan(u) or np.isnan(v)):
                last_valid = (int(round(u)), int(round(v)))
        if last_valid is None:
            continue
        cv2.circle(img, last_valid, 3 if is_chosen else 2, color, -1, cv2.LINE_AA)

        # Q value text at the endpoint
        q_str = f"{values[k_idx]:+.2f}"
        # outline (black) then fill (color) for readability over the image
        text_org = (last_valid[0] + 4, last_valid[1] - 2)
        cv2.putText(img, q_str, text_org, cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, q_str, text_org, cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, color, 1, cv2.LINE_AA)

    return img


def _write_mp4(frames: Iterable[np.ndarray], path: Path, fps: int) -> bool:
    arr = np.stack(list(frames), axis=0)
    if arr.size == 0:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
        with imageio.get_writer(str(path), fps=fps, codec="libx264",
                                quality=8, macro_block_size=1) as w:
            for f in arr:
                w.append_data(f)
        return True
    except Exception as e:
        try:
            import mediapy
            mediapy.write_video(str(path), arr, fps=fps)
            return True
        except Exception as e2:
            print(f"[render] failed to write {path.name}: imageio={e!r}; mediapy={e2!r}")
            return False


def render_npz(path: Path, out_dir: Path, fps: int) -> Optional[Path]:
    z = np.load(str(path), allow_pickle=False)
    frames = z["frames"]
    cand = z["candidate_actions"]
    vals = z["values"]
    sel_i = z["selected_index"]
    ep_idx = int(z["ep_idx"])
    seed = int(z["seed"])
    succ = bool(int(z["success"])) if "success" in z.files else False
    trunc = bool(int(z["truncated"])) if "truncated" in z.files else False
    status = "succ" if succ else ("trunc" if trunc else "fail")

    missing = [k for k in ("ee_xyz", "cam_K", "cam_T_wc") if k not in z.files]
    if missing:
        print(f"[render] {path.name}: missing keys {missing} — re-run eval "
              f"with the patched script. Skipping.")
        return None

    ee_xyz = z["ee_xyz"]
    K_mat = z["cam_K"]
    T_wc = z["cam_T_wc"]
    mode = str(z["mode"]) if "mode" in z.files else "argmax"
    # SimplerEnv widowx tasks mount the arm with a 180° yaw vs. the world
    # frame, so action xyz deltas (in robot BASE frame) need (-x, -y, +z).
    # If the eval saved an explicit base_pose_world we use that; otherwise
    # we fall back to the SimplerEnv-widowx default.
    if "base_pose_world" in z.files:
        bp = z["base_pose_world"]  # (7,) [px, py, pz, qw, qx, qy, qz]
        R_wb = _quat_wxyz_to_R(bp[3:7])
    else:
        R_wb = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)

    R = frames.shape[0]
    rendered: List[np.ndarray] = []
    for r in range(R):
        out_frame = _render_one_replan(
            frame=frames[r],
            cand=cand[r],
            values=vals[r],
            selected_index=int(sel_i[r]),
            ee_xyz=ee_xyz[r],
            K=K_mat,
            T_wc=T_wc,
            mode=mode,
            R_wb=R_wb,
        )
        rendered.append(out_frame)

    out_path = out_dir / f"ep{ep_idx:03d}_seed{seed}__N{int(vals.shape[1])}_{status}.mp4"
    ok = _write_mp4(rendered, out_path, fps=fps)
    return out_path if ok else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", help="search_q/ directory or a single .npz file")
    p.add_argument("--out-dir", default=None,
                   help="Output dir (default: <input>/videos)")
    p.add_argument("--fps", type=int, default=3,
                   help="Frames per second (one frame per replan, default 3)")
    p.add_argument("--limit", type=int, default=0,
                   help="Render at most this many episodes (0 = all)")
    args = p.parse_args()

    src = Path(args.input)
    if src.is_dir():
        npz_paths = sorted(src.glob("ep*.npz"))
        out_dir = Path(args.out_dir) if args.out_dir else src / "videos"
    elif src.is_file() and src.suffix == ".npz":
        npz_paths = [src]
        out_dir = Path(args.out_dir) if args.out_dir else src.parent / "videos"
    else:
        raise SystemExit(f"input not found or not .npz/dir: {src}")

    if args.limit > 0:
        npz_paths = npz_paths[: args.limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[render] {len(npz_paths)} episode(s) -> {out_dir}")
    ok_n = 0
    for npz in npz_paths:
        out = render_npz(npz, out_dir, fps=args.fps)
        if out is not None:
            ok_n += 1
            print(f"  {npz.name} -> {out.name}")
    print(f"[render] done: {ok_n}/{len(npz_paths)} videos written")


if __name__ == "__main__":
    main()
