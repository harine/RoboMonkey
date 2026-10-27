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


def _draw_dashed_polyline(img, pts_xy: np.ndarray, color, thickness: int,
                          dash: int = 7, gap: int = 5):
    """Dashed polyline through valid (non-NaN) pixel points (used for a*)."""
    import cv2
    valid = [(int(round(u)), int(round(v))) for u, v in pts_xy
             if not (np.isnan(u) or np.isnan(v))]
    for (x0, y0), (x1, y1) in zip(valid[:-1], valid[1:]):
        seg = float(np.hypot(x1 - x0, y1 - y0))
        if seg < 1e-3:
            continue
        n = max(1, int(seg // (dash + gap)))
        for i in range(n + 1):
            a = (i * (dash + gap)) / seg
            b = min(1.0, a + dash / seg)
            if a >= 1.0:
                break
            pa = (int(round(x0 + (x1 - x0) * a)), int(round(y0 + (y1 - y0) * a)))
            pb = (int(round(x0 + (x1 - x0) * b)), int(round(y0 + (y1 - y0) * b)))
            cv2.line(img, pa, pb, color, thickness, cv2.LINE_AA)


# Frames are RGB end-to-end (zarr agentview_image / get_image_from_maniskill2 are
# RGB, imageio writes RGB), and cv2 just writes the raw tuple into channels, so
# colors here are specified in RGB order. (_color_for_rank keeps its legacy
# tuple — the rank gradient and the candidate polylines use the *same* formula,
# so they stay mutually consistent regardless of channel naming; the legend bar
# below mirrors it and is labeled explicitly.)
_MEDAL_RGB = [(255, 215, 0), (192, 192, 192), (205, 127, 50)]   # gold, silver, bronze
_MEDAL_TAG = ["#1", "#2", "#3"]
_ASTAR_RGB = (255, 255, 255)   # a* drawn dashed white
_OK_RGB = (60, 200, 60)        # true / success
_BAD_RGB = (235, 80, 80)       # false / failure


def _rmse_to_astar(cand: np.ndarray, a_star: np.ndarray) -> np.ndarray:
    """Per-candidate RMSE of the action chunk vs a*. cand (K,T_a,7), a_star
    (T_a,7); a* may be NaN-padded past the episode end (those entries ignored).
    Returns (K,)."""
    diff = cand.astype(np.float64) - a_star.astype(np.float64)[None]   # (K,T_a,7)
    mask = ~np.isnan(a_star)[None]                                     # (1,T_a,7)
    se = np.where(mask, diff ** 2, np.nan)
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.nanmean(se.reshape(se.shape[0], -1), axis=1))


def _overlay_legend_box(scene: np.ndarray, *, seed, branch_t, status, n_cand,
                        mode, has_astar, a_star_rank=None, a_star_q=None,
                        top1_match=None, top3_match=None) -> np.ndarray:
    """Render the color key + per-frame metrics in a solid panel to the RIGHT of
    the scene (OUTSIDE the image) and hstack it on, so nothing is drawn over the
    frame. The diagnostic minRMSE-match rows (previously shown in red) are
    omitted. (top1_match / top3_match are accepted but unused.)"""
    import cv2
    H, W = scene.shape[:2]
    white = (235, 235, 235)
    grey = (170, 170, 170)
    st_color = {"succ": _OK_RGB, "fail": _BAD_RGB}.get(status, grey)

    # Row spec: ("text", text, color) | ("bar",) | ("medals",) | ("astar",)
    rows: List = [
        ("text", f"seed {seed}  t={branch_t}  {status.upper()}", st_color),
        ("text", f"N={n_cand}  {mode}", grey),
        ("bar",),
        ("text", "Q:  red=low  blue=high", grey),
        ("medals",),
    ]
    if has_astar:
        rk = "n/a" if a_star_rank is None else f"{int(a_star_rank)}/{n_cand}"
        qv = "n/a" if (a_star_q is None or np.isnan(a_star_q)) else f"{a_star_q:+.3f}"
        rows.append(("astar",))
        rows.append(("text", f"a* rank: {rk}", white))
        rows.append(("text", f"a* Q: {qv}", white))

    fs, dy, pad = 0.44, 20, 10
    panel_w = 232                       # even -> keeps libx264 happy after hstack
    panel = np.full((H, panel_w, 3), 24, dtype=np.uint8)   # solid dark side panel
    x = pad
    cy = pad
    bw = panel_w - 2 * pad
    for row in rows:
        kind = row[0]
        if kind == "bar":
            # Q-rank gradient (same formula as _color_for_rank -> matches polylines)
            for i in range(bw):
                t = i / max(1, bw - 1)
                col = (int(255 * (1 - t)), int(160 * (1 - abs(2 * t - 1))),
                       int(255 * t))
                cv2.line(panel, (x + i, cy + 4), (x + i, cy + 13), col, 1)
        elif kind == "medals":
            cx = x + 5
            for r in range(3):
                cv2.circle(panel, (cx, cy + 8), 5, _MEDAL_RGB[r], -1, cv2.LINE_AA)
                cx += 13
            cv2.putText(panel, "top-3 by Q", (cx + 3, cy + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, white, 1, cv2.LINE_AA)
        elif kind == "astar":
            _draw_dashed_polyline(panel, np.array([[x, cy + 8], [x + 16, cy + 8]],
                                                  dtype=np.float32), _ASTAR_RGB, 2)
            cv2.putText(panel, "a* (expert)", (x + 22, cy + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, white, 1, cv2.LINE_AA)
        else:
            _, text, color = row
            cv2.putText(panel, text, (x, cy + 13), cv2.FONT_HERSHEY_SIMPLEX,
                        fs, color, 1, cv2.LINE_AA)
        cy += dy
    return np.hstack([scene, panel])


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
        a_star: Optional[np.ndarray] = None,  # (T_a, 7) recorded chunk (offline)
    ) -> np.ndarray:
    """Compose a single overlay frame.

    Top-3 candidates by Q get a gold/silver/bronze ring + #1/#2/#3 tag at their
    endpoint. When ``a_star`` is given (offline train/val) it is drawn as a
    dashed white polyline labeled ``a*``.
    """
    import cv2

    img = frame.copy()
    Kc = int(values.shape[0])

    # rank order (ascending in value)
    order = np.argsort(values)
    rank_of = {int(k): int(r) for r, k in enumerate(order)}
    # top-3 by Q: order[-1] is best -> medal 0 (gold)
    top3 = [int(k) for k in order[::-1][:3]]
    medal_of = {k: r for r, k in enumerate(top3)}

    # project all candidate trajectories at once
    world_traj = _candidate_world_traj(ee_xyz, cand, R_wb=R_wb)   # (K, T_a+1, 3)
    px = _project(world_traj.reshape(-1, 3), K, T_wc).reshape(*world_traj.shape[:2], 2)

    # Draw order: every non-chosen candidate worst-to-best (so rank gradient
    # reads correctly), then the chosen one last on top.
    draw_order = [int(k) for k in order if int(k) != int(selected_index)]
    draw_order.append(int(selected_index))

    endpoints: dict[int, tuple[int, int]] = {}
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
        endpoints[k_idx] = last_valid
        cv2.circle(img, last_valid, 3 if is_chosen else 2, color, -1, cv2.LINE_AA)

        # Q value text at the endpoint
        q_str = f"{values[k_idx]:+.2f}"
        # outline (black) then fill (color) for readability over the image
        text_org = (last_valid[0] + 4, last_valid[1] - 2)
        cv2.putText(img, q_str, text_org, cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, q_str, text_org, cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, color, 1, cv2.LINE_AA)

    # Highlight the top-3 by Q with a medal ring + tag (drawn last, on top).
    for k_idx in top3[::-1]:                      # bronze first, gold last
        ep = endpoints.get(k_idx)
        if ep is None:
            continue
        mc = _MEDAL_RGB[medal_of[k_idx]]
        cv2.circle(img, ep, 6, mc, 2, cv2.LINE_AA)
        cv2.putText(img, _MEDAL_TAG[medal_of[k_idx]], (ep[0] - 6, ep[1] + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, _MEDAL_TAG[medal_of[k_idx]], (ep[0] - 6, ep[1] + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, mc, 1, cv2.LINE_AA)

    # a* overlay (offline only): dashed white polyline + label.
    if a_star is not None:
        a_traj = _candidate_world_traj(ee_xyz, a_star[None], R_wb=R_wb)  # (1,T_a+1,3)
        a_px = _project(a_traj.reshape(-1, 3), K, T_wc).reshape(-1, 2)
        _draw_dashed_polyline(img, a_px, _ASTAR_RGB, 2)
        a_last = None
        for u, v in a_px:
            if not (np.isnan(u) or np.isnan(v)):
                a_last = (int(round(u)), int(round(v)))
        if a_last is not None:
            cv2.putText(img, "a*", (a_last[0] + 4, a_last[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, "a*", (a_last[0] + 4, a_last[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, _ASTAR_RGB, 1, cv2.LINE_AA)

    return img


def _write_mp4(frames: Iterable[np.ndarray], path: Path, fps: int) -> bool:
    arr = np.stack(list(frames), axis=0)
    if arr.size == 0:
        return False
    # libx264 needs even H/W; the in-frame legend keeps the scene dims, so pad
    # by 1px if either is odd.
    h, w = arr.shape[1], arr.shape[2]
    if (h % 2) or (w % 2):
        arr = np.pad(arr, ((0, 0), (0, h % 2), (0, w % 2), (0, 0)), mode="constant")
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


def render_npz(path: Path, out_dir: Path, fps: int, scale: float = 0.0, action_steps: int = 0) -> Optional[Path]:
    z = np.load(str(path), allow_pickle=False)
    frames = z["frames"]
    cand = z["candidate_actions"]
    if action_steps:
        cand = cand[:, :, :action_steps]
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
    branch_t = z["branch_t"] if "branch_t" in z.files else np.arange(frames.shape[0])

    # Offline (train/val) extras: recorded a* chunk + its Q / rank on the Q scale.
    has_astar = "a_star" in z.files
    a_star = z["a_star"] if has_astar else None              # (R, T_a, 7)
    if a_star is not None and action_steps:
        a_star = a_star[:, :action_steps]
    a_star_q = z["a_star_q"] if "a_star_q" in z.files else None    # (R,)
    a_star_rank = z["a_star_rank"] if "a_star_rank" in z.files else None  # (R,)
    # SimplerEnv widowx tasks mount the arm with a 180° yaw vs. the world
    # frame, so action xyz deltas (in robot BASE frame) need (-x, -y, +z).
    # If the eval saved an explicit base_pose_world we use that; otherwise
    # we fall back to the SimplerEnv-widowx default.
    if "base_pose_world" in z.files:
        bp = z["base_pose_world"]  # (7,) [px, py, pz, qw, qx, qy, qz]
        R_wb = _quat_wxyz_to_R(bp[3:7])
    else:
        R_wb = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)

    # Upscale small (offline 224x224) frames so the candidate polylines + Q
    # labels aren't cramped. cam_K scales with the frame; ee_xyz / T_wc don't.
    fh, fw = int(frames.shape[1]), int(frames.shape[2])
    if scale <= 0:
        scale = 2.0 if max(fh, fw) < 300 else 1.0
    if scale != 1.0:
        import cv2
        nh, nw = int(round(fh * scale)), int(round(fw * scale))
        frames = np.stack([cv2.resize(f, (nw, nh), interpolation=cv2.INTER_LINEAR)
                           for f in frames])
        K_mat = K_mat.copy()
        K_mat[:2] *= scale

    R = frames.shape[0]
    draw_cand=cand.copy()
    verifier_scale=z['verifier_action_pos_scale'] if 'verifier_action_pos_scale' in z.files else np.ones(3)
    draw_cand[...,:3]*=verifier_scale
    draw_astar=None if a_star is None else a_star.copy()
    if draw_astar is not None:draw_astar[...,:3]*=verifier_scale
    rendered: List[np.ndarray] = []
    for r in range(R):
        a_star_r = a_star[r] if has_astar else None
        scene = _render_one_replan(
            frame=frames[r],
            cand=draw_cand[r],
            values=vals[r],
            selected_index=int(sel_i[r]),
            ee_xyz=ee_xyz[r],
            K=K_mat,
            T_wc=T_wc,
            mode=mode,
            R_wb=R_wb,
            a_star=draw_astar[r] if draw_astar is not None else None,
        )

        # Per-frame ranking-correctness metrics (only when a* is available).
        a_rank = top1_match = top3_match = None
        a_q = None
        if has_astar and a_star_r is not None and not np.all(np.isnan(a_star_r)):
            rmse_k = _rmse_to_astar(cand[r], a_star_r)        # (K,)
            closest = int(np.nanargmin(rmse_k))
            order_desc = np.argsort(vals[r])[::-1]
            top1 = int(order_desc[0])
            top3 = set(int(k) for k in order_desc[:3])
            top1_match = bool(top1 == closest)
            top3_match = bool(closest in top3)
            if a_star_q is not None:
                a_q = float(a_star_q[r])
            if a_star_rank is not None:
                a_rank = int(a_star_rank[r])
            elif a_star_q is not None and not np.isnan(a_q):
                a_rank = int((vals[r] > a_q).sum())

        out_frame = _overlay_legend_box(
            scene, seed=seed, branch_t=int(branch_t[r]), status=status,
            n_cand=int(vals.shape[1]), mode=mode, has_astar=has_astar,
            a_star_rank=a_rank, a_star_q=a_q,
            top1_match=top1_match, top3_match=top3_match,
        )
        # Keep every score legible even when projected endpoints overlap.
        panel = np.full((out_frame.shape[0], 460, 3), 24, dtype=np.uint8)
        rows = ['Estimated EE paths (not physics rollouts)',
                f'{cand.shape[2]} actions shown; higher verifier value is better',
                'ID      value      raw sum dx/dy/dz (m)     grip mean']
        for k in range(len(vals[r])):
            delta = cand[r, k, :, :3].sum(axis=0)
            grip = cand[r, k, :, -1].mean()
            mark = '*' if k == int(sel_i[r]) else ' '
            rows.append(f'{mark}{k:02d} {vals[r,k]:9.3f}  '
                        f'{delta[0]:+.3f}/{delta[1]:+.3f}/{delta[2]:+.3f}  {grip:+.2f}')
        import cv2
        spacing = min(20, (panel.shape[0] - 12) // max(len(rows), 1))
        for i, line in enumerate(rows):
            cv2.putText(panel, line, (8, 16 + spacing*i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.40, (235, 235, 235), 1, cv2.LINE_AA)
        out_frame = np.hstack([out_frame, panel])
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
    p.add_argument("--action-steps", type=int, default=0,
                   help="Show only the scored/executed prefix (0 = full prediction)")
    p.add_argument("--scale", type=float, default=0.0,
                   help="Frame upscale factor (0 = auto: 2x for frames <300px, "
                        "else 1x). cam_K scales with it so projection stays exact.")
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
        out = render_npz(npz, out_dir, fps=args.fps, scale=args.scale, action_steps=args.action_steps)
        if out is not None:
            ok_n += 1
            print(f"  {npz.name} -> {out.name}")
    print(f"[render] done: {ok_n}/{len(npz_paths)} videos written")


if __name__ == "__main__":
    main()
