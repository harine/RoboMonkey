"""Project candidate action chunks onto the recorded camera frame and
color-code by verifier Q-value. The chosen candidate is drawn in red.

Input: an augmented bon_q .npz file produced by augment_bon_q.py (or by a
future eval_diffusion.py run that records EE pose + camera params per branch).

Required fields:
  candidate_actions      (R, K, T, 7)     raw 7D actions (xyz delta, rpy, grip)
  per_candidate_mean_reward (R, K)        verifier Q-value (mean over scored act.)
  selected_index         (R,)             chosen candidate at each branch
  selected_reward        (R,)             chosen Q
  frames                 (R, H, W, 3)     the verifier-scored RGB frame
  tcp_world_p            (R, 3)           EE position at the branch (world)
  cam_intrinsic          (3, 3)
  cam_extrinsic          (R, 3, 4)        world->camera (OpenCV convention)
  bon_replan_every_n_steps (scalar int)
  branch_t               (R,)

Output: one PNG per branch (or per *sampled* branch), and a contact-sheet PDF.

Conventions
-----------
For viz purposes we treat ``action[:3]`` as a *world-frame* xyz delta (the
diffusion_policy training data records actions in the robot base frame; for
widowx-on-bridge the base_link is fixed in world, so base ~ world). We
integrate ``cumsum`` of the K candidate deltas starting from ``tcp_world_p``
to get a 3D trajectory per candidate, then project via the recorded camera
intrinsic + extrinsic.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# --------------------------------------------------------------------------- #
# Projection helpers
# --------------------------------------------------------------------------- #

def integrate_chunks(start_p: np.ndarray, candidate_actions: np.ndarray) -> np.ndarray:
    """Cumsum the xyz deltas to produce 3D waypoints.

    start_p: (3,)
    candidate_actions: (K, T, 7)
    returns: (K, T+1, 3) -- the start point repeated once at index 0.
    """
    deltas = candidate_actions[..., :3].astype(np.float64)           # (K, T, 3)
    # The policy emits xyz deltas in the WidowX base frame, which is yaw-flipped
    # (180 deg about z) relative to the world/camera frame used for projection.
    # Empirically (scriptsv2/bon/check_action_frame.py) the mapping is
    # (x, y, z) -> (-x, -y, z); without this the drawn branches point opposite
    # to the robot's actual motion in x/y. Scale is left as-is (the env doesn't
    # execute the full commanded delta during contact, so no constant gain).
    deltas = deltas * np.array([-1.0, -1.0, 1.0])
    K, T, _ = deltas.shape
    waypoints = np.zeros((K, T + 1, 3), dtype=np.float64)
    waypoints[:, 0, :] = start_p[None, :]
    waypoints[:, 1:, :] = start_p[None, None, :] + np.cumsum(deltas, axis=1)
    return waypoints


def project_world_points(
    points_w: np.ndarray,           # (..., 3)
    intrinsic: np.ndarray,          # (3, 3)
    extrinsic: np.ndarray,          # (3, 4)  world->cam, CV convention
) -> tuple[np.ndarray, np.ndarray]:
    """Return (uv (..., 2), z_cam (...,)) for the projected points.

    z_cam <= 0 means the point is behind / on the image plane; the viz code
    should mask those out.
    """
    P = np.asarray(points_w, dtype=np.float64)
    shape = P.shape[:-1]
    P_flat = P.reshape(-1, 3)
    R = extrinsic[:, :3]
    t = extrinsic[:, 3]
    P_cam = P_flat @ R.T + t              # (N, 3)
    z = P_cam[:, 2]
    safe_z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    uv_h = P_cam @ intrinsic.T            # (N, 3)
    uv = uv_h[:, :2] / safe_z[:, None]
    return uv.reshape(*shape, 2), z.reshape(shape)


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #

# A modest qualitative-but-monotonic palette: viridis-like (low->purple, high->yellow).
# We pre-bake 256 RGB triples so we don't require matplotlib at runtime.
def _viridis_lut() -> np.ndarray:
    """8-stop viridis approximation -> 256x3 uint8 LUT."""
    stops = np.array([
        [ 68,  1,  84],
        [ 72, 35, 116],
        [ 64, 67, 135],
        [ 52, 94, 141],
        [ 41,120, 142],
        [ 32,144, 140],
        [ 34,167, 132],
        [ 68,190, 112],
        [121,209,  81],
        [189,222,  38],
        [253,231,  37],
    ], dtype=np.float32)
    xs = np.linspace(0.0, 1.0, stops.shape[0])
    out = np.zeros((256, 3), dtype=np.uint8)
    for c in range(3):
        out[:, c] = np.clip(np.interp(np.linspace(0,1,256), xs, stops[:, c]), 0, 255).astype(np.uint8)
    return out


_VIRIDIS = _viridis_lut()


def q_to_rgb(q: float, qmin: float, qmax: float) -> tuple[int, int, int]:
    if qmax <= qmin:
        idx = 128
    else:
        idx = int(np.clip(round(255 * (q - qmin) / (qmax - qmin)), 0, 255))
    r, g, b = _VIRIDIS[idx]
    return int(r), int(g), int(b)


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    for path in (
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_branch(
    frame: np.ndarray,            # (H, W, 3) uint8
    candidate_actions: np.ndarray,  # (K, T, 7)
    q_values: np.ndarray,          # (K,)
    selected_index: int,
    start_p_world: np.ndarray,    # (3,)
    intrinsic: np.ndarray,        # (3, 3)
    extrinsic: np.ndarray,        # (3, 4)
    title: str = "",
) -> Image.Image:
    img = Image.fromarray(frame.copy()).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    waypoints = integrate_chunks(start_p_world, candidate_actions)  # (K, T+1, 3)
    uv, z = project_world_points(waypoints, intrinsic, extrinsic)   # (K, T+1, 2), (K, T+1)

    qmin = float(q_values.min())
    qmax = float(q_values.max())

    K = candidate_actions.shape[0]
    order = np.argsort(q_values)  # draw low-Q first so high-Q is on top
    # ensure chosen is drawn LAST
    order = [i for i in order if i != selected_index] + [selected_index]

    H, W = frame.shape[:2]
    for k in order:
        is_chosen = (k == selected_index)
        color = (255, 30, 30) if is_chosen else q_to_rgb(float(q_values[k]), qmin, qmax)
        line_w = 4 if is_chosen else 2
        pts = uv[k]
        zs = z[k]
        # Mask out points behind the camera
        valid = zs > 0.01
        # connect successive valid pairs
        prev = None
        for i in range(pts.shape[0]):
            if not valid[i]:
                prev = None
                continue
            x, y = float(pts[i, 0]), float(pts[i, 1])
            if not (-50 <= x <= W + 50 and -50 <= y <= H + 50):
                prev = None
                continue
            r = 4 if is_chosen else 3
            fill = color + ((255,) if is_chosen else (220,))
            draw.ellipse([x - r, y - r, x + r, y + r], fill=fill)
            if prev is not None:
                px, py = prev
                draw.line([px, py, x, y], fill=fill, width=line_w)
            prev = (x, y)

        # Q label at the chunk endpoint
        if pts.shape[0] > 0 and valid[-1]:
            x, y = float(pts[-1, 0]), float(pts[-1, 1])
            text = f"{float(q_values[k]):+.2f}"
            font = _get_font(14 if is_chosen else 11)
            bbox = draw.textbbox((x + 6, y - 6), text, font=font)
            bg = (200, 0, 0, 230) if is_chosen else (40, 40, 40, 200)
            draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=bg)
            draw.text((x + 6, y - 6), text, fill=(255, 255, 255, 255), font=font)

    # Header
    if title:
        font = _get_font(14)
        bbox = draw.textbbox((6, 6), title, font=font)
        draw.rectangle([2, 2, bbox[2] + 6, bbox[3] + 6], fill=(0, 0, 0, 200))
        draw.text((6, 6), title, fill=(255, 255, 255, 255), font=font)

    # Q-range legend
    font_s = _get_font(10)
    legend_h = 10
    legend_w = 120
    lx, ly = 6, frame.shape[0] - legend_h - 22
    grad = np.zeros((legend_h, legend_w, 3), dtype=np.uint8)
    for i in range(legend_w):
        grad[:, i] = _VIRIDIS[int(255 * i / max(1, legend_w - 1))]
    img.paste(Image.fromarray(grad), (lx, ly))
    draw.text((lx, ly + legend_h + 1), f"Q  {qmin:+.2f}", fill=(255, 255, 255), font=font_s)
    draw.text((lx + legend_w - 38, ly + legend_h + 1), f"{qmax:+.2f}", fill=(255, 255, 255), font=font_s)
    # chosen swatch
    sw_x = lx + legend_w + 12
    draw.rectangle([sw_x, ly, sw_x + 14, ly + legend_h], fill=(255, 30, 30))
    draw.text((sw_x + 18, ly), "chosen", fill=(255, 255, 255), font=font_s)
    return img


# --------------------------------------------------------------------------- #
# Main per-file
# --------------------------------------------------------------------------- #

def viz_one(
    npz_path: Path,
    out_dir: Path,
    every: int = 1,
    max_branches: int = 0,
    video: bool = False,
    fps: float = 2.0,
) -> dict:
    with np.load(npz_path, allow_pickle=False) as z:
        if "tcp_world_p" not in z.files:
            raise RuntimeError(
                f"{npz_path}: missing 'tcp_world_p' -- run augment_bon_q.py first."
            )
        seed = int(z["seed"])
        success = bool(int(z["success"]))
        truncated = bool(int(z["truncated"]))
        branch_t = z["branch_t"].astype(np.int32)
        cand = z["candidate_actions"].astype(np.float32)         # (R, K, T, 7)
        Q = z["per_candidate_mean_reward"].astype(np.float32)    # (R, K)
        sel = z["selected_index"].astype(np.int32)               # (R,)
        frames = z["frames"]                                     # (R, H, W, 3)
        tcp_p = z["tcp_world_p"].astype(np.float32)              # (R, 3)
        K_int = z["cam_intrinsic"].astype(np.float32)
        K_ext = z["cam_extrinsic"].astype(np.float32)            # (R, 3, 4)

    out_dir.mkdir(parents=True, exist_ok=True)
    status = "success" if success else ("truncated" if truncated else "fail")
    stem = npz_path.stem.replace("_aug", "")

    R = min(cand.shape[0], tcp_p.shape[0], frames.shape[0])
    indices = list(range(0, R, max(1, every)))
    if max_branches > 0:
        indices = indices[:max_branches]

    writer = None
    video_path: Path | None = None
    if video:
        import imageio.v2 as imageio
        video_path = out_dir / f"{stem}.mp4"
        writer = imageio.get_writer(
            video_path, fps=fps, codec="libx264",
            quality=8, macro_block_size=1,
        )

    n_frames = 0
    try:
        for r in indices:
            title = (
                f"[{status.upper()}]  {stem}  t={int(branch_t[r])}  "
                f"branch={r}/{R-1}  K={cand.shape[1]}  chosen={int(sel[r])}"
            )
            img = draw_branch(
                frame=frames[r],
                candidate_actions=cand[r],
                q_values=Q[r],
                selected_index=int(sel[r]),
                start_p_world=tcp_p[r],
                intrinsic=K_int,
                extrinsic=K_ext[r] if K_ext.ndim == 3 else K_ext,
                title=title,
            )
            if writer is not None:
                arr = np.asarray(img)
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                if arr.shape[-1] == 4:
                    arr = arr[..., :3]
                writer.append_data(arr)
            else:
                outp = out_dir / f"{stem}_b{r:03d}_t{int(branch_t[r]):03d}.png"
                img.save(outp)
            n_frames += 1
    finally:
        if writer is not None:
            writer.close()

    if video_path is not None:
        print(f"[viz] {npz_path.name} -> {video_path} ({n_frames} frames @ {fps} fps)")
    else:
        print(f"[viz] {npz_path.name} -> {n_frames} PNGs in {out_dir}")

    return {"seed": seed, "branches_written": n_frames, "out_dir": str(out_dir)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", type=Path,
                   help=".npz files or directories containing them.")
    p.add_argument("--out-dir", type=Path,
                   default=Path("data/eval/bon/_summaries/bon_viz"))
    p.add_argument("--every", type=int, default=1,
                   help="Plot every Nth branch (1 = all).")
    p.add_argument("--max-branches", type=int, default=0,
                   help="Cap branches per episode (0 = all).")
    p.add_argument("--limit-files", type=int, default=0,
                   help="If >0, only process this many input files.")
    p.add_argument("--video", action="store_true",
                   help="Write a single MP4 per episode instead of one PNG "
                        "per branch (no PNGs are saved).")
    p.add_argument("--fps", type=float, default=2.0,
                   help="Frames per second for --video (default 2).")
    args = p.parse_args()

    files: List[Path] = []
    for inp in args.inputs:
        if inp.is_dir():
            # Prefer *_aug.npz files; fall back to ep*_seed*.npz.
            aug = sorted(inp.glob("*_aug.npz"))
            files.extend(aug if aug else sorted(inp.glob("ep*_seed*.npz")))
        else:
            files.append(inp)
    if args.limit_files > 0:
        files = files[: args.limit_files]
    if not files:
        print("no input .npz files", file=sys.stderr)
        sys.exit(2)

    for f in files:
        ep_out_dir = args.out_dir / f.stem.replace("_aug", "")
        try:
            viz_one(f, ep_out_dir, every=args.every,
                    max_branches=args.max_branches,
                    video=args.video, fps=args.fps)
        except Exception as e:
            print(f"[viz] FAILED {f.name}: {e!r}", file=sys.stderr)


if __name__ == "__main__":
    main()
