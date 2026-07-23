#!/usr/bin/env python3
"""Render closed-loop rollout videos with the DEMO's expert trajectory overlaid.

Each frame shows, projected into the env camera:
  * the N candidate action chunks, colored by Q rank (red = best)   [render_q._color_for_rank]
  * the CHOSEN (argmax-Q) candidate, thick red
  * the demo's full expert EE path, dashed white  (static reference)
  * the policy's executed EE path so far, solid cyan
  * a legend: seed, replan t, Q(chosen), and the running divergence
    ||ee_policy(t) - ee_expert(t)||

The rollout was reset at the demo's own seed (see the seed gate), so the two paths start
from the same state and any separation is genuine policy-vs-expert divergence.

Reuses render_q.py verbatim for projection / candidate world-trajectories / drawing / mp4.

All videos for a task land in ONE folder, <task>/videos/, with the split encoded in the
filename (read from the npz's `demo_split`, written by augment_npz_with_expert.py):
    <task>/videos/<split>_seed<seed>_N<N>_<succ|trunc|fail>.mp4

  python scriptsv2/analysis/render_rollout_vs_expert.py \
      --dir data/eval/train_test_analysis/spoon/rollout/train \
      --out data/eval/train_test_analysis/spoon/videos [--fps 3]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scriptsv2" / "eval_search"))

from render_q import (  # type: ignore  # noqa: E402
    _candidate_world_traj, _color_for_rank, _draw_dashed_polyline, _draw_polyline,
    _project, _quat_wxyz_to_R, _write_mp4,
)

WIDOWX_R_WB = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)   # render_q's fallback


def _scale(img, K, factor):
    import cv2
    if factor <= 1.0:
        return img, K
    H, W = img.shape[:2]
    img = cv2.resize(img, (int(W * factor), int(H * factor)), interpolation=cv2.INTER_NEAREST)
    K = K.copy(); K[0, :] *= factor; K[1, :] *= factor
    return img, K


def render_one(npz: Path, out_dir: Path, fps: int, scale: float,
               tag: str = "") -> Path | None:
    import cv2
    d = np.load(npz, allow_pickle=True)
    if "expert_ee_world" not in d.files:
        print(f"  !! {npz.name}: no expert_ee_world (run augment_npz_with_expert.py first)")
        return None

    frames = d["frames"]                       # (R,H,W,3)
    cand = d["candidate_actions"]              # (R,K,T_a,7)
    vals = d["values"]                         # (R,K)
    sel = np.asarray(d["selected_index"])      # (R,)
    ee = d["ee_xyz"]                           # (R,3) policy EE, world
    bt = np.asarray(d["branch_t"])             # (R,) env step of each replan
    K_cam = np.asarray(d["cam_K"], np.float32)
    T_wc = np.asarray(d["cam_T_wc"], np.float32)
    exp_ee = np.asarray(d["expert_ee_world"], np.float32)   # (T,3)

    R_wb = WIDOWX_R_WB
    if "base_pose_world" in d.files:
        bp = np.asarray(d["base_pose_world"], np.float32)
        if bp.size >= 7:
            R_wb = _quat_wxyz_to_R(bp[3:7])

    # (R,K) bool: candidates that came from the random-augmentation pool
    # (scriptsv2/eval_search/random_action_aug.py). Absent on pre-augmentation
    # rollouts -> all-False, so those render exactly as before.
    is_rand = (np.asarray(d["candidate_is_random"], bool)
               if "candidate_is_random" in d.files
               else np.zeros(cand.shape[:2], bool))

    seed = int(d["seed"]); succ = int(d["success"]); trunc = int(d["truncated"])
    status = "succ" if succ else ("trunc" if trunc else "fail")
    # split lives in the npz (written by augment_npz_with_expert.py); it goes in the
    # FILENAME so every video for a task sits in one folder.
    split = str(d["demo_split"]) if "demo_split" in d.files else "unk"

    out_frames = []
    R, Kc = cand.shape[0], cand.shape[1]
    for r in range(R):
        img = np.ascontiguousarray(frames[r][..., ::-1])      # RGB -> BGR for cv2
        img, Kp = _scale(img, K_cam, scale)

        # expert path (static, dashed white) + policy path so far (solid cyan)
        _draw_dashed_polyline(img, _project(exp_ee, Kp, T_wc), (255, 255, 255), 1)
        if r > 0:
            _draw_polyline(img, _project(ee[: r + 1], Kp, T_wc), (255, 255, 0), 2)

        # candidate fan, colored by Q rank; chosen thick red
        trajs = _candidate_world_traj(ee[r], cand[r], R_wb)   # (K, T_a+1, 3)
        order = np.argsort(vals[r])                            # ascending Q
        rank_of = np.empty(Kc, int); rank_of[order] = np.arange(Kc)
        for k in range(Kc):
            if k == int(sel[r]):
                continue
            col = _color_for_rank(int(rank_of[k]), Kc)
            pts = _project(trajs[k], Kp, T_wc)
            # random-pool candidates dashed, policy candidates solid
            if bool(is_rand[r, k]):
                _draw_dashed_polyline(img, pts, col, 1)
            else:
                _draw_polyline(img, pts, col, 1)
        _draw_polyline(img, _project(trajs[int(sel[r])], Kp, T_wc), (0, 0, 255), 2)

        # divergence at this env step
        t = int(min(bt[r], len(exp_ee) - 1))
        div = float(np.linalg.norm(ee[r] - exp_ee[t]))

        n_rand = int(is_rand[r].sum())
        lines = [f"{split}  seed {seed}  {status}",
                 f"t={int(bt[r])}  N={Kc}" + (f" ({Kc - n_rand}p+{n_rand}r)" if n_rand else ""),
                 f"Q(chosen)={float(vals[r][int(sel[r])]):.2f}"
                 + ("  [RANDOM]" if n_rand and bool(is_rand[r, int(sel[r])]) else ""),
                 f"divergence={div*100:.1f} cm"]
        y = 14
        for ln in lines:
            cv2.putText(img, ln, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, ln, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (255, 255, 255), 1, cv2.LINE_AA)
            y += 13
        legend = ("-- expert   - chosen   . policy cand   .. random cand"
                  if n_rand else "-- expert   - chosen   . candidates")
        cv2.putText(img, legend, (6, img.shape[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, legend, (6, img.shape[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (230, 230, 230), 1, cv2.LINE_AA)

        out_frames.append(img[..., ::-1])                      # BGR -> RGB

    out_dir.mkdir(parents=True, exist_ok=True)
    suf = f"_{tag}" if tag else ""
    path = out_dir / f"{split}_seed{seed}_N{Kc}{suf}_{status}.mp4"
    return path if _write_mp4(out_frames, path, fps) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="rollout out_dir (contains search_q/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=3)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--tag", default="",
                    help="Suffix in the filename (e.g. k50) so several variants "
                         "of the same seed can share one --out folder.")
    a = ap.parse_args()

    sq = Path(a.dir) / "search_q"
    made = 0
    for p in sorted(sq.glob("*.npz")):
        r = render_one(p, Path(a.out), a.fps, a.scale, tag=str(a.tag))
        if r:
            made += 1
            print(f"  -> {r}")
    print(f"[render] {made} videos -> {a.out}")


if __name__ == "__main__":
    main()
