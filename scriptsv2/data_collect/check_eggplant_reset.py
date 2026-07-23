#!/usr/bin/env python
"""Spread-check for the widened eggplant reset region (verification step 1-2).

Resets the wide eggplant task over many seeds, records the settled eggplant start
XY, and asserts the spawn region matches the intended reachable-core grid with the
basket footprint excluded. Optionally dumps a few reset frames + a scatter plot so
the placement can be eyeballed (rests flat in the sink, in view, not in the basket).

Needs a GPU compute node with Vulkan (the reset runs the settle sim). On Klone:
  conda activate simpler_env
  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  xvfb-run --auto-servernum -s "-screen 0 640x480x24" \
  python scriptsv2/data_collect/check_eggplant_reset.py --n 200 --out /tmp/eggplant_reset_check
"""
import argparse
import os

import numpy as np


# Intended reachable-core region + basket footprint (must match
# PutEggplantInBasketWideScene in put_on_in_scene.py).
REGION = dict(x_lo=-0.22, x_hi=-0.06, y_lo=0.116, y_hi=0.26)  # y_lo after basket-band exclusion
TARGET_XY = np.array([-0.125, 0.025])
BASKET_HALF = np.array([0.065, 0.04])  # dummy_sink_target_plane bbox half-extents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="widowx_put_eggplant_in_basket_wide")
    ap.add_argument("--n", type=int, default=200, help="number of reset seeds")
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="dir for scatter plot + reset frames")
    ap.add_argument("--frames", type=int, default=16, help="how many reset frames to save")
    args = ap.parse_args()

    import simpler_env

    env = simpler_env.make(args.task)
    print(f"[check] task={args.task}  instruction={env.get_language_instruction()!r}")

    xys = []
    frames = []
    for i in range(args.n):
        seed = args.start_seed + i
        obs, info = env.reset(seed=seed)
        src = getattr(env, "episode_source_obj", None)
        if src is None:
            raise SystemExit("env has no episode_source_obj; wrong task?")
        xys.append(np.asarray(src.pose.p[:2], dtype=np.float64))
        if args.out and len(frames) < args.frames:
            img = _try_get_image(env, obs)
            if img is not None:
                frames.append((seed, img))
    xys = np.array(xys)

    # --- stats ---
    print(f"[check] resets: {len(xys)}")
    print(f"[check] x range : [{xys[:,0].min():.3f}, {xys[:,0].max():.3f}]  "
          f"(intended [{REGION['x_lo']:.3f}, {REGION['x_hi']:.3f}])")
    print(f"[check] y range : [{xys[:,1].min():.3f}, {xys[:,1].max():.3f}]  "
          f"(intended [{REGION['y_lo']:.3f}, {REGION['y_hi']:.3f}])")
    uniq = {(round(x, 3), round(y, 3)) for x, y in xys}
    print(f"[check] unique settled XY (1mm): {len(uniq)}")

    # basket overlap: settled eggplant center inside the basket footprint
    in_basket = np.sum(
        (np.abs(xys[:, 0] - TARGET_XY[0]) < BASKET_HALF[0])
        & (np.abs(xys[:, 1] - TARGET_XY[1]) < BASKET_HALF[1])
    )
    print(f"[check] resets with eggplant center inside basket footprint: {in_basket}  (want 0)")

    # coverage margin: settle physics nudges things a bit, so allow a small slop
    slop = 0.02
    out_of_region = np.sum(
        (xys[:, 0] < REGION["x_lo"] - slop) | (xys[:, 0] > REGION["x_hi"] + slop)
        | (xys[:, 1] < REGION["y_lo"] - slop) | (xys[:, 1] > REGION["y_hi"] + slop)
    )
    print(f"[check] resets outside intended region (+/-{slop:.0f}cm slop): {out_of_region}")

    ok = (in_basket == 0) and (out_of_region == 0)
    print(f"[check] {'PASS' if ok else 'FAIL'}")

    # --- artifacts ---
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        _save_scatter(xys, os.path.join(args.out, "starts.png"))
        for seed, img in frames:
            _save_png(img, os.path.join(args.out, f"reset_seed{seed:04d}.png"))
        print(f"[check] wrote scatter + {len(frames)} reset frames to {args.out}")

    raise SystemExit(0 if ok else 1)


def _try_get_image(env, obs):
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "openvla-mini"))
        from experiments.robot.simpler.simpler_utils import get_simpler_img
        return np.asarray(get_simpler_img(env, obs, 480))
    except Exception:
        # fall back to digging the rgb out of the obs dict
        try:
            cam = obs["image"]
            key = next(iter(cam))
            return np.asarray(cam[key]["rgb"])
        except Exception:
            return None


def _save_scatter(xys, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 6))
        ax.scatter(xys[:, 0], xys[:, 1], s=8, alpha=0.5, label="eggplant start")
        # basket footprint
        bx, by = TARGET_XY - BASKET_HALF
        ax.add_patch(plt.Rectangle((bx, by), 2 * BASKET_HALF[0], 2 * BASKET_HALF[1],
                                   fill=False, ec="red", label="basket"))
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_aspect("equal"); ax.legend(); ax.set_title("wide eggplant reset spread")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[check] scatter skipped: {e}")


def _save_png(img, path):
    try:
        from PIL import Image
        Image.fromarray(np.asarray(img, dtype=np.uint8)).save(path)
    except Exception as e:
        print(f"[check] frame save skipped: {e}")


if __name__ == "__main__":
    main()
