#!/usr/bin/env python3
"""Capture a SimplerEnv task's fixed camera model (K, world->camera extrinsic, frame size,
robot base pose) into a json the Veritas verifier and the training-time target builder
load without a simulator.

The 3rd-person camera is mounted on the robot base link and the base is placed at a
fixed per-task pose, so the model is a constant of the task; this asserts that over
``--n-seeds`` resets before writing it.

    MUJOCO_GL=osmesa python scriptsv2/veritas/capture_simpler_camera.py \\
        --task widowx_put_eggplant_in_basket
Writes diffusion_policy/diffusion_policy/config/veritas/camera_<task>.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'diffusion_policy'))

from diffusion_policy.env.simpler.veritas.camera import CameraModel  # noqa: E402

OUT_DIR = REPO / 'diffusion_policy' / 'diffusion_policy' / 'config' / 'veritas'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--n-seeds', type=int, default=20)
    ap.add_argument('--seed0', type=int, default=100000)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    import simpler_env  # noqa: F401
    from simpler_env import make as make_env
    env = make_env(a.task)
    ref = None
    for k in range(a.n_seeds):
        obs, _ = env.reset(seed=a.seed0 + k)
        cam = CameraModel.from_obs(obs)
        if ref is None:
            ref = cam
        else:
            ref.assert_matches(obs)
    env.close()
    out = Path(a.out) if a.out else OUT_DIR / f'camera_{a.task}.json'
    ref.save(out, task=a.task, n_seeds_checked=a.n_seeds, seed0=a.seed0)
    print(f'[camera] {a.task}: frame {ref.image_hw[1]}x{ref.image_hw[0]}, '
          f'K diag {np.diag(ref.K)[:2].round(2).tolist()}, base xyz '
          f'{ref.base_pose[:3].round(4).tolist()} -> {out}')


if __name__ == '__main__':
    main()
