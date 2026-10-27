#!/usr/bin/env python3
"""Per-frame Veritas targets for a strict dataset: the tracker state at every recorded
frame of every planned episode, replayed from the zarr's end-effector path through the
fixed camera (``SimplerTrackerSnapshotStore``). Training attaches the result as the
``veritas_targets`` obs (``task.dataset.extra_obs_arrays``), so the ST's verifier sees
the same targets encoding at train and eval.

Also reports the expert chunk's value (-cost) statistics for ``value_mean``/``value_std``
and each episode's final tracker progress (does the demo actually traverse the plan?).

    python scriptsv2/veritas/build_veritas_targets.py --task widowx_put_eggplant_in_basket \\
        --split-file .../random_train400.json --plan-dir .../veritas_plans/eggplant/v1_upstream \\
        --horizon 4 [--action-pos-scale 0.72 0.74 0.87]
Writes <plan-dir>/veritas_targets.npy + .json (episode_ends_checksum, K, stats).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'diffusion_policy'))

from diffusion_policy.common.split_manifest import episode_ends_checksum  # noqa: E402
from diffusion_policy.common.veritas_calibration import training_window_starts  # noqa: E402
from diffusion_policy.env.simpler.veritas.camera import CameraModel  # noqa: E402
from diffusion_policy.env.simpler.veritas.tracker import (  # noqa: E402
    LOOKAHEAD_K, SimplerTrackerSnapshotStore)
from diffusion_policy.veritas.plan_store import has_plan, load_plan  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--split-file', required=True)
    ap.add_argument('--plan-dir', required=True)
    ap.add_argument('--camera-file', default=None)
    ap.add_argument('--require-splits', nargs='+', default=['train', 'val'],
                    help='splits every episode of which must have a plan')
    ap.add_argument('--horizon', type=int, default=4)
    ap.add_argument('--action-pos-scale', type=float, nargs=3, default=None)
    ap.add_argument('--lookahead-k', type=int, default=LOOKAHEAD_K)
    ap.add_argument('--lookahead-alpha', type=float, default=0.5)
    ap.add_argument('--stack-position', action='store_true', help='Calibrate the shared pixel + cube-alignment verifier')
    a = ap.parse_args()

    import zarr
    manifest = json.loads(Path(a.split_file).read_text())
    z = zarr.open(manifest['zarr_path'], 'r')
    ends = z['meta/episode_ends'][:]
    ee = np.asarray(z['data/obs/end_effector_pose'])[:, :3]
    actions = np.asarray(z['data/actions'])
    plan_dir = Path(a.plan_dir)
    cam_file = a.camera_file or (REPO / 'diffusion_policy/diffusion_policy/config/veritas'
                                 / f'camera_{a.task}.json')
    camera = CameraModel.load(cam_file)

    required = sorted({int(i) for s in a.require_splits for i in manifest[s]})
    missing = [i for i in required if not has_plan(plan_dir / f'ep{i}.json')]
    if missing:
        raise SystemExit(f'{len(missing)} required episode(s) have no plan in {plan_dir}: '
                         f'{missing[:20]}... Run plan_simpler_episodes.py first.')
    plans = {i: load_plan(plan_dir / f'ep{i}.json') for i in range(len(ends))
             if has_plan(plan_dir / f'ep{i}.json')}
    store = SimplerTrackerSnapshotStore(ee, ends, plans, camera, lookahead_k=a.lookahead_k)

    out_npy = plan_dir / 'veritas_targets.npy'

    # expert-chunk value stats through the real verifier, on frames whose H-step window
    # stays inside the episode
    from diffusion_policy.policy.veritas_verifier import VeritasPixelVerifier
    if a.stack_position:
        from diffusion_policy.policy.veritas_stack_position_verifier import VeritasStackPositionVerifier
        VeritasPixelVerifier = VeritasStackPositionVerifier
    ver = VeritasPixelVerifier(camera_file=str(cam_file), action_pos_scale=a.action_pos_scale,
                               lookahead_k=a.lookahead_k, lookahead_alpha=a.lookahead_alpha,
                               horizon=a.horizon, health_check=False)
    H = a.horizon
    ok = training_window_starts(ends, manifest['train'], plans, H)
    if len(ok):
        chunks = torch.tensor(np.stack([actions[i:i + H] for i in ok]), dtype=torch.float32)
        obs = {'end_effector_pose': torch.tensor(ee[ok], dtype=torch.float32),
               'veritas_targets': torch.tensor(store.targets[ok], dtype=torch.float32)}
        if a.stack_position:
            for key in ('insertive_asset_pose','receptive_asset_pose','last_gripper_action'):
                obs[key]=torch.tensor(np.asarray(z['data/obs/'+key])[ok],dtype=torch.float32)
        vals = ver.get_value(obs, chunks).numpy()
    else:
        raise ValueError('No complete training chunks available for Veritas calibration')
    if not np.isfinite(vals).all() or vals.std() <= 0:
        raise ValueError('Veritas training scores must be finite with positive variance')
    prog = np.array(list(store.final_progress.values()))
    meta = dict(task=a.task, split_file=str(a.split_file), plan_dir=str(plan_dir),
                camera_file=str(Path(cam_file).resolve()), zarr_path=manifest['zarr_path'],
                verifier_version='veritas_stack_position_v1' if a.stack_position else 'veritas_pixel',
                calibration_split='train',
                calibration_episodes=sorted(set(map(int, manifest['train']))),
                calibration_n_windows=len(ok), plan_done_mode='zero',
                episode_ends_checksum=episode_ends_checksum(ends), lookahead_k=a.lookahead_k,
                lookahead_alpha=a.lookahead_alpha, horizon=H,
                action_pos_scale=a.action_pos_scale, n_planned=len(plans),
                planned_episodes=sorted(plans),
                value_mean=float(vals.mean()) if len(vals) else None,
                value_std=float(vals.std()) if len(vals) else None,
                final_progress={str(k): float(v) for k, v in store.final_progress.items()},
                final_progress_mean=float(prog.mean()) if len(prog) else None,
                final_progress_done_frac=float((prog >= 1.0).mean()) if len(prog) else None)
    np.save(out_npy, store.targets.astype(np.float32))
    (plan_dir / 'veritas_targets.json').write_text(json.dumps(meta, indent=1))
    print(f'[targets] {store.targets.shape} -> {out_npy}; planned {len(plans)} eps; '
          f'expert value mean {meta["value_mean"]} std {meta["value_std"]}; '
          f'final progress mean {meta["final_progress_mean"]}, '
          f'done frac {meta["final_progress_done_frac"]}')


if __name__ == '__main__':
    main()
