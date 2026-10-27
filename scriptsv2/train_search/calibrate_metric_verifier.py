"""Calibrate actor-frame placement offsets and 8-step motion using train50 only."""
import hashlib
import json
from pathlib import Path

import numpy as np
import zarr
from diffusion_policy.env.simpler.veritas.camera import CameraModel, quat_wxyz_to_R

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/gscratch/robotics/harine/RoboMonkey_data/experiments/metric_verifier_pilot_20261004')
TASKS = {'stack': 'widowx_stack_cube', 'eggplant': 'widowx_put_eggplant_in_basket'}


def calibrate(task):
    split_path = REPO / f'diffusion_policy/diffusion_policy/config/splits/simpler/{task}/random_train50.json'
    split = json.loads(split_path.read_text())
    z = zarr.open(split['zarr_path'], mode='r')
    ends = np.asarray(z['meta/episode_ends'])
    obs = z['data/obs']
    camera = CameraModel.load(REPO / f'diffusion_policy/diffusion_policy/config/veritas/camera_{TASKS[task]}.json')
    local, commands, measured = [], [], []
    for ep in split['train']:
        start, end = int(ends[ep-1]) if ep else 0, int(ends[ep])
        src = np.asarray(obs['insertive_asset_pose'][end-1])
        tgt = np.asarray(obs['receptive_asset_pose'][end-1])
        local.append(quat_wxyz_to_R(tgt[3:]).T @ (src[:3]-tgt[:3]))
        actions = np.asarray(z['data/actions'][start:end])
        tcp = np.asarray(obs['end_effector_pose'][start:end, :3])
        for t in range(end-start-8):
            commands.append(actions[t:t+8, :3].sum(axis=0))
            measured.append((tcp[t+8]-tcp[t]) @ camera.R_world_from_base)
    commands, measured = np.asarray(commands), np.asarray(measured)
    scale = (commands*measured).sum(axis=0)/(commands**2).sum(axis=0)
    return dict(version='metric_train50_calibration_v1', split=str(split_path),
                split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
                train_episodes=split['train'], horizon=8, windows=len(commands),
                goal_offset_target_frame=np.median(local, axis=0).tolist(),
                goal_offset_quantiles=np.quantile(local, [.1, .9], axis=0).tolist(),
                action_pos_scale=scale.tolist(),
                unscaled_rmse_m=float(np.sqrt(np.mean((commands-measured)**2))),
                fitted_rmse_m=float(np.sqrt(np.mean((commands*scale-measured)**2))),
                note='Successful training terminal actor poses; no evaluation states, rewards or contacts used.')


if __name__ == '__main__':
    for task in TASKS:
        path = ROOT / f'calibration_{task}.json'
        path.write_text(json.dumps(calibrate(task), indent=2))
        print(path)
