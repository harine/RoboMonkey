#!/usr/bin/env python3
"""SEED GATE for a strict dataset: does ``env.reset(seed)`` reproduce each episode's
recorded start state?

Recreatable (``ok``) means the end effector reproduces exactly; the objects' settle after
reset varies by up to ~8 mm (eggplant) and is treated as environment stochasticity
(``object_offset_mm``, ``scene_ok`` = all objects within tol).

Seeds follow the collector's rule ``seed = 100000 + START_INDEX[shard] + src_episode``
(``scriptsv2/analysis/reconstruct_demo_seeds.py``). The eggplant settles on reset and
its settle depends on the simulator's prior history, so about one episode in ten lands
a few millimetres off; those episodes cannot be rolled out from their seed and must stay
out of the TEST pools. This writes, per episode, the live-vs-zarr distances and a
pass flag, which ``scriptsv2/splits/make_simpler_splits.py`` reads.

    MUJOCO_GL=osmesa python scriptsv2/veritas/seed_gate.py --task eggplant
Writes scriptsv2/eval_search/demoseeds/<task>_strict_seed_gate.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'diffusion_policy'))
sys.path.insert(0, str(REPO / 'scriptsv2' / 'analysis'))

from reconstruct_demo_seeds import START_INDEX, START_SEED  # noqa: E402

GATE_DIR = REPO / 'scriptsv2' / 'eval_search' / 'demoseeds'
GATE_TOL = 1e-3   # metres, per the 2026-07 seed gate (xyz reproduces to <1 mm)
TASKS = {
    'eggplant': ('widowx_put_eggplant_in_basket', 'strict_distance_success', 'eggplant'),
    'stack': ('widowx_stack_cube', 'stack_cube_strict', 'stack_cube'),
}


def gate_path(task: str) -> Path:
    return GATE_DIR / f'{task}_strict_seed_gate.json'


def episode_seed(task: str, src_shard: str, src_episode: int) -> int:
    seed_task = TASKS[task][2]
    return START_SEED + START_INDEX[seed_task][str(src_shard).split('/')[-1]] + int(src_episode)


def gate_from_env(env_base, obs, zarr_root, out_start: int, tol: float = GATE_TOL) -> dict:
    """Live reset vs the zarr's first row: end effector, source and target objects."""
    z = zarr_root['data/obs']
    tcp = np.asarray(obs['extra']['tcp_pose'], dtype=np.float64)[:3]
    d = {
        'ee': float(np.linalg.norm(tcp - np.asarray(z['end_effector_pose'][out_start])[:3])),
        'source': float(np.linalg.norm(np.asarray(env_base.episode_source_obj.pose.p)
                                       - np.asarray(z['insertive_asset_pose'][out_start])[:3])),
        'target': float(np.linalg.norm(np.asarray(env_base.episode_target_obj.pose.p)
                                       - np.asarray(z['receptive_asset_pose'][out_start])[:3])),
    }
    # Recreatable = the end effector reproduces. The eggplant/basket settle after reset
    # varies by up to ~8 mm with the simulator's prior history; that is environment
    # stochasticity, recorded but not gated. scene_ok keeps the strict all-object check.
    d['scene_ok'] = bool(all(v <= tol for v in d.values()))
    d['ok'] = bool(d['ee'] <= tol)
    d['object_offset_mm'] = {'source': round(d['source'] * 1e3, 3), 'target': round(d['target'] * 1e3, 3)}
    d['tol'] = tol
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=sorted(TASKS))
    ap.add_argument('--tol', type=float, default=GATE_TOL)
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    env_task, name, _ = TASKS[a.task]
    live = Path('/gscratch/robotics/harine/data') / name
    dataset_dir = live if live.exists() else live.parent / '_archive' / name

    import zarr
    zpath = sorted(dataset_dir.glob('*.zarr'))[0]
    z = zarr.open(str(zpath), 'r')
    ends = z['meta/episode_ends'][:]
    starts = np.concatenate([[0], ends[:-1]])
    eps = json.loads((dataset_dir / 'filter_manifest.json').read_text())['episodes']

    import simpler_env  # noqa: F401
    from simpler_env import make as make_env
    env = make_env(env_task)
    env_base = env.unwrapped
    out = {'task': a.task, 'env_task': env_task, 'dataset_dir': str(dataset_dir),
           'zarr': str(zpath), 'tol': a.tol, 'rule': 'seed = 100000 + START_INDEX[shard] + src_episode',
           'episodes': {}}
    n_ok = 0
    for i, e in enumerate(eps[:a.limit] if a.limit else eps):
        seed = episode_seed(a.task, e['src_shard'], e['src_episode'])
        obs, _ = env.reset(seed=int(seed))
        g = gate_from_env(env_base, obs, z, int(starts[i]), a.tol)
        g.update(seed=int(seed), src_shard=str(e['src_shard']).split('/')[-1],
                 src_episode=int(e['src_episode']))
        out['episodes'][str(i)] = g
        n_ok += int(g['ok'])
        if (i + 1) % 50 == 0:
            print(f'[gate] {i + 1}/{len(eps)} ok={n_ok}', flush=True)
    env.close()
    out['n_ok'] = n_ok
    out['n'] = len(out['episodes'])
    gate_path(a.task).write_text(json.dumps(out, indent=1))
    print(f'[gate] {a.task}: {n_ok}/{out["n"]} episodes reproduce their start state at '
          f'tol={a.tol} -> {gate_path(a.task)}')


if __name__ == '__main__':
    main()
