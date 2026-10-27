#!/usr/bin/env python3
"""Train / val / test split manifests for the strict-filtered SimplerEnv datasets.

One manifest per (rule, train size), in the PushT split-manifest format read by
``diffusion_policy.dataset.pusht_image_dataset.load_split_manifest`` (``train``/``val``/
``test`` episode-index lists, ``n_episodes``, ``episode_ends_checksum``, ``checksum``),
plus per-episode provenance (``episodes[idx] = {src_shard, src_episode, seed, start_xy,
start_cell, goal_cell}``) so eval can roll the test episodes out from their env seeds and
training checkpoints can record which seeds they saw.

Rules (``--task eggplant``; ``--task stack`` has no ``inner`` rule):
  random    test/val stratified over start cells; train = the remaining pool.
  quadrant  test pool = one corner of the start grid; train from the rest.
  inner     test pool = a disc around the grid centre; train from outside it.

Test (50) and val (20) are drawn stratified by start cell (round-robin over cells with
a seeded shuffle within each). The train episodes are ONE stratified ordering of the
remaining pool, and each train size is a prefix of it, so train50 ⊂ train100 ⊂ train400.
Episodes that fail the seed gate (``scriptsv2/veritas/seed_gate.py``: the seed does not
reproduce the recorded start state) are never placed in test, since test episodes are
rolled out from their seeds; they may still train.

    python scriptsv2/splits/make_simpler_splits.py --task eggplant
    python scriptsv2/splits/make_simpler_splits.py --task stack
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import zarr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'diffusion_policy'))
sys.path.insert(0, str(REPO / 'scriptsv2' / 'analysis'))

from diffusion_policy.common.split_manifest import (  # noqa: E402
    episode_ends_checksum, split_checksum)
from reconstruct_demo_seeds import START_INDEX, START_SEED  # noqa: E402

DATA = Path('/gscratch/robotics/harine/data')


def strict_dir(name: str) -> Path:
    """The strict-filtered source dataset; archived under _archive/ once the split-only
    datasets (scriptsv2/data/make_split_dataset.py) replaced it."""
    live = DATA / name
    return live if live.exists() else DATA / '_archive' / name
SPLIT_ROOT = REPO / 'diffusion_policy' / 'diffusion_policy' / 'config' / 'splits' / 'simpler'
sys.path.insert(0, str(REPO / 'scriptsv2' / 'veritas'))
from seed_gate import gate_path  # noqa: E402

TASKS = {
    'eggplant': dict(env_task='widowx_put_eggplant_in_basket',
                     dataset_dir=strict_dir('strict_distance_success'), seed_task='eggplant',
                     train_sizes={'random': (400, 100, 50), 'quadrant': (100, 50), 'inner': (100, 50)},
                     # cell-based, not decimal medians: the column at x~-0.1168 must fall on
                     # one side, and the 79-episode cell at y=0.213 is excluded from the corner
                     quadrant=dict(x_max=-0.1165, y_max=0.2125),
                     inner=dict(center=(-0.1135, 0.2140), radius=0.006)),
    'stack': dict(env_task='widowx_stack_cube',
                  dataset_dir=strict_dir('stack_cube_strict'), seed_task='stack_cube',
                  train_sizes={'random': (200, 100, 50), 'quadrant': (50,)},
                  quadrant=dict(y_min=0.0)),
}
N_TEST, N_VAL = 50, 20


def episode_table(task: str):
    t = TASKS[task]
    zpath = sorted(t['dataset_dir'].glob('*.zarr'))[0]
    z = zarr.open(str(zpath), 'r')
    ends = z['meta/episode_ends'][:]
    starts = np.concatenate([[0], ends[:-1]])
    src = z['data/obs/insertive_asset_pose'][starts, :2]
    goal = z['data/obs/receptive_asset_pose'][starts, :2]
    manifest = json.loads((t['dataset_dir'] / 'filter_manifest.json').read_text())
    eps = manifest['episodes']
    assert len(eps) == len(ends), (len(eps), len(ends))
    gate_file = gate_path(task)
    if not gate_file.exists():
        raise SystemExit(f'{gate_file} missing: run scriptsv2/veritas/seed_gate.py --task {task} '
                         f'first (test episodes must reproduce their start state from their seed)')
    gate = json.loads(gate_file.read_text())['episodes']
    rows = {}
    for i, e in enumerate(eps):
        assert int(e['out_start']) == int(starts[i]), (i, e['out_start'], starts[i])
        shard = str(e['src_shard']).split('/')[-1]
        k = int(e['src_episode'])
        seed = START_SEED + START_INDEX[t['seed_task']][shard] + k
        g = gate.get(str(i))
        assert g is not None and int(g['seed']) == seed, (i, g, seed)
        rows[i] = dict(src_shard=shard, src_episode=k, seed=seed,
                       seed_gate=bool(g.get('scene_ok', g['ok'])),
                       start_xy=[round(float(v), 4) for v in src[i]],
                       goal_xy=[round(float(v), 4) for v in goal[i]],
                       start_cell=[round(float(v), 3) for v in src[i]],
                       goal_cell=[round(float(v), 2) for v in goal[i]])
    return zpath, ends, rows


def stratified_order(idxs, cell_of, rng):
    """Round-robin over cells (shuffled within each, cell order shuffled per round)."""
    by_cell = defaultdict(list)
    for i in idxs:
        by_cell[cell_of[i]].append(i)
    queues = {c: list(rng.permutation(v)) for c, v in by_cell.items()}
    out = []
    while queues:
        keys = sorted(queues, key=str)
        for j in rng.permutation(len(keys)):
            c = keys[int(j)]
            out.append(int(queues[c].pop()))
            if not queues[c]:
                del queues[c]
    return out


def test_pool(task, rule, rows):
    t = TASKS[task]
    xy = {i: r['start_xy'] for i, r in rows.items()}
    if rule == 'random':
        return list(rows)
    if rule == 'quadrant':
        q = t['quadrant']
        return [i for i, (x, y) in xy.items()
                if ('x_max' not in q or x < q['x_max']) and ('y_max' not in q or y < q['y_max'])
                and ('y_min' not in q or y > q['y_min'])]
    if rule == 'inner':
        c, r = np.array(t['inner']['center']), t['inner']['radius']
        return [i for i, p in xy.items() if np.linalg.norm(np.array(p) - c) <= r]
    raise ValueError(rule)


def build(task, rule, seed):
    t = TASKS[task]
    zpath, ends, rows = episode_table(task)
    rng = np.random.default_rng(seed)
    cell_of = {i: tuple(r['start_cell']) for i, r in rows.items()}
    pool = [i for i in test_pool(task, rule, rows) if rows[i]['seed_gate']]
    if len(pool) < N_TEST:
        raise SystemExit(f'{task}/{rule}: test pool has {len(pool)} gate-passing episodes < {N_TEST}')
    test = sorted(stratified_order(pool, cell_of, rng)[:N_TEST])
    rest = [i for i in rows if i not in set(test)]
    if rule != 'random':
        # val and train come from OUTSIDE the held-out region
        region = set(test_pool(task, rule, rows))
        rest = [i for i in rest if i not in region]
    order = stratified_order(rest, cell_of, rng)
    val = sorted(order[:N_VAL])
    train_order = order[N_VAL:]
    sizes = t['train_sizes'][rule]
    if max(sizes) > len(train_order):
        raise SystemExit(f'{task}/{rule}: {len(train_order)} episodes left for train, '
                         f'need {max(sizes)}')
    ck = episode_ends_checksum(ends)
    out = {}
    for n in sizes:
        train = sorted(train_order[:n])
        m = dict(
            task=task, env_task=t['env_task'], dataset_dir=str(t['dataset_dir']),
            zarr_path=str(zpath), n_episodes=int(len(ends)), episode_ends_checksum=ck,
            train=train, val=val, test=test,
            checksum=split_checksum(train, val, test),
            derivation=dict(rule=rule, seed=int(seed), n_test=N_TEST, n_val=N_VAL,
                            n_train=int(n), train_sizes=list(sizes),
                            nested='train sizes are prefixes of one stratified ordering',
                            region=(t.get(rule) if rule != 'random' else None),
                            test_pool_size=len(pool),
                            n_gate_failed=sum(not r['seed_gate'] for r in rows.values())),
            episodes={str(i): rows[i] for i in sorted(rows)},
        )
        out[n] = m
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=sorted(TASKS))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-dir', default=None)
    a = ap.parse_args()
    out_dir = Path(a.out_dir) if a.out_dir else SPLIT_ROOT / a.task
    out_dir.mkdir(parents=True, exist_ok=True)
    for rule in TASKS[a.task]['train_sizes']:
        for n, m in build(a.task, rule, a.seed).items():
            path = out_dir / f'{rule}_train{n}.json'
            path.write_text(json.dumps(m, indent=1))
            cells = lambda idxs: len({tuple(m['episodes'][str(i)]['start_cell']) for i in idxs})
            print(f'[split] {a.task:8s} {rule:8s} train{n:<4d} -> {path.name}: '
                  f'train {len(m["train"])} ({cells(m["train"])} cells)  val {len(m["val"])}  '
                  f'test {len(m["test"])} ({cells(m["test"])} cells)  '
                  f'gate-failed {m["derivation"]["n_gate_failed"]}')


if __name__ == '__main__':
    main()
