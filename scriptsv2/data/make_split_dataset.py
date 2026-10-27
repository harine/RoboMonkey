#!/usr/bin/env python3
"""Write the split-only dataset for a task: the strict-filtered episodes that are in at
least one split manifest (train, val or test of any rule/size), in their original order,
with every zarr key. Also remaps the manifests onto it.

Episodes outside every split are not copied; they stay only in the strict dataset, which
is then archived (scriptsv2/data/archive_datasets.sh).

    python scriptsv2/data/make_split_dataset.py --task eggplant [--dry-run]

Writes /gscratch/robotics/harine/data/<strict>_split/{<strict>_split.zarr,
filter_manifest.json, README.md} and rewrites
diffusion_policy/diffusion_policy/config/splits/simpler/<task>/*.json in place (same
episodes and seeds; indices, dataset_dir, zarr_path and checksums for the new zarr). The
pre-remap manifests are kept under splits/simpler/<task>/_pre_split_dataset/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import zarr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'diffusion_policy'))

from diffusion_policy.common.split_manifest import (  # noqa: E402
    episode_ends_checksum, load_split_manifest, split_checksum)

DATA = Path('/gscratch/robotics/harine/data')
SPLIT_ROOT = REPO / 'diffusion_policy' / 'diffusion_policy' / 'config' / 'splits' / 'simpler'
TASKS = {'eggplant': 'strict_distance_success', 'stack': 'stack_cube_strict'}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=sorted(TASKS))
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    manifests = sorted((SPLIT_ROOT / a.task).glob('*.json'))
    ms = {p: json.loads(p.read_text()) for p in manifests}
    in_dir = Path(next(iter(ms.values()))['dataset_dir'])
    assert all(Path(m['dataset_dir']) == in_dir for m in ms.values()), 'manifests disagree on dataset_dir'
    zin = zarr.open(next(iter(ms.values()))['zarr_path'], 'r')
    ends = zin['meta/episode_ends'][:]
    starts = np.concatenate([[0], ends[:-1]])
    for p, m in ms.items():   # every manifest must match the zarr it claims
        load_split_manifest(p, episode_ends=ends)

    keep = sorted({int(i) for m in ms.values() for s in ('train', 'val', 'test') for i in m[s]})
    old2new = {old: new for new, old in enumerate(keep)}
    n_steps = int(sum(ends[i] - starts[i] for i in keep))
    print(f'[split-dataset] {a.task}: {len(keep)} of {len(ends)} episodes ({n_steps} steps) are in a split; '
          f'{len(ends) - len(keep)} excluded')
    if a.dry_run:
        return

    name = f'{TASKS[a.task]}_split'
    out_dir = DATA / name
    out_zarr = out_dir / f'{name}.zarr'
    if out_zarr.exists():
        shutil.rmtree(out_zarr)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = zarr.open(str(out_zarr), mode='w')
    for k, v in dict(zin.attrs).items():
        root.attrs[k] = v
    dgrp = root.create_group('data')
    ogrp = dgrp.create_group('obs')
    data_keys = [k for k in zin['data'].keys() if k != 'obs']
    obs_keys = list(zin['data/obs'].keys())

    def make_out(parent, src, key):
        return parent.create_dataset(key, shape=(n_steps,) + src.shape[1:],
                                     chunks=(src.chunks[0],) + src.shape[1:],
                                     dtype=src.dtype, compressor=src.compressor)

    out_data = {k: make_out(dgrp, zin['data'][k], k) for k in data_keys}
    out_obs = {k: make_out(ogrp, zin['data/obs'][k], k) for k in obs_keys}
    in_eps = json.loads((in_dir / 'filter_manifest.json').read_text())['episodes']
    seeds = next(iter(ms.values()))['episodes']
    rows, cur = [], 0
    for old in keep:
        s, e = int(starts[old]), int(ends[old])
        n = e - s
        for k in data_keys:
            out_data[k][cur:cur + n] = zin['data'][k][s:e]
        for k in obs_keys:
            out_obs[k][cur:cur + n] = zin['data/obs'][k][s:e]
        rows.append({'out_episode': len(rows), 'in_episode': old,
                     'src_shard': in_eps[old].get('src_shard'),
                     'src_episode': in_eps[old].get('src_episode'),
                     'seed': seeds[str(old)]['seed'], 'n_steps': n,
                     'out_start': cur, 'out_end': cur + n})
        cur += n
    assert cur == n_steps
    new_ends = np.array([r['out_end'] for r in rows], dtype=np.int64)
    root.create_group('meta').create_dataset('episode_ends', data=new_ends,
                                             chunks=(len(rows),), dtype=np.int64)
    (out_dir / 'filter_manifest.json').write_text(json.dumps({
        'in_dir': str(in_dir), 'out_zarr': str(out_zarr), 'n_input': int(len(ends)),
        'n_episodes': len(rows), 'n_steps': n_steps,
        'selection': 'episodes in any split manifest under config/splits/simpler/' + a.task,
        'manifests': [p.name for p in manifests], 'episodes': rows}, indent=1))
    (out_dir / 'README.md').write_text(f"""# {name}

The episodes of `{in_dir.name}` that appear in at least one split manifest
(`diffusion_policy/diffusion_policy/config/splits/simpler/{a.task}/`): {len(rows)} of {len(ends)},
{n_steps} steps, original order, all zarr keys. Every episode is recreatable from its seed
(end effector reproduces exactly; object settle offsets are environment stochasticity, see
`scriptsv2/eval_search/demoseeds/{a.task}_strict_seed_gate.json`).

`filter_manifest.json` maps each episode to its index in `{in_dir.name}` (`in_episode`), its
raw shard coordinates and env seed. Built by `scriptsv2/data/make_split_dataset.py`; the
{len(ends) - len(rows)} excluded episodes are archived with `{in_dir.name}` under `_archive/`.
""")

    # remap manifests; keep the originals for provenance
    backup = SPLIT_ROOT / a.task / '_pre_split_dataset'
    backup.mkdir(exist_ok=True)
    ck = episode_ends_checksum(new_ends)
    for p, m in ms.items():
        shutil.copy2(p, backup / p.name)
        for s in ('train', 'val', 'test'):
            m[s] = sorted(old2new[int(i)] for i in m[s])
        m['episodes'] = {str(old2new[int(i)]): {**r, 'in_episode': int(i)}
                         for i, r in m['episodes'].items() if int(i) in old2new}
        m.update(dataset_dir=str(out_dir), zarr_path=str(out_zarr), n_episodes=len(rows),
                 episode_ends_checksum=ck, checksum=split_checksum(m['train'], m['val'], m['test']))
        m.setdefault('derivation', {})['remapped_from'] = str(in_dir)
        p.write_text(json.dumps(m, indent=1))
        load_split_manifest(p, episode_ends=new_ends)
    # first-row poses must be identical under the map
    zout = zarr.open(str(out_zarr), 'r')
    for k in ('end_effector_pose', 'insertive_asset_pose', 'receptive_asset_pose'):
        a_ = zin['data/obs'][k][[int(starts[o]) for o in keep]]
        b_ = zout['data/obs'][k][[r['out_start'] for r in rows]]
        assert np.array_equal(a_, b_), k
    print(f'[split-dataset] wrote {out_zarr}; remapped {len(ms)} manifests (originals in {backup})')


if __name__ == '__main__':
    main()
