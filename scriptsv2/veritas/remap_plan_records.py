#!/usr/bin/env python3
"""Re-key plan records after scriptsv2/data/make_split_dataset.py: ``ep{old}.json/png``
(indices of the strict dataset) become ``ep{new}`` (indices of the split-only dataset),
``seed{seed}.json`` links are re-pointed, and records of episodes outside every split move
to ``<plan_dir>/_not_in_split/``.

    python scriptsv2/veritas/remap_plan_records.py --task eggplant \\
        --plan-dir /gscratch/robotics/harine/data/veritas_plans/eggplant/v1_upstream
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'diffusion_policy'))

from diffusion_policy.veritas.plan_store import link_seed  # noqa: E402

DATA = Path('/gscratch/robotics/harine/data')
SPLIT_DIRS = {'eggplant': 'strict_distance_success_split', 'stack': 'stack_cube_strict_split'}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=sorted(SPLIT_DIRS))
    ap.add_argument('--plan-dir', required=True)
    a = ap.parse_args()
    plan_dir = Path(a.plan_dir)
    marker = plan_dir / 'REMAPPED.json'
    if marker.exists():
        sys.exit(f'{plan_dir} already remapped ({marker})')
    fm = json.loads((DATA / SPLIT_DIRS[a.task] / 'filter_manifest.json').read_text())
    old2new = {int(r['in_episode']): int(r['out_episode']) for r in fm['episodes']}

    stash = plan_dir / '_not_in_split'
    staging = plan_dir / '_remap_staging'
    staging.mkdir(exist_ok=True)
    for link in plan_dir.glob('seed*.json'):
        link.unlink()
    moved, kept = [], []
    for rec in sorted(plan_dir.glob('ep*.json')):
        old = int(rec.stem[2:])
        d = json.loads(rec.read_text())
        png = rec.with_suffix('.png')
        if old not in old2new:
            stash.mkdir(exist_ok=True)
            rec.rename(stash / rec.name)
            if png.exists():
                png.rename(stash / png.name)
            moved.append(old)
            continue
        new = old2new[old]
        d.update(ep_idx=new, in_episode=old, dataset_dir=str(DATA / SPLIT_DIRS[a.task]))
        (staging / f'ep{new}.json').write_text(json.dumps(d, indent=2))
        rec.unlink()
        if png.exists():
            png.rename(staging / f'ep{new}.png')
        kept.append((old, new, d.get('seed'), 'plan' in d))
    for f in staging.iterdir():
        f.rename(plan_dir / f.name)
    staging.rmdir()
    for old, new, seed, planned in kept:
        if planned and seed is not None:
            link_seed(plan_dir, int(seed), f'ep{new}.json')
    marker.write_text(json.dumps({'dataset_dir': str(DATA / SPLIT_DIRS[a.task]),
                                  'kept': [k[:2] for k in kept], 'not_in_split': moved}, indent=1))
    print(f'[remap] {a.task}: kept {len(kept)} records ({sum(k[3] for k in kept)} with plans), '
          f'moved {len(moved)} not-in-split records to {stash}')


if __name__ == '__main__':
    main()
