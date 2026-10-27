#!/usr/bin/env python3
"""Idempotent N32/64 extension; immutable cell lists separate from active arrays."""
import sys
import time
from pathlib import Path
from veritas_h16_campaign import read, write, submit


def tick(root):
    campaign = read(root / 'campaign.json')
    base = read(root / 'evaluations.json')
    registry = root / 'extended_evaluations.json'
    submitted = read(registry) if registry.exists() else {}
    for key, entry in base.items():
        cells = read(entry['cells'])
        first = cells[0]
        spec = campaign['runs'][first['run']]
        if spec['kind'] not in ('bc', 'uniform') or key in submitted:
            continue
        modes = ['argmax'] if spec['kind'] == 'bc' else ['argmax', 'final']
        extra = [dict(first, n=n, mode=mode) for n in (32, 64) for mode in modes]
        path = root / 'cells' / f'extended_{key}.json'
        write(path, extra)
        job = submit(root, f'ex64_{key}', 'eval', path, array=f'0-{len(extra)-1}%2')
        submitted[key] = dict(job=job, checkpoint=entry['checkpoint'], cells=str(path))
        write(registry, submitted)
    expected = sum(s.get('max_gradient_steps', 50000) // 10000
                   for s in campaign['runs'].values() if s['kind'] in ('bc', 'uniform'))
    return len(submitted) == expected


if __name__ == '__main__':
    root = Path(sys.argv[1]).resolve()
    while not tick(root):
        time.sleep(60)
