#!/usr/bin/env python3
"""Evaluate noisy-trained ST with clean observations at each immutable milestone."""
import sys
import time
from pathlib import Path
from veritas_h16_campaign import read, write, submit


def tick(root):
    campaign = read(root / 'campaign.json')
    registry = root / 'clean_evaluations.json'
    done = read(registry) if registry.exists() else {}
    for key, entry in read(root / 'evaluations.json').items():
        source = read(entry['cells'])
        if not source:
            continue
        spec = campaign['runs'][source[0]['run']]
        if spec['kind'] not in ('flat10', 'ramp2to10') or key in done:
            continue
        cells = [dict(cell, output_group='eval_clean', obs_corruption='clean', viz_q=False)
                 for cell in source]
        assert all(Path(cell['checkpoint']).name.startswith('step_') for cell in cells)
        file = root / 'cells' / f'clean_{key}.json'
        write(file, cells)
        job = submit(root, f'clean_{key}', 'eval', file, array=f'0-{len(cells)-1}%2')
        done[key] = dict(job=job, checkpoint=entry['checkpoint'], cells=str(file))
        write(registry, done)
    expected = sum(s.get('max_gradient_steps', 50000) // 10000
                   for s in campaign['runs'].values() if s['kind'] in ('flat10', 'ramp2to10'))
    return len(done) == expected


if __name__ == '__main__':
    root = Path(sys.argv[1])
    while not tick(root):
        time.sleep(60)
