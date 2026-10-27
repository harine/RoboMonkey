#!/usr/bin/env python3
"""One-screen status of the Veritas BC best-of-N sweep: plan coverage per manifest,
job counts, and every finished cell's success rate (task / split rule / checkpoint step /
scale arm / selection mode / N), as a compact table.

    python3 scriptsv2/eval_search/veritas_bon_status.py [--since-file .seen]
"""
import argparse
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'data' / 'eval' / 'veritas_bon'
SPLITS = REPO / 'diffusion_policy' / 'diffusion_policy' / 'config' / 'splits' / 'simpler'
PLANS = Path('/gscratch/robotics/harine/data/veritas_plans')


def plan_coverage():
    rows = []
    for task_dir in sorted(SPLITS.iterdir()):
        if not task_dir.is_dir():
            continue
        for m in sorted(task_dir.glob('*.json')):
            d = json.loads(m.read_text())
            pd = PLANS / task_dir.name / 'v1_upstream'
            have = lambda idxs: sum((pd / f"seed{d['episodes'][str(i)]['seed']}.json").exists() for i in idxs)
            rows.append(f"{task_dir.name}/{m.stem}: test {have(d['test'])}/{len(d['test'])} "
                        f"train+val {have(d['train'] + d['val'])}/{len(d['train']) + len(d['val'])}")
    return rows


def results():
    cells = defaultdict(dict)   # (task, rule, run, step, arm, mode) -> {n: (sr, n_eps)}
    for log in OUT.glob('*/*/*/*/*/*/n*/eval_log.json'):
        task, rule, run, step, arm, mode, n = log.parent.relative_to(OUT).parts
        try:
            d = json.loads(log.read_text())
        except ValueError:
            continue
        sr = d.get('success_rate')
        if sr is None:
            continue
        neps = d.get('num_episodes') or d.get('episodes') or ''
        cells[(task, rule, step, arm, mode)][int(n[1:])] = (sr, neps)
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    q = subprocess.run(['squeue', '-u', os.environ.get('USER', 'harine'), '-h', '-o', '%j %t'],
                       capture_output=True, text=True).stdout.split('\n')
    cnt = defaultdict(int)
    for line in q:
        if not line.strip():
            continue
        name, st = line.split()
        kind = 'vbon' if name.startswith('vbon_') else ('bc' if name.startswith('bc_unet') else
                                                        ('plan' if name.startswith('plan') else None))
        if kind:
            cnt[f'{kind}:{st}'] += 1
    print('jobs: ' + ', '.join(f'{k}={v}' for k, v in sorted(cnt.items())))
    print('plans: ' + ' | '.join(plan_coverage()))
    cells = results()
    if not cells:
        print('results: none yet')
        return
    ns = sorted({n for v in cells.values() for n in v})
    print('results (success rate over test episodes; N=1 shared by both modes):')
    print('  ' + 'task/rule/step/arm/mode'.ljust(58) + ''.join(f'N={n:<6}' for n in ns))
    for key in sorted(cells):
        row = cells[key]
        base = cells.get(key[:4] + ('argmax',), {}).get(1)
        vals = []
        for n in ns:
            v = row.get(n) or (base if n == 1 else None)
            vals.append(f'{v[0]:<8.2f}' if v else '-'.ljust(8))
        print('  ' + '/'.join(key).ljust(58) + ''.join(vals))


if __name__ == '__main__':
    main()
