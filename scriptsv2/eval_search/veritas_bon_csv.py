#!/usr/bin/env python3
"""One row per Veritas BoN eval cell (live + offline protocols), rewritten in place.

Cells are <root>/<task>/<rule>/<run>/<step>/<arm>/<mode>/n<N>/episodes.jsonl under
  data/eval/veritas_bon_upstream  (protocol=live: Gemini plan at each rollout reset)
  data/eval/veritas_bon           (protocol=offline: cached per-seed plans)
Partial cells are included (done=0) so the file tracks progress too.

    python scriptsv2/eval_search/veritas_bon_csv.py [--out data/eval/veritas_bon_results.csv]
"""
import argparse
import csv
import glob
import json
import os
import re
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOTS = (('live', 'data/eval/veritas_bon_upstream'), ('offline', 'data/eval/veritas_bon'))
FIELDS = ['protocol', 'task', 'rule', 'train_size', 'run', 'step', 'arm', 'obs_noise', 'mode', 'N',
          'successes', 'episodes', 'success_rate', 'done', 'errors', 'updated', 'path']


def rows():
    for protocol, root in ROOTS:
        for f in glob.glob(os.path.join(REPO, root, '*/*/*/*/*/*/n*/episodes.jsonl')):
            cell = os.path.dirname(f)
            task, rule, run, step, arm, mode, ndir = os.path.relpath(cell, os.path.join(REPO, root)).split(os.sep)
            eps = [json.loads(l) for l in open(f) if l.strip()]
            succ = sum(int(e.get('success', 0)) for e in eps)
            m = re.search(r'_obsnoise([0-9.]+)$', arm)
            size = re.search(r'train(\d+)$', rule)
            yield {
                'protocol': protocol, 'task': task, 'rule': rule.split('_train')[0],
                'train_size': size.group(1) if size else '', 'run': run,
                'step': int(step.split('_')[-1]), 'arm': arm.split('_obsnoise')[0],
                'obs_noise': m.group(1) if m else '0', 'mode': mode, 'N': int(ndir[1:]),
                'successes': succ, 'episodes': len(eps),
                'success_rate': f'{succ / len(eps):.3f}' if eps else '',
                'done': int(os.path.exists(os.path.join(cell, 'eval_log.json'))),
                'errors': sum(1 for e in eps if e.get('error')),
                'updated': time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(f))),
                'path': os.path.relpath(cell, REPO),
            }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(REPO, 'data/eval/veritas_bon_results.csv'))
    a = ap.parse_args()
    rs = sorted(rows(), key=lambda r: (r['protocol'], r['task'], r['rule'], r['train_size'], r['step'],
                                       r['arm'], r['obs_noise'], r['mode'], r['N']))
    tmp = a.out + '.tmp'
    with open(tmp, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rs)
    os.replace(tmp, a.out)                      # readers never see a half-written file
    print(f'{len(rs)} cells -> {a.out}')


if __name__ == '__main__':
    main()
