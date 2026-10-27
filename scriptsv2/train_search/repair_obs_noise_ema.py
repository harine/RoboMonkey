#!/usr/bin/env python3
"""Create a separate checkpoint with online observation-noise statistics in EMA.

Original checkpoint and all learned weights remain unchanged. This repairs only
the two running-statistic buffers omitted by the old EMA implementation.
"""
import argparse
import json
from pathlib import Path
import torch
import dill


def repair(source, destination):
    if destination.exists():
        raise FileExistsError(destination)
    payload = torch.load(source, map_location='cpu', pickle_module=dill, weights_only=False)
    online = payload['state_dicts']['model']
    ema = payload['state_dicts']['ema_model']
    names = ('obs_feature_std', 'obs_feature_std_inited')
    if not bool(online[names[1]]):
        raise ValueError('Online observation statistics were never initialized')
    stats = online[names[0]]
    if not torch.isfinite(stats).all() or not (stats > 0).all():
        raise ValueError('Invalid online feature scales')
    changes = {n: dict(before=ema[n].tolist(), after=online[n].tolist()) for n in names}
    for n in names:
        ema[n] = online[n].clone()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    torch.save(payload, temporary, pickle_module=dill)
    temporary.replace(destination)
    destination.with_suffix('.repair.json').write_text(json.dumps(
        dict(source=str(source.resolve()), destination=str(destination.resolve()),
             buffers=changes, note='EMA parameters unchanged; online running statistics copied'), indent=2))
    print(destination, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    repair(args.source, args.destination)
