#!/usr/bin/env python3
"""Summarize saved candidates in native units, keeping episode as the unit of analysis.

The first decision is a matched environment state across observation ablations.
Later states diverge, so their diversity is descriptive rather than a paired effect.
Verifier improvements are not counterfactual task-success measurements.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(z, first=False):
    a = z['candidate_actions'][:1] if first else z['candidate_actions']
    a = a[:, :, :8]  # This campaign scores and executes the first eight commands.
    q = z['values'][:1] if first else z['values']
    k = a.shape[1]
    i, j = np.triu_indices(k, 1)
    endpoints = a[..., :3].sum(axis=2)
    translation_rmse = np.sqrt(np.mean((a[:, i, :, :3] - a[:, j, :, :3])**2, axis=(2, 3)))
    grip_disagree = (a[:, i, :, -1] > 0) != (a[:, j, :, -1] > 0)
    return dict(pairwise_translation_rmse_m=float(translation_rmse.mean()),
                pairwise_endpoint_distance_m=float(np.linalg.norm(endpoints[:, i]-endpoints[:, j], axis=-1).mean()),
                gripper_sign_disagreement=float(grip_disagree.mean()),
                mean_value_by_slot=q.mean(axis=0).tolist(),
                final_minus_first_value=float((q[:, -1]-q[:, 0]).mean()),
                best_minus_first_value=float((q.max(axis=1)-q[:, 0]).mean()),
                tied_values_fraction=float((np.ptp(q, axis=1) < 1e-8).mean()))


def analyze(source):
    rows = []
    for path in sorted(source.glob('ep*.npz')):
        with np.load(path, allow_pickle=False) as z:
            rows.append(dict(file=str(path), seed=int(z['seed']), success=bool(z['success']),
                             first_decision=summarize(z, True), all_decisions=summarize(z)))
    out = source.parent / 'candidate_diversity.json'
    out.write_text(json.dumps(rows, indent=2, allow_nan=False))
    print(out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    analyze(p.parse_args().source)
