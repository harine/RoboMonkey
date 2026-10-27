#!/usr/bin/env python3
"""Fixed-state, fixed-history sensitivity to shuffled verifier feedback.

Compare normalized outputs with identical diffusion/observation random draws.
This measures dependence on value assignments, not whether that dependence helps
task success. Both clean/noisy conditions use exactly the same saved history.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'eval_diffusion'))
from eval_diffusion import load_policy


@torch.inference_mode()
def analyze(checkpoint, source):
    paths = sorted(source.glob('ep*.npz'))[:5]
    if not paths:
        raise ValueError('No diagnostic episodes')
    with np.load(paths[0], allow_pickle=False) as z:
        if 'candidate_actions_full' not in z:
            print('Saved observations/history unavailable in this older diagnostic')
            return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy, _ = load_policy(str(checkpoint), device, use_ema=True)
    if not hasattr(policy, 'search_candidates'):
        print('BC has no history-feedback conditioning; skipping sensitivity test')
        return
    rows = []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            # First decision is matched across rollout conditions.
            obs = {k.split('__', 1)[1]: torch.as_tensor(z[k][0], device=device)
                   for k in z.files if k.startswith('policy_obs__')}
            actions = torch.as_tensor(z['candidate_actions_full'][0, :-1], device=device)[None]
            values = torch.as_tensor(z['values'][0, :-1], device=device)[None]
            seed = int(z['seed'])
        record = dict(seed=seed, file=str(path), value_std=float(values.std()))
        encoded = policy._encode_obs_features(obs)
        for corruption in (False, True):
            policy.corrupt_obs_eval = corruption
            distances = []
            for repeat in range(3):
                outputs = []
                for feedback in (values, values.flip(1)):
                    policy.set_sample_seeds([seed + repeat * 1000003])
                    with policy._corrupt_scope():
                        # Separate observation-noise stream: toggling corruption
                        # must not shift the action-diffusion draws across conditions.
                        generator = torch.Generator(device=device).manual_seed(seed + repeat + 173)
                        policy._corrupt_eps = torch.randn(encoded.shape, device=device,
                                                          dtype=encoded.dtype, generator=generator)
                        prediction = policy.predict_action(obs, actions=actions, values=feedback,
                                                           obs_features=encoded,
                                                           **policy._slot_kwargs(actions.shape[1]))['action_pred']
                    outputs.append(policy.normalizer['action'].normalize(prediction)[:, :8])
                    policy.set_sample_seeds(None)
                distances.append(float((outputs[0] - outputs[1]).square().mean().sqrt()))
            record['noisy' if corruption else 'clean'] = dict(
                normalized_action_rmse=float(np.mean(distances)), repeats=distances)
        rows.append(record)
    out = source.parent / 'feedback_sensitivity.json'
    out.write_text(json.dumps(rows, indent=2, allow_nan=False))
    print(out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint', type=Path)
    p.add_argument('source', type=Path)
    a = p.parse_args()
    analyze(a.checkpoint, a.source)
