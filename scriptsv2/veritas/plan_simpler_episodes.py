#!/usr/bin/env python3
"""Gemini waypoint plans for the episodes of a split manifest, one call per episode.

For each episode the env is reset with the episode's reconstructed seed, the reset
state is checked against the zarr's first row (the SEED GATE: object and end-effector
positions must match, else the seed is wrong and the episode is refused rather than
planned on the wrong scene), the 640x480 first frame is captured, the end-effector and
the source/target objects are projected to pixels, and Gemini returns a plan. Records
land as ``<outdir>/ep{idx}.json`` (raw response + clipped ``plan`` + provenance), with
``seed{seed}.json`` symlinks so eval resolves plans by seed, an overlay png per episode
and an overview grid per split, and a usage/cost summary.

    export GEMINI_API_KEY=...
    MUJOCO_GL=osmesa python scriptsv2/veritas/plan_simpler_episodes.py \\
        --task widowx_put_eggplant_in_basket \\
        --split-file diffusion_policy/diffusion_policy/config/splits/simpler/eggplant/random_train400.json \\
        --splits train val test --outdir /gscratch/robotics/harine/data/veritas_plans/eggplant/v1_upstream \\
        --skip-planned
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'diffusion_policy'))

from diffusion_policy.env.pusht.veritas.draw import draw_waypoints_rgb  # noqa: E402
from diffusion_policy.env.simpler.veritas.camera import CameraModel, tcp_xyz  # noqa: E402
from diffusion_policy.env.simpler.veritas.hints import object_hints  # noqa: E402
from diffusion_policy.env.simpler.veritas.simpler_client import (  # noqa: E402
    PROMPTS, SimplerVeritasClient)
from diffusion_policy.veritas.plan_store import (  # noqa: E402
    cost_usd, has_plan, link_seed, read_record, usage_summary, write_plan_record)
sys.path.insert(0, str(REPO / 'scriptsv2' / 'veritas'))
from seed_gate import gate_from_env  # noqa: E402

def overlay(frame, plan, eef_px, hints, caption):
    import cv2
    dbg = [{'uv': tuple(wp.uv), 'idx': i + 1} for i, wp in enumerate(plan.waypoints)] if plan else []
    img = draw_waypoints_rgb(frame.copy(), dbg, draw_curve=True) if dbg else frame.copy()
    cv2.circle(img, (int(eef_px[0]), int(eef_px[1])), 6, (255, 255, 255), 2)
    for h in hints:
        u, v = int(h['uv'][0]), int(h['uv'][1])
        cv2.drawMarker(img, (u, v), (0, 255, 0) if h['role'] == 'source' else (255, 200, 0),
                       cv2.MARKER_CROSS, 16, 2)
    strip = np.full((22, img.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(strip, caption, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([img, strip], axis=0)


def grid(tiles, cols=5):
    if not tiles:
        return None
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.full((rows * h, cols * w, 3), 255, dtype=np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--split-file', required=True)
    ap.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    ap.add_argument('--episodes', type=int, nargs='*', default=None,
                    help='explicit episode indices (overrides --splits)')
    ap.add_argument('--prompt', default='v1_upstream', choices=sorted(PROMPTS))
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--camera-file', default=None,
                    help='default: diffusion_policy/diffusion_policy/config/veritas/camera_<task>.json')
    ap.add_argument('--model', default=os.environ.get('GEMINI_MODEL', 'gemini-3.8-flash'))
    ap.add_argument('--skip-planned', action='store_true')
    ap.add_argument('--redraw', action='store_true', help='overlays only, no Gemini calls')
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()

    import zarr
    manifest = json.loads(Path(a.split_file).read_text())
    zroot = zarr.open(manifest['zarr_path'], 'r')
    ends = zroot['meta/episode_ends'][:]
    starts = np.concatenate([[0], ends[:-1]])
    eps = manifest['episodes']
    if a.episodes is not None:
        todo = [(int(i), None) for i in a.episodes]
    else:
        todo = [(int(i), s) for s in a.splits for i in manifest[s]]
    if a.limit:
        todo = todo[:a.limit]

    cam_file = a.camera_file or (REPO / 'diffusion_policy/diffusion_policy/config/veritas'
                                 / f'camera_{a.task}.json')
    camera = CameraModel.load(cam_file)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'prompt.txt').write_text(f'# prompt: {a.prompt}\n# model: {a.model}\n\n{PROMPTS[a.prompt]}')

    client = None
    if not a.redraw:
        try:
            client = SimplerVeritasClient(model_name=a.model)
        except ValueError as e:
            sys.exit(f'{e}\nExport the key first: export GEMINI_API_KEY=...')

    import cv2
    import simpler_env  # noqa: F401
    from simpler_env import make as make_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    env = make_env(a.task)
    env_base = env.unwrapped
    instruction = env_base.get_language_instruction()

    tiles, called, gate_fail = {}, [], []
    for ep_idx, split in todo:
        row = eps[str(ep_idx)]
        seed = row.get('seed')
        rec_path = out / f'ep{ep_idx}.json'
        if seed is None:
            print(f'ep{ep_idx}: no reconstructable seed, skipped')
            continue
        obs, _ = env.reset(seed=int(seed))
        camera.assert_matches(obs)
        gate = gate_from_env(env_base, obs, zroot, int(starts[ep_idx]))
        if not gate['ok']:
            gate_fail.append(ep_idx)
            print(f'ep{ep_idx} seed {seed}: SEED GATE FAILED {gate} -> not planned')
            write_plan_record(rec_path, None, seed=int(seed), ep_idx=ep_idx, split=split,
                              seed_gate=gate, src_shard=row['src_shard'],
                              src_episode=row['src_episode'])
            continue
        frame = np.asarray(get_image_from_maniskill2_obs_dict(env_base, obs), dtype=np.uint8)
        eef_px = camera.project(tcp_xyz(obs)).tolist()
        hints = object_hints(env_base, camera)

        if a.redraw or (a.skip_planned and has_plan(rec_path)):
            from diffusion_policy.veritas.plan_store import load_plan
            plan = load_plan(rec_path) if has_plan(rec_path) else None
        else:
            plan = client.call_plan(frame, instruction, eef_pixel=eef_px, object_hints=hints,
                                    prompt=a.prompt, log_dir=str(out), log_name=f'ep{ep_idx}')
            called.append(ep_idx)
            write_plan_record(
                rec_path, plan, model=client.model_name, image_size=list(frame.shape[:2][::-1]),
                prompt=client.last_prompt, hints=hints, eef_pixel=eef_px, usage=client.last_usage,
                seed=int(seed), ep_idx=ep_idx, split=split, seed_gate=gate,
                src_shard=row['src_shard'], src_episode=row['src_episode'],
                instruction=instruction, task=a.task, camera_file=str(cam_file))
            if plan is not None:
                link_seed(out, int(seed), rec_path.name)
        rec = read_record(rec_path) if rec_path.exists() else {}
        usage = rec.get('usage') or {}
        cost = cost_usd(usage, rec.get('model'))
        cap = (f'ep{ep_idx} seed {seed} | ' + (f'{len(plan.waypoints)} wp conf {plan.confidence:.2f}'
                                                if plan else 'NO PLAN')
               + (f' | ${cost:.4f}' if cost is not None else ''))
        print(cap, flush=True)
        tile = overlay(frame, plan, eef_px, hints, cap)
        cv2.imwrite(str(out / f'ep{ep_idx}.png'), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
        tiles.setdefault(split or 'explicit', []).append(tile)
    env.close()

    for split, ts in tiles.items():
        g = grid(ts)
        if g is not None:
            cv2.imwrite(str(out / f'overlays_{split}.png'), cv2.cvtColor(g, cv2.COLOR_RGB2BGR))
    print(f'wrote {sum(map(len, tiles.values()))} overlays to {out}/; seed-gate failures: {gate_fail}')
    print(usage_summary(out, called, a.model))


if __name__ == '__main__':
    main()
