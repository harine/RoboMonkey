#!/usr/bin/env python3
"""Pipeline smoke fixture WITHOUT Gemini: a tiny split manifest over seed-exact (state1)
episodes and a synthetic straight-line plan per episode (end-effector pixel -> target
object pixel, 3 waypoints, upstream defaults), written in the real record format so
build_veritas_targets.py, the ST trainer and the BoN eval run end-to-end on them.
These plans are NOT a verifier; they only exercise the plumbing.

    python scriptsv2/veritas/smoke_synthetic_plans.py --task eggplant --out data/veritas/_smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'diffusion_policy'))
sys.path.insert(0, str(REPO / 'scriptsv2' / 'splits'))

from diffusion_policy.common.split_manifest import episode_ends_checksum, split_checksum  # noqa: E402
from diffusion_policy.env.simpler.veritas.camera import CameraModel  # noqa: E402
from diffusion_policy.veritas.plan_store import link_seed, write_plan_record  # noqa: E402
from diffusion_policy.veritas.schemas import VeritasPlan, VeritasWaypoint  # noqa: E402
import make_simpler_splits as ms  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='eggplant', choices=sorted(ms.TASKS))
    ap.add_argument('--out', default='data/veritas/_smoke')
    ap.add_argument('--n-train', type=int, default=6)
    ap.add_argument('--n-val', type=int, default=2)
    ap.add_argument('--n-test', type=int, default=2)
    a = ap.parse_args()
    t = ms.TASKS[a.task]
    zpath, ends, rows = ms.episode_table(a.task)
    seeded = [i for i, r in rows.items() if r['seed'] is not None and r['src_shard'] == 'state1.zarr']
    n = a.n_train + a.n_val + a.n_test
    pick = seeded[:n]
    train, val, test = pick[:a.n_train], pick[a.n_train:a.n_train + a.n_val], pick[a.n_train + a.n_val:]
    out = Path(a.out) / a.task
    out.mkdir(parents=True, exist_ok=True)
    m = dict(task=a.task, env_task=t['env_task'], dataset_dir=str(t['dataset_dir']), zarr_path=str(zpath),
             n_episodes=int(len(ends)), episode_ends_checksum=episode_ends_checksum(ends),
             train=train, val=val, test=test, checksum=split_checksum(train, val, test),
             derivation=dict(rule='smoke', seed=0), episodes={str(i): rows[i] for i in sorted(rows)})
    (out / 'smoke_split.json').write_text(json.dumps(m, indent=1))

    import zarr
    z = zarr.open(str(zpath), 'r')
    starts = np.concatenate([[0], ends[:-1]])
    cam = CameraModel.load(REPO / 'diffusion_policy/diffusion_policy/config/veritas' / f"camera_{t['env_task']}.json")
    plan_dir = out / 'plans'
    plan_dir.mkdir(exist_ok=True)
    for i in pick:
        s = int(starts[i])
        ee = cam.project(np.asarray(z['data/obs/end_effector_pose'][s])[:3])
        src = cam.project(np.asarray(z['data/obs/insertive_asset_pose'][s])[:3])
        tgt = cam.project(np.asarray(z['data/obs/receptive_asset_pose'][s])[:3])
        pts = [0.5 * (ee + src), src, 0.5 * (src + tgt), tgt]
        plan = VeritasPlan(waypoints=[
            VeritasWaypoint(uv=(float(p[0]), float(p[1])), tol_px=60.0, min_hold=1 if k < 3 else 2,
                            skippable=(k < 3)) for k, p in enumerate(pts)], confidence=0.0, notes='synthetic')
        rec = plan_dir / f'ep{i}.json'
        write_plan_record(rec, plan, model='synthetic', seed=int(rows[i]['seed']), ep_idx=i,
                          synthetic=True, usage=None)
        link_seed(plan_dir, int(rows[i]['seed']), rec.name)
    print(f'[smoke] manifest {out / "smoke_split.json"} (train {train} val {val} test {test}); '
          f'{len(pick)} synthetic plans in {plan_dir}')


if __name__ == '__main__':
    main()
