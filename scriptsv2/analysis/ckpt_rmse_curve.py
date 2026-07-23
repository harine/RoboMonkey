#!/usr/bin/env python3
"""Training curves: best-of-N action RMSE vs training step, on TRAIN and TEST demos.

For each selected checkpoint of a sub50 SEARCH policy, sample N candidate action chunks on a
fixed set of demo states and measure how close the candidates get to the expert action a*:

  best_of_N  = min RMSE among candidates 0..N-1        (the oracle a perfect ranker would get)
  qpick_N    = RMSE of argmax_Q among candidates 0..N-1 (what Best-of-N actually selects)
  q_best_N   = max Q among candidates 0..N-1

Reported for every N = 1..n_samples, so one pass gives the whole best-of-N curve.
`train` = the 50 demos the policy trained on; `test` = the held-out testset_not_in_sub100.

Two things make this affordable:
  * the policy + its baked DQC verifier (JAX init) are built ONCE and the checkpoint weights are
    swapped in per step -- not re-instantiated 20 times;
  * the observation windows and expert chunks are precomputed ONCE from the zarr.

  python scriptsv2/analysis/ckpt_rmse_curve.py --ckpt-dir <run>/checkpoints --zarr <ds>.zarr \
      --task widowx_spoon_on_towel --instruction "put the spoon on the towel" \
      --split train --out <csv> --n-samples 64 --min-step 5000 --ckpt-stride 5
Out CSV: step,split,N,best_of_N,qpick_N,q_best_N,q_mean_i0,n_frames
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
# `diffusion_policy` must resolve to the INNER package (<repo>/diffusion_policy/diffusion_policy).
# The outer submodule dir has no __init__.py, so if it wins the import it becomes a namespace
# package with no `policy` subdir and hydra cannot locate SearchPolicyRoboMonkeyDiffusion.
# Don't rely on PYTHONPATH being set by the caller -- put the package root on sys.path here.
sys.path.insert(0, str(_REPO / "diffusion_policy"))
sys.path.insert(0, str(_REPO / "scriptsv2" / "q_fn_eval"))
sys.path.insert(0, str(_REPO / "scriptsv2" / "eval_diffusion"))

import diffusion_policy as _dp  # noqa: E402
print(f"[curve] diffusion_policy -> {_dp.__file__}", flush=True)

from sample_offline_candidates import (  # type: ignore  # noqa: E402
    build_obs_dict, build_policy, pick_episodes,
)


def rmse_to_astar(cand: np.ndarray, a_star: np.ndarray) -> np.ndarray:
    """(K,H,7) vs (H,7) -> (K,) RMSE over all dims, NaN-pad aware."""
    diff = cand.astype(np.float64) - a_star[None].astype(np.float64)
    mask = ~np.isnan(a_star[None])
    se = np.where(mask, diff ** 2, np.nan)
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.nanmean(se.reshape(se.shape[0], -1), axis=1))


def select_ckpts(ckpt_dir: str, min_step: int, stride: int):
    rows = []
    for p in sorted(Path(ckpt_dir).glob("step_*.ckpt")):
        m = re.search(r"step_0*(\d+)\.ckpt$", p.name)
        if m:
            rows.append((int(m.group(1)), str(p)))
    rows.sort()
    rows = [r for r in rows if r[0] >= min_step]
    return rows[::stride]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--n-action-steps", type=int, default=4)
    ap.add_argument("--num-episodes", type=int, default=50)
    ap.add_argument("--stride", type=int, default=4, help="frame stride within an episode")
    ap.add_argument("--min-step", type=int, default=5000)
    ap.add_argument("--ckpt-stride", type=int, default=5)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    import zarr

    ckpts = select_ckpts(a.ckpt_dir, a.min_step, a.ckpt_stride)
    if not ckpts:
        raise SystemExit(f"no checkpoints >= {a.min_step} in {a.ckpt_dir}")
    ckpts = ckpts[a.shard::a.nshards]     # interleaved so every shard spans the whole x-axis
    if not ckpts:
        raise SystemExit(f"shard {a.shard}/{a.nshards} is empty")
    print(f"[curve] shard {a.shard}/{a.nshards}: {len(ckpts)} checkpoints: "
          f"{[s for s, _ in ckpts]}", flush=True)

    device = torch.device(a.device)
    # Build ONCE from the first checkpoint (this constructs the baked DQC verifier / JAX).
    policy, verifier, To, chunk_h, image_key, known = build_policy(
        ckpts[0][1], "search", device, use_ema=False,
        n_samples=a.n_samples, n_action_steps=a.n_action_steps, instruction=a.instruction)

    # ---- precompute the frame set once ----
    z = zarr.open(a.zarr, "r")
    zobs, acts = z["data"]["obs"], z["data"]["actions"]
    dones = z["data"]["dones"][:]
    ep_ends = np.where(dones)[0]
    ep_starts = np.concatenate([[0], ep_ends[:-1] + 1])
    A = int(acts.shape[-1])

    frames = []   # (obs_dict, a_star)
    for ep in pick_episodes(ep_starts, ep_ends, a.num_episodes):
        s, en = int(ep_starts[ep]), int(ep_ends[ep]) + 1
        for g in range(s, en, a.stride):
            obs = build_obs_dict(zobs, g, s, known, image_key, To, device)
            n = min(chunk_h, en - g)
            a_star = np.full((chunk_h, A), np.nan, np.float32)
            a_star[:n] = np.asarray(acts[g:g + n], np.float32)
            frames.append((obs, a_star))
    print(f"[curve] {len(frames)} frames from {a.num_episodes} demos ({a.split})", flush=True)

    import dill
    K = a.n_samples
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write("step,split,N,best_of_N,qpick_N,best_of_N_norm,qpick_N_norm,q_best_N,q_mean_i0,n_frames\n")

        for si, (step, path) in enumerate(ckpts):
            if si > 0:   # swap weights only; keep policy + verifier + JAX state
                payload = torch.load(open(path, "rb"), pickle_module=dill,
                                     map_location="cpu", weights_only=False)
                sd = payload["state_dicts"]
                state = sd.get("model", sd.get("ema_model"))
                missing = policy.load_state_dict(state, strict=False)
                del payload, sd, state
                if si == 1:
                    print(f"[curve] weight swap ok (missing={len(missing.missing_keys)}, "
                          f"unexpected={len(missing.unexpected_keys)})", flush=True)
            policy.to(device).eval()
            # action normalizer (limits: x*scale+offset -> [-1,1]) for normalized RMSE
            _na = policy.normalizer['action']
            _sc = _na.params_dict['scale'].detach().cpu().numpy().astype(np.float64)
            _of = _na.params_dict['offset'].detach().cpu().numpy().astype(np.float64)

            best = np.zeros((len(frames), K)); qpick = np.zeros_like(best)
            best_n = np.zeros_like(best); qpick_n = np.zeros_like(best)
            qbest = np.zeros_like(best); q0 = np.zeros(len(frames))
            for fi, (obs, a_star) in enumerate(frames):
                torch.manual_seed(1_000_003 * fi + 17)   # fixed candidate noise across ckpts
                with torch.no_grad():
                    acts_t, vals_t = policy.predict_n_actions(obs, verifier, K)
                cand = acts_t[0, :, :chunk_h].cpu().float().numpy()
                vals = vals_t[0].cpu().float().numpy()
                sel_idx = [int(np.argmax(vals[:n + 1])) for n in range(K)]
                r = rmse_to_astar(cand, a_star)                            # raw action units
                rn = rmse_to_astar(cand * _sc + _of, a_star * _sc + _of)   # normalized [-1,1]
                best[fi] = np.minimum.accumulate(r); qpick[fi] = r[sel_idx]
                best_n[fi] = np.minimum.accumulate(rn); qpick_n[fi] = rn[sel_idx]
                qbest[fi] = np.maximum.accumulate(vals)
                q0[fi] = vals[0]

            for n in range(K):
                fh.write(f"{step},{a.split},{n+1},{best[:,n].mean():.6f},"
                         f"{qpick[:,n].mean():.6f},{best_n[:,n].mean():.6f},"
                         f"{qpick_n[:,n].mean():.6f},{qbest[:,n].mean():.5f},"
                         f"{q0.mean():.5f},{len(frames)}\n")
            fh.flush()
            print(f"[curve] step {step:6d}: bo1={best[:,0].mean():.4f} "
                  f"bo{K}={best[:,-1].mean():.4f}  qpick{K}={qpick[:,-1].mean():.4f}",
                  flush=True)

    print(f"[curve] done -> {a.out}")


if __name__ == "__main__":
    main()
