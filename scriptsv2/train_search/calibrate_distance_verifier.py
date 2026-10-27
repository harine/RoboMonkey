#!/usr/bin/env python
"""Calibrate DistanceVerifier against a dataset. Prints the hydra overrides to paste.

Two constants in ``DistanceVerifier`` are dataset properties, not design choices, and
getting either wrong degrades the verifier SILENTLY rather than loudly:

  action_pos_scale   metres of end-effector travel per action unit. The verifier
                     predicts the post-chunk gripper position by integrating the
                     candidate's translation channels; if the unit conversion is off,
                     every candidate's predicted displacement is off by that factor.
                     Too small and all candidates land on top of each other, so the
                     verifier ranks noise. Estimated here by regressing the OBSERVED
                     ee displacement on the commanded action over every transition
                     (least squares through the origin, per axis and pooled).

  value_mean/std     the policy CONDITIONS on verifier values, so their scale is part
                     of the model input. DQC values were O(0-1); raw distances here are
                     negative metres. Estimated as the mean/std of the raw value under
                     the EXPERT action, which is the distribution the context is drawn
                     from at convergence.

Usage:
    python scriptsv2/train_search/calibrate_distance_verifier.py \
        --dataset-dir /gscratch/robotics/harine/data/strict_distance_success_sub50_diverse_start \
        --horizon 8
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import zarr


def _open(dataset_dir: str):
    roots = sorted(glob.glob(os.path.join(dataset_dir, "*.zarr")))
    if not roots:
        raise SystemExit(f"no *.zarr under {dataset_dir}")
    return zarr.open(roots[0], mode="r")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--ee-key", default="end_effector_pose")
    ap.add_argument("--object-key", default="insertive_asset_pose")
    ap.add_argument("--target-key", default="receptive_asset_pose")
    ap.add_argument("--grasp-key", default="last_gripper_action")
    # Measured, not assumed: last_gripper_action is {0.0, 0.996} on this data, so
    # "closed" is POSITIVE and 0.5 splits it. The Bridge gripper<0 convention gives a
    # grasped fraction of exactly 0.000 here, which freezes the object and makes the
    # placement distance constant across candidates — a silent, not a loud, failure.
    ap.add_argument("--grasp-threshold", type=float, default=0.5)
    ap.add_argument("--grasp-closed-is-negative", action="store_true")
    ap.add_argument("--json-out", default=None)
    # --fit horizon_cumulative regresses the H-step cumulative displacement on the summed
    # commanded translation (what a chunk-endpoint verifier such as VeritasPixelVerifier
    # integrates); the default regresses single steps (what DistanceVerifier was
    # calibrated with). --frame base rotates observed world displacements into the robot
    # base frame first (needs --camera-file for the base pose), giving a positive scale
    # to combine with R_world_from_base; world keeps the raw world-frame fit.
    ap.add_argument("--fit", choices=("step", "horizon_cumulative"), default="step")
    ap.add_argument("--frame", choices=("world", "base"), default="world")
    ap.add_argument("--camera-file", default=None,
                    help="diffusion_policy/.../config/veritas/camera_<task>.json (for --frame base)")
    args = ap.parse_args()

    root = _open(args.dataset_dir)
    obs = root["data/obs"]
    actions = np.asarray(root["data/actions"])
    ends = np.asarray(root["meta/episode_ends"])
    ee = np.asarray(obs[args.ee_key])[:, :3]
    objp = np.asarray(obs[args.object_key])[:, :3]
    tgtp = np.asarray(obs[args.target_key])[:, :3]
    grasp = np.asarray(obs[args.grasp_key])[:, :1]
    N = len(actions)
    print(f"steps={N}  episodes={len(ends)}  action_dim={actions.shape[-1]}")

    # ---- action_pos_scale -----------------------------------------------------
    # Transitions only WITHIN an episode: the step across an episode boundary pairs the
    # last frame of one trajectory with the first of the next, whose displacement is a
    # reset, not a response to the action. On 50 episodes of ~24 steps that is 4% of the
    # data and it is pure outlier — enough to drag a least-squares fit noticeably.
    starts = np.concatenate([[0], ends[:-1]])
    valid = np.ones(N, dtype=bool)
    valid[ends - 1] = False          # last frame of each episode has no in-episode successor
    valid[-1] = False
    idx = np.flatnonzero(valid)

    if args.fit == "horizon_cumulative":
        # windows whose full horizon stays inside one episode: displacement over H steps
        # against the sum of the H commanded translations
        Hf = args.horizon
        ep_of_f = np.searchsorted(ends, np.arange(N), side="right")
        idx = np.flatnonzero((np.arange(N) + Hf < N)
                             & (ep_of_f == ep_of_f[np.clip(np.arange(N) + Hf, 0, N - 1)]))
        d_obs = ee[idx + Hf] - ee[idx]
        a_cmd = np.stack([actions[i:i + Hf, :3].sum(0) for i in idx])
    else:
        d_obs = ee[idx + 1] - ee[idx]            # observed displacement
        a_cmd = actions[idx, :3]                 # commanded translation
    if args.frame == "base":
        if not args.camera_file:
            raise SystemExit("--frame base needs --camera-file")
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "diffusion_policy"))
        from diffusion_policy.env.simpler.veritas.camera import CameraModel
        R_wb = CameraModel.load(args.camera_file).R_world_from_base
        d_obs = d_obs @ R_wb          # world -> base: R_bw = R_wb^T, applied to row vectors
    print(f"fit={args.fit} frame={args.frame} windows={len(idx)}")
    # least squares through the origin: scale = <a, d> / <a, a>
    pooled = float((a_cmd * d_obs).sum() / np.clip((a_cmd * a_cmd).sum(), 1e-12, None))
    per_axis = [
        float((a_cmd[:, k] * d_obs[:, k]).sum() / np.clip((a_cmd[:, k] ** 2).sum(), 1e-12, None))
        for k in range(3)
    ]
    ss_tot = float(((d_obs - d_obs.mean(0)) ** 2).sum())
    r2_pooled = 1.0 - float(((d_obs - pooled * a_cmd) ** 2).sum()) / max(ss_tot, 1e-12)
    scale = np.array(per_axis)
    r2_axis = 1.0 - float(((d_obs - a_cmd * scale) ** 2).sum()) / max(ss_tot, 1e-12)
    corr = [float(np.corrcoef(a_cmd[:, k], d_obs[:, k])[0, 1]) for k in range(3)]

    print("\n--- action_pos_scale ---")
    print(f"  PER-AXIS (x,y,z) : {per_axis[0]:.6f} {per_axis[1]:.6f} {per_axis[2]:.6f}   "
          f"R^2 = {r2_axis:.4f}   <- USE THIS")
    print(f"  pooled scalar    : {pooled:.6f}                        R^2 = {r2_pooled:.4f}")
    print(f"  per-axis corr    : {corr[0]:+.3f} {corr[1]:+.3f} {corr[2]:+.3f}")

    # A pooled scalar is not merely less accurate than per-axis here — when the axes
    # disagree in SIGN it averages them toward zero and reports a near-zero fit for a
    # relationship that is actually strong. That is what a single number hid on the
    # eggplant set: R^2 0.03 pooled vs 0.70 per-axis, because x,y are negative (a
    # 180-degree flip about z between the action frame and the pose frame) and z is
    # positive. Judge the model by r2_axis; r2_pooled is printed only to show the trap.
    if len(set(np.sign(per_axis))) > 1:
        print("  NOTE: per-axis scales DISAGREE IN SIGN — the action frame is rotated "
              "relative to the pose frame. A pooled scalar is meaningless here; the "
              "per-axis vector is the only correct form.")
    if r2_axis < 0.5:
        print("  WARNING: even per-axis, the linear delta-EE model explains <50% of "
              "observed motion. Actions may be absolute rather than delta, or pass "
              "through a controller with real dynamics. The verifier assumes "
              "displacement is linear in the commanded action; treat its ranking with "
              "suspicion while this stays low.")

    # ---- value_mean / value_std under the expert action ------------------------
    # Windows whose full horizon lies inside one episode, scored exactly as the verifier
    # scores a candidate — same forward model, expert action in place of the candidate.
    H = args.horizon
    ep_of = np.searchsorted(ends, np.arange(N), side="right")
    ok = np.flatnonzero((np.arange(N) + H < N) & (ep_of == ep_of[np.clip(np.arange(N) + H, 0, N - 1)]))
    delta = np.stack([actions[i:i + H, :3].sum(0) for i in ok]) * scale
    held = ((grasp[ok] < args.grasp_threshold) if args.grasp_closed_is_negative
            else (grasp[ok] > args.grasp_threshold)).astype(np.float64)
    ee_f = ee[ok] + delta
    obj_f = objp[ok] + held * delta
    raw = -(np.linalg.norm(ee_f - obj_f, axis=-1) + np.linalg.norm(obj_f - tgtp[ok], axis=-1))
    print(f"\n--- raw value under the expert action ({len(ok)} windows, H={H}) ---")
    print(f"  mean {raw.mean():.6f}   std {raw.std():.6f}   "
          f"min {raw.min():.4f}   max {raw.max():.4f}")
    print(f"  grasped fraction : {held.mean():.3f}")
    if held.mean() < 0.05:
        print("  WARNING: the gripper reads as closed on <5% of windows, so the object "
              "almost never moves with it and the placement term is near-constant. Check "
              "--grasp-key / --grasp-threshold before trusting the second distance term.")

    out = {
        "fit": args.fit, "frame": args.frame,
        "action_pos_scale": per_axis,          # the per-axis vector IS the constant
        "action_pos_scale_pooled": pooled,     # kept only for diagnostics
        "action_pos_scale_r2": r2_axis,
        "action_pos_scale_r2_pooled": r2_pooled,
        "action_pos_corr": corr,
        "value_mean": float(raw.mean()),
        "value_std": float(raw.std()),
        "horizon": H,
        "n_windows": int(len(ok)),
        "grasped_fraction": float(held.mean()),
        "dataset_dir": args.dataset_dir,
    }
    print("\n--- paste into the launcher ---")
    print(f"  ACTION_POS_SCALE='[{per_axis[0]:.6f},{per_axis[1]:.6f},{per_axis[2]:.6f}]'")
    print(f"  VALUE_MEAN={raw.mean():.6f}")
    print(f"  VALUE_STD={raw.std():.6f}")
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
