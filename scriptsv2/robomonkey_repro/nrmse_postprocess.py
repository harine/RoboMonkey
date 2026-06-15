#!/usr/bin/env python
"""Recompute action-error metrics from a cached candidate pool (raw_pool.npz).

NO GPU / NO re-sampling: reads the F x M x 7 pool of candidate actions, their
verifier rewards, and the reference (executed) action per frame, then computes:

  * NORMALIZED RMSE (NRMSE): z-score each of the 7 action dims by the dataset's
    per-dim std before the RMSE, so dims with larger raw range (e.g. gripper)
    don't dominate the 7-D number. (`--norm range` uses q99-q01 bounds instead.)

  * SHUFFLED-POOL selection (RoboMonkey Fig.8 protocol): instead of "best of the
    FIRST N", take the expected best-of-N over a uniformly random size-N subset
    of the M-candidate pool. We use the EXACT closed form: rank candidates by
    reward (0=best); the rank-j candidate wins a random size-N subset with
    probability w_j(N) = C(M-1-j, N-1) / C(M, N) (it must be drawn AND all N-1
    others drawn from the M-1-j worse ones). So E[error](N) = sum_j w_j * err_j.
    N=1 -> uniform 1/M (random pick); N=M -> the top-1. No Monte-Carlo noise.

  * FIRST-N (argmax over rewards[:N]) and UNNORMALIZED variants, for comparison.

  * Per-dimension NRMSE(N) for all 7 dims, to see exactly which dims scale.

Selection is absolute argmax of the verifier's scalar reward -- matching the
deployed RoboMonkey verifier (run_simpler_eval does np.argmax(rewards); the
Bradley-Terry pairwise loss is training-time only).

Usage:
  python scriptsv2/robomonkey_repro/nrmse_postprocess.py \
      --raw data/eval/robomonkey_action_error/M64_f2000_s0/raw_pool.npz
"""
import argparse, json, os
from math import comb
import numpy as np
import zarr

DIM_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip"]


def norm_stats(dataset, mode):
    """Per-dim normalization scale from the full dataset's executed actions."""
    g = zarr.open(dataset, mode="r")
    a = np.asarray(g["data"]["actions"][:], dtype=np.float64)   # (T, 7)
    mean = a.mean(0)
    std = a.std(0)
    q01, q99 = np.quantile(a, 0.01, 0), np.quantile(a, 0.99, 0)
    if mode == "std":
        scale = std.copy()
    elif mode == "range":
        scale = (q99 - q01)
    else:
        raise ValueError(mode)
    scale[scale < 1e-8] = 1.0          # guard degenerate dims
    return scale, dict(mean=mean, std=std, q01=q01, q99=q99)


def shuffled_weights(M, N):
    """w_j(N) = C(M-1-j, N-1)/C(M, N) for rank j=0..M-1 (0=best reward)."""
    denom = comb(M, N)
    w = np.array([comb(M - 1 - j, N - 1) if (M - 1 - j) >= (N - 1) else 0
                  for j in range(M)], dtype=np.float64)
    return w / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw_pool.npz from eval_action_error.py")
    ap.add_argument("--dataset",
        default="/gscratch/robotics/harine/data/eggplant_in_basket_val_success/val_success.zarr",
        help="source of per-dim normalization stats")
    ap.add_argument("--norm", choices=["std", "range"], default="std")
    ap.add_argument("--out", default=None, help="output json (default next to --raw)")
    args = ap.parse_args()

    d = np.load(args.raw)
    acts, rewards, ref = d["actions"], d["rewards"], d["ref"]   # (F,M,7)(F,M)(F,7)
    F, M, A = acts.shape
    Ns = [1 << k for k in range(int(np.log2(M)) + 1)]
    scale, raw_stats = norm_stats(args.dataset, args.norm)
    print(f"[cfg] raw={args.raw}")
    print(f"[cfg] F={F} frames  M={M} candidates  norm={args.norm}")
    print(f"[cfg] per-dim scale ({args.norm}): "
          + "  ".join(f"{n}={s:.4f}" for n, s in zip(DIM_NAMES, scale)))

    # per-(frame, candidate, dim) absolute error, normalized
    err = np.abs(acts - ref[:, None, :]) / scale[None, None, :]    # (F, M, 7)
    sq = err ** 2
    # per-candidate group errors (RMS across the group's dims)
    grp = {
        "overall_7d": np.sqrt(sq.mean(axis=2)),                    # (F, M)
        "arm_6d":     np.sqrt(sq[:, :, :6].mean(axis=2)),
        "gripper_1d": err[:, :, 6],
    }
    for k in range(A):
        grp[f"dim_{DIM_NAMES[k]}"] = err[:, :, k]

    # rank candidates by reward (desc) per frame; reorder group errors by rank
    order = np.argsort(-rewards, axis=1, kind="stable")           # (F, M), 0=best
    rows = np.arange(F)[:, None]

    def agg(per_frame_vals):
        v = np.asarray(per_frame_vals)
        return {"mean": float(v.mean()), "sem": float(v.std() / np.sqrt(len(v)))}

    results = {"shuffled": {}, "firstN": {}}
    for gname, gvals in grp.items():
        gv_ranked = gvals[rows, order]            # (F, M) ranked best->worst
        # SHUFFLED-POOL: exact expectation E[err](N) = sum_j w_j(N) * err_rankj
        sh = {}
        for N in Ns:
            w = shuffled_weights(M, N)            # (M,)
            per_frame = gv_ranked @ w             # (F,)
            sh[str(N)] = agg(per_frame)
        results["shuffled"][gname] = sh
        # FIRST-N: argmax over rewards[:N] in original order
        fn = {}
        for N in Ns:
            sel = np.argmax(rewards[:, :N], axis=1)    # (F,)
            per_frame = gvals[rows[:, 0], sel]
            fn[str(N)] = agg(per_frame)
        results["firstN"][gname] = fn

    out = {"raw": args.raw, "dataset": args.dataset, "norm": args.norm,
           "F": F, "M": M, "Ns": Ns, "dim_names": DIM_NAMES,
           "scale": scale.tolist(),
           "norm_stats": {k: v.tolist() for k, v in raw_stats.items()},
           "selection": "absolute argmax of verifier scalar reward",
           "metrics": results}
    out_path = args.out or os.path.join(os.path.dirname(args.raw),
                                        f"nrmse_{args.norm}.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)

    # ---- pretty print ------------------------------------------------------
    def table(title, group):
        sh = results["shuffled"][group]
        fn = results["firstN"][group]
        b_sh = sh[str(Ns[0])]["mean"]
        print(f"\n  {title}")
        print(f"    {'N':>4} | {'shuffled':>9} {'±sem':>7} {'vsN=1':>7} | "
              f"{'firstN':>9} {'±sem':>7}")
        print("    " + "-" * 54)
        for N in Ns:
            s, fnn = sh[str(N)], fn[str(N)]
            dl = (s["mean"] - b_sh) / b_sh * 100
            print(f"    {N:>4} | {s['mean']:>9.4f} {s['sem']:>7.4f} {dl:>+6.1f}% | "
                  f"{fnn['mean']:>9.4f} {fnn['sem']:>7.4f}")

    print(f"\n{'='*70}")
    print(f" NORMALIZED RMSE (norm={args.norm}, z-scored per dim)  F={F}  M={M}")
    print(f" reference = executed successful action;  shuffled = exact E over size-N subsets")
    print('='*70)
    table("NRMSE  overall (7D)", "overall_7d")
    table("NRMSE  arm (6D)", "arm_6d")
    table("NRMSE  gripper (1D)", "gripper_1d")

    # per-dim compact view (shuffled only)
    print(f"\n  per-dimension NRMSE (shuffled-pool), N=1 -> N={M}:")
    print(f"    {'dim':>7} | " + " ".join(f"{('N='+str(N)):>8}" for N in Ns) + " |  %vsN1")
    print("    " + "-" * (12 + 9 * len(Ns) + 9))
    for k in range(A):
        sh = results["shuffled"][f"dim_{DIM_NAMES[k]}"]
        vals = [sh[str(N)]["mean"] for N in Ns]
        dl = (vals[-1] - vals[0]) / vals[0] * 100 if vals[0] else 0.0
        print(f"    {DIM_NAMES[k]:>7} | " + " ".join(f"{v:>8.4f}" for v in vals)
              + f" | {dl:>+6.1f}%")
    print('='*70)
    print(f"[done] -> {out_path}")


if __name__ == "__main__":
    main()
