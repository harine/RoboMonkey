#!/usr/bin/env python3
"""Three diagnostics that turn "the DQC chunk verifier is a poor ranker for BC-H8"
from inference into a mechanism proof. Reads VIZ_Q dumps (search_q/*.npz) produced by
eval_search.py for a BC-H8 rollout set and a diffusion-search rollout set, plus the
eggplant demo chunks as the in-distribution reference.

EXP 1  Correlation probe (decisive). Score each rollout's chunks with the critic and
       correlate critic value with episode outcome. Prediction: corr<=0 for BC,
       corr>0 for search. Same critic, opposite sign => proposal-distribution-dependent,
       not a critic bug.
EXP 2  Joint-support diagnostic. Marginals match (already shown). Measure the JOINT:
       chunk smoothness (step-to-step delta magnitude) and log-likelihood under a
       full-covariance Gaussian fit on demo chunks. Prediction: BC chunks are less
       smooth / lower-LL than search & demos despite in-bounds marginals.
EXP 3  Argmax-selection OOD test. Are the critic's argmax-preferred candidates more OOD
       (jerkier / lower-LL) than the candidate pool? If the critic's favorites are the
       weird ones, that's the overestimation mechanism caught red-handed.

Usage:
  python verifier_mechanism_probe.py \
     --bc data/eval/_probe/bc_h8_vizq/search_q \
     --search data/eval/_probe/diff10k_vizq/search_q
"""
import argparse, glob, os
import numpy as np

ARM = slice(0, 6)          # 6 arm dims; gripper (dim 6) is bang-bang -> excluded from smoothness
ZARRS = ["/gscratch/robotics/harine/data/eggplant_in_basket/state0.zarr",
         "/gscratch/robotics/harine/data/eggplant_in_basket/state1.zarr"]


# ---------- metrics ----------
def smoothness(chunk):
    """Mean L2 of consecutive arm-action deltas over a chunk (S,7). Higher = jerkier."""
    a = chunk[:, ARM]
    if a.shape[0] < 2:
        return np.nan
    d = np.diff(a, axis=0)                      # (S-1, 6)
    return float(np.mean(np.linalg.norm(d, axis=1)))


def load_npz_dir(d):
    eps = []
    for p in sorted(glob.glob(os.path.join(d, "ep*.npz"))):
        z = np.load(p)
        eps.append(dict(
            success=int(z["success"]),
            cand=z["candidate_actions"].astype(np.float32),   # (R,K,S,7)
            vals=z["values"].astype(np.float32),              # (R,K)
            sel=z["selected_index"].astype(int),              # (R,)
        ))
    return eps


def demo_windows(S, max_n=20000):
    """Raw S-step arm-action windows from the eggplant demos (in-distribution ref)."""
    import zarr
    W = []
    for zp in ZARRS:
        if not os.path.isdir(zp):
            continue
        g = zarr.open(zp, "r")
        acts = g["data"]["actions"][:]                        # (T,7)
        ends = list(g["meta"]["episode_ends"][:])
        start = 0
        for e in ends:
            ep = acts[start:e]
            for t in range(0, len(ep) - S + 1):
                W.append(ep[t:t + S])
            start = e
            if len(W) >= max_n:
                break
        if len(W) >= max_n:
            break
    return np.asarray(W[:max_n], dtype=np.float32)            # (N,S,7)


# ---------- joint Gaussian on demo chunks (arm dims) ----------
def fit_gauss(windows):
    X = windows[:, :, ARM].reshape(len(windows), -1)          # (N, S*6)
    mu = X.mean(0)
    cov = np.cov(X, rowvar=False) + 1e-6 * np.eye(X.shape[1])
    sign, logdet = np.linalg.slogdet(cov)
    inv = np.linalg.inv(cov)
    k = X.shape[1]
    const = -0.5 * (k * np.log(2 * np.pi) + logdet)

    def ll(chunks):                                            # chunks (M,S,7)
        Y = chunks[:, :, ARM].reshape(len(chunks), -1)
        d = Y - mu
        maha = np.einsum("ij,jk,ik->i", d, inv, d)
        return const - 0.5 * maha
    return ll


# ---------- experiments ----------
def point_biserial(values, binary):
    v = np.asarray(values, float); b = np.asarray(binary, int)
    if len(np.unique(b)) < 2 or np.std(v) == 0:
        return float("nan")
    return float(np.corrcoef(v, b)[0, 1])


def auc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney U / (|pos||neg|)
    order = np.argsort(s); ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def exp1_correlation(eps, name):
    # per-episode mean critic value (all candidates AND selected) vs success
    mean_all, mean_sel, succ = [], [], []
    step_sel_val, step_succ = [], []
    for e in eps:
        mean_all.append(float(e["vals"].mean()))
        sv = e["vals"][np.arange(len(e["sel"])), e["sel"]]    # selected value per replan
        mean_sel.append(float(sv.mean()))
        succ.append(e["success"])
        step_sel_val += list(sv); step_succ += [e["success"]] * len(sv)
    print(f"\n=== EXP1 correlation — {name}  (episodes={len(eps)}, succ={sum(succ)}/{len(succ)}) ===")
    print(f"  per-episode mean ALL-candidate value vs success : r={point_biserial(mean_all, succ):+.3f}  AUC={auc(mean_all, succ):.3f}")
    print(f"  per-episode mean SELECTED value     vs success  : r={point_biserial(mean_sel, succ):+.3f}  AUC={auc(mean_sel, succ):.3f}")
    print(f"  per-replan-step SELECTED value      vs ep-success: r={point_biserial(step_sel_val, step_succ):+.3f}  AUC={auc(step_sel_val, step_succ):.3f}")
    succ_v = np.array([m for m, s in zip(mean_all, succ) if s == 1])
    fail_v = np.array([m for m, s in zip(mean_all, succ) if s == 0])
    if len(succ_v) and len(fail_v):
        print(f"  mean value:  SUCCESS={succ_v.mean():+.3f}   FAIL={fail_v.mean():+.3f}   (success-fail={succ_v.mean()-fail_v.mean():+.3f})")
    return dict(r_all=point_biserial(mean_all, succ), auc_all=auc(mean_all, succ))


def exp2_joint(eps, name, demo_ll, demo_smooth):
    # smoothness of selected chunks vs demos
    sm = []
    ll_chunks = []
    for e in eps:
        sel = e["cand"][np.arange(len(e["sel"])), e["sel"]]   # (R,S,7) selected chunks
        for c in sel:
            sm.append(smoothness(c))
            ll_chunks.append(c)
    sm = np.array(sm, float)
    lls = demo_ll(np.asarray(ll_chunks, np.float32))
    print(f"\n=== EXP2 joint-support — {name} ===")
    print(f"  chunk smoothness (mean |Δarm|, higher=jerkier): {np.nanmean(sm):.4f}   [demo ref={demo_smooth:.4f}]")
    print(f"  log-lik under demo joint Gaussian (higher=in-support): mean={lls.mean():+.1f}  median={np.median(lls):+.1f}")
    return dict(smooth=float(np.nanmean(sm)), ll=float(lls.mean()))


def exp3_argmax_ood(eps, name, demo_ll):
    # per replan step: argmax(value) candidate's smoothness vs pool mean; value~smoothness corr
    argmax_sm, pool_sm = [], []
    argmax_ll, pool_ll = [], []
    val_smooth_corr = []
    for e in eps:
        for r in range(e["cand"].shape[0]):
            cands = e["cand"][r]                              # (K,S,7)
            vals = e["vals"][r]                               # (K,)
            sms = np.array([smoothness(c) for c in cands])
            lls = demo_ll(cands)
            ai = int(np.argmax(vals))
            argmax_sm.append(sms[ai]); pool_sm.append(np.nanmean(sms))
            argmax_ll.append(lls[ai]); pool_ll.append(lls.mean())
            if np.std(vals) > 0 and np.std(sms) > 0:
                val_smooth_corr.append(np.corrcoef(vals, sms)[0, 1])
    argmax_sm = np.array(argmax_sm); pool_sm = np.array(pool_sm)
    argmax_ll = np.array(argmax_ll); pool_ll = np.array(pool_ll)
    print(f"\n=== EXP3 argmax-selection OOD — {name} ===")
    print(f"  smoothness:  argmax-pick={np.nanmean(argmax_sm):.4f}  pool-mean={np.nanmean(pool_sm):.4f}  "
          f"Δ={np.nanmean(argmax_sm-pool_sm):+.4f}  ({'JERKIER' if np.nanmean(argmax_sm-pool_sm)>0 else 'smoother'} than pool)")
    print(f"  demo-LL:     argmax-pick={argmax_ll.mean():+.1f}  pool-mean={pool_ll.mean():+.1f}  "
          f"Δ={(argmax_ll-pool_ll).mean():+.1f}  ({'more OOD' if (argmax_ll-pool_ll).mean()<0 else 'more in-support'})")
    print(f"  within-step corr(critic value, jerk) across candidates: mean={np.nanmean(val_smooth_corr):+.3f}  "
          f"({'critic REWARDS jerk' if np.nanmean(val_smooth_corr)>0 else 'critic penalizes jerk'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc", required=True)
    ap.add_argument("--search", required=True)
    args = ap.parse_args()

    bc = load_npz_dir(args.bc)
    se = load_npz_dir(args.search)
    if not bc or not se:
        print(f"[probe] BC dumps={len(bc)}  search dumps={len(se)} — need both. Wait for jobs.")
        return
    S = bc[0]["cand"].shape[2]
    demos = demo_windows(S)
    demo_ll = fit_gauss(demos)
    demo_smooth = float(np.nanmean([smoothness(w) for w in demos[:4000]]))
    print(f"[probe] BC eps={len(bc)}  search eps={len(se)}  demo windows={len(demos)} (S={S})")

    exp1_correlation(bc, "BC-H8")
    exp1_correlation(se, "diffusion-search")
    exp2_joint(bc, "BC-H8", demo_ll, demo_smooth)
    exp2_joint(se, "diffusion-search", demo_ll, demo_smooth)
    exp3_argmax_ood(bc, "BC-H8", demo_ll)
    exp3_argmax_ood(se, "diffusion-search", demo_ll)


if __name__ == "__main__":
    main()
