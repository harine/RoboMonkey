"""Verify the viz's action-frame / sign assumption against ground truth.

viz_action_branches.py assumes candidate_actions[..., :3] is a world-frame
xyz delta, integrated by plain cumsum from the recorded EE position. This
script tests that assumption empirically:

For each replan r in an augmented bon_q npz, the env actually executed the
chosen candidate's first `replan_every_n` actions. So:

    predicted_disp[r]  = sum(chosen_candidate_actions[r, :replan_every_n, :3])
    actual_disp[r]     = tcp_world_p[r+1] - tcp_world_p[r]

If the viz's frame/sign is correct these match (per-axis slope ~ +1,
correlation ~ +1). A sign flip shows as slope/correlation ~ -1 on that axis.
A body-vs-world frame mismatch shows as low/mixed per-axis correlation but
similar vector magnitudes (because a rotation preserves length).

Usage:
  python scriptsv2/bon/check_action_frame.py <ep..._aug.npz> [more npz ...]
  python scriptsv2/bon/check_action_frame.py <cell_dir>/bon_q   # all *_aug.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def analyze(npz_path: Path) -> None:
    with np.load(npz_path, allow_pickle=False) as z:
        if "tcp_world_p" not in z.files:
            print(f"[skip] {npz_path.name}: not augmented (no tcp_world_p)")
            return
        cand = z["candidate_actions"].astype(np.float64)        # (R,K,T,7)
        sel = z["selected_index"].astype(int)                   # (R,)
        tcp = z["tcp_world_p"].astype(np.float64)               # (R,3)
        rpn = int(z["bon_replan_every_n_steps"])

    R = min(cand.shape[0], tcp.shape[0])
    if R < 2:
        print(f"[skip] {npz_path.name}: only {R} replan(s)")
        return

    pred = np.zeros((R - 1, 3))
    act = np.zeros((R - 1, 3))
    for r in range(R - 1):
        ksel = sel[r]
        chunk = cand[r, ksel, :rpn, :3]          # (rpn,3) xyz deltas
        pred[r] = chunk.sum(axis=0)
        act[r] = tcp[r + 1] - tcp[r]

    axes = "xyz"
    print(f"\n=== {npz_path.name}  (R={R}, replan_every_n={rpn}) ===")
    print(f"{'axis':>4} {'corr':>7} {'slope':>8} {'pred_std':>9} "
          f"{'act_std':>9}  interpretation")
    for i in range(3):
        p, a = pred[:, i], act[:, i]
        if p.std() < 1e-9 or a.std() < 1e-9:
            print(f"{axes[i]:>4} {'n/a':>7} {'n/a':>8} "
                  f"{p.std():>9.4f} {a.std():>9.4f}  (degenerate)")
            continue
        corr = float(np.corrcoef(p, a)[0, 1])
        slope = float(np.polyfit(p, a, 1)[0])
        if corr > 0.7 and slope > 0:
            verdict = "OK (same dir)"
        elif corr < -0.7:
            verdict = "SIGN FLIPPED"
        else:
            verdict = "MISMATCH (frame? scale? axis-swap?)"
        print(f"{axes[i]:>4} {corr:>7.3f} {slope:>8.3f} "
              f"{p.std():>9.4f} {a.std():>9.4f}  {verdict}")

    # Magnitude check: if it's a pure rotation (body vs world) the per-step
    # vector LENGTHS still match even when per-axis correlation is poor.
    pn = np.linalg.norm(pred, axis=1)
    an = np.linalg.norm(act, axis=1)
    if pn.std() > 1e-9 and an.std() > 1e-9:
        mcorr = float(np.corrcoef(pn, an)[0, 1])
        mratio = float(np.median(an[pn > 1e-6] / pn[pn > 1e-6])) if (pn > 1e-6).any() else float("nan")
        print(f"  |disp| corr={mcorr:.3f}  median(|act|/|pred|)={mratio:.3f}")
        print("  -> high |disp| corr but poor per-axis corr  ==> "
              "rotated frame (likely body/EE-frame delta, not world).")
        print("  -> per-axis corr ~ +1, slope ~ +1            ==> "
              "viz assumption correct.")
        print("  -> slope ~ -1 on an axis                      ==> "
              "that axis sign is flipped in viz.")
        print(f"  -> median ratio != 1 ({mratio:.2f})           ==> "
              "action scale/units differ (viz lengths off, dirs maybe OK).")


def search_transforms(npz_path: Path) -> None:
    """Brute-force the sign/permutation (and body->world) transform that best
    maps integrated chosen-candidate deltas onto actual EE displacement."""
    import itertools

    with np.load(npz_path, allow_pickle=False) as z:
        if "tcp_world_p" not in z.files:
            return
        cand = z["candidate_actions"].astype(np.float64)
        sel = z["selected_index"].astype(int)
        tcp = z["tcp_world_p"].astype(np.float64)
        rpn = int(z["bon_replan_every_n_steps"])
        tcp_q = z["tcp_world_q"].astype(np.float64) if "tcp_world_q" in z.files else None

    R = min(cand.shape[0], tcp.shape[0])
    if R < 3:
        return
    raw = np.zeros((R - 1, 3))
    act = np.zeros((R - 1, 3))
    for r in range(R - 1):
        raw[r] = cand[r, sel[r], :rpn, :3].sum(axis=0)
        act[r] = tcp[r + 1] - tcp[r]

    def score(P):  # mean abs per-axis correlation after transform P (3x3)
        pr = raw @ P.T
        cs = []
        for i in range(3):
            if pr[:, i].std() < 1e-9 or act[:, i].std() < 1e-9:
                return -1
            cs.append(np.corrcoef(pr[:, i], act[:, i])[0, 1])
        return float(np.mean(cs))  # want high & positive

    best = []
    perms = list(itertools.permutations(range(3)))
    signs = list(itertools.product([1, -1], repeat=3))
    for perm in perms:
        for sg in signs:
            P = np.zeros((3, 3))
            for i, (pj, s) in enumerate(zip(perm, sg)):
                P[i, pj] = s
            sc = score(P)
            pr = raw @ P.T
            ratio = np.median(
                np.linalg.norm(act, axis=1)[np.linalg.norm(pr, axis=1) > 1e-6]
                / np.linalg.norm(pr, axis=1)[np.linalg.norm(pr, axis=1) > 1e-6]
            )
            best.append((sc, perm, sg, ratio))
    best.sort(key=lambda x: -x[0])
    print(f"\n--- transform search: {npz_path.name} ---")
    print(f"{'meanCorr':>9} {'perm(x<-,y<-,z<-)':>18} {'signs':>10} {'scale(act/pred)':>16}")
    for sc, perm, sg, ratio in best[:5]:
        print(f"{sc:>9.3f} {str(perm):>18} {str(sg):>10} {ratio:>16.3f}")
    print("  perm = which RAW axis feeds out-axis (x,y,z); signs applied after.")


def iter_npz(target: Path):
    if target.is_file():
        yield target
    elif target.is_dir():
        yield from sorted(target.glob("*_aug.npz"))


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    any_found = False
    for arg in argv:
        for f in iter_npz(Path(arg)):
            any_found = True
            analyze(f)
            search_transforms(f)
    if not any_found:
        print("no *_aug.npz inputs found", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
