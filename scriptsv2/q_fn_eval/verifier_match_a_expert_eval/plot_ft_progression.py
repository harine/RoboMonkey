#!/usr/bin/env python
"""plot_ft_progression.py — track verifier quality of a DQC finetune across its
saved checkpoints. For each `params_<step>.pkl` in a finetune run dir, re-score the
cached offline candidate logs (no rollouts) and compute, vs training step:
  * train a*-Q AUC (success vs failure episodes, per-episode mean Q(a*))  [verifier quality]
  * advantage A = Q(a*) - Q(a_pred)  (mean + frac>0)                       [expert vs deployed]
  * val best-of-N RMSE-to-a* (argmax-Q candidate)                          [selection quality]
The base (non-finetuned) DQC value is drawn as a horizontal reference using the
`values`/`a_star_q` already cached in the source NPZs.

Runs in dqc_jax. Usage:
  python plot_ft_progression.py --ft-run-dir data/viz_q/ft/dqc/v2_all/v2_all \
     --label v2_all --src-root data/viz --out data/viz_q/_prog \
     --configs bc_h8 diff10k diff33k gauss20k gauss50k
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "diffusion_policy"))
DEFAULT_TEXT_EMB = str(_REPO / "scriptsv2" / "q_fn_eval" / "task_emb_use3.npy")
DEFAULT_INSTR = "put the eggplant in the basket"


def _auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def _rmse_to_astar(cand, a_star):
    diff = cand.astype(np.float64) - a_star[None].astype(np.float64)
    mask = ~np.isnan(a_star)[None]
    se = np.where(mask, diff ** 2, np.nan)
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.nanmean(se.reshape(se.shape[0], -1), axis=1))


def load_split(src_root, configs, split):
    """Return list of per-episode dicts across configs for a split."""
    eps = []
    for cfg in configs:
        sq = Path(src_root) / cfg / split / "search_q"
        for p in sorted(sq.glob("ep*.npz")):
            z = np.load(p, allow_pickle=False)
            eps.append(dict(
                cfg=cfg, path=str(p),
                cand=z["candidate_actions"], frames=z["frames"],
                a_star=z["a_star"], sel=z["selected_index"],
                success=bool(int(z["success"])),
                base_values=z["values"], base_aq=z["a_star_q"],
            ))
    return eps


def metrics_from_scored(eps, get_vals):
    """get_vals(ep) -> (values (R,K), aq (R,)). Compute AUC/advantage/RMSE."""
    A_sel, succ_epQ, fail_epQ = [], [], []
    val_rmse, top1 = [], []
    for ep in eps:
        v, aq = get_vals(ep)
        sel = ep["sel"]
        # advantage A = Q(a*) - Q(deployed pick) per frame
        for r in range(v.shape[0]):
            if np.isfinite(aq[r]):
                A_sel.append(aq[r] - v[r, int(sel[r])])
        # per-episode mean a*-Q for success/failure AUC
        m = float(np.nanmean(aq))
        (succ_epQ if ep["success"] else fail_epQ).append(m)
        # selection RMSE-to-a* (argmax-Q candidate)
        for r in range(v.shape[0]):
            rmse = _rmse_to_astar(ep["cand"][r], ep["a_star"][r])
            pick = int(np.argmax(v[r]))
            val_rmse.append(rmse[pick])
            top1.append(int(pick == int(np.nanargmin(rmse))))
    A_sel = np.asarray(A_sel, float)
    return dict(
        auc=_auc(succ_epQ, fail_epQ),
        A_mean=float(A_sel.mean()) if A_sel.size else float("nan"),
        fracA=float((A_sel > 0).mean()) if A_sel.size else float("nan"),
        bon_rmse=float(np.nanmean(val_rmse)) if val_rmse else float("nan"),
        top1=float(np.mean(top1)) if top1 else float("nan"),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ft-run-dir", required=True, help="dir with params_<step>.pkl + flags.json")
    ap.add_argument("--label", required=True)
    ap.add_argument("--src-root", default="data/viz")
    ap.add_argument("--configs", nargs="+",
                    default=["bc_h8", "diff10k", "diff33k", "gauss20k", "gauss50k"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, nargs="*", default=None, help="default: all saved")
    ap.add_argument("--image-key", default="agentview_image")
    ap.add_argument("--instruction", default=DEFAULT_INSTR)
    ap.add_argument("--text-emb", default=DEFAULT_TEXT_EMB)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from diffusion_policy.policy.dqc_verifier import DQCVerifier
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    tr = load_split(args.src_root, args.configs, "train")
    va = load_split(args.src_root, args.configs, "val")
    print(f"[prog] {args.label}: train eps={len(tr)} val eps={len(va)}")

    # base reference (cached base-DQC values)
    base_tr = metrics_from_scored(tr, lambda e: (e["base_values"], e["base_aq"]))
    base_va = metrics_from_scored(va, lambda e: (e["base_values"], e["base_aq"]))

    epochs = args.epochs
    if not epochs:
        epochs = sorted(int(re.search(r"params_(\d+)\.pkl", p).group(1))
                        for p in glob.glob(os.path.join(args.ft_run_dir, "params_*.pkl")))
    print(f"[prog] checkpoints: {epochs}")

    def scorer(verifier, chunk_h):
        def score(img_u8, chunks):
            M = chunks.shape[0]
            it = torch.from_numpy(img_u8.astype(np.uint8)).to(device)
            obs = {args.image_key: it.unsqueeze(0).unsqueeze(0).expand(M, 1, *img_u8.shape)}
            a = torch.from_numpy(chunks.astype(np.float32)).to(device)
            return verifier.get_value(obs, a).detach().cpu().numpy().astype(np.float32)

        def get_vals(ep):
            R, K = ep["cand"].shape[0], ep["cand"].shape[1]
            v = np.zeros((R, K), np.float32); aq = np.zeros((R,), np.float32)
            for r in range(R):
                v[r] = score(ep["frames"][r], ep["cand"][r, :, :chunk_h])
                aq[r] = score(ep["frames"][r], ep["a_star"][r][None, :chunk_h])[0]
            return v, aq
        return get_vals

    rows = []
    rows.append(dict(step=-1, **{f"train_{k}": v for k, v in base_tr.items()},
                     **{f"val_{k}": v for k, v in base_va.items()}))  # base = step -1
    for ep_i in epochs:
        verifier = DQCVerifier(
            ckpt_dir=args.ft_run_dir, restore_epoch=ep_i, instruction=args.instruction,
            image_obs_key=args.image_key, text_emb_file=args.text_emb,
            critic_type="chunk", score_full_horizon=True, health_check=False)
        gv = scorer(verifier, int(verifier.chunk_horizon))
        mt = metrics_from_scored(tr, gv)
        mv = metrics_from_scored(va, gv)
        rows.append(dict(step=ep_i, **{f"train_{k}": v for k, v in mt.items()},
                         **{f"val_{k}": v for k, v in mv.items()}))
        print(f"[prog] step {ep_i}: train_auc={mt['auc']:.3f} train_A={mt['A_mean']:+.2f} "
              f"fracA={mt['fracA']:.2f} val_bon_rmse={mv['bon_rmse']:.4f}")
        del verifier

    # CSV
    csv_path = out / f"progression_{args.label}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"[out] {csv_path}")

    # plots: AUC, advantage, val RMSE vs step (base = dashed reference)
    ft = [r for r in rows if r["step"] >= 0]
    steps = [r["step"] for r in ft]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    panels = [("train_auc", "train a*-Q AUC (succ vs fail)", base_tr["auc"], 0.5),
              ("train_A_mean", "advantage Q(a*)-Q(a_pred), train", base_tr["A_mean"], 0.0),
              ("val_bon_rmse", "val best-of-N RMSE-to-a*", base_va["bon_rmse"], None)]
    for ax, (key, title, base_v, ref) in zip(axes, panels):
        ax.plot(steps, [r[key] for r in ft], "-o", color="crimson", label=args.label)
        ax.axhline(base_v, color="grey", ls="--", label=f"base={base_v:.3f}")
        if ref is not None:
            ax.axhline(ref, color="k", ls=":", lw=1, label="chance" if ref == 0.5 else "A=0")
        ax.set_xlabel("finetune step"); ax.set_title(title); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"DQC finetune progression — {args.label}", fontsize=13)
    fig.tight_layout()
    p = out / f"progression_{args.label}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[out] {p}")


if __name__ == "__main__":
    main()
