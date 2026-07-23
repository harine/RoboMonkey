#!/usr/bin/env python
"""rescore_offline_candidates.py — re-score the cached offline candidate logs
(search_q/*.npz from sample_offline_candidates.py) with a given DQC checkpoint, so
the advantage/AUC/RMSE diagnostics can be recomputed for a finetuned Q without any
new rollouts or policy sampling.

For each frame it re-scores the K cached `candidate_actions` chunks + the recorded
`a_star` chunk with the chunk critic (`DQCVerifier.get_value`, score_full_horizon),
and writes a lean NPZ (no frames/cam) with `values` + `a_star_q` under the standard
keys, so plot_advantage_investigation.py / plot_offline_diagnostics.py read it
directly (point them at the rescore dir).

Runs in the dqc_jax env. Usage:
  python rescore_offline_candidates.py --src data/viz/diff33k/val/search_q \
     --out data/viz_q/rescore/v2_all/diff33k/val/search_q \
     --ckpt-dir data/viz_q/ft/dqc/v2_all/v2_all --restore-epoch 50000
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "diffusion_policy"))

DEFAULT_TEXT_EMB = str(_REPO / "scriptsv2" / "q_fn_eval" / "task_emb_use3.npy")
DEFAULT_INSTR = "put the eggplant in the basket"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="source search_q dir (or its parent)")
    ap.add_argument("--out", required=True, help="output search_q dir for rescored NPZs")
    ap.add_argument("--ckpt-dir", required=True, help="DQC ckpt dir (has flags.json + params_<ep>.pkl)")
    ap.add_argument("--restore-epoch", type=int, default=100000)
    ap.add_argument("--image-key", default="agentview_image")
    ap.add_argument("--instruction", default=DEFAULT_INSTR)
    ap.add_argument("--text-emb", default=DEFAULT_TEXT_EMB)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from diffusion_policy.policy.dqc_verifier import DQCVerifier

    src = Path(args.src)
    src = src if src.name == "search_q" else src / "search_q"
    npzs = sorted(src.glob("ep*.npz"))
    if not npzs:
        raise SystemExit(f"no ep*.npz in {src}")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    verifier = DQCVerifier(
        ckpt_dir=args.ckpt_dir, restore_epoch=args.restore_epoch,
        instruction=args.instruction, image_obs_key=args.image_key,
        text_emb_file=args.text_emb, critic_type="chunk",
        score_full_horizon=True, health_check=True,
    )
    chunk_h = int(verifier.chunk_horizon)
    print(f"[rescore] {len(npzs)} eps from {src}  ckpt={args.ckpt_dir}@{args.restore_epoch}  "
          f"chunk_h={chunk_h} -> {out}")

    def score(img_u8: np.ndarray, chunks: np.ndarray) -> np.ndarray:
        """chunks (M, chunk_h, 7) -> Q (M,). img_u8 (H,W,3)."""
        M = chunks.shape[0]
        it = torch.from_numpy(img_u8.astype(np.uint8)).to(device)
        obs = {args.image_key: it.unsqueeze(0).unsqueeze(0).expand(M, 1, *img_u8.shape)}
        a = torch.from_numpy(chunks.astype(np.float32)).to(device)        # (M, chunk_h, 7)
        return verifier.get_value(obs, a).detach().cpu().numpy().astype(np.float32)

    for p in npzs:
        z = np.load(p, allow_pickle=False)
        cand = z["candidate_actions"]        # (R,K,chunk_h,7)
        a_star = z["a_star"]                  # (R,chunk_h,7)
        frames = z["frames"]                  # (R,H,W,3) uint8
        R, K = cand.shape[0], cand.shape[1]
        vals = np.zeros((R, K), np.float32)
        aq = np.zeros((R,), np.float32)
        for r in range(R):
            vals[r] = score(frames[r], cand[r, :, :chunk_h])
            aq[r] = score(frames[r], a_star[r][None, :chunk_h])[0]
        # Pass through every source key (frames, cam_K, cam_T_wc, ee_xyz, seed,
        # truncated, num_steps, mode, ...) so the rescored NPZ is a drop-in for
        # render_q + the plot scripts; overwrite ONLY the finetuned-Q fields.
        out_d = {k: z[k] for k in z.files}
        out_d["values"] = vals
        out_d["a_star_q"] = aq
        sel = np.asarray(out_d.get("selected_index", vals.argmax(1))).astype(np.int32).reshape(-1)
        out_d["selected_index"] = sel
        out_d["selected_value"] = vals[np.arange(R), sel].astype(np.float32)
        np.savez_compressed(out / p.name, **out_d)
        print(f"[rescore]   {p.name}: R={R} K={K}  Qrange=[{vals.min():.2f},{vals.max():.2f}] "
              f"a*Q mean={aq.mean():.2f}")
    print(f"[rescore] done -> {out}")


if __name__ == "__main__":
    main()
