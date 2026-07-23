#!/usr/bin/env python
"""filtered_score_dqc.py — score the 3-group pairs with the DQC critics.

Run in the ``dqc_jax`` conda env. Loads the REAL BridgeDQCChunkAgent from the
checkpoint (faithful forward, same as eval_dqc_critics.py / dqc_verifier.py) and
scores three variants of the SAME checkpoint per (s, a):

  q_action  single-step action critic   Q(s, a_t)             forward_action_critic
  q_chunk1  chunk critic, 1-step chunk   Q(s, [a_t, 0,..,0])  score_chunks (slot 0 only)
  q_chunk8  chunk critic, 8-step chunk   Q(s, a_{t:t+8})      score_chunks (full chunk)

The only DQC checkpoint has backup_horizon=8, so "chunk-8" vs "chunk-1" both come
from the chunk critic head — chunk-1 fills slot 0 with the action and zero-fills
slots 1..7 in NORMALIZED space (identical to how training assembled the terminal
chunks of an episode: an action followed by the infinite-horizon zero pad).
q_action is the genuinely-distinct single-step action-critic head (reported too,
since "chunk-1" can be read either way).

Reads ``pairs.npz`` (written by filtered_sample_pairs.py). Writes
``dqc_scores_filtered.csv`` row-aligned with pairs.npz order:
  row, ep_uid, group, success, branch_t, q_action, q_chunk1, q_chunk8

Usage: see run_filtered_verifier_comp.sbatch.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_TEXT_EMB = str(_REPO / "scriptsv2" / "q_fn_eval" / "task_emb_use3.npy")

# Bridge action normalization (matches action_normalization.json / dqc bridge stats).
LOW = np.array([-0.05, -0.05, -0.05, -0.25, -0.25, -0.25, 0.0], np.float32)
HIGH = np.array([0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1.0], np.float32)
CLIP_EPS = 1e-5
GROUP_NAMES = {0: "filtered_success", 1: "other_success", 2: "failure"}


def normalize(a):
    n = 2.0 * (a - LOW) / (HIGH - LOW) - 1.0
    return np.clip(n, -1.0 + CLIP_EPS, 1.0 - CLIP_EPS)


def load_text_emb(text_emb_file: str, instruction: str) -> np.ndarray:
    """Load the precomputed 512-d MUSE (USE-3) embedding (same one dqc_verifier.py
    uses) — avoids importing tensorflow_text live, which dqc_jax lacks."""
    p = Path(text_emb_file)
    if p.is_file():
        emb = np.load(p).astype(np.float32)
        if emb.shape == (512,):
            print(f"[dqc] loaded precomputed MUSE (USE-3) embedding from {p}")
            return emb
        print(f"[dqc] {p} has shape {emb.shape}, expected (512,); encoding live ...")
    sys.path.insert(0, str(_REPO / "dqc"))
    from utils.text_processing import make_text_processor
    tp = make_text_processor("muse_embedding")
    emb = np.asarray(tp.encode([instruction]), np.float32)[0]
    assert emb.shape == (512,), emb.shape
    return emb


def make_agent(ckpt_dir: str, image_hw, restore_epoch: int):
    sys.path.insert(0, str(_REPO / "dqc"))
    from ml_collections import ConfigDict
    from agents.bridge_dqc_chunk import BridgeDQCChunkAgent
    from utils.flax_utils import restore_agent

    cfg = ConfigDict(json.load(open(Path(ckpt_dir) / "flags.json"))["agent"])
    H = int(cfg["backup_horizon"]); A = 7; Hh, Ww = image_hw
    ex = {
        "observations": {"image": np.zeros((1, Hh, Ww, 3), np.uint8)},
        "next_observations": {"image": np.zeros((1, Hh, Ww, 3), np.uint8)},
        "high_value_next_observations": {"image": np.zeros((1, Hh, Ww, 3), np.uint8)},
        "goals": {"language": np.zeros((1, 512), np.float32)},
        "high_value_goals": {"language": np.zeros((1, 512), np.float32)},
        "actions": np.zeros((1, A), np.float32),
        "high_value_action_chunks": np.zeros((1, H * A), np.float32),
        "rewards": np.zeros((1,), np.float32), "masks": np.ones((1,), np.float32),
        "high_value_rewards": np.zeros((1,), np.float32), "high_value_masks": np.ones((1,), np.float32),
        "high_value_backup_horizon": np.full((1,), H, np.float32),
        "valids": np.ones((1, H), np.float32), "mc_returns": np.zeros((1,), np.float32),
    }
    agent = BridgeDQCChunkAgent.create(0, ex, cfg)
    agent = restore_agent(agent, ckpt_dir, restore_epoch)
    return agent, H


def per_episode_auc(ep_uid, success, metric):
    """AUC = P(success-ep mean > failure-ep mean) over the two-class success label
    (= the 'before filtering' success-vs-failure baseline). Returns (auc, mu_s, mu_f)."""
    per, succ_ep = {}, {}
    for u, s, m in zip(ep_uid, success, metric):
        per.setdefault(int(u), []).append(float(m)); succ_ep[int(u)] = bool(s)
    means = {k: float(np.mean(v)) for k, v in per.items()}
    pos = np.array([means[k] for k in means if succ_ep[k]])
    neg = np.array([means[k] for k in means if not succ_ep[k]])
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan"), float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size), pos.mean(), neg.mean()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ckpt-dir",
                    default="/gscratch/robotics/harine/dqc_checkpoint/sd000s_137913.0.20260605_163017")
    ap.add_argument("--restore-epoch", type=int, default=100000)
    ap.add_argument("--instruction", default="put the eggplant in the basket")
    ap.add_argument("--text-emb", default=DEFAULT_TEXT_EMB,
                    help="precomputed (512,) MUSE USE-3 embedding for the instruction")
    ap.add_argument("--q-agg", default="min", choices=["min", "mean"])
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    out = Path(args.out_dir)
    d = np.load(out / "pairs.npz", allow_pickle=True)
    images = d["images"]                                    # (G,H,W,3) uint8
    actions = d["actions"].astype(np.float32)              # (G,7) raw
    chunks_raw = d["action_chunks"].astype(np.float32)     # (G,8,7) raw, NaN pad
    ep_uid = d["ep_uid"]; branch_t = d["branch_t"]
    group = d["group"]; success = d["success"].astype(bool)
    instruction = args.instruction or str(d["instruction"])
    G = len(actions)

    # ---- normalize (same as training chunk assembly) -------------------------
    single_norm = normalize(actions)                       # (G,7)
    cn = normalize(chunks_raw)                              # (G,8,7); NaN stays NaN
    cn = np.where(np.isnan(chunks_raw), 0.0, cn).astype(np.float32)
    chunk8_norm = cn.reshape(G, -1)                        # (G,56) time-major, 8 real (NaN->0)

    # chunk-1: action in slot 0, zero pad slots 1..7 in normalized space.
    chunk1 = np.zeros((G, chunks_raw.shape[1], 7), np.float32)
    chunk1[:, 0, :] = single_norm
    chunk1_norm = chunk1.reshape(G, -1)                   # (G,56)

    sys.path.insert(0, str(_REPO / "dqc"))
    agent, H = make_agent(args.ckpt_dir, images.shape[1:3], args.restore_epoch)
    assert chunk8_norm.shape[1] == H * 7, (chunk8_norm.shape, H)
    lang1 = load_text_emb(args.text_emb, instruction)[None]          # (1,512)
    print(f"[dqc] agent restored (backup_horizon={H}); MUSE emb {lang1.shape} "
          f"for {instruction!r}; scoring {G} pairs (q_agg={args.q_agg}) ...")

    q_action = np.zeros(G, np.float32)
    q_chunk1 = np.zeros(G, np.float32)
    q_chunk8 = np.zeros(G, np.float32)
    for s in range(0, G, args.batch):
        e = min(s + args.batch, G)
        obs = {"image": np.asarray(images[s:e])}
        goals = {"language": np.repeat(lang1, e - s, axis=0)}
        qa = np.asarray(agent.forward_action_critic(obs, goals, single_norm[s:e], train=False))
        q_action[s:e] = qa.min(axis=0) if args.q_agg == "min" else qa.mean(axis=0)
        q_chunk1[s:e] = np.asarray(agent.score_chunks(obs, goals, chunk1_norm[s:e], q_agg=args.q_agg))
        q_chunk8[s:e] = np.asarray(agent.score_chunks(obs, goals, chunk8_norm[s:e], q_agg=args.q_agg))
        if (s // args.batch) % 5 == 0:
            print(f"  {e}/{G}", flush=True)

    csv_path = out / "dqc_scores_filtered.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "ep_uid", "group", "group_name", "success",
                    "branch_t", "q_action", "q_chunk1", "q_chunk8"])
        for i in range(G):
            w.writerow([i, int(ep_uid[i]), int(group[i]), GROUP_NAMES[int(group[i])],
                        bool(success[i]), int(branch_t[i]),
                        float(q_action[i]), float(q_chunk1[i]), float(q_chunk8[i])])
    print(f"[dqc] {G} rows -> {csv_path}")

    # quick success-vs-failure (before-filtering) AUC for each variant.
    for name, m in [("q_action", q_action), ("q_chunk1", q_chunk1), ("q_chunk8", q_chunk8)]:
        a, sp, sf = per_episode_auc(ep_uid, success, m)
        print(f"[auc] DQC {name}: success-vs-failure per-ep AUC={a:.3f} "
              f"(succ mu={sp:.3f}, fail mu={sf:.3f})")


if __name__ == "__main__":
    main()
