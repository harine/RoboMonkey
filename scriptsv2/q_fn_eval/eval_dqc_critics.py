"""eval_dqc_critics.py — Faithful DQC eval: single-step AND chunk critic.

Run in the `dqc` conda env (jax 0.10 / flax 0.12). Loads the REAL
BridgeDQCChunkAgent from the checkpoint and scores both heads in the same
checkpoint on the eggplant pairs sampled by eval_dqc.py:

  * action_critic  Q(s_t, a_t)           — single 7-D action  (forward_action_critic)
  * chunk_critic   Q(s_t, a_{t:t+8})      — 56-D 8-step chunk  (score_chunks)

This replaces the earlier hand-rolled pure-JAX forward pass (eval_q.py), which
had several mismatches vs the true encoder (image norm /127.5-1 not /255,
GroupNorm groups=4 not 32, a stem max-pool, FiLM applied after each block, and
the MUSE model is multilingual USE-3 not USE-4). Using the real agent + real
text processor removes all of that guesswork.

Inputs come from pairs.npz (written by eval_dqc.py): images (uint8, fed raw —
the encoder normalizes internally), raw single actions, and raw 8-step chunks
with NaN past the episode end. We normalize with the Bridge action stats and
zero-fill the NaN pad slots **in normalized space**, exactly as the training
dataset assembled chunks (infinite-horizon backup).

Writes dqc_scores.csv (ep_idx, success, branch_t, q_value, q_chunk). Then run
`eval_dqc.py --plot-only` (simpler_env) for comparison.png.

Usage
-----
conda run -n dqc python eval_dqc_critics.py \
    --out-dir results/dqc_vs_verifier \
    --ckpt-dir /home/harine/RoboMonkey/dqc_checkpoint/sd000s_137913.0.20260605_163017 \
    --instruction "put the eggplant in the basket"
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]   # /home/harine/RoboMonkey

# Bridge action normalization (matches checkpoint action_normalization.json and
# dqc/utils/bridge_lerobot_dataset.py BRIDGE_ACTION_LOW/HIGH).
LOW  = np.array([-0.05, -0.05, -0.05, -0.25, -0.25, -0.25, 0.0], np.float32)
HIGH = np.array([ 0.05,  0.05,  0.05,  0.25,  0.25,  0.25, 1.0], np.float32)
CLIP_EPS = 1e-5


def normalize(a):
    n = 2.0 * (a - LOW) / (HIGH - LOW) - 1.0
    return np.clip(n, -1.0 + CLIP_EPS, 1.0 - CLIP_EPS)


def make_agent(ckpt_dir: str, image_hw):
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
    agent = restore_agent(agent, ckpt_dir, 100000)
    return agent, H


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results/dqc_vs_verifier")
    ap.add_argument("--ckpt-dir",
                    default="/home/harine/RoboMonkey/dqc_checkpoint/sd000s_137913.0.20260605_163017")
    ap.add_argument("--instruction", default="put the eggplant in the basket")
    ap.add_argument("--q-agg", default="min", choices=["min", "mean"])
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    out = Path(args.out_dir)
    d = np.load(out / "pairs.npz", allow_pickle=True)
    images  = d["images"]                                   # (G, H, W, 3) uint8
    actions = d["actions"].astype(np.float32)               # (G, 7) raw
    chunks_raw = d["action_chunks"].astype(np.float32)      # (G, 8, 7) raw, NaN pad
    ep_idx, t_real, success = d["ep_idx"], d["t_real"], d["success"].astype(bool)
    instruction = args.instruction or str(d["instruction"])
    G = len(actions)

    # Normalize. Chunk: normalize valid entries, zero-fill NaN pad in NORMALIZED
    # space (= training's infinite-horizon zero pad), then flatten time-major.
    single_norm = normalize(actions)                        # (G, 7)
    cn = normalize(chunks_raw)                              # (G, 8, 7); NaN stays NaN
    cn = np.where(np.isnan(chunks_raw), 0.0, cn).astype(np.float32)
    chunk_norm = cn.reshape(G, -1)                          # (G, 56) time-major

    # Real agent + MUSE (multilingual USE-3) embedding.
    sys.path.insert(0, str(_REPO / "dqc"))
    agent, H = make_agent(args.ckpt_dir, images.shape[1:3])
    assert chunk_norm.shape[1] == H * 7, (chunk_norm.shape, H)
    from utils.text_processing import make_text_processor
    text_proc = make_text_processor("muse_embedding")
    lang1 = np.asarray(text_proc.encode([instruction]), np.float32)  # (1, 512)
    print(f"[dqc] agent restored; MUSE emb {lang1.shape} for {instruction!r}")
    print(f"[dqc] scoring {G} pairs (action critic + chunk critic, q_agg={args.q_agg}) ...")

    q_single = np.zeros(G, np.float32)
    q_chunk  = np.zeros(G, np.float32)
    for s in range(0, G, args.batch):
        e = min(s + args.batch, G)
        obs = {"image": np.asarray(images[s:e])}
        goals = {"language": np.repeat(lang1, e - s, axis=0)}
        qa = np.asarray(agent.forward_action_critic(obs, goals, single_norm[s:e], train=False))
        q_single[s:e] = qa.min(axis=0) if args.q_agg == "min" else qa.mean(axis=0)
        q_chunk[s:e] = np.asarray(agent.score_chunks(obs, goals, chunk_norm[s:e], q_agg=args.q_agg))
        if (s // args.batch) % 5 == 0:
            print(f"  {e}/{G}", flush=True)

    csv_path = out / "dqc_scores.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ep_idx", "success", "branch_t", "q_value", "q_chunk"])
        for i in range(G):
            w.writerow([int(ep_idx[i]), bool(success[i]), int(t_real[i]),
                        float(q_single[i]), float(q_chunk[i])])
    print(f"[dqc] {G} rows → {csv_path}")

    # Per-episode AUC for both critics (numpy only; full plot via eval_dqc.py --plot-only).
    def auc(metric):
        per = {}
        for i in range(G):
            per.setdefault(int(ep_idx[i]), []).append(metric[i])
        means = {k: float(np.mean(v)) for k, v in per.items()}
        succ_ep = {int(ep_idx[i]): bool(success[i]) for i in range(G)}
        pos = np.array([means[k] for k in means if succ_ep[k]])
        neg = np.array([means[k] for k in means if not succ_ep[k]])
        diff = pos[:, None] - neg[None, :]
        return ((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size, pos.mean(), neg.mean()
    for name, m in [("single-step Q", q_single), ("chunk Q", q_chunk)]:
        a, sp, sf = auc(m)
        print(f"[auc] DQC {name}: per-episode AUC={a:.3f}  (success μ={sp:.3f}, failure μ={sf:.3f})")


if __name__ == "__main__":
    main()
