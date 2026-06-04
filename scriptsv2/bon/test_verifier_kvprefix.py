"""Exactness + speedup test for the per-image prefix-KV verifier path.

Simulates one image-chunk of training: C distinct images scored across ROUNDS
autoregressive verifier calls (new actions each round). The prefix-KV path
encodes each image's (image + instruction) prefix once and reuses its KV across
rounds, forwarding only the action suffix.

Checks:
  1. Exactness: get_rewards_paired_kvprefix == get_rewards_paired (per round).
  2. Speedup: time ROUNDS rounds, prefix-KV vs baseline (full forward each round).

Run on a GPU node in the monkey-verifier env:
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate monkey-verifier
  cd /mmfs1/home/harine/RoboMonkey
  python scriptsv2/bon/test_verifier_kvprefix.py
Env: KV_C (chunk size, default 16), KV_ROUNDS (default 15).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "monkey-verifier" / "src"))
os.environ.setdefault("MODEL_DIR", str(REPO_ROOT / "monkey-verifier" / "model_dir"))

INSTRUCTION = "put the eggplant in the basket"
C = int(os.environ.get("KV_C", 16))
ROUNDS = int(os.environ.get("KV_ROUNDS", 15))


def main() -> int:
    from infer_server import RobotRewardModel
    print(f"[kv] C={C} ROUNDS={ROUNDS}", flush=True)
    rrm = RobotRewardModel()
    print("[kv] verifier loaded.", flush=True)

    rng = np.random.RandomState(0)
    images = [rng.randint(0, 256, size=(256, 256, 3), dtype=np.uint8) for _ in range(C)]
    keys = [rrm._raw_image_key(im) for im in images]
    action_rounds = [rng.uniform(-1, 1, size=(C, 7)).astype(np.float32)
                     for _ in range(ROUNDS)]

    # --- 1) Exactness: per round, prefix-KV vs full forward ---
    max_diff = 0.0
    rrm.clear_prefix_cache()
    for r, actions in enumerate(action_rounds):
        base = np.asarray(rrm.get_rewards_paired(INSTRUCTION, images, actions), dtype=np.float64)
        kv = np.asarray(rrm.get_rewards_paired_kvprefix(INSTRUCTION, images, actions, keys),
                        dtype=np.float64)
        d = float(np.max(np.abs(base - kv)))
        max_diff = max(max_diff, d)
    print(f"[kv] exactness: max |baseline - kvprefix| over {ROUNDS} rounds = {max_diff:.3e}")

    # --- 2) Speedup: ROUNDS rounds, baseline vs prefix-KV ---
    def _timed(fn):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for actions in action_rounds:
            fn(actions)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    # warmup both paths
    rrm.get_rewards_paired(INSTRUCTION, images, action_rounds[0])
    rrm.clear_prefix_cache()
    rrm.get_rewards_paired_kvprefix(INSTRUCTION, images, action_rounds[0], keys)

    torch.cuda.reset_peak_memory_stats()
    base_t = _timed(lambda a: rrm.get_rewards_paired(INSTRUCTION, images, a))
    base_mem = torch.cuda.max_memory_allocated() / 1e9

    torch.cuda.reset_peak_memory_stats()
    rrm.clear_prefix_cache()
    kv_t = _timed(lambda a: rrm.get_rewards_paired_kvprefix(INSTRUCTION, images, a, keys))
    kv_mem = torch.cuda.max_memory_allocated() / 1e9

    print(f"[kv] baseline  : {base_t:.2f}s for {ROUNDS} rounds  "
          f"({1000*base_t/(ROUNDS*C):.1f} ms/score)  peakmem={base_mem:.1f}G")
    print(f"[kv] prefix-KV : {kv_t:.2f}s for {ROUNDS} rounds  "
          f"({1000*kv_t/(ROUNDS*C):.1f} ms/score)  peakmem={kv_mem:.1f}G")
    print(f"[kv] speedup={base_t/kv_t:.2f}x  (C={C} chunk; extrapolate to B=256 by "
          f"running 256/C chunks)")

    ok = max_diff < 0.05
    print("[kv] PASS (exact within tol)" if ok else f"[kv] FAIL: diff {max_diff:.3e} too large")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
