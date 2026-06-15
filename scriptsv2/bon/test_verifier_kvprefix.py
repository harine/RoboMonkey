"""Exactness + speedup test for the per-image prefix-KV verifier path.

Simulates one image-chunk of training: C distinct images scored across ROUNDS
autoregressive verifier calls (new actions each round). The prefix-KV path
encodes each image's (image + instruction) prefix once and reuses its KV across
rounds, forwarding only the action suffix.

Checks:
  1. Determinism: the prefix-KV path is repeatable (kv == kv).
  2. Ranking preservation: prefix-KV is NOT bit-exact vs the full forward (a
     chunked/cached forward rounds differently than one full forward in bf16,
     and cuBLAS GEMMs are shape-dependent; the gap compounds over 32 layers to
     ~0.1 on rewards of O(1)). The action the search SELECTS is unchanged, so we
     assert top-1 agreement + Spearman vs the full-forward baseline instead.
  3. Speedup: time ROUNDS rounds, prefix-KV vs baseline (full forward each round).

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

    # --- 1) Correctness: NOT bit-exact vs the full forward (impossible — a
    # chunked/cached forward rounds differently than one full forward in bf16,
    # and the per-row cuBLAS GEMMs are shape-dependent; the gap compounds over 32
    # layers to ~0.1 on rewards of O(1)). What must hold is (a) determinism — the
    # KV path is repeatable — and (b) the RANKING the search uses is preserved.
    def _spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rb = np.argsort(np.argsort(b)).astype(np.float64)
        ra -= ra.mean(); rb -= rb.mean()
        den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
        return float((ra * rb).sum() / den) if den > 0 else 1.0

    rrm.clear_prefix_cache()
    kv_a = np.asarray(rrm.get_rewards_paired_kvprefix(INSTRUCTION, images, action_rounds[0], keys))
    rrm.clear_prefix_cache()
    kv_b = np.asarray(rrm.get_rewards_paired_kvprefix(INSTRUCTION, images, action_rounds[0], keys))
    determinism = float(np.max(np.abs(kv_a - kv_b)))

    max_diff, top1_agree, spearmans = 0.0, 0, []
    rrm.clear_prefix_cache()
    for actions in action_rounds:
        base = np.asarray(rrm.get_rewards_paired(INSTRUCTION, images, actions), dtype=np.float64)
        kv = np.asarray(rrm.get_rewards_paired_kvprefix(INSTRUCTION, images, actions, keys),
                        dtype=np.float64)
        max_diff = max(max_diff, float(np.max(np.abs(base - kv))))
        top1_agree += int(base.argmax() == kv.argmax())
        spearmans.append(_spearman(base, kv))
    mean_sp = float(np.mean(spearmans))
    print(f"[kv] determinism (kv vs kv)   : {determinism:.3e}  (must be ~0)")
    print(f"[kv] max |baseline - kvprefix|: {max_diff:.3e}  (bf16 noise, not bit-exact)")
    print(f"[kv] top-1 agreement / Spearman over {ROUNDS} rounds (rank over {C} imgs): "
          f"{top1_agree}/{ROUNDS}, mean rho={mean_sp:.4f}")

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

    ok = (determinism < 1e-4) and (top1_agree == ROUNDS) and (mean_sp >= 0.95)
    print("[kv] PASS (deterministic + rankings preserved)" if ok else
          f"[kv] FAIL: determinism={determinism:.2e} top1={top1_agree}/{ROUNDS} "
          f"rho={mean_sp:.3f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
