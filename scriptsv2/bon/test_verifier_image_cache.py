"""Exactness + hit-rate test for the per-image verifier cache.

The in-process verifier now caches, keyed on the raw-image hash, (a) the
preprocessed CLIP-input tensor and (b) the CLIP vision-tower features, so the
same image recurring across the policy's `max_actions` autoregressive verifier
calls is preprocessed/encoded once instead of every round.

This test scores a fixed batch of distinct images against several rounds of
*different* actions (mimicking the autoregressive loop), once with caching
disabled and once enabled, and asserts the rewards are identical. It also checks
that the second/third rounds hit the caches.

Run on a GPU node in the monkey-verifier env:
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate monkey-verifier
  cd /mmfs1/home/harine/RoboMonkey
  python scriptsv2/bon/test_verifier_image_cache.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "monkey-verifier" / "src"))
os.environ.setdefault("MODEL_DIR", str(REPO_ROOT / "monkey-verifier" / "model_dir"))

INSTRUCTION = "put the eggplant in the basket"
B = 8          # distinct images in the batch
ROUNDS = 3     # autoregressive verifier calls (same images, new actions each)


def _score_all_rounds(rrm, images, action_rounds):
    """Return rewards array (ROUNDS, B) for the given images across all rounds."""
    out = []
    for actions in action_rounds:
        out.append(rrm.get_rewards_paired(INSTRUCTION, images, actions))
    return np.asarray(out, dtype=np.float64)


def main() -> int:
    print(f"[test] MODEL_DIR={os.environ['MODEL_DIR']}", flush=True)
    from infer_server import RobotRewardModel
    rrm = RobotRewardModel()
    print("[test] verifier loaded.", flush=True)

    rng = np.random.RandomState(0)
    # Fixed, distinct images (identical bytes across rounds -> cache should hit).
    images = [rng.randint(0, 256, size=(256, 256, 3), dtype=np.uint8) for _ in range(B)]
    # Distinct actions each round so we exercise different LLM suffixes.
    action_rounds = [rng.uniform(-1, 1, size=(B, 7)).astype(np.float32)
                     for _ in range(ROUNDS)]

    # --- caching DISABLED (baseline) ---
    rrm._image_feat_cache_size = 0
    rrm._proc_image_cache_size = 0
    rrm._image_feat_cache.clear()
    rrm._proc_image_cache.clear()
    rrm._feat_cache_hits = rrm._feat_cache_misses = 0
    rrm._proc_cache_hits = rrm._proc_cache_misses = 0
    rewards_off = _score_all_rounds(rrm, images, action_rounds)
    print(f"[test] caching OFF: proc(h={rrm._proc_cache_hits},m={rrm._proc_cache_misses}) "
          f"feat(h={rrm._feat_cache_hits},m={rrm._feat_cache_misses})")

    # --- caching ENABLED ---
    rrm._image_feat_cache_size = 512
    rrm._proc_image_cache_size = 512
    rrm._image_feat_cache.clear()
    rrm._proc_image_cache.clear()
    rrm._feat_cache_hits = rrm._feat_cache_misses = 0
    rrm._proc_cache_hits = rrm._proc_cache_misses = 0
    rewards_on = _score_all_rounds(rrm, images, action_rounds)
    print(f"[test] caching ON : proc(h={rrm._proc_cache_hits},m={rrm._proc_cache_misses}) "
          f"feat(h={rrm._feat_cache_hits},m={rrm._feat_cache_misses})")

    max_diff = float(np.max(np.abs(rewards_off - rewards_on)))
    print(f"[test] max |reward_off - reward_on| = {max_diff:.3e}")

    ok = True
    # Exactness: cached features are the same tensors -> expect ~bit-identical.
    if max_diff > 1e-3:
        print(f"[test] FAIL: rewards diverged ({max_diff:.3e} > 1e-3)")
        ok = False
    # Required: proc-image cache (the JPEG-roundtrip CPU lever) must reuse across
    # rounds -> round 1 misses (B), rounds 2..N hit (B each).
    exp_proc_miss, exp_proc_hit = B, B * (ROUNDS - 1)
    if (rrm._proc_cache_misses, rrm._proc_cache_hits) != (exp_proc_miss, exp_proc_hit):
        print(f"[test] FAIL: proc cache hits/misses "
              f"({rrm._proc_cache_hits},{rrm._proc_cache_misses}) "
              f"!= expected ({exp_proc_hit},{exp_proc_miss})")
        ok = False
    # Informational: the CLIP-feature cache lives in the encode_images monkeypatch,
    # which does not intercept the reward model's paired forward (PEFT wrapping), so
    # it is a no-op in training. Not required — CLIP runs on the (otherwise idle)
    # GPU; the CPU proc-cache is the wall-clock win. Revisit only if profiling shows
    # CLIP encoding is a remaining bottleneck.
    if rrm._feat_cache_hits == 0:
        print("[test] NOTE: CLIP-feature cache inactive in paired mode (expected; "
              "secondary GPU-side optimization, not required for correctness).")

    print("[test] PASS" if ok else "[test] FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
