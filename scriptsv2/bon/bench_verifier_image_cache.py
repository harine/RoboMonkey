"""Wall-clock A/B for the per-image verifier cache.

Simulates the verifier load of ONE training step: the policy scores the same B
images across `ROUNDS` autoregressive verifier calls (new actions each round).
Times it with the preprocessed-image cache OFF vs ON to see whether eliminating
the per-round `_process_image` (PIL resize + JPEG roundtrip + CLIP preprocess)
actually cuts wall-clock — i.e. whether preprocessing was the bottleneck.

Run on a GPU node in the monkey-verifier env:
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate monkey-verifier
  cd /mmfs1/home/harine/RoboMonkey
  python scriptsv2/bon/bench_verifier_image_cache.py
Env: BENCH_B (batch, default 128), BENCH_ROUNDS (default 6),
     ROBOMONKEY_PAIRED_CHUNK_SIZE (default 4).
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
B = int(os.environ.get("BENCH_B", 128))
ROUNDS = int(os.environ.get("BENCH_ROUNDS", 6))


def _run(rrm, images, action_rounds):
    """Score all rounds; return (total_s, round1_s, steady_s_per_round)."""
    torch.cuda.synchronize()
    per_round = []
    for actions in action_rounds:
        t0 = time.perf_counter()
        rrm.get_rewards_paired(INSTRUCTION, images, actions)
        torch.cuda.synchronize()
        per_round.append(time.perf_counter() - t0)
    total = sum(per_round)
    steady = sum(per_round[1:]) / max(1, len(per_round) - 1)
    return total, per_round[0], steady


def main() -> int:
    from infer_server import RobotRewardModel
    print(f"[bench] B={B} ROUNDS={ROUNDS} "
          f"chunk={os.environ.get('ROBOMONKEY_PAIRED_CHUNK_SIZE', '4')}", flush=True)
    rrm = RobotRewardModel()
    print("[bench] verifier loaded.", flush=True)

    rng = np.random.RandomState(0)
    images = [rng.randint(0, 256, size=(256, 256, 3), dtype=np.uint8) for _ in range(B)]
    action_rounds = [rng.uniform(-1, 1, size=(B, 7)).astype(np.float32)
                     for _ in range(ROUNDS)]

    def _reset(size):
        rrm._image_feat_cache_size = size
        rrm._proc_image_cache_size = size
        rrm._image_feat_cache.clear()
        rrm._proc_image_cache.clear()

    # warmup (kernels/autotune) — not timed
    _reset(0)
    rrm.get_rewards_paired(INSTRUCTION, images, action_rounds[0])

    _reset(0)
    off_total, off_r1, off_steady = _run(rrm, images, action_rounds)
    _reset(max(B + 8, 512))
    on_total, on_r1, on_steady = _run(rrm, images, action_rounds)

    print(f"\n[bench] cache OFF: total={off_total:.2f}s  round1={off_r1:.2f}s  "
          f"steady/round={off_steady:.2f}s")
    print(f"[bench] cache ON : total={on_total:.2f}s  round1={on_r1:.2f}s  "
          f"steady/round={on_steady:.2f}s")
    # Extrapolate to a real 15-round (max_actions=16) training step.
    R = 15
    off_step = off_steady * R
    on_step = on_r1 + on_steady * (R - 1)
    speed = off_step / on_step if on_step else float("nan")
    print(f"\n[bench] extrapolated {R}-round step:  OFF={off_step:.1f}s  "
          f"ON={on_step:.1f}s  speedup={speed:.2f}x")
    print(f"[bench] steady/round speedup={off_steady/on_steady:.2f}x "
          f"(isolates the per-round preprocessing saved)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
