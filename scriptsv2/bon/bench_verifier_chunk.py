"""Sweep ROBOMONKEY_PAIRED_CHUNK_SIZE to find the GPU-batching sweet spot.

The verifier forward (LLaVA-7B over ~576 image tokens + action suffix) dominates
training step time, and `get_rewards_paired` currently chunks B into groups of 4
-> many tiny underutilized forwards. This times one round of B (image, action)
pairs at several chunk sizes (model loaded once) to see how much larger chunks
cut per-round wall-clock, and watches for OOM.

  source ~/miniconda3/etc/profile.d/conda.sh && conda activate monkey-verifier
  cd /mmfs1/home/harine/RoboMonkey
  python scriptsv2/bon/bench_verifier_chunk.py
Env: BENCH_B (default 128), BENCH_CHUNKS (csv, default "4,8,16,32,64").
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
CHUNKS = [int(c) for c in os.environ.get("BENCH_CHUNKS", "4,8,16,32,64").split(",")]


def main() -> int:
    from infer_server import RobotRewardModel
    print(f"[chunk] B={B} chunks={CHUNKS}", flush=True)
    rrm = RobotRewardModel()
    print("[chunk] verifier loaded.", flush=True)

    rng = np.random.RandomState(0)
    images = [rng.randint(0, 256, size=(256, 256, 3), dtype=np.uint8) for _ in range(B)]
    actions = rng.uniform(-1, 1, size=(B, 7)).astype(np.float32)
    rrm._proc_image_cache_size = max(B + 8, 512)  # remove preprocessing noise

    def _time_round(chunk):
        os.environ["ROBOMONKEY_PAIRED_CHUNK_SIZE"] = str(chunk)
        rrm._proc_image_cache.clear()
        # warmup at this chunk size (autotune), not timed
        rrm.get_rewards_paired(INSTRUCTION, images, actions)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rrm.get_rewards_paired(INSTRUCTION, images, actions)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        mem = torch.cuda.max_memory_allocated() / 1e9
        return dt, mem

    base = None
    for c in CHUNKS:
        try:
            torch.cuda.reset_peak_memory_stats()
            dt, mem = _time_round(c)
        except RuntimeError as e:
            print(f"[chunk] chunk={c:>3}: OOM/err ({str(e)[:60]})")
            continue
        if base is None:
            base = dt
        per_img_ms = 1000 * dt / B
        step15 = dt * 15
        print(f"[chunk] chunk={c:>3}: round/B={dt:6.2f}s  "
              f"{per_img_ms:6.1f} ms/img  peakmem={mem:5.1f}G  "
              f"~15-round step={step15:6.1f}s  speedup_vs_chunk{CHUNKS[0]}={base/dt:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
