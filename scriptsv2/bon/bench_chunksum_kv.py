"""Same-process A/B speedup of the CHUNK-SUM verifier loop: KV-prefix on vs off.

Chunk-sum scores each image's frame against M action steps and sums; across the
search's R rounds the SAME C frames recur. KV-off re-encodes the image tokens
every score; KV-on prefills each frame once and forwards only the action suffix.

Because both halves are timed back-to-back on the same GPU, node contention
cancels out (unlike a cross-job wall-clock comparison). Reports ms/score and the
speedup, plus a determinism/ranking sanity check.

Env: CS_C (images, default 16), CS_M (frames/actions per image, default 8),
     CS_R (rounds, default 8).
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "monkey-verifier" / "src"))
os.environ.setdefault("MODEL_DIR", str(REPO_ROOT / "monkey-verifier" / "model_dir"))

INSTRUCTION = "put the eggplant in the basket"
C = int(os.environ.get("CS_C", 16))
M = int(os.environ.get("CS_M", 8))
R = int(os.environ.get("CS_R", 8))


def main() -> int:
    from infer_server import RobotRewardModel
    rrm = RobotRewardModel()
    print(f"[cs] C={C} images x M={M} frames, R={R} rounds  "
          f"({C*M} pairs/round)", flush=True)
    rng = np.random.RandomState(0)

    imgs = [rng.randint(0, 256, size=(256, 256, 3), dtype=np.uint8) for _ in range(C)]
    keys = [rrm._raw_image_key(im) for im in imgs]
    # Chunk-sum pair layout: image c repeated M times (same frame, M actions).
    images_flat = [imgs[c] for c in range(C) for _ in range(M)]
    keys_flat = [keys[c] for c in range(C) for _ in range(M)]
    rounds = [rng.uniform(-1, 1, size=(C * M, 7)).astype(np.float32) for _ in range(R)]

    def _timed(fn):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for a in rounds:
            fn(a)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    # warmup + correctness (round 0)
    base0 = np.asarray(rrm.get_rewards_paired(INSTRUCTION, images_flat, rounds[0]))
    rrm.clear_prefix_cache()
    kv0 = np.asarray(rrm.get_rewards_paired_kvprefix(INSTRUCTION, images_flat, rounds[0], keys_flat))
    print(f"[cs] round0 max|base-kv|={np.abs(base0-kv0).max():.3f}  "
          f"top1-agree={int(base0.argmax()==kv0.argmax())}")

    torch.cuda.reset_peak_memory_stats()
    off_t = _timed(lambda a: rrm.get_rewards_paired(INSTRUCTION, images_flat, a))
    off_mem = torch.cuda.max_memory_allocated() / 1e9

    torch.cuda.reset_peak_memory_stats()
    rrm.clear_prefix_cache()
    on_t = _timed(lambda a: rrm.get_rewards_paired_kvprefix(INSTRUCTION, images_flat, a, keys_flat))
    on_mem = torch.cuda.max_memory_allocated() / 1e9

    npair = C * M
    print(f"[cs] KV-off : {off_t:.2f}s / {R} rounds  "
          f"({1000*off_t/(R*npair):.1f} ms/score)  peakmem={off_mem:.1f}G")
    print(f"[cs] KV-on  : {on_t:.2f}s / {R} rounds  "
          f"({1000*on_t/(R*npair):.1f} ms/score)  peakmem={on_mem:.1f}G")
    print(f"[cs] speedup = {off_t/on_t:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
