"""Diagnose the prefix-KV exactness gap. Runs several configs in one model load:

  1. C=1  (no broadcast possible, no cross-image batching) — isolates the
     prefix/suffix KV-split mechanism itself.
  2. C=2  — adds batching across distinct images.
  3. Per-row reward dump (baseline vs kvprefix) so we see whether ALL rows
     diverge (mechanism bug) or only rows>0 (image/broadcast/batch bug).
  4. Fresh-prefill-each-round vs the same call — both clear cache first, so this
     is purely the single-call split, not cross-round reuse.

Run on a GPU node (a40+, bf16) in the monkey-verifier env.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "monkey-verifier" / "src"))
os.environ.setdefault("MODEL_DIR", str(REPO_ROOT / "monkey-verifier" / "model_dir"))

INSTRUCTION = "put the eggplant in the basket"


def main() -> int:
    from infer_server import RobotRewardModel
    rrm = RobotRewardModel()
    print("[dbg] verifier loaded.", flush=True)
    rng = np.random.RandomState(0)

    for C in (1, 2, 4):
        images = [rng.randint(0, 256, size=(256, 256, 3), dtype=np.uint8) for _ in range(C)]
        keys = [rrm._raw_image_key(im) for im in images]
        actions = rng.uniform(-1, 1, size=(C, 7)).astype(np.float32)

        base = np.asarray(rrm.get_rewards_paired(INSTRUCTION, images, actions), dtype=np.float64)
        rrm.clear_prefix_cache()
        kv = np.asarray(rrm.get_rewards_paired_kvprefix(INSTRUCTION, images, actions, keys),
                        dtype=np.float64)
        diff = np.abs(base - kv)
        print(f"\n[dbg] C={C}: max|diff|={diff.max():.4e}")
        for i in range(C):
            print(f"      row {i}: base={base[i]:+.5f}  kv={kv[i]:+.5f}  d={diff[i]:.4e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
