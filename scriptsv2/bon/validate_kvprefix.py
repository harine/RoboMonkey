"""Functional validation of the prefix-KV verifier path.

Bit-exactness vs the full forward is impossible (a chunked/cached forward rounds
differently than a single full forward in bf16; the gap compounds over 32
layers). What the search actually uses is the *ranking* of candidate actions per
image, so we validate that instead:

For C images x K candidate actions each, score every (image, action) pair with
  base = get_rewards_paired            (full forward, the reference)
  kv   = get_rewards_paired_kvprefix   (prefix reused across the K candidates)
and report, per image:
  - top-1 agreement: does kv pick the same best action as base?
  - top-1 reward gap: |base - kv| on base's chosen action
  - Spearman rank correlation over the K candidates
plus the global max|base-kv|.

Env: VAL_C (images, default 16), VAL_K (candidates/image, default 16).
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "monkey-verifier" / "src"))
os.environ.setdefault("MODEL_DIR", str(REPO_ROOT / "monkey-verifier" / "model_dir"))

INSTRUCTION = "put the eggplant in the basket"
C = int(os.environ.get("VAL_C", 16))
K = int(os.environ.get("VAL_K", 16))


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 1.0


def main() -> int:
    from infer_server import RobotRewardModel
    rrm = RobotRewardModel()
    print(f"[val] C={C} images x K={K} candidates", flush=True)
    rng = np.random.RandomState(0)

    imgs = [rng.randint(0, 256, size=(256, 256, 3), dtype=np.uint8) for _ in range(C)]
    keys = [rrm._raw_image_key(im) for im in imgs]
    # C*K pairs: row c*K+k is image c, candidate action k.
    images_flat, actions_flat, keys_flat = [], [], []
    for c in range(C):
        for _ in range(K):
            images_flat.append(imgs[c])
            actions_flat.append(rng.uniform(-1, 1, size=(7,)).astype(np.float32))
            keys_flat.append(keys[c])
    actions_flat = np.stack(actions_flat, axis=0)

    base = np.asarray(rrm.get_rewards_paired(INSTRUCTION, images_flat, actions_flat),
                      dtype=np.float64).reshape(C, K)
    rrm.clear_prefix_cache()
    kv = np.asarray(
        rrm.get_rewards_paired_kvprefix(INSTRUCTION, images_flat, actions_flat, keys_flat),
        dtype=np.float64).reshape(C, K)

    top1_agree = 0
    top1_gaps, spearmans = [], []
    for c in range(C):
        ba, ka = int(base[c].argmax()), int(kv[c].argmax())
        top1_agree += int(ba == ka)
        top1_gaps.append(abs(base[c, ba] - kv[c, ba]))
        spearmans.append(spearman(base[c], kv[c]))

    maxd = float(np.abs(base - kv).max())
    print(f"[val] top-1 action agreement : {top1_agree}/{C} "
          f"({100.0*top1_agree/C:.1f}%)")
    print(f"[val] Spearman over {K} cands  : mean={np.mean(spearmans):.4f} "
          f"min={np.min(spearmans):.4f}")
    print(f"[val] top-1 reward gap        : mean={np.mean(top1_gaps):.3f} "
          f"max={np.max(top1_gaps):.3f}")
    print(f"[val] global max|base-kv|     : {maxd:.3f}  "
          f"(reward range [{base.min():.2f},{base.max():.2f}])")
    ok = (top1_agree >= 0.9 * C) and (np.mean(spearmans) >= 0.98)
    print("[val] PASS (rankings preserved)" if ok else "[val] WARN (check rankings)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
