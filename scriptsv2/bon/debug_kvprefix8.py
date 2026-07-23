"""Pin down the KV-split divergence: logic vs cuBLAS non-determinism (C=1).

  - determinism: run base x2 and kv x2; if a path differs from itself, cuBLAS
    is non-deterministic (shape/workspace dependent).
  - exact-cache: feed the suffix forward the EXACT full-forward prefix K/V. If
    that == base, the suffix path is correct and only the prefill K/V is the
    (numerical) error source.
  - layer-0 prefix input diff: are the prefill's prefix embeds bit-identical to
    the full forward's first-P hidden input?

Set CUBLAS_WORKSPACE_CONFIG=:4096:8 in the environment to test whether forcing
deterministic cuBLAS shrinks the gap.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "monkey-verifier" / "src"))
os.environ.setdefault("MODEL_DIR", str(REPO_ROOT / "monkey-verifier" / "model_dir"))

INSTRUCTION = "put the eggplant in the basket"


def main() -> int:
    from infer_server import RobotRewardModel
    rrm = RobotRewardModel()
    print(f"[dbg8] CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')!r}")
    print(f"[dbg8] deterministic_algorithms={torch.are_deterministic_algorithms_enabled()}")
    rng = np.random.RandomState(0)
    images = [rng.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)]
    keys = [rrm._raw_image_key(images[0])]
    actions = rng.uniform(-1, 1, size=(1, 7)).astype(np.float32)

    decoder = rrm.model.backbone_model.get_decoder()
    layers = decoder.layers
    nL = len(layers)

    def kvp():
        rrm.clear_prefix_cache()
        return float(rrm.get_rewards_paired_kvprefix(INSTRUCTION, images, actions, keys)[0])

    def basep():
        return float(rrm.get_rewards_paired(INSTRUCTION, images, actions)[0])

    # 1) determinism
    b1, b2 = basep(), basep()
    k1, k2 = kvp(), kvp()
    print(f"[dbg8] base x2  = {b1:+.6f}, {b2:+.6f}  (selfdiff {abs(b1-b2):.2e})")
    print(f"[dbg8] kv   x2  = {k1:+.6f}, {k2:+.6f}  (selfdiff {abs(k1-k2):.2e})")
    print(f"[dbg8] base-vs-kv = {abs(b1-k1):.3e}")

    # 2) capture exact full prefix K/V + full layer-0 input
    captured, l0 = {}, []
    def make_hook(idx):
        def hook(module, inp, out):
            pkv = out[2]
            if pkv is not None:
                captured[idx] = (pkv[0].detach().clone(), pkv[1].detach().clone(), int(pkv[2]))
        return hook
    def l0_hook(module, args):
        l0.append(args[0].detach().clone())
    handles = [layers[i].self_attn.register_forward_hook(make_hook(i)) for i in range(nL)]
    h0 = layers[0].register_forward_pre_hook(l0_hook)
    try:
        basep()
    finally:
        for h in handles:
            h.remove()
        h0.remove()
    full_l0 = max(l0, key=lambda t: t.shape[1])   # the full 418-length layer-0 input
    T = captured[0][2]

    # 3) prefill, capture its prefix (346-len) layer-0 input
    l0p = []
    h1 = layers[0].register_forward_pre_hook(lambda m, a: l0p.append(a[0].detach().clone()))
    try:
        kvp()
    finally:
        h1.remove()
    pre_l0 = max(l0p, key=lambda t: t.shape[1])   # prefill prefix input (346)
    P = pre_l0.shape[1]
    d_l0 = (full_l0[:, :P, :].float() - pre_l0.float()).abs().max().item()
    print(f"[dbg8] P={P} T={T}  layer0 prefix-input max|full[:P]-prefill| = {d_l0:.3e}")

    # 4) exact-cache: suffix forward with EXACT full prefix K/V
    exact = [(captured[i][0][:, :, :P, :].contiguous(),
              captured[i][1][:, :, :P, :].contiguous(), P) for i in range(nL)]
    rrm._prefix_kv_cache = {keys[0]: exact}
    kv_exact = float(rrm.get_rewards_paired_kvprefix(INSTRUCTION, images, actions, keys)[0])
    print(f"[dbg8] kv(EXACT prefix K/V) = {kv_exact:+.6f}  base={b1:+.6f}  "
          f"d={abs(kv_exact-b1):.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
