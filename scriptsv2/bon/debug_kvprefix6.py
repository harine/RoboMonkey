"""Definitive KV-cache comparison (C=1).

Capture per-layer present_key_value from a FULL forward (base path), take the
prefix slice [:, :, :P], and compare against the prefill's CACHED prefix K/V.

  cache matches full-forward prefix -> bug is in suffix consumption.
  cache differs                     -> bug is in prefill caching.

Prints per-layer max|dK|, max|dV| (first/last few layers) + the worst layer.
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
    print("[dbg6] verifier loaded.", flush=True)
    rng = np.random.RandomState(0)
    images = [rng.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)]
    keys = [rrm._raw_image_key(images[0])]
    actions = rng.uniform(-1, 1, size=(1, 7)).astype(np.float32)

    decoder = rrm.model.backbone_model.get_decoder()
    layers = decoder.layers
    nL = len(layers)

    # 1) Capture per-layer present_key_value from the FULL forward.
    captured = {}
    def make_hook(idx):
        def hook(module, inp, out):
            pkv = out[2]
            if pkv is not None:
                captured[idx] = (pkv[0].detach().clone(), pkv[1].detach().clone(), int(pkv[2]))
        return hook
    handles = [layers[i].self_attn.register_forward_hook(make_hook(i)) for i in range(nL)]
    try:
        rrm.get_rewards_paired(INSTRUCTION, images, actions)
    finally:
        for h in handles:
            h.remove()
    if not captured:
        print("[dbg6] FULL forward returned no present_key_value (use_cache off?)")
        return 1
    Tfull = captured[0][2]
    print(f"[dbg6] full-forward kv_seq_len T={Tfull}, captured {len(captured)} layers")

    # 2) Run the prefill (populates rrm._prefix_kv_cache) and read the cache.
    rrm.clear_prefix_cache()
    rrm.get_rewards_paired_kvprefix(INSTRUCTION, images, actions, keys)
    cached = rrm._prefix_kv_cache[keys[0]]   # list[(k,v,n)] per layer
    P = cached[0][0].shape[-2]
    print(f"[dbg6] cached prefix len P={P}  (suffix S={Tfull - P})")

    # 3) Compare prefix slice of full K/V vs cached K/V, per layer.
    worst = (-1, 0.0)
    for i in range(nL):
        fullK, fullV, _ = captured[i]
        fK = fullK[:, :, :P, :].float()
        fV = fullV[:, :, :P, :].float()
        cK = cached[i][0].float()
        cV = cached[i][1].float()
        dK = (fK - cK).abs().max().item()
        dV = (fV - cV).abs().max().item()
        m = max(dK, dV)
        if m > worst[1]:
            worst = (i, m)
        if i < 3 or i >= nL - 3:
            print(f"       layer {i:2d}: max|dK|={dK:.3e}  max|dV|={dV:.3e}")
    print(f"[dbg6] WORST layer={worst[0]}  max|d|={worst[1]:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
