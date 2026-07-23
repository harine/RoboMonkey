"""Test the suffix-consumption logic in isolation (C=1).

Feed the suffix forward the EXACT prefix K/V captured from a full forward
(instead of the prefill's own cache). If the resulting reward == base, the
suffix-with-past path is correct and the only error source is the prefill's
approximate K/V. If it still differs, the suffix path itself has a bug.

Also reports, for layer 0, whether the prefill's prefix EMBEDS (layer-0 input)
are bit-identical to the full forward's first-P hidden input.
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
    print("[dbg7] verifier loaded.", flush=True)
    rng = np.random.RandomState(0)
    images = [rng.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)]
    keys = [rrm._raw_image_key(images[0])]
    actions = rng.uniform(-1, 1, size=(1, 7)).astype(np.float32)

    decoder = rrm.model.backbone_model.get_decoder()
    layers = decoder.layers
    nL = len(layers)

    base = np.asarray(rrm.get_rewards_paired(INSTRUCTION, images, actions), dtype=np.float64)

    # Capture exact full-forward per-layer K/V + layer-0 input hidden.
    captured = {}
    l0_input = {}
    def make_hook(idx):
        def hook(module, inp, out):
            pkv = out[2]
            if pkv is not None:
                captured[idx] = (pkv[0].detach().clone(), pkv[1].detach().clone(), int(pkv[2]))
        return hook
    def l0_pre_hook(module, args):
        # decoder layer forward(hidden_states, ...) -> capture hidden_states
        l0_input["full"] = args[0].detach().clone()
    handles = [layers[i].self_attn.register_forward_hook(make_hook(i)) for i in range(nL)]
    h0 = layers[0].register_forward_pre_hook(l0_pre_hook)
    try:
        rrm.get_rewards_paired(INSTRUCTION, images, actions)
    finally:
        for h in handles:
            h.remove()
        h0.remove()
    P = None

    # Run the prefill to populate the cache, and capture the prefill's layer-0 input.
    l0_pre = {}
    def l0_pre_hook2(module, args):
        l0_pre["prefill"] = args[0].detach().clone()
    h1 = layers[0].register_forward_pre_hook(l0_pre_hook2)
    try:
        rrm.clear_prefix_cache()
        kv_native = np.asarray(
            rrm.get_rewards_paired_kvprefix(INSTRUCTION, images, actions, keys),
            dtype=np.float64)
    finally:
        h1.remove()
    P = rrm._prefix_kv_cache[keys[0]][0][0].shape[-2]

    # Compare layer-0 inputs (prefill prefix-embeds vs full first-P hidden).
    full_l0 = l0_input["full"][:, :P, :].float()
    pre_l0 = l0_pre["prefill"].float()
    print(f"[dbg7] P={P}  layer0-input max|full[:P]-prefill| = "
          f"{(full_l0 - pre_l0).abs().max().item():.3e}  "
          f"(shapes {tuple(full_l0.shape)} vs {tuple(pre_l0.shape)})")

    # Build a cache from the EXACT full-forward prefix K/V and run the suffix.
    exact_cache = [(captured[i][0][:, :, :P, :].contiguous(),
                    captured[i][1][:, :, :P, :].contiguous(),
                    P) for i in range(nL)]
    rrm._prefix_kv_cache[keys[0]] = exact_cache
    kv_exact = np.asarray(
        rrm.get_rewards_paired_kvprefix(INSTRUCTION, images, actions, keys),
        dtype=np.float64)

    print(f"[dbg7] base       = {base[0]:+.6f}")
    print(f"[dbg7] kv (native)= {kv_native[0]:+.6f}  d={abs(kv_native[0]-base[0]):.3e}")
    print(f"[dbg7] kv (EXACT prefix K/V) = {kv_exact[0]:+.6f}  d={abs(kv_exact[0]-base[0]):.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
