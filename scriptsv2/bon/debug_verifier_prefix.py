"""Standalone diagnostic for the verifier's KV-cache prefix forward path.

The in-process verifier's `_forward_kv_cache_prefix` raises
  TypeError("'NoneType' object is not subscriptable")
under some configurations, and the code silently disables the prefix path
forever after the first failure. This script forces a single attempt at the
prefix forward, captures the full traceback, and inspects the shape and
contents of `prefix_out.past_key_values` so we can see exactly what is None.

Run from the repo root on a GPU node (e.g. the salloc'd compute node) inside
the monkey-verifier conda env:

  source ~/miniconda3/etc/profile.d/conda.sh && conda activate monkey-verifier
  cd /mmfs1/home/harine/RoboMonkey
  python scriptsv2/bon/debug_verifier_prefix.py

It prints (in order):
  1. type of `prefix_out`, list of its attrs, and `past_key_values` type
  2. layer-by-layer dump of past_key_values (whether each entry is None,
     and shapes of the tensors inside)
  3. the full traceback if the prefix forward raises
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "monkey-verifier" / "src"))

# The verifier needs MODEL_DIR set so AutoTokenizer.from_pretrained finds
# the local LLaVA checkpoint. Match what bon_eval_sweep.sbatch sets.
os.environ.setdefault("MODEL_DIR", str(REPO_ROOT / "monkey-verifier" / "model_dir"))


def main() -> int:
    print(f"[debug] MODEL_DIR={os.environ['MODEL_DIR']}")
    print("[debug] importing RobotRewardModel…", flush=True)
    from infer_server import RobotRewardModel  # noqa: WPS433
    rrm = RobotRewardModel(use_kv_cache_prefix=True)
    print("[debug] verifier loaded.")

    # Build a tiny batch (K=4) of identical actions. We just need
    # `_forward_kv_cache_prefix` to *execute*; the values aren't checked.
    instruction = "put eggplant into yellow basket"
    actions = np.zeros((4, 7), dtype=np.float32)
    # cache the template (populates rrm._template_cache[instruction])
    rrm._build_input_ids(instruction, actions)
    input_ids = rrm._build_input_ids(instruction, actions)
    attention_mask = input_ids.ne(rrm.tokenizer.pad_token_id).long()

    # Fake image: 384x384 white square.
    fake_img = np.full((384, 384, 3), 255, dtype=np.uint8)
    image_tensor = rrm._process_image(fake_img)
    images = image_tensor.unsqueeze(0).expand(actions.shape[0], -1, -1, -1)

    device = next(rrm.model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    images = images.to(device=device, dtype=torch.bfloat16)

    # --- Reproduce the early lines of _forward_kv_cache_prefix manually so we
    # can inspect past_key_values before it gets indexed.
    _, action_slot_idx = rrm._template_cache[instruction]
    K = input_ids.shape[0]
    suffix_start = action_slot_idx
    backbone = rrm.model.backbone_model
    backbone.set_adapter(rrm.model.adapter_name)
    backbone.config.use_cache = True
    print(f"[debug] backbone class: {type(backbone).__name__}")
    print(f"[debug] backbone.config.use_cache = {backbone.config.use_cache}")
    print(f"[debug] action_slot_idx = {action_slot_idx}  K = {K}")

    prefix_ids = input_ids[:1, :suffix_start]
    prefix_mask = attention_mask[:1, :suffix_start]
    print(f"[debug] prefix_ids.shape = {tuple(prefix_ids.shape)}")

    with torch.inference_mode():
        prefix_out = backbone(
            input_ids=prefix_ids,
            attention_mask=prefix_mask,
            images=images[:1],
            use_cache=True,
            return_dict=True,
        )

    print(f"\n[debug] prefix_out type: {type(prefix_out).__name__}")
    if hasattr(prefix_out, "__dict__"):
        attrs = list(vars(prefix_out).keys())
    elif hasattr(prefix_out, "keys"):
        attrs = list(prefix_out.keys())
    else:
        attrs = [a for a in dir(prefix_out) if not a.startswith("_")]
    print(f"[debug] prefix_out attrs: {attrs}")

    pkv = getattr(prefix_out, "past_key_values", None)
    print(f"\n[debug] past_key_values type: {type(pkv).__name__ if pkv is not None else 'None'}")

    if pkv is None:
        print("[debug] >>> past_key_values is None — model isn't propagating KV through this code path.")
        print("[debug] Likely fix paths:")
        print("        - call backbone.base_model(...) to bypass PEFT wrapping")
        print("        - or call backbone.get_decoder()(...) directly")
        return 0

    # DynamicCache path
    if hasattr(pkv, "key_cache"):
        print(f"[debug] DynamicCache: num key_cache entries = {len(pkv.key_cache)}")
        for i, (k, v) in enumerate(zip(pkv.key_cache, pkv.value_cache)):
            print(f"  layer {i}: key={ 'None' if k is None else tuple(k.shape) }  "
                  f"value={ 'None' if v is None else tuple(v.shape) }")
        return 0

    # Legacy tuple path
    print(f"[debug] legacy tuple: len(pkv) = {len(pkv)}")
    for i, layer in enumerate(pkv):
        if layer is None:
            print(f"  layer {i}: None")
            continue
        if not isinstance(layer, (tuple, list)):
            print(f"  layer {i}: type={type(layer).__name__}")
            continue
        ks = ", ".join(
            "None" if t is None else f"shape={tuple(t.shape)}"
            for t in layer
        )
        print(f"  layer {i}: {ks}")

    # Finally try the actual prefix forward and dump the traceback if it fails.
    print("\n[debug] now attempting _forward_kv_cache_prefix(...)")
    try:
        with torch.inference_mode():
            rewards = rrm._forward_kv_cache_prefix(
                instruction, input_ids, attention_mask, images,
            )
        print(f"[debug] OK — rewards shape: {tuple(rewards.shape)}  values: {rewards.float().cpu().tolist()}")
    except Exception:
        print("[debug] _forward_kv_cache_prefix raised:")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
