"""Bisect the prefix-KV exactness gap with a single (C=2) image set.

Three rewards per row:
  base    = get_rewards_paired                      (production full forward)
  direct  = decoder(full multimodal embeds)          NO KV split — drives
            backbone.get_decoder() directly on the WHOLE sequence
  kvpref  = get_rewards_paired_kvprefix              prefix/suffix KV split

Interpretation:
  direct != base  -> bug is in driving the decoder directly (embeds / norm /
                     adapter / hidden-state readout), NOT the KV split.
  direct == base but kvpref != base -> bug is in the prefix/suffix KV split
                     (rotary / mask / cache stacking).
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


def reward_direct_full(rrm, instruction, images, actions, keys):
    """Drive backbone.get_decoder() on the FULL multimodal sequence (no split)."""
    actions = np.asarray(actions)
    B = actions.shape[0]
    device = next(rrm.model.parameters()).device
    backbone = rrm.model.backbone_model
    backbone.set_adapter(rrm.model.adapter_name)
    llava = rrm._llava_model()
    decoder = backbone.get_decoder()
    decoder.config.use_cache = True

    input_ids = rrm._build_input_ids(instruction, actions).to(device)
    attn = input_ids.ne(rrm.tokenizer.pad_token_id).long()
    img_tensors = torch.stack([
        rrm._process_image_cached(images[i], keys[i]) for i in range(B)
    ]).to(device=device, dtype=torch.bfloat16)

    prev_b, prev_k = rrm._broadcast_image, rrm._cur_image_keys
    rrm._broadcast_image = False
    rrm._cur_image_keys = list(keys)
    try:
        with torch.inference_mode():
            prep = llava.prepare_inputs_labels_for_multimodal(
                input_ids, attn, None, None, img_tensors)
            full_embeds, full_attn = prep[3], prep[1]
            dec_out = decoder(
                inputs_embeds=full_embeds,
                attention_mask=full_attn,
                use_cache=False,
                return_dict=True,
            )
            last_hidden = dec_out.last_hidden_state[:, -1, :]
            last_hidden = last_hidden.type_as(rrm.model.reward_head.weight)
            rewards = rrm.model.reward_head(last_hidden).squeeze(-1)
    finally:
        rrm._broadcast_image, rrm._cur_image_keys = prev_b, prev_k
    return rewards.detach().float().cpu().tolist()


def main() -> int:
    from infer_server import RobotRewardModel
    rrm = RobotRewardModel()
    print("[dbg3] verifier loaded.", flush=True)
    rng = np.random.RandomState(0)

    for C in (1, 2):
        images = [rng.randint(0, 256, size=(256, 256, 3), dtype=np.uint8) for _ in range(C)]
        keys = [rrm._raw_image_key(im) for im in images]
        actions = rng.uniform(-1, 1, size=(C, 7)).astype(np.float32)

        base = np.asarray(rrm.get_rewards_paired(INSTRUCTION, images, actions), dtype=np.float64)
        direct = np.asarray(reward_direct_full(rrm, INSTRUCTION, images, actions, keys),
                            dtype=np.float64)
        rrm.clear_prefix_cache()
        kv = np.asarray(rrm.get_rewards_paired_kvprefix(INSTRUCTION, images, actions, keys),
                        dtype=np.float64)
        print(f"\n[dbg3] C={C}: max|direct-base|={np.abs(direct-base).max():.4e}  "
              f"max|kv-base|={np.abs(kv-base).max():.4e}")
        for i in range(C):
            print(f"       row {i}: base={base[i]:+.5f}  direct={direct[i]:+.5f}  "
                  f"kv={kv[i]:+.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
