"""Localize the KV-split bug to embeds-reconstruction vs past-reuse (C=1).

  base   = get_rewards_paired
  recon  = prefill prefix_embeds, cat with suffix embeds, ONE decoder forward,
           NO past_key_values  (tests embeds reconstruction only)
  kv     = get_rewards_paired_kvprefix  (prefill + suffix WITH past)

  recon == base, kv != base  -> bug is in past_key_values consumption.
  recon != base              -> bug is in prefix_embeds reconstruction.

Also prints P (prefix embed len), S (suffix len), T (full len) to confirm
T == P + S, and dumps prefix_attn uniqueness.
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


def _prefix_embeds(rrm, instruction, images, keys, suffix_start):
    """Reproduce the kvprefix prefill's prefix embeds (no decoder)."""
    device = next(rrm.model.parameters()).device
    backbone = rrm.model.backbone_model
    backbone.set_adapter(rrm.model.adapter_name)
    llava = rrm._llava_model()
    template_ids, _ = rrm._template_cache[instruction]
    B = len(images)
    miss_imgs = torch.stack([
        rrm._process_image_cached(images[i], keys[i]) for i in range(B)
    ]).to(device=device, dtype=torch.bfloat16)
    prefix_ids = template_ids[:suffix_start].unsqueeze(0).repeat(B, 1).to(device)
    prefix_mask = torch.ones_like(prefix_ids)
    prev_b, prev_k = rrm._broadcast_image, rrm._cur_image_keys
    rrm._broadcast_image = False
    rrm._cur_image_keys = list(keys)
    try:
        prep = llava.prepare_inputs_labels_for_multimodal(
            prefix_ids, prefix_mask, None, None, miss_imgs)
    finally:
        rrm._broadcast_image, rrm._cur_image_keys = prev_b, prev_k
    return prep[3], prep[1]   # prefix_embeds, prefix_attn


def reward_recon(rrm, instruction, images, actions, keys):
    actions = np.asarray(actions)
    B = actions.shape[0]
    device = next(rrm.model.parameters()).device
    backbone = rrm.model.backbone_model
    decoder = backbone.get_decoder()
    decoder.config.use_cache = True
    embed_tokens = decoder.embed_tokens
    if instruction not in rrm._template_cache:
        rrm._template_cache[instruction] = rrm._build_template(instruction)
    _, suffix_start = rrm._template_cache[instruction]

    with torch.inference_mode():
        prefix_embeds, prefix_attn = _prefix_embeds(rrm, instruction, images, keys, suffix_start)
        input_ids = rrm._build_input_ids(instruction, actions).to(device)
        suffix_ids = input_ids[:, suffix_start:]
        suffix_embeds = embed_tokens(suffix_ids)
        P, S = prefix_embeds.shape[1], suffix_embeds.shape[1]
        full_embeds = torch.cat([prefix_embeds, suffix_embeds], dim=1)
        full_attn = torch.cat(
            [prefix_attn, torch.ones((B, S), dtype=prefix_attn.dtype, device=device)], dim=1)
        dec = decoder(inputs_embeds=full_embeds, attention_mask=full_attn,
                      use_cache=False, return_dict=True)
        last_hidden = dec.last_hidden_state[:, -1, :].type_as(rrm.model.reward_head.weight)
        rewards = rrm.model.reward_head(last_hidden).squeeze(-1)
    print(f"       [recon] P={P} S={S} P+S={P+S}")
    return rewards.detach().float().cpu().tolist()


def main() -> int:
    from infer_server import RobotRewardModel
    rrm = RobotRewardModel()
    print("[dbg5] verifier loaded.", flush=True)
    rng = np.random.RandomState(0)
    C = 1
    images = [rng.randint(0, 256, size=(256, 256, 3), dtype=np.uint8) for _ in range(C)]
    keys = [rrm._raw_image_key(im) for im in images]
    actions = rng.uniform(-1, 1, size=(C, 7)).astype(np.float32)

    base = np.asarray(rrm.get_rewards_paired(INSTRUCTION, images, actions), dtype=np.float64)
    recon = np.asarray(reward_recon(rrm, INSTRUCTION, images, actions, keys), dtype=np.float64)
    rrm.clear_prefix_cache()
    kv = np.asarray(rrm.get_rewards_paired_kvprefix(INSTRUCTION, images, actions, keys),
                    dtype=np.float64)
    print(f"[dbg5] max|recon-base|={np.abs(recon-base).max():.4e}  "
          f"max|kv-base|={np.abs(kv-base).max():.4e}")
    for i in range(C):
        print(f"       row {i}: base={base[i]:+.5f}  recon={recon[i]:+.5f}  kv={kv[i]:+.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
