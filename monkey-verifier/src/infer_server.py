import hashlib
import io
import os
import time
from collections import OrderedDict
from typing import Union

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, Request
from transformers import AutoTokenizer, set_seed
import uvicorn
import json_numpy as json
import torch

from reward_model_utils import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    DisableLogger,
)
from lora_utils import print_trainable_parameters
from models.reward_model import RewardConfig, RewardModel
from action_processing import ActionTokenizer
from llava import conversation as conversation_lib
from llava.conversation import conv_templates
from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.train.train import smart_tokenizer_and_embedding_resize


PLACEHOLDER_TOKEN_ID = 12983  # tokenizer id of " placeholder"


# Resolve MODEL_DIR at import time so the in-process verifier finds the
# checkpoint regardless of CWD; previously this default was only set inside
# the `__main__` block, so importing `RobotRewardModel` from another process
# fell back to a CWD-relative "./model_dir" that doesn't exist.
os.environ.setdefault(
    "MODEL_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "model_dir")),
)


class RobotRewardModel:
    """LLaVA-based reward model. Supports either an HTTP server (`infer_server.py`)
    or in-process use (`from infer_server import RobotRewardModel`)."""

    def __init__(self, use_kv_cache_prefix: bool = False):
        # Parse arguments from environment variables and defaults based on the shell script
        model_args = ModelArguments(
            model_name_or_path=os.path.join(os.environ.get("MODEL_DIR", "./model_dir"),
                                          "llava-v1.5-7b/sft_model/"),
            vision_tower="openai/clip-vit-large-patch14-336",
            mm_vision_select_layer=-2,
            mm_use_im_start_end=False,
            mm_use_im_patch_token=False,
            version="v1"
        )

        data_args = DataArguments(
            image_aspect_ratio='pad',
            is_multimodal=True,
            reward_prompt_file="./prompts/robot_reward_prompt.txt"
        )

        training_args = TrainingArguments(
            model_max_length=2048,
            query_len=1280,
            response_len=768,
            bits=16,
            lora_r=64,
            lora_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            output_dir=os.path.join(os.environ.get("MODEL_DIR", "./model_dir"),
                                   "llava-v1.5-7b"),
            freeze_mm_mlp_adapter=True,
            group_by_length=False,
            bf16=True,
            seed=42
        )

        # Set seed for deterministic behavior
        set_seed(42)
        torch.manual_seed(42)
        np.random.seed(42)

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="left",
            truncation_side="right",
            use_fast=False,
        )

        # Handle tokenizer configuration
        if model_args.version == "v0":
            if tokenizer.pad_token is None:
                smart_tokenizer_and_embedding_resize(
                    special_tokens_dict=dict(pad_token="[PAD]"),
                    tokenizer=tokenizer,
                    model=None,
                )
        elif model_args.version == "v0.5":
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.unk_token
            if model_args.version in conversation_lib.conv_templates:
                conversation_lib.default_conversation = conversation_lib.conv_templates[
                    model_args.version
                ]
            else:
                conversation_lib.default_conversation = conversation_lib.conv_templates[
                    "vicuna_v1"
                ]

        # Initialize model
        if model_args.vision_tower is not None:
            config = RewardConfig(backbone_model_name_or_path=model_args.model_name_or_path)

            with DisableLogger():
                args = type('Args', (), {})()
                for key, value in vars(model_args).items():
                    setattr(args, key, value)
                for key, value in vars(data_args).items():
                    setattr(args, key, value)
                for key, value in vars(training_args).items():
                    setattr(args, key, value)

                model = RewardModel(
                    args=args,
                    config=config,
                    qlora=True,
                    checkpoint_dir=os.path.join(os.environ.get("MODEL_DIR", "./model_dir"), "lora_adapter"),
                    tokenizer=tokenizer,
                ).to(torch.bfloat16)

            model.backbone_model.config.use_cache = True
            print_trainable_parameters(args, model)
            print("Loaded model")

            with DisableLogger():
                model_temp = model.backbone_model

            vision_tower = model_temp.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()

            data_args.image_processor = vision_tower.image_processor
            model_temp.config.mm_use_im_start_end = model_args.mm_use_im_start_end

            self.tokenizer = tokenizer
            self.model = model
            self.model.eval()
            self.data_args = data_args

            # Cached helpers (formerly rebuilt every request).
            self._action_tokenizer = ActionTokenizer(tokenizer)
            self._conv_template = conv_templates["vicuna_v1"]
            self._template_cache: dict = {}
            self.use_kv_cache_prefix = bool(use_kv_cache_prefix)

            # Encode image only once per request even when batched K times.
            self._patch_encode_images_broadcast()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _llava_model(self):
        """Return the inner module that owns `encode_images` (under PEFT wrappers)."""
        m = self.model.backbone_model
        seen = set()
        while id(m) not in seen:
            seen.add(id(m))
            if hasattr(m, "encode_images") and callable(getattr(m, "encode_images")):
                return m
            for attr in ("base_model", "model"):
                if hasattr(m, attr):
                    m = getattr(m, attr)
                    break
            else:
                break
        raise RuntimeError("Could not locate LlavaMetaForCausalLM with encode_images")

    def _patch_encode_images_broadcast(self):
        """Encode the unique image once when all rows of the K-batch share it.
        Toggle off via `self._broadcast_image = False` for paired scoring
        where each row has a distinct image.

        In paired mode also serves an LRU cache of per-image features so the
        same image (e.g. one batch element revisited across L2S's `max_actions`
        verifier calls) only hits the vision tower once.
        """
        ll = self._llava_model()
        orig = ll.encode_images
        self._broadcast_image = True
        self._image_feat_cache: "OrderedDict[bytes, torch.Tensor]" = OrderedDict()
        # Default raised 128 -> 512 so the per-image CLIP-feature cache survives a
        # full L2S training batch (B=256) and is reused across the policy's
        # `max_actions` autoregressive verifier calls instead of thrashing.
        self._image_feat_cache_size = int(
            os.environ.get("ROBOMONKEY_IMAGE_FEAT_CACHE_SIZE", 512)
        )
        # Per-row raw-image keys for the *current* paired forward, set by
        # `_paired_chunk_forward`. When present (and length-matched) the feature
        # cache keys on the cheap raw-image hash instead of re-hashing the large
        # preprocessed float tensor every round. None outside paired scoring.
        self._cur_image_keys = None
        self._feat_cache_hits = 0
        self._feat_cache_misses = 0
        # Cache of preprocessed CLIP-input tensors (CPU), keyed by raw-image hash,
        # so `_process_image` (PIL resize + JPEG roundtrip + CLIP preprocess) runs
        # once per unique image instead of once per (image, action) verifier call.
        self._proc_image_cache: "OrderedDict[bytes, torch.Tensor]" = OrderedDict()
        self._proc_image_cache_size = int(
            os.environ.get("ROBOMONKEY_PROC_IMAGE_CACHE_SIZE",
                           self._image_feat_cache_size)
        )
        self._proc_cache_hits = 0
        self._proc_cache_misses = 0

        def _hash_row(t: torch.Tensor) -> bytes:
            arr = t.detach().to(torch.float32).cpu().numpy()
            return hashlib.blake2b(np.ascontiguousarray(arr).tobytes(),
                                   digest_size=16).digest()

        def encode_images_broadcast(images):
            if self._broadcast_image and images.shape[0] > 1:
                feats = orig(images[:1])
                return feats.expand(images.shape[0], *feats.shape[1:]).contiguous()

            cache = self._image_feat_cache
            cap = self._image_feat_cache_size
            if cap <= 0 or images.shape[0] == 0:
                return orig(images)

            B = images.shape[0]
            if self._cur_image_keys is not None and len(self._cur_image_keys) == B:
                keys = self._cur_image_keys
            else:
                keys = [_hash_row(images[i]) for i in range(B)]
            feat_rows: list = [None] * B
            missing: list = []
            for i, k in enumerate(keys):
                cached = cache.get(k)
                if cached is not None:
                    cache.move_to_end(k)
                    feat_rows[i] = cached
                    self._feat_cache_hits += 1
                else:
                    missing.append(i)
            if missing:
                self._feat_cache_misses += len(missing)
                new_feats = orig(torch.stack([images[i] for i in missing], dim=0))
                for j, i in enumerate(missing):
                    f = new_feats[j].detach()
                    feat_rows[i] = f
                    cache[keys[i]] = f
                    if len(cache) > cap:
                        cache.popitem(last=False)
            return torch.stack(feat_rows, dim=0)

        ll.encode_images = encode_images_broadcast

    def _build_template(self, instruction: str):
        """Tokenize the full prompt once. Return (template_ids, action_slot_idx).
        Token IDs at [action_slot_idx, action_slot_idx+7) are placeholders to be
        overwritten with per-candidate action token IDs."""
        instruction = instruction.lower().rstrip('.')
        action_holder = ' '.join(['placeholder'] * 7)

        inp = (f"shows the current observation from the robot's wrist-mounted camera. "
               f"The robot manipulation arm is attempting to {instruction}. "
               f"What action should the robot take to effectively accomplish the task? "
               f"ASSISTANT: The robot should take the action: {action_holder} </s> "
               f"USER: Please evaluate the quality of the robot action. "
               f"A good robot action should consider different factors, "
               f"especially interactions with surrounding objects and human preferences.\n"
               f"ASSISTANT: Based on how humans would control the robot arm and the "
               f"awareness of the situation, the quality score of the robot action is")
        inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv = self._conv_template.copy()
        conv.append_message(conv.roles[0], inp)
        prompt = conv.get_prompt()
        prompt = prompt.replace("<image>", " placeholder ")

        in_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length + 2,
            truncation=True,
        ).input_ids

        placeholders = (in_ids == PLACEHOLDER_TOKEN_ID).nonzero()
        image_idx = placeholders[0][1].item()
        in_ids[0, image_idx:image_idx + 1] = -200
        action_slot_idx = (in_ids == PLACEHOLDER_TOKEN_ID).nonzero()[0][1].item()
        in_ids = in_ids[:, :-1]
        return in_ids.squeeze(0).to(torch.long), int(action_slot_idx)

    def _process_image(self, image_input: Union[str, np.ndarray, Image.Image]):
        processor = self.data_args.image_processor
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            # Match the HTTP path's resize-to-256 + JPEG quality=95 roundtrip
            # in `eval_diffusion.save_reward_image` so in-process scoring
            # produces the same CLIP input as the HTTP path.
            image = Image.fromarray(np.ascontiguousarray(image_input).astype(np.uint8)).convert("RGB")
            if image.size != (256, 256):
                image = image.resize((256, 256), Image.LANCZOS)
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=95)
            buf.seek(0)
            image = Image.open(buf).convert("RGB")
        else:
            image = image_input.convert("RGB")

        if self.data_args.image_aspect_ratio == "pad":
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(
                image, tuple(int(x * 255) for x in processor.image_mean)
            )
        return processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _build_input_ids(self, instruction: str, actions: np.ndarray) -> torch.Tensor:
        if instruction not in self._template_cache:
            self._template_cache[instruction] = self._build_template(instruction)
        template_ids, action_slot_idx = self._template_cache[instruction]
        K = actions.shape[0]

        input_ids = template_ids.unsqueeze(0).repeat(K, 1)
        # Encode actions: float -> ActionTokenizer; int -> assume pre-tokenized.
        if actions.dtype.kind == "f":
            tok_rows = np.stack([self._action_tokenizer(row) for row in actions], axis=0)
        else:
            tok_rows = actions.astype(np.int64, copy=False)
        action_tokens = torch.from_numpy(tok_rows - 1000).to(torch.long)
        input_ids[:, action_slot_idx:action_slot_idx + 7] = action_tokens
        return input_ids

    def get_rewards(self, instruction: str, image, actions) -> list:
        """Score K candidate 7D actions for a single (image, instruction).

        `image` may be a path (str), a numpy uint8 array (HxWx3), or a PIL image.
        `actions` is (K, 7); float rows are tokenized via `ActionTokenizer`,
        int rows are treated as OpenVLA token IDs.
        """
        actions = np.asarray(actions)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"actions must be (K, 7); got {actions.shape}")
        K = actions.shape[0]

        input_ids = self._build_input_ids(instruction, actions)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        image_tensor = self._process_image(image)
        # Vision tower runs once on the first row; patched encode_images broadcasts.
        images = image_tensor.unsqueeze(0).expand(K, -1, -1, -1)

        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        images = images.to(device=device, dtype=torch.bfloat16)

        with torch.inference_mode():
            rewards = None
            if self.use_kv_cache_prefix and K > 1:
                try:
                    rewards = self._forward_kv_cache_prefix(
                        instruction, input_ids, attention_mask, images
                    )
                except Exception as e:
                    # Disable on the first failure so we don't keep paying the
                    # prefix forward only to fall back each call.
                    print(f"[verifier] KV-cache prefix path failed ({e!r}); "
                          f"falling back to single batched forward.")
                    self.use_kv_cache_prefix = False
            if rewards is None:
                scores = self.model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                )
                rewards = scores.rewards
        return rewards.detach().float().cpu().tolist()

    def get_rewards_paired(
        self,
        instruction: str,
        images: list,
        actions,
        chunk_size: int = None,
    ) -> list:
        """Score B (image, action) pairs.

        Splits the B rows into chunks of at most `chunk_size` so the GPU
        memory cost is bounded by chunk_size full LLaVA forwards rather than
        B (a 256-batch L2S training call OOMs a 32 GB GPU easily). Default
        chunk size is `ROBOMONKEY_PAIRED_CHUNK_SIZE` (env var, default 4).

        `images` is a length-B list (each entry: path str, HxWx3 uint8 ndarray,
        or PIL.Image). `actions` is (B, 7), same dtype semantics as
        `get_rewards`.
        """
        actions = np.asarray(actions)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"actions must be (B, 7); got {actions.shape}")
        B = actions.shape[0]
        if len(images) != B:
            raise ValueError(
                f"len(images)={len(images)} must match actions B={B}"
            )
        if chunk_size is None:
            chunk_size = int(os.environ.get("ROBOMONKEY_PAIRED_CHUNK_SIZE", 4))
        chunk_size = max(1, int(chunk_size))

        # Each row is a distinct image: disable the encode-once broadcast.
        # KV-cache prefix sharing also doesn't apply (image differs per row).
        prev_broadcast = self._broadcast_image
        self._broadcast_image = False
        try:
            rewards: list = []
            for start in range(0, B, chunk_size):
                end = min(start + chunk_size, B)
                rewards.extend(self._paired_chunk_forward(
                    instruction, images[start:end], actions[start:end],
                ))
        finally:
            self._broadcast_image = prev_broadcast
        return rewards

    @staticmethod
    def _raw_image_key(image_input) -> bytes:
        """Cheap content hash of a raw verifier image (path str / HWC uint8
        ndarray / PIL.Image), used to dedupe identical images across the policy's
        autoregressive verifier calls. Same observation -> same bytes -> same key."""
        if isinstance(image_input, str):
            payload = image_input.encode("utf-8")
        elif isinstance(image_input, np.ndarray):
            payload = np.ascontiguousarray(image_input).tobytes()
        else:  # PIL.Image or array-like
            payload = np.ascontiguousarray(np.asarray(image_input)).tobytes()
        return hashlib.blake2b(payload, digest_size=16).digest()

    def _process_image_cached(self, image_input, key: bytes):
        """`_process_image` with an LRU cache keyed on the raw-image hash so the
        PIL/JPEG/CLIP preprocessing runs once per unique image, not once per
        (image, action) pair. Returns a CPU tensor (cloned from the cache)."""
        cache = self._proc_image_cache
        cap = self._proc_image_cache_size
        if cap <= 0:
            self._proc_cache_misses += 1
            return self._process_image(image_input)
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            self._proc_cache_hits += 1
            return cached
        self._proc_cache_misses += 1
        t = self._process_image(image_input)
        cache[key] = t
        if len(cache) > cap:
            cache.popitem(last=False)
        return t

    def _paired_chunk_forward(self, instruction: str, images: list, actions: np.ndarray) -> list:
        input_ids = self._build_input_ids(instruction, actions)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        # Dedupe preprocessing + CLIP encoding across the same images recurring
        # over the policy's autoregressive verifier calls: hash each raw image
        # once, reuse the preprocessed tensor, and hand the keys to the feature
        # cache (via `_cur_image_keys`) so the vision tower is skipped on hits.
        keys = [self._raw_image_key(img) for img in images]
        image_tensors = torch.stack(
            [self._process_image_cached(img, k) for img, k in zip(images, keys)]
        )

        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        image_tensors = image_tensors.to(device=device, dtype=torch.bfloat16)

        self._cur_image_keys = keys
        try:
            with torch.inference_mode():
                scores = self.model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=image_tensors,
                )
                rewards = scores.rewards
        finally:
            self._cur_image_keys = None
        return rewards.detach().float().cpu().tolist()

    @staticmethod
    def _expand_kv(past_kv, K: int):
        """Broadcast a (1, ...) KV cache across batch K. Supports both legacy
        tuple-of-tuples and `DynamicCache` returns from transformers."""
        if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
            from copy import copy as _copy
            new_cache = _copy(past_kv)
            new_cache.key_cache = [k.expand(K, -1, -1, -1).contiguous()
                                   for k in past_kv.key_cache]
            new_cache.value_cache = [v.expand(K, -1, -1, -1).contiguous()
                                     for v in past_kv.value_cache]
            return new_cache
        return tuple(
            (k.expand(K, -1, -1, -1).contiguous(),
             v.expand(K, -1, -1, -1).contiguous())
            for (k, v) in past_kv
        )

    def _forward_kv_cache_prefix(
        self,
        instruction: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill the shared prefix (image + text up to the action slot) once,
        then run a single batched forward over the K-candidate suffixes using
        the expanded KV cache."""
        _, action_slot_idx = self._template_cache[instruction]
        K = input_ids.shape[0]
        suffix_start = action_slot_idx

        backbone = self.model.backbone_model
        backbone.set_adapter(self.model.adapter_name)
        backbone.config.use_cache = True

        prefix_ids = input_ids[:1, :suffix_start]
        prefix_mask = attention_mask[:1, :suffix_start]
        prefix_out = backbone(
            input_ids=prefix_ids,
            attention_mask=prefix_mask,
            images=images[:1],
            use_cache=True,
            return_dict=True,
        )
        past_kv = prefix_out.past_key_values
        if hasattr(past_kv, "key_cache"):
            kv_len = past_kv.key_cache[0].shape[-2]
        else:
            kv_len = past_kv[0][0].shape[-2]

        expanded_kv = self._expand_kv(past_kv, K)

        suffix_ids = input_ids[:, suffix_start:]
        suffix_mask = attention_mask[:, suffix_start:]
        full_mask = torch.cat(
            [
                torch.ones((K, kv_len), dtype=attention_mask.dtype,
                           device=attention_mask.device),
                suffix_mask,
            ],
            dim=1,
        )

        suffix_out = backbone(
            input_ids=suffix_ids,
            attention_mask=full_mask,
            past_key_values=expanded_kv,
            images=None,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = suffix_out.hidden_states[-1][:, -1, :]
        last_hidden = last_hidden.type_as(self.model.reward_head.weight)
        return self.model.reward_head(last_hidden).squeeze(-1)

    # ------------------------------------------------------------------
    # Per-image prefix-KV cache (training: same images scored across the
    # policy's `max_actions` autoregressive rounds). The image+instruction
    # prefix (~96% of the sequence) is encoded once per image and its KV is
    # reused across rounds; only the short action suffix is forwarded per round.
    # ------------------------------------------------------------------
    @staticmethod
    def _slice_kv(past_kv, j: int):
        """Slice row j of a batched KV cache -> list[(k,v)] per layer, batch 1."""
        if hasattr(past_kv, "key_cache"):
            pairs = zip(past_kv.key_cache, past_kv.value_cache)
        else:
            pairs = past_kv
        return [(k[j:j + 1].contiguous(), v[j:j + 1].contiguous()) for (k, v) in pairs]

    def _stack_kv(self, per_image):
        """Concatenate N per-image list[(k,v)] caches into one batch-N cache,
        matching the type the backbone produced (DynamicCache or legacy tuple)."""
        n_layers = len(per_image[0])
        Ks = [torch.cat([img[l][0] for img in per_image], dim=0) for l in range(n_layers)]
        Vs = [torch.cat([img[l][1] for img in per_image], dim=0) for l in range(n_layers)]
        proto = self._kv_proto
        if proto is not None and hasattr(proto, "key_cache"):
            from copy import copy as _copy
            new = _copy(proto)
            new.key_cache = Ks
            new.value_cache = Vs
            return new
        return tuple((Ks[l], Vs[l]) for l in range(n_layers))

    def clear_prefix_cache(self):
        """Drop the per-image prefix-KV cache (call between image chunks so GPU
        memory stays bounded by the chunk size, not the full batch)."""
        self._prefix_kv_cache = {}
        self._kv_proto = None

    def get_rewards_paired_kvprefix(self, instruction: str, images: list,
                                    actions, image_keys: list) -> list:
        """Score B (image, action) pairs reusing cached image+instruction prefix
        KV across calls with the same images (keyed by `image_keys`). The caller
        must pass a small enough chunk that B prefixes fit in GPU memory and must
        call `clear_prefix_cache()` between independent image chunks."""
        if not hasattr(self, "_prefix_kv_cache"):
            self._prefix_kv_cache = {}
            self._kv_proto = None
        actions = np.asarray(actions)
        B = actions.shape[0]
        if len(images) != B or len(image_keys) != B:
            raise ValueError("images, actions, image_keys must share length B")
        if instruction not in self._template_cache:
            self._template_cache[instruction] = self._build_template(instruction)
        template_ids, action_slot_idx = self._template_cache[instruction]
        suffix_start = action_slot_idx

        device = next(self.model.parameters()).device
        backbone = self.model.backbone_model
        backbone.set_adapter(self.model.adapter_name)
        # The PEFT/LLaVA *wrapper* forward drops past_key_values (returns a tuple
        # of None per layer). So we drive the inner Llama decoder directly: build
        # multimodal inputs_embeds via prepare_inputs_labels_for_multimodal, then
        # call get_decoder() with use_cache=True (which does populate KV).
        llava = self._llava_model()
        decoder = backbone.get_decoder()
        decoder.config.use_cache = True
        embed_tokens = decoder.embed_tokens
        cache = self._prefix_kv_cache

        with torch.inference_mode():
            # 1) Prefill prefix KV for any images not yet cached.
            miss = [i for i in range(B) if image_keys[i] not in cache]
            if miss:
                miss_imgs = torch.stack([
                    self._process_image_cached(images[i], image_keys[i]) for i in miss
                ]).to(device=device, dtype=torch.bfloat16)
                prefix_ids = template_ids[:suffix_start].unsqueeze(0).repeat(
                    len(miss), 1).to(device)
                prefix_mask = torch.ones_like(prefix_ids)
                # -> (input_ids, attention_mask, past, inputs_embeds, labels)
                prep = llava.prepare_inputs_labels_for_multimodal(
                    prefix_ids, prefix_mask, None, None, miss_imgs)
                prefix_embeds = prep[3]
                prefix_attn = prep[1]
                dec_out = decoder(
                    inputs_embeds=prefix_embeds,
                    attention_mask=prefix_attn,
                    use_cache=True,
                    return_dict=True,
                )
                past = dec_out.past_key_values
                self._kv_proto = past
                for j, i in enumerate(miss):
                    cache[image_keys[i]] = self._slice_kv(past, j)

            # 2) Stack the B images' cached prefixes and forward only the suffix.
            input_ids = self._build_input_ids(instruction, actions).to(device)
            attn = input_ids.ne(self.tokenizer.pad_token_id).long()
            suffix_ids = input_ids[:, suffix_start:]
            suffix_mask = attn[:, suffix_start:]
            stacked = self._stack_kv([cache[k] for k in image_keys])
            kv_len = (stacked.key_cache[0].shape[-2] if hasattr(stacked, "key_cache")
                      else stacked[0][0].shape[-2])
            full_mask = torch.cat(
                [torch.ones((B, kv_len), dtype=attn.dtype, device=device), suffix_mask],
                dim=1,
            )
            suffix_embeds = embed_tokens(suffix_ids)
            dec_suffix = decoder(
                inputs_embeds=suffix_embeds,
                attention_mask=full_mask,
                past_key_values=stacked,
                use_cache=False,
                return_dict=True,
            )
            last_hidden = dec_suffix.last_hidden_state[:, -1, :]
            last_hidden = last_hidden.type_as(self.model.reward_head.weight)
            rewards = self.model.reward_head(last_hidden).squeeze(-1)
        return rewards.detach().float().cpu().tolist()


# FastAPI application
app = FastAPI()
reward_model = None


@app.on_event("startup")
async def startup_event():
    global reward_model
    reward_model = RobotRewardModel()


@app.get("/")
async def read_root():
    return {"message": "RM server up"}


@app.post("/process")
async def process_data(request: Request):
    body = await request.body()
    data = json.loads(body)

    instruction = data.get("instruction")
    image_path = data.get("image_path")
    action = data.get("action")

    if not isinstance(instruction, str):
        raise HTTPException(status_code=400, detail="Instruction must be a string")
    if not isinstance(image_path, str):
        raise HTTPException(status_code=400, detail="Image path must be a string")

    action_array = np.array(action)

    if action_array.ndim != 2:
        raise HTTPException(status_code=400, detail="Action must be a 2D array")

    start_time = time.time()

    rewards = reward_model.get_rewards(instruction, image_path, action_array)

    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.4f} seconds")

    return {"rewards": rewards}


if __name__ == "__main__":
    # MODEL_DIR is already defaulted at module top-level so in-process imports
    # work too; here we just set the remaining server-only env vars.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("GPUS_PER_NODE", "1")

    uvicorn.run(app, host="0.0.0.0", port=3100)
