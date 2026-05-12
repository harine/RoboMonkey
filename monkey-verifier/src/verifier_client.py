"""Unified client for the RoboMonkey reward model.

Used by:
  * `RoboMonkey/scriptsv2/eval_diffusion/eval_diffusion.py`  (BoN eval)
  * `diffusion_policy/diffusion_policy/policy/verifiers.py`  (L2S train)

Two backends, chosen by `server_url`:
  * ``"in_process"`` (or `""` / `None`) — load `RobotRewardModel` in this
    Python process. No HTTP, no disk image; all K (or B) rows go through
    a single batched GPU call.
  * ``"http://host:port"`` — POST to `/process` on the verifier server.

Use ``VerifierClient.from_port(port)`` for a numeric switch:
``port <= 0`` → in-process, otherwise → ``http://127.0.0.1:<port>``.
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

_HERE = Path(__file__).resolve().parent
ImageLike = Union[str, np.ndarray, Any]  # str path | HxWx3 uint8 | PIL.Image


def _ensure_on_path() -> None:
    p = str(_HERE)
    if p not in sys.path:
        sys.path.insert(0, p)


def _hash_image(img: ImageLike) -> Tuple:
    """Stable key for an image input. Path strings short-circuit hashing."""
    if isinstance(img, str):
        return ("path", img)
    if isinstance(img, np.ndarray):
        h = hashlib.blake2b(
            np.ascontiguousarray(img).tobytes(), digest_size=16
        ).digest()
        return ("nd", img.shape, img.dtype.str, h)
    try:
        from PIL.Image import Image as _PILImage
        if isinstance(img, _PILImage):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return ("pil", hashlib.blake2b(buf.getvalue(), digest_size=16).digest())
    except Exception:
        pass
    return ("repr", repr(img))


class VerifierClient:
    """Process-local cache of an in-process verifier + thin HTTP client."""

    _IN_PROCESS_MODEL: Optional[Any] = None
    _LOAD_LOCK = threading.Lock()

    def __init__(self, server_url: Optional[str], request_timeout: float = 300.0):
        if server_url in (None, "", "in_process", "0", "in-process"):
            self.server_url = "in_process"
            self.in_process = True
        else:
            self.server_url = server_url.rstrip("/")
            self.in_process = False
        self.request_timeout = float(request_timeout)

        # (instruction, image_hash, action_tokens) -> reward LRU. Helps when
        # the same (image, action_token_tuple) is queried multiple times
        # (e.g. converged-policy L2S, BoN sweeps over the same env state).
        cache_size = int(os.environ.get("ROBOMONKEY_REWARD_CACHE_SIZE", 100_000))
        self._reward_cache: Optional["OrderedDict[tuple, float]"] = (
            OrderedDict() if cache_size > 0 else None
        )
        self._reward_cache_size = cache_size
        self._reward_cache_hits = 0
        self._reward_cache_misses = 0
        self._reward_cache_lock = threading.Lock()

        if self.in_process:
            self._ensure_loaded()

    def _cache_key(self, instruction: str, image: ImageLike, action_row) -> tuple:
        return (instruction, _hash_image(image),
                tuple(int(x) for x in np.asarray(action_row).reshape(-1)))

    def _cache_get(self, key: tuple) -> Optional[float]:
        if self._reward_cache is None:
            return None
        with self._reward_cache_lock:
            v = self._reward_cache.get(key)
            if v is not None:
                self._reward_cache.move_to_end(key)
                self._reward_cache_hits += 1
                return v
            self._reward_cache_misses += 1
            return None

    def _cache_put(self, key: tuple, reward: float) -> None:
        if self._reward_cache is None:
            return
        with self._reward_cache_lock:
            self._reward_cache[key] = float(reward)
            while len(self._reward_cache) > self._reward_cache_size:
                self._reward_cache.popitem(last=False)

    def cache_stats(self) -> dict:
        total = self._reward_cache_hits + self._reward_cache_misses
        return {
            "size": 0 if self._reward_cache is None else len(self._reward_cache),
            "capacity": self._reward_cache_size,
            "hits": self._reward_cache_hits,
            "misses": self._reward_cache_misses,
            "hit_rate": (self._reward_cache_hits / total) if total else 0.0,
        }

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_port(cls, port: int, **kw: Any) -> "VerifierClient":
        port = int(port)
        if port <= 0:
            return cls("in_process", **kw)
        return cls(f"http://127.0.0.1:{port}", **kw)

    # ------------------------------------------------------------------
    # In-process loading
    # ------------------------------------------------------------------

    @classmethod
    def _ensure_loaded(cls) -> Any:
        if cls._IN_PROCESS_MODEL is None:
            with cls._LOAD_LOCK:
                if cls._IN_PROCESS_MODEL is None:
                    _ensure_on_path()
                    from infer_server import RobotRewardModel  # noqa: WPS433
                    print("[verifier_client] Loading in-process RoboMonkey verifier...")
                    cls._IN_PROCESS_MODEL = RobotRewardModel()
                    print("[verifier_client] In-process verifier ready.")
        return cls._IN_PROCESS_MODEL

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_check(self) -> None:
        if self.in_process:
            self._ensure_loaded()
            return
        import requests
        try:
            r = requests.get(self.server_url + "/", timeout=5)
            r.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"Verifier server unreachable at {self.server_url}/: {e!r}. "
                "Start `infer_server.py` or pass server_url='in_process'."
            ) from e

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_candidates(
        self,
        instruction: str,
        image: ImageLike,
        actions: np.ndarray,
    ) -> List[float]:
        """K candidate 7-D actions for a single (instruction, image).

        In-process: one batched GPU call.
        HTTP: one POST; `image` must be a filesystem path.
        """
        actions = np.asarray(actions)
        if not self.in_process and not isinstance(image, str):
            raise TypeError(
                "HTTP verifier mode requires `image` to be a filesystem path; "
                f"got {type(image)!r}."
            )

        if self._reward_cache is None:
            if self.in_process:
                return self._IN_PROCESS_MODEL.get_rewards(instruction, image, actions)
            return self._http_post(instruction, image, actions)

        K = actions.shape[0]
        rewards: List[Optional[float]] = [None] * K
        miss_idxs: List[int] = []
        miss_keys: List[tuple] = []
        for i in range(K):
            key = self._cache_key(instruction, image, actions[i])
            cached = self._cache_get(key)
            if cached is not None:
                rewards[i] = cached
            else:
                miss_idxs.append(i)
                miss_keys.append(key)
        if miss_idxs:
            miss_actions = actions[miss_idxs]
            if self.in_process:
                new_rewards = self._IN_PROCESS_MODEL.get_rewards(
                    instruction, image, miss_actions
                )
            else:
                new_rewards = self._http_post(instruction, image, miss_actions)
            for j, i in enumerate(miss_idxs):
                rewards[i] = new_rewards[j]
                self._cache_put(miss_keys[j], new_rewards[j])
        return [float(r) for r in rewards]

    def score_paired(
        self,
        instruction: str,
        images: Union[Sequence[ImageLike], np.ndarray],
        actions: np.ndarray,
        max_workers: int = 8,
    ) -> List[float]:
        """B (image, action) pairs.

        In-process: one batched GPU call (each row has its own image).
        HTTP: parallel single-action POSTs (one per pair).
        """
        actions = np.asarray(actions)
        B = actions.shape[0]
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images_list: List[ImageLike] = [images] * B
        else:
            images_list = list(images)
        if len(images_list) != B:
            raise ValueError(
                f"len(images)={len(images_list)} != actions B={B}"
            )

        if not self.in_process:
            for img in images_list:
                if not isinstance(img, str):
                    raise TypeError(
                        "HTTP verifier mode requires every image to be a filesystem "
                        f"path; got {type(img)!r}."
                    )

        if self._reward_cache is None:
            if self.in_process:
                return self._IN_PROCESS_MODEL.get_rewards_paired(
                    instruction, images_list, actions
                )
            return self._http_post_paired(instruction, images_list, actions, max_workers)

        # Cached path: look up each row, forward only the missing ones.
        rewards: List[Optional[float]] = [None] * B
        miss_idxs: List[int] = []
        miss_keys: List[tuple] = []
        for i in range(B):
            key = self._cache_key(instruction, images_list[i], actions[i])
            cached = self._cache_get(key)
            if cached is not None:
                rewards[i] = cached
            else:
                miss_idxs.append(i)
                miss_keys.append(key)
        if miss_idxs:
            miss_images = [images_list[i] for i in miss_idxs]
            miss_actions = actions[miss_idxs]
            if self.in_process:
                new_rewards = self._IN_PROCESS_MODEL.get_rewards_paired(
                    instruction, miss_images, miss_actions
                )
            else:
                new_rewards = self._http_post_paired(
                    instruction, miss_images, miss_actions, max_workers
                )
            for j, i in enumerate(miss_idxs):
                rewards[i] = new_rewards[j]
                self._cache_put(miss_keys[j], new_rewards[j])
        return [float(r) for r in rewards]

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _http_post(self, instruction: str, image_path: str, actions: np.ndarray) -> List[float]:
        import requests
        payload = {
            "instruction": instruction,
            "image_path": image_path,
            "action": actions.tolist(),
        }
        r = requests.post(
            self.server_url + "/process", json=payload, timeout=self.request_timeout
        )
        r.raise_for_status()
        return [float(x) for x in r.json()["rewards"]]

    def _http_post_paired(
        self,
        instruction: str,
        image_paths: List[str],
        actions: np.ndarray,
        max_workers: int,
    ) -> List[float]:
        import concurrent.futures
        B = actions.shape[0]
        rewards = [0.0] * B

        def _one(idx: int) -> None:
            rewards[idx] = self._http_post(
                instruction, image_paths[idx], actions[idx:idx + 1]
            )[0]

        if B == 1 or max_workers <= 1:
            for i in range(B):
                _one(i)
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(max_workers, B)
            ) as pool:
                list(pool.map(_one, range(B)))
        return rewards
