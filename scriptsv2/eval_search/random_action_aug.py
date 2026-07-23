"""Stratified-random action candidates to augment a search policy's pool.

Motivation (see data/eval/train_test_analysis/spoon/RESULTS.md): the search
transformer's N candidates are near-collapsed (mean pairwise cosine of their net
Δxyz ≈ 1), so the verifier has nothing to choose between. This module adds a
second pool of *deliberately diverse* candidates that covers the whole
translation-direction sphere, scores them with the SAME verifier, and lets the
caller argmax over the union.

Candidate construction (one chunk per sector per gripper state):
  * the unit sphere is partitioned into ``K`` equal-area cells (equal-area in
    cos(elevation) x azimuth) and ONE direction is drawn uniformly inside each
    cell -> near-360 coverage by construction, not by luck;
  * dims 0:3 (xyz delta) = center + dir * u * half_range, u ~ U(mag_min, 1),
    held CONSTANT across the horizon -> a coherent straight-line push, not jitter;
  * dims 3:6 (rpy) are copied from a reference chunk (the policy's argmax-Q
    candidate) so the rotation profile stays sane;
  * dim 6 (gripper) is forced open and closed (``gripper="both"``, M = 2K) so the
    verifier also gets to pick the grasp state, or copied (``gripper="copy"``, M = K).
All dims are clipped to the training action range read off the policy normalizer.

The scoring path mirrors ``eval_search.EVAL_CHUNK_SELECT``: the image obs is
repeated per candidate and ``verifier.get_value`` is called on raw (unnormalized)
``(M, horizon, action_dim)`` chunks -> values on the identical scale as the
policy's own candidate values. (DQCVerifier caches frame encodings by pixel
bytes, so repeating one frame M times costs one encode.)

Env knobs (read by :func:`config_from_env`):
  RAND_AUG_K        number of sectors; 0 / unset = augmentation off
  RAND_AUG_GRIPPER  both (default, M=2K) | copy (M=K)
  RAND_AUG_ONLY     1 = drop the policy candidates (verifier-only control)
  RAND_AUG_MAG_MIN  lower bound of the magnitude fraction u (default 0.25)
  RAND_AUG_BATCH    verifier scoring sub-batch (default 64)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

__all__ = ["AugConfig", "config_from_env", "action_bounds", "sector_dirs",
           "sample_random_chunks", "score_chunks", "augment"]


@dataclass
class AugConfig:
    k: int = 0                     # sectors; 0 = off
    gripper: str = "both"          # both | copy
    only: bool = False             # drop policy candidates
    mag_min: float = 0.25
    batch: int = 64

    @property
    def enabled(self) -> bool:
        return int(self.k) > 0

    @property
    def m(self) -> int:
        """Number of random candidates produced per env."""
        return int(self.k) * (2 if self.gripper == "both" else 1)


def config_from_env(env: Optional[Dict[str, str]] = None) -> AugConfig:
    e = os.environ if env is None else env
    return AugConfig(
        k=int(e.get("RAND_AUG_K", "0") or 0),
        gripper=str(e.get("RAND_AUG_GRIPPER", "both")).lower(),
        only=str(e.get("RAND_AUG_ONLY", "0")) == "1",
        mag_min=float(e.get("RAND_AUG_MAG_MIN", "0.25")),
        batch=int(e.get("RAND_AUG_BATCH", "64")),
    )


# --------------------------------------------------------------------------- #
# candidate construction
# --------------------------------------------------------------------------- #
def action_bounds(policy) -> Tuple[np.ndarray, np.ndarray]:
    """Per-dim raw action range [lo, hi] from the policy's action normalizer."""
    stats = policy.normalizer["action"].params_dict["input_stats"]
    lo = np.asarray(stats["min"].detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    hi = np.asarray(stats["max"].detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    return lo, hi


def _sector_grid(k: int) -> list:
    """Cells per elevation band: ``n_el`` bands uniform in cos(theta), with the K
    cells spread over them so EVERY band is fully wedged (no missing sector)."""
    k = int(k)
    n_el = max(1, int(round(np.sqrt(k / 2.0))))
    base, rem = divmod(k, n_el)
    return [base + (1 if i < rem else 0) for i in range(n_el)]


def sector_dirs(k: int, rng: np.random.Generator) -> np.ndarray:
    """K unit vectors, one drawn uniformly inside each of K near-equal-area cells.

    The sphere is cut into bands uniform in cos(theta) (equal area per band) and
    each band into its own azimuth wedges; one direction is drawn uniformly inside
    every cell, so the returned set covers all directions by construction.
    """
    per_band = _sector_grid(k)
    n_el = len(per_band)
    out = np.empty((int(k), 3), dtype=np.float64)
    i = 0
    for ie, n_az in enumerate(per_band):
        for ia in range(n_az):
            c = rng.uniform(-1.0 + 2.0 * ie / n_el, -1.0 + 2.0 * (ie + 1) / n_el)
            phi = rng.uniform(2.0 * np.pi * ia / n_az, 2.0 * np.pi * (ia + 1) / n_az)
            s = float(np.sqrt(max(0.0, 1.0 - c * c)))
            out[i] = (s * np.cos(phi), s * np.sin(phi), c)
            i += 1
    return out


def sample_random_chunks(bounds: Tuple[np.ndarray, np.ndarray], cfg: AugConfig,
                         horizon: int, ref_chunk: np.ndarray,
                         rng: np.random.Generator) -> np.ndarray:
    """-> (M, horizon, action_dim) raw action chunks. ``ref_chunk`` is (horizon, D)."""
    lo, hi = bounds
    D = int(ref_chunk.shape[-1])
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)

    dirs = sector_dirs(cfg.k, rng)                                   # (K,3)
    u = rng.uniform(cfg.mag_min, 1.0, size=(cfg.k, 1))               # (K,1)
    xyz = center[:3] + dirs * u * half[:3]                           # (K,3)

    reps = 2 if cfg.gripper == "both" else 1
    m = cfg.k * reps
    chunks = np.repeat(np.asarray(ref_chunk, dtype=np.float64)[None], m, axis=0)
    chunks[:, :, :3] = np.repeat(xyz, reps, axis=0)[:, None, :]      # constant over horizon
    if cfg.gripper == "both":
        # even rows = gripper closed (lo), odd rows = gripper open (hi)
        chunks[0::2, :, D - 1] = lo[D - 1]
        chunks[1::2, :, D - 1] = hi[D - 1]

    np.clip(chunks, lo, hi, out=chunks)
    return chunks.astype(np.float32)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
@torch.no_grad()
def score_chunks(verifier, obs_dict: Dict[str, Any], chunks: torch.Tensor,
                 batch: int = 64) -> torch.Tensor:
    """Score (B, M, H, D) raw chunks against the current frame -> (B, M).

    The image obs (B, To, ...) is repeat_interleaved M times so row ``b*M+j``
    sees env b's frame — the same layout the flattened chunks use.
    """
    img_key = getattr(verifier, "image_obs_key", None)
    if img_key is None or img_key not in obs_dict:
        raise KeyError(f"random_action_aug: obs_dict has no image key {img_key!r}")
    img = obs_dict[img_key]
    B, M, H, D = chunks.shape
    flat = chunks.reshape(B * M, H, D)
    big = img.repeat_interleave(M, dim=0)                            # (B*M, To, ...)
    vals = []
    step = max(1, int(batch))
    for s in range(0, B * M, step):
        vals.append(verifier.get_value({img_key: big[s:s + step]}, flat[s:s + step]))
    return torch.cat(vals, dim=0).reshape(B, M)


# --------------------------------------------------------------------------- #
# top-level
# --------------------------------------------------------------------------- #
@torch.no_grad()
def augment(policy, obs_dict: Dict[str, Any], actions: torch.Tensor,
            values: torch.Tensor, cfg: AugConfig,
            seeds: Optional[Any] = None,
            verifier=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Append stratified-random candidates to a policy's candidate pool.

    ``actions`` (B, N, H, D) raw, ``values`` (B, N). ``seeds`` may be a scalar or
    a per-env list (tiled path) — the random pool is derived from it so it is
    reproducible exactly like the diffusion init noise.

    Returns ``(actions', values', is_random)`` with is_random (B, N+M) bool.
    With ``cfg.only`` the policy candidates are dropped (is_random all True).
    """
    verifier = verifier if verifier is not None else policy.verifier
    B, N, H, D = actions.shape
    dev = actions.device

    if isinstance(seeds, (list, tuple)):
        per_env = [int(s) for s in seeds]
    elif seeds is None:
        per_env = [0] * B
    else:
        per_env = [int(seeds)] * B
    if len(per_env) != B:                      # tiled batch shrinks as envs finish
        per_env = (per_env * B)[:B] if per_env else [0] * B

    bounds = action_bounds(policy)
    best = values.argmax(dim=1)                                       # (B,)
    ref = actions[torch.arange(B, device=dev), best].detach().cpu().numpy()  # (B,H,D)

    rand = np.stack([
        sample_random_chunks(bounds, cfg, H, ref[b],
                             np.random.default_rng((per_env[b] * 1_000_003 + 7919)
                                                   % (2 ** 63 - 1)))
        for b in range(B)
    ])                                                                # (B,M,H,D)
    rand_t = torch.as_tensor(rand, dtype=actions.dtype, device=dev)
    rand_v = score_chunks(verifier, obs_dict, rand_t, cfg.batch).to(values.dtype).to(dev)

    if cfg.only:
        is_rand = torch.ones(B, rand_t.shape[1], dtype=torch.bool, device=dev)
        return rand_t, rand_v, is_rand

    out_a = torch.cat([actions, rand_t], dim=1)
    out_v = torch.cat([values, rand_v], dim=1)
    is_rand = torch.zeros(B, out_a.shape[1], dtype=torch.bool, device=dev)
    is_rand[:, N:] = True
    return out_a, out_v, is_rand
