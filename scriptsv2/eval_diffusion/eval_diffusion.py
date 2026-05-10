"""
Evaluate a `diffusion_policy` lowdim checkpoint (DiffusionUnetImagePolicy /
MLPImagePolicy trained with `bridge_v2_carrot_lowdim` or
`eggplant_in_basket_lowdim`) on the corresponding SimplerEnv task. Task is
chosen via the ``--task`` CLI flag (default ``widowx_carrot_on_plate``).

The checkpoints expect a 53-dim state vector built from privileged simulator
state and re-create the same per-step obs that `collect_trajectories.py`
recorded into the training Zarr.

Per-rollout flow:
  * Reset env at a deterministic seed (start_seed + i).
  * Each control step, build the 9 obs fields (end_effector_pose,
    end_effector_vel_lin_ang_b, arm_joint_pos, joint_vel, last_arm_action,
    last_gripper_action, insertive_asset_pose, receptive_asset_pose,
    insertive_asset_in_receptive_asset_frame). Maintain a sliding window of
    `n_obs_steps=4` past observations (left-padded by repeating the first
    obs).
  * Call `policy.predict_action(...)` to get a chunk of length
    `n_action_steps=8` raw 7D actions (xyz, rpy, gripper in [0,1]). Convert
    each via `convert_maniskill` (axis-angle rot, gripper -> {-1,+1}) and
    step the env. Re-plan after each chunk.
  * Episode is a success if the env returns done=True before max_steps.

Loads only the policy submodule from the workspace checkpoint, so neither
`hydra` `Accelerator` nor `wandb` is required at eval time.

Usage
-----
    python scriptsv2/eval_diffusion/eval_diffusion.py \
        --checkpoint /home/harine/diffusion_policy/data/outputs/2026.04.28/16.22.54_train_diffusion_unet_bridge_v2_carrot_lowdim_bridge_v2_carrot_lowdim/checkpoints/latest.ckpt \
        --num-episodes 100 \
        --output-dir data/eval/diffusion_unet_carrot
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import requests
import torch


# ---------------------------------------------------------------------------
#  Make `experiments.robot.token_action_converter` importable so we can use
#  the OpenVLA-compatible action -> token-id encoding when calling the
#  RoboMonkey verifier. Required because the verifier server's type check
#  mishandles numpy floats: it expects either Python floats (for which it
#  invokes its own ActionTokenizer) or integer token IDs. We send the latter.
# ---------------------------------------------------------------------------

_OPENVLA_MINI_DIR = Path(__file__).resolve().parents[2] / "openvla-mini"
if str(_OPENVLA_MINI_DIR) not in sys.path:
    sys.path.insert(0, str(_OPENVLA_MINI_DIR))


_TOKEN_ACTION_CONVERTER: Optional[Any] = None


def _get_token_action_converter():
    """Lazily build the bridge_orig TokenActionConverter (HF download/cached)."""
    global _TOKEN_ACTION_CONVERTER
    if _TOKEN_ACTION_CONVERTER is None:
        from experiments.robot.token_action_converter import TokenActionConverter
        _TOKEN_ACTION_CONVERTER = TokenActionConverter(
            n_action_bins=256, unnorm_key="bridge_orig",
        )
    return _TOKEN_ACTION_CONVERTER


# ---------------------------------------------------------------------------
#  Constants matching `bridge_v2_carrot_lowdim` task / collect_trajectories.py
# ---------------------------------------------------------------------------

TASK_NAME = "widowx_carrot_on_plate"
NUM_ARM_JOINTS = 6  # WidowX

OBS_KEYS_AND_DIMS: Dict[str, int] = {
    "end_effector_pose": 7,
    "end_effector_vel_lin_ang_b": 6,
    "arm_joint_pos": NUM_ARM_JOINTS,
    "joint_vel": NUM_ARM_JOINTS,
    "last_arm_action": 6,
    "last_gripper_action": 1,
    "insertive_asset_pose": 7,
    "receptive_asset_pose": 7,
    "insertive_asset_in_receptive_asset_frame": 7,
}


# ---------------------------------------------------------------------------
#  Action transform (mirror of `simpler_utils.convert_maniskill`)
# ---------------------------------------------------------------------------

def convert_maniskill(action: np.ndarray) -> np.ndarray:
    """Raw 7D VLA action -> ManiSkill `widowx_*` env action.

    * action[0:3]  : xyz delta (passed through)
    * action[3:6]  : rpy euler -> axis-angle (axis * angle)
    * action[6]    : gripper [0, 1] -> {-1, +1} via threshold at 0.5
    """
    from transforms3d.euler import euler2axangle

    assert action.shape[0] == 7, f"expected 7D action, got {action.shape}"
    a = action.astype(np.float32, copy=True)

    roll, pitch, yaw = float(a[3]), float(a[4]), float(a[5])
    axis, angle = euler2axangle(roll, pitch, yaw)
    a[3:6] = (np.asarray(axis, dtype=np.float32) * np.float32(angle))

    # gripper [0, 1] -> [-1, +1] then binarize via sign (with 0 -> +1)
    g = 2.0 * float(a[6]) - 1.0
    a[6] = 1.0 if g >= 0.0 else -1.0
    return a


def save_reward_image(image: np.ndarray, path: Path) -> None:
    """Write the current SimplerEnv RGB frame in the verifier's expected format."""
    import tensorflow as tf
    from PIL import Image

    image = tf.convert_to_tensor(image, dtype=tf.uint8)
    image = tf.image.encode_jpeg(image)
    image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8)
    image = tf.image.resize(image, (256, 256), method="lanczos3", antialias=True)
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    image = tf.io.encode_jpeg(image, quality=95)

    image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8)
    image = tf.image.resize(image, (256, 256), method="lanczos3", antialias=True)
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def _actions_to_openvla_token_ids(actions: np.ndarray) -> np.ndarray:
    """Encode 7D continuous actions as OpenVLA token IDs in [31744, 32000].

    The verifier server's `/process` handler treats integer rows as token IDs
    directly and only tokenizes rows that have Python `float` elements. Sending
    pre-tokenized integers sidesteps a server-side type check bug that
    misroutes numpy-float arrays through the integer path with garbage IDs.
    """
    converter = _get_token_action_converter()
    return np.stack(
        [np.asarray(converter.action_to_token(row), dtype=np.int64) for row in actions],
        axis=0,
    )


def get_verifier_rewards(
    instruction: str,
    image_path: Path,
    actions: np.ndarray,
    reward_server_port: int,
    reward_batch_size: int,
) -> List[float]:
    """Score candidate 7D actions with the RoboMonkey verifier server.

    Sends ceil(N / reward_batch_size) HTTP requests. Each request is
    dispatched in a thread so that TCP + server-side CPU work (image load,
    tokenisation) overlaps across batches. GPU forward passes still serialize
    inside the server, but the I/O overhead is amortised.

    Set reward_batch_size >= total number of actions to send everything in a
    single round-trip (fastest when GPU memory allows).
    """
    import concurrent.futures

    actions = np.asarray(actions, dtype=np.float32)
    token_ids = _actions_to_openvla_token_ids(actions)

    num_batches = math.ceil(len(token_ids) / reward_batch_size)
    batches = [
        token_ids[i * reward_batch_size: min((i + 1) * reward_batch_size, len(token_ids))]
        for i in range(num_batches)
    ]

    url = f"http://127.0.0.1:{reward_server_port}/process"

    def _post_batch(batch: np.ndarray) -> List[float]:
        payload = {
            "instruction": instruction,
            "image_path": str(image_path),
            "action": batch.tolist(),
        }
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        return [float(r) for r in response.json()["rewards"]]

    if num_batches == 1:
        return _post_batch(batches[0])

    all_rewards: List[float] = [0.0] * len(token_ids)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_batches) as pool:
        futures = {
            pool.submit(_post_batch, batch): i * reward_batch_size
            for i, batch in enumerate(batches)
        }
        for future in concurrent.futures.as_completed(futures):
            start_idx = futures[future]
            rewards = future.result()
            all_rewards[start_idx: start_idx + len(rewards)] = rewards

    return all_rewards


def check_verifier_health(reward_server_port: int) -> None:
    """Ping the verifier server to fail-fast if it isn't running."""
    url = f"http://127.0.0.1:{reward_server_port}/"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"Verifier server health check failed at {url}: {e!r}. "
            "Start the RoboMonkey verifier server before running with bon_k > 1."
        ) from e
    print(f"[eval] Verifier server reachable at {url}")


# ---------------------------------------------------------------------------
#  Per-step state extractors (match collect_trajectories.py)
# ---------------------------------------------------------------------------

def _pose_to_vec(pose) -> np.ndarray:
    return np.concatenate(
        [np.asarray(pose.p, dtype=np.float32),
         np.asarray(pose.q, dtype=np.float32)]
    ).astype(np.float32)


def _tcp_pose_vec(obs_or_pose) -> np.ndarray:
    if hasattr(obs_or_pose, "p") and hasattr(obs_or_pose, "q"):
        return _pose_to_vec(obs_or_pose)
    return np.asarray(obs_or_pose, dtype=np.float32).reshape(-1)[:7]


def _ee_body_velocity(tcp_link) -> np.ndarray:
    R_bw = tcp_link.pose.to_transformation_matrix()[:3, :3]
    v_w = np.asarray(tcp_link.get_velocity(), dtype=np.float64)
    w_w = np.asarray(tcp_link.get_angular_velocity(), dtype=np.float64)
    v_b = R_bw.T @ v_w
    w_b = R_bw.T @ w_w
    return np.concatenate([v_b, w_b]).astype(np.float32)


def _rel_pose_vec(source_pose, target_pose) -> np.ndarray:
    rel = target_pose.inv() * source_pose
    return _pose_to_vec(rel)


def build_obs_step(
    env: Any,
    obs: Dict[str, Any],
    prev_arm_action: np.ndarray,
    prev_gripper_action: np.ndarray,
    source_obj: Any,
    target_obj: Any,
    tcp_link: Any,
) -> Dict[str, np.ndarray]:
    """Build one (state-only) observation matching the training shape_meta."""
    qpos = np.asarray(env.agent.robot.get_qpos(), dtype=np.float32)
    qvel = np.asarray(env.agent.robot.get_qvel(), dtype=np.float32)
    arm_q = qpos[:NUM_ARM_JOINTS].copy()
    arm_qv = qvel[:NUM_ARM_JOINTS].copy()

    return {
        "end_effector_pose":
            _tcp_pose_vec(obs["extra"]["tcp_pose"]),
        "end_effector_vel_lin_ang_b":
            _ee_body_velocity(tcp_link),
        "arm_joint_pos": arm_q,
        "joint_vel": arm_qv,
        "last_arm_action": prev_arm_action.astype(np.float32, copy=True),
        "last_gripper_action": prev_gripper_action.astype(np.float32, copy=True),
        "insertive_asset_pose": _pose_to_vec(source_obj.pose),
        "receptive_asset_pose": _pose_to_vec(target_obj.pose),
        "insertive_asset_in_receptive_asset_frame":
            _rel_pose_vec(source_obj.pose, target_obj.pose),
    }


# ---------------------------------------------------------------------------
#  Sliding window of past observations (length n_obs_steps, padded by repeat)
# ---------------------------------------------------------------------------

class ObsWindow:
    def __init__(self, n_obs_steps: int):
        self.n_obs_steps = n_obs_steps
        self.buffer: Deque[Dict[str, np.ndarray]] = deque(maxlen=n_obs_steps)

    def push(self, obs_step: Dict[str, np.ndarray]) -> None:
        self.buffer.append(obs_step)

    def is_empty(self) -> bool:
        return len(self.buffer) == 0

    def to_tensor_dict(
        self, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> Dict[str, torch.Tensor]:
        """Return dict of (1, n_obs_steps, dim) tensors. If <n_obs_steps real
        obs are buffered, pad on the LEFT by repeating the earliest obs (this
        matches diffusion_policy's `pad_before` behaviour for replay buffers).
        """
        assert not self.is_empty()
        steps = list(self.buffer)
        pad = self.n_obs_steps - len(steps)
        if pad > 0:
            steps = [steps[0]] * pad + steps

        out: Dict[str, torch.Tensor] = {}
        for k in OBS_KEYS_AND_DIMS:
            arr = np.stack([s[k] for s in steps], axis=0).astype(np.float32)
            out[k] = torch.from_numpy(arr).to(device=device, dtype=dtype).unsqueeze(0)
        return out


# ---------------------------------------------------------------------------
#  Checkpoint loading: instantiate the policy module *only*
# ---------------------------------------------------------------------------

def load_policy(checkpoint: str, device: torch.device, use_ema: bool):
    """Load a diffusion_policy checkpoint and return a ready-to-run policy.

    Avoids constructing the full training workspace (no Accelerator / wandb /
    optimizer). We instantiate just `cfg.policy` via Hydra and load weights
    from `state_dicts['ema_model' | 'model']`.
    """
    import dill
    import hydra
    from omegaconf import OmegaConf

    # diffusion_policy configs use `${eval:...}` interpolations (registered by
    # its train.py entry point). Re-register here so we can instantiate
    # `cfg.policy` standalone without importing the training workspace.
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval, replace=True)

    print(f"[eval] Loading checkpoint: {checkpoint}")
    payload = torch.load(
        open(checkpoint, "rb"),
        pickle_module=dill,
        map_location="cpu",
        weights_only=False,
    )
    cfg = payload["cfg"]
    state_dicts = payload["state_dicts"]

    print(f"[eval] policy._target_ = {cfg.policy._target_}")
    print(
        f"[eval] horizon={cfg.horizon}  "
        f"n_obs_steps={cfg.n_obs_steps}  "
        f"n_action_steps={cfg.n_action_steps}  "
        f"action_norm_mode={cfg.get('action_norm_mode', 'gaussian')}"
    )

    policy = hydra.utils.instantiate(cfg.policy)

    src_key = "ema_model" if (use_ema and "ema_model" in state_dicts) else "model"
    print(f"[eval] Loading weights from state_dicts['{src_key}']")
    state = state_dicts[src_key]

    missing, unexpected = policy.load_state_dict(state, strict=False)
    if missing:
        print(f"[eval]   missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        print(f"[eval]   unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")

    policy = policy.to(device)
    policy.eval()
    return policy, cfg


# ---------------------------------------------------------------------------
#  Single rollout
# ---------------------------------------------------------------------------

def rollout_episode(
    env: Any,
    policy: Any,
    cfg: Any,
    device: torch.device,
    seed: int,
    max_steps: int,
    instruction: str,
    deterministic: bool = True,
    capture_frames: bool = False,
    bon_k: int = 1,
    reward_server_port: int = 3100,
    reward_batch_size: int = 2,
    reward_image_path: Path | None = None,
    bon_replan_every_n_steps: int = 0,
    bon_score_num_actions: int = 1,
    viz_q: bool = False,
) -> Dict[str, Any]:
    """Run one episode and return per-step diagnostics.

    If `capture_frames=True`, also collects per-step camera RGB frames into
    the returned dict under key `"frames"` (list of HxWx3 uint8 numpy arrays).

    If `viz_q=True` *and* BoN is enabled, also captures per-branch data under
    `"bon_branches"`: at each replan we record all K candidate action chunks,
    the verifier rewards for each, the chosen index, and the camera frame
    that the verifier scored. Caller can `np.savez` this for later
    visualization of branching / action selection.
    """
    n_obs_steps = int(cfg.n_obs_steps)
    n_action_steps = int(cfg.n_action_steps)

    obs, _reset_info = env.reset(seed=int(seed))

    source_obj = getattr(env, "episode_source_obj", None)
    target_obj = getattr(env, "episode_target_obj", None)
    tcp_link = getattr(env, "tcp", None)
    if source_obj is None or target_obj is None or tcp_link is None:
        raise RuntimeError(
            "env is missing episode_source_obj / episode_target_obj / tcp; "
            f"got {source_obj=}, {target_obj=}, {tcp_link=}"
        )

    prev_arm_action = np.zeros(6, dtype=np.float32)
    prev_gripper_action = np.zeros(1, dtype=np.float32)

    window = ObsWindow(n_obs_steps=n_obs_steps)
    success = False
    truncated = False
    t = 0
    step_log: List[Dict[str, Any]] = []

    pending_actions: List[np.ndarray] = []  # raw 7D actions queued from last predict
    chunk_actions_executed = 0  # how many actions of the current chunk we've executed
    frames: List[np.ndarray] = []

    replan_every_n = int(bon_replan_every_n_steps)
    if replan_every_n < 0:
        replan_every_n = 0

    use_bon = int(bon_k) > 1
    if use_bon and reward_image_path is None:
        reward_image_path = Path("./transfer_images/reward_img.jpg").absolute()

    bon_branches: Optional[List[Dict[str, Any]]] = (
        [] if (viz_q and use_bon) else None
    )

    # Lazy import (only when we actually need camera frames)
    if capture_frames or use_bon:
        from simpler_env.utils.env.observation_utils import (
            get_image_from_maniskill2_obs_dict,
        )

    while t < max_steps:
        step_obs = build_obs_step(
            env=env,
            obs=obs,
            prev_arm_action=prev_arm_action,
            prev_gripper_action=prev_gripper_action,
            source_obj=source_obj,
            target_obj=target_obj,
            tcp_link=tcp_link,
        )
        window.push(step_obs)

        current_frame: np.ndarray | None = None
        if capture_frames or use_bon:
            try:
                img = get_image_from_maniskill2_obs_dict(env, obs)
                arr = np.asarray(img, dtype=np.uint8)
                if arr.ndim == 3 and arr.shape[-1] == 3:
                    current_frame = arr
                    if capture_frames:
                        frames.append(arr)
            except Exception as e:
                if use_bon:
                    raise RuntimeError(f"failed to capture verifier image: {e!r}") from e
                if capture_frames:
                    if t == 0:
                        print(f"[eval] frame capture disabled (error: {e!r})")
                    capture_frames = False

        selected_index = 0
        selected_reward = None
        bon_rewards: List[float] | None = None
        score_actions_per_candidate = max(
            1, min(int(bon_score_num_actions), n_action_steps)
        )

        if not pending_actions:
            obs_dict = window.to_tensor_dict(device=device)

            def _sample_chunk(candidate_idx: int = 0) -> np.ndarray:
                if use_bon:
                    sample_seed = int(seed) * 100000 + int(t) * 1000 + candidate_idx
                else:
                    sample_seed = int(seed) + int(t)
                with torch.no_grad():
                    result = _sample_action_chunk(
                        policy=policy,
                        obs_dict=obs_dict,
                        use_bon=use_bon,
                        deterministic=deterministic,
                        device=device,
                        sample_seed=sample_seed,
                    )
                chunk = result["action"][0].detach().cpu().numpy()
                assert chunk.shape == (n_action_steps, 7), \
                    f"expected ({n_action_steps}, 7), got {chunk.shape}"
                return chunk.astype(np.float32, copy=False)

            if use_bon:
                assert current_frame is not None
                assert reward_image_path is not None
                save_reward_image(current_frame, reward_image_path)

                candidate_chunks = np.stack(
                    [_sample_chunk(j) for j in range(int(bon_k))],
                    axis=0,
                )
                # Score the first `score_actions_per_candidate` actions of each
                # candidate and average. Shape: (K * N, 7) flattened request
                # to the verifier, then reshape (K, N) -> mean over N.
                actions_to_score = candidate_chunks[
                    :, :score_actions_per_candidate, :
                ].reshape(-1, 7)
                flat_rewards = get_verifier_rewards(
                    instruction=instruction,
                    image_path=reward_image_path,
                    actions=actions_to_score,
                    reward_server_port=int(reward_server_port),
                    reward_batch_size=int(reward_batch_size),
                )
                rewards_matrix = np.asarray(flat_rewards, dtype=np.float32).reshape(
                    int(bon_k), score_actions_per_candidate
                )
                per_candidate_reward = rewards_matrix.mean(axis=1)
                selected_index = int(np.argmax(per_candidate_reward))
                selected_reward = float(per_candidate_reward[selected_index])
                bon_rewards = per_candidate_reward.tolist()
                actions_chunk = candidate_chunks[selected_index]

                if bon_branches is not None and current_frame is not None:
                    bon_branches.append({
                        "t": int(t),
                        "candidate_actions": candidate_chunks.astype(np.float32, copy=True),
                        "per_candidate_rewards": rewards_matrix.astype(np.float32, copy=True),
                        "per_candidate_mean_reward": per_candidate_reward.astype(np.float32, copy=True),
                        "selected_index": int(selected_index),
                        "selected_reward": float(selected_reward),
                        "frame": current_frame.astype(np.uint8, copy=True),
                    })
            else:
                actions_chunk = _sample_chunk(0)
            pending_actions = list(actions_chunk)
            chunk_actions_executed = 0

        action_vla = pending_actions.pop(0).astype(np.float32)
        chunk_actions_executed += 1
        env_action = convert_maniskill(action_vla.copy())

        obs, reward, done, trunc, info = env.step(env_action)

        prev_arm_action = action_vla[:6].copy()
        prev_gripper_action = action_vla[6:7].copy()

        step_log.append({
            "t": t,
            "reward": float(reward),
            "done": bool(done),
            "trunc": bool(trunc),
            "bon_k": int(bon_k),
            "bon_score_num_actions": int(score_actions_per_candidate),
            "bon_replan_every_n_steps": int(replan_every_n),
        })
        if use_bon and selected_reward is not None and bon_rewards is not None:
            step_log[-1].update({
                "bon_selected_index": int(selected_index),
                "bon_selected_reward": float(selected_reward),
                "bon_reward_min": float(np.min(bon_rewards)),
                "bon_reward_max": float(np.max(bon_rewards)),
            })

        if bool(done):
            success = True
            break
        if bool(trunc):
            truncated = True
            break

        if replan_every_n > 0 and chunk_actions_executed >= replan_every_n:
            pending_actions = []

        t += 1

    return {
        "success": bool(success),
        "truncated": bool(truncated),
        "num_steps": int(t + 1 if (success or truncated) else t),
        "steps": step_log,
        "frames": frames,
        "bon_branches": bon_branches,
    }


def save_video(frames: List[np.ndarray], path: Path, fps: int = 10) -> bool:
    """Write a list of HxWx3 uint8 frames to `path` as an mp4.

    Tries `imageio` (with imageio-ffmpeg) first; falls back to `mediapy`.
    Returns True on success.
    """
    if not frames:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.stack(frames, axis=0)  # (T, H, W, 3) uint8

    try:
        import imageio.v2 as imageio
        with imageio.get_writer(
            str(path),
            fps=int(fps),
            codec="libx264",
            quality=8,
            macro_block_size=1,
        ) as w:
            for f in arr:
                w.append_data(f)
        return True
    except Exception as e_imageio:
        try:
            import mediapy
            mediapy.write_video(str(path), arr, fps=int(fps))
            return True
        except Exception as e_media:
            print(
                f"[eval] failed to save video {path.name}: "
                f"imageio={e_imageio!r}; mediapy={e_media!r}"
            )
            return False


# ---------------------------------------------------------------------------
#  Best-of-N sampler dispatch.
#
#  Default behaviour: reseed torch RNG and call `policy.predict_action`. This
#  works out-of-the-box for any policy whose `predict_action` is internally
#  stochastic (e.g., diffusion U-Net and transformer policies that call
#  `torch.randn`).
#
#  For policies whose `predict_action` is deterministic but that expose a
#  learned distribution (e.g., a Gaussian MLP that returns `dist.mean`),
#  register a custom sampler in `_BON_SAMPLER_REGISTRY` keyed by class name.
#  The sampler must take `(policy, obs_dict)` and return a dict with the same
#  shape as `policy.predict_action`, i.e. `{"action": Tensor[B, T, action_dim]}`.
# ---------------------------------------------------------------------------

_PRINTED_MLP_DIST_SCALE = False


def _maybe_log_mlp_scale(dist) -> None:
    """One-shot diagnostic so the user can see whether MLP candidates differ."""
    global _PRINTED_MLP_DIST_SCALE
    if _PRINTED_MLP_DIST_SCALE:
        return
    scale_mean = float(dist.scale.mean().item())
    print(f"[eval] MLP candidate diversity check: dist.scale.mean()={scale_mean:.4f}")
    if scale_mean < 1e-3:
        print(
            "[eval] WARNING: very small std; MLP candidates may be near-identical. "
            "If you want true diversity, retrain with NLL loss or unfreeze log_std."
        )
    _PRINTED_MLP_DIST_SCALE = True


def _sample_mlp_action(policy, obs_dict) -> Dict[str, torch.Tensor]:
    """Sample from MLPImagePolicy's learned Normal action distribution.

    Note: depends on private internals of `diffusion_policy.MLPImagePolicy`
    (`_encode_obs`, `forward`, `autoregressive`, `normalizer["action"]`).
    """
    obs_input = policy._encode_obs(obs_dict)
    batch_size = obs_input.shape[0]

    if policy.autoregressive:
        action_steps = policy.n_action_steps
        action_dim = policy.action_dim
        action_slots = torch.zeros(
            batch_size,
            action_steps * action_dim,
            device=obs_input.device,
            dtype=obs_input.dtype,
        )
        predicted_actions = []
        for step_idx in range(action_steps):
            trunk_input = torch.cat([obs_input, action_slots], dim=-1)
            dist = policy.forward(trunk_input)
            _maybe_log_mlp_scale(dist)
            next_action = dist.sample()
            predicted_actions.append(next_action)
            slot_idx = step_idx * action_dim
            action_slots = action_slots.clone()
            action_slots[:, slot_idx:slot_idx + action_dim] = next_action
        action_pred = torch.stack(predicted_actions, dim=1)
    else:
        dist = policy.forward(obs_input)
        _maybe_log_mlp_scale(dist)
        action_pred = dist.sample()

    action = policy.normalizer["action"].unnormalize(action_pred)
    return {
        "action": action,
        "action_pred": action_pred,
    }


_BON_SAMPLER_REGISTRY: Dict[str, Any] = {
    "MLPImagePolicy": _sample_mlp_action,
}


def _sample_action_chunk(
    policy: Any,
    obs_dict: Dict[str, torch.Tensor],
    use_bon: bool,
    deterministic: bool,
    device: torch.device,
    sample_seed: int,
) -> Dict[str, torch.Tensor]:
    """Draw one action chunk for either baseline rollout or BON ranking.

    - If `deterministic`, reseed the torch RNG (CUDA + CPU) with `sample_seed`.
    - If `use_bon` and the policy class is registered, use the registered
      sampler (e.g., MLP samples from its learned Normal distribution).
    - Otherwise call `policy.predict_action` directly.
    """
    if deterministic:
        torch.manual_seed(int(sample_seed))
        if device.type == "cuda":
            torch.cuda.manual_seed(int(sample_seed))

    if use_bon:
        sampler = _BON_SAMPLER_REGISTRY.get(policy.__class__.__name__)
        if sampler is not None:
            return sampler(policy, obs_dict)
    return policy.predict_action(obs_dict)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to diffusion_policy .ckpt")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=1000,
                        help="Episode i uses seed = start_seed + i. "
                             "Default 1000 matches openvla-mini eval.")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated list of explicit episode seeds "
                             "(e.g. '17,50,3,9'). When set, --num-episodes "
                             "and --start-seed are ignored and one episode is "
                             "run per listed seed in order.")
    parser.add_argument("--max-steps", type=int, default=120,
                        help="Per-episode env step cap (env may also truncate).")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=None,
                        help="If set, write eval_log.json + per-episode jsonl.")
    parser.add_argument("--no-ema", action="store_true",
                        help="Use raw model weights instead of EMA weights.")
    parser.add_argument("--non-deterministic", action="store_true",
                        help="Do not reseed RNG before each predict_action.")
    parser.add_argument("--num-inference-steps", type=int, default=None,
                        help="Override DDPM inference steps (default: cfg).")
    parser.add_argument("--task", default=TASK_NAME,
                        help=f"SimplerEnv task name (default: {TASK_NAME})")
    parser.add_argument("--save-videos", type=int, default=0,
                        help="Save MP4 videos for the first N episodes "
                             "(0 = no videos; default 0). Videos go to "
                             "<output_dir>/videos/ep{idx}_seed{seed}_<status>.mp4")
    parser.add_argument("--video-fps", type=int, default=10,
                        help="Frame rate for saved videos (default 10).")
    parser.add_argument("--bon-k", type=int, default=1,
                        help="Best-of-N candidate chunks per replan. "
                             "1 disables verifier ranking.")
    parser.add_argument("--bon-replan-every-n-steps", type=int, default=0,
                        help="If >0, replan after this many actions of the "
                             "current chunk are executed. 0 (default) means "
                             "replan only when the chunk is fully drained "
                             "(i.e., every n_action_steps env steps).")
    parser.add_argument("--bon-score-num-actions", type=int, default=1,
                        help="Number of leading actions to score per candidate "
                             "(rewards are averaged). Default 1.")
    parser.add_argument("--reward-server-port", type=int, default=3100,
                        help="RoboMonkey verifier server port (default 3100).")
    parser.add_argument("--reward-batch-size", type=int, default=2,
                        help="Verifier scoring batch size (default 2).")
    parser.add_argument("--reward-image-path",
                        default="./transfer_images/reward_img.jpg",
                        help="Temporary JPEG path sent to the verifier.")
    parser.add_argument("--viz-q", action="store_true",
                        help="When BoN is enabled, save per-replan candidate "
                             "actions, verifier rewards (Q-values), the "
                             "selected index, and the verifier-scored frame "
                             "to <output_dir>/bon_q/ep<idx>_seed<seed>.npz "
                             "for later branching/action-selection viz.")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[eval] CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")

    policy, cfg = load_policy(
        checkpoint=args.checkpoint, device=device, use_ema=not args.no_ema,
    )
    if args.num_inference_steps is not None and hasattr(policy, "num_inference_steps"):
        print(f"[eval] Overriding num_inference_steps -> {args.num_inference_steps}")
        policy.num_inference_steps = int(args.num_inference_steps)

    # Lazy import (TF + sapien only loaded after policy is on GPU).
    import simpler_env  # noqa: F401  (registers envs)
    from simpler_env import make as make_env

    print(f"[eval] Creating SimplerEnv task: {args.task}")
    env = make_env(args.task)
    instr = env.get_language_instruction() or args.task
    print(f"[eval] instruction: {instr!r}")
    bon_k = max(1, int(args.bon_k))
    bon_score_num_actions = max(1, int(args.bon_score_num_actions))
    bon_replan_every_n_steps = max(0, int(args.bon_replan_every_n_steps))
    if bon_k > 1:
        print(
            f"[eval] Best-of-N enabled: k={bon_k}, "
            f"replan_every_n_steps={bon_replan_every_n_steps} "
            f"(0 = chunk replan), "
            f"score_num_actions={bon_score_num_actions}, "
            f"reward_server_port={args.reward_server_port}"
        )
        check_verifier_health(int(args.reward_server_port))
        # Warm up TokenActionConverter so the HF download/cache happens before
        # any rollout, not mid-episode.
        try:
            _get_token_action_converter()
            print("[eval] TokenActionConverter (bridge_orig, 256 bins) ready.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load OpenVLA TokenActionConverter: {e!r}. "
                "BON requires the openvla/openvla-7b config (HF cache or download)."
            ) from e

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        ep_log_path = output_dir / "episodes.jsonl"
        ep_log_file = open(ep_log_path, "w")
    else:
        ep_log_file = None

    save_videos_n = max(0, int(args.save_videos))
    if save_videos_n > 0 and output_dir is None:
        print("[eval] --save-videos>0 requires --output-dir; disabling.")
        save_videos_n = 0
    video_dir = (output_dir / "videos") if (output_dir is not None and save_videos_n > 0) else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)
        print(f"[eval] saving videos for first {save_videos_n} episode(s) -> {video_dir}")

    if args.seeds:
        explicit_seeds: Optional[List[int]] = [
            int(s) for s in args.seeds.split(",") if s.strip()
        ]
        num_episodes = len(explicit_seeds)
        if num_episodes == 0:
            raise SystemExit("--seeds was empty after parsing")
        print(
            f"[eval] using --seeds={explicit_seeds} "
            f"(overrides --num-episodes={args.num_episodes} and "
            f"--start-seed={args.start_seed}); running {num_episodes} episode(s)."
        )
    else:
        explicit_seeds = None
        num_episodes = int(args.num_episodes)

    viz_q = bool(args.viz_q)
    bon_q_dir: Path | None = None
    if viz_q and bon_k > 1:
        if output_dir is None:
            print("[eval] --viz-q requires --output-dir; disabling.")
            viz_q = False
        else:
            bon_q_dir = output_dir / "bon_q"
            bon_q_dir.mkdir(parents=True, exist_ok=True)
            print(f"[eval] viz_q enabled -> saving per-branch Q-values to {bon_q_dir}")
    elif viz_q:
        print("[eval] --viz-q has no effect when bon_k <= 1; disabling.")
        viz_q = False

    successes = 0
    truncations = 0
    durations: List[float] = []
    t0 = time.time()

    for i in range(num_episodes):
        ep_seed = explicit_seeds[i] if explicit_seeds is not None else int(args.start_seed) + i
        capture = (i < save_videos_n)
        ep_t0 = time.time()
        try:
            ep = rollout_episode(
                env=env,
                policy=policy,
                cfg=cfg,
                device=device,
                seed=ep_seed,
                max_steps=int(args.max_steps),
                instruction=str(instr),
                deterministic=not args.non_deterministic,
                capture_frames=capture,
                bon_k=bon_k,
                reward_server_port=int(args.reward_server_port),
                reward_batch_size=max(1, int(args.reward_batch_size)),
                reward_image_path=Path(args.reward_image_path).absolute(),
                bon_replan_every_n_steps=bon_replan_every_n_steps,
                bon_score_num_actions=bon_score_num_actions,
                viz_q=viz_q,
            )
        except Exception as e:
            print(f"[eval] episode {i} (seed={ep_seed}) raised: {e!r}")
            ep = {"success": False, "truncated": False, "num_steps": 0,
                  "error": repr(e), "steps": [], "frames": [],
                  "bon_branches": None}

        durations.append(time.time() - ep_t0)
        successes += int(ep["success"])
        truncations += int(ep["truncated"])

        sr_so_far = successes / float(i + 1)
        print(
            f"[eval] ep={i+1:3d}/{num_episodes}  seed={ep_seed}  "
            f"success={int(ep['success'])}  trunc={int(ep['truncated'])}  "
            f"steps={ep['num_steps']:3d}  elapsed={durations[-1]:5.1f}s  "
            f"running_sr={sr_so_far:.3f}",
            flush=True,
        )

        if bon_q_dir is not None and ep.get("bon_branches"):
            branches = ep["bon_branches"]
            npz_path = bon_q_dir / f"ep{i:03d}_seed{ep_seed}.npz"
            np.savez_compressed(
                npz_path,
                seed=np.int32(ep_seed),
                ep_idx=np.int32(i),
                success=np.int32(int(ep["success"])),
                truncated=np.int32(int(ep["truncated"])),
                num_steps=np.int32(int(ep["num_steps"])),
                bon_k=np.int32(int(bon_k)),
                bon_replan_every_n_steps=np.int32(int(bon_replan_every_n_steps)),
                bon_score_num_actions=np.int32(int(bon_score_num_actions)),
                branch_t=np.asarray([b["t"] for b in branches], dtype=np.int32),
                candidate_actions=np.stack([b["candidate_actions"] for b in branches], axis=0),
                per_candidate_rewards=np.stack([b["per_candidate_rewards"] for b in branches], axis=0),
                per_candidate_mean_reward=np.stack([b["per_candidate_mean_reward"] for b in branches], axis=0),
                selected_index=np.asarray([b["selected_index"] for b in branches], dtype=np.int32),
                selected_reward=np.asarray([b["selected_reward"] for b in branches], dtype=np.float32),
                frames=np.stack([b["frame"] for b in branches], axis=0),
            )
            print(f"[eval]   saved Q-values -> {npz_path} ({len(branches)} branches)")

        video_path: Path | None = None
        if video_dir is not None and capture and ep.get("frames"):
            status = "success" if ep["success"] else (
                "truncated" if ep["truncated"] else "fail"
            )
            video_path = video_dir / f"ep{i:03d}_seed{ep_seed}_{status}.mp4"
            ok = save_video(ep["frames"], video_path, fps=int(args.video_fps))
            if ok:
                print(f"[eval]   saved video -> {video_path}")
            else:
                video_path = None

        if ep_log_file is not None:
            ep_compact = {
                "ep_idx": i,
                "seed": ep_seed,
                "success": ep["success"],
                "truncated": ep["truncated"],
                "num_steps": ep["num_steps"],
                "duration_s": durations[-1],
            }
            if video_path is not None:
                ep_compact["video"] = str(video_path)
            if "error" in ep:
                ep_compact["error"] = ep["error"]
            ep_log_file.write(json.dumps(ep_compact) + "\n")
            ep_log_file.flush()

    total_t = time.time() - t0
    success_rate = successes / float(num_episodes)
    summary = {
        "checkpoint": str(args.checkpoint),
        "task": args.task,
        "instruction": str(instr),
        "num_episodes": int(num_episodes),
        "start_seed": int(args.start_seed),
        "seeds": (
            [int(s) for s in explicit_seeds] if explicit_seeds is not None else None
        ),
        "max_steps": int(args.max_steps),
        "use_ema": (not args.no_ema),
        "bon_k": int(bon_k),
        "bon_replan_every_n_steps": int(bon_replan_every_n_steps),
        "bon_score_num_actions": int(bon_score_num_actions),
        "reward_server_port": int(args.reward_server_port),
        "num_successes": int(successes),
        "num_truncated": int(truncations),
        "success_rate": float(success_rate),
        "mean_episode_time_s": float(np.mean(durations)) if durations else 0.0,
        "total_time_s": float(total_t),
    }
    print("\n[eval] ===== summary =====")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if output_dir is not None:
        with open(output_dir / "eval_log.json", "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"\n[eval] Wrote {output_dir / 'eval_log.json'}")
        if ep_log_file is not None:
            ep_log_file.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    main()
