"""Run policy + verifier on training-data observations to produce the same
candidate-action / Q-value / chosen-in-red overlays as for eval rollouts.

For each of N selected training episodes from the eggplant_in_basket zarr,
walk through the steps with a sliding window of `n_obs_steps` past obs (as
during training), and at every `branch_every` steps:
  1. Build the policy obs dict from the zarr (the 53-dim low-dim state).
  2. Sample K candidate action chunks in a single batched predict_action.
  3. Score the first `score_num_actions` of each candidate with the verifier
     using the agentview_image at this step.
  4. Pick argmax(Q) as the "chosen" candidate.

The output is an NPZ per episode in the same schema as eval_diffusion.py's
bon_q files, with the extra fields (tcp_world_p, cam_intrinsic, cam_extrinsic)
already populated. After running this, point `viz_action_branches.py` at the
output directory to render the overlay images.

Camera intrinsics are read live from the env once at startup (3rd_view_camera
for widowx_sink_camera_setup, which is fixed because the camera is attached to
base_link). The agentview_image in the zarr is resized to 224x224, so the
intrinsic is scaled accordingly. EE pose comes directly from the zarr
(end_effector_pose), which is recorded in the world frame.

Usage
-----
    bash scriptsv2/viz_bon/run_viz_training.sh <zarr_path> <ckpt> <out_dir>

or manually after activating monkey-verifier:
    python scriptsv2/viz_bon/viz_training_data.py \\
        --zarr /gscratch/robotics/harine/data/eggplant_in_basket/state0.zarr \\
        --checkpoint <path/to/diffusion_policy_latest.ckpt> \\
        --out-dir data/eval/training_viz \\
        --num-episodes 5 --bon-k 8 --score-num-actions 4 --branch-every 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

# Reuse eval_diffusion.py helpers (policy loading, verifier scoring, action
# conversion to OpenVLA token ids). The path setup at the top of that module
# already adds openvla-mini and monkey-verifier to sys.path.
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scriptsv2" / "eval_diffusion"))
import eval_diffusion as _ed  # noqa: E402


# --------------------------------------------------------------------------- #
# Observation building straight from the zarr (no env required).
# --------------------------------------------------------------------------- #

_OBS_KEYS = list(_ed.OBS_KEYS_AND_DIMS.keys())


def build_obs_window_from_zarr(z, idx: int, n_obs_steps: int) -> Dict[str, torch.Tensor]:
    """Sliding window of past observations from the zarr, pad-left by repeat.

    Returns a dict matching the policy's expected obs_dict shape: each value
    is (1, n_obs_steps, dim).
    """
    out: Dict[str, torch.Tensor] = {}
    start = max(0, idx - n_obs_steps + 1)
    indices = list(range(start, idx + 1))
    pad = n_obs_steps - len(indices)
    if pad > 0:
        indices = [indices[0]] * pad + indices
    for key in _OBS_KEYS:
        arr_2d = np.asarray(z[f"data/obs/{key}"][indices], dtype=np.float32)
        if arr_2d.ndim == 1:
            arr_2d = arr_2d[:, None]
        out[key] = torch.from_numpy(arr_2d).unsqueeze(0)
    return out


# --------------------------------------------------------------------------- #
# Camera params: query the env once at startup.
# --------------------------------------------------------------------------- #

def get_widowx_camera_params(task: str, target_img_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (intrinsic 3x3 scaled to target_img_size, extrinsic 3x4)."""
    import simpler_env  # noqa: F401 registers envs
    env = simpler_env.make(task)
    env.reset(seed=0)
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    cp = base.get_camera_params()
    cam_name = "3rd_view_camera" if "3rd_view_camera" in cp else next(iter(cp.keys()))
    K_orig = np.asarray(cp[cam_name]["intrinsic_cv"], dtype=np.float32).copy()
    ext = np.asarray(cp[cam_name]["extrinsic_cv"], dtype=np.float32)
    if ext.shape == (4, 4):
        ext = ext[:3, :].copy()

    # The zarr stores agentview_image at target_img_size x target_img_size,
    # downsampled from the original env render (640x480). Scale intrinsics.
    # Read raw render size from the env's camera config.
    try:
        cam_obj = base._cameras[cam_name]
        H_orig = int(cam_obj.camera.height)
        W_orig = int(cam_obj.camera.width)
    except Exception:
        H_orig, W_orig = 480, 640  # widowx default
    sx = float(target_img_size) / float(W_orig)
    sy = float(target_img_size) / float(H_orig)
    K_scaled = K_orig.copy()
    K_scaled[0, :] *= sx
    K_scaled[1, :] *= sy
    print(f"[viz_train] camera={cam_name}  raw=({W_orig}x{H_orig}) -> "
          f"target=({target_img_size}x{target_img_size})  sx={sx:.4f} sy={sy:.4f}")
    return K_scaled, ext


# --------------------------------------------------------------------------- #
# Main per-episode pipeline.
# --------------------------------------------------------------------------- #

def run_episode(
    z,
    ep_idx: int,
    ep_start: int,
    ep_end: int,
    policy,
    cfg,
    device,
    instruction: str,
    bon_k: int,
    score_num_actions: int,
    branch_every: int,
    reward_server_port: int,
    reward_batch_size: int,
    reward_image_path: Path,
    out_path: Path,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> None:
    n_obs_steps = int(cfg.n_obs_steps)
    n_action_steps = int(cfg.n_action_steps)

    branch_indices = list(range(ep_start, ep_end, branch_every))
    print(
        f"[viz_train] episode {ep_idx}: steps {ep_start}..{ep_end} "
        f"({ep_end-ep_start} steps), {len(branch_indices)} branches"
    )

    candidate_actions = np.zeros((len(branch_indices), bon_k, n_action_steps, 7), dtype=np.float32)
    per_candidate_rewards = np.zeros((len(branch_indices), bon_k, score_num_actions), dtype=np.float32)
    per_candidate_mean = np.zeros((len(branch_indices), bon_k), dtype=np.float32)
    selected_index = np.zeros((len(branch_indices),), dtype=np.int32)
    selected_reward = np.zeros((len(branch_indices),), dtype=np.float32)
    frames = np.zeros((len(branch_indices),) + z["data/obs/agentview_image"].shape[1:], dtype=np.uint8)
    tcp_world_p = np.zeros((len(branch_indices), 3), dtype=np.float32)
    tcp_world_q = np.zeros((len(branch_indices), 4), dtype=np.float32)

    for r, step_idx in enumerate(branch_indices):
        # Frame + EE pose from zarr
        frame = np.asarray(z["data/obs/agentview_image"][step_idx])
        ee = np.asarray(z["data/obs/end_effector_pose"][step_idx], dtype=np.float32)  # (7,)
        frames[r] = frame
        tcp_world_p[r] = ee[:3]
        tcp_world_q[r] = ee[3:]

        # Save the JPEG the verifier expects (256x256).
        _ed.save_reward_image(frame, reward_image_path)

        # Build obs dict relative to this step, then move to device.
        obs_dict = build_obs_window_from_zarr(z, step_idx, n_obs_steps)
        obs_dict = {k: v.to(device=device) for k, v in obs_dict.items()}

        # Sample K candidates batched.
        sample_seed = 17 * 100000 + step_idx * 1000  # match eval_diffusion pattern
        torch.manual_seed(sample_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(sample_seed)
        batched = {k: v.expand(bon_k, *v.shape[1:]).contiguous() for k, v in obs_dict.items()}
        with torch.no_grad():
            result = policy.predict_action(batched)
        chunks = result["action"].detach().cpu().numpy()
        assert chunks.shape == (bon_k, n_action_steps, 7), \
            f"expected ({bon_k}, {n_action_steps}, 7), got {chunks.shape}"
        candidate_actions[r] = chunks.astype(np.float32)

        # Score with verifier.
        score = max(1, min(int(score_num_actions), n_action_steps))
        actions_flat = chunks[:, :score, :].reshape(-1, 7)
        flat_rewards = _ed.get_verifier_rewards(
            instruction=instruction,
            image_path=reward_image_path,
            actions=actions_flat,
            reward_server_port=int(reward_server_port),
            reward_batch_size=int(reward_batch_size),
        )
        Q_per = np.asarray(flat_rewards, dtype=np.float32).reshape(bon_k, score)
        Q = Q_per.mean(axis=1)
        per_candidate_rewards[r] = Q_per
        per_candidate_mean[r] = Q
        sel = int(np.argmax(Q))
        selected_index[r] = sel
        selected_reward[r] = float(Q[sel])

        if r % 5 == 0 or r == len(branch_indices) - 1:
            print(
                f"[viz_train]   ep={ep_idx} branch {r+1}/{len(branch_indices)} "
                f"step={step_idx} chosen={sel} Q[min,max]=[{Q.min():.3f},{Q.max():.3f}]",
                flush=True,
            )

    # Write npz in bon_q format.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        seed=np.int32(ep_idx),  # use ep_idx as "seed" stand-in
        ep_idx=np.int32(ep_idx),
        success=np.int32(0),
        truncated=np.int32(0),
        num_steps=np.int32(int(ep_end - ep_start)),
        bon_k=np.int32(int(bon_k)),
        bon_replan_every_n_steps=np.int32(int(branch_every)),
        bon_score_num_actions=np.int32(int(score_num_actions)),
        branch_t=np.asarray([s - ep_start for s in branch_indices], dtype=np.int32),
        candidate_actions=candidate_actions,
        per_candidate_rewards=per_candidate_rewards,
        per_candidate_mean_reward=per_candidate_mean,
        selected_index=selected_index,
        selected_reward=selected_reward,
        frames=frames,
        tcp_world_p=tcp_world_p,
        tcp_world_q=tcp_world_q,
        cam_intrinsic=intrinsic,
        cam_extrinsic=np.broadcast_to(extrinsic, (len(branch_indices), 3, 4)).copy(),
        cam_name=np.asarray("3rd_view_camera"),
    )
    print(f"[viz_train]   wrote -> {out_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zarr", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--task", default="widowx_put_eggplant_in_basket")
    p.add_argument("--instruction", default="put eggplant into yellow basket")
    p.add_argument("--num-episodes", type=int, default=5)
    p.add_argument("--ep-start", type=int, default=0,
                   help="Index of the first zarr episode to process.")
    p.add_argument("--bon-k", type=int, default=8)
    p.add_argument("--score-num-actions", type=int, default=4)
    p.add_argument("--branch-every", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--reward-server-port", type=int, default=0,
                   help="0 => in-process verifier.")
    p.add_argument("--reward-batch-size", type=int, default=16)
    p.add_argument("--reward-image-path", default="./transfer_images/viz_train_img.jpg")
    p.add_argument("--target-img-size", type=int, default=224,
                   help="Size of agentview_image in the zarr (square).")
    args = p.parse_args()

    import zarr
    z = zarr.open(str(args.zarr), "r")
    ep_ends = np.asarray(z["meta/episode_ends"][:])
    print(f"[viz_train] zarr has {len(ep_ends)} episodes; total steps={int(ep_ends[-1])}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    # Camera (one-time)
    intrinsic, extrinsic = get_widowx_camera_params(args.task, int(args.target_img_size))

    # Policy
    policy, cfg = _ed.load_policy(
        checkpoint=str(args.checkpoint), device=device, use_ema=not args.no_ema,
    )

    # Verifier
    _ed.check_verifier_health(int(args.reward_server_port))
    _ed._get_token_action_converter()
    print("[viz_train] TokenActionConverter ready.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    reward_image_path = Path(args.reward_image_path).absolute()

    for k in range(int(args.num_episodes)):
        ep_idx = int(args.ep_start) + k
        ep_start = int(ep_ends[ep_idx - 1]) if ep_idx > 0 else 0
        ep_end = int(ep_ends[ep_idx])
        out = args.out_dir / f"train_ep{ep_idx:04d}_aug.npz"
        run_episode(
            z=z,
            ep_idx=ep_idx,
            ep_start=ep_start,
            ep_end=ep_end,
            policy=policy,
            cfg=cfg,
            device=device,
            instruction=str(args.instruction),
            bon_k=int(args.bon_k),
            score_num_actions=int(args.score_num_actions),
            branch_every=int(args.branch_every),
            reward_server_port=int(args.reward_server_port),
            reward_batch_size=int(args.reward_batch_size),
            reward_image_path=reward_image_path,
            out_path=out,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
        )


if __name__ == "__main__":
    main()
