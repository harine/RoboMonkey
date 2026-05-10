"""
analyze_variance.py

Measure the action variance/uncertainty produced by a diffusion_policy
checkpoint on the `widowx_put_eggplant_in_basket` task. Run via the
``analyze_variance.sh`` wrapper to compare both architectures side-by-side.

Two complementary measurements are made depending on architecture:

  DiffusionUnet
  -------------
  Stochastic at inference time (DDPM noise).  We run N independent forward
  passes from the SAME observation and compute the empirical std across them.
  This gives the aleatoric/sampling variance baked into the diffusion process.

  MLP (Gaussian)
  --------------
  The network outputs an explicit Normal distribution.  We read the
  predicted std directly from `log_std_head` (no need to sample).  We also
  draw N samples and compute the empirical std for cross-check.

Output
------
  * Per-step per-dimension std table in the terminal.
  * A PNG with two panels: (1) mean ± std over time for each action dim,
    (2) per-dimension aggregate std distribution across rollout steps.
  * A JSON with raw numbers: mean, std, min, max per action dimension.

Usage
-----
  python scriptsv2/action_variance/analyze_variance.py \\
      --checkpoint <ckpt>.ckpt \\
      --n-samples 32 \\
      --n-rollout-steps 60 \\
      --n-episodes 5 \\
      --output-dir data/eval/variance_eggplant_unet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

# ---------------------------------------------------------------------------
ACTION_LABELS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
TASK_NAME = "widowx_put_eggplant_in_basket"

OBS_KEYS_AND_DIMS: Dict[str, int] = {
    "end_effector_pose": 7,
    "end_effector_vel_lin_ang_b": 6,
    "arm_joint_pos": 6,
    "joint_vel": 6,
    "last_arm_action": 6,
    "last_gripper_action": 1,
    "insertive_asset_pose": 7,
    "receptive_asset_pose": 7,
    "insertive_asset_in_receptive_asset_frame": 7,
}
NUM_ARM_JOINTS = 6


# ---------------------------------------------------------------------------
#  Shared helpers (mirror of eval_diffusion/eval_diffusion.py)
# ---------------------------------------------------------------------------

def _pose_to_vec(pose) -> np.ndarray:
    return np.concatenate([np.asarray(pose.p, dtype=np.float32),
                           np.asarray(pose.q, dtype=np.float32)]).astype(np.float32)

def _tcp_pose_vec(obs_or_pose) -> np.ndarray:
    if hasattr(obs_or_pose, "p") and hasattr(obs_or_pose, "q"):
        return _pose_to_vec(obs_or_pose)
    return np.asarray(obs_or_pose, dtype=np.float32).reshape(-1)[:7]

def _ee_body_velocity(tcp_link) -> np.ndarray:
    R_bw = tcp_link.pose.to_transformation_matrix()[:3, :3]
    v_b = R_bw.T @ np.asarray(tcp_link.get_velocity(), dtype=np.float64)
    w_b = R_bw.T @ np.asarray(tcp_link.get_angular_velocity(), dtype=np.float64)
    return np.concatenate([v_b, w_b]).astype(np.float32)

def _rel_pose_vec(src, tgt) -> np.ndarray:
    return _pose_to_vec((tgt.inv() * src))

def build_obs_step(env, obs, prev_arm, prev_grip, src_obj, tgt_obj, tcp):
    qpos = np.asarray(env.agent.robot.get_qpos(), dtype=np.float32)
    qvel = np.asarray(env.agent.robot.get_qvel(), dtype=np.float32)
    return {
        "end_effector_pose":                      _tcp_pose_vec(obs["extra"]["tcp_pose"]),
        "end_effector_vel_lin_ang_b":              _ee_body_velocity(tcp),
        "arm_joint_pos":                           qpos[:NUM_ARM_JOINTS].copy(),
        "joint_vel":                               qvel[:NUM_ARM_JOINTS].copy(),
        "last_arm_action":                         prev_arm.astype(np.float32, copy=True),
        "last_gripper_action":                     prev_grip.astype(np.float32, copy=True),
        "insertive_asset_pose":                    _pose_to_vec(src_obj.pose),
        "receptive_asset_pose":                    _pose_to_vec(tgt_obj.pose),
        "insertive_asset_in_receptive_asset_frame": _rel_pose_vec(src_obj.pose, tgt_obj.pose),
    }

class ObsWindow:
    def __init__(self, n):
        import collections
        self.n = n
        self.buf = collections.deque(maxlen=n)
    def push(self, s):
        self.buf.append(s)
    def to_tensor(self, device):
        steps = list(self.buf)
        if len(steps) < self.n:
            steps = [steps[0]] * (self.n - len(steps)) + steps
        out = {}
        for k in OBS_KEYS_AND_DIMS:
            arr = np.stack([s[k] for s in steps], axis=0).astype(np.float32)
            out[k] = torch.from_numpy(arr).unsqueeze(0).to(device)
        return out


# ---------------------------------------------------------------------------
#  Checkpoint loading
# ---------------------------------------------------------------------------

def load_policy(checkpoint: str, device: torch.device):
    import dill, hydra
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill,
                         map_location="cpu", weights_only=False)
    cfg = payload["cfg"]
    state_dicts = payload["state_dicts"]

    policy_type = cfg.policy._target_.split(".")[-1]
    use_ema = bool(cfg.training.use_ema) and "ema_model" in state_dicts
    src_key = "ema_model" if use_ema else "model"
    print(f"[var] policy={policy_type}  weights={src_key}  "
          f"n_obs={cfg.n_obs_steps}  n_act={cfg.n_action_steps}")

    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(state_dicts[src_key], strict=False)
    policy.to(device).eval()
    return policy, cfg, policy_type


# ---------------------------------------------------------------------------
#  Variance measurements
# ---------------------------------------------------------------------------

def diffusion_variance(policy, obs_dict: Dict[str, torch.Tensor],
                       n_samples: int, device: torch.device,
                       n_action_steps: int) -> Dict[str, np.ndarray]:
    """Run N forward passes and return empirical stats over the first
    `n_action_steps` actions.  Returns dict with 'mean', 'std', 'min', 'max'
    each of shape (n_action_steps, 7)."""
    samples = []
    with torch.no_grad():
        for seed in range(n_samples):
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)
            out = policy.predict_action(obs_dict)
            a = out["action"][0, :n_action_steps].cpu().numpy()  # (T, 7)
            samples.append(a)
    arr = np.stack(samples, axis=0)  # (N, T, 7)
    return {
        "samples": arr,
        "mean":    arr.mean(axis=0),
        "std":     arr.std(axis=0),
        "min":     arr.min(axis=0),
        "max":     arr.max(axis=0),
    }


def mlp_variance(policy, obs_dict: Dict[str, torch.Tensor],
                 n_samples: int, device: torch.device,
                 n_action_steps: int, n_obs_steps: int,
                 normalizer: Any) -> Dict[str, np.ndarray]:
    """Read the Normal distribution parameters directly from the MLP heads,
    then also draw N samples for cross-check."""
    from diffusion_policy.common.pytorch_util import dict_apply

    with torch.no_grad():
        nobs = normalizer.normalize(obs_dict)
        To = n_obs_steps
        this_nobs = dict_apply(nobs,
                               lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        feats = policy.obs_encoder(this_nobs)
        feats = feats.reshape(1, To, -1).reshape(1, -1)

        h = policy.trunk(feats)
        mean_n = policy.mean_head(h)   # (1, horizon*7)
        lstd_n = policy.log_std_head(h).clamp(
            min=policy.log_std_limits[0], max=policy.log_std_limits[1])
        std_n  = torch.exp(lstd_n)

        start = To - 1
        end   = start + n_action_steps

        mean_n = mean_n.reshape(1, -1, 7)[:, start:end, :]   # (1, T, 7)
        std_n  = std_n.reshape(1, -1, 7)[:, start:end, :]

        # Unnormalize mean; std scales by the action normalizer scale
        mean_unnorm = normalizer["action"].unnormalize(mean_n)[0].cpu().numpy()
        scale = normalizer["action"].params_dict["scale"].cpu().numpy()  # (7,)
        std_unnorm  = (std_n[0].cpu().numpy() / scale)        # (T, 7)

    # Empirical samples for cross-check
    samples = []
    with torch.no_grad():
        for s in range(n_samples):
            torch.manual_seed(s)
            out = policy.predict_action(obs_dict)
            samples.append(out["action"][0, :n_action_steps].cpu().numpy())
    emp = np.stack(samples, axis=0)

    return {
        "dist_mean":     mean_unnorm,           # (T, 7)  analytic
        "dist_std":      std_unnorm,            # (T, 7)  analytic
        "mean":          emp.mean(axis=0),      # (T, 7)  empirical
        "std":           emp.std(axis=0),       # (T, 7)  empirical
        "min":           emp.min(axis=0),
        "max":           emp.max(axis=0),
        "samples":       emp,
    }


# ---------------------------------------------------------------------------
#  Per-step collection loop
# ---------------------------------------------------------------------------

def _annotate_frame(frame: np.ndarray, t: int, per_dim_std: List[float],
                    mean_std: float, success: bool) -> np.ndarray:
    """Draw per-dim std bars and text onto a copy of `frame` (HxWx3 uint8)."""
    try:
        import cv2
    except ImportError:
        return frame  # no OpenCV → return unannotated

    img = frame.copy()
    H, W = img.shape[:2]

    # Semi-transparent dark banner at top (for text)
    banner_h = 20 + len(ACTION_LABELS) * 16 + 10
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (W, banner_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    # Title line
    status = "SUCCESS" if success else f"t={t:03d}"
    cv2.putText(img, f"step {t:03d}  mean_std={mean_std:.4f}  {status}",
                (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1,
                cv2.LINE_AA)

    # Per-dim std bars (coloured)
    COLORS = [
        (100, 200, 255), (100, 255, 160), (255, 200,  80),
        (255, 110, 110), (200, 100, 255), (100, 220, 220),
        (255, 180, 100),
    ]
    max_std = max(per_dim_std) if max(per_dim_std) > 0 else 1.0
    bar_max_w = W // 2 - 8
    for d, (label, val) in enumerate(zip(ACTION_LABELS, per_dim_std)):
        y_base = 26 + d * 16
        bar_w  = int(val / max_std * bar_max_w)
        cv2.rectangle(img, (6, y_base - 1), (6 + max(bar_w, 2), y_base + 11),
                      COLORS[d], -1)
        cv2.putText(img, f"{label}: {val:.4f}",
                    (10, y_base + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                    (240, 240, 240), 1, cv2.LINE_AA)

    return img


def collect_variance_over_rollout(
    env, policy, cfg, device, policy_type, n_samples, n_rollout_steps, seed,
    capture_frames: bool = False,
):
    """Step the env for `n_rollout_steps` (using the policy's mean action),
    and at each step measure the action variance.

    Returns a list of per-step variance dicts. If `capture_frames=True`,
    each dict also contains an 'annotated_frame' key with a HxWx3 uint8
    numpy array overlaid with the per-dim std bars.
    """
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    n_obs   = int(cfg.n_obs_steps)
    n_act   = int(cfg.n_action_steps)
    obs, _  = env.reset(seed=seed)

    src_obj = getattr(env, "episode_source_obj", None)
    tgt_obj = getattr(env, "episode_target_obj", None)
    tcp     = getattr(env, "tcp", None)
    if None in (src_obj, tgt_obj, tcp):
        raise RuntimeError("env missing episode_source_obj / episode_target_obj / tcp")

    prev_arm  = np.zeros(6, dtype=np.float32)
    prev_grip = np.zeros(1, dtype=np.float32)
    window    = ObsWindow(n=n_obs)
    step_records: List[Dict] = []
    pending_actions: List[np.ndarray] = []
    success = False

    from transforms3d.euler import euler2axangle

    def _maniskill(a):
        a = a.copy().astype(np.float32)
        ax, ang = euler2axangle(float(a[3]), float(a[4]), float(a[5]))
        a[3:6] = np.asarray(ax, dtype=np.float32) * np.float32(ang)
        g = 2.0 * float(a[6]) - 1.0
        a[6] = 1.0 if g >= 0.0 else -1.0
        return a

    for t in range(n_rollout_steps):
        step_obs = build_obs_step(env, obs, prev_arm, prev_grip,
                                  src_obj, tgt_obj, tcp)
        window.push(step_obs)
        obs_dict = window.to_tensor(device)

        # ---- capture raw camera frame BEFORE variance sampling ----
        raw_frame: np.ndarray | None = None
        if capture_frames:
            try:
                raw_frame = np.asarray(
                    get_image_from_maniskill2_obs_dict(env, obs), dtype=np.uint8
                )
            except Exception:
                raw_frame = None

        # ---- measure variance at this observation ----
        if policy_type.lower().startswith("mlp"):
            vstats = mlp_variance(policy, obs_dict, n_samples, device,
                                  n_act, n_obs, policy.normalizer)
        else:
            vstats = diffusion_variance(policy, obs_dict, n_samples, device, n_act)

        per_dim_std = vstats["std"].mean(axis=0).tolist()   # (7,) averaged over chunk
        mean_std    = float(vstats["std"].mean())

        rec: Dict[str, Any] = {
            "t":           t,
            "mean_std":    mean_std,
            "per_dim_std": per_dim_std,
            "vstats":      vstats,
        }

        # Annotate the frame with std bars
        if raw_frame is not None and raw_frame.ndim == 3 and raw_frame.shape[-1] == 3:
            rec["annotated_frame"] = _annotate_frame(
                raw_frame, t, per_dim_std, mean_std, success=False
            )

        step_records.append(rec)

        # ---- execute the mean action (one step) ----
        if not pending_actions:
            with torch.no_grad():
                torch.manual_seed(0)
                out = policy.predict_action(obs_dict)
            chunk = out["action"][0].detach().cpu().numpy()    # (n_act, 7)
            pending_actions = list(chunk)

        action_raw = pending_actions.pop(0).astype(np.float32)
        obs, _reward, done, trunc, _info = env.step(_maniskill(action_raw))
        prev_arm  = action_raw[:6].copy()
        prev_grip = action_raw[6:7].copy()

        if bool(done):
            success = True
            # Annotate last frame as success
            if "annotated_frame" in rec:
                rec["annotated_frame"] = _annotate_frame(
                    raw_frame, t, per_dim_std, mean_std, success=True
                )
            print(f"  SUCCESS at t={t}")
            break
        if bool(trunc):
            print(f"  truncated at t={t}")
            break

    return step_records, success


# ---------------------------------------------------------------------------
#  Video writer (reuse from eval_diffusion/eval_diffusion.py logic)
# ---------------------------------------------------------------------------

def save_video_from_records(records: List[Dict], path: Path, fps: int = 10) -> bool:
    """Collect annotated_frame from each record and write to mp4."""
    frames = [r["annotated_frame"] for r in records if "annotated_frame" in r]
    if not frames:
        return False
    arr = np.stack(frames, axis=0)   # (T, H, W, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
        with imageio.get_writer(str(path), fps=fps, codec="libx264",
                                quality=8, macro_block_size=1) as w:
            for f in arr:
                w.append_data(f)
        return True
    except Exception as e_iio:
        try:
            import mediapy
            mediapy.write_video(str(path), arr, fps=fps)
            return True
        except Exception as e_mp:
            print(f"[var] video write failed: imageio={e_iio!r}  mediapy={e_mp!r}")
            return False


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def plot_variance(all_records: List[List[Dict]], policy_type: str,
                  out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Flatten all records from all episodes
    all_steps = [r for ep in all_records for r in ep]
    T = len(all_steps)
    if T == 0:
        print("[var] no steps to plot")
        return

    # Per-step per-dim std (7,) averaged over chunk
    std_over_time = np.array([s["per_dim_std"] for s in all_steps])  # (T, 7)
    timesteps     = np.array([s["t"] for s in all_steps])

    # Per-dim aggregate std (collapse over time)
    per_dim_agg   = std_over_time.mean(axis=0)   # (7,)
    per_dim_p25   = np.percentile(std_over_time, 25, axis=0)
    per_dim_p75   = np.percentile(std_over_time, 75, axis=0)

    colors = cm.tab10(np.linspace(0, 1, 7))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Action variance — {policy_type}  ({TASK_NAME})\n"
        f"{len(all_records)} episodes, {T} total steps",
        fontsize=13, fontweight="bold",
    )

    # ── Left: std over rollout time ──────────────────────────────────────
    ax = axes[0]
    for d, (label, c) in enumerate(zip(ACTION_LABELS, colors)):
        ax.plot(timesteps, std_over_time[:, d], label=label, color=c, lw=1.5, alpha=0.85)
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Action std (un-normalized)")
    ax.set_title("Per-dim std over time")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── Right: per-dim aggregate bar chart ───────────────────────────────
    ax = axes[1]
    x = np.arange(7)
    bars = ax.bar(x, per_dim_agg, color=colors, alpha=0.85)
    ax.errorbar(x, per_dim_agg,
                yerr=[np.clip(per_dim_agg - per_dim_p25, 0, None),
                      np.clip(per_dim_p75 - per_dim_agg, 0, None)],
                fmt="none", color="black", capsize=4, lw=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ACTION_LABELS, fontsize=9)
    ax.set_ylabel("Mean std (un-normalized)")
    ax.set_title("Per-dim std (mean ± IQR)")
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate each bar
    for bar, val in zip(bars, per_dim_agg):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[var] saved plot -> {out_path}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default=TASK_NAME)
    parser.add_argument("--n-samples", type=int, default=32,
                        help="Number of independent samples per observation.")
    parser.add_argument("--n-rollout-steps", type=int, default=60,
                        help="Env steps per episode during measurement.")
    parser.add_argument("--n-episodes", type=int, default=5,
                        help="Number of env resets to average over.")
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--save-videos", type=int, default=0,
                        help="Save annotated MP4 videos for the first N episodes "
                             "(0 = off). Frames show per-dim std bars overlaid. "
                             "Requires --output-dir.")
    parser.add_argument("--video-fps", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[var] CUDA unavailable, falling back to CPU")
        device = torch.device("cpu")

    policy, cfg, policy_type = load_policy(args.checkpoint, device)

    import simpler_env  # noqa
    env = simpler_env.make(args.task)

    save_videos_n = max(0, int(args.save_videos))
    if save_videos_n > 0 and args.output_dir is None:
        print("[var] --save-videos requires --output-dir; disabling.")
        save_videos_n = 0
    video_dir = (Path(args.output_dir) / "videos") if (args.output_dir and save_videos_n > 0) else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)

    print(f"[var] task={args.task}  n_samples={args.n_samples}  "
          f"n_rollout_steps={args.n_rollout_steps}  n_episodes={args.n_episodes}  "
          f"save_videos={save_videos_n}")

    # Check OpenCV is available if we need frame annotation; warn otherwise.
    if save_videos_n > 0:
        try:
            import cv2  # noqa
        except ImportError:
            print("[var] WARNING: opencv-python not found. "
                  "Videos will be saved without std-bar overlay. "
                  "Install with: pip install opencv-python-headless")

    all_records: List[List[Dict]] = []
    for ep_i in range(args.n_episodes):
        seed = args.start_seed + ep_i
        capture = ep_i < save_videos_n
        print(f"\n[var] === episode {ep_i+1}/{args.n_episodes}  seed={seed}  "
              f"capture_frames={capture} ===")
        recs, ep_success = collect_variance_over_rollout(
            env=env, policy=policy, cfg=cfg, device=device,
            policy_type=policy_type,
            n_samples=args.n_samples,
            n_rollout_steps=args.n_rollout_steps,
            seed=seed,
            capture_frames=capture,
        )
        all_records.append(recs)
        mean_std = np.mean([r["mean_std"] for r in recs])
        print(f"  mean_std over episode = {mean_std:.5f}")

        if capture and video_dir is not None:
            status = "success" if ep_success else "fail"
            vpath = video_dir / f"ep{ep_i:03d}_seed{seed}_{status}.mp4"
            ok = save_video_from_records(recs, vpath, fps=int(args.video_fps))
            if ok:
                print(f"  saved video  -> {vpath}")
            else:
                print(f"  video write failed for ep{ep_i}")

    # ── Aggregate across all episodes ────────────────────────────────────
    all_steps = [r for ep in all_records for r in ep]
    per_dim_std = np.array([s["per_dim_std"] for s in all_steps])  # (T_total, 7)
    agg_mean = per_dim_std.mean(axis=0)   # (7,)
    agg_std  = per_dim_std.std(axis=0)
    agg_min  = per_dim_std.min(axis=0)
    agg_max  = per_dim_std.max(axis=0)
    agg_p25  = np.percentile(per_dim_std, 25, axis=0)
    agg_p75  = np.percentile(per_dim_std, 75, axis=0)

    print("\n[var] ══════════════ VARIANCE SUMMARY ══════════════")
    print(f"  policy: {policy_type}   task: {args.task}")
    print(f"  {len(all_steps)} total measurement steps  "
          f"({args.n_episodes} episodes × ~{args.n_rollout_steps} steps)\n")
    header = f"  {'dim':<10} {'mean_std':>10} {'std_of_std':>11} "
    header += f"{'p25':>8} {'p75':>8} {'min':>10} {'max':>10}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for d, label in enumerate(ACTION_LABELS):
        print(f"  {label:<10} {agg_mean[d]:>10.5f} {agg_std[d]:>11.5f} "
              f"{agg_p25[d]:>8.5f} {agg_p75[d]:>8.5f} "
              f"{agg_min[d]:>10.5f} {agg_max[d]:>10.5f}")
    print(f"\n  Overall mean std (all dims): {agg_mean.mean():.5f}")
    print("═" * 54)

    out = {
        "checkpoint": args.checkpoint,
        "policy_type": policy_type,
        "task": args.task,
        "n_samples": args.n_samples,
        "n_episodes": args.n_episodes,
        "n_rollout_steps": args.n_rollout_steps,
        "total_steps": len(all_steps),
        "per_dim": {
            label: {
                "mean_std":  float(agg_mean[d]),
                "std_of_std": float(agg_std[d]),
                "p25":       float(agg_p25[d]),
                "p75":       float(agg_p75[d]),
                "min":       float(agg_min[d]),
                "max":       float(agg_max[d]),
            }
            for d, label in enumerate(ACTION_LABELS)
        },
        "overall_mean_std": float(agg_mean.mean()),
    }

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "variance_summary.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[var] wrote JSON -> {out_dir / 'variance_summary.json'}")
        plot_variance(
            all_records, policy_type,
            out_path=out_dir / "variance_plot.png",
        )


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    main()
