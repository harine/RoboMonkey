"""SimplerEnv eval for ``SearchPolicyRoboMonkey`` (state-based search policy).

The search policy has its own in-process verifier (built into the policy at
hydra-instantiate time). Each ``predict_action`` call runs ``max_actions``
candidate samples, each scored by the verifier, then the transformer trunk
emits a final action chunk conditioned on the *full* (action, value) context
— so there is **no argmax over candidates**; we read out the last token's
mean.

This script mirrors :mod:`scriptsv2.eval_diffusion.eval_diffusion` for the
rollout / obs-window / video plumbing, but replaces the policy invocation
with the search-aware call.

Usage:
    python eval_search_policy.py \\
        --checkpoint <ckpt.ckpt> --num-episodes 100 --task widowx_put_eggplant_in_basket \\
        --output-dir data/eval/eggplant_search_state_corrupt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Reuse helpers from eval_diffusion (build_obs_step / ObsWindow / convert_maniskill / etc.)
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "eval_diffusion"))
from eval_diffusion import (  # type: ignore
    ObsWindow,
    build_obs_step,
    convert_maniskill,
    load_policy,
    save_video,
)

TASK_NAME = "widowx_put_eggplant_in_basket"


@torch.no_grad()
def _search_policy_predict(
    policy,
    obs_dict: Dict[str, torch.Tensor],
    mode: str = "argmax",
    viz_q: bool = False,
) -> Dict[str, Any]:
    """Run the search policy.

    Two modes:
      * ``"argmax"``: run the search loop (``max_actions`` autoregressive
        samples, each verifier-scored), then pick the candidate with the
        highest verifier value. The chosen index reveals how much context
        was used (0 = no prior samples, max_actions-1 = full search trace).
      * ``"refine"``: run ``max_actions - 1`` search samples, then do one
        more transformer forward pass conditioned on the full search trace
        and read out the last token's ``dist.mean`` — the trained policy's
        refined prediction. No argmax.

    Returns ``{"chunk": (1, n_action_steps, action_dim) tensor,
                "selected_index": int (only in argmax mode),
                "selected_value": float (only in argmax mode),
                "values": list[float] (verifier values for all candidates)}``.
    """
    # Drop low-dim obs keys the trained normalizer doesn't know about, but
    # KEEP the image key — the verifier reads it directly.
    known = set(policy.normalizer.params_dict.keys())
    image_obs_key = getattr(policy.verifier, "image_obs_key", None)
    obs_dict = {
        k: v for k, v in obs_dict.items()
        if k in known or k == image_obs_key
    }
    nobs_only = {k: v for k, v in obs_dict.items() if k in known}

    To = int(policy.n_obs_steps)
    n_action_steps = int(policy.n_action_steps)
    n_samples = int(policy.max_actions) if mode == "argmax" else int(policy.max_actions) - 1

    actions, values = policy.predict_action(
        obs_dict, policy.verifier, n_samples
    )  # (B, K, horizon, action_dim) ; (B, K)
    B = actions.shape[0]
    start = To - 1
    end = start + n_action_steps

    out: Dict[str, Any] = {"values": values[0].detach().cpu().tolist()}

    if viz_q:
        # All K candidate chunks, sliced to the executed action window so they
        # are directly comparable to the chosen chunk (mirrors BoN viz_q).
        out["candidate_actions"] = (
            actions[0, :, start:end].detach().cpu().float().numpy()
        )  # (K, n_action_steps, action_dim)

    if mode == "argmax":
        best = values.argmax(dim=1)  # (B,)
        best_action = actions[torch.arange(B, device=actions.device), best]  # (B, horizon, action_dim)
        out["chunk"] = best_action[:, start:end]
        out["selected_index"] = int(best[0].item())
        out["selected_value"] = float(values[0, int(best[0].item())].item())
        return out

    # --- refine mode ---
    from diffusion_policy.common.pytorch_util import dict_apply

    nobs = policy.normalizer.normalize(nobs_only)
    if isinstance(nobs, dict):
        this_nobs = dict_apply(
            nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
        )
    else:
        this_nobs = nobs[:, :To, ...].reshape(-1, *nobs.shape[2:])
    nobs_features = policy.obs_encoder(this_nobs)
    obs_features = nobs_features.reshape(B, -1)
    obs_features = policy.obs_projection(obs_features)

    nactions = policy.normalizer["action"].normalize(actions)
    nactions_flat = nactions.reshape(B, nactions.shape[1], -1).to(obs_features.device)
    values_in = values.to(obs_features.device)
    action_value_input = torch.cat([nactions_flat, values_in.unsqueeze(-1)], dim=-1)
    action_value_features = policy.act_projection(action_value_input)

    dist = policy.forward(obs_features, action_value_features)  # (B, T, horizon, action_dim)
    naction_pred = dist.mean[:, -1]
    action_pred = policy.normalizer["action"].unnormalize(naction_pred)
    out["chunk"] = action_pred[:, start:end]
    return out


def rollout_episode(env, policy, cfg, device: torch.device, seed: int,
                    max_steps: int, instruction: str,
                    capture_frames: bool = False,
                    mode: str = "argmax",
                    viz_q: bool = False) -> Dict[str, Any]:
    n_obs_steps = int(cfg.n_obs_steps)
    n_action_steps = int(cfg.n_action_steps)

    obs, _ = env.reset(seed=int(seed))
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
    pending_actions: List[np.ndarray] = []
    frames: List[np.ndarray] = []
    branches: Optional[List[Dict[str, Any]]] = [] if viz_q else None

    # Always need a frame for the in-process RoboMonkey verifier; capturing
    # for video output is a strict subset.
    from simpler_env.utils.env.observation_utils import (  # type: ignore
        get_image_from_maniskill2_obs_dict,
    )

    while t < max_steps:
        step_obs = build_obs_step(
            env=env, obs=obs,
            prev_arm_action=prev_arm_action,
            prev_gripper_action=prev_gripper_action,
            source_obj=source_obj, target_obj=target_obj, tcp_link=tcp_link,
        )
        window.push(step_obs)

        current_frame: Optional[np.ndarray] = None
        try:
            img = get_image_from_maniskill2_obs_dict(env, obs)
            arr = np.asarray(img, dtype=np.uint8)
            if arr.ndim == 3 and arr.shape[-1] == 3:
                current_frame = arr
                if capture_frames:
                    frames.append(arr)
        except Exception as e:
            raise RuntimeError(f"failed to capture verifier image: {e!r}") from e

        if not pending_actions:
            obs_dict = window.to_tensor_dict(device=device)
            # Verifier expects (B, To, H, W, 3). Repeat the current frame
            # across the obs window (only the last step is read by the
            # verifier anyway).
            assert current_frame is not None
            img_t = torch.from_numpy(current_frame).to(device=device)
            obs_dict["agentview_image"] = img_t.unsqueeze(0).unsqueeze(0).expand(
                1, n_obs_steps, *current_frame.shape
            )
            pred = _search_policy_predict(
                policy, obs_dict, mode=mode, viz_q=viz_q
            )
            chunk = pred["chunk"][0].detach().cpu().numpy()
            if branches is not None and current_frame is not None:
                branches.append({
                    "t": int(t),
                    "candidate_actions": pred["candidate_actions"].astype(
                        np.float32, copy=True
                    ),
                    "values": np.asarray(pred["values"], dtype=np.float32),
                    "selected_index": int(pred.get("selected_index", -1)),
                    "selected_value": float(pred.get("selected_value", float("nan"))),
                    "frame": current_frame.astype(np.uint8, copy=True),
                })
            replan_t = t
            replan_selected_index = pred.get("selected_index")
            replan_selected_value = pred.get("selected_value")
            replan_values = pred.get("values")
            assert chunk.shape == (n_action_steps, 7), \
                f"expected ({n_action_steps}, 7), got {chunk.shape}"
            pending_actions = list(chunk.astype(np.float32, copy=False))

        action_vla = pending_actions.pop(0).astype(np.float32)
        env_action = convert_maniskill(action_vla.copy())
        obs, reward, done, trunc, info = env.step(env_action)

        prev_arm_action = action_vla[:6].copy()
        prev_gripper_action = action_vla[6:7].copy()

        is_replan = (t == replan_t)
        step_entry: Dict[str, Any] = {
            "t": t,
            "reward": float(reward),
            "done": bool(done),
            "trunc": bool(trunc),
        }
        if is_replan and replan_selected_index is not None:
            step_entry["selected_index"] = int(replan_selected_index)
            step_entry["selected_value"] = float(replan_selected_value)
        if is_replan and replan_values is not None:
            step_entry["values"] = list(replan_values)
        step_log.append(step_entry)
        if bool(done):
            success = True
            break
        if bool(trunc):
            truncated = True
            break
        t += 1

    return {
        "success": bool(success),
        "truncated": bool(truncated),
        "num_steps": int(t + 1 if (success or truncated) else t),
        "steps": step_log,
        "frames": frames,
        "branches": branches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--task", default=TASK_NAME)
    parser.add_argument("--save-videos", type=int, default=0)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--repeat-seed", action="store_true",
                        help="If set, every episode uses --start-seed (same "
                             "asset layout, different stochastic rollouts). "
                             "Otherwise episode i uses --start-seed + i.")
    parser.add_argument("--mode", choices=["argmax", "refine"], default="argmax",
                        help="argmax: pick the highest-verifier-value sample from "
                             "the search loop (and log selected_index). "
                             "refine: final transformer forward pass over the "
                             "search trace, take dist.mean of last token.")
    parser.add_argument("--viz-q", action="store_true",
                        help="Save per-replan sampled candidate actions + "
                             "verifier values + frame to "
                             "<output_dir>/search_q/ep<idx>_seed<seed>.npz "
                             "for later visualization (mirrors BoN viz_q).")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[eval] CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")

    policy, cfg = load_policy(
        checkpoint=args.checkpoint, device=device, use_ema=not args.no_ema,
    )
    print(f"[eval] max_actions={int(policy.max_actions)}  (in-process verifier)")

    import simpler_env  # noqa: F401
    from simpler_env import make as make_env
    print(f"[eval] Creating SimplerEnv task: {args.task}")
    env = make_env(args.task)
    instr = env.get_language_instruction() or args.task
    print(f"[eval] instruction: {instr!r}")

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        ep_log_file = open(output_dir / "episodes.jsonl", "w")
    else:
        ep_log_file = None

    save_videos_n = max(0, int(args.save_videos))
    if save_videos_n > 0 and output_dir is None:
        print("[eval] --save-videos>0 requires --output-dir; disabling.")
        save_videos_n = 0
    video_dir = (output_dir / "videos") if (output_dir is not None and save_videos_n > 0) else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)

    viz_q = bool(args.viz_q)
    if viz_q and output_dir is None:
        print("[eval] --viz-q requires --output-dir; disabling.")
        viz_q = False
    viz_q_dir = (output_dir / "search_q") if (output_dir is not None and viz_q) else None
    if viz_q_dir is not None:
        viz_q_dir.mkdir(parents=True, exist_ok=True)
        print(f"[eval] viz_q enabled -> saving sampled actions to {viz_q_dir}")

    if args.seeds:
        explicit_seeds: Optional[List[int]] = [
            int(s) for s in args.seeds.split(",") if s.strip()
        ]
        num_episodes = len(explicit_seeds)
    else:
        explicit_seeds = None
        num_episodes = int(args.num_episodes)

    successes = 0
    truncations = 0
    durations: List[float] = []
    t0 = time.time()
    all_selected_indices: List[int] = []

    for i in range(num_episodes):
        if explicit_seeds is not None:
            ep_seed = explicit_seeds[i]
        elif args.repeat_seed:
            ep_seed = int(args.start_seed)
        else:
            ep_seed = int(args.start_seed) + i
        capture = (i < save_videos_n)
        ep_t0 = time.time()
        try:
            ep = rollout_episode(
                env=env, policy=policy, cfg=cfg, device=device,
                seed=ep_seed, max_steps=int(args.max_steps),
                instruction=str(instr), capture_frames=capture,
                mode=str(args.mode), viz_q=viz_q,
            )
        except Exception as e:
            print(f"[eval] episode {i} (seed={ep_seed}) raised: {e!r}")
            ep = {"success": False, "truncated": False, "num_steps": 0,
                  "error": repr(e), "steps": [], "frames": [], "branches": None}

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

        if viz_q_dir is not None and ep.get("branches"):
            br = ep["branches"]
            npz_path = viz_q_dir / f"ep{i:03d}_seed{ep_seed}.npz"
            np.savez_compressed(
                npz_path,
                seed=np.int32(ep_seed),
                ep_idx=np.int32(i),
                success=np.int32(int(ep["success"])),
                truncated=np.int32(int(ep["truncated"])),
                num_steps=np.int32(int(ep["num_steps"])),
                mode=str(args.mode),
                max_actions=np.int32(int(policy.max_actions)),
                branch_t=np.asarray([b["t"] for b in br], dtype=np.int32),
                candidate_actions=np.stack(
                    [b["candidate_actions"] for b in br], axis=0
                ),
                values=np.stack([b["values"] for b in br], axis=0),
                selected_index=np.asarray(
                    [b["selected_index"] for b in br], dtype=np.int32
                ),
                selected_value=np.asarray(
                    [b["selected_value"] for b in br], dtype=np.float32
                ),
                frames=np.stack([b["frame"] for b in br], axis=0),
            )
            print(f"[eval]   saved sampled actions -> {npz_path} ({len(br)} replans)")

        ep_selected_indices = [
            int(s["selected_index"]) for s in ep.get("steps", [])
            if "selected_index" in s
        ]
        ep_selected_values = [
            float(s["selected_value"]) for s in ep.get("steps", [])
            if "selected_value" in s
        ]
        ep_values_per_replan = [
            list(s["values"]) for s in ep.get("steps", [])
            if "values" in s
        ]
        all_selected_indices.extend(ep_selected_indices)

        if ep_log_file is not None:
            ep_compact = {
                "ep_idx": i,
                "seed": ep_seed,
                "success": ep["success"],
                "truncated": ep["truncated"],
                "num_steps": ep["num_steps"],
                "duration_s": durations[-1],
            }
            if ep_selected_indices:
                ep_compact["selected_indices"] = ep_selected_indices
            if ep_selected_values:
                ep_compact["selected_values"] = ep_selected_values
            if ep_values_per_replan:
                ep_compact["q_values"] = ep_values_per_replan
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
        "seeds": ([int(s) for s in explicit_seeds] if explicit_seeds is not None else None),
        "max_steps": int(args.max_steps),
        "use_ema": (not args.no_ema),
        "mode": str(args.mode),
        "max_actions": int(policy.max_actions),
        "num_successes": int(successes),
        "num_truncated": int(truncations),
        "success_rate": float(success_rate),
        "mean_episode_time_s": float(np.mean(durations)) if durations else 0.0,
        "total_time_s": float(total_t),
    }
    if all_selected_indices:
        idx_arr = np.asarray(all_selected_indices, dtype=np.int64)
        counts = np.bincount(idx_arr, minlength=int(policy.max_actions)).tolist()
        summary["selected_index_histogram"] = counts
        summary["selected_index_mean"] = float(idx_arr.mean())
        summary["selected_index_median"] = float(np.median(idx_arr))
        summary["selected_index_min"] = int(idx_arr.min())
        summary["selected_index_max"] = int(idx_arr.max())
        summary["num_replans"] = int(idx_arr.size)
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
