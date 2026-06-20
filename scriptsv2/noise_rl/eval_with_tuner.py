"""Evaluate a noise-conditioned search policy with a trained SAC noise tuner.

Same SimplerEnv rollout loop as ``scriptsv2/eval_search/eval_search.py``, but
``tcont_context`` is sampled from the trained ``NoiseTunerAgent`` at every
replan instead of being a fixed scalar. Pass ``--tuner-mode mean`` to use the
deterministic actor mean; ``sample`` to sample stochastically; ``fixed`` to
ignore the tuner and use ``--fixed-tcont``.
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

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_ROOT / "eval_diffusion"))

from eval_diffusion import (  # type: ignore
    ObsWindow,
    build_obs_step,
    convert_maniskill,
    load_policy,
)

from noise_tuner import NoiseTunerAgent, SACConfig  # noqa: E402
from train_noise_tuner import encode_tuner_obs  # noqa: E402


def _resolve_tcont(
        tuner: Optional[NoiseTunerAgent],
        obs_feat: torch.Tensor,
        mode: str,
        fixed: float,
    ) -> torch.Tensor:
    if mode == "fixed" or tuner is None:
        return torch.full(
            (obs_feat.shape[0],), float(fixed),
            device=obs_feat.device, dtype=torch.float32,
        )
    if mode == "sample":
        return tuner.select_action(obs_feat, deterministic=False).view(-1)
    if mode == "mean":
        return tuner.select_action(obs_feat, deterministic=True).view(-1)
    raise ValueError(f"unknown tuner-mode: {mode}")


def run_episode(env, policy, tuner, args, cfg, device, seed: int) -> Dict[str, Any]:
    n_obs_steps = int(cfg.n_obs_steps)
    n_action_steps = int(cfg.n_action_steps)

    obs, _ = env.reset(seed=int(seed))
    source_obj = getattr(env, "episode_source_obj", None)
    target_obj = getattr(env, "episode_target_obj", None)
    tcp_link = getattr(env, "tcp", None)
    if source_obj is None or target_obj is None or tcp_link is None:
        raise RuntimeError("env missing episode_source_obj / episode_target_obj / tcp")

    prev_arm_action = np.zeros(6, dtype=np.float32)
    prev_gripper_action = np.zeros(1, dtype=np.float32)
    window = ObsWindow(n_obs_steps=n_obs_steps)
    success = False
    truncated = False
    pending_actions: List[np.ndarray] = []
    tcont_log: List[float] = []
    selected_log: List[int] = []
    total_reward = 0.0

    from simpler_env.utils.env.observation_utils import (  # type: ignore
        get_image_from_maniskill2_obs_dict,
    )

    t = 0
    while t < args.max_steps:
        step_obs = build_obs_step(
            env=env, obs=obs,
            prev_arm_action=prev_arm_action,
            prev_gripper_action=prev_gripper_action,
            source_obj=source_obj, target_obj=target_obj, tcp_link=tcp_link,
        )
        window.push(step_obs)
        img = np.asarray(get_image_from_maniskill2_obs_dict(env, obs), dtype=np.uint8)

        if not pending_actions:
            obs_dict = window.to_tensor_dict(device=device)
            img_t = torch.from_numpy(img).to(device=device)
            obs_dict["agentview_image"] = img_t.unsqueeze(0).unsqueeze(0).expand(
                1, n_obs_steps, *img.shape
            )
            with torch.no_grad():
                feat = encode_tuner_obs(policy, obs_dict)
            tcont = _resolve_tcont(tuner, feat, args.tuner_mode, args.fixed_tcont)
            tcont_log.append(float(tcont[0].item()))
            actions, values = policy.predict_n_actions(
                obs_dict, policy.verifier, int(args.n_samples),
                tcont_context=tcont,
            )
            best = int(values.argmax(dim=1).item())
            selected_log.append(best)
            start = n_obs_steps - 1
            end = start + n_action_steps
            chunk = actions[0, best, start:end].detach().cpu().numpy()
            pending_actions = list(chunk.astype(np.float32, copy=False))

        action_vla = pending_actions.pop(0).astype(np.float32)
        obs, reward, done, trunc, _ = env.step(convert_maniskill(action_vla.copy()))
        prev_arm_action = action_vla[:6].copy()
        prev_gripper_action = action_vla[6:7].copy()
        total_reward += float(reward)
        if bool(done):
            success = True
            break
        if bool(trunc):
            truncated = True
            break
        t += 1

    return {
        "success": success,
        "truncated": truncated,
        "steps": int(t + (1 if (success or truncated) else 0)),
        "tcont_mean": float(np.mean(tcont_log)) if tcont_log else float("nan"),
        "tcont_std": float(np.std(tcont_log)) if tcont_log else float("nan"),
        "tcont_min": float(np.min(tcont_log)) if tcont_log else float("nan"),
        "tcont_max": float(np.max(tcont_log)) if tcont_log else float("nan"),
        "total_reward": total_reward,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Frozen noise-conditioned search policy ckpt.")
    p.add_argument("--tuner-ckpt", default=None,
                   help="Trained NoiseTunerAgent .pt; omit for --tuner-mode fixed.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--task", default="widowx_put_eggplant_in_basket")
    p.add_argument("--num-episodes", type=int, default=50)
    p.add_argument("--start-seed", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--tuner-mode", choices=["mean", "sample", "fixed"], default="mean",
                   help="mean: deterministic actor; sample: stochastic; "
                        "fixed: ignore tuner and use --fixed-tcont.")
    p.add_argument("--fixed-tcont", type=float, default=0.0,
                   help="Used when --tuner-mode=fixed.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-ema", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    policy, cfg = load_policy(
        checkpoint=args.checkpoint, device=device, use_ema=not args.no_ema,
    )
    policy.eval()
    for prm in policy.parameters():
        prm.requires_grad_(False)

    tuner = None
    if args.tuner_mode != "fixed":
        if args.tuner_ckpt is None:
            raise ValueError("--tuner-ckpt required unless --tuner-mode=fixed")
        # Probe obs_dim via a reset (same trick as train_noise_tuner).
        import simpler_env  # noqa: F401
        from simpler_env import make as make_env
        probe_env = make_env(args.task)
        obs, _ = probe_env.reset(seed=args.start_seed)
        src = getattr(probe_env, "episode_source_obj", None)
        tgt = getattr(probe_env, "episode_target_obj", None)
        tcp = getattr(probe_env, "tcp", None)
        win = ObsWindow(n_obs_steps=int(cfg.n_obs_steps))
        s = build_obs_step(
            env=probe_env, obs=obs,
            prev_arm_action=np.zeros(6, dtype=np.float32),
            prev_gripper_action=np.zeros(1, dtype=np.float32),
            source_obj=src, target_obj=tgt, tcp_link=tcp,
        )
        for _ in range(int(cfg.n_obs_steps)):
            win.push(s)
        probe_feat = encode_tuner_obs(policy, win.to_tensor_dict(device=device))
        obs_dim = int(probe_feat.shape[-1])
        del probe_env
        tuner = NoiseTunerAgent(obs_dim=obs_dim, device=device, cfg=SACConfig())
        tuner.load_state_dict(torch.load(args.tuner_ckpt, map_location=device))
        tuner.actor.eval()
        tuner.critic.eval()
        print(f"[eval_tuner] loaded tuner: {args.tuner_ckpt}")

    import simpler_env  # noqa: F401
    from simpler_env import make as make_env
    env = make_env(args.task)

    episodes: List[Dict[str, Any]] = []
    t0 = time.time()
    for i in range(args.num_episodes):
        seed = args.start_seed + i
        ep = run_episode(env, policy, tuner, args, cfg, device, seed)
        ep_record = {"ep_idx": i, "seed": seed, **ep}
        episodes.append(ep_record)
        with open(out_dir / "episodes.jsonl", "a") as fh:
            fh.write(json.dumps(ep_record) + "\n")
        sr = sum(int(e["success"]) for e in episodes) / len(episodes)
        print(
            f"[eval_tuner] ep={i:3d}/{args.num_episodes} seed={seed}  "
            f"success={int(ep['success'])}  "
            f"tcont_mean={ep['tcont_mean']:.3f}  running_sr={sr:.3f}"
        )

    summary = {
        "checkpoint": args.checkpoint,
        "tuner_ckpt": args.tuner_ckpt,
        "tuner_mode": args.tuner_mode,
        "fixed_tcont": args.fixed_tcont,
        "task": args.task,
        "num_episodes": len(episodes),
        "num_successes": sum(int(e["success"]) for e in episodes),
        "success_rate": float(np.mean([int(e["success"]) for e in episodes])),
        "mean_tcont_mean": float(np.mean([e["tcont_mean"] for e in episodes
                                          if not np.isnan(e["tcont_mean"])])),
        "total_time_s": round(time.time() - t0, 2),
    }
    with open(out_dir / "eval_log.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[eval_tuner] done. summary -> {out_dir/'eval_log.json'}")


if __name__ == "__main__":
    main()
