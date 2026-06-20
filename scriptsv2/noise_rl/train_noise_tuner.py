"""SAC fine-tuning of the obs-noise level for SearchPolicyRoboMonkeyDiffusionNoiseCond.

Loads a frozen ``SearchPolicyRoboMonkeyDiffusionNoiseCond`` checkpoint, rolls
out in SimplerEnv, and trains a small SAC actor that picks ``tcont_context``
per replan based on the current state. The base policy + verifier are never
updated — only the tuner is. Inference at deploy time becomes:

    tcont = tuner.act(obs_feat)
    chunk, _ = policy.predict_n_actions(obs_dict, policy.verifier, n,
                                        tcont_context=tcont)

Usage
-----
    bash scriptsv2/noise_rl/train_noise_tuner.sh <ckpt> [num_env_steps]

Per-chunk SMDP: one transition per ``predict_n_actions`` call. Reward is the
sum of env rewards across the executed action steps. Episodes terminate on
``done`` or ``trunc``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# eval_diffusion / eval_search helpers (obs window, action conversion, policy loader).
SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_ROOT / "eval_diffusion"))

from eval_diffusion import (  # type: ignore
    ObsWindow,
    build_obs_step,
    convert_maniskill,
    load_policy,
)

from noise_tuner import (  # noqa: E402
    NoiseTunerAgent,
    ReplayBuffer,
    SACConfig,
)


# ---------------------------------------------------------------------------
#  Obs feature extraction — uses the frozen policy's normalizer + encoder so
#  the tuner sees exactly the same state representation the policy was trained
#  on, just without the noise injection.
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_tuner_obs(policy, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Clean (un-noised) flattened state features used as tuner obs.

    Mirrors ``SearchPolicyRoboMonkeyDiffusionNoiseCond.encode_obs_cond`` but
    skips the corruption + sinusoidal-injection steps so the tuner can
    *decide* the noise level from a clean view of the state.
    """
    from diffusion_policy.common.pytorch_util import dict_apply

    nobs = policy.normalizer.normalize(obs_dict)
    obs_value = next(iter(nobs.values()))
    B = obs_value.shape[0]
    To = int(policy.n_obs_steps)
    this_nobs = dict_apply(
        {k: nobs[k] for k in policy._state_keys},
        lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]),
    )
    feat = policy.obs_encoder(this_nobs)        # (B*To, D)
    return feat.reshape(B, -1)                   # (B, To*D)


# ---------------------------------------------------------------------------
#  Single-env training rollout.
# ---------------------------------------------------------------------------

def run_one_episode(
        env,
        policy,
        tuner: NoiseTunerAgent,
        buffer: ReplayBuffer,
        cfg,
        device: torch.device,
        max_steps: int,
        n_samples: int,
        seed: int,
        deterministic_tuner: bool,
    ) -> Dict[str, Any]:
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

    # SMDP transition bookkeeping.
    pending_obs_feat: Optional[np.ndarray] = None
    pending_tcont: Optional[float] = None
    chunk_reward = 0.0
    transitions = 0
    tcont_log: List[float] = []
    total_reward = 0.0

    from simpler_env.utils.env.observation_utils import (  # type: ignore
        get_image_from_maniskill2_obs_dict,
    )

    t = 0
    while t < max_steps:
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

            # 1. Encode obs for the tuner (clean view).
            cur_feat = encode_tuner_obs(policy, obs_dict)  # (1, obs_dim)

            # 2. If we already executed a previous chunk, store its transition
            #    (s_t, a_t, r_t -> s_{t+1}) with done=False — this is "chunk end".
            if pending_obs_feat is not None:
                next_feat_np = cur_feat[0].detach().cpu().numpy()
                buffer.push(pending_obs_feat, pending_tcont, chunk_reward,
                            next_feat_np, 0.0)
                transitions += 1

            # 3. Pick tcont via tuner.
            tcont_t = tuner.select_action(cur_feat, deterministic=deterministic_tuner)
            tcont_f = float(tcont_t.detach().cpu().item())
            tcont_log.append(tcont_f)

            # 4. Search with that tcont; argmax candidates by verifier value.
            actions, values = policy.predict_n_actions(
                obs_dict, policy.verifier, int(n_samples),
                tcont_context=tcont_t.view(-1),
            )  # (1, K, horizon, action_dim), (1, K)
            best = int(values.argmax(dim=1).item())
            start = n_obs_steps - 1
            end = start + n_action_steps
            chunk = actions[0, best, start:end].detach().cpu().numpy()
            assert chunk.shape == (n_action_steps, 7)
            pending_actions = list(chunk.astype(np.float32, copy=False))

            pending_obs_feat = cur_feat[0].detach().cpu().numpy()
            pending_tcont = tcont_f
            chunk_reward = 0.0

        # Step env.
        action_vla = pending_actions.pop(0).astype(np.float32)
        env_action = convert_maniskill(action_vla.copy())
        obs, reward, done, trunc, _info = env.step(env_action)
        prev_arm_action = action_vla[:6].copy()
        prev_gripper_action = action_vla[6:7].copy()
        chunk_reward += float(reward)
        total_reward += float(reward)

        if bool(done):
            success = True
            break
        if bool(trunc):
            truncated = True
            break
        t += 1

    # Flush the trailing transition (last chunk → terminal).
    if pending_obs_feat is not None:
        # next state = current encoded state at episode end.
        step_obs = build_obs_step(
            env=env, obs=obs,
            prev_arm_action=prev_arm_action,
            prev_gripper_action=prev_gripper_action,
            source_obj=source_obj, target_obj=target_obj, tcp_link=tcp_link,
        )
        window.push(step_obs)
        obs_dict = window.to_tensor_dict(device=device)
        img2 = np.asarray(get_image_from_maniskill2_obs_dict(env, obs), dtype=np.uint8)
        img_t = torch.from_numpy(img2).to(device=device)
        obs_dict["agentview_image"] = img_t.unsqueeze(0).unsqueeze(0).expand(
            1, n_obs_steps, *img2.shape
        )
        next_feat = encode_tuner_obs(policy, obs_dict)[0].detach().cpu().numpy()
        done_flag = 1.0 if (success or truncated) else 0.0
        buffer.push(pending_obs_feat, pending_tcont, chunk_reward, next_feat, done_flag)
        transitions += 1

    return {
        "success": success,
        "truncated": truncated,
        "steps": int(t + (1 if (success or truncated) else 0)),
        "transitions": transitions,
        "tcont_mean": float(np.mean(tcont_log)) if tcont_log else float("nan"),
        "tcont_std": float(np.std(tcont_log)) if tcont_log else float("nan"),
        "total_reward": total_reward,
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Frozen SearchPolicyRoboMonkeyDiffusionNoiseCond .ckpt")
    p.add_argument("--output-dir", required=True,
                   help="Where tuner checkpoints + training log go.")
    p.add_argument("--task", default="widowx_put_eggplant_in_basket")
    p.add_argument("--num-env-steps", type=int, default=20000,
                   help="Total env interactions (across episodes).")
    p.add_argument("--max-episode-steps", type=int, default=120)
    p.add_argument("--n-samples", type=int, default=16,
                   help="K passed to predict_n_actions per chunk.")
    p.add_argument("--start-seed", type=int, default=2000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-ema", action="store_true")

    # SAC knobs.
    p.add_argument("--buffer-size", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--warmup-transitions", type=int, default=200,
                   help="Skip SAC updates until the buffer has this many.")
    p.add_argument("--updates-per-chunk", type=int, default=1)
    p.add_argument("--init-alpha", type=float, default=0.1)
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--alpha-lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)

    p.add_argument("--log-every-episodes", type=int, default=1)
    p.add_argument("--ckpt-every-episodes", type=int, default=10)

    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "tuner_log.jsonl"

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[tuner] CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")

    print(f"[tuner] Loading frozen policy: {args.checkpoint}")
    policy, cfg = load_policy(
        checkpoint=args.checkpoint, device=device, use_ema=not args.no_ema,
    )
    policy.eval()
    for prm in policy.parameters():
        prm.requires_grad_(False)

    cls = policy.__class__.__name__
    if "NoiseCond" not in cls:
        print(
            f"[tuner] WARNING: policy class is {cls} — the tuner only makes "
            "sense for a noise-CONDITIONED variant (SearchPolicyRoboMonkey"
            "DiffusionNoiseCond). Continuing anyway; tcont_context will be "
            "treated as a fixed-corruption knob."
        )

    # Lazy import of SimplerEnv so TF/sapien only load after policy is on GPU.
    import simpler_env  # noqa: F401
    from simpler_env import make as make_env

    env = make_env(args.task)
    print(f"[tuner] task = {args.task}  instruction = {env.get_language_instruction()!r}")

    # Probe obs_dim with a dummy reset + window.
    obs, _ = env.reset(seed=int(args.start_seed))
    source_obj = getattr(env, "episode_source_obj", None)
    target_obj = getattr(env, "episode_target_obj", None)
    tcp_link = getattr(env, "tcp", None)
    if source_obj is None or target_obj is None or tcp_link is None:
        raise RuntimeError(
            "env missing episode_source_obj / episode_target_obj / tcp — cannot infer obs_dim"
        )
    probe_window = ObsWindow(n_obs_steps=int(cfg.n_obs_steps))
    step_obs = build_obs_step(
        env=env, obs=obs,
        prev_arm_action=np.zeros(6, dtype=np.float32),
        prev_gripper_action=np.zeros(1, dtype=np.float32),
        source_obj=source_obj, target_obj=target_obj, tcp_link=tcp_link,
    )
    for _ in range(int(cfg.n_obs_steps)):
        probe_window.push(step_obs)
    probe_obs_dict = probe_window.to_tensor_dict(device=device)
    # Image isn't read by encode_tuner_obs (only low-dim keys), so omit safely.
    obs_feat = encode_tuner_obs(policy, probe_obs_dict)
    obs_dim = int(obs_feat.shape[-1])
    print(f"[tuner] obs_dim = {obs_dim}")

    sac_cfg = SACConfig(
        gamma=args.gamma, tau=args.tau,
        init_alpha=args.init_alpha,
        actor_lr=args.actor_lr, critic_lr=args.critic_lr, alpha_lr=args.alpha_lr,
    )
    agent = NoiseTunerAgent(obs_dim=obs_dim, device=device, cfg=sac_cfg)
    buffer = ReplayBuffer(args.buffer_size, obs_dim=obs_dim, device=device)

    total_env_steps = 0
    episode_idx = 0
    t0 = time.time()

    while total_env_steps < args.num_env_steps:
        seed = args.start_seed + episode_idx
        # During the warmup phase, the actor is essentially random; once we
        # have enough transitions, sample stochastically from the trained policy.
        deterministic = False
        ep = run_one_episode(
            env=env, policy=policy, tuner=agent, buffer=buffer, cfg=cfg,
            device=device, max_steps=args.max_episode_steps,
            n_samples=args.n_samples, seed=seed,
            deterministic_tuner=deterministic,
        )
        total_env_steps += ep["steps"]

        # SAC updates: one per stored chunk transition this episode (scaled).
        metrics = None
        if len(buffer) >= args.warmup_transitions:
            for _ in range(ep["transitions"] * args.updates_per_chunk):
                batch = buffer.sample(args.batch_size)
                metrics = agent.update(batch)

        if episode_idx % args.log_every_episodes == 0:
            log_row = {
                "episode": episode_idx,
                "seed": seed,
                "env_steps": total_env_steps,
                "success": bool(ep["success"]),
                "truncated": bool(ep["truncated"]),
                "ep_steps": ep["steps"],
                "ep_transitions": ep["transitions"],
                "ep_tcont_mean": ep["tcont_mean"],
                "ep_tcont_std": ep["tcont_std"],
                "ep_total_reward": ep["total_reward"],
                "buffer_size": len(buffer),
                "elapsed_s": round(time.time() - t0, 2),
            }
            if metrics is not None:
                log_row.update({
                    "critic_loss": metrics.critic_loss,
                    "actor_loss": metrics.actor_loss,
                    "alpha_loss": metrics.alpha_loss,
                    "alpha": metrics.alpha,
                    "q_mean": metrics.q_mean,
                    "target_q_mean": metrics.target_q_mean,
                })
            with open(log_path, "a") as fh:
                fh.write(json.dumps(log_row) + "\n")
            print(
                f"[tuner] ep={episode_idx:5d}  env_steps={total_env_steps:6d}  "
                f"success={int(ep['success'])}  "
                f"tcont_mean={ep['tcont_mean']:.3f}  "
                f"buf={len(buffer):5d}  "
                f"loss_c={getattr(metrics, 'critic_loss', float('nan')):.3f}  "
                f"loss_a={getattr(metrics, 'actor_loss', float('nan')):.3f}  "
                f"alpha={getattr(metrics, 'alpha', float('nan')):.3f}"
            )

        if (episode_idx + 1) % args.ckpt_every_episodes == 0:
            ckpt_path = out_dir / f"tuner_ep{episode_idx+1:05d}.pt"
            torch.save(agent.state_dict(), ckpt_path)
            torch.save(agent.state_dict(), out_dir / "tuner_latest.pt")
            print(f"[tuner] saved {ckpt_path.name}")

        episode_idx += 1

    torch.save(agent.state_dict(), out_dir / "tuner_latest.pt")
    print(f"[tuner] done — {episode_idx} episodes, {total_env_steps} env steps; "
          f"tuner -> {out_dir / 'tuner_latest.pt'}")


if __name__ == "__main__":
    main()
