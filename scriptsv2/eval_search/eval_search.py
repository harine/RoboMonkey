"""SimplerEnv eval for ``SearchPolicyRoboMonkey`` (Gaussian + Diffusion
variants of the state-based search policy).

The search policy has its own in-process verifier (built into the policy at
hydra-instantiate time). ``predict_n_actions`` runs N autoregressive
candidate samples, each scored by the verifier. Modes:

  * ``argmax``  : pick the candidate with the highest verifier value.
  * ``softmax`` : sample an index from ``softmax(values / T)``.
  * ``refine``  : final transformer forward pass conditioned on the full
                  (action, value) context, read the last token's
                  ``dist.mean`` (Gaussian policy only).

Reuses obs/action plumbing from ``scriptsv2.eval_diffusion.eval_diffusion``.

Usage:
    python eval_search.py \\
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
    n_samples: Optional[int] = None,
    softmax_temp: float = 1.0,
    tcont: Optional[float] = None,
    replan_seed: Optional[int] = None,
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
    if n_samples is None:
        n_samples = int(policy.max_actions) if mode == "argmax" else int(policy.max_actions) - 1
    else:
        n_samples = int(n_samples)
        if mode == "refine":
            # refine mode reads dist.mean of the last transformer token; it
            # needs at most (max_actions - 1) prior samples in the context.
            n_samples = min(n_samples, int(policy.max_actions) - 1)

    # predict_n_actions handles n_samples > max_actions by sliding a fixed
    # (max_actions - 1) window of context over the autoregressive loop,
    # which matches training-time context length while still returning all
    # n_samples candidates for argmax selection.
    #
    # Verifier speedup (same as training's _run_search): when ROBOMONKEY_KV_PREFIX=1
    # and the in-process verifier supports it, route all of this replan's N
    # candidate scorings through the cached image-prefix KV. Every get_value
    # here scores the SAME single frame (B=1), so the image+instruction prefix
    # KV is computed once and reused across the N rounds. Clear before (fresh
    # frame) and after (bound GPU memory). Bit-equivalent to the plain path.
    import os as _os
    _verifier = policy.verifier
    _kv = (_os.environ.get("ROBOMONKEY_KV_PREFIX", "0") == "1"
           and getattr(_verifier, "supports_kv_prefix", False))
    if _kv:
        _verifier.set_kv_prefix(True)
        _verifier.clear_prefix_cache()
    # tcont (noise level in [0,1]) only applies to the noise-conditioned (tmrl)
    # policy, whose predict_n_actions takes a tcont_context kwarg. The
    # tunable_corruption policies don't accept it, so pass only when supported.
    import inspect as _inspect
    _pred_kw: Dict[str, Any] = {}
    if tcont is not None and "tcont_context" in _inspect.signature(
        policy.predict_n_actions
    ).parameters:
        _pred_kw["tcont_context"] = float(tcont)
    # Deterministic candidate sampling: seed the global RNG from a per-replan
    # seed so the N diffusion candidate draws are reproducible run-to-run.
    # Applied here, at the harness level, it governs every policy variant
    # (tunable_corruption / noise-conditioned / BC) since they all draw their
    # init noise from the default generator. The candidates stay diverse (N
    # sequential draws from the seeded stream) and nested across N. The softmax
    # selection below is intentionally left stochastic — see the restore call.
    if replan_seed is not None:
        torch.manual_seed(int(replan_seed))
    try:
        actions, values = policy.predict_n_actions(
            obs_dict, _verifier, n_samples, **_pred_kw
        )  # (B, K, horizon, action_dim) ; (B, K)
    finally:
        if _kv:
            _verifier.set_kv_prefix(False)
            _verifier.clear_prefix_cache()
    B = actions.shape[0]
    start = To - 1
    end = start + n_action_steps

    # Chunk-scoring selection (parity with the BC BoN fix): re-score every
    # candidate by the verifier over ALL executed actions (steps [start:end])
    # and aggregate, then select on that. The autoregressive search policy
    # already generated its candidates using its own per-candidate values; this
    # only changes the SELECTION criterion so it matches the BC chunk scoring
    # (where the verifier collapses on a single far-horizon step). Enabled with
    # EVAL_CHUNK_SELECT=1. BC (BCSearchWrapper) already returns chunk values, so
    # it leaves this off. Rollout is B=1.
    if _os.environ.get("EVAL_CHUNK_SELECT", "0") == "1":
        _agg = _os.environ.get("BC_VERIFIER_AGG", "mean").lower()
        _imgk = getattr(_verifier, "image_obs_key", None)
        _img = obs_dict[_imgk]                       # (1, To, ...)
        _execs = actions[:, :, start:end, :]         # (B, N, S, 7)
        _N, _S = _execs.shape[1], _execs.shape[2]
        assert B == 1, "EVAL_CHUNK_SELECT assumes rollout batch B=1"
        _M = _N * _S
        _flat = _execs.reshape(_M, 1, 7)             # get_value reads [:, -1]
        _bimg = _img.expand(_M, *_img.shape[1:])
        _fv = _verifier.get_value({_imgk: _bimg}, _flat)  # (M,)
        _ps = _fv.reshape(_N, _S)
        if _agg == "sum":
            _cv = _ps.sum(dim=1)
        elif _agg == "min":
            _cv = _ps.min(dim=1).values
        elif _agg == "last":
            _cv = _ps[:, -1]
        elif _agg == "first":
            _cv = _ps[:, 0]
        else:
            _cv = _ps.mean(dim=1)
        values = _cv.unsqueeze(0)                    # (1, N) chunk values

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

    if mode == "softmax":
        # Restore entropy-seeded RNG so the multinomial selection below stays
        # stochastic even though candidate sampling above was seeded. (Only the
        # candidate sampling is made deterministic — selection is part of the
        # method and intentionally keeps its run-to-run randomness.)
        if replan_seed is not None:
            torch.seed()
        # Sample a candidate index from softmax(values / temp). temp -> 0 is
        # argmax, temp -> inf is uniform. Default temp=1.0.
        T = max(float(softmax_temp), 1e-6)
        probs = torch.softmax(values / T, dim=1)  # (B, K)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(1)  # (B,)
        chosen_action = actions[torch.arange(B, device=actions.device), sampled]
        out["chunk"] = chosen_action[:, start:end]
        out["selected_index"] = int(sampled[0].item())
        out["selected_value"] = float(values[0, int(sampled[0].item())].item())
        out["softmax_temp"] = float(T)
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
                    viz_q: bool = False,
                    n_samples: Optional[int] = None,
                    softmax_temp: float = 1.0,
                    tcont: Optional[float] = None,
                    deterministic_sampling: bool = True,
                    sample_seed_base: int = 0) -> Dict[str, Any]:
    n_obs_steps = int(cfg.n_obs_steps)
    n_action_steps = int(cfg.n_action_steps)

    obs, _ = env.reset(seed=int(seed))
    # Step/reset go through the wrapper, but the SimplerEnv-specific attributes
    # live on the unwrapped base env (mirrors eval_diffusion.py).
    env_base = getattr(env, "unwrapped", env)
    source_obj = getattr(env_base, "episode_source_obj", None)
    target_obj = getattr(env_base, "episode_target_obj", None)
    tcp_link = getattr(env_base, "tcp", None)
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

    # Camera params (intrinsic_cv (3,3) + extrinsic_cv (4,4) world->cam) for
    # projecting candidate action trajectories into the rendered frame later.
    # SimplerEnv widowx tasks use `3rd_view_camera`; fall back to whatever's
    # available if that key is missing.
    # base_pose: (7,) [px, py, pz, qw, qx, qy, qz] — the robot's base pose in
    # world frame. Action xyz deltas are in BASE frame (e.g. widowx is yawed
    # 180° relative to world), so the renderer needs R_world_base to project
    # them correctly.
    cam_K: Optional[np.ndarray] = None
    cam_T_wc: Optional[np.ndarray] = None
    base_pose_world: Optional[np.ndarray] = None
    if viz_q:
        try:
            cam_params = obs.get("camera_param", {})
            cam_name = "3rd_view_camera" if "3rd_view_camera" in cam_params \
                else next(iter(cam_params), None)
            if cam_name is not None:
                cam_K = np.asarray(cam_params[cam_name]["intrinsic_cv"], dtype=np.float32)
                cam_T_wc = np.asarray(cam_params[cam_name]["extrinsic_cv"], dtype=np.float32)
        except Exception as e:
            print(f"[eval] could not read camera_param for projection: {e!r}")
        try:
            bp = obs.get("agent", {}).get("base_pose", None)
            if bp is not None:
                base_pose_world = np.asarray(bp, dtype=np.float32)
        except Exception as e:
            print(f"[eval] could not read agent.base_pose: {e!r}")

    # Always need a frame for the in-process RoboMonkey verifier; capturing
    # for video output is a strict subset.
    from simpler_env.utils.env.observation_utils import (  # type: ignore
        get_image_from_maniskill2_obs_dict,
    )

    while t < max_steps:
        step_obs = build_obs_step(
            env=env_base, obs=obs,
            prev_arm_action=prev_arm_action,
            prev_gripper_action=prev_gripper_action,
            source_obj=source_obj, target_obj=target_obj, tcp_link=tcp_link,
        )
        window.push(step_obs)

        current_frame: Optional[np.ndarray] = None
        try:
            img = get_image_from_maniskill2_obs_dict(env_base, obs)
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
            # Per-replan deterministic seed = f(episode seed, t). Unique per
            # (episode, replan step) and stable across runs, so the candidate
            # set at each replan is reproducible. None -> stochastic (legacy).
            replan_seed = (
                (int(seed) * 1_000_003 + int(t) + int(sample_seed_base))
                % (2 ** 31 - 1)
                if deterministic_sampling else None
            )
            pred = _search_policy_predict(
                policy, obs_dict, mode=mode, viz_q=viz_q,
                n_samples=n_samples, softmax_temp=softmax_temp,
                tcont=tcont, replan_seed=replan_seed,
            )
            chunk = pred["chunk"][0].detach().cpu().numpy()
            if branches is not None and current_frame is not None:
                # Current EE world-frame xyz (start of every candidate's traj).
                try:
                    tcp_world = np.asarray(obs["extra"]["tcp_pose"], dtype=np.float32)[:3]
                except Exception:
                    tcp_world = np.zeros(3, dtype=np.float32)
                branches.append({
                    "t": int(t),
                    "candidate_actions": pred["candidate_actions"].astype(
                        np.float32, copy=True
                    ),
                    "values": np.asarray(pred["values"], dtype=np.float32),
                    "selected_index": int(pred.get("selected_index", -1)),
                    "selected_value": float(pred.get("selected_value", float("nan"))),
                    "frame": current_frame.astype(np.uint8, copy=True),
                    "ee_xyz": tcp_world,
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
        "cam_K": cam_K,
        "cam_T_wc": cam_T_wc,
        "base_pose_world": base_pose_world,
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
    parser.add_argument("--mode", choices=["argmax", "softmax", "refine"], default="argmax",
                        help="argmax: pick the highest-verifier-value sample. "
                             "softmax: sample index ~ softmax(values / temp), "
                             "tunable via --softmax-temp. "
                             "refine: final transformer forward pass over the "
                             "search trace, take dist.mean of last token "
                             "(Gaussian policy only).")
    parser.add_argument("--softmax-temp", type=float, default=1.0,
                        help="Temperature for --mode softmax. Lower -> more "
                             "argmax-like; higher -> more uniform. Default 1.0.")
    parser.add_argument("--viz-q", action="store_true",
                        help="Save per-replan sampled candidate actions + "
                             "verifier values + frame to "
                             "<output_dir>/search_q/ep<idx>_seed<seed>.npz "
                             "for later visualization (mirrors BoN viz_q).")
    parser.add_argument("--viz-q-episodes", type=int, default=0,
                        help="When --viz-q is set, only collect/save the "
                             "render npz for the first K episodes (the same "
                             "seeds across every N cell, since episode i uses "
                             "start_seed+i). 0 = all episodes (legacy). Set to "
                             "e.g. 10 to keep full success-rate stats over all "
                             "--num-episodes while saving render data for just "
                             "the first 10 fixed-seed episodes.")
    parser.add_argument("--tcont", type=float, default=None,
                        help="Noise level in [0,1] for the noise-conditioned "
                             "(tmrl) policy. Maps to DDPM timestep "
                             "round(tcont*(T-1)). 0.0 = clean obs. Ignored by "
                             "the tunable_corruption policies, which don't take "
                             "a tcont input. Default: policy's eval default "
                             "(0.0 / clean for the noise-conditioned policy).")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Override the number of search samples per "
                             "replan. Default: policy.max_actions (argmax) "
                             "or policy.max_actions-1 (refine). For values "
                             "> max_actions, predict_n_actions slides a "
                             "fixed (max_actions-1) context window.")
    parser.add_argument("--stochastic-sampling", action="store_true",
                        help="Legacy behavior: do NOT seed the diffusion "
                             "candidate sampling, so the N candidates vary "
                             "run-to-run. Default (off): candidate sampling "
                             "is seeded per-replan from f(episode_seed, t) for "
                             "reproducible candidates. (Softmax selection stays "
                             "stochastic in either case.)")
    parser.add_argument("--sample-seed-base", type=int, default=0,
                        help="Offset added into the per-replan candidate-"
                             "sampling seed. Change to draw a different "
                             "deterministic realization of the candidate sets. "
                             "Ignored under --stochastic-sampling.")
    parser.add_argument("--score-window", choices=["last", "executed"],
                        default="last",
                        help="Which action(s) the verifier scores per "
                             "candidate. 'last': the single configured "
                             "action_chunk_index (default, =-1, last horizon "
                             "step). 'executed': score EACH executed step "
                             "(start..start+n_action_steps) against the current "
                             "frame and aggregate via --score-agg, so the value "
                             "reflects the whole executed chunk. NOTE: scores "
                             "n_action_steps x more verifier forwards.")
    parser.add_argument("--score-agg", choices=["mean", "sum", "min"],
                        default="mean",
                        help="Aggregation over the executed-window step scores "
                             "when --score-window executed. Default mean.")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[eval] CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")

    policy, cfg = load_policy(
        checkpoint=args.checkpoint, device=device, use_ema=not args.no_ema,
    )
    # BC checkpoint (e.g. DiffusionUnetImagePolicy) has no predict_n_actions or
    # verifier. Wrap it with BCSearchWrapper so the rest of the eval harness works
    # unchanged: N independent BC samples are scored by the in-process verifier.
    if not hasattr(policy, "predict_n_actions"):
        import os as _os
        from diffusion_policy.policy.verifiers import RoboMonkeyVerifier
        from diffusion_policy.policy.search_policy_robomonkey import BCSearchWrapper
        # BC sample diversity: the trained DDIM-8 sampler is near-deterministic,
        # so the N candidates collapse and the verifier can't rank them (BoN
        # degenerates). BC_SAMPLER=ddpm swaps in full ancestral DDPM sampling at
        # inference (same epsilon model + betas), which injects per-step noise
        # and yields genuinely diverse candidates. BC_SAMPLER=ddim (default)
        # keeps the trained deterministic sampler (the control run).
        _sampler = _os.environ.get("BC_SAMPLER", "ddim").lower()
        if _sampler == "ddpm":
            from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
            _old = policy.noise_scheduler
            policy.noise_scheduler = DDPMScheduler(
                num_train_timesteps=_old.config.num_train_timesteps,
                beta_start=_old.config.beta_start,
                beta_end=_old.config.beta_end,
                beta_schedule=_old.config.beta_schedule,
                variance_type="fixed_small",
                clip_sample=_old.config.clip_sample,
                prediction_type=_old.config.prediction_type,
            )
            policy.num_inference_steps = int(
                _os.environ.get("BC_SAMPLER_STEPS",
                                str(_old.config.num_train_timesteps)))
            print(f"[eval] BC_SAMPLER=ddpm -> ancestral DDPM sampling "
                  f"(num_inference_steps={policy.num_inference_steps}, "
                  f"stochastic candidates)")
        else:
            print(f"[eval] BC_SAMPLER=ddim -> trained deterministic sampler "
                  f"(num_inference_steps={policy.num_inference_steps})")
        _instr = _os.environ.get("VERIFIER_INSTRUCTION", "put the eggplant in the basket")
        _img_key = _os.environ.get("VERIFIER_IMAGE_KEY", "agentview_image")
        _verifier = RoboMonkeyVerifier(
            server_url="in_process",
            instruction=_instr,
            image_obs_key=_img_key,
            image_save_dir="/tmp/robomonkey_bc_obs",
            action_chunk_index=-1,
            request_timeout=300,
            max_workers=8,
            keep_jpegs=False,
            health_check=True,
        )
        _max_actions = max(128, int(_os.environ.get("N_SAMPLES", "128")))
        policy = BCSearchWrapper(policy, _verifier, max_actions=_max_actions)
        print(f"[eval] BC mode: wrapped with in-process verifier "
              f"(instruction={_instr!r}, max_actions={_max_actions})")
    print(f"[eval] max_actions={int(policy.max_actions)}  (in-process verifier)")

    # Configure what the verifier scores. 'executed': score each of the
    # n_action_steps actions that actually get executed (the chunk window
    # start..start+n_action_steps) against the current frame, aggregated via
    # --score-agg, so the candidate value = quality of the whole executed
    # chunk. 'last': leave the verifier's single action_chunk_index (default).
    _score_indices = None
    if args.score_window == "executed":
        _To = int(policy.n_obs_steps)
        _nas = int(policy.n_action_steps)
        _start = _To - 1
        _score_indices = list(range(_start, _start + _nas))
        policy.verifier.score_indices = _score_indices
        policy.verifier.score_agg = str(args.score_agg)
        print(f"[eval] verifier scoring = EXECUTED window steps "
              f"{_score_indices[0]}..{_score_indices[-1]} "
              f"(agg={args.score_agg}, {_nas}x verifier forwards/candidate)")
    else:
        # Explicitly clear so a stale attr can't leak windowed scoring.
        policy.verifier.score_indices = None
        print(f"[eval] verifier scoring = single step "
              f"action_chunk_index={getattr(policy.verifier, 'action_chunk_index', -1)}")

    import simpler_env  # noqa: F401
    from simpler_env import make as make_env
    print(f"[eval] Creating SimplerEnv task: {args.task}")
    env = make_env(args.task)
    # make_env returns a gym TimeLimit wrapper; the SimplerEnv API lives on the
    # unwrapped base env (mirrors eval_diffusion.py).
    _base_env = getattr(env, "unwrapped", env)
    instr = (
        _base_env.get_language_instruction()
        if hasattr(_base_env, "get_language_instruction") else None
    ) or args.task
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
        # Cap render-data collection to the first K episodes (same fixed seeds
        # across every N cell). Keeps SR over all episodes but avoids dumping
        # frames/candidate actions for all of them. 0 = all (legacy).
        viz_q_lim = int(args.viz_q_episodes)
        ep_viz_q = viz_q and (viz_q_lim <= 0 or i < viz_q_lim)
        ep_t0 = time.time()
        try:
            ep = rollout_episode(
                env=env, policy=policy, cfg=cfg, device=device,
                seed=ep_seed, max_steps=int(args.max_steps),
                instruction=str(instr), capture_frames=capture,
                mode=str(args.mode), viz_q=ep_viz_q,
                n_samples=args.n_samples,
                softmax_temp=float(args.softmax_temp),
                tcont=args.tcont,
                deterministic_sampling=(not args.stochastic_sampling),
                sample_seed_base=int(args.sample_seed_base),
            )
        except Exception as e:
            print(f"[eval] episode {i} (seed={ep_seed}) raised: {e!r}")
            import traceback as _tb
            _tb.print_exc()
            ep = {"success": False, "truncated": False, "num_steps": 0,
                  "error": repr(e), "steps": [], "frames": [], "branches": None,
                  "cam_K": None, "cam_T_wc": None, "base_pose_world": None}

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
            extra_proj = {}
            if ep.get("cam_K") is not None and ep.get("cam_T_wc") is not None:
                extra_proj["cam_K"] = ep["cam_K"]
                extra_proj["cam_T_wc"] = ep["cam_T_wc"]
            if ep.get("base_pose_world") is not None:
                extra_proj["base_pose_world"] = ep["base_pose_world"]
            np.savez_compressed(
                npz_path,
                seed=np.int32(ep_seed),
                ep_idx=np.int32(i),
                success=np.int32(int(ep["success"])),
                truncated=np.int32(int(ep["truncated"])),
                num_steps=np.int32(int(ep["num_steps"])),
                ee_xyz=np.stack([b.get("ee_xyz", np.zeros(3, dtype=np.float32))
                                 for b in br], axis=0),
                **extra_proj,
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
        "tcont": (float(args.tcont) if args.tcont is not None else None),
        "deterministic_sampling": (not args.stochastic_sampling),
        "sample_seed_base": int(args.sample_seed_base),
        "score_window": str(args.score_window),
        "score_indices": _score_indices,
        "score_agg": str(args.score_agg),
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
