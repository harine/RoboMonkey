"""SimplerEnv eval for ``SearchPolicyRoboMonkey`` (Gaussian + Diffusion
variants of the state-based search policy).

The search policy has its own in-process verifier (built into the policy at
hydra-instantiate time). ``predict_n_actions`` runs N autoregressive
candidate samples, each scored by the verifier. Selection is over ALL N
candidates (N may exceed ``policy.max_actions``). Modes:

  * ``argmax``  : pick the candidate with the highest verifier value.
  * ``softmax`` : sample an index from ``softmax(values / T)``.

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

    Runs the search loop (``n_samples`` autoregressive candidate samples, each
    verifier-scored), then selects ONE of those ``n_samples`` candidates. Two
    selection modes:
      * ``"argmax"``: pick the candidate with the highest verifier value.
      * ``"softmax"``: sample an index from ``softmax(values / softmax_temp)``.

    ``n_samples`` is independent of ``policy.max_actions`` (the transformer's
    candidate token budget) and may exceed it — see the note below.

    Returns ``{"chunk": (1, n_action_steps, action_dim) tensor,
                "selected_index": int,
                "selected_value": float,
                "values": list[float] (verifier values for all n_samples candidates)}``.
    """
    # Drop low-dim obs keys the trained normalizer doesn't know about, but
    # KEEP the image key — the verifier reads it directly.
    known = set(policy.normalizer.params_dict.keys())
    image_obs_key = getattr(policy.verifier, "image_obs_key", None)
    obs_dict = {
        k: v for k, v in obs_dict.items()
        if k in known or k == image_obs_key
    }

    To = int(policy.n_obs_steps)
    n_action_steps = int(policy.n_action_steps)
    # n_samples is NOT clamped: N > max_actions is supported by design.
    n_samples = int(policy.max_actions) if n_samples is None else int(n_samples)

    # predict_n_actions handles n_samples > max_actions by sliding a fixed
    # (max_actions - 1) window of context over the autoregressive loop,
    # which matches training-time context length while still returning all
    # n_samples candidates for selection.
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
    # replan_seed may be a scalar (seed the global RNG; env(s) share the stream and
    # get row-position slices) OR a per-env list (tiled path with TILED_SEED_PER_ENV):
    # in the latter case build per-env init noise anchored to each episode's OWN seed
    # and inject it, so a candidate's noise no longer depends on its batch position or
    # on other envs' done-timing. DDIM is deterministic, so the init noise fully
    # determines the candidate -> this matches the single-env path per episode.
    _per_env_seeds = None
    if replan_seed is not None and isinstance(replan_seed, (list, tuple)):
        _per_env_seeds = [int(s) for s in replan_seed]
    _supports_noise = "noise_seq" in _inspect.signature(policy.predict_n_actions).parameters
    if _per_env_seeds is not None and _supports_noise:
        _dev = next(policy.parameters()).device
        _H, _D = int(policy.horizon), int(policy.action_dim)
        _noise_seq = torch.empty(n_samples, len(_per_env_seeds), _H, _D,
                                 device=_dev, dtype=torch.float32)
        for _k, _s in enumerate(_per_env_seeds):
            _g = torch.Generator(device=_dev)
            _g.manual_seed(int(_s) % (2 ** 63 - 1))
            for _j in range(n_samples):
                _noise_seq[_j, _k] = torch.randn(_H, _D, generator=_g,
                                                 device=_dev, dtype=torch.float32)
        _pred_kw["noise_seq"] = _noise_seq
    elif _per_env_seeds is not None:
        # policy without noise_seq support: fall back to a shared scalar seed.
        torch.manual_seed(int(_per_env_seeds[0]))
    elif replan_seed is not None:
        torch.manual_seed(int(replan_seed))
    # Stratified-random candidate augmentation (RAND_AUG_K>0): append a pool of
    # deliberately diverse actions covering the whole translation-direction
    # sphere, scored by the SAME verifier, so the selection below ranks the
    # union. Done inside the KV-prefix block so the cached image prefix is reused;
    # done BEFORE the EVAL_CHUNK_SELECT re-score so both pools share one scoring
    # path. Inert (bit-identical to before) when RAND_AUG_K is unset.
    from random_action_aug import augment as _rand_augment, config_from_env as _rand_cfg
    _aug = _rand_cfg()
    _n_policy = 0
    _is_random = None
    try:
        actions, values = policy.predict_n_actions(
            obs_dict, _verifier, n_samples, **_pred_kw
        )  # (B, K, horizon, action_dim) ; (B, K)
        if _aug.enabled:
            _n_policy = 0 if _aug.only else int(actions.shape[1])
            actions, values, _is_random = _rand_augment(
                policy, obs_dict, actions, values, _aug,
                seeds=(_per_env_seeds if _per_env_seeds is not None else replan_seed),
                verifier=_verifier,
            )
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
    if _is_random is not None:
        out["n_policy"] = int(_n_policy)
        out["n_random"] = int(actions.shape[1]) - int(_n_policy)

    if viz_q:
        # All K candidate chunks, sliced to the executed action window so they
        # are directly comparable to the chosen chunk (mirrors BoN viz_q).
        out["candidate_actions"] = (
            actions[0, :, start:end].detach().cpu().float().numpy()
        )  # (K, n_action_steps, action_dim)
        if _is_random is not None:
            out["candidate_is_random"] = _is_random[0].detach().cpu().numpy()

    def _mark_random(o, idx):
        """Did an augmented (random) candidate win this replan? (per batch row)"""
        if _is_random is not None:
            won = _is_random.gather(1, idx.view(-1, 1)).squeeze(1)   # (B,)
            o["random_won"] = bool(won[0].item())
            o["random_won_all"] = [bool(x) for x in won.detach().cpu().tolist()]
        return o

    if mode == "argmax":
        best = values.argmax(dim=1)  # (B,)
        best_action = actions[torch.arange(B, device=actions.device), best]  # (B, horizon, action_dim)
        out["chunk"] = best_action[:, start:end]
        out["selected_index"] = int(best[0].item())
        out["selected_value"] = float(values[0, int(best[0].item())].item())
        return _mark_random(out, best)

    if mode == "softmax":
        # Sample a candidate index from softmax(values / temp). temp -> 0 is
        # argmax, temp -> inf is uniform. Default temp=1.0.
        T = max(float(softmax_temp), 1e-6)
        probs = torch.softmax(values / T, dim=1)  # (B, K)
        # SEED the stochastic selection so the softmax pick is reproducible
        # run-to-run, using a dedicated RNG stream derived from replan_seed
        # (distinct from the candidate-generation seed applied above). Previously
        # this reseeded the global RNG from OS entropy (torch.seed()), which made
        # the selection — and hence the entire rollout — non-reproducible across
        # runs even with identical env + candidate seeds (see the 120-vs-500
        # divergence analysis: same candidate set at t=0, different pick). Set
        # SOFTMAX_SELECT_UNSEEDED=1 to restore the old entropy-seeded behavior.
        if _per_env_seeds is not None and _os.environ.get("SOFTMAX_SELECT_UNSEEDED", "0") != "1":
            # Per-env selection RNG, anchored to each episode's own seed — so the
            # softmax pick (like the candidate noise) is reproducible and NOT
            # correlated across envs / shifted by done-timing in a tiled batch.
            sampled = torch.empty(probs.shape[0], dtype=torch.long, device=probs.device)
            for _k in range(probs.shape[0]):
                _sg = torch.Generator(device=probs.device)
                _sg.manual_seed((int(_per_env_seeds[_k]) + 999331) % (2 ** 31 - 1))
                sampled[_k] = torch.multinomial(probs[_k:_k + 1], num_samples=1,
                                                generator=_sg).squeeze(1)
        elif replan_seed is not None and _os.environ.get("SOFTMAX_SELECT_UNSEEDED", "0") != "1":
            _sel_g = torch.Generator(device=probs.device)
            _sel_g.manual_seed((int(replan_seed) + 999331) % (2 ** 31 - 1))
            sampled = torch.multinomial(probs, num_samples=1, generator=_sel_g).squeeze(1)  # (B,)
        else:
            if replan_seed is not None:
                torch.seed()
            sampled = torch.multinomial(probs, num_samples=1).squeeze(1)  # (B,)
        chosen_action = actions[torch.arange(B, device=actions.device), sampled]
        out["chunk"] = chosen_action[:, start:end]
        out["selected_index"] = int(sampled[0].item())
        out["selected_value"] = float(values[0, int(sampled[0].item())].item())
        out["softmax_temp"] = float(T)
        return _mark_random(out, sampled)

    raise ValueError(
        f"unknown selection mode {mode!r}; expected 'argmax' or 'softmax'"
    )


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

    init_state = None   # captured at the first step: asset + arm start poses
    obj_goal_dist = float("nan")       # source-object -> target distance, latest step
    obj_goal_dist_min = float("inf")   # closest the object ever got to the goal
    while t < max_steps:
        step_obs = build_obs_step(
            env=env_base, obs=obs,
            prev_arm_action=prev_arm_action,
            prev_gripper_action=prev_gripper_action,
            source_obj=source_obj, target_obj=target_obj, tcp_link=tcp_link,
        )
        window.push(step_obs)
        # object -> goal distance (eggplant->basket, cube->cube, spoon->towel):
        # ||insertive_asset - receptive_asset|| in world xyz, tracked each step.
        _src = step_obs.get("insertive_asset_pose", None)
        _tgt = step_obs.get("receptive_asset_pose", None)
        if _src is not None and _tgt is not None:
            obj_goal_dist = float(np.linalg.norm(
                np.asarray(_src, dtype=np.float32)[:3] - np.asarray(_tgt, dtype=np.float32)[:3]))
            obj_goal_dist_min = min(obj_goal_dist_min, obj_goal_dist)
        if init_state is None:
            init_state = {}
            for _name, _key in (("insertive_xyz", "insertive_asset_pose"),
                                ("receptive_xyz", "receptive_asset_pose"),
                                ("ee_xyz", "end_effector_pose")):
                _v = step_obs.get(_key, None)
                init_state[_name] = (np.asarray(_v, dtype=np.float32)[:3].tolist()
                                     if _v is not None else None)
            _aq = step_obs.get("arm_joint_pos", None)
            init_state["arm_q"] = (np.asarray(_aq, dtype=np.float32).tolist()
                                   if _aq is not None else None)

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
                    "candidate_is_random": np.asarray(
                        pred.get("candidate_is_random",
                                 np.zeros(len(pred["values"]), dtype=bool)), dtype=bool),
                    "selected_index": int(pred.get("selected_index", -1)),
                    "selected_value": float(pred.get("selected_value", float("nan"))),
                    "frame": current_frame.astype(np.uint8, copy=True),
                    "ee_xyz": tcp_world,
                    # object (source) + target xyz for trajectory plots (already in step_obs)
                    "src_xyz": (np.asarray(step_obs["insertive_asset_pose"], dtype=np.float32)[:3]
                                if step_obs.get("insertive_asset_pose") is not None else np.zeros(3, dtype=np.float32)),
                    "tgt_xyz": (np.asarray(step_obs["receptive_asset_pose"], dtype=np.float32)[:3]
                                if step_obs.get("receptive_asset_pose") is not None else np.zeros(3, dtype=np.float32)),
                })
            replan_t = t
            replan_selected_index = pred.get("selected_index")
            replan_selected_value = pred.get("selected_value")
            replan_values = pred.get("values")
            replan_random_won = pred.get("random_won")
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
        if is_replan and replan_random_won is not None:
            step_entry["random_won"] = bool(replan_random_won)
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
        "obj_goal_dist_final": obj_goal_dist,
        "obj_goal_dist_min": (None if obj_goal_dist_min == float("inf") else obj_goal_dist_min),
        "steps": step_log,
        "frames": frames,
        "branches": branches,
        "cam_K": cam_K,
        "cam_T_wc": cam_T_wc,
        "base_pose_world": base_pose_world,
        "init_state": init_state,
    }


def _override_env_timelimit(env, max_steps):
    """Override the env TimeLimit so the episode horizon == max_steps (the registered
    cap is the task-native 60/120 and would truncate regardless of MAX_STEPS)."""
    ms = int(max_steps); w = env; base = getattr(env, "unwrapped", env)
    while w is not None:
        if hasattr(w, "_max_episode_steps"):
            w._max_episode_steps = ms
        sp = getattr(w, "spec", None)
        if sp is not None and getattr(sp, "max_episode_steps", None) is not None:
            try: sp.max_episode_steps = ms
            except Exception: pass
        w = getattr(w, "env", None)
    for a in ("max_episode_steps", "_max_episode_steps"):
        if hasattr(base, a):
            try: setattr(base, a, ms)
            except Exception: pass


def rollout_episodes_tiled(envs, policy, cfg, device, seeds, max_steps, instruction,
                           mode="softmax", n_samples=8, softmax_temp=1.0, tcont=None,
                           deterministic_sampling=True, sample_seed_base=0):
    """Vectorized rollout: run len(envs) episodes in LOCKSTEP, batching the policy +
    verifier over the env dimension (B = #active envs) at each replan. Handles variable
    episode length via a per-env done mask (finished envs are frozen, not stepped).
    Fast path: SR + object->goal distance only (no per-step frames/viz). Returns a list
    of per-env result dicts aligned with `seeds`."""
    from simpler_env.utils.env.observation_utils import (  # type: ignore
        get_image_from_maniskill2_obs_dict,
    )
    N = len(envs)
    import os as _os_pe
    # TILED_SEED_PER_ENV=1 (default) anchors each env's candidate noise to its own
    # episode seed — reproducible + decorrelated regardless of done-timing, and
    # identical to the single-env path's formula. The legacy alternative
    # (TILED_SEED_PER_ENV=0) uses a shared per-step seed that DROPS the episode
    # seed, so candidate noise correlates across envs and shifts whenever
    # done-timing changes the batch composition. Set 0 only to reproduce old runs.
    _tiled_per_env_seed = _os_pe.environ.get("TILED_SEED_PER_ENV", "1") == "1"
    n_obs_steps = int(cfg.n_obs_steps); n_action_steps = int(cfg.n_action_steps)
    st = [dict(done=False, success=False, truncated=False, num_steps=0, dmin=float("inf"),
               pending=[], prev_arm=np.zeros(6, np.float32), prev_grip=np.zeros(1, np.float32))
          for _ in range(N)]
    obs_l, base_l, src_l, tgt_l, tcp_l, win_l = [], [], [], [], [], []
    for k, env in enumerate(envs):
        o, _ = env.reset(seed=int(seeds[k]))
        _override_env_timelimit(env, max_steps)
        be = getattr(env, "unwrapped", env)
        obs_l.append(o); base_l.append(be)
        src_l.append(getattr(be, "episode_source_obj", None))
        tgt_l.append(getattr(be, "episode_target_obj", None))
        tcp_l.append(getattr(be, "tcp", None))
        win_l.append(ObsWindow(n_obs_steps=n_obs_steps))
    t = 0
    while t < max_steps and not all(s["done"] for s in st):
        frames = {}; need = []
        for k in range(N):
            if st[k]["done"]:
                continue
            step_obs = build_obs_step(
                env=base_l[k], obs=obs_l[k],
                prev_arm_action=st[k]["prev_arm"], prev_gripper_action=st[k]["prev_grip"],
                source_obj=src_l[k], target_obj=tgt_l[k], tcp_link=tcp_l[k])
            win_l[k].push(step_obs)
            _s = step_obs.get("insertive_asset_pose"); _g = step_obs.get("receptive_asset_pose")
            if _s is not None and _g is not None:
                d = float(np.linalg.norm(np.asarray(_s, np.float32)[:3] - np.asarray(_g, np.float32)[:3]))
                st[k]["dmin"] = min(st[k]["dmin"], d)
            img = get_image_from_maniskill2_obs_dict(base_l[k], obs_l[k])
            frames[k] = np.asarray(img, dtype=np.uint8)
            if not st[k]["pending"]:
                need.append(k)
        if need:                                    # batched replan over active envs
            ods = [win_l[k].to_tensor_dict(device=device) for k in need]
            batch = {key: torch.cat([od[key] for od in ods], dim=0) for key in ods[0]}
            imgs = torch.stack([torch.from_numpy(frames[k]).to(device=device) for k in need], dim=0)
            batch["agentview_image"] = imgs.unsqueeze(1).expand(
                len(need), n_obs_steps, *frames[need[0]].shape)
            if not deterministic_sampling:
                replan_seed = None
            elif _tiled_per_env_seed:
                # Per-env candidate seed = f(episode seed, t) — identical formula to
                # the single-env path — so each active env's candidate noise is anchored
                # to its OWN episode seed, independent of its batch position / other
                # envs' done-timing. Aligned with the `need` order = the batch rows.
                replan_seed = [((int(seeds[k]) * 1_000_003 + int(t) + int(sample_seed_base))
                                % (2 ** 31 - 1)) for k in need]
            else:
                # legacy: shared per-step seed (drops the episode seed -> candidate noise
                # correlates across envs and shifts when done-timing changes the batch).
                replan_seed = ((t + int(sample_seed_base)) * 1_000_003) % (2 ** 31 - 1)
            pred = _search_policy_predict(policy, batch, mode=mode, viz_q=False,
                                          n_samples=n_samples, softmax_temp=softmax_temp,
                                          tcont=tcont, replan_seed=replan_seed)
            chunks = pred["chunk"].detach().cpu().numpy()   # (len(need), n_action_steps, 7)
            _rw = pred.get("random_won_all")
            for bi, k in enumerate(need):
                st[k]["pending"] = list(chunks[bi].astype(np.float32, copy=False))
                if _rw is not None:
                    st[k]["n_replans"] = st[k].get("n_replans", 0) + 1
                    st[k]["n_rand_won"] = st[k].get("n_rand_won", 0) + int(bool(_rw[bi]))
        for k in range(N):                          # step every active env
            if st[k]["done"]:
                continue
            a = st[k]["pending"].pop(0).astype(np.float32)
            obs_l[k], _r, dn, tr, _info = envs[k].step(convert_maniskill(a.copy()))
            st[k]["prev_arm"] = a[:6].copy(); st[k]["prev_grip"] = a[6:7].copy()
            st[k]["num_steps"] = t + 1
            if bool(dn):
                st[k]["success"] = True; st[k]["done"] = True
            elif bool(tr):
                st[k]["truncated"] = True; st[k]["done"] = True
        t += 1
    return [dict(seed=int(seeds[k]), success=st[k]["success"], truncated=st[k]["truncated"],
                 num_steps=st[k]["num_steps"],
                 obj_goal_dist_final=(None if st[k]["dmin"] == float("inf") else st[k]["dmin"]),
                 obj_goal_dist_min=(None if st[k]["dmin"] == float("inf") else st[k]["dmin"]),
                 n_replans=st[k].get("n_replans"), n_rand_won=st[k].get("n_rand_won"),
                 branches=None, frames=[], steps=[], init_state=None) for k in range(N)]


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
    parser.add_argument("--mode", choices=["argmax", "softmax"], default="argmax",
                        help="Selection over all --n-samples candidates. "
                             "argmax: pick the highest-verifier-value sample. "
                             "softmax: sample index ~ softmax(values / temp), "
                             "tunable via --softmax-temp.")
    parser.add_argument("--softmax-temp", type=float, default=1.0,
                        help="Temperature for --mode softmax. Lower -> more "
                             "argmax-like; higher -> more uniform. Default 1.0.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Disable per-episode resume. By default, if "
                             "<output_dir>/episodes.jsonl exists, episodes "
                             "already completed for a seed are reused (same "
                             "seed -> identical deterministic rollout) and only "
                             "the NEW seeds are run — so re-running a cell with "
                             "a larger --num-episodes just adds episodes.")
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
                        help="Number of search samples per replan; selection "
                             "(argmax/softmax) runs over all of them. "
                             "Default: policy.max_actions. Values > max_actions "
                             "are supported — predict_n_actions slides a fixed "
                             "(max_actions-1) context window.")
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
                        default=None,
                        help="Aggregation over the executed-window step scores "
                             "when --score-window executed. Default: use the "
                             "checkpoint's baked score_agg (matches training).")
    parser.add_argument("--score-discount", type=float, default=None,
                        help="Geometric discount on later chunk steps: step t is "
                             "weighted by score_discount**t before sum/mean. "
                             "Default: use the checkpoint's baked value (1.0).")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[eval] CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")

    # Determinism (default ON; DETERMINISTIC=0 to disable): make the torch policy +
    # verifier forward passes use deterministic cuDNN/cuBLAS kernels so a given
    # (env seed, ckpt, N) reproduces the SAME rollout run-to-run. Combined with the
    # seeded softmax selection, this removes the residual near-tie flips we saw.
    # (CUBLAS_WORKSPACE_CONFIG must be set in the env before CUDA init -> the sbatch.)
    import os as _os_det
    if _os_det.environ.get("DETERMINISTIC", "1") == "1":
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
            print("[eval] deterministic torch kernels ON (cudnn.deterministic, use_deterministic_algorithms)")
        except Exception as _e:
            print(f"[eval] could not fully enable determinism: {_e!r}")

    policy, cfg = load_policy(
        checkpoint=args.checkpoint, device=device, use_ema=not args.no_ema,
    )
    # Inference-time override of how many predicted-horizon steps are executed
    # per replan. The policy ALWAYS predicts the full horizon; n_action_steps
    # only sets the executed window [n_obs_steps-1 : +n_action_steps] (and, for
    # --score-window executed, how many steps the verifier scores). This lets a
    # trained 8-step policy be evaluated at e.g. 4 steps without retraining.
    # Set via the N_ACTION_STEPS env var (eval_search.sh passes the env through).
    import os as _os_env
    _nas_env = _os_env.environ.get("N_ACTION_STEPS")
    if _nas_env:
        _new_nas = int(_nas_env)
        try:
            from omegaconf import OmegaConf as _OC
            _OC.set_struct(cfg, False)
        except Exception:
            pass
        _old_nas = int(getattr(policy, "n_action_steps", cfg.n_action_steps))
        if hasattr(policy, "n_action_steps"):
            policy.n_action_steps = _new_nas
        cfg.n_action_steps = _new_nas
        print(f"[eval] N_ACTION_STEPS override: n_action_steps {_old_nas} -> "
              f"{_new_nas} (execute {_new_nas} of horizon="
              f"{int(cfg.get('horizon', 0))} per replan)")
    # BC checkpoint (e.g. DiffusionUnetImagePolicy) has no predict_n_actions or
    # verifier. Wrap it with BCSearchWrapper so the rest of the eval harness works
    # unchanged: N independent BC samples are scored by the in-process verifier.
    _bc_wrapped = False
    if not hasattr(policy, "predict_n_actions"):
        _bc_wrapped = True
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
        _vtype = _os.environ.get("VERIFIER_TYPE", "robomonkey").lower()
        if _vtype == "dqc":
            # DQC Q-function verifier (JAX, in-process). Scores each executed
            # action against the current frame; BCSearchWrapper aggregates the
            # per-step Q-values (BC_VERIFIER_AGG, default discounted_sum here).
            from diffusion_policy.policy.dqc_verifier import DQCVerifier
            _ckpt = _os.environ["DQC_CKPT_DIR"]
            # REAL BridgeDQCChunkAgent verifier. critic_type: chunk (8-step chunk
            # critic, default) | action (single-step). DQC_CRITIC_MODULE kept as a
            # back-compat alias for DQC_CRITIC_TYPE.
            _critic_type = _os.environ.get(
                "DQC_CRITIC_TYPE", _os.environ.get("DQC_CRITIC_MODULE", "chunk")).lower()
            _verifier = DQCVerifier(
                ckpt_dir=_ckpt,
                instruction=_instr,
                image_obs_key=_img_key,
                text_emb_file=_os.environ.get("DQC_TEXT_EMB"),
                critic_type=_critic_type,
                # Chunk critic scores ALL the executed steps the BC wrapper passes
                # (S = N_ACTION_STEPS = chunk_horizon) as one chunk.
                score_full_horizon=(_critic_type == "chunk"),
                health_check=True,
            )
            if _critic_type == "chunk":
                _os.environ.setdefault("BC_VERIFIER_SCORE", "chunk_critic")
            else:
                _os.environ.setdefault("BC_VERIFIER_AGG", "discounted_sum")
            print(f"[eval] BC verifier = REAL DQC ({_ckpt}, critic_type={_critic_type}, "
                  f"BC_VERIFIER_SCORE={_os.environ.get('BC_VERIFIER_SCORE', 'chunk')})")
        else:
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
        if _vtype == "dqc":
            # DQC eval: candidate count = N_SAMPLES (default 16), no 128 floor.
            _max_actions = int(_os.environ.get("N_SAMPLES", "16"))
        else:
            _max_actions = max(128, int(_os.environ.get("N_SAMPLES", "128")))
        policy = BCSearchWrapper(policy, _verifier, max_actions=_max_actions)
        print(f"[eval] BC mode: wrapped with in-process verifier "
              f"(instruction={_instr!r}, max_actions={_max_actions})")
    else:
        # Native search policy (predict_n_actions + baked verifier). Optionally
        # swap the baked LLaVA verifier for the DQC Q-critic so the
        # autoregressive search scores candidates by the discounted chunk sum
        # of DQC Q-values. Enabled with VERIFIER_TYPE=dqc; the score_window
        # block below sets score_indices to the executed chunk on it.
        import os as _os
        if (_os.environ.get("VERIFIER_TYPE", "robomonkey").lower() == "dqc"
                and type(getattr(policy, "verifier", None)).__name__ != "DQCVerifier"):
            from diffusion_policy.policy.dqc_verifier import DQCVerifier
            _img_key = (getattr(policy.verifier, "image_obs_key", None)
                        or _os.environ.get("VERIFIER_IMAGE_KEY", "agentview_image"))
            _critic_type = _os.environ.get(
                "DQC_CRITIC_TYPE", _os.environ.get("DQC_CRITIC_MODULE", "chunk")).lower()
            policy.verifier = DQCVerifier(
                ckpt_dir=_os.environ["DQC_CKPT_DIR"],
                instruction=_os.environ.get("VERIFIER_INSTRUCTION",
                                            "put the eggplant in the basket"),
                image_obs_key=_img_key,
                text_emb_file=_os.environ.get("DQC_TEXT_EMB"),
                critic_type=_critic_type,
                # search policy sets score_indices to the executed window below.
                score_executed_chunk=True,
                health_check=True,
            )
            print(f"[eval] SEARCH verifier swapped -> REAL DQC "
                  f"(critic_type={_critic_type})")
        elif (_os.environ.get("DQC_VERIFIER_CKPT")
              and type(getattr(policy, "verifier", None)).__name__ == "DQCVerifier"):
            # Re-point the policy's BAKED DQC verifier at a different (e.g. FINETUNED)
            # ckpt, PRESERVING all its scoring config (critic_type, score_full_horizon,
            # q_agg, discount, score_agg, score_indices) so the ONLY change vs the
            # baked base run is the verifier WEIGHTS — an apples-to-apples comparison.
            # Uses DQC_VERIFIER_EPOCH (NOT DQC_RESTORE_EPOCH, which would also be read
            # by the baked verifier's load above and break it, since the base ckpt has
            # no params_<that>.pkl).
            from diffusion_policy.policy.dqc_verifier import DQCVerifier
            _v = policy.verifier
            _ovr = _os.environ["DQC_VERIFIER_CKPT"]
            _ep = int(_os.environ.get("DQC_VERIFIER_EPOCH", "30000"))
            policy.verifier = DQCVerifier(
                ckpt_dir=_ovr,
                instruction=_v.instruction,
                image_obs_key=_v.image_obs_key,
                text_emb_file=_os.environ.get("DQC_TEXT_EMB"),
                critic_type=_v.critic_type,
                score_indices=_v.score_indices,
                score_agg=_v.score_agg,
                score_executed_chunk=_v.score_executed_chunk,
                score_full_horizon=_v.score_full_horizon,
                action_chunk_index=_v.action_chunk_index,
                chunk_horizon=_v._chunk_horizon_cfg,
                restore_epoch=_ep,
                discount=_v.discount,
                q_agg=_v.q_agg,
                image_hw=_v.image_hw,
                health_check=True,
            )
            print(f"[eval] baked DQC verifier RE-POINTED -> {_ovr} @ epoch {_ep} "
                  f"(scoring preserved: critic={_v.critic_type}, "
                  f"full_horizon={_v.score_full_horizon}, q_agg={_v.q_agg})")
    print(f"[eval] max_actions={int(policy.max_actions)}  (in-process verifier)")

    # Configure what the verifier scores. 'executed': score each of the
    # n_action_steps actions that actually get executed (the chunk window
    # start..start+n_action_steps) against the current frame, aggregated via
    # --score-agg, so the candidate value = quality of the whole executed
    # chunk. 'last': leave the verifier's single action_chunk_index (default).
    _score_indices = None
    if _bc_wrapped:
        # BCSearchWrapper does its OWN per-step chunk scoring: it slices the
        # executed window itself and calls verifier.get_value with single-step
        # actions (shape (M, 1, 7)), aggregating via BC_VERIFIER_SCORE/AGG.
        # Setting a multi-index score_indices here would make get_value do
        # action[:, [1..nas], :] on a horizon-1 tensor -> CUDA device-side
        # assert (index out of bounds). Leave it single-step for BC.
        policy.verifier.score_indices = None
        print(f"[eval] verifier scoring = BC wrapper chunk scoring "
              f"(BC_VERIFIER_SCORE={_os.environ.get('BC_VERIFIER_SCORE', 'chunk')}, "
              f"BC_VERIFIER_AGG={_os.environ.get('BC_VERIFIER_AGG', 'mean')}); "
              f"verifier.score_indices left single-step")
    elif args.score_window == "executed":
        _To = int(policy.n_obs_steps)
        _nas = int(policy.n_action_steps)
        _start = _To - 1
        _score_indices = list(range(_start, _start + _nas))
        policy.verifier.score_indices = _score_indices
        # Default to the checkpoint's baked score_agg (matches training); only
        # override when --score-agg is passed explicitly.
        if args.score_agg:
            policy.verifier.score_agg = str(args.score_agg)
        if args.score_discount is not None:
            policy.verifier.score_discount = float(args.score_discount)
        _agg_used = getattr(policy.verifier, "score_agg", "mean")
        # RoboMonkeyVerifier exposes score_discount; DQCVerifier its own `discount`.
        _disc_used = getattr(policy.verifier, "score_discount", None)
        if _disc_used is None:
            _disc_used = getattr(policy.verifier, "discount", 1.0)
        print(f"[eval] verifier scoring = EXECUTED window steps "
              f"{_score_indices[0]}..{_score_indices[-1]} "
              f"(agg={_agg_used}, discount={_disc_used}, "
              f"{_nas}x verifier forwards/candidate)")
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
    # The env's registered TimeLimit truncates at the TASK-NATIVE cap (60 for
    # stack/spoon, 120 for eggplant) — which caps the episode REGARDLESS of
    # args.max_steps. Override every wrapper's _max_episode_steps (+ spec + base)
    # so the true episode horizon equals max_steps.
    _override_env_timelimit(env, args.max_steps)
    print(f"[eval] TimeLimit overridden -> max_episode_steps={int(args.max_steps)} (was task-native 60/120)")
    instr = (
        _base_env.get_language_instruction()
        if hasattr(_base_env, "get_language_instruction") else None
    ) or args.task
    print(f"[eval] instruction: {instr!r}")

    output_dir = Path(args.output_dir) if args.output_dir else None
    done_by_seed: Dict[int, Dict[str, Any]] = {}
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _ep_path = output_dir / "episodes.jsonl"
        # Per-episode resume: load episodes already completed (keyed by seed)
        # before truncating, so a re-run with more episodes reuses them.
        if (not args.no_resume) and _ep_path.exists():
            _skipped_errors = 0
            for _line in _ep_path.read_text().splitlines():
                try:
                    _r = json.loads(_line)
                except Exception:
                    continue
                if not (isinstance(_r, dict) and "seed" in _r and "success" in _r):
                    continue
                # Do NOT treat an errored episode as completed. A CUDA
                # device-side assert poisons the process context, so once one
                # fires every remaining episode in that run is logged as an
                # instant failure ({"error": ..., "num_steps": 0}). Reusing
                # those would pin the cell's SR at a garbage value forever;
                # drop them here so this run actually re-runs those seeds.
                if _r.get("error") or _r.get("num_steps") in (None, 0):
                    _skipped_errors += 1
                    continue
                done_by_seed[int(_r["seed"])] = _r
            if done_by_seed or _skipped_errors:
                print(f"[eval] resume: reusing {len(done_by_seed)} completed "
                      f"episode(s) from {_ep_path.name}"
                      + (f"; re-running {_skipped_errors} previously-errored "
                         f"episode(s)" if _skipped_errors else ""))
        ep_log_file = open(_ep_path, "w")
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
    obj_goal_dists_min: List[float] = []    # closest object->goal distance per episode
    t0 = time.time()
    all_selected_indices: List[int] = []
    rand_replans = 0
    rand_wins = 0

    # --- TILED parallel path: run TILED_ENVS episodes concurrently (batched policy +
    # verifier over the env dim), precompute all results, then reuse the same
    # accumulation/logging below. Fast SR + distance path (no per-step videos). ---
    import os as _os_tiled
    TILED = int(_os_tiled.environ.get("TILED_ENVS", "1"))
    tiled_eps: List[Dict[str, Any]] = []
    if TILED > 1:
        import simpler_env  # noqa: F401
        from simpler_env import make as make_env
        _all_seeds = ([int(s) for s in explicit_seeds] if explicit_seeds is not None
                      else ([int(args.start_seed)] * num_episodes if args.repeat_seed
                            else [int(args.start_seed) + i for i in range(num_episodes)]))
        tiled_envs = [env] + [make_env(args.task) for _ in range(TILED - 1)]
        print(f"[eval] TILED mode: {TILED} parallel envs over {num_episodes} episodes", flush=True)
        for b in range(0, num_episodes, TILED):
            bseeds = _all_seeds[b:b + TILED]
            res = rollout_episodes_tiled(
                tiled_envs[:len(bseeds)], policy, cfg, device, bseeds, int(args.max_steps),
                str(instr), mode=str(args.mode), n_samples=args.n_samples,
                softmax_temp=float(args.softmax_temp), tcont=args.tcont,
                deterministic_sampling=(not args.stochastic_sampling),
                sample_seed_base=int(args.sample_seed_base))
            tiled_eps.extend(res)
            print(f"[eval]   tiled batch {b//TILED+1}: seeds {bseeds} -> "
                  f"succ={[int(r['success']) for r in res]} steps={[r['num_steps'] for r in res]}",
                  flush=True)

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

        # Resume: reuse a previously-completed episode for this seed (its
        # render data / video, if any, are already on disk). Accumulate its
        # outcome and re-emit its record at the current position.
        if ep_seed in done_by_seed:
            rec = dict(done_by_seed[ep_seed]); rec["ep_idx"] = i
            successes += int(rec.get("success", 0))
            truncations += int(rec.get("truncated", 0))
            if rec.get("selected_indices"):
                all_selected_indices.extend(int(x) for x in rec["selected_indices"])
            durations.append(float(rec.get("duration_s", 0.0)))
            if ep_log_file is not None:
                ep_log_file.write(json.dumps(rec) + "\n"); ep_log_file.flush()
            sr_so_far = successes / float(i + 1)
            print(f"[eval] ep={i+1:3d}/{num_episodes}  seed={ep_seed}  REUSED  "
                  f"success={int(rec.get('success', 0))}  "
                  f"running_sr={sr_so_far:.3f}", flush=True)
            continue

        ep_t0 = time.time()
        try:
            if TILED > 1:
                ep = tiled_eps[i]                    # precomputed by the tiled batch loop
            else:
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
            # A CUDA device-side assert (or any CUDA error) corrupts the
            # process's CUDA context: every subsequent kernel launch throws the
            # same error, so continuing would just log all remaining seeds as
            # instant phantom failures and tank the cell's SR. Abort the run
            # hard instead. We do NOT write a record for this seed, so on the
            # next launch resume reuses the good episodes and re-runs this one
            # (and the rest) in a fresh process.
            if "CUDA error" in str(e) or "device-side assert" in str(e):
                print(f"[eval] FATAL: CUDA context corrupted at seed={ep_seed}; "
                      f"aborting run without logging phantom failures. "
                      f"Re-launch to resume the remaining seeds.", flush=True)
                if ep_log_file is not None:
                    ep_log_file.flush()
                raise
            ep = {"success": False, "truncated": False, "num_steps": 0,
                  "obj_goal_dist_final": None, "obj_goal_dist_min": None,
                  "error": repr(e), "steps": [], "frames": [], "branches": None,
                  "cam_K": None, "cam_T_wc": None, "base_pose_world": None}

        durations.append(time.time() - ep_t0)
        successes += int(ep["success"])
        truncations += int(ep["truncated"])
        if ep.get("obj_goal_dist_min") is not None:
            obj_goal_dists_min.append(float(ep["obj_goal_dist_min"]))
        sr_so_far = successes / float(i + 1)
        _dmin = ep.get("obj_goal_dist_min")
        print(
            f"[eval] ep={i+1:3d}/{num_episodes}  seed={ep_seed}  "
            f"success={int(ep['success'])}  trunc={int(ep['truncated'])}  "
            f"steps={ep['num_steps']:3d}  obj_goal_dist(min={_dmin if _dmin is None else round(_dmin,4)})  "
            f"elapsed={durations[-1]:5.1f}s  "
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
                src_xyz=np.stack([b.get("src_xyz", np.zeros(3, dtype=np.float32))
                                  for b in br], axis=0),
                tgt_xyz=np.stack([b.get("tgt_xyz", np.zeros(3, dtype=np.float32))
                                  for b in br], axis=0),
                **extra_proj,
                mode=str(args.mode),
                max_actions=np.int32(int(policy.max_actions)),
                branch_t=np.asarray([b["t"] for b in br], dtype=np.int32),
                candidate_actions=np.stack(
                    [b["candidate_actions"] for b in br], axis=0
                ),
                values=np.stack([b["values"] for b in br], axis=0),
                # (R, K) bool: which candidates came from the random-augmentation
                # pool (all-False when RAND_AUG_K is off) -- the renderer draws
                # those dashed.
                candidate_is_random=np.stack(
                    [b["candidate_is_random"] for b in br], axis=0
                ),
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
        # Random-augmentation bookkeeping: how often an augmented candidate beat
        # every policy candidate. Single-env path records it per replan step;
        # the tiled path aggregates counts per env.
        ep_rand_won = [bool(s["random_won"]) for s in ep.get("steps", [])
                       if "random_won" in s]
        if ep_rand_won:
            rand_replans += len(ep_rand_won); rand_wins += int(sum(ep_rand_won))
        elif ep.get("n_replans") is not None:
            rand_replans += int(ep["n_replans"]); rand_wins += int(ep.get("n_rand_won") or 0)

        if ep_log_file is not None:
            ep_compact = {
                "ep_idx": i,
                "seed": ep_seed,
                "success": ep["success"],
                "truncated": ep["truncated"],
                "num_steps": ep["num_steps"],
                "obj_goal_dist_final": ep.get("obj_goal_dist_final"),
                "obj_goal_dist_min": ep.get("obj_goal_dist_min"),
                "duration_s": durations[-1],
            }
            if ep.get("init_state") is not None:
                ep_compact["init_state"] = ep["init_state"]
            if ep_selected_indices:
                ep_compact["selected_indices"] = ep_selected_indices
            if ep_selected_values:
                ep_compact["selected_values"] = ep_selected_values
            if ep_values_per_replan:
                ep_compact["q_values"] = ep_values_per_replan
            if ep_rand_won:
                ep_compact["random_won"] = ep_rand_won
            elif ep.get("n_replans") is not None:
                ep_compact["n_replans"] = int(ep["n_replans"])
                ep_compact["n_rand_won"] = int(ep.get("n_rand_won") or 0)
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
        "score_agg": str(getattr(policy.verifier, "score_agg", args.score_agg)),
        "score_discount": float(getattr(policy.verifier, "score_discount", None)
                                 or getattr(policy.verifier, "discount", 1.0)),
        "max_actions": int(policy.max_actions),
        "num_successes": int(successes),
        "num_truncated": int(truncations),
        "success_rate": float(success_rate),
        "mean_obj_goal_dist_min": (float(np.mean(obj_goal_dists_min)) if obj_goal_dists_min else None),
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
    # Random-augmentation config + how often the augmented pool won the argmax.
    from random_action_aug import config_from_env as _rand_cfg
    _acfg = _rand_cfg()
    if _acfg.enabled:
        summary["rand_aug_k"] = int(_acfg.k)
        summary["rand_aug_m"] = int(_acfg.m)
        summary["rand_aug_gripper"] = str(_acfg.gripper)
        summary["rand_aug_only"] = bool(_acfg.only)
        summary["rand_aug_mag_min"] = float(_acfg.mag_min)
        summary["frac_replans_random_won"] = (
            float(rand_wins) / rand_replans if rand_replans else None)
        summary["num_replans_random_scored"] = int(rand_replans)
    print("\n[eval] ===== summary =====")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if output_dir is not None:
        with open(output_dir / "eval_log.json", "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"\n[eval] Wrote {output_dir / 'eval_log.json'}")
        if ep_log_file is not None:
            ep_log_file.close()
        # Diagnostic: start positions of assets + arm, train vs eval, with the
        # visualized (search_q) episodes highlighted. Never fail the eval on it.
        try:
            from plot_start_positions import plot_start_positions as _plot_starts
            _plot_starts(str(output_dir), str(args.task),
                         viz_episodes=int(args.viz_q_episodes),
                         out_path=str(output_dir / "start_positions.png"))
        except Exception as _e:  # noqa: BLE001
            print(f"[eval] start-positions plot skipped: {_e!r}")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    main()
