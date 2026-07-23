#!/usr/bin/env python
"""sample_offline_candidates.py — offline analog of the eval_search rollout, for
visualizing candidate actions + verifier Q + the recorded action a* on demo
(train) / held-out (val) frames.

Runs in the ``dqc_jax`` conda env (torch policy + JAX DQC verifier in one
process). For a handful of fixed episodes in a zarr it strides over frames and,
at each frame:

  1. reconstructs the policy obs from the stored state obs (the same arrays the
     training loader feeds the policy — no SimplerEnv rollout needed);
  2. samples N candidate action chunks via ``policy.predict_n_actions`` (the
     same call eval_search uses), scored by the (baked, or BC-wrapped) DQC chunk
     verifier — these Q-values ARE the ones the AUC eval ranks on;
  3. scores a* (the recorded next-``chunk_horizon`` actions, NaN-padded past the
     episode end) with the SAME verifier so it lands on the candidates' Q scale;

then writes one render_q-compatible ``.npz`` per episode under
``<out-dir>/search_q/`` (extra offline keys ``a_star`` / ``a_star_q`` /
``a_star_rank``). The camera intrinsics are rescaled to the 224x224 zarr frame
(the agentview_image is a 640x480 -> 128 -> 224 squish of the 3rd_view_camera).

Verifier wiring mirrors the eval sbatches:
  * search policy (predict_n_actions present): use the BAKED DQC chunk verifier
    as-is (critic_type=chunk, score_full_horizon=True, q_agg=min). USE_EMA=0.
  * BC policy (DiffusionUnetImagePolicy): DDPM-swap for sample diversity, wrap
    with BCSearchWrapper + a DQCVerifier(critic_type=chunk). USE_EMA=1.

Usage (see viz_pipeline.sbatch):
  python sample_offline_candidates.py --ckpt <ckpt> --policy-kind search \
      --zarr /gscratch/robotics/harine/data/eggplant_in_basket/state0.zarr \
      --split train --config-label diff33k --out-dir data/viz/diff33k/train \
      --n-samples 16 --stride 4 --num-episodes 10
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
# Reuse load_policy + helpers from eval_diffusion (same path trick eval_search uses).
sys.path.insert(0, str(_REPO / "scriptsv2" / "eval_diffusion"))

DQC_CKPT_DIR = os.environ.get(
    "DQC_CKPT_DIR", "/gscratch/robotics/harine/dqc_checkpoint/sd000s_137913.0.20260605_163017")
DQC_TEXT_EMB = os.environ.get(
    "DQC_TEXT_EMB", str(_REPO / "scriptsv2" / "q_fn_eval" / "task_emb_use3.npy"))
DEFAULT_INSTR = "put the eggplant in the basket"


# --------------------------------------------------------------------------- #
#  Policy construction (search = baked verifier; BC = wrap with DQCVerifier)   #
# --------------------------------------------------------------------------- #
def build_policy(ckpt: str, kind: str, device: torch.device, use_ema: bool,
                 n_samples: int, n_action_steps: int, instruction: str,
                 dqc_text_emb: str = DQC_TEXT_EMB):
    """Return (policy, verifier, To, chunk_h, image_key, known_keys).

    `dqc_text_emb` is used ONLY by the BC branch (the search policy carries a baked
    verifier with its own per-task embedding). Passing the wrong task's embedding
    silently makes BC's Q incomparable to search's -- always pass it explicitly.
    """
    from eval_diffusion import load_policy  # type: ignore

    policy, cfg = load_policy(ckpt, device, use_ema)

    if kind == "bc" or not hasattr(policy, "predict_n_actions"):
        # ---- BC: mirror eval_search.py's BC-wrap block ----
        from diffusion_policy.policy.dqc_verifier import DQCVerifier
        from diffusion_policy.policy.search_policy_robomonkey import BCSearchWrapper
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

        # DDPM ancestral sampling -> diverse candidates (DDIM-8 collapses them).
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
        _steps = int(os.environ.get("BC_SAMPLER_STEPS", str(_old.config.num_train_timesteps)))
        policy.num_inference_steps = _steps
        # The wrapper scores the first chunk_h PREDICTED actions as the chunk.
        os.environ.setdefault("BC_VERIFIER_SCORE", "chunk_critic")
        os.environ.setdefault("BC_CHUNK_FROM_PRED", "1")
        verifier = DQCVerifier(
            ckpt_dir=DQC_CKPT_DIR, instruction=instruction,
            image_obs_key="agentview_image", text_emb_file=dqc_text_emb,
            critic_type="chunk", score_full_horizon=True, health_check=True,
        )
        policy = BCSearchWrapper(policy, verifier, max_actions=int(n_samples))
        print(f"[offline] BC wrapped: BCSearchWrapper + DQC chunk verifier "
              f"(ddpm {_steps} steps, max_actions={n_samples})")
        print(f"[offline] BC text_emb = {dqc_text_emb}")
    else:
        verifier = policy.verifier   # baked DQC chunk verifier (path-fixed ckpt)
        print(f"[offline] search policy: baked verifier "
              f"{type(verifier).__name__} (critic_type="
              f"{getattr(verifier, 'critic_type', '?')}, "
              f"score_full_horizon={getattr(verifier, 'score_full_horizon', '?')})")
        print(f"[offline] search text_emb = "
              f"{getattr(verifier, 'text_emb_file', '?')}")

    # n_action_steps override (predict 8, rollout n_action_steps) — same as eval.
    if hasattr(policy, "n_action_steps"):
        policy.n_action_steps = int(n_action_steps)

    To = int(getattr(policy, "n_obs_steps", getattr(cfg, "n_obs_steps", 1)))
    chunk_h = int(getattr(verifier, "chunk_horizon", 8))
    image_key = getattr(verifier, "image_obs_key", "agentview_image")
    known = set(policy.normalizer.params_dict.keys())
    return policy, verifier, To, chunk_h, image_key, known


# --------------------------------------------------------------------------- #
#  Camera (captured once; widowx 3rd_view_camera is fixed on base_link)        #
# --------------------------------------------------------------------------- #
def capture_camera(task: str, img_hw: tuple):
    """Reset SimplerEnv once and read the fixed camera intrinsic/extrinsic +
    base pose. Rescale K from the native render res to the zarr frame res."""
    import simpler_env  # noqa: F401
    from simpler_env import make as make_env

    env = make_env(task)
    obs, _ = env.reset(seed=0)
    cam_params = obs.get("camera_param", {})
    cam_name = "3rd_view_camera" if "3rd_view_camera" in cam_params \
        else next(iter(cam_params), None)
    if cam_name is None:
        raise RuntimeError("no camera_param available from env reset")
    K = np.asarray(cam_params[cam_name]["intrinsic_cv"], dtype=np.float32).copy()
    T_wc = np.asarray(cam_params[cam_name]["extrinsic_cv"], dtype=np.float32)
    base_pose = np.asarray(obs.get("agent", {}).get("base_pose"), dtype=np.float32)
    try:
        env.close()
    except Exception:
        pass

    # Native render res implied by the principal point (cx ~ W/2, cy ~ H/2).
    cam_w, cam_h = 2.0 * float(K[0, 2]), 2.0 * float(K[1, 2])
    img_h, img_w = float(img_hw[0]), float(img_hw[1])
    sx, sy = img_w / cam_w, img_h / cam_h
    K[0, :] *= sx
    K[1, :] *= sy
    print(f"[offline] camera={cam_name}  native~{cam_w:.0f}x{cam_h:.0f} -> "
          f"frame {img_w:.0f}x{img_h:.0f}  (sx={sx:.4f}, sy={sy:.4f})")
    return K.astype(np.float32), T_wc.astype(np.float32), base_pose


# --------------------------------------------------------------------------- #
#  obs reconstruction from zarr                                                #
# --------------------------------------------------------------------------- #
def build_obs_dict(zobs, g: int, start: int, known_keys, image_key: str,
                   To: int, device: torch.device) -> Dict[str, torch.Tensor]:
    """Assemble the policy obs_dict for global step g from raw zarr arrays.
    To-step window = [g-To+1 .. g], clamped at the episode start (repeat first)."""
    idxs = [max(start, g - (To - 1) + i) for i in range(To)]   # length To, ascending
    obs_dict: Dict[str, torch.Tensor] = {}
    for k in known_keys:
        if k == "action" or k == image_key or k not in zobs:
            continue
        arr = np.stack([np.asarray(zobs[k][j], dtype=np.float32) for j in idxs])  # (To, dim)
        obs_dict[k] = torch.from_numpy(arr).unsqueeze(0).to(device)               # (1, To, dim)
    img = np.asarray(zobs[image_key][g], dtype=np.uint8)                           # (H, W, 3)
    img_t = torch.from_numpy(img).to(device)
    obs_dict[image_key] = img_t.unsqueeze(0).unsqueeze(0).expand(1, To, *img.shape)
    return obs_dict


def score_astar(verifier, image_key: str, img_u8: np.ndarray, a_star: np.ndarray,
                To: int, device: torch.device) -> float:
    """Score the recorded a* chunk on the candidates' Q scale (NaN pad -> the
    verifier zero-fills in normalized space, matching training assembly)."""
    img_t = torch.from_numpy(img_u8.astype(np.uint8)).to(device)
    obs = {image_key: img_t.unsqueeze(0).unsqueeze(0).expand(1, To, *img_u8.shape)}
    a = torch.from_numpy(a_star.astype(np.float32)).unsqueeze(0).to(device)  # (1, chunk_h, 7)
    return float(verifier.get_value(obs, a).detach().cpu().item())


# --------------------------------------------------------------------------- #
def pick_episodes(ep_starts, ep_ends, num: int) -> np.ndarray:
    n = len(ep_ends)
    if n <= num:
        return np.arange(n)
    return np.linspace(0, n - 1, num).round().astype(int)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--zarr", required=True, help="single zarr shard")
    ap.add_argument("--policy-kind", choices=["bc", "search"], required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], required=True,
                    help="label only (filename + npz field); the zarr is --zarr")
    ap.add_argument("--config-label", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--n-action-steps", type=int, default=4)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--num-episodes", type=int, default=10)
    ap.add_argument("--max-frames-per-ep", type=int, default=0, help="0 = all")
    ap.add_argument("--task", default="widowx_put_eggplant_in_basket")
    ap.add_argument("--instruction", default=DEFAULT_INSTR)
    ap.add_argument("--dqc-text-emb", default=DQC_TEXT_EMB,
                    help="DQC task text embedding, used ONLY for --policy-kind bc "
                         "(search carries a baked per-task verifier). Must match the "
                         "task or BC's Q is not comparable to search's.")
    ap.add_argument("--use-ema", type=int, default=-1,
                    help="-1 = auto (bc->1, search->0); else 0/1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rand-aug-k", type=int, default=0,
                    help="If >0, augment each frame's candidate pool with "
                         "stratified-random actions covering the whole "
                         "translation-direction sphere (K sectors x gripper "
                         "open/closed), scored by the SAME verifier. See "
                         "scriptsv2/eval_search/random_action_aug.py.")
    args = ap.parse_args()

    import zarr

    device = torch.device(args.device)
    use_ema = bool(args.use_ema) if args.use_ema in (0, 1) else (args.policy_kind == "bc")

    policy, verifier, To, chunk_h, image_key, known = build_policy(
        args.ckpt, args.policy_kind, device, use_ema,
        args.n_samples, args.n_action_steps, args.instruction,
        dqc_text_emb=args.dqc_text_emb)

    # Optional stratified-random candidate augmentation (same module the online
    # eval uses, so the two pools are constructed identically).
    aug_cfg = None
    if int(args.rand_aug_k) > 0:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval_search"))
        from random_action_aug import AugConfig, augment as _rand_augment_fn  # noqa: E402
        globals()["_rand_augment"] = _rand_augment_fn
        aug_cfg = AugConfig(
            k=int(args.rand_aug_k),
            gripper=os.environ.get("RAND_AUG_GRIPPER", "both").lower(),
            only=os.environ.get("RAND_AUG_ONLY", "0") == "1",
            mag_min=float(os.environ.get("RAND_AUG_MAG_MIN", "0.25")),
            batch=int(os.environ.get("RAND_AUG_BATCH", "64")),
        )
        print(f"[offline] random augmentation ON: K={aug_cfg.k} -> M={aug_cfg.m} "
              f"extra candidates/frame (gripper={aug_cfg.gripper}, only={aug_cfg.only})")

    z = zarr.open(args.zarr, "r")
    zobs = z["data"]["obs"]
    if image_key not in list(zobs.keys()):
        raise SystemExit(f"{args.zarr} obs has no '{image_key}' (keys={list(zobs.keys())})")
    acts = z["data"]["actions"]
    A = int(acts.shape[-1])
    dones = z["data"]["dones"][:]
    rewards = z["data"]["rewards"][:]
    ep_ends = np.where(dones)[0]
    ep_starts = np.concatenate([[0], ep_ends[:-1] + 1])
    ep_success = rewards[ep_ends] > 0
    img_hw = tuple(np.asarray(zobs[image_key][0]).shape[:2])

    cam_K, cam_T_wc, base_pose = capture_camera(args.task, img_hw)

    sel_eps = pick_episodes(ep_starts, ep_ends, args.num_episodes)
    out_dir = Path(args.out_dir) / "search_q"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[offline] {args.config_label}/{args.split}: {len(sel_eps)} episodes "
          f"from {args.zarr}  (To={To}, chunk_h={chunk_h}, N={args.n_samples}, "
          f"stride={args.stride}) -> {out_dir}")

    for ep in sel_eps:
        s, en = int(ep_starts[ep]), int(ep_ends[ep]) + 1   # [s, en)
        gs = list(range(s, en, args.stride))
        if args.max_frames_per_ep > 0:
            gs = gs[: args.max_frames_per_ep]

        cand_l, val_l, sel_i_l, sel_v_l, ee_l, bt_l = [], [], [], [], [], []
        israndl = []
        astar_l, astar_q_l, astar_rank_l = [], [], []
        for fi, g in enumerate(gs):
            torch.manual_seed(int(args.seed) * 1_000_003 + int(ep) * 1_009 + int(fi))
            obs_dict = build_obs_dict(zobs, g, s, known, image_key, To, device)
            with torch.no_grad():
                actions, values = policy.predict_n_actions(
                    obs_dict, verifier, int(args.n_samples))   # (1,K,H,7),(1,K)
                if aug_cfg is not None:
                    actions, values, is_rand = _rand_augment(
                        policy, obs_dict, actions, values, aug_cfg,
                        seeds=int(args.seed) * 1_000_003 + int(ep) * 1_009 + int(fi),
                        verifier=verifier)
            cand = actions[0, :, :chunk_h].detach().cpu().float().numpy()  # (K,chunk_h,7)
            vals = values[0].detach().cpu().float().numpy()                # (K,)
            israndl.append(
                is_rand[0].detach().cpu().numpy() if aug_cfg is not None
                else np.zeros(vals.shape[0], bool))

            n = min(chunk_h, en - g)
            a_star = np.full((chunk_h, A), np.nan, np.float32)
            a_star[:n] = np.asarray(acts[g:g + n], np.float32)
            img_u8 = np.asarray(zobs[image_key][g], np.uint8)
            a_q = score_astar(verifier, image_key, img_u8, a_star, To, device)
            a_rank = int((vals > a_q).sum())

            cand_l.append(cand.astype(np.float32))
            val_l.append(vals.astype(np.float32))
            sel = int(np.argmax(vals))
            sel_i_l.append(sel); sel_v_l.append(float(vals[sel]))
            ee_l.append(np.asarray(zobs["end_effector_pose"][g], np.float32)[:3])
            bt_l.append(int(g - s))
            astar_l.append(a_star)
            astar_q_l.append(np.float32(a_q)); astar_rank_l.append(np.int32(a_rank))

        if not cand_l:
            continue
        frames = np.stack([np.asarray(zobs[image_key][g], np.uint8) for g in gs])
        npz_path = out_dir / f"ep{int(ep):03d}_{args.split}__N{args.n_samples}.npz"
        np.savez_compressed(
            npz_path,
            seed=np.int32(int(ep)), ep_idx=np.int32(int(ep)),
            success=np.int32(int(ep_success[ep])), truncated=np.int32(0),
            num_steps=np.int32(en - s),
            mode="argmax", max_actions=np.int32(int(args.n_samples)),
            branch_t=np.asarray(bt_l, np.int32),
            candidate_actions=np.stack(cand_l),          # (R,K,chunk_h,7)
            values=np.stack(val_l),                      # (R,K)
            candidate_is_random=np.stack(israndl),       # (R,K) bool
            rand_aug_k=np.int32(int(args.rand_aug_k)),
            selected_index=np.asarray(sel_i_l, np.int32),
            selected_value=np.asarray(sel_v_l, np.float32),
            ee_xyz=np.stack(ee_l),                       # (R,3)
            frames=frames,                               # (R,H,W,3)
            cam_K=cam_K, cam_T_wc=cam_T_wc, base_pose_world=base_pose,
            a_star=np.stack(astar_l),                    # (R,chunk_h,7)
            a_star_q=np.asarray(astar_q_l, np.float32),  # (R,)
            a_star_rank=np.asarray(astar_rank_l, np.int32),
            split=str(args.split), config_label=str(args.config_label),
            chunk_horizon=np.int32(chunk_h), n_action_steps=np.int32(args.n_action_steps),
        )
        print(f"[offline]   ep{int(ep):03d} ({'succ' if ep_success[ep] else 'fail'}): "
              f"{len(gs)} frames -> {npz_path.name}")

    print(f"[offline] done -> {out_dir}")


if __name__ == "__main__":
    main()
