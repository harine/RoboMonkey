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
import hashlib
import json
import os
import platform
import subprocess
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

# ONE seeding path. Episode seeds come from the split manifest (config/splits/simpler/), the
# per-replan seed is SeedSequence([episode_seed, t]), and the policy draws obs-corruption
# noise + every candidate's noise from its per-row stream keyed on that seed
# (SearchProcedureMixin.set_sample_seeds), identically in the single-env and tiled paths.
_SEED_KEY_SELECT = 1

from diffusion_policy.common.dataset_provenance import load_split_file  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]


def _resolve_split_file(path: str) -> Path:
    """A split_file as written in a checkpoint cfg: absolute, or relative to the repo or
    the diffusion_policy submodule (hydra's original cwd at training time)."""
    p = Path(path)
    for cand in ((p,) if p.is_absolute() else (p, _REPO / p, _REPO / "diffusion_policy" / p)):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"split_file not found: {path}")


def _replan_seed(episode_seed: int, t: int) -> int:
    return int(np.random.SeedSequence([int(episode_seed), int(t)]).generate_state(1)[0])


def _selection_generator(replan_seed: int) -> torch.Generator:
    """CPU generator for the softmax pick, on its own stream so selection never shifts the
    candidate noise."""
    g = torch.Generator()
    g.manual_seed(int(np.random.SeedSequence(
        [int(replan_seed), _SEED_KEY_SELECT]).generate_state(1)[0]))
    return g


# Every environment variable that changes what this script measures, with the default the
# code falls back to. This is the single place that enumerates them; `_write_run_config`
# records the EFFECTIVE value of each one into the output dir so a result is traceable to
# the configuration that produced it.
#
# These belong in a config file, not the environment -- a number produced here otherwise
# depends on the shell it ran in, and nothing in the output says which. Recording them is
# the first step; moving them into Hydra is tracked in diffusion_policy/AUDIT.md (P1-7).
#
# NOTE on `N_SAMPLES` (= N in best-of-N, the candidates drawn per control step). It is read
# only on the BC branch, in an if/else on the verifier, so its default is verifier-dependent:
# 16 for VERIFIER_TYPE=dqc, 128 otherwise. The non-DQC branch is `max(128, N_SAMPLES)` -- a
# FLOOR, not a default -- so with the RoboMonkey/LLaVA verifier, N_SAMPLES=4 silently yields
# 128 candidates. Currently latent (every small-N sweep sets VERIFIER_TYPE=dqc), but it would
# silently flatten a BoN curve. The native search path takes N from --n-samples instead.
EVAL_ENV_VARS = (
    ("VERIFIER_TYPE", "robomonkey"),
    ("VERIFIER_IMAGE_KEY", "agentview_image"),
    ("VERIFIER_INSTRUCTION", "put the eggplant in the basket"),
    ("DQC_CKPT_DIR", None),
    ("DQC_VERIFIER_CKPT", None),
    ("DQC_VERIFIER_EPOCH", "30000"),
    ("DQC_CRITIC_MODULE", "chunk"),
    ("DQC_TEXT_EMB", None),
    ("N_SAMPLES", "(BC branch only; 16 if VERIFIER_TYPE=dqc, else floored at 128)"),
    ("BC_SAMPLER", "ddim"),
    ("BC_SAMPLER_STEPS", None),
    ("BC_BATCH_CANDIDATES", "0"),
    ("SOFTMAX_NORM", "none"),
    ("BC_VERIFIER_SCORE", "chunk"),
    ("BC_VERIFIER_AGG", "mean"),
    ("BC_CHUNK_FROM_PRED", "0"),
    ("BC_VERIFIER_STEP", "0"),
    ("BC_VERIFIER_DISCOUNT", "0.98"),
    ("EVAL_CHUNK_SELECT", "0"),
    ("ROBOMONKEY_KV_PREFIX", "0"),
    ("ROBOMONKEY_KV_CHUNK", "16"),
    ("VERITAS_CAMERA_FILE", None),
    ("VERITAS_PLAN_DIR", None),
    ("VERITAS_ALLOW_LIVE", "0"),
    ("VERITAS_PROMPT", "v1_upstream"),
    ("VERITAS_ACTION_POS_SCALE", "null"),
    ("VERITAS_LOOKAHEAD_K", "3"),
    ("VERITAS_LOOKAHEAD_ALPHA", "0.5"),
    ("VERITAS_HORIZON", None),
    ("VERITAS_PLAN_DONE_MODE", "zero"),
    ("BC_CANDIDATE_POOL", None),
    ("BC_GRIPPER_AUG", None),
    ("STRICT_TASK_WEIGHT", None),
    ("FULL_FILTER_LOOKAHEAD", None),
    ("SIM_LOOKAHEAD_MODE", None),
    ("METRIC_PILOT_ROOT", None),
    ("METRIC_TRACE_DIR", None),
    ("METRIC_VERIFIER_REVISION", "v1"),
    ("METRIC_CALIBRATION", None),
    ("VERITAS_GRIPPER_WEIGHT", None),
    ("SUBTASK_RECIPE", None),
    ("BON_SCORE", "veritas"),
    ("BON_TRACE_DIR", None),
)


def _build_veritas_verifier(existing=None, n_action_steps=None):
    """VeritasPixelVerifier from the VERITAS_* environment (see EVAL_ENV_VARS)."""
    from diffusion_policy.policy.veritas_verifier import VeritasPixelVerifier
    # Preserve checkpoint normalization and other settings unless explicitly overridden.
    keys = ('camera_file', 'action_pos_scale', 'lookahead_k', 'lookahead_alpha',
            'default_tol_px', 'horizon', 'ee_key', 'targets_key', 'image_obs_key',
            'normalize', 'value_mean', 'value_std', 'unprojectable_cost', 'plan_done_mode')
    kwargs = {k: getattr(existing, k) for k in keys if hasattr(existing, k)}
    if n_action_steps is not None:
        kwargs['horizon'] = int(n_action_steps)
    for env, key, cast in (
            ('VERITAS_CAMERA_FILE', 'camera_file', str),
            ('VERITAS_LOOKAHEAD_K', 'lookahead_k', int),
            ('VERITAS_LOOKAHEAD_ALPHA', 'lookahead_alpha', float),
            ('VERITAS_HORIZON', 'horizon', int),
            ('VERITAS_PLAN_DONE_MODE', 'plan_done_mode', str)):
        if os.environ.get(env):
            kwargs[key] = cast(os.environ[env])
    if 'VERITAS_ACTION_POS_SCALE' in os.environ:
        scale = os.environ['VERITAS_ACTION_POS_SCALE'].strip()
        kwargs['action_pos_scale'] = (None if scale.lower() in ('', 'null', 'none')
                                      else json.loads(scale))
    if os.environ.get('VERIFIER_TYPE') in ('veritas_pixel_gripper','veritas_subtask'):
        from diffusion_policy.policy.veritas_gripper_verifier import VeritasGripperVerifier
        return VeritasGripperVerifier(**kwargs,health_check=False,
            gripper_weight=float(os.environ['VERITAS_GRIPPER_WEIGHT']),
            task_weight=float(os.environ.get('STRICT_TASK_WEIGHT','0')))
    if os.environ.get('VERITAS_STACK_POSITION') == '1' or type(existing).__name__ == 'VeritasStackPositionVerifier':
        from diffusion_policy.policy.veritas_stack_position_verifier import VeritasStackPositionVerifier
        for key in ('stack_weight','position_sigma','cube_height','grasp_radius','pixel_scale'):
            if hasattr(existing,key):kwargs[key]=getattr(existing,key)
        return VeritasStackPositionVerifier(**kwargs,health_check=False)
    if os.environ.get('VERIFIER_TYPE') == 'veritas_metric':
        from diffusion_policy.policy.veritas_metric_verifier import VeritasMetricVerifier
        revision=os.environ.get('METRIC_VERIFIER_REVISION','v1')
        calibration=None
        if revision=='v2':
            calibration=json.loads(Path(os.environ['METRIC_CALIBRATION']).read_text())
            kwargs['action_pos_scale']=calibration['action_pos_scale']
        return VeritasMetricVerifier(**kwargs, revision=revision, calibration=calibration, health_check=False)
    return VeritasPixelVerifier(**kwargs, health_check=True)


def _veritas_env_wrapper(task: str):
    """A callable wrapping a SimplerEnv gym env in VeritasTrackerWrapper, planning from
    the VERITAS_PLAN_DIR cache (live Gemini only with VERITAS_ALLOW_LIVE=1)."""
    from diffusion_policy.env.simpler.veritas.camera import CameraModel
    from diffusion_policy.env.simpler.veritas.tracker import VeritasTrackerWrapper
    from diffusion_policy.veritas.plan_store import CachedPlanProvider
    camera = CameraModel.load(os.environ["VERITAS_CAMERA_FILE"])
    plan_dir = os.environ["VERITAS_PLAN_DIR"]
    allow_live = os.environ.get("VERITAS_ALLOW_LIVE", "0") == "1"
    planner = None
    hint_provider = None
    # VERITAS_HINTS: sim (object pixels from projected sim poses) | gsam2 (upstream: Gemini
    # picks the objects, Grounded-SAM-2 localizes them, Gemini bboxes fill gaps; implies the
    # exact upstream prompt formatting)
    hints = os.environ.get("VERITAS_HINTS", "sim")
    if hints not in ("sim", "gsam2"):
        raise ValueError(f"VERITAS_HINTS={hints!r} (want sim|gsam2)")
    if allow_live:
        from diffusion_policy.env.simpler.veritas.simpler_client import (
            SimplerVeritasClient, make_live_planner)
        client = SimplerVeritasClient()
        prompt = os.environ.get("VERITAS_PROMPT",
                                "v1_upstream_exact" if hints == "gsam2" else "v1_upstream")
        planner = make_live_planner(
            client, prompt=prompt, log_dir=plan_dir, task=task,
            camera_file=os.environ["VERITAS_CAMERA_FILE"], hint_mode=hints)
        if hints == "gsam2":
            from diffusion_policy.env.simpler.veritas.gsam.upstream_hints import \
                UpstreamHintProvider
            hint_provider = UpstreamHintProvider(client, log_dir=plan_dir)
    elif hints == "gsam2":
        raise ValueError("VERITAS_HINTS=gsam2 needs VERITAS_ALLOW_LIVE=1 (hints feed live planning)")
    provider = CachedPlanProvider(plan_dir, planner=planner, allow_live=allow_live)
    print(f"[eval] veritas object hints: {hints}")
    return lambda env: VeritasTrackerWrapper(env, camera, provider, hint_provider=hint_provider)


def _git_info(repo_root: Path) -> Dict[str, Any]:
    """Commit sha + dirty flag for a repo, or an error string. Never raises."""
    def _run(*cmd):
        return subprocess.run(cmd, cwd=str(repo_root), capture_output=True,
                              text=True, timeout=15).stdout.strip()
    try:
        return {
            "sha": _run("git", "rev-parse", "HEAD"),
            "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            # a dirty tree means the sha does not identify the code that ran
            "dirty": bool(_run("git", "status", "--porcelain")),
        }
    except Exception as e:  # missing git, not a repo, timeout
        return {"error": f"{type(e).__name__}: {e}"}


def _resume_identity(args: argparse.Namespace) -> Dict[str, Any]:
    """What must match for a stored episode to be reused on resume."""
    return {
        "tiled_envs": int(os.environ.get("TILED_ENVS", "1")),
        "obs_corruption": str(args.obs_corruption),
        # absent from records written before the flag existed; None == clean there too
        **({"eval_obs_noise_level": float(args.eval_obs_noise_level)}
           if args.eval_obs_noise_level is not None else {}),
        **({"veritas_hints": os.environ["VERITAS_HINTS"]}
           if os.environ.get("VERITAS_HINTS", "sim") != "sim" else {}),
        **({"metric_verifier_revision": os.environ['METRIC_VERIFIER_REVISION'],
            "metric_calibration": os.environ.get('METRIC_CALIBRATION')}
           if os.environ.get('METRIC_VERIFIER_REVISION','v1') != 'v1' else {}),
        **({'pixel_gripper_version':'v1_normalized','gripper_weight':float(os.environ['VERITAS_GRIPPER_WEIGHT'])}
           if os.environ.get('VERIFIER_TYPE')=='veritas_pixel_gripper' else {}),
    }


def _write_run_config(output_dir: Path, args: argparse.Namespace,
                      extra: Optional[Dict[str, Any]] = None) -> Path:
    """Record the full configuration of this evaluation next to its results.

    Without this an eval result is not reproducible and not even interpretable: the
    behaviour of this script is driven by ~20 environment variables (which verifier
    checkpoint to load, how candidates are scored and selected, how many are drawn) that
    are recorded nowhere, so two rows in the same results table can have been produced by
    materially different procedures with nothing to distinguish them.

    Writes ``<output_dir>/run_config.json``. Env vars are recorded as their EFFECTIVE
    value together with whether it came from the environment or from the code default, so
    the record is unambiguous either way.
    """
    ckpt = Path(args.checkpoint)
    ckpt_info: Dict[str, Any] = {"path": str(ckpt)}
    try:
        st = ckpt.stat()
        ckpt_info.update(size_bytes=st.st_size, mtime=int(st.st_mtime))
        # cheap identity for the checkpoint: hashing multi-GB files every eval is not
        # worth it, so fingerprint the head instead -- enough to catch "same path,
        # different file" between runs.
        with open(ckpt, "rb") as fh:
            ckpt_info["head_md5"] = hashlib.md5(fh.read(1 << 20)).hexdigest()
    except OSError as e:
        ckpt_info["error"] = str(e)

    env: Dict[str, Any] = {}
    for name, default in EVAL_ENV_VARS:
        raw = os.environ.get(name)
        env[name] = {"value": raw if raw is not None else default,
                     "source": "env" if raw is not None else "default"}

    repo_root = Path(__file__).resolve().parents[2]
    payload = {
        "script": str(Path(__file__).resolve()),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv,
        "args": vars(args),
        "checkpoint": ckpt_info,
        "env": env,
        "git": {"RoboMonkey": _git_info(repo_root),
                "diffusion_policy": _git_info(repo_root / "diffusion_policy")},
        "slurm": {k: v for k, v in os.environ.items() if k.startswith("SLURM_JOB")},
        "host": platform.node(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_device": (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else None),
    }
    if extra:
        payload.update(extra)

    path = output_dir / "run_config.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[eval] wrote run config -> {path}")
    return path


@torch.no_grad()
def _rescore_veritas_bon(pred, verifier, obs_dict, bon_score, sim_look, env_base, source_obj,
                         step_obs, goal_d0, device, seed, t):
    """Rescore the BC candidates for a BON_SCORE arm and reselect by argmax.

    Pixel term: Veritas lookahead cost / 105 px, bounded to [0, 1) as c / (1 + c).
    Distance term: ||source - target|| (world xyz) / d0, d0 = initial distance (>= 5 cm).
    """
    acts = torch.as_tensor(np.asarray(pred['candidate_actions']), device=device, dtype=torch.float32)
    n = acts.shape[0]
    rep = {k: v.expand(n, *v.shape[1:]) for k, v in obs_dict.items()}
    tcp = torch.as_tensor(np.asarray(step_obs['end_effector_pose'])[:3], device=device, dtype=torch.float32)
    src = np.asarray(step_obs['insertive_asset_pose'], dtype=np.float64)[:3]
    tgt = np.asarray(step_obs['receptive_asset_pose'], dtype=np.float64)[:3]
    sim = None
    if sim_look is not None:
        sim = sim_look.score(pred['candidate_actions'])
        # Encode the simulated endpoint as one displacement so the Veritas cost reads it
        # instead of the integrated action translation.
        R, scale, _, _ = verifier._consts(acts)
        endpoints = torch.tensor([x['eef_endpoint'] for x in sim], device=device, dtype=torch.float32)
        scored = torch.zeros_like(acts)
        scored[:, 0, :3] = ((endpoints - tcp) @ R) / scale
    else:
        scored = acts
    with torch.no_grad():
        pixel = verifier.cost(rep, scored) / 105.
    pixel = (pixel / (1 + pixel)).cpu().numpy().astype(float)
    dist = np.zeros(n)
    if bon_score == 'sim_veritas_distance':
        dist = np.array([np.linalg.norm(np.asarray(x['object_endpoint']) - tgt) for x in sim]) / goal_d0
    elif bon_score == 'veritas_distance':
        # Held source (grasped now, gripper closed over the whole chunk) follows the
        # predicted end-effector displacement; otherwise it stays where it is.
        ee_end = verifier.endpoints(rep, acts).cpu().numpy().astype(float)
        held = bool(env_base.agent.check_grasp(source_obj))
        closed = (np.asarray(pred['candidate_actions'])[:, :, 6] < .5).all(axis=1)
        moved = ee_end - tcp.cpu().numpy()
        src_end = src[None] + np.where((held & closed)[:, None], moved, 0.)
        dist = np.linalg.norm(src_end - tgt[None], axis=1) / goal_d0
    values = -pixel - dist
    best = int(values.argmax())
    pred = dict(pred, values=values.tolist(), selected_index=best, selected_value=float(values[best]),
                chunk=acts[best][None])
    trace_dir = os.environ.get('BON_TRACE_DIR')
    if trace_dir:
        row = dict(t=int(t), bon_score=bon_score, pixel=pixel.tolist(), distance=dist.tolist(),
                   values=values.tolist(), selected_index=best)
        if sim is not None:
            row['sim_success'] = [bool(x.get('checks', {}).get('success', False)) for x in sim]
            row['restore_error'] = sim_look.validation_errors[-1]
        with (Path(trace_dir) / f'seed{seed}.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
    return pred


def _search_policy_predict(
    policy,
    obs_dict: Dict[str, torch.Tensor],
    mode: str = "argmax",
    viz_q: bool = False,
    n_samples: Optional[int] = None,
    softmax_temp: float = 1.0,
    replan_seed: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Run the search policy.

    Runs the search loop (``n_samples`` autoregressive candidate samples, each
    verifier-scored), then selects ONE of those ``n_samples`` candidates. Three
    selection modes:
      * ``"argmax"``: pick the candidate with the highest verifier value.
      * ``"final"``: pick the final candidate, preserving all earlier search feedback.
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
    # Pose-based verifiers (Veritas) read low-dim keys the policy may not normalize,
    # plus the per-step waypoint targets the env wrapper injects.
    verifier_keys = set(getattr(policy.verifier, "obs_keys", ()) or ())
    obs_dict = {
        k: v for k, v in obs_dict.items()
        if k in known or k == image_obs_key or k in verifier_keys
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
    # Per-env replan seeds (one per batch row). Policies with per-row streams draw every
    # noise from them; the rest (Gaussian ST, BC wrapper) fall back to the global RNG
    # seeded from row 0.
    _pred_kw: Dict[str, Any] = {}
    _per_env_seeds = None if replan_seed is None else [int(x) for x in replan_seed]
    _policy_seeded = (_per_env_seeds is not None
                      and getattr(policy, "supports_sample_seeds", False))
    if _policy_seeded:
        policy.set_sample_seeds(_per_env_seeds)
    elif _per_env_seeds is not None:
        torch.manual_seed(_per_env_seeds[0])
    if getattr(policy, "eval_obs_noise_level", None) is not None:
        # BC wrapper's eval-only state noise: its own stream keyed on the replan seed
        _pred_kw["obs_noise_seed"] = None if _per_env_seeds is None else _per_env_seeds[0]
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
        if _policy_seeded:
            policy.set_sample_seeds(None)
        if _kv:
            _verifier.set_kv_prefix(False)
            _verifier.clear_prefix_cache()
    if _os.environ.get("BC_GRIPPER_AUG", "0") == "1":
        if type(policy).__name__ != "BCSearchWrapper" or To != 1 or _aug.enabled:
            raise ValueError("gripper diagnostic requires BC, one observation, and no random augmentation")
        from diffusion_policy.policy.gripper_candidate_aug import augment_gripper_candidates
        actions=augment_gripper_candidates(actions)
        assert actions.shape[0]==1
        rep={k:v.expand(actions.shape[1],*v.shape[1:]) for k,v in obs_dict.items()}
        values=_verifier.get_value(rep,actions[0]).unsqueeze(0)
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
        out['candidate_actions_full'] = actions[0].detach().cpu().float().numpy()
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

    if mode in ("argmax", "final"):
        best = (values.argmax(dim=1) if mode == "argmax" else
                torch.full((B,), actions.shape[1] - 1, device=actions.device, dtype=torch.long))
        best_action = actions[torch.arange(B, device=actions.device), best]  # (B, horizon, action_dim)
        out["chunk"] = best_action[:, start:end]
        out["selected_index"] = int(best[0].item())
        out["selected_value"] = float(values[0, int(best[0].item())].item())
        return _mark_random(out, best)

    if mode == "softmax":
        # Sample a candidate index from softmax(values / temp). temp -> 0 is
        # argmax, temp -> inf is uniform. Default temp=1.0.
        T = max(float(softmax_temp), 1e-6)
        # SOFTMAX_NORM=zscore standardizes the K values within each replan (per row)
        # before the temperature, so T is scale-free across verifiers, tasks and plans.
        # A row whose values are all equal becomes uniform (std floored).
        _norm = _os.environ.get("SOFTMAX_NORM", "none").lower()
        if _norm == "zscore":
            _v = values.float()
            logits = (_v - _v.mean(dim=1, keepdim=True)) / _v.std(
                dim=1, unbiased=False, keepdim=True).clamp_min(1e-8)
        elif _norm == "none":
            logits = values
        else:
            raise ValueError(f"SOFTMAX_NORM must be none|zscore, got {_norm!r}")
        probs = torch.softmax(logits / T, dim=1)  # (B, K)
        # Seeded per env from its own replan seed (own CPU stream), so the pick is
        # reproducible and independent of batch position.
        _probs_cpu = probs.detach().float().cpu()
        if _per_env_seeds is not None:
            sampled = torch.stack([
                torch.multinomial(_probs_cpu[_k:_k + 1], num_samples=1,
                                  generator=_selection_generator(_per_env_seeds[_k])).squeeze(1)[0]
                for _k in range(_probs_cpu.shape[0])])
        else:
            sampled = torch.multinomial(_probs_cpu, num_samples=1).squeeze(1)  # (B,)
        sampled = sampled.to(actions.device)
        chosen_action = actions[torch.arange(B, device=actions.device), sampled]
        out["chunk"] = chosen_action[:, start:end]
        out["selected_index"] = int(sampled[0].item())
        out["selected_value"] = float(values[0, int(sampled[0].item())].item())
        out["softmax_temp"] = float(T)
        out["softmax_norm"] = _norm
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
                    softmax_temp: float = 1.0) -> Dict[str, Any]:
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

    full_filter=None
    if os.environ.get('FULL_FILTER_LOOKAHEAD')=='1':
        from full_filter_lookahead import FullFilterLookahead
        full_filter=FullFilterLookahead(env_base,source_obj,target_obj,convert_maniskill,seed)
    # BON_SCORE: Veritas BoN arms that rescore the BC candidates.
    #   veritas_distance      -bounded pixel cost - predicted ||src-tgt||/d0 (held source follows the gripper)
    #   sim_veritas           -bounded pixel cost of the simulated (cloned-scene) endpoint
    #   sim_veritas_distance  sim_veritas - simulated ||src-tgt||/d0
    # Unset / 'veritas' keeps the verifier's own argmax.
    bon_score = os.environ.get('BON_SCORE', 'veritas') or 'veritas'
    if bon_score not in ('veritas', 'veritas_distance', 'sim_veritas', 'sim_veritas_distance'):
        raise ValueError(f'BON_SCORE={bon_score!r}')
    sim_look = None
    if bon_score.startswith('sim_'):
        from full_filter_lookahead import FullFilterLookahead
        sim_look = FullFilterLookahead(env_base, source_obj, target_obj, convert_maniskill, seed,
                                       task=os.environ['TASK'])
    goal_d0 = None
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

    metric_tracker = None
    metric_provider = None
    pixel_gripper = type(policy.verifier).__name__ == 'VeritasGripperVerifier'
    subtask_mode = os.environ.get('VERIFIER_TYPE') == 'veritas_subtask'
    if subtask_mode:
        from diffusion_policy.env.simpler.veritas.subtask_controller import SubtaskController
        policy.verifier.camera.assert_matches(obs)
        if os.environ.get('SUBTASK_RECIPE')=='two_physical_v2':
            from diffusion_policy.env.simpler.veritas.subtask_two_controller import TwoSubtaskController
            SubtaskController=TwoSubtaskController
        metric_tracker=SubtaskController(os.environ['METRIC_PILOT_ROOT'],os.environ['METRIC_TRACE_DIR'],seed,instruction,policy.verifier.camera)
        if 'eggplant' in instruction.lower():
            from diffusion_policy.env.simpler.veritas.subtask_eggplant_controller import EggplantSubtaskController,BasketPositionChecks
            metric_tracker=EggplantSubtaskController(os.environ['METRIC_PILOT_ROOT'],os.environ['METRIC_TRACE_DIR'],seed,instruction,policy.verifier.camera)
            metric_tracker.checks=BasketPositionChecks(env_base.episode_source_obj_bbox_world,env_base.episode_target_obj_bbox_world)
        (Path(os.environ['METRIC_TRACE_DIR'])/f'seed{seed}.jsonl').write_text('')
    if pixel_gripper and not subtask_mode:
        from diffusion_policy.env.simpler.veritas.pixel_gripper_plan import PixelGripperPlanProvider
        from diffusion_policy.policy.veritas_gripper_verifier import PixelGripperTracker
        from diffusion_policy.env.simpler.veritas.hints import object_hints
        policy.verifier.camera.assert_matches(obs)
        metric_provider=PixelGripperPlanProvider(os.environ['METRIC_PILOT_ROOT'],instruction,
                                               os.environ['METRIC_TRACE_DIR'])
        if os.environ.get('SIM_LOOKAHEAD_MODE') in ('trajectory_distance','trajectory_full_filter'):
            metric_provider.budget_group='simple_sim_distance_eggplant_20261004'
            metric_provider.group_cap_usd=10.
        metric_provider.context='Native-pixel spatial hints: '+json.dumps(
            dict(eef=policy.verifier.camera.project(np.asarray(obs['extra']['tcp_pose'])[:3]).tolist(),
                 objects=object_hints(env_base,policy.verifier.camera)))
    if type(policy.verifier).__name__ == 'VeritasMetricVerifier':
        from diffusion_policy.env.simpler.veritas.metric_plan import PlanProvider
        from diffusion_policy.policy.veritas_metric_verifier import MetricTracker
        policy.verifier.camera.assert_matches(obs)
        metric_provider = PlanProvider(os.environ['METRIC_PILOT_ROOT'], instruction,
                                       os.environ['METRIC_TRACE_DIR'])
    init_state = None   # captured at the first step: asset + arm start poses
    obj_goal_dist = float("nan")       # source-object -> target distance, latest step
    strict_first_grasp_distance = None
    strict_basket_arrived = False
    obj_goal_dist_min = float("inf")   # closest the object ever got to the goal
    while t < max_steps:
        step_obs = build_obs_step(
            env=env_base, obs=obs,
            prev_arm_action=prev_arm_action,
            prev_gripper_action=prev_gripper_action,
            source_obj=source_obj, target_obj=target_obj, tcp_link=tcp_link,
        )
        window.push(step_obs)
        if full_filter is not None:
            full_filter.observe(bool(prev_gripper_action[0]>=.5) if t else True,success)
        # object -> goal distance (eggplant->basket, cube->cube, spoon->towel):
        # ||insertive_asset - receptive_asset|| in world xyz, tracked each step.
        _src = step_obs.get("insertive_asset_pose", None)
        _tgt = step_obs.get("receptive_asset_pose", None)
        if _src is not None and _tgt is not None:
            obj_goal_dist = float(np.linalg.norm(
                np.asarray(_src, dtype=np.float32)[:3] - np.asarray(_tgt, dtype=np.float32)[:3]))
            obj_goal_dist_min = min(obj_goal_dist_min, obj_goal_dist)
            if goal_d0 is None:
                goal_d0 = max(obj_goal_dist, .05)
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

        metric_state = None
        if subtask_mode:
            metric_state=metric_tracker.observe(current_frame,t,step_obs['end_effector_pose'],
                step_obs['insertive_asset_pose'],step_obs['receptive_asset_pose'],
                bool(prev_gripper_action[0]>=.5) if t else True,can_request=not pending_actions)
            if getattr(policy.verifier,'task_weight',0):
                contact=bool(env_base.agent.check_grasp(source_obj))
                c=metric_tracker.metrics['position_checks']
                if contact and strict_first_grasp_distance is None:
                    strict_first_grasp_distance=c['tcp_source_distance_m']
                strict_basket_arrived |= bool(c['basket_center_inside'] and c['basket_height_valid'])
                metric_tracker.metrics['strict_grasp_contact']=contact
                metric_tracker.metrics['strict_first_grasp_distance']=strict_first_grasp_distance
                metric_tracker.metrics['strict_basket_arrived']=strict_basket_arrived
                strict_log=dict(t=t,contact=contact,gripper_open=c['gripper_open'],
                    gripper_object_distance=c['tcp_source_distance_m'],object_target_distance=c['source_target_distance_m'],
                    grasp_reference_distance=strict_first_grasp_distance,basket_arrived=strict_basket_arrived)
                with (Path(os.environ['METRIC_TRACE_DIR'])/f'seed{seed}_strict_metric_steps.jsonl').open('w' if t==0 else 'a') as f:
                    f.write(json.dumps(strict_log)+'\n')
            if metric_tracker.complete or metric_tracker.budget_exhausted:break
        if metric_provider is not None:
            if metric_tracker is None:
                plan=metric_provider.get(seed,current_frame,instruction)
                if pixel_gripper:
                    metric_tracker=PixelGripperTracker(plan,policy.verifier.camera,step_obs['end_effector_pose'][:3])
                else:
                    metric_tracker = MetricTracker(plan,
                        revision=policy.verifier.revision, calibration=policy.verifier.calibration)
                # A retry/resume starts a new actual rollout; do not append its
                # traces to a previously interrupted rollout of the same seed.
                (Path(os.environ['METRIC_TRACE_DIR'])/f'seed{seed}.jsonl').write_text('')
            # Use gripper proprioception after reset, then the executed gripper command.
            # Possession additionally needs observed object lift and co-motion.
            opened = bool(prev_gripper_action[0] >= .5) if t else True
            metric_state = metric_tracker.update(step_obs['end_effector_pose'],
                step_obs['insertive_asset_pose'], step_obs['receptive_asset_pose'], opened)
        if not pending_actions:
            obs_dict = window.to_tensor_dict(device=device)
            if getattr(policy.verifier,'task_weight',0):
                if not subtask_mode or 'eggplant' not in instruction.lower():raise ValueError('strict metric requires eggplant subtask controller')
                c=metric_tracker.metrics['position_checks']
                d0=float(np.linalg.norm(np.asarray(init_state['ee_xyz'])-np.asarray(init_state['insertive_xyz'])))
                task_state=np.r_[step_obs['insertive_asset_pose'][:3],step_obs['receptive_asset_pose'][:3],float(contact),d0,strict_first_grasp_distance or 0.,float(strict_basket_arrived),float(strict_first_grasp_distance is not None)]
                obs_dict['strict_task_state']=torch.as_tensor(task_state,dtype=torch.float32,device=device)[None,None]
            if metric_state is not None:
                if pixel_gripper:
                    obs_dict['veritas_targets']=torch.as_tensor(metric_state[:8],device=device)[None,None]
                    obs_dict['veritas_gripper_targets']=torch.as_tensor(metric_state[8:],device=device)[None,None]
                else:
                    obs_dict['veritas_metrics'] = torch.as_tensor(metric_state,device=device)[None,None]

            # Verifier expects (B, To, H, W, 3). Repeat the current frame
            # across the obs window (only the last step is read by the
            # verifier anyway).
            assert current_frame is not None
            img_t = torch.from_numpy(current_frame).to(device=device)
            obs_dict["agentview_image"] = img_t.unsqueeze(0).unsqueeze(0).expand(
                1, n_obs_steps, *current_frame.shape
            )
            if hasattr(env, "targets_vector"):
                # Veritas: the live tracker's upcoming waypoints, advanced on executed
                # steps only (VeritasTrackerWrapper), repeated over the obs window.
                _tv = torch.from_numpy(np.asarray(env.targets_vector(), dtype=np.float32))
                obs_dict["veritas_targets"] = _tv.to(device=device)[None, None].expand(
                    1, n_obs_steps, _tv.shape[0])
            # One-row per-env list, same formula as the tiled path, so TILED_ENVS=1 and >1
            # draw the same candidates for an episode.
            pred = _search_policy_predict(
                policy, obs_dict, mode=mode, viz_q=(viz_q or metric_tracker is not None or bon_score != 'veritas'),
                n_samples=n_samples, softmax_temp=softmax_temp,
                replan_seed=[_replan_seed(seed, t)],
            )
            full_results=None
            if bon_score != 'veritas' and n_samples > 1:
                pred = _rescore_veritas_bon(pred, policy.verifier, obs_dict,
                                            bon_score, sim_look, env_base, source_obj, step_obs,
                                            goal_d0, device, seed, t)
            if full_filter is not None:
                full_results=full_filter.score(pred['candidate_actions'])
                trajectory=np.asarray(pred['values'],dtype=float)
                if os.environ.get('SIM_LOOKAHEAD_MODE') in ('trajectory_distance','trajectory_full_filter'):
                    # Encode actual simulated endpoint as a single displacement;
                    # reuse the exact Veritas pixel cost, without action integration error.
                    endpoints=torch.tensor([x['eef_endpoint'] for x in full_results],device=device,dtype=torch.float32)
                    R,scale,_,_=policy.verifier._consts(endpoints)
                    synthetic=torch.zeros((len(full_results),8,7),device=device)
                    tcp=torch.as_tensor(step_obs['end_effector_pose'][:3],device=device)
                    synthetic[:,0,:3]=((endpoints-tcp)@R)/scale
                    rep={k:v.expand(len(full_results),*v.shape[1:]) for k,v in obs_dict.items()}
                    trajectory=policy.verifier.get_value(rep,synthetic).detach().cpu().numpy()
                    if os.environ.get('SIM_LOOKAHEAD_MODE')=='trajectory_full_filter':
                        task_values=np.array([x['score'] for x in full_results])
                    else:
                        d0=max(full_filter.history[0]['deb'],.05)
                        task_values=-np.array([x['final_distance'] for x in full_results])/d0
                        for x,q in zip(full_results,task_values):x['distance_score']=float(q)
                    values=trajectory+task_values
                else:
                    values=trajectory+np.array([x['score'] for x in full_results])
                best=int(values.argmax())
                pred.update(values=values.tolist(),selected_index=best,selected_value=float(values[best]),
                    chunk=torch.as_tensor(pred['candidate_actions'][best],device=device)[None])
                metric_tracker.metrics['full_filter_selected']=full_results[best]
                metric_tracker.metrics['full_filter_restore_error']=full_filter.validation_errors[-1]
                with (Path(os.environ['METRIC_TRACE_DIR'])/f'seed{seed}_full_filter.jsonl').open('w' if t==0 else 'a') as f:
                    f.write(json.dumps(dict(t=t,trajectory_values=trajectory.tolist(),combined_values=values.tolist(),selected_index=best,candidates=full_results))+'\n')
            if metric_tracker is not None:
                acts = torch.as_tensor(pred['candidate_actions'],device=device)
                repeated = {k:v.expand(len(acts), *v.shape[1:]) for k,v in obs_dict.items()}
                components = {k:v.detach().cpu().tolist() for k,v in
                              policy.verifier.components(repeated,acts).items()}
                trace = dict(seed=int(seed), t=int(t), metrics=metric_tracker.metrics,
                    tcp_xyz=step_obs["end_effector_pose"][:3].tolist(),
                    source_xyz=step_obs["insertive_asset_pose"][:3].tolist(),
                    target_xyz=step_obs["receptive_asset_pose"][:3].tolist(),
                    state=metric_state.tolist(), actions=pred['candidate_actions'].tolist(),
                    selected_index=pred['selected_index'], values=pred['values'], components=components)
                with (Path(os.environ['METRIC_TRACE_DIR']) / f'seed{seed}.jsonl').open('a') as f:
                    f.write(json.dumps(trace)+'\n')
            chunk = pred["chunk"][0].detach().cpu().numpy()
            if branches is not None and current_frame is not None:
                # Current EE world-frame xyz (start of every candidate's traj).
                try:
                    tcp_world = np.asarray(obs["extra"]["tcp_pose"], dtype=np.float32)[:3]
                except Exception:
                    tcp_world = np.zeros(3, dtype=np.float32)
                branches.append({
                    "t": int(t),
                    "policy_obs": {k: v.detach().cpu().numpy() for k, v in obs_dict.items()
                                   if v.ndim <= 3},
                    "candidate_actions_full": pred['candidate_actions_full'],
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
            if not subtask_mode and full_filter is None:break
        if bool(trunc):
            truncated = True
            break
        t += 1

    if sim_look is not None:
        sim_look.close()
    if full_filter is not None:
        from full_filter_lookahead import evaluate_filter
        full_filter.observe(bool(prev_gripper_action[0]>=.5),success)
        report=dict(history=full_filter.history,filter=evaluate_filter(full_filter.history),restoration_errors=full_filter.validation_errors,score_variation=full_filter.score_variation)
        (Path(os.environ['METRIC_TRACE_DIR'])/f'seed{seed}_full_filter_outcome.json').write_text(json.dumps(report,indent=2))
        full_filter.close()
    return {
        "success": bool(success),
        "success_while_grasped": bool(success and info.get("is_src_obj_grasped", False)),
        "truncated": bool(truncated),
        "num_steps": len(step_log),
        "subtasks": metric_tracker.metrics if subtask_mode else None,
        "obj_goal_dist_final": obj_goal_dist,
        "obj_goal_dist_min": (None if obj_goal_dist_min == float("inf") else obj_goal_dist_min),
        "steps": step_log,
        "frames": frames,
        "branches": branches,
        "cam_K": cam_K,
        "cam_T_wc": cam_T_wc,
        "base_pose_world": base_pose_world,
        "init_state": init_state,
        "veritas": (env.tracker_status() if hasattr(env, "tracker_status") else None),
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
                           mode="softmax", n_samples=8, softmax_temp=1.0):
    """Vectorized rollout: run len(envs) episodes in LOCKSTEP, batching the policy +
    verifier over the env dimension (B = #active envs) at each replan. Handles variable
    episode length via a per-env done mask (finished envs are frozen, not stepped).
    Fast path: SR + object->goal distance only (no per-step frames/viz). Returns a list
    of per-env result dicts aligned with `seeds`."""
    from simpler_env.utils.env.observation_utils import (  # type: ignore
        get_image_from_maniskill2_obs_dict,
    )
    N = len(envs)
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
            if all(hasattr(envs[k], "targets_vector") for k in need):
                _tv = torch.from_numpy(np.stack(
                    [np.asarray(envs[k].targets_vector(), dtype=np.float32) for k in need]))
                batch["veritas_targets"] = _tv.to(device=device)[:, None].expand(
                    len(need), n_obs_steps, _tv.shape[1])
            # Per-env seed = f(episode seed, t), aligned with `need` = the batch rows, so a
            # row's candidates do not depend on batch position or other envs' done-timing.
            pred = _search_policy_predict(policy, batch, mode=mode, viz_q=False,
                                          n_samples=n_samples, softmax_temp=softmax_temp,
                                          replan_seed=[_replan_seed(seeds[k], t) for k in need])
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
                 veritas=(envs[k].tracker_status() if hasattr(envs[k], "tracker_status") else None),
                 branches=None, frames=[], steps=[], init_state=None) for k in range(N)]


def _select_episode_seeds(args, cfg):
    """(seeds, seed -> raw-shard coordinates, split_info, checkpoint dataset_provenance)."""
    # Episode seeds: the split manifest (or an explicit list). Each seed reproduces one
    # demo's start state, so the same split is comparable across every checkpoint / N.
    # The checkpoint carries the split it trained on (dataset_provenance, stamped by the
    # training workspace); that manifest is the default, and its train seeds are what
    # test seeds must be disjoint from. Seeds, not episode indices: the manifests were
    # remapped onto the split-only datasets, which renumbered episodes but kept seeds.
    from omegaconf import OmegaConf as _OCp
    train_prov = cfg.get("dataset_provenance", None)
    if train_prov is not None:
        train_prov = _OCp.to_container(train_prov, resolve=True)
    ckpt_split_file = _OCp.select(cfg, "task.dataset.split_file", default=None)
    ckpt_split = (train_prov or {}).get("split_manifest")
    split_info: Dict[str, Any] = {"split": None}
    seed_src: Dict[int, Dict[str, Any]] = {}
    if args.seeds:
        explicit_seeds: List[int] = [int(s) for s in args.seeds.split(",") if s.strip()]
        split_info["split"] = "explicit"
    else:
        _sf = args.split_file or (ckpt_split or {}).get("path") or ckpt_split_file
        if _sf is None:
            raise ValueError(
                "no split manifest: the checkpoint has no task.dataset.split_file. Pass "
                "--split-file diffusion_policy/diffusion_policy/config/splits/simpler/<task>/"
                "<rule>_train<N>.json, or --seeds (the archived sub50 demo sets: --seeds "
                "$(jq -r '[.episodes[].seed]|join(\",\")' "
                "scriptsv2/eval_search/demoseeds/<task>_test.json)).")
        _sm = load_split_file(_resolve_split_file(_sf))
        explicit_seeds = list(_sm["seeds"][args.split])
        seed_src = {int(r["seed"]): {"src_shard": r.get("src_shard"),
                                     "src_episode": r.get("src_episode")}
                    for r in (_sm["episodes"][str(i)] for i in _sm[args.split])}
        split_info = {"split": args.split, "manifest": _sm["path"], "manifest_md5": _sm["md5"],
                      "checksum": _sm.get("checksum"), "dataset_dir": _sm.get("dataset_dir"),
                      "zarr": _sm.get("zarr_path"),
                      "rule": (_sm.get("derivation") or {}).get("rule"),
                      "episode_indices": [int(i) for i in _sm[args.split]],
                      "is_ckpt_split": args.split_file is None}
    if args.num_episodes is not None:
        explicit_seeds = explicit_seeds[:int(args.num_episodes)]
    split_info["seeds"] = list(explicit_seeds)
    # The checkpoint's training seeds: stamped > its split_file read now > (pre-manifest
    # checkpoints) the demoseeds train set matched to its dataset_dir.
    train_seeds, train_src = set(), None
    if ckpt_split is not None:
        train_seeds, train_src = set(ckpt_split["train"]["seeds"]), "stamped"
    elif ckpt_split_file is not None:
        train_seeds = set(load_split_file(_resolve_split_file(ckpt_split_file))["seeds"]["train"])
        train_src = "ckpt split_file"
    elif train_prov is not None and train_prov.get("train_split") is not None:
        train_seeds, train_src = set(int(x) for x in train_prov["train_split"]["seeds"]), "demoseeds"
    overlap = sorted(train_seeds & set(explicit_seeds))
    split_info["ckpt_train_seeds_source"] = train_src
    split_info["overlap_with_ckpt_train_seeds"] = overlap
    if split_info["split"] == "test" and overlap:
        raise ValueError(f"--split test but {len(overlap)} seed(s) are in the checkpoint's "
                         f"training demos: {overlap[:10]}")
    return explicit_seeds, seed_src, split_info, train_prov


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-file", type=str, default=None,
                        help="Split manifest (diffusion_policy/config/splits/simpler/<task>/"
                             "<rule>_train<N>.json) whose --split episodes are rolled out from "
                             "their seeds. Default: the checkpoint's own "
                             "task.dataset.split_file, so test is its held-out set.")
    parser.add_argument("--split", "--split-name", dest="split",
                        choices=["test", "val", "train"], default="test",
                        help="Which list of the split manifest to roll out.")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Explicit comma-separated env seeds; overrides the manifest.")
    parser.add_argument("--num-episodes", type=int, default=None,
                        help="Cap: first K seeds of the split (default all).")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--task", default=TASK_NAME)
    parser.add_argument("--save-videos", type=int, default=0)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--mode", choices=["argmax", "softmax", "final"], default="argmax",
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
                             "split seeds across every N cell). 0 = all. Set to "
                             "e.g. 10 to keep full success-rate stats over all "
                             "--num-episodes while saving render data for just "
                             "the first 10 fixed-seed episodes.")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of search samples per replan; selection "
                             "(argmax/softmax) runs over all of them. "
                             "Default: policy.max_actions. Values > max_actions "
                             "are supported — predict_n_actions slides a fixed "
                             "(max_actions-1) context window.")
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
    parser.add_argument("--obs-corruption", choices=["as_trained", "clean"],
                        default="as_trained",
                        help="as_trained: apply the policy's trained obs corruption at "
                             "eval (seeded per episode). clean: feed uncorrupted obs. "
                             "Only affects policies trained with a fixed "
                             "obs_noise_level.")
    parser.add_argument("--eval-obs-noise-level", type=float, default=None,
                        help="BC only: noise the policy's low-dim state obs at rollout with "
                             "the fixed-level DDPM obs-noise schedule (100 steps, beta "
                             "1e-3->2e-2; level x -> t=round(x*99)). One draw per replan "
                             "shared by all N candidates; the verifier sees clean obs. "
                             "Default: off.")
    args = parser.parse_args()

    # Fix every global stream up front; the per-episode candidate / selection / obs-noise
    # draws are seeded separately, so nothing here depends on which episodes ran before.
    import random as _random
    _random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

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
            # TF32 changes matmul results with the GPU generation and batch shape.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.use_deterministic_algorithms(True, warn_only=True)
            print("[eval] deterministic torch kernels ON (cudnn.deterministic, use_deterministic_algorithms)")
        except Exception as _e:
            print(f"[eval] could not fully enable determinism: {_e!r}")

    policy, cfg = load_policy(
        checkpoint=args.checkpoint, device=device, use_ema=not args.no_ema,
    )
    if hasattr(policy, "corrupt_obs_eval"):
        policy.corrupt_obs_eval = (args.obs_corruption == "as_trained")
        print(f"[eval] obs corruption: {args.obs_corruption} (policy corrupt_obs="
              f"{getattr(policy, 'corrupt_obs', None)}, obs_noise_level="
              f"{getattr(policy, 'obs_noise_level', None)})")
    elif args.obs_corruption == "clean":
        raise ValueError(f"--obs-corruption clean: {type(policy).__name__} has no "
                         f"corrupt_obs_eval switch")
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
        if _vtype in ("veritas", "veritas_metric", "veritas_pixel_gripper", "veritas_subtask"):
            # Veritas pixel-waypoint verifier: scores the FULL predicted chunk's
            # end-effector endpoint against the live tracker's upcoming waypoints
            # (targets injected per replan by VeritasTrackerWrapper below).
            _verifier = _build_veritas_verifier(n_action_steps=policy.n_action_steps)
            _os.environ.setdefault("BC_VERIFIER_SCORE", "full")
            print(f"[eval] BC verifier = Veritas pixel-waypoint "
                  f"(scale={_verifier.action_pos_scale}, k={_verifier.lookahead_k}, "
                  f"BC_VERIFIER_SCORE={_os.environ.get('BC_VERIFIER_SCORE')})")
        elif _vtype == "dqc":
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
        if _vtype in ("dqc", "veritas", "veritas_metric", "veritas_pixel_gripper", "veritas_subtask"):
            # DQC/Veritas eval: candidate count = N_SAMPLES (default 16), no 128 floor.
            _max_actions = int(_os.environ.get("N_SAMPLES", "16"))
        else:
            _max_actions = max(128, int(_os.environ.get("N_SAMPLES", "128")))
        policy = BCSearchWrapper(policy, _verifier, max_actions=_max_actions)
        print(f"[eval] BC mode: wrapped with in-process verifier "
              f"(instruction={_instr!r}, max_actions={_max_actions})")
        if args.eval_obs_noise_level is not None:
            policy.eval_obs_noise_level = float(args.eval_obs_noise_level)
            print(f"[eval] eval-only state obs noise: level={policy.eval_obs_noise_level} "
                  f"on {list(getattr(policy._base, 'obs_keys', ()))} (verifier sees clean obs)")
    else:
        # Native search policy (predict_n_actions + baked verifier). Optionally
        # swap the baked LLaVA verifier for the DQC Q-critic so the
        # autoregressive search scores candidates by the discounted chunk sum
        # of DQC Q-values. Enabled with VERIFIER_TYPE=dqc; the score_window
        # block below sets score_indices to the executed chunk on it.
        import os as _os
        if (_os.environ.get("VERIFIER_TYPE", "robomonkey").lower() == "veritas"
                or (_os.environ.get("VERIFIER_TYPE") is None
                    and type(getattr(policy, "verifier", None)).__name__ in ("VeritasPixelVerifier", "VeritasStackPositionVerifier"))):
            existing = getattr(policy, 'verifier', None)
            if type(existing).__name__ not in ('VeritasPixelVerifier','VeritasStackPositionVerifier'):
                existing = None
            policy.verifier = _build_veritas_verifier(existing, policy.n_action_steps)
            print("[eval] SEARCH verifier swapped -> Veritas pixel-waypoint")
        elif (_os.environ.get("VERIFIER_TYPE", "robomonkey").lower() == "dqc"
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
    if type(policy.verifier).__name__ in ('VeritasPixelVerifier', 'VeritasStackPositionVerifier', 'VeritasMetricVerifier', 'VeritasGripperVerifier'):
        print(f"[eval] Veritas joint endpoint scoring: horizon={policy.verifier.horizon}, "
              f"plan_done_mode={policy.verifier.plan_done_mode}, "
              f"normalize={policy.verifier.normalize}")
    elif _bc_wrapped:
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
    if args.eval_obs_noise_level is not None and not hasattr(policy, "eval_obs_noise_level"):
        raise ValueError(f"--eval-obs-noise-level is BC-only; {type(policy).__name__} "
                         f"has its own obs corruption (--obs-corruption)")
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
    _veritas_wrap = _veritas_env_wrapper(args.task) \
        if type(getattr(policy, "verifier", None)).__name__ in ("VeritasPixelVerifier", "VeritasStackPositionVerifier") else None
    if _veritas_wrap is not None:
        env = _veritas_wrap(env)
        print(f"[eval] env wrapped with VeritasTrackerWrapper "
              f"(plans: {_os.environ.get('VERITAS_PLAN_DIR')}, "
              f"live={_os.environ.get('VERITAS_ALLOW_LIVE', '0')})")
    instr = (
        _base_env.get_language_instruction()
        if hasattr(_base_env, "get_language_instruction") else None
    ) or args.task
    print(f"[eval] instruction: {instr!r}")

    explicit_seeds, seed_src, split_info, train_prov = _select_episode_seeds(args, cfg)
    num_episodes = len(explicit_seeds)
    _overlap, _train_src = (split_info["overlap_with_ckpt_train_seeds"],
                            split_info["ckpt_train_seeds_source"])
    print(f"[eval] split={split_info['split']} n={num_episodes} "
          f"manifest={split_info.get('manifest')} "
          f"ckpt_train_seeds={_train_src or 'unknown'} overlap={len(_overlap)}")

    output_dir = Path(args.output_dir) if args.output_dir else None
    done_by_seed: Dict[int, Dict[str, Any]] = {}
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Record the configuration BEFORE any episode runs, so a crashed or preempted eval
        # still leaves behind what it was trying to measure.
        _write_run_config(output_dir, args, extra={
            "instruction": instr, "episodes": split_info,
            "checkpoint_dataset_provenance": train_prov,
            **_resume_identity(args)})
        _ep_path = output_dir / "episodes.jsonl"
        # Per-episode resume: load episodes already completed (keyed by seed)
        # before truncating, so a re-run with more episodes reuses them.
        if (not args.no_resume) and _ep_path.exists():
            _skipped_errors = 0
            _skipped_mismatch = 0
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
                # Only reuse episodes produced under the same seeding scheme, tiling and
                # obs corruption; anything else is a different realization.
                if any(_r.get(k) != v for k, v in _resume_identity(args).items()):
                    _skipped_mismatch += 1
                    continue
                done_by_seed[int(_r["seed"])] = _r
            if done_by_seed or _skipped_errors or _skipped_mismatch:
                print(f"[eval] resume: reusing {len(done_by_seed)} completed "
                      f"episode(s) from {_ep_path.name}"
                      + (f"; re-running {_skipped_errors} previously-errored "
                         f"episode(s)" if _skipped_errors else "")
                      + (f"; re-running {_skipped_mismatch} episode(s) from a different "
                         f"seed scheme / TILED_ENVS / obs corruption"
                         if _skipped_mismatch else ""))
        ep_log_file = open(_ep_path, "w")
    else:
        ep_log_file = None
        print("[eval] WARNING: no --output-dir, so this run's configuration "
              "(checkpoint, verifier, and ~20 scoring/selection env vars) is NOT "
              "recorded anywhere. Its numbers will not be reproducible.")

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
        _all_seeds = list(explicit_seeds)
        tiled_envs = [env] + [
            (_veritas_wrap(make_env(args.task)) if _veritas_wrap is not None else make_env(args.task))
            for _ in range(TILED - 1)]
        print(f"[eval] TILED mode: {TILED} parallel envs over {num_episodes} episodes", flush=True)
        for b in range(0, num_episodes, TILED):
            bseeds = _all_seeds[b:b + TILED]
            res = rollout_episodes_tiled(
                tiled_envs[:len(bseeds)], policy, cfg, device, bseeds, int(args.max_steps),
                str(instr), mode=str(args.mode), n_samples=args.n_samples,
                softmax_temp=float(args.softmax_temp))
            tiled_eps.extend(res)
            print(f"[eval]   tiled batch {b//TILED+1}: seeds {bseeds} -> "
                  f"succ={[int(r['success']) for r in res]} steps={[r['num_steps'] for r in res]}",
                  flush=True)

    for i in range(num_episodes):
        ep_seed = int(explicit_seeds[i])
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
            # Same for a live Gemini plan that never arrived (overload outlasting every
            # retry): the episode never started, so it is not a task failure.
            if "live Gemini planning failed" in str(e) or "metric verifier:" in str(e):
                print(f"[eval] FATAL: no Gemini plan for seed={ep_seed}; aborting run "
                      f"without logging a failure. Re-launch to resume.", flush=True)
                if ep_log_file is not None:
                    ep_log_file.flush()
                raise
            if "CUDA error" in str(e) or "device-side assert" in str(e):
                print(f"[eval] FATAL: CUDA context corrupted at seed={ep_seed}; "
                      f"aborting run without logging phantom failures. "
                      f"Re-launch to resume the remaining seeds.", flush=True)
                if ep_log_file is not None:
                    ep_log_file.flush()
                raise
            if os.environ.get("VERIFIER_TYPE") in ("veritas", "veritas_metric", "veritas_pixel_gripper", "veritas_subtask") or type(policy.verifier).__name__ == "VeritasStackPositionVerifier":
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
            if type(policy.verifier).__name__ == 'VeritasMetricVerifier':
                extra_proj['verifier_action_pos_scale'] = np.asarray(
                    policy.verifier.action_pos_scale if policy.verifier.action_pos_scale is not None else [1.,1.,1.],
                    dtype=np.float32)
            np.savez_compressed(
                npz_path,
                **{'policy_obs__' + k: np.stack([b['policy_obs'][k] for b in br])
                   for k in br[0]['policy_obs']},
                candidate_actions_full=np.stack([b['candidate_actions_full'] for b in br]),
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
                "success_while_grasped": ep.get("success_while_grasped"),
                "truncated": ep["truncated"],
                "num_steps": ep["num_steps"],
                "obj_goal_dist_final": ep.get("obj_goal_dist_final"),
                "obj_goal_dist_min": ep.get("obj_goal_dist_min"),
                "duration_s": durations[-1],
                **_resume_identity(args),
                **seed_src.get(ep_seed, {}),
            }
            if ep.get("subtasks") is not None:ep_compact["subtasks"]=ep["subtasks"]
            if ep.get("init_state") is not None:
                ep_compact["init_state"] = ep["init_state"]
            if ep.get("veritas") is not None:
                ep_compact["veritas"] = ep["veritas"]
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
        "split": split_info["split"],
        "seeds": [int(x) for x in explicit_seeds],
        "max_steps": int(args.max_steps),
        "use_ema": (not args.no_ema),
        "mode": str(args.mode),
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
