"""eval_q.py — Evaluate the DQC Q function on robot manipulation trajectories.

Reconstructs the ResNet-34 + FiLM encoder and MLP critic from the JAX DQC
checkpoint in pure JAX (no Flax required), then scores each (image, action)
pair from trajectory data split by episode outcome (success / failure).

Two data sources are supported via --mode:

  npz   Load search-policy eval NPZ files (ep*.npz).  Each file has R replan
        steps with K candidate actions and frame snapshots.

        Bug fixed here: these files store selected_index = -1 as a sentinel
        meaning the selection metadata was not saved.  The old code passed -1
        directly to NumPy, which silently returned the *last* candidate — an
        arbitrary choice unrelated to what the policy actually ran.  Fix:
        detect sel < 0 and substitute argmax(values[r]) so we always score
        the candidate the policy would have chosen (highest verifier score).

  zarr  Load a zarr shard that contains agentview_image (save_images=True
        collection runs).  Gives per-step (image, action) pairs at the full
        episode granularity rather than at replan cadence, and typically
        provides 10–100× more episodes than the NPZ set.  Carrot data only
        has state-only shards (no images); a clear error is raised if the
        image key is absent.

Architecture (from checkpoint flags.json + param shapes):
  Encoder : ResNet-34, GroupNorm + FiLM (text → scale+shift per block)
            7×7 stride-2 stem, 3 RGB + 2 spatial-coord channels → 512-dim
  Critic  : 5-layer MLP (1024 hidden), LayerNorm + SiLU, 2-ensemble min
  Action  : 7D, normalized to [-1,1] via action_normalization.json
  Text    : 512-dim MUSE (USE-4) embedding, or placeholder random vector

Usage
-----
# NPZ mode (fixed selection bug):
conda run -n simpler_env python eval_q.py --mode npz --text-emb-file task_emb.npy

# Zarr mode — eggplant (has images):
conda run -n simpler_env python eval_q.py --mode zarr \\
    --zarr-path /home/harine/data/eggplant_in_basket/all_data/state0.zarr \\
    --instruction "put the eggplant in the basket" \\
    --text-emb-file task_emb.npy \\
    --out-dir results/eggplant_zarr_q

# Zarr mode — carrot (state-only, no images → will raise a clear error):
conda run -n simpler_env python eval_q.py --mode zarr \\
    --zarr-path /home/harine/data/carrot_on_plate/state_only/state0.zarr
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ── Image preprocessing ───────────────────────────────────────────────────────

def preprocess_image(img_hwc: np.ndarray, size: int = 128) -> np.ndarray:
    """Resize uint8 (H,W,3) → (size,size,5) float32 with spatial-coord channels."""
    pil = Image.fromarray(img_hwc).resize((size, size), Image.BILINEAR)
    rgb = np.asarray(pil, dtype=np.float32) / 255.0
    H, W = rgb.shape[:2]
    xs = np.linspace(-1.0, 1.0, W, dtype=np.float32)[None, :] * np.ones((H, 1), np.float32)
    ys = np.linspace(-1.0, 1.0, H, dtype=np.float32)[:, None] * np.ones((1, W), np.float32)
    return np.concatenate([rgb, xs[..., None], ys[..., None]], axis=-1)  # (H,W,5)


# ── Action normalization ───────────────────────────────────────────────────────

def normalize_action(action: np.ndarray, norm: Dict) -> np.ndarray:
    low  = np.array(norm["low"],  dtype=np.float32)
    high = np.array(norm["high"], dtype=np.float32)
    eps  = float(norm["clip_eps"])
    a = np.clip(action.astype(np.float32), low + eps, high - eps)
    return 2.0 * (a - low) / (high - low) - 1.0


# ── Text embedding ─────────────────────────────────────────────────────────────

def make_text_embedding(instruction: str, dim: int = 512) -> np.ndarray:
    """Deterministic unit-norm placeholder (hash-seeded). Use real MUSE for
    meaningful absolute Q values; relative comparisons are valid either way."""
    import hashlib
    seed = int(hashlib.md5(instruction.encode()).hexdigest(), 16) % (2 ** 31)
    rng  = np.random.default_rng(seed)
    emb  = rng.standard_normal(dim).astype(np.float32)
    return emb / np.linalg.norm(emb)


# ── Pure-JAX forward pass ─────────────────────────────────────────────────────

def _num_groups(C: int, max_g: int = 32) -> int:
    g = max_g
    while C % g != 0:
        g -= 1
    return g

def _group_norm(x, scale, bias):
    H, W, C = x.shape
    G = _num_groups(C)
    x = x.reshape(H, W, G, C // G)
    mean = x.mean(axis=(0, 1, 3), keepdims=True)
    var  = x.var (axis=(0, 1, 3), keepdims=True)
    x = (x - mean) / jnp.sqrt(var + 1e-5)
    return x.reshape(H, W, C) * scale + bias

def _layer_norm(x, scale, bias):
    mean = x.mean(); var = x.var()
    return (x - mean) / jnp.sqrt(var + 1e-6) * scale + bias

def _conv2d(x, kernel, stride=1):
    return jax.lax.conv_general_dilated(
        x[None], kernel,
        window_strides=(stride, stride), padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )[0]

def _film(x, text, fp):
    gamma = text @ fp["Dense_0"]["kernel"] + fp["Dense_0"]["bias"]
    beta  = text @ fp["Dense_1"]["kernel"] + fp["Dense_1"]["bias"]
    return x * gamma + beta

def _resnet_block(x, text, bp, fp, stride=1):
    """Post-activation bridge ResNet block: Conv→GN→FiLM→SiLU→Conv→GN→SiLU→(+skip)."""
    residual = x
    y = _conv2d(x, bp["Conv_0"]["kernel"], stride=stride)
    y = _group_norm(y, bp["GroupNorm_0"]["scale"], bp["GroupNorm_0"]["bias"])
    y = _film(y, text, fp)
    y = jax.nn.silu(y)
    y = _conv2d(y, bp["Conv_1"]["kernel"], stride=1)
    y = _group_norm(y, bp["GroupNorm_1"]["scale"], bp["GroupNorm_1"]["bias"])
    y = jax.nn.silu(y)
    if "conv_proj" in bp:
        residual = _conv2d(residual, bp["conv_proj"]["kernel"], stride=stride)
        residual = _group_norm(residual, bp["norm_proj"]["scale"], bp["norm_proj"]["bias"])
    return y + residual

# ResNet-34: stages (n_blocks, channels, first_stride)
_STRIDES: List[int] = []
for n, _c, s0 in [(3, 64, 1), (4, 128, 2), (6, 256, 2), (3, 512, 2)]:
    _STRIDES.extend([s0] + [1] * (n - 1))

def _encode(img5, text, enc_p):
    ie = enc_p["image_encoder"]
    x  = _conv2d(img5, ie["conv_init"]["kernel"], stride=2)
    x  = _group_norm(x, ie["norm_init"]["scale"], ie["norm_init"]["bias"])
    x  = jax.nn.silu(x)
    for i, s in enumerate(_STRIDES):
        x = _resnet_block(x, text, ie[f"ResNetBlock_{i}"], ie[f"FilmConditioning_{i}"], s)
    return x.mean(axis=(0, 1))  # global avg pool → (512,)

def _mlp(feat, mlp_p, head_p):
    x = feat
    for i in range(5):
        x = x @ mlp_p[f"Dense_{i}"]["kernel"] + mlp_p[f"Dense_{i}"]["bias"]
        x = _layer_norm(x, mlp_p[f"LayerNorm_{i}"]["scale"], mlp_p[f"LayerNorm_{i}"]["bias"])
        x = jax.nn.silu(x)
    return (x @ head_p["kernel"] + head_p["bias"])[0]

def _q_one(img5, act, text, p):
    feat = jnp.concatenate([_encode(img5, text, p["encoder"]), act])
    return _mlp(feat, p["MLP_0"], p["Dense_0"])


# ── Scorer ────────────────────────────────────────────────────────────────────

def build_scorer(ckpt_dir: str):
    ckpt_path = Path(ckpt_dir) / "params_100000.pkl"
    norm_path = Path(ckpt_dir) / "action_normalization.json"
    with open(ckpt_path, "rb") as f:
        d = pickle.load(f)
    all_params = d["agent"]["network"]["params"]["modules_action_critic"]
    with open(norm_path) as f:
        norm_cfg = json.load(f)

    params_list = [
        jax.tree_util.tree_map(lambda x: jnp.array(x[e]), all_params)
        for e in range(2)
    ]

    @jax.jit
    def _batch(imgs5, acts, text, p):
        return jax.vmap(lambda i, a: _q_one(i, a, text, p))(imgs5, acts)

    def score(frames_hwc: np.ndarray, actions_raw: np.ndarray,
              text_emb: np.ndarray, img_size: int = 128) -> np.ndarray:
        imgs5 = jnp.array(np.stack([preprocess_image(f, img_size) for f in frames_hwc]))
        acts  = jnp.array(np.stack([normalize_action(a, norm_cfg) for a in actions_raw]))
        txt   = jnp.array(text_emb)
        qs    = np.stack([np.array(_batch(imgs5, acts, txt, p)) for p in params_list])
        return qs.min(axis=0)

    return score


# ── Data loading ──────────────────────────────────────────────────────────────

def load_npz_episodes(npz_dir: str) -> List[Dict]:
    """Load search-policy eval NPZ files.

    Fix applied: selected_index stores -1 as a sentinel when the selection was
    not recorded.  NumPy's -1 indexing silently returns the *last* element of
    the array — an arbitrary candidate unrelated to the policy's actual choice.
    We detect this and substitute argmax(values[r]) so we always score the
    candidate with the highest verifier score, which is what weighted-topk
    selection would have chosen.
    """
    npz_dir = Path(npz_dir)
    meta: Dict[int, bool] = {}
    jsonl = npz_dir.parent / "episodes.jsonl"
    if jsonl.exists():
        with open(jsonl) as f:
            for line in f:
                ep = json.loads(line)
                meta[ep["ep_idx"]] = bool(ep["success"])

    episodes = []
    for npz_path in sorted(npz_dir.glob("ep*.npz")):
        z       = np.load(npz_path, allow_pickle=False)
        ep_idx  = int(z["ep_idx"])
        success = bool(z["success"]) if "success" in z.files else meta.get(ep_idx, False)

        cand   = z["candidate_actions"]   # (R, K, T, 7)
        sel    = z["selected_index"]       # (R,)  — all -1 sentinel (never recorded)
        values = z["values"]               # (R, K)  RoboMonkey verifier scores per candidate
        R      = cand.shape[0]

        # FIX: selected_index is an all -1 sentinel. The TRUE selection is
        # recorded in topk_indices/topk_weights (weighted top-k sampling — here
        # weighted_topk2, so 2 of K candidates with weights summing to 1). The
        # "selected" candidate is the highest-weighted top-k entry; fall back to
        # argmax(values) only if topk is unavailable. (NumPy's -1 indexing would
        # have silently returned the LAST candidate — an arbitrary score.)
        if "topk_indices" in z.files and "topk_weights" in z.files:
            ti = z["topk_indices"]         # (R, topk) — unused slots are -1
            tw = z["topk_weights"]         # (R, topk)
            sel_fixed = ti[np.arange(R), np.argmax(tw, axis=1)]
            sel_fixed = np.where(sel_fixed < 0, np.argmax(values, axis=1), sel_fixed)
        else:
            sel_fixed = np.argmax(values, axis=1)

        selected_actions  = cand[np.arange(R), sel_fixed, 0, :]   # (R, 7) first action step
        selected_verifier = values[np.arange(R), sel_fixed]        # (R,) selected-candidate score

        episodes.append({
            "ep_idx":           ep_idx,
            "success":          success,
            "frames":           z["frames"],          # (R, H, W, 3)
            "selected_actions": selected_actions,     # (R, 7)
            "branch_t":         z["branch_t"],        # (R,)
            "verifier_score":   selected_verifier,    # (R,) — selected-candidate verifier score
            "verifier_max":     values.max(axis=1),   # (R,) — best-of-K
            "verifier_mean":    values.mean(axis=1),  # (R,) — mean-of-K (better discriminator)
            "source":           "npz",
        })

    print(f"[npz] loaded {len(episodes)} episodes from {npz_dir}")
    return episodes


def load_zarr_episodes(zarr_path: str, max_per_class: int = 150, seed: int = 42) -> List[Dict]:
    """Load per-step (image, action) episodes from a zarr shard.

    Raises ValueError immediately if agentview_image is absent — the DQC Q
    function cannot run without images (carrot state-only data hits this).
    """
    import zarr as _zarr
    z = _zarr.open(zarr_path, "r")

    obs_keys = list(z["data"]["obs"].keys())
    if "agentview_image" not in obs_keys:
        raise ValueError(
            f"zarr at {zarr_path} has no agentview_image.\n"
            f"  Available obs keys: {obs_keys}\n"
            f"  The DQC Q function requires images. Carrot data was collected "
            f"with save_images=False (state_only). Cannot evaluate."
        )

    dones   = z["data"]["dones"][:]
    rewards = z["data"]["rewards"][:]
    ep_ends = np.where(dones)[0]
    ep_starts = np.concatenate([[0], ep_ends[:-1] + 1])
    success_labels = rewards[ep_ends] > 0

    n_s = success_labels.sum()
    n_f = (~success_labels).sum()
    print(f"[zarr] {zarr_path}")
    print(f"       {len(ep_ends)} total eps — {n_s} success, {n_f} failure")

    rng = np.random.default_rng(seed)
    sel_s = rng.choice(np.where( success_labels)[0], min(max_per_class, int(n_s)), replace=False)
    sel_f = rng.choice(np.where(~success_labels)[0], min(max_per_class, int(n_f)), replace=False)
    selected = np.sort(np.concatenate([sel_s, sel_f]))
    print(f"       sampling {len(sel_s)} success + {len(sel_f)} failure")

    episodes = []
    for ep_i in selected:
        s, e    = int(ep_starts[ep_i]), int(ep_ends[ep_i]) + 1
        frames  = z["data"]["obs"]["agentview_image"][s:e]  # (T, H, W, 3)
        actions = z["data"]["actions"][s:e]                 # (T, 7)
        T       = e - s
        episodes.append({
            "ep_idx":           int(ep_i),
            "success":          bool(success_labels[ep_i]),
            "frames":           frames,
            "selected_actions": actions,
            "branch_t":         np.arange(T),
            "verifier_score":   np.full(T, np.nan),  # no verifier scores in zarr
            "verifier_max":     np.full(T, np.nan),
            "verifier_mean":    np.full(T, np.nan),
            "source":           "zarr",
        })

    return episodes


# ── Plotting (shared, jax-free) ───────────────────────────────────────────────

from plot_utils import make_plot, _auc  # noqa: E402


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["npz", "zarr"], default="zarr")
    # NPZ mode
    parser.add_argument("--npz-dir", default=(
        "/home/harine/data/verifier_scored_last_action_only/eval_results/"
        "search_sweep_20260517_181853/weighted_topk2_n16/search_q"))
    # Zarr mode
    parser.add_argument("--zarr-path", default=
        "/home/harine/data/eggplant_in_basket/all_data/state0.zarr")
    parser.add_argument("--max-eps-per-class", type=int, default=150,
        help="Max episodes per success/failure class to sample (zarr mode).")
    # Shared
    parser.add_argument("--ckpt-dir", default=
        "/home/harine/RoboMonkey/dqc_checkpoint/sd000s_137913.0.20260605_163017")
    parser.add_argument("--out-dir",  default="results/q_eval")
    parser.add_argument("--instruction", default="put the eggplant in the basket")
    parser.add_argument("--text-emb-file", default=None)
    parser.add_argument("--img-size", type=int, default=128)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Text embedding
    if args.text_emb_file:
        text_emb = np.load(args.text_emb_file).astype(np.float32)
        assert text_emb.shape == (512,)
        print(f"[text] loaded MUSE embedding from {args.text_emb_file}")
    else:
        text_emb = make_text_embedding(args.instruction)
        print(f"[text] placeholder embedding for '{args.instruction}' (pass --text-emb-file for MUSE)")

    # Checkpoint
    print(f"[ckpt] loading {args.ckpt_dir} ...")
    scorer = build_scorer(args.ckpt_dir)
    print("[ckpt] compiled")

    # Data
    if args.mode == "npz":
        episodes = load_npz_episodes(args.npz_dir)
    else:
        episodes = load_zarr_episodes(args.zarr_path, args.max_eps_per_class)

    n_suc = sum(e["success"] for e in episodes)
    n_fail = len(episodes) - n_suc
    source = episodes[0]["source"] if episodes else args.mode
    print(f"[data] {len(episodes)} episodes: {n_suc} success, {n_fail} failure  (source={source})")

    # Score
    import pandas as pd
    rows = []
    for i, ep in enumerate(episodes):
        if (i + 1) % 50 == 0:
            print(f"  scoring ep {i+1}/{len(episodes)} ...", flush=True)
        q_vals = scorer(ep["frames"], ep["selected_actions"], text_emb, args.img_size)
        T = len(q_vals)
        for t in range(T):
            rows.append({
                "ep_idx":         ep["ep_idx"],
                "success":        ep["success"],
                "branch_t":       int(ep["branch_t"][t]),
                "q_value":        float(q_vals[t]),
                "verifier_score": float(ep["verifier_score"][t]),
                "source":         ep["source"],
            })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "q_values.csv"
    df.to_csv(csv_path, index=False)
    print(f"[out] {len(df)} rows → {csv_path}")

    # Summary
    for label, mask in [("SUCCESS", df["success"]), ("FAILURE", ~df["success"])]:
        sub = df[mask]
        print(f"\n{label} ({sub['ep_idx'].nunique()} eps, {len(sub)} steps):")
        print(f"  Q value  mean={sub['q_value'].mean():.4f}  "
              f"std={sub['q_value'].std():.4f}  "
              f"[{sub['q_value'].min():.4f}, {sub['q_value'].max():.4f}]")
        if not sub["verifier_score"].isna().all():
            print(f"  Verifier mean={sub['verifier_score'].mean():.4f}  "
                  f"std={sub['verifier_score'].std():.4f}")

    print(f"\nQ mean  success={df[df['success']]['q_value'].mean():.4f}  "
          f"failure={df[~df['success']]['q_value'].mean():.4f}  "
          f"Δ={df[df['success']]['q_value'].mean() - df[~df['success']]['q_value'].mean():+.4f}")

    title_task = args.instruction
    title = (f"DQC Q fn — {title_task}\n"
             f"({n_suc} success / {n_fail} failure  |  source={source}  |  "
             f"{'MUSE emb' if args.text_emb_file else 'placeholder emb'})")
    make_plot(df, out_dir / "q_values.png", title, source)


if __name__ == "__main__":
    main()
