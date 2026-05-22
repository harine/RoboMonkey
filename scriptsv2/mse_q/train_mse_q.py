"""train_mse_q.py — Calibrate the RoboMonkey verifier to the MSE-error scale.

The state-based search policy is trained with ``MSEVerifier``, whose score is
the negative MSE between a candidate action chunk and the dataset expert
action.  At eval time there is no ground-truth action, so ``MSEVerifier``
cannot run — only the RoboMonkey reward model is available, and its scores
live on a different scale.

This script learns a 1-D transformation ``g(reward) -> -mse`` (an
``MLPCalibrator``) so the RoboMonkey verifier can stand in for ``MSEVerifier``
when evaluating the MSE-trained search policy.

Calibration data
----------------
For every action window in the offline dataset (the same shards the search
policy was trained on) we form noisy candidate actions:

    candidate = expert + sigma * per_dim_action_std * N(0, 1)

over a grid of ``sigma`` values (``sigma=0`` keeps the expert action, giving
the mse=0 / max-reward anchor).  Each candidate is scored twice:

    mse_target = -mean((candidate - expert) ** 2)      # MSEVerifier scale
    reward     = RoboMonkeyVerifier(image, candidate)  # reward-model scale

and the resulting ``(reward, mse_target)`` pairs train the MLP.

Usage
-----
    python scriptsv2/mse_q/train_mse_q.py \
        --output-dir data/mse_q/eggplant \
        --noise-levels 0.0 0.25 0.5 1.0 1.5 2.5 \
        --max-samples 20000

Run via the ``train_mse_q.sh`` wrapper so the conda env / paths are set up.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- repo paths --------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROBOMONKEY = _HERE.parents[1]                       # .../RoboMonkey
_DP_REPO = _ROBOMONKEY / "diffusion_policy"          # diffusion_policy submodule
_DP_CONFIG = _DP_REPO / "diffusion_policy" / "config"
for _p in (str(_DP_REPO), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hydra  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from calibrator import MLPCalibrator  # noqa: E402
from diffusion_policy.policy.verifiers import RoboMonkeyVerifier  # noqa: E402

# `${eval:'...'}` resolver used by the diffusion_policy configs.
OmegaConf.register_new_resolver("eval", eval, replace=True)

CONFIG_NAME = "robomonkey_eggplant_search_state_diffusion_mse"


# --- calibration-data collection --------------------------------------------
def build_dataset(dataset_dir: str):
    """Instantiate the offline dataset from the search-policy training config."""
    from hydra import compose, initialize_config_dir

    overrides = [
        f"task.dataset.dataset_dir={dataset_dir}",
        "task.dataset.val_ratio=0.0",  # use all windows for calibration
    ]
    with initialize_config_dir(version_base=None, config_dir=str(_DP_CONFIG)):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    return dataset


@torch.no_grad()
def collect_pairs(
    dataset,
    verifier: RoboMonkeyVerifier,
    noise_levels: List[float],
    batch_size: int,
    num_workers: int,
    max_samples: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Score noisy candidate actions -> (reward, mse_target, sigma) arrays.

    ``mse_target`` follows the MSEVerifier convention: ``-mean((cand-expert)^2)``
    over the (horizon, action_dim) axes, so higher = closer to the expert.
    """
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    rewards: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    sigmas: List[torch.Tensor] = []
    action_sq_sum = action_sum = None
    action_count = 0
    seen = 0
    t0 = time.time()

    for bi, batch in enumerate(loader):
        expert = batch["action"].float()              # (B, T, Da), raw scale
        obs = batch["obs"]
        B = expert.shape[0]

        # Per-dim action std drives the noise scale (raw action units differ
        # across dims; gripper vs deltas). Per-batch estimate is stable at the
        # batch sizes used here.
        std = expert.std(dim=(0, 1)).clamp_min(1e-6)   # (Da,)

        # Running global action stats (reported in the checkpoint metadata).
        flat = expert.reshape(-1, expert.shape[-1])
        if action_sum is None:
            action_sum = flat.sum(dim=0)
            action_sq_sum = (flat ** 2).sum(dim=0)
        else:
            action_sum += flat.sum(dim=0)
            action_sq_sum += (flat ** 2).sum(dim=0)
        action_count += flat.shape[0]

        for sigma in noise_levels:
            if sigma == 0.0:
                candidate = expert.clone()
            else:
                noise = torch.randn(expert.shape, generator=gen) * (sigma * std)
                candidate = expert + noise

            # MSEVerifier-scale target: negative MSE over (horizon, action_dim).
            mse_target = -((candidate - expert) ** 2).mean(dim=(-1, -2))  # (B,)
            reward = verifier.get_value(obs, candidate).float().cpu()      # (B,)

            rewards.append(reward)
            targets.append(mse_target)
            sigmas.append(torch.full((B,), float(sigma)))

        seen += B
        if (bi + 1) % 10 == 0 or seen >= max_samples:
            print(
                f"  [collect] batch {bi + 1}  windows={seen}  "
                f"pairs={seen * len(noise_levels)}  "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
        if seen >= max_samples:
            break

    action_mean = (action_sum / action_count).tolist()
    action_std = (
        (action_sq_sum / action_count - (action_sum / action_count) ** 2)
        .clamp_min(0.0)
        .sqrt()
        .tolist()
    )
    return {
        "reward": torch.cat(rewards).numpy().astype(np.float32),
        "mse_target": torch.cat(targets).numpy().astype(np.float32),
        "sigma": torch.cat(sigmas).numpy().astype(np.float32),
        "n_windows": seen,
        "action_mean": action_mean,
        "action_std": action_std,
    }


# --- training ----------------------------------------------------------------
def train_calibrator(
    reward: np.ndarray,
    target: np.ndarray,
    hidden: List[int],
    epochs: int,
    lr: float,
    batch_size: int,
    val_ratio: float,
    device: str,
    seed: int,
) -> Dict:
    """Fit an MLPCalibrator to (reward -> mse_target). Returns model + metrics."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    n = reward.shape[0]
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_ratio)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    r = torch.from_numpy(reward)
    t = torch.from_numpy(target)
    r_mean, r_std = r[train_idx].mean().item(), r[train_idx].std().item()
    t_mean, t_std = t[train_idx].mean().item(), t[train_idx].std().item()
    r_std = r_std if r_std > 1e-8 else 1.0
    t_std = t_std if t_std > 1e-8 else 1.0

    model = MLPCalibrator(
        hidden=hidden,
        reward_mean=r_mean,
        reward_std=r_std,
        target_mean=t_mean,
        target_std=t_std,
    ).to(device)

    def make_loader(idx, shuffle):
        ds = TensorDataset(r[idx].to(device), t[idx].to(device))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(train_idx, True)
    val_loader = make_loader(val_idx, False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    best_val = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        for rb, tb in train_loader:
            # Train in standardized space for a well-conditioned objective.
            pred_std = model.net_std(model.standardize_reward(rb))
            tgt_std = (tb - model.target_mean) / model.target_std
            loss = loss_fn(pred_std, tgt_std)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            v_sq = v_n = 0.0
            for rb, tb in val_loader:
                pred = model(rb)  # mse-scale
                v_sq += ((pred - tb) ** 2).sum().item()
                v_n += tb.numel()
            val_mse = v_sq / max(1, v_n)
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"  [train] epoch {epoch + 1}/{epochs}  val_mse={val_mse:.6e}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Final metrics on the val split, in original (mse) units.
    with torch.no_grad():
        cal_val = model(r[val_idx].to(device)).cpu().numpy()
    tgt_val = target[val_idx]
    ss_res = float(np.sum((tgt_val - cal_val) ** 2))
    ss_tot = float(np.sum((tgt_val - tgt_val.mean()) ** 2)) or 1e-12
    metrics = {
        "n_pairs": int(n),
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "val_mse": float(best_val),
        "val_mae": float(np.mean(np.abs(tgt_val - cal_val))),
        "val_r2": float(1.0 - ss_res / ss_tot),
        "pearson_reward_vs_mse": float(
            np.corrcoef(reward[val_idx], tgt_val)[0, 1]
        ),
        "pearson_calibrated_vs_mse": float(
            np.corrcoef(cal_val, tgt_val)[0, 1]
        ),
        "reward_mean": r_mean,
        "reward_std": r_std,
        "target_mean": t_mean,
        "target_std": t_std,
    }
    return {"model": model, "metrics": metrics}


# --- plotting ----------------------------------------------------------------
def save_plot(path: str, data: Dict, model: MLPCalibrator, device: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reward, target, sigma = data["reward"], data["mse_target"], data["sigma"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(reward, target, c=sigma, cmap="viridis", s=6, alpha=0.4)
    fig.colorbar(sc, ax=ax, label="noise level σ (× per-dim action std)")

    grid = np.linspace(reward.min(), reward.max(), 400).astype(np.float32)
    with torch.no_grad():
        curve = model(torch.from_numpy(grid).to(device)).cpu().numpy()
    ax.plot(grid, curve, color="crimson", lw=2.5, label="MLP calibrator g(reward)")

    ax.set_xlabel("RoboMonkey verifier reward")
    ax.set_ylabel("MSEVerifier score  (-mse to expert action)")
    ax.set_title("Verifier-reward → MSE-scale calibration")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --- main --------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/mse_q/eggplant",
                        help="Directory for the checkpoint, plot, and summary.")
    parser.add_argument("--dataset-dir", default="/home/harine/data/eggplant_in_basket",
                        help="Offline dataset dir (search-policy training shards).")
    parser.add_argument("--server-url", default="http://127.0.0.1:3100",
                        help="RoboMonkey verifier backend: http URL of infer_server.py, "
                             "or 'in_process' (needs reward-model deps in this env).")
    parser.add_argument("--instruction", default="put the eggplant in the basket")
    parser.add_argument("--image-obs-key", default="agentview_image")
    parser.add_argument("--noise-levels", type=float, nargs="+",
                        default=[0.0, 0.25, 0.5, 1.0, 1.5, 2.5],
                        help="σ grid; candidate = expert + σ·per_dim_std·N(0,1).")
    parser.add_argument("--max-samples", type=int, default=20000,
                        help="Cap on dataset windows scored (pairs = windows × #σ).")
    parser.add_argument("--collect-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  mse_q — calibrate RoboMonkey verifier to the MSE-error scale")
    print(f"  dataset      : {args.dataset_dir}")
    print(f"  verifier     : {args.server_url}")
    print(f"  noise levels : {args.noise_levels}")
    print(f"  max samples  : {args.max_samples}")
    print(f"  output dir   : {out_dir}")
    print("=" * 70, flush=True)

    # 1. offline dataset + RoboMonkey verifier
    dataset = build_dataset(args.dataset_dir)
    print(f"[mse_q] dataset windows available: {len(dataset)}", flush=True)
    verifier = RoboMonkeyVerifier(
        server_url=args.server_url,
        instruction=args.instruction,
        image_obs_key=args.image_obs_key,
        health_check=True,
    )

    # 2. collect (reward, mse_target) pairs
    print("[mse_q] collecting calibration pairs ...", flush=True)
    data = collect_pairs(
        dataset=dataset,
        verifier=verifier,
        noise_levels=args.noise_levels,
        batch_size=args.collect_batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    n_pairs = data["reward"].shape[0]
    print(f"[mse_q] collected {n_pairs} pairs "
          f"from {data['n_windows']} windows", flush=True)
    np.savez(
        out_dir / "calibration_pairs.npz",
        reward=data["reward"], mse_target=data["mse_target"], sigma=data["sigma"],
    )

    # 3. train the calibrator
    print("[mse_q] training MLP calibrator ...", flush=True)
    result = train_calibrator(
        reward=data["reward"],
        target=data["mse_target"],
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.train_batch_size,
        val_ratio=args.val_ratio,
        device=args.device,
        seed=args.seed,
    )
    model, metrics = result["model"], result["metrics"]

    # per-σ sanity table
    per_sigma = {}
    for s in args.noise_levels:
        mask = data["sigma"] == s
        if mask.any():
            per_sigma[str(s)] = {
                "mean_reward": float(data["reward"][mask].mean()),
                "mean_mse_target": float(data["mse_target"][mask].mean()),
                "count": int(mask.sum()),
            }

    # 4. save artifacts
    ckpt_path = out_dir / "mse_q_calibrator.pt"
    metadata = {
        "dataset_dir": args.dataset_dir,
        "instruction": args.instruction,
        "image_obs_key": args.image_obs_key,
        "noise_levels": list(args.noise_levels),
        "n_windows": data["n_windows"],
        "action_mean": data["action_mean"],
        "action_std": data["action_std"],
        "metrics": metrics,
        "purpose": "g(robomonkey_reward) -> MSEVerifier-scale score (-mse)",
    }
    model.save(str(ckpt_path), metadata=metadata)
    save_plot(str(out_dir / "calibration.png"), data, model, args.device)

    summary = {
        "checkpoint": str(ckpt_path),
        "config_name": CONFIG_NAME,
        "args": vars(args),
        "metrics": metrics,
        "per_sigma": per_sigma,
        "action_mean": data["action_mean"],
        "action_std": data["action_std"],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print("  RESULTS")
    print(f"  pairs                : {metrics['n_pairs']} "
          f"(train {metrics['n_train']} / val {metrics['n_val']})")
    print(f"  val MSE              : {metrics['val_mse']:.6e}")
    print(f"  val MAE              : {metrics['val_mae']:.6e}")
    print(f"  val R^2              : {metrics['val_r2']:.4f}")
    print(f"  pearson reward~mse   : {metrics['pearson_reward_vs_mse']:.4f}")
    print(f"  pearson calib~mse    : {metrics['pearson_calibrated_vs_mse']:.4f}")
    print("  per-σ  mean_reward -> mean_mse_target")
    for s, v in per_sigma.items():
        print(f"    σ={s:<6} {v['mean_reward']:+.4f} -> {v['mean_mse_target']:+.6f}")
    print("-" * 70)
    print(f"  checkpoint : {ckpt_path}")
    print(f"  plot       : {out_dir / 'calibration.png'}")
    print(f"  summary    : {out_dir / 'summary.json'}")
    print(f"  pairs      : {out_dir / 'calibration_pairs.npz'}")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
