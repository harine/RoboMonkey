"""Diagnose BC candidate diversity: do N predict_action calls produce diverse
trajectories, and where (executed window vs far-horizon step -1) do they
collapse? Mirrors the eval's deterministic per-replan seeding.
"""
import sys, torch, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval_diffusion"))
# Pre-import so hydra's _locate can resolve the FlattenObsEncoder _target_.
import diffusion_policy.model.vision.multi_image_obs_encoder  # noqa: F401
from eval_diffusion import load_policy  # type: ignore

CKPT = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
device = torch.device("cuda:0")
policy, cfg = load_policy(checkpoint=CKPT, device=device, use_ema=True)
policy.eval()

# Build a dummy obs dict (B=1, To=n_obs_steps) from shape_meta low-dim keys.
To = int(cfg.n_obs_steps)
sm = cfg.shape_meta["obs"]
torch.manual_seed(0)
obs = {}
for k, v in sm.items():
    if v.get("type", "low_dim") != "low_dim":
        continue
    dim = int(v["shape"][0])
    obs[k] = torch.randn(1, To, dim, device=device)
# keep only keys the normalizer knows
known = set(policy.normalizer.params_dict.keys())
obs = {k: v for k, v in obs.items() if k in known}
print(f"[diag] obs keys: {list(obs)}  To={To}")

# Mimic eval: seed once, then draw N candidates sequentially.
torch.manual_seed(123456)
chunks = []
for i in range(N):
    out = policy.predict_action(obs)
    chunks.append(out["action_pred"][0].detach().cpu().numpy())  # (horizon, Da)
A = np.stack(chunks, 0)  # (N, horizon, Da)
print(f"[diag] action_pred stack: {A.shape}  (N, horizon, action_dim)")

# Per-step spread across the N candidates.
spread = (A.max(0) - A.min(0)).mean(axis=-1)  # (horizon,)
var = A.var(axis=0).mean(axis=-1)             # (horizon,)
print("[diag] per-horizon-step spread (max-min, mean over dims):")
for t in range(A.shape[1]):
    print(f"    step {t:2d}: spread={spread[t]:.6f}  var={var[t]:.8f}")

print(f"[diag] step -1 (scored by verifier) spread = {spread[-1]:.6f}")
print(f"[diag] all candidates identical at step -1? "
      f"{np.allclose(A[:, -1], A[0, -1])}")
print(f"[diag] all candidates fully identical? {np.allclose(A, A[0])}")
