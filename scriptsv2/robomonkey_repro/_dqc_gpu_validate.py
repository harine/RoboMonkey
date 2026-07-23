import os, sys, json
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION","0.35")
sys.path.insert(0,"/mmfs1/home/harine/RoboMonkey/dqc")
sys.path.insert(0,"/mmfs1/home/harine/RoboMonkey/diffusion_policy")
import numpy as np, torch
print("torch cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
import jax
print("jax devices:", jax.devices())
# a real torch GPU op + jax GPU op in the same process (cudnn coexistence)
x = torch.randn(1024,1024, device="cuda"); print("torch matmul ok:", float((x@x).sum().abs()>0))
import jax.numpy as jnp; y = jnp.ones((1024,1024)); print("jax matmul ok:", float((y@y).sum())>0)

from diffusion_policy.policy.dqc_verifier import DQCVerifier
CK="/gscratch/robotics/harine/dqc_checkpoint/sd000s_137913.0.20260605_163017"
v = DQCVerifier(ckpt_dir=CK, instruction="put the eggplant in the basket",
                image_obs_key="agentview_image", critic_type="chunk",
                score_full_horizon=True, health_check=True)
# score a few real eggplant 8-step chunks
import zarr
z = zarr.open("/gscratch/robotics/harine/data/eggplant_in_basket_complete/state0.zarr","r")
frames = np.stack([np.asarray(z["data"]["obs"]["agentview_image"][i]) for i in range(0,160,10)])  # (16,H,W,3)
chunks = np.stack([np.asarray(z["data"]["actions"][i:i+8]) for i in range(0,160,10)]).astype(np.float32)  # (16,8,7)
N=len(frames)
obs={"agentview_image":torch.from_numpy(frames).unsqueeze(1)}        # (N,1,H,W,3)
v.score_indices=list(range(8))
q=v.get_value(obs, torch.from_numpy(chunks)).cpu().numpy()
print(f"CHUNK Q: N={N} min/mean/max={q.min():.3f}/{q.mean():.3f}/{q.max():.3f} std={q.std():.4f}")
print("VALIDATE_OK" if np.isfinite(q).all() and q.std()>0.05 else "VALIDATE_WEAK")
