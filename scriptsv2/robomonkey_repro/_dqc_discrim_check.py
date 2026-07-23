"""Standalone sanity check: does the DQC critic discriminate between candidate
actions on a real eval frame? Loads the same DQCVerifier the OpenVLA repro uses,
scores a spread of bridge-space actions on the smoke run's saved reward_img.jpg,
and prints the Q spread. A near-flat spread => verifier not separating candidates.
"""
import os, sys
import numpy as np
import torch

dp_root = os.environ.get("DIFFUSION_POLICY_ROOT")
if dp_root and dp_root not in sys.path:
    sys.path.insert(0, dp_root)
from diffusion_policy.policy.dqc_verifier import DQCVerifier
from PIL import Image

img_path = sys.argv[1] if len(sys.argv) > 1 else "openvla-mini/transfer_images/reward_img.jpg"
img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
print(f"frame: {img_path}  shape={img.shape}")

v = DQCVerifier(
    ckpt_dir=os.environ["DQC_CKPT_DIR"],
    instruction=os.environ.get("VERIFIER_INSTRUCTION", "put the eggplant in the basket"),
    image_obs_key="agentview_image",
    text_emb_file=os.environ.get("DQC_TEXT_EMB"),
    discount=0.98,
    health_check=True,
)

# A spread of candidate actions covering the bridge augmented-sample range.
rng = np.random.RandomState(0)
cands = []
labels = []
cands.append(np.zeros(7, np.float32)); labels.append("zeros")
cands.append(np.array([0.03, 0, 0, 0, 0, 0, 1], np.float32)); labels.append("+x open")
cands.append(np.array([-0.03, 0, 0, 0, 0, 0, 1], np.float32)); labels.append("-x open")
cands.append(np.array([0, 0, -0.03, 0, 0, 0, 0], np.float32)); labels.append("-z close")
cands.append(np.array([0, 0, 0.03, 0, 0, 0, 1], np.float32)); labels.append("+z open")
# 16 random augmented-style candidates around a small downward grasp motion.
base = np.array([0.005, 0.0, -0.01, 0, 0, 0, 0.0], np.float32)
for _ in range(16):
    a = base + rng.normal(0, [0.02, 0.02, 0.02, 0.08, 0.08, 0.15, 0], 7).astype(np.float32)
    a[-1] = float(rng.rand() > 0.5)
    cands.append(a.astype(np.float32)); labels.append("rand")
acts = np.stack(cands)                       # (N, 7)
N = acts.shape[0]

img_t = torch.from_numpy(img.copy())
obs = {"agentview_image": img_t.unsqueeze(0).unsqueeze(0).expand(N, 1, *img.shape)}
q = v.get_value(obs, torch.from_numpy(acts).unsqueeze(1)).cpu().numpy()

order = np.argsort(-q)
print(f"\nN={N}  Q: min={q.min():.3f} mean={q.mean():.3f} max={q.max():.3f} "
      f"std={q.std():.3f}  range={q.max()-q.min():.3f}")
print("top-5 by Q:")
for i in order[:5]:
    print(f"  Q={q[i]:7.3f}  {labels[i]:8s}  {np.array2string(acts[i], precision=3)}")
print("bottom-3 by Q:")
for i in order[-3:]:
    print(f"  Q={q[i]:7.3f}  {labels[i]:8s}  {np.array2string(acts[i], precision=3)}")
