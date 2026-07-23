#!/bin/bash
# Build the `dqc` conda env: ONE env that runs BOTH the diffusion-policy trainer
# (torch) AND the REAL DQC chunk-critic verifier (flax/jax BridgeDQCChunkAgent),
# in-process, sharing the GPU. Used by scriptsv2/train_search/train_dqc_chunk.sbatch
# (CONFIG=robomonkey_eggplant_search_dqc[_gaussian]).
#
# Why a bespoke env (none of the existing ones work):
#   - dqc_jax:     torch + jax 0.6.2  -> jax too new, breaks the repo's flax/distrax
#                  (`jax.experimental.maps` removed).
#   - dqc_repo:    correct jax0.4.28/flax0.8.5 but no torch / no trainer deps.
#   - simpler_env: torch + jax0.6.2 + flax0.8.1 (incompatible) and 21G (the sim).
#
# Hard-won pin lessons (do NOT relax without re-testing):
#   * Python 3.11  -> hydra-core MUST be 1.3.2 (1.2.0 fails to import: "mutable
#       default ... override_dirname" dataclass error).
#   * torch cu121  -> its cuDNN 8.9 MATCHES jax 0.4.28's, so they share one
#       libcudnn (no torch-cuDNN8 / jax-cuDNN9 split like dqc_jax needed). Do NOT
#       use a cu118 torch or a torch>=2.4 (pulls cuDNN 9) alongside jax 0.4.28.
#   * numpy MUST stay <2 (jax 0.4.28 + numba need it). Therefore pin AWAY from
#       pandas>=3, scikit-image>=0.23, opencv>=4.10, imagecodecs — they hard-require
#       numpy>=2 (and imagecodecs has no py311 wheel). Use the pinned versions below.
#   * Do NOT install robomimic (pulls egl_probe, which fails to build a wheel) — the
#       FlattenObsEncoder / search-policy training path doesn't need it.
#   * The diffusion_policy package and the dqc submodule are used via PYTHONPATH /
#       sys.path (the launcher sets DIFFUSION_POLICY_ROOT; dqc_verifier.py adds the
#       dqc submodule), so they are NOT pip-installed here.
#
# Usage:  bash scriptsv2/setup_dqc_env.sh          # creates env `dqc`
#         CONDA_ENV=dqc_repo bash ...              # or extend an existing env in place
set -euo pipefail
ENV_NAME="${CONDA_ENV:-dqc}"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

if ! conda env list | grep -qE "^${ENV_NAME} "; then
  echo "[setup] creating env ${ENV_NAME} (python 3.11) ..."
  conda create -y -n "${ENV_NAME}" python=3.11
fi
conda activate "${ENV_NAME}"

echo "[setup] 1/3 DQC verifier stack (jax/flax/distrax — repo-compatible) ..."
pip install --no-input "jax[cuda12]==0.4.28" "flax==0.8.5" "distrax==0.1.5" \
  "optax==0.2.2" "chex==0.1.86" "ml_collections" "tensorflow-probability==0.24.0"

echo "[setup] 2/3 torch 2.2.0 + cu121 (cuDNN 8.9, matches jax0.4.28) ..."
pip install --no-input torch==2.2.0 torchvision==0.17.0 \
  --index-url https://download.pytorch.org/whl/cu121

echo "[setup] 3/3 diffusion-policy trainer deps (numpy<2; py311-safe pins) ..."
pip install --no-input \
  "numpy==1.26.4" "diffusers==0.33.1" "transformers==4.31.0" \
  "hydra-core==1.3.2" "omegaconf==2.3.0" "einops==0.6.1" "wandb" "dill==0.3.8" \
  "numba==0.61.2" "scipy" "threadpoolctl==3.6.0" "tqdm" "accelerate" \
  "huggingface_hub" "safetensors" \
  "matplotlib==3.8.4" "pandas==2.1.4" "opencv-python-headless==4.9.0.80" \
  "zarr==2.18.3" "numcodecs==0.13.1"

echo "[setup] verifying torch + jax + repo agent + trainer all import together ..."
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JAX_PLATFORMS=cpu python - "$REPO" <<'PY'
import sys
repo = sys.argv[1]
sys.path.insert(0, f"{repo}/dqc"); sys.path.insert(0, f"{repo}/diffusion_policy")
import numpy, torch, jax, flax, distrax            # noqa
from agents.bridge_dqc_chunk import BridgeDQCChunkAgent           # noqa
from diffusion_policy.workspace.train_mlp_image_workspace import TrainMLPImageWorkspace  # noqa
from diffusion_policy.policy.dqc_verifier import DQCVerifier      # noqa
assert numpy.__version__.startswith("1.26"), numpy.__version__
print(f"OK: torch {torch.__version__} | jax {jax.__version__} | flax {flax.__version__} "
      f"| numpy {numpy.__version__}")
PY
echo "[setup] DONE — env '${ENV_NAME}' ready for train_dqc_chunk.sbatch (CONDA_ENV=${ENV_NAME})."
