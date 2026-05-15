#!/bin/bash
# Install SimplerEnv + diffusion_policy eval deps INTO the existing
# `monkey-verifier` conda env, so the BoN eval can use the in-process
# RoboMonkey verifier (REWARD_SERVER_PORT=0) without spinning up infer_server.
#
# Idempotent: re-running just hits the pip cache.
#
# Usage:
#   bash scripts/install_verifier_eval_deps.sh
#
# Notes:
#   * monkey-verifier pins transformers==4.31, peft==0.4, torch==2.0.1+cu118,
#     numpy==1.26.4. We do NOT touch those. If diffusion_policy needs newer,
#     re-pinning is your problem.
#   * SimplerEnv's own setup pins numpy==1.24.4; we skip its pip block and
#     install only the runtime modules diffusion_policy + eval_diffusion.py
#     actually import.

set -euo pipefail

export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

repo_root="$(cd "$(dirname "$(realpath "$0")")/.." && pwd)"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
if ! conda env list | grep -qE "^\s*monkey-verifier\s"; then
    echo "ERROR: monkey-verifier conda env not found. Run scripts/env_verifier.sh first." >&2
    exit 1
fi
conda activate monkey-verifier

echo "[install] active env: $CONDA_DEFAULT_ENV ($(python --version))"

# ------------------------------------------------------------------
# 1. SimplerEnv + ManiSkill2_real2sim (editable installs only; no requirements_full_install
#    which would clobber pinned torch/transformers/numpy).
# ------------------------------------------------------------------
cd "$repo_root/SimplerEnv"
pip install --no-deps -e ./ManiSkill2_real2sim/
pip install --no-deps -e .

# ------------------------------------------------------------------
# 2. diffusion_policy runtime deps (hydra, dill, diffusers in the supported
#    range, plus a few odds-and-ends eval_diffusion.py imports).
# ------------------------------------------------------------------
pip install \
    "hydra-core==1.2.0" \
    "dill" \
    "diffusers>=0.30,<0.34" \
    "omegaconf" \
    "einops" \
    "imageio[ffmpeg]" \
    "mediapy" \
    "requests" \
    "opencv-python<4.11" \
    "zarr<3" \
    "numcodecs<0.14"

# ------------------------------------------------------------------
# 3. SimplerEnv runtime deps that are NOT in monkey-verifier already.
#    Skip anything that would downgrade torch/transformers/numpy.
# ------------------------------------------------------------------
pip install --no-deps \
    "sapien>=2.2.2,<3" \
    "trimesh" \
    "rtree" \
    "pyglet<2" \
    "shapely" \
    "h5py" \
    "gymnasium" \
    "scipy"

# ------------------------------------------------------------------
# 4. Smoke imports (run each in a fresh subprocess so a partial failure
#    doesn't poison sys.modules).
# ------------------------------------------------------------------
python - <<'PY'
import subprocess, sys
for mod in ["sapien", "mani_skill2_real2sim", "simpler_env",
            "diffusers", "hydra", "dill", "omegaconf",
            "verifier_client"]:
    code = "import sys; sys.path.insert(0, 'monkey-verifier/src'); import " + mod
    r = subprocess.run([sys.executable, "-c", code])
    status = "ok" if r.returncode == 0 else "FAIL"
    print(f"[install] import {mod}: {status}")
    if r.returncode != 0:
        sys.exit(1)
PY

echo
echo "[install] done. To use, set CONDA_ENV=monkey-verifier when invoking eval_diffusion.sh."
