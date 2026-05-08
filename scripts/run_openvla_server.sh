#!/bin/bash
# Launch the OpenVLA sglang action server on port 3200 (foreground).
#
# Usage:
#   bash scripts/run_openvla_server.sh                  # default: GPU 0, seed 1
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_openvla_server.sh
#   SEED=42 bash scripts/run_openvla_server.sh
#
# First launch JIT-compiles FlashInfer kernels for your GPU (~60-90 s); later
# launches hit the cache at ~/.cache/flashinfer/ and start much faster.

set -e

GPU=${CUDA_VISIBLE_DEVICES:-0}
SEED=${SEED:-1}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
# Conda's cuda-nvcc activation hook returns non-zero when CXX is unset
# (just a warning about missing conda compiler packages; harmless for our
# setup since we fall back to system gcc below). Guard with `set +e` so the
# non-zero exit doesn't abort the launcher.
set +e
conda activate sglang-vla
set -e

# Point torch's cpp_extension / FlashInfer JIT at the CUDA 12.6 toolkit that
# was installed into this env (matches the torch 2.7.1+cu126 ABI).
if [ ! -x "$CONDA_PREFIX/cuda_home/bin/nvcc" ]; then
    echo "ERROR: $CONDA_PREFIX/cuda_home/bin/nvcc not found."
    echo "Install CUDA 12.6 into the env first, e.g.:"
    echo "  conda install -c nvidia --override-channels 'cuda-version=12.6' \\"
    echo "      cuda-nvcc cuda-cudart-dev cuda-nvrtc-dev cuda-cccl cuda-driver-dev"
    echo "  mkdir -p \$CONDA_PREFIX/cuda_home"
    echo "  ln -sf \$CONDA_PREFIX/bin                         \$CONDA_PREFIX/cuda_home/bin"
    echo "  ln -sf \$CONDA_PREFIX/targets/x86_64-linux/include \$CONDA_PREFIX/cuda_home/include"
    echo "  ln -sf \$CONDA_PREFIX/targets/x86_64-linux/lib     \$CONDA_PREFIX/cuda_home/lib64"
    exit 1
fi

# Conda's cuda-nvcc activation script injects a -ccbin= via NVCC_PREPEND_FLAGS
# and points CC/CXX at x86_64-conda-linux-gnu-* binaries that aren't installed.
# Clear both so torch falls back to the system gcc (11.4 on Ubuntu 22.04).
unset NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS CC CXX

export CUDA_HOME="$CONDA_PREFIX/cuda_home"
export PATH="$CUDA_HOME/bin:/usr/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

full_path=$(realpath "$0")
dir_path=$(dirname "$full_path")
cd "$dir_path/../sglang-vla"

echo "CUDA_HOME=$CUDA_HOME"
echo "nvcc:  $(which nvcc)"
echo "gcc:   $(which gcc)"
echo "GPU:   $GPU   seed: $SEED"
echo

exec env CUDA_VISIBLE_DEVICES="$GPU" python openvla_server.py --seed "$SEED"
