#!/bin/bash
# Launch the sglang OpenVLA action server (port 3200) on an H200 (sm90) node.
#
# Differs from scripts/run_openvla_server.sh in one way that matters on H200:
# FlashInfer JIT-compiles sm90 kernels on first launch, and that compile needs
# GCC >= 9 (system gcc here is 8.5). We use the sglang-vla conda env's own
# gcc 11.4 (x86_64-conda-linux-gnu-gcc) as the nvcc host compiler, which also
# avoids libstdc++ path conflicts. CUDA 12.6 toolkit comes from the env's
# `cuda_home` symlink (-> /sw/cuda/12.6.3), matching torch 2.7.1+cu126.
#
# Usage: CUDA_VISIBLE_DEVICES=0 SEED=1 bash run_openvla_server_h200.sh
set -e

export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

GPU=${CUDA_VISIBLE_DEVICES:-0}
SEED=${SEED:-1}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
set +e
conda activate sglang-vla
set -e

# --- CUDA 12.6 toolkit (matches torch cu126) -------------------------------
export CUDA_HOME="$CONDA_PREFIX/cuda_home"
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "ERROR: nvcc not found at $CUDA_HOME/bin/nvcc."
    echo "Create the symlink once:  ln -sfn /sw/cuda/12.6.3 $CONDA_PREFIX/cuda_home"
    exit 1
fi
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# --- host compiler for FlashInfer sm90 JIT (needs gcc>=9) -------------------
CONDA_CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
CONDA_CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
if [ -x "$CONDA_CXX" ]; then
    export CC="$CONDA_CC"
    export CXX="$CONDA_CXX"
    export NVCC_PREPEND_FLAGS="-ccbin $CONDA_CXX"
else
    # fallback: cluster gcc module install (gcc 11.2)
    GCC_PREFIX="${GCC_PREFIX:-/sw/gcc/11.2.0}"
    export CC="$GCC_PREFIX/bin/gcc"
    export CXX="$GCC_PREFIX/bin/g++"
    export PATH="$GCC_PREFIX/bin:$PATH"
    export LD_LIBRARY_PATH="$GCC_PREFIX/lib64:$LD_LIBRARY_PATH"
    export NVCC_PREPEND_FLAGS="-ccbin $CXX"
fi
unset NVCC_APPEND_FLAGS

full_path=$(realpath "$0")
dir_path=$(dirname "$full_path")
cd "$dir_path/../../sglang-vla"

echo "CUDA_HOME=$CUDA_HOME"
echo "nvcc: $(which nvcc)  ($(nvcc --version | grep -oE 'release [0-9.]+'))"
echo "host gcc (CC): $CC  ($($CC --version | head -1))"
echo "GPU: $GPU   seed: $SEED"
echo

exec env CUDA_VISIBLE_DEVICES="$GPU" python openvla_server.py --seed "$SEED"
