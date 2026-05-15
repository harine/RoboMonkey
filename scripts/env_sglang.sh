#!/bin/bash

# Exit on error
set -e

# Ensure loopback traffic bypasses the cluster's HTTP proxy (e.g. Squid on
# Klone). Outbound proxy is preserved so pip/conda still work.
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

echo "Starting setup process..."

# Function to check command status
check_status() {
    if [ $? -eq 0 ]; then
        echo "✓ $1 completed successfully"
    else
        echo "✗ Error during $1"
        exit 1
    fi
}

# Initialize conda in the current shell
echo "Initializing conda..."
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc

full_path=$(realpath $0)
dir_path=$(dirname $full_path)

echo "Creating sglang-vla environment..."
# Create and activate environment
if ! conda env list | grep -qE "^\s*sglang-vla\s"; then
    $HOME/miniconda3/bin/conda create -n sglang-vla python=3.10 -y
else
    echo "Conda environment 'sglang-vla' already exists. Skipping creation."
fi

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate sglang-vla

cd "$dir_path/../sglang-vla"
pip install --upgrade pip
pip install -e "python[all]"
pip install timm==0.9.10
pip install json_numpy
pip install flask
sudo apt-get update
sudo apt-get install -y libnuma1 numactl
check_status "SGLang setup"
