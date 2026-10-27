#!/bin/bash
#SBATCH --job-name=veritas_prep
#SBATCH --account=ckpt-robotics
#SBATCH --partition=ckpt
#SBATCH --qos=ckpt-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|l40s|l40|a100
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=6:00:00
#SBATCH --output=data/veritas/_logs/slurm-%x-%j.out
#SBATCH --error=data/veritas/_logs/slurm-%x-%j.out
# Run one SimplerEnv-backed Veritas prep command on a GPU node (SAPIEN needs Vulkan;
# the login nodes have none). CMD is the command line; CONDA_ENV defaults to simpler_env.
#   sbatch --job-name=cam_egg --export=ALL,CMD="python scriptsv2/veritas/capture_simpler_camera.py --task widowx_put_eggplant_in_basket" scriptsv2/veritas/prep.sbatch
set -euo pipefail
# Gemini key: from the environment, else the user's key file (never passed via --export,
# which would expose it in the job's recorded environment).
if [ -z "${GEMINI_API_KEY:-}" ] && [ -f "$HOME/.config/gemini/api_key" ]; then
  export GEMINI_API_KEY="$(tr -d '[:space:]' < "$HOME/.config/gemini/api_key")"
fi
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
CONDA_ENV="${CONDA_ENV:-simpler_env}"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa DISPLAY=""
export LD_LIBRARY_PATH="${HOME}/miniconda3/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/diffusion_policy:${PYTHONPATH:-}"
export PYTHONWARNINGS="ignore"
nvidia-smi -L || true
echo "[prep] env=$CONDA_ENV cmd: ${CMD:?}"
eval "$CMD" 2> >(grep -v -E "GLFW|svulkan2|Continue without" >&2)
echo "[prep] done"
