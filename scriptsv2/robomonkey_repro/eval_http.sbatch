#!/bin/bash
#SBATCH --job-name=rmky_http
#SBATCH --account=socialrl
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=16:00:00
#SBATCH --output=/mmfs1/home/harine/RoboMonkey/data/eval/robomonkey_repro/slurm-http-%j.out
# -----------------------------------------------------------------------------
# RoboMonkey reproduction, HTTP verifier (the README path, reward_server_port=3100).
# GPU0 = sglang OpenVLA policy server; GPU1 = infer_server.py LLaVA verifier +
# eval process (SAPIEN), co-located (H200 has room). Override at submit time, e.g.:
#   sbatch --export=ALL,SEEDS="1",NUM_TRIALS=2 eval_http.sbatch        # smoke
# -----------------------------------------------------------------------------
export MODE=http
export TASK="${TASK:-simpler_put_eggplant_in_basket}"
export SEEDS="${SEEDS:-1 2 3}"
export INIT_SAMPLES="${INIT_SAMPLES:-9}"
export AUG_SAMPLES="${AUG_SAMPLES:-32}"
export NUM_TRIALS="${NUM_TRIALS:-50}"

bash /mmfs1/home/harine/RoboMonkey/scriptsv2/robomonkey_repro/run_robomonkey_eval.sh
