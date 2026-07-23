#!/bin/bash
#SBATCH --job-name=rmky_inproc
#SBATCH --account=socialrl
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --output=/mmfs1/home/harine/RoboMonkey/data/eval/robomonkey_repro/slurm-inprocess-%j.out
# -----------------------------------------------------------------------------
# RoboMonkey reproduction, IN-PROCESS verifier (reward_server_port=0).
# Single H200: sglang OpenVLA policy server (mem-fraction capped) co-located
# with the eval process (SAPIEN + in-process LLaVA verifier). Override at
# submit time, e.g.:
#   sbatch --export=ALL,SEEDS="1",NUM_TRIALS=2 eval_inprocess.sbatch   # smoke
# -----------------------------------------------------------------------------
export MODE=inprocess
export TASK="${TASK:-simpler_put_eggplant_in_basket}"
export SEEDS="${SEEDS:-1 2 3}"
export INIT_SAMPLES="${INIT_SAMPLES:-9}"
export AUG_SAMPLES="${AUG_SAMPLES:-32}"
export NUM_TRIALS="${NUM_TRIALS:-50}"

bash /mmfs1/home/harine/RoboMonkey/scriptsv2/robomonkey_repro/run_robomonkey_eval.sh
