#!/bin/bash
#SBATCH --job-name=rmky_dqc
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/mmfs1/home/harine/RoboMonkey/data/eval/robomonkey_repro/slurm-dqc-%j.out
# -----------------------------------------------------------------------------
# RoboMonkey reproduction with the DQC Q-function verifier (reward_server_port=0).
#
# Identical to eval_inprocess.sbatch EXCEPT the verifier: OpenVLA-7B policy
# (sglang) generates the same init+aug candidate set, but Best-of-N selection is
# done by the in-process DQC chunk Q-critic instead of the LLaVA token verifier.
# Everything else (task, init=9, aug=32, 50 trials, seeds 1/2/3, argmax select)
# matches the LLaVA run, so results are directly comparable to
#   data/eval/robomonkey_repro/inprocess  (LLaVA: 70 / 72 / 66 %).
#
# The eval runs in simpler_env (the only env with both JAX [DQC] and torch
# [SAPIEN]); the DQC critic loads in-process, sharing the H200 with the sglang
# policy server (both mem-fraction capped).
#
# Smoke:  sbatch --export=ALL,SEEDS="1",NUM_TRIALS=2 eval_dqc.sbatch
# Full:   sbatch eval_dqc.sbatch
# -----------------------------------------------------------------------------
export MODE=dqc
export TASK="${TASK:-simpler_put_eggplant_in_basket}"
export SEEDS="${SEEDS:-1 2 3}"
export INIT_SAMPLES="${INIT_SAMPLES:-9}"
export AUG_SAMPLES="${AUG_SAMPLES:-32}"
export NUM_TRIALS="${NUM_TRIALS:-50}"

# DQC verifier checkpoint (bridge_dqc_chunk, discount 0.98) + eggplant MUSE emb.
export DQC_CKPT_DIR="${DQC_CKPT_DIR:-/gscratch/robotics/harine/dqc_checkpoint/sd000s_137913.0.20260605_163017}"
export DQC_TEXT_EMB="${DQC_TEXT_EMB:-/mmfs1/home/harine/RoboMonkey/scriptsv2/q_fn_eval/task_emb.npy}"
export DQC_DISCOUNT="${DQC_DISCOUNT:-0.98}"

bash /mmfs1/home/harine/RoboMonkey/scriptsv2/robomonkey_repro/run_robomonkey_eval.sh
