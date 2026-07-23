#!/bin/bash
#SBATCH --job-name=rmky_bondqc
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=13:00:00
#SBATCH --output=/mmfs1/home/harine/RoboMonkey/data/eval/robomonkey_repro/slurm-bondqc-%j.out
# -----------------------------------------------------------------------------
# RoboMonkey Best-of-N test-time scaling sweep with the DQC Q-function verifier.
# OpenVLA-7B policy (sglang) generates init=9 + augmented=N candidates; the
# in-process DQC critic scores all N and argmax-selects. Sweeps N=2^0..2^9.
# Directly comparable to the LLaVA sweep at data/eval/robomonkey_repro/bon_sweep
# (which peaks ~76% at N=64 then degrades).
#
# RESUMABLE: progress.json accumulates (success, episodes) per N across relaunches
# and checkpoints every BLOCK_TRIALS=10 episodes, so a wall-clock cutoff (e.g. the
# Jun16 maintenance) loses at most 10 episodes. Re-submit after maintenance to
# finish the remaining N points / top up to the target trials.
#
# Runs in simpler_env (JAX+torch). One sglang server stays up for the whole sweep.
# Smoke:  sbatch --export=ALL,N_LIST="1 4",NUM_TRIALS=2 bon_sweep_dqc.sbatch
# Full:   sbatch bon_sweep_dqc.sbatch
# -----------------------------------------------------------------------------
export VERIFIER_TYPE=dqc
export TASK="${TASK:-simpler_put_eggplant_in_basket}"
export SEED="${SEED:-1}"
export INIT_SAMPLES="${INIT_SAMPLES:-9}"
export NUM_TRIALS="${NUM_TRIALS:-50}"
export N_LIST="${N_LIST:-1 2 4 8 16 32 64 128 256 512}"
export N_LIST="${N_LIST//[,_]/ }"   # accept '_' or ',' separated (parallel launcher) or space
export DQC_CKPT_DIR="${DQC_CKPT_DIR:-/gscratch/robotics/harine/dqc_checkpoint/sd000s_137913.0.20260605_163017}"
export DQC_TEXT_EMB="${DQC_TEXT_EMB:-/mmfs1/home/harine/RoboMonkey/scriptsv2/q_fn_eval/task_emb.npy}"
# Own working dir so it can run alongside the other RoboMonkey jobs race-free.
export EVAL_CWD="${EVAL_CWD:-/mmfs1/home/harine/RoboMonkey/data/eval/robomonkey_repro/bon_sweep_dqc/_cwd}"

bash /mmfs1/home/harine/RoboMonkey/scriptsv2/robomonkey_repro/run_bon_sweep.sh
