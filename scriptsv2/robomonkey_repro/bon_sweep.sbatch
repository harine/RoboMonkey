#!/bin/bash
#SBATCH --job-name=rmky_bon
#SBATCH --account=socialrl
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=20:00:00
#SBATCH --output=/mmfs1/home/harine/RoboMonkey/data/eval/robomonkey_repro/slurm-bon-%j.out
# -----------------------------------------------------------------------------
# RoboMonkey Best-of-N scaling sweep: VLA policy + in-process verifier.
# Single H200 (sglang server mem-fraction-capped, co-located with eval+verifier).
# N = 2^0..2^10, fixed initial_samples, 50 trials, seed 1, eggplant.
#   sbatch --export=ALL,NUM_TRIALS=2,N_LIST="1 1024" bon_sweep.sbatch   # smoke
# -----------------------------------------------------------------------------
export TASK="${TASK:-simpler_put_eggplant_in_basket}"
export SEED="${SEED:-1}"
export INIT_SAMPLES="${INIT_SAMPLES:-9}"
export NUM_TRIALS="${NUM_TRIALS:-50}"
export N_LIST="${N_LIST:-1 2 4 8 16 32 64 128 256 512 1024}"
export REWARD_BATCH_SIZE="${REWARD_BATCH_SIZE:-64}"

bash /mmfs1/home/harine/RoboMonkey/scriptsv2/robomonkey_repro/run_bon_sweep.sh
