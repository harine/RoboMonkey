#!/bin/bash
# Sweep the BC + DQC discounted-chunk-sum eval over search width N = 2^0 .. 2^8.
# n=16 is skipped by default (already run -> data/eval/bc_dqc/h16_n16_disc098).
# One SLURM cell per N, round-robined across the cse / weirdlab / robotics GPU
# pools for parallelism. Each cell: horizon-16 BC policy, n_action_steps=8,
# DQC discounted chunk sum (gamma=0.98), argmax best-of-N, NUM_EPISODES episodes.
#
#   bash scriptsv2/eval_search/sweep_bc_dqc.sh
#   NS="128 256" bash scriptsv2/eval_search/sweep_bc_dqc.sh   # resume just these
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"

CKPT="${CHECKPOINT:-diffusion_policy/data/outputs/2026.06.11/13.52.09_robomonkey_eggplant_bc_robomonkey_eggplant_state/checkpoints/best.ckpt}"
NUM_EPISODES="${NUM_EPISODES:-50}"
NS="${NS:-1 2 4 8 32 64 128 256}"     # 2^0..2^8 minus 16
WALL="${WALL:-14:00:00}"              # fits before the 2026-06-16 09:00 maintenance
mkdir -p data/eval/bc_dqc

# pool: "account partition gres"
POOLS=("cse gpu-l40s l40s" "weirdlab gpu-l40 l40" "robotics gpu-a40 a40")
i=0
for n in $NS; do
    read -r acct part gres <<< "${POOLS[$((i % ${#POOLS[@]}))]}"
    out="data/eval/bc_dqc/h16_n${n}_disc098"
    jid=$(sbatch --parsable --time="$WALL" \
        --account="$acct" --partition="$part" --gres="gpu:${gres}:1" \
        --job-name="bcdqc_n${n}" \
        --export=ALL,CHECKPOINT="$CKPT",NUM_EPISODES="$NUM_EPISODES",N_SAMPLES="$n",OUT_DIR="$out" \
        scriptsv2/eval_search/eval_bc_dqc.sbatch)
    echo "n=${n}  job ${jid}  ${acct}/${part}  -> ${out}"
    i=$((i + 1))
done
echo "submitted ${NS} (skipped 16). track: squeue -u $USER -n $(echo $NS | sed 's/ /,/g' | sed 's/[0-9]*/bcdqc_n&/g')"
