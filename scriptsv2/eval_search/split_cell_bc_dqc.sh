#!/bin/bash
# Split one BC+DQC eval cell into K parallel seed-range parts so 50 episodes
# finish in ~1/K the wall-clock (e.g. to land before a maintenance window).
# Each part writes data/eval/bc_dqc/h16_n<N>_disc098/part<k>/. Combine the
# parts' episodes.jsonl afterwards (merge_cells_bc_dqc.sh / report loop).
#
#   N=64 K=2 bash scriptsv2/eval_search/split_cell_bc_dqc.sh
#   N=256 K=4 WALL=08:30:00 bash scriptsv2/eval_search/split_cell_bc_dqc.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"

N="${N:?set N (search width)}"
K="${K:-2}"
TOTAL="${TOTAL:-50}"
START="${START_SEED:-1000}"
WALL="${WALL:-08:30:00}"                 # must end before the 2026-06-16 09:00 maint
ACCT="${ACCT:-robotics-ckpt}"; PART="${PART:-ckpt}"; GRES="${GRES:-gpu:a40:1}"
CKPT="${CHECKPOINT:-diffusion_policy/data/outputs/2026.06.11/13.52.09_robomonkey_eggplant_bc_robomonkey_eggplant_state/checkpoints/best.ckpt}"

base=$(( TOTAL / K )); rem=$(( TOTAL % K )); off=0
for k in $(seq 0 $((K-1))); do
    cnt=$base; [ "$k" -lt "$rem" ] && cnt=$((base+1))
    ss=$(( START + off )); off=$(( off + cnt ))
    out="data/eval/bc_dqc/h16_n${N}_disc098/part${k}"
    jid=$(sbatch --parsable --time="$WALL" \
        --account="$ACCT" --partition="$PART" --gres="$GRES" --requeue \
        --job-name="bcdqc_n${N}_p${k}" \
        --export=ALL,CHECKPOINT="$CKPT",NUM_EPISODES="$cnt",START_SEED="$ss",N_SAMPLES="$N",OUT_DIR="$out" \
        scriptsv2/eval_search/eval_bc_dqc.sbatch)
    echo "n=$N part$k: seeds ${ss}..$((ss+cnt-1)) (${cnt} eps) -> job ${jid} -> ${out}"
done
