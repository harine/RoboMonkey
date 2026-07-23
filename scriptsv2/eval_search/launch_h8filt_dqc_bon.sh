#!/bin/bash
# -----------------------------------------------------------------------------
# DQC chunk-8 verifier Best-of-N sweep on the H8 FILTERED BC (latest.ckpt = 50k).
#   predict 8 / rollout 4 / DQC chunk critic scores the full predicted 8-chunk
#   (BC_CHUNK_FROM_PRED=1), ddpm sampler (diverse candidates), softmax selection
#   by DEFAULT (set MODE=argmax for classic best-of-N), 50 episodes seeds
#   1000-1049. One SLURM job per N (eval_bc_dqc_chunk.sbatch, dqc_jax env).
#   Idempotent: skips cells already complete or queued; routes across free GPUs.
#   MODE is encoded in the cell label so softmax/argmax results never collide.
#
#   bash scriptsv2/eval_search/launch_h8filt_dqc_bon.sh          # submit gaps (softmax)
#   MODE=argmax bash .../launch_h8filt_dqc_bon.sh                # classic best-of-N
#   DRY=1 bash .../launch_h8filt_dqc_bon.sh                      # report only
#   N_LIST="1 8 64" bash .../launch_h8filt_dqc_bon.sh            # custom N set
# Results: data/eval/bc_dqc_chunk/bcH8filt50k_chunk8_predh8_roll4_ddpm_<mode>_n<N>_seed1000_ep50/
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey

SB=scriptsv2/eval_search/eval_bc_dqc_chunk.sbatch
CKPT="$PWD/diffusion_policy/data/train/bc_denoiser/eggplant_bc_unet_h8_50k_filtered/checkpoints/latest.ckpt"
N_LIST="${N_LIST:-1 2 4 8 16 32 64}"
NUM_EPISODES="${NUM_EPISODES:-50}"
START_SEED="${START_SEED:-1000}"
MODE="${MODE:-softmax}"                 # selection: softmax (default) | argmax (best-of-N)
SOFTMAX_TEMP="${SOFTMAX_TEMP:-1.0}"
LABEL="${LABEL:-bcH8filt50k_chunk8_predh8_roll4_ddpm_${MODE}}"
DRY="${DRY:-0}"

ROUTES=( "robotics|gpu-a40|gpu:a40:1" "weirdlab|gpu-l40|gpu:l40:1" "weirdlab|gpu-l40s|gpu:l40s:1" )
[[ -f "$CKPT" ]] || { echo "ERROR: ckpt not found: $CKPT" >&2; exit 1; }

QNAMES="$(squeue -u "$USER" -h -o '%j')"
i=0; sub=0; sd=0; sq=0
done_cell(){ python3 -c "import json,sys
try: sys.exit(0 if int(json.load(open('$1/eval_log.json')).get('num_episodes',0))>=$NUM_EPISODES else 1)
except Exception: sys.exit(1)" 2>/dev/null; }

for N in $N_LIST; do
  out="data/eval/bc_dqc_chunk/${LABEL}_n${N}_seed${START_SEED}_ep${NUM_EPISODES}"
  jn="h8filt_dqc_${MODE}_n${N}"
  if done_cell "$out"; then echo "[done] n=$N ($out)"; sd=$((sd+1)); continue; fi
  if grep -qxF "$jn" <<<"$QNAMES"; then echo "[queued] $jn"; sq=$((sq+1)); continue; fi
  IFS='|' read -r a p g <<< "${ROUTES[$((i % ${#ROUTES[@]}))]}"; i=$((i+1))
  if [[ "$DRY" == "1" ]]; then echo "WOULD submit $jn (N=$N, $a/$p) -> $out"; sub=$((sub+1)); continue; fi
  jid=$(sbatch --parsable --job-name="$jn" --account="$a" --partition="$p" --gres="$g" \
        --export=ALL,CHECKPOINT="$CKPT",N_SAMPLES="$N",NUM_EPISODES="$NUM_EPISODES",START_SEED="$START_SEED",BC_SAMPLER=ddpm,MODE="$MODE",SOFTMAX_TEMP="$SOFTMAX_TEMP",OUT_DIR="$out" \
        "$SB")
  echo "submitted $jn -> $jid ($a/$p, N=$N)"; sub=$((sub+1)); sleep 1
done
echo "=== h8filt DQC BoN: submitted=$sub done=$sd queued=$sq ==="
echo "results -> data/eval/bc_dqc_chunk/${LABEL}_n<N>_seed${START_SEED}_ep${NUM_EPISODES}/eval_log.json"
