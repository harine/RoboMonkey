#!/bin/bash
# -----------------------------------------------------------------------------
# DQC chunk-8 verifier Best-of-N sweep on an ARBITRARY BC checkpoint.
#   predict-8 / rollout-4 / DQC chunk critic scores the full predicted 8-chunk
#   (BC_CHUNK_FROM_PRED=1 — CLEAN a_{t:t+8} when the ckpt is n_obs=1), ddpm
#   sampler (diverse candidates), MODE default softmax, 50 ep seeds 1000-1049.
#   One SLURM job per N (eval_bc_dqc_chunk.sbatch, dqc_jax env), routed to
#   robotics/gpu-a40. Idempotent: skips done/queued cells.
#
# Env:
#   CKPT     (required) absolute path to the BC checkpoint
#   LABEL    (required) OUT_DIR prefix under data/eval/bc_dqc_chunk/
#   JOBTAG   (required) short slurm job-name stem
#   N_LIST   (default "1 2 4 8 16 32 64")
#   MODE     (default softmax)  | argmax
#   NUM_EPISODES (50)  START_SEED (1000)
#   DRY=1    report only
# Results: data/eval/bc_dqc_chunk/${LABEL}_n<N>_seed1000_ep50/eval_log.json
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey

SB=scriptsv2/eval_search/eval_bc_dqc_chunk.sbatch
: "${CKPT:?set CKPT}"; : "${LABEL:?set LABEL}"; : "${JOBTAG:?set JOBTAG}"
[[ -f "$CKPT" ]] || { echo "ERROR: ckpt not found: $CKPT" >&2; exit 1; }
N_LIST="${N_LIST:-1 2 4 8 16 32 64}"
MODE="${MODE:-softmax}"
ACCT="${ACCT:-robotics}"; PART="${PART:-gpu-a40}"; GRES="${GRES:-gpu:a40:1}"
SOFTMAX_TEMP="${SOFTMAX_TEMP:-1.0}"
NUM_EPISODES="${NUM_EPISODES:-50}"
START_SEED="${START_SEED:-1000}"
# Correct env per task (default = eggplant). For stack/spoon set BOTH the
# SimplerEnv rollout TASK and the DQC verifier's task embedding + instruction.
TASK="${TASK:-widowx_put_eggplant_in_basket}"
DQC_TEXT_EMB="${DQC_TEXT_EMB:-$PWD/scriptsv2/q_fn_eval/task_emb_use3.npy}"
VERIFIER_INSTRUCTION="${VERIFIER_INSTRUCTION:-put the eggplant in the basket}"
DRY="${DRY:-0}"

QNAMES="$(squeue -u "$USER" -h -o '%j')"
sub=0; sd=0; sq=0
done_cell(){ python3 -c "import json,sys
try: sys.exit(0 if int(json.load(open('$1/eval_log.json')).get('num_episodes',0))>=$NUM_EPISODES else 1)
except Exception: sys.exit(1)" 2>/dev/null; }

for N in $N_LIST; do
  out="data/eval/bc_dqc_chunk/${LABEL}_n${N}_seed${START_SEED}_ep${NUM_EPISODES}"
  jn="${JOBTAG}_n${N}"
  if done_cell "$out"; then echo "[done] n=$N"; sd=$((sd+1)); continue; fi
  if grep -qxF "$jn" <<<"$QNAMES"; then echo "[queued] $jn"; sq=$((sq+1)); continue; fi
  if [[ "$DRY" == "1" ]]; then echo "WOULD submit $jn (N=$N) -> $out"; sub=$((sub+1)); continue; fi
  jid=$(sbatch --parsable --job-name="$jn" --account="$ACCT" --partition="$PART" --gres="$GRES" --cpus-per-task=2 --mem=16G \
        --export=ALL,CHECKPOINT="$CKPT",N_SAMPLES="$N",NUM_EPISODES="$NUM_EPISODES",START_SEED="$START_SEED",BC_SAMPLER=ddpm,MODE="$MODE",SOFTMAX_TEMP="$SOFTMAX_TEMP",OUT_DIR="$out",TASK="$TASK",DQC_TEXT_EMB="$DQC_TEXT_EMB",VERIFIER_INSTRUCTION="$VERIFIER_INSTRUCTION" \
        "$SB")
  echo "submitted $jn -> $jid (N=$N)"; sub=$((sub+1)); sleep 1
done
echo "=== $LABEL DQC BoN: submitted=$sub done=$sd queued=$sq ==="