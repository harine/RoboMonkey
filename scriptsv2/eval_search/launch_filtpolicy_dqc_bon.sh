#!/bin/bash
# -----------------------------------------------------------------------------
# BoN sim-eval of the FILTERED-DATA-trained 10k diffusion search policy
# (diff_search_rmse_filtered @ step 10000) with the DQC chunk verifier, over
# N=1..64. Two verifiers per N so the finetuned curve has a baseline:
#   * base            = the policy's BAKED base DQC verifier (no re-point)
#   * v4filt50k       = re-point baked verifier -> finetuned v4_filt_ramp @ 50k
# Same protocol as launch_ft_verifier_bon.sh (softmax T=1.0, score_window=last,
# baked score_agg=discounted_sum, 50 ep seeds 1000..1049) so it is directly
# comparable to the unfiltered diff10k rows already in the BoN table.
#
# The filtered policy was trained ON-cluster, so its baked verifier paths are
# valid -> NO pathfix needed (verified). Idempotent + scavenger, same as the
# sibling launcher.
#
#   bash scriptsv2/eval_search/launch_filtpolicy_dqc_bon.sh          # submit gaps
#   DRY=1 bash scriptsv2/eval_search/launch_filtpolicy_dqc_bon.sh    # report only
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
REPO="$PWD"
SB_SEARCH=scriptsv2/eval_search/eval_search_dqc_chunk.sbatch
# Policy is explicit/overridable (POLICY_CKPT) so this always evals the policy you name,
# never a stale one; it must exist (checked below). Default = filtered diff-10k.
POLICY_CKPT="${POLICY_CKPT:-/gscratch/robotics/harine/RoboMonkey_data/train/rmse_filtered/diff_search_rmse_filtered_data/checkpoints/step_0010000.ckpt}"
V4DIR=/gscratch/robotics/harine/RoboMonkey_data/viz_q/ft/dqc_finetune/v4_filt_ramp/v4_filt_ramp_b256_a100
TEMB="$REPO/scriptsv2/q_fn_eval/task_emb_use3.npy"
N_LIST="${N_LIST:-1 2 4 8 16 32 64}"
# Which finetune epochs of v4_filt_ramp to score (tag = v4filt<E/1000>k). Default 50k;
# set EPOCHS="10000 20000 30000 40000 50000" to sweep checkpoints for best-of.
EPOCHS="${EPOCHS:-50000}"
DO_BASE="${DO_BASE:-1}"   # also run the baked base verifier (0 to skip if already done)
DRY="${DRY:-0}"
[[ -f "$POLICY_CKPT" ]] || { echo "missing policy ckpt $POLICY_CKPT"; exit 1; }

# out-prefix base (in data/eval/search_dqc_chunk); table verifier tag column
#   base      -> difffilt10k_chunk8_...              (baked base verifier)
#   v4filt50k -> difffilt10k_v4filt50k_chunk8_...     (finetuned verifier)
PBASE="difffilt10k"

if [[ -n "${ROUTES_OVERRIDE:-}" ]]; then read -r -a ROUTES <<< "$ROUTES_OVERRIDE"; else
  ROUTES=( "weirdlab-ckpt|ckpt-all|gpu:l40s:1|ckpt-gpu" \
           "weirdlab-ckpt|ckpt-all|gpu:l40:1|ckpt-gpu" \
           "weirdlab-ckpt|ckpt-all|gpu:a40:1|ckpt-gpu" \
           "weirdlab-ckpt|ckpt-all|gpu:a100:1|ckpt-gpu" ); fi

QNAMES="$(squeue -u "$USER" -h -o '%j' 2>/dev/null)"
i=0; sub=0; skip_done=0; skip_q=0
done_cell () { python3 - "$1" <<'PY' 2>/dev/null
import json,sys
try:
    d=json.load(open(sys.argv[1]+"/eval_log.json")); sys.exit(0 if int(d.get("num_episodes",0))>=50 else 1)
except Exception: sys.exit(1)
PY
}
submit () {  # $1=jobname $2=exports
  IFS='|' read -r acct part gres qos <<< "${ROUTES[$((i % ${#ROUTES[@]}))]}"; i=$((i+1))
  local qosflag=(); [[ -n "${qos:-}" ]] && qosflag=(--qos="$qos")
  if [ "$DRY" = "1" ]; then printf 'WOULD %-30s %s/%s\n' "$1" "$part" "$gres"; sub=$((sub+1)); return; fi
  jid=$(sbatch --parsable --job-name="$1" --account="$acct" --partition="$part" --gres="$gres" "${qosflag[@]}" \
        --cpus-per-task=2 --mem=16G --time=1-00:00:00 --export=ALL,"$2" "$SB_SEARCH")
  printf 'submit %-30s %-14s -> %s\n' "$1" "$part/$gres" "$jid"; sub=$((sub+1))
}

for N in $N_LIST; do
  # ---- base (baked) verifier ----
  if [ "$DO_BASE" = 1 ]; then
    out="data/eval/search_dqc_chunk/${PBASE}_chunk8_predh8_roll4_softmax_n${N}_seed1000_ep50"
    jn="ffilt_base_n${N}"
    if done_cell "$out"; then skip_done=$((skip_done+1)); elif grep -qxF "$jn" <<< "$QNAMES"; then skip_q=$((skip_q+1)); else
      submit "$jn" "CHECKPOINT=$POLICY_CKPT,N_SAMPLES=$N,MODE=softmax,SOFTMAX_TEMP=1.0,NUM_EPISODES=50,START_SEED=1000,OUT_DIR=$out"
    fi
  fi
  # ---- finetuned v4_filt_ramp @ each requested epoch (re-point) ----
  for E in $EPOCHS; do
    tag="v4filt$((E/1000))k"
    out="data/eval/search_dqc_chunk/${PBASE}_${tag}_chunk8_predh8_roll4_softmax_n${N}_seed1000_ep50"
    jn="ffilt_${tag}_n${N}"
    if done_cell "$out"; then skip_done=$((skip_done+1)); elif grep -qxF "$jn" <<< "$QNAMES"; then skip_q=$((skip_q+1)); else
      submit "$jn" "CHECKPOINT=$POLICY_CKPT,N_SAMPLES=$N,MODE=softmax,SOFTMAX_TEMP=1.0,NUM_EPISODES=50,START_SEED=1000,OUT_DIR=$out,DQC_VERIFIER_CKPT=$V4DIR,DQC_VERIFIER_EPOCH=$E,DQC_TEXT_EMB=$TEMB"
    fi
  done
done
echo "=== submitted=$sub  skip_done=$skip_done  skip_queued=$skip_q  (2 verifiers x N) ==="