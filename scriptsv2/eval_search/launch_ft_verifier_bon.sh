#!/bin/bash
# -----------------------------------------------------------------------------
# BoN sim-eval of the 4 policies (bc_h8, diff10k, diff33k, gauss20k) with the
# FINETUNED DQC verifiers (v2_all@30k, v3_dist@30k), softmax selection, N=1..64.
# Matches the june19 base config EXACTLY (mode=softmax, score_window=last,
# baked score_agg=discounted_sum/0.98, 50 ep, seeds 1000..1049) — the ONLY change
# vs base is the verifier WEIGHTS, so curves are directly comparable to the
# already-run base curves (NOT rerun here).
#
#   * search policies (diff/gauss): baked DQC verifier RE-POINTED to the finetuned
#     ckpt via DQC_VERIFIER_CKPT + DQC_VERIFIER_EPOCH (scoring preserved).
#   * BC (bc_h8): VERIFIER_TYPE=dqc path builds the verifier from DQC_CKPT_DIR +
#     DQC_RESTORE_EPOCH (same code path as the base bc_h8 softmax run).
#
# 1 GPU / cell, --cpus-per-task=2 (16-cpu/8-gpu l40 nodes -> all GPUs keep CPUs),
# round-robin over weirdlab l40s/l40 (both infinite walltime). Idempotent: a cell
# is skipped if its eval_log.json already has >=50 ep, or a same-named job is PD/R.
#
#   bash scriptsv2/eval_search/launch_ft_verifier_bon.sh         # submit gaps
#   DRY=1 bash scriptsv2/eval_search/launch_ft_verifier_bon.sh   # report only
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
REPO="$PWD"
SB_SEARCH=scriptsv2/eval_search/eval_search_dqc_chunk.sbatch
SB_BC=scriptsv2/eval_search/eval_bc_dqc_chunk.sbatch
TEMB="$REPO/scriptsv2/q_fn_eval/task_emb_use3.npy"
N_LIST="${N_LIST:-1 2 4 8 16 32 64}"
DRY="${DRY:-0}"

# verifier: tag | finetuned ckpt dir | epoch
# Override with VERIFIERS_OVERRIDE="tag|dir|epoch tag2|dir2|epoch2 ..." to sweep a
# different set (e.g. the v4_filt_ramp finetune) without touching the defaults.
if [[ -n "${VERIFIERS_OVERRIDE:-}" ]]; then
  read -r -a VERIFIERS <<< "$VERIFIERS_OVERRIDE"
else
  VERIFIERS=(
    "v2all30k|/gscratch/robotics/harine/RoboMonkey_data/viz_q/ft/dqc_finetune/v2_all/v2_all|30000"
    "v3dist30k|/gscratch/robotics/harine/RoboMonkey_data/viz_q/ft/dqc_finetune/v3_dist/v3_dist_b256_a100|30000"
  )
fi
# ---------------------------------------------------------------------------
# POLICIES ARE REQUIRED — no hardcoded defaults. The unfiltered success policies
# were ARCHIVED to /gscratch/robotics/harine/unfiltered_success_policies/ and must
# NOT be evaluated; this launcher ERRORS unless you name the policy to evaluate, so
# it always runs the policy you want (never a stale one). Filtered policies live under
#   /gscratch/robotics/harine/RoboMonkey_data/train/rmse_filtered/{diff_search_rmse_filtered,gauss_search_rmse_filtered,diff_search_rmse_sub100_diverse}_data/checkpoints/
# Specify at least one of:
#   SEARCH_POLICIES : space-sep "name|ckpt" for native search policies (baked DQC
#                     verifier re-pointed to each VERIFIER). Result dir prefix =
#                     <name>_<verifier-tag>_chunk8_predh8_roll4_softmax.
#   BC_CKPT + BC_NAME : a BC policy ckpt + short name (DQC verifier built fresh;
#                     prefix = <BC_NAME>_<verifier-tag>_chunk8_predh8_roll4_softmax_ddpm).
# Example (filtered diffusion policy @ latest):
#   SEARCH_POLICIES="difffilt|/gscratch/robotics/harine/RoboMonkey_data/train/rmse_filtered/diff_search_rmse_filtered_data/checkpoints/latest.ckpt" \
#     bash scriptsv2/eval_search/launch_ft_verifier_bon.sh
# ---------------------------------------------------------------------------
read -r -a SEARCH_POLICIES <<< "${SEARCH_POLICIES:-}"
BC_CKPT="${BC_CKPT:-}"; BC_NAME="${BC_NAME:-}"
if [[ ${#SEARCH_POLICIES[@]} -eq 0 && -z "$BC_CKPT" ]]; then
  echo "ERROR: no policy specified — set SEARCH_POLICIES=\"name|ckpt ...\" and/or BC_CKPT=<ckpt> BC_NAME=<name>." >&2
  echo "  Unfiltered policies were ARCHIVED to /gscratch/robotics/harine/unfiltered_success_policies/ (do NOT eval them)." >&2
  echo "  Filtered policies: /gscratch/robotics/harine/RoboMonkey_data/train/rmse_filtered/*/checkpoints/" >&2
  exit 2
fi
[[ -n "$BC_CKPT" && -z "$BC_NAME" ]] && { echo "ERROR: BC_CKPT set but BC_NAME missing (short name for result dirs)." >&2; exit 2; }
# fail fast on a missing/typo'd/moved checkpoint rather than silently eval the wrong thing.
for pol in "${SEARCH_POLICIES[@]}"; do
  IFS='|' read -r _pn _pc _ <<< "$pol"
  [[ -n "$_pc" && -f "$_pc" ]] || { echo "ERROR: search policy '${_pn:-?}' ckpt not found: ${_pc:-<empty>}" >&2; exit 2; }
done
[[ -n "$BC_CKPT" && ! -f "$BC_CKPT" ]] && { echo "ERROR: BC_CKPT not found: $BC_CKPT" >&2; exit 2; }
# account|partition|gres|qos  — default: ckpt-all scavenger (own cap, infinite walltime,
# idle L40S/L40/A40/A100 right now). evals resume per-episode so preemption is safe.
# Override with ROUTES_OVERRIDE="acct|part|gres|qos ..." to use weirdlab pools instead.
if [[ -n "${ROUTES_OVERRIDE:-}" ]]; then
  read -r -a ROUTES <<< "$ROUTES_OVERRIDE"
else
  ROUTES=( "weirdlab-ckpt|ckpt-all|gpu:l40s:1|ckpt-gpu" \
           "weirdlab-ckpt|ckpt-all|gpu:l40:1|ckpt-gpu" \
           "weirdlab-ckpt|ckpt-all|gpu:a40:1|ckpt-gpu" \
           "weirdlab-ckpt|ckpt-all|gpu:a100:1|ckpt-gpu" )
fi

QNAMES="$(squeue -u "$USER" -h -o '%j' 2>/dev/null)"
i=0; sub=0; skip_done=0; skip_q=0
done_cell () {  # $1 = full eval dir -> 0 if eval_log has >=50 ep
  python3 - "$1" <<'PY' 2>/dev/null
import json,sys
try:
    d=json.load(open(sys.argv[1]+"/eval_log.json")); sys.exit(0 if int(d.get("num_episodes",0))>=50 else 1)
except Exception: sys.exit(1)
PY
}
submit () {  # $1=jobname $2=sbatch $3=exports
  IFS='|' read -r acct part gres qos <<< "${ROUTES[$((i % ${#ROUTES[@]}))]}"; i=$((i+1))
  local qosflag=(); [[ -n "${qos:-}" ]] && qosflag=(--qos="$qos")
  if [ "$DRY" = "1" ]; then printf 'WOULD %-26s %s/%s/%s\n' "$1" "$acct" "$part" "$gres"; sub=$((sub+1)); return; fi
  jid=$(sbatch --parsable --job-name="$1" --account="$acct" --partition="$part" --gres="$gres" "${qosflag[@]}" \
        --cpus-per-task=2 --mem=16G --time=1-00:00:00 --export=ALL,"$3" "$2")
  printf 'submit %-26s %-14s -> %s\n' "$1" "$part/$gres" "$jid"; sub=$((sub+1))
}

for vf in "${VERIFIERS[@]}"; do
  IFS='|' read -r vtag vdir vep <<< "$vf"
  # ---- search policies (re-point baked DQC verifier) ----
  for pol in "${SEARCH_POLICIES[@]}"; do
    IFS='|' read -r pname pckpt pbase <<< "$pol"
    for N in $N_LIST; do
      out="data/eval/search_dqc_chunk/${pname}_${vtag}_chunk8_predh8_roll4_softmax_n${N}_seed1000_ep50"
      jn="ft_${vtag}_${pname}_n${N}"
      if done_cell "$out"; then skip_done=$((skip_done+1)); continue; fi
      if grep -qxF "$jn" <<< "$QNAMES"; then skip_q=$((skip_q+1)); continue; fi
      submit "$jn" "$SB_SEARCH" \
        "CHECKPOINT=$pckpt,N_SAMPLES=$N,MODE=softmax,SOFTMAX_TEMP=1.0,NUM_EPISODES=50,START_SEED=1000,OUT_DIR=$out,DQC_VERIFIER_CKPT=$vdir,DQC_VERIFIER_EPOCH=$vep,DQC_TEXT_EMB=$TEMB"
    done
  done
  # ---- BC policy (build verifier from DQC_CKPT_DIR) — only if BC_CKPT given ----
  if [[ -n "$BC_CKPT" ]]; then
    for N in $N_LIST; do
      out="data/eval/bc_dqc_chunk/${BC_NAME}_${vtag}_chunk8_predh8_roll4_softmax_ddpm_n${N}_seed1000_ep50"
      jn="ft_${vtag}_${BC_NAME}_n${N}"
      if done_cell "$out"; then skip_done=$((skip_done+1)); continue; fi
      if grep -qxF "$jn" <<< "$QNAMES"; then skip_q=$((skip_q+1)); continue; fi
      submit "$jn" "$SB_BC" \
        "CHECKPOINT=$BC_CKPT,BC_SAMPLER=ddpm,N_SAMPLES=$N,MODE=softmax,SOFTMAX_TEMP=1.0,NUM_EPISODES=50,START_SEED=1000,OUT_DIR=$out,DQC_CKPT_DIR=$vdir,DQC_RESTORE_EPOCH=$vep,DQC_TEXT_EMB=$TEMB"
    done
  fi
done
echo "=== submitted=$sub  skip_done=$skip_done  skip_queued=$skip_q  (verifiers x policies x N) ==="
