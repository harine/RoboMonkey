#!/bin/bash
# -----------------------------------------------------------------------------
# SEEDED-selection reruns for a clean, reproducible BoN comparison. The softmax
# candidate selection in eval_search.py is now seeded from replan_seed (was
# reseeded from OS entropy), so a given (env seed, checkpoint, N) yields the SAME
# rollout every run -> the only variable across the 500- vs 120-step tables is
# episode length. N=1..64.
#
#   SEARCH: 3 tasks x {sub50,sub100} x 8 ckpts (10/20/30/50/60/70/90/100k)
#           x N(1..64) x {500,120} steps   (100k -> latest.ckpt)
#   BC:     3 tasks x {sub50,sub100} best-n=1 ckpt x N(1..64) x 500 only
#
# Output -> data/eval/_seeded/... (kept separate from the OLD unseeded cells,
# which are archived in the results doc). No videos (SR sweep only). Idempotent:
# a cell with >=50 ep or already queued is skipped. Submission order: search@500,
# then BC@500, then search@120 (start search with 500 steps first, per request).
#
# CPU-per-task = 4 (lowered from the per-node ratio): the eval only needs a few CPUs
# (SAPIEN sim + policy + verifier), and a smaller ask lets ~4x more jobs run under the
# account's AssocGrpCpuLimit quota -> far higher GPU concurrency. DRY=1 to preview.
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
SBS=scriptsv2/eval_search/eval_search_dqc_chunk.sbatch
SBB=scriptsv2/eval_search/eval_bc_dqc_chunk.sbatch
MT=/gscratch/robotics/harine/RoboMonkey_data/train/rmse_multitask
FL=/gscratch/robotics/harine/RoboMonkey_data/train/rmse_filtered
BCD=/mmfs1/home/harine/RoboMonkey/diffusion_policy/data/train/bc_denoiser
Q=/mmfs1/home/harine/RoboMonkey/scriptsv2/q_fn_eval
SEED_ROOT=data/eval/_seeded
# N=1 has a single candidate -> no softmax selection -> deterministic, identical to
# the old unseeded runs. So REUSE old N=1 at 120 steps (the milestone evals already
# cover N=1 for every ckpt); still run N=1 at 500 steps (no old 500-step milestone data).
N_LIST_500="1 2 4 8 16 32 64"
N_LIST_120="2 4 8 16 32 64"
MILES="10 20 30 50 60 70 90 100"           # k (40k/80k intentionally skipped)
DRY="${DRY:-0}"
# acct|partition|gres|cpus  (cpus = partition CPU/GPU ratio -> all node GPUs get CPUs)
ROUTES=( "weirdlab|gpu-l40|gpu:l40:1|4" "weirdlab|gpu-l40s|gpu:l40s:1|4" "cse|gpu-l40s|gpu:l40s:1|4" "weirdlab|gpu-a40|gpu:a40:1|4" "cse|gpu-a100|gpu:a100:1|4" "robotics|gpu-a40|gpu:a40:1|4" )

# task|sub|ckpt_dir|TASK_ENV
SEARCH_POL=(
  "stack|sub50|$MT/diff_search_rmse_stack_sub50_diverse|widowx_stack_cube"
  "stack|sub100|$MT/diff_search_rmse_stack_sub100_diverse|widowx_stack_cube"
  "spoon|sub50|$MT/diff_search_rmse_spoon_sub50_diverse|widowx_spoon_on_towel"
  "spoon|sub100|$MT/diff_search_rmse_spoon_sub100_diverse|widowx_spoon_on_towel"
  "eggplant|sub50|$FL/diff_search_rmse_sub50_diverse_data|widowx_put_eggplant_in_basket"
  "eggplant|sub100|$FL/diff_search_rmse_sub100_diverse_data|widowx_put_eggplant_in_basket"
)
# task|sub|ckpt|TASK_ENV|EMB|INSTR  (BC best n=1)
BC_POL=(
  "stack|sub50|$BCD/bc_unet_h8_50k_stack_sub50_nobs1/checkpoints/step_0030002.ckpt|widowx_stack_cube|$Q/task_emb_stack_use3.npy|stack the green block on the yellow block"
  "stack|sub100|$BCD/bc_unet_h8_50k_stack_sub100_nobs1/checkpoints/step_0040005.ckpt|widowx_stack_cube|$Q/task_emb_stack_use3.npy|stack the green block on the yellow block"
  "spoon|sub50|$BCD/bc_unet_h8_50k_spoon_sub50_nobs1/checkpoints/step_0010001.ckpt|widowx_spoon_on_towel|$Q/task_emb_spoon_use3.npy|put the spoon on the towel"
  "spoon|sub100|$BCD/bc_unet_h8_50k_spoon_sub100_nobs1/checkpoints/step_0020002.ckpt|widowx_spoon_on_towel|$Q/task_emb_spoon_use3.npy|put the spoon on the towel"
  "eggplant|sub50|$BCD/eggplant_bc_unet_h8_50k_sub50_diverse_nobs1/checkpoints/latest.ckpt|widowx_put_eggplant_in_basket|$Q/task_emb_use3.npy|put the eggplant in the basket"
  "eggplant|sub100|$BCD/eggplant_bc_unet_h8_50k_sub100_diverse_nobs1/checkpoints/step_0030001.ckpt|widowx_put_eggplant_in_basket|$Q/task_emb_use3.npy|put the eggplant in the basket"
)

QNAMES="$(squeue -u "$USER" -h -o '%j')"
i=0; NSUB=0; sd=0; sq=0; noc=0
# eval_log.json is written only at successful completion -> its presence == done (fast, no python).
done_cell () { [ -s "$1/eval_log.json" ]; }
find_ckpt () {  # $1=ckpt_dir $2=milestone(k) -> path (100k -> latest.ckpt)
  local d="$1/checkpoints" M=$(( $2 * 1000 ))
  [ -d "$d" ] || return 0
  if [ "$2" -ge 100 ]; then [ -f "$d/latest.ckpt" ] && echo "$d/latest.ckpt"; return 0; fi
  ls "$d"/step_*.ckpt 2>/dev/null | awk -v M="$M" '
    { s=$0; n=$0; sub(/.*step_0*/,"",n); sub(/\.ckpt/,"",n); n+=0;
      if (n>=M-200 && n<=M+1200){dd=(n>M)?n-M:M-n; if(best==""||dd<bd){best=s;bd=dd}} }
    END{ if(best!="") print best }'
}
submit () {  # $1=jobname $2=out $3=sbatch $4=export-str  (routes + cpus)
  IFS='|' read -r a p g c <<< "${ROUTES[$((i % ${#ROUTES[@]}))]}"; i=$((i+1))
  if [ "$DRY" = 1 ]; then echo "WOULD $1 -> $a/$p ${c}cpu"; NSUB=$((NSUB+1)); return; fi
  jid=$(sbatch --parsable --job-name="$1" --account="$a" --partition="$p" --gres="$g" \
        --cpus-per-task="$c" --mem=32G --time=48:00:00 --export=ALL,"$4" "$3" 2>/dev/null) \
    && { echo "sub $1 -> $jid ($a/$p ${c}cpu)"; NSUB=$((NSUB+1)); } \
    || echo "SUBMIT-FAIL $1 (queue cap?)"
}

# ---- SEARCH: iterate step-length outer so 500 goes first, then 120 ----
search_pass () {  # $1=steplen
  local sl="$1" nl
  [ "$sl" = 500 ] && nl="$N_LIST_500" || nl="$N_LIST_120"
  for spec in "${SEARCH_POL[@]}"; do
    IFS='|' read -r task sub dir tenv <<< "$spec"
    for Mk in $MILES; do
      ck="$(find_ckpt "$dir" "$Mk")"; [ -z "$ck" ] && { noc=$((noc+1)); continue; }
      for N in $nl; do
        out="$SEED_ROOT/${task}_${sub}_search_${Mk}k_st${sl}_n${N}_seed1000_ep50"
        jn="ss${sl}_${task:0:2}${sub#sub}_${Mk}k_n${N}"
        done_cell "$out" && { sd=$((sd+1)); continue; }
        grep -qxF "$jn" <<< "$QNAMES" && { sq=$((sq+1)); continue; }
        submit "$jn" "$out" "$SBS" \
          "CHECKPOINT=$ck,N_SAMPLES=$N,MODE=softmax,SOFTMAX_TEMP=1.0,NUM_EPISODES=50,START_SEED=1000,DQC_RESTORE_EPOCH=100000,TASK=$tenv,MAX_STEPS=$sl,SAVE_VIDEOS=0,VIZ_Q=0,RENDER_Q=0,OUT_DIR=$out"
      done
    done
  done
}
bc_pass () {  # 500 only
  local sl=500
  for spec in "${BC_POL[@]}"; do
    IFS='|' read -r task sub ck tenv emb instr <<< "$spec"
    [ -f "$ck" ] || { echo "MISSING BC ckpt $task $sub"; continue; }
    for N in $N_LIST_500; do
      out="$SEED_ROOT/${task}_${sub}_bc_st${sl}_n${N}_seed1000_ep50"
      jn="bs${sl}_${task:0:2}${sub#sub}_n${N}"
      done_cell "$out" && { sd=$((sd+1)); continue; }
      grep -qxF "$jn" <<< "$QNAMES" && { sq=$((sq+1)); continue; }
      submit "$jn" "$out" "$SBB" \
        "CHECKPOINT=$ck,N_SAMPLES=$N,NUM_EPISODES=50,START_SEED=1000,BC_SAMPLER=ddpm,MODE=softmax,SOFTMAX_TEMP=1.0,MAX_STEPS=$sl,SAVE_VIDEOS=0,VIZ_Q=0,RENDER_Q=0,TASK=$tenv,DQC_TEXT_EMB=$emb,VERIFIER_INSTRUCTION=$instr,OUT_DIR=$out"
    done
  done
}

mkdir -p "$SEED_ROOT"
search_pass 500
bc_pass
search_pass 120
echo "=== seeded ckpt evals: submitted=$NSUB skip_done=$sd skip_queued=$sq ckpt_not_yet=$noc ==="
