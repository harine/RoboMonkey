#!/bin/bash
# -----------------------------------------------------------------------------
# Deterministic sub50 eval driver (search + BC), two modes:
#   VIZ=0 (SR)  : 50 ep, tiled (search only), no videos -> SR + obj->goal distance.
#   VIZ=1 (viz) : 10 ep, NON-tiled, VIZ_Q=1 + RENDER_Q=1 -> 10 videos + trajectory
#                 npz (ee_xyz/src_xyz/tgt_xyz/values) per cell for the analysis.
# KIND=search : spoon 90k, stack 60k, eggplant 90k  (eval_search_dqc_chunk.sbatch)
# KIND=bc     : each task's best n=1 ckpt           (eval_bc_dqc_chunk.sbatch, ddpm)
# STEPS_MODE=native|500|120.  N=1..64.  Deterministic + distance always on.
# Routes: robotics + cse + ckpt (NOT weirdlab).
#   KIND=bc STEPS_MODE=120 VIZ=0 bash scriptsv2/eval_search/launch_det_eval2.sh
#   KIND=search STEPS_MODE=native VIZ=1 bash .../launch_det_eval2.sh   (DRY=1 preview)
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
SBS=scriptsv2/eval_search/eval_search_dqc_chunk.sbatch
SBB=scriptsv2/eval_search/eval_bc_dqc_chunk.sbatch
MT=/gscratch/robotics/harine/RoboMonkey_data/train/rmse_multitask
FL=/gscratch/robotics/harine/RoboMonkey_data/train/rmse_filtered
BCD=/mmfs1/home/harine/RoboMonkey/diffusion_policy/data/train/bc_denoiser
Q=/mmfs1/home/harine/RoboMonkey/scriptsv2/q_fn_eval
N_LIST="1 2 4 8 16 32 64"
KIND="${KIND:-search}"; STEPS_MODE="${STEPS_MODE:-native}"; VIZ="${VIZ:-0}"; DRY="${DRY:-0}"
# SEL_MODE=softmax|argmax : verifier-BoN selection. argmax = deterministic top-Q (no
#   multinomial) -> removes the softmax pool-size resampling noise.
# PIN_NODE=<node>        : pin every cell to ONE physical node (same GPU) to remove
#   cross-GPU FP variance. weirdlab owns the g30xx a40s -> force weirdlab/gpu-a40.
SEL_MODE="${SEL_MODE:-softmax}"; PIN_NODE="${PIN_NODE:-}"
# PER_ENV=1 (default): tiled search anchors each env's candidate noise to its own
# episode seed (bug fix for the shared-per-step seed). Ignored on non-tiled (BC/viz).
PER_ENV="${PER_ENV:-1}"
ROUTES=( "robotics|gpu-a100|gpu:a100:1" "cse|gpu-a100|gpu:a100:1" "cse|gpu-l40s|gpu:l40s:1" "robotics|ckpt|gpu:a40:1" "robotics|ckpt|gpu:a100:1" )
# VIZ renders route to weirdlab/robotics a40 (own CPU quota -> unblocks AssocGrpCpuLimit).
VIZ_ROUTES=( "weirdlab|gpu-a40|gpu:a40:1" "robotics|gpu-a40|gpu:a40:1" )
[ "${VIZ:-0}" = 1 ] && ROUTES=( "${VIZ_ROUTES[@]}" )
PIN_ACCT="${PIN_ACCT:-robotics}"; PIN_PART="${PIN_PART:-gpu-a40}"; PIN_GRES="${PIN_GRES:-gpu:a40:1}"
[ -n "$PIN_NODE" ] && ROUTES=( "$PIN_ACCT|$PIN_PART|$PIN_GRES" )   # single route for a pinned node (robotics by default)
find_ckpt(){ ls "$1"/checkpoints/step_*.ckpt 2>/dev/null | awk -v M=$(($2*1000)) \
  '{n=$0;sub(/.*step_0*/,"",n);sub(/\.ckpt/,"",n);n+=0; if(n>=M-200&&n<=M+1200){d=(n>M)?n-M:M-n; if(b==""||d<bd){b=$0;bd=d}}} END{if(b)print b}'; }
# task_folder|TASK_ENV|native_steps|search_dir|search_mk|bc_ckpt|EMB|INSTR
POLS=(
  "spoon_on_towel|widowx_spoon_on_towel|60|$MT/diff_search_rmse_spoon_sub50_diverse|90|$BCD/bc_unet_h8_50k_spoon_sub50_nobs1/checkpoints/step_0010001.ckpt|$Q/task_emb_spoon_use3.npy|put the spoon on the towel"
  "stack_cube|widowx_stack_cube|60|$MT/diff_search_rmse_stack_sub50_diverse|60|$BCD/bc_unet_h8_50k_stack_sub50_nobs1/checkpoints/step_0030002.ckpt|$Q/task_emb_stack_use3.npy|stack the green block on the yellow block"
  "eggplant|widowx_put_eggplant_in_basket|120|$FL/diff_search_rmse_sub50_diverse_data|90|$BCD/eggplant_bc_unet_h8_50k_sub50_diverse_nobs1/checkpoints/latest.ckpt|$Q/task_emb_use3.npy|put the eggplant in the basket"
)
QN="$(squeue -u "$USER" -h -o '%j')"; i=0; nsub=0; sk=0
for spec in "${POLS[@]}"; do
  IFS='|' read -r folder tenv native dir smk bcck emb instr <<< "$spec"
  case "$STEPS_MODE" in 500) steps=500;; 120) steps=120;; *) steps=$native;; esac
  if [ "$KIND" = search ]; then
    ck=$(find_ckpt "$dir" "$smk"); mk=$smk; SB="$SBS"
  else
    ck="$bcck"; mk=bc; SB="$SBB"
  fi
  [ -f "$ck" ] || { echo "NO CKPT $KIND $folder"; continue; }
  for N in $N_LIST; do
    asuf=""; [ "$SEL_MODE" = argmax ] && asuf="_argmax"      # SR argmax cells get a suffix
    if [ "$VIZ" = 1 ]; then
      vsub=$([ "$SEL_MODE" = argmax ] && echo "_viz_argmax" || echo "_viz")   # argmax videos -> own subfolder
      out="data/eval/$folder/$vsub/${folder}_${mk}_st${steps}_n${N}"; ep=10; tiled=1; vq=1; tag=viz
    else
      out="data/eval/$folder/${folder}_${mk}_st${steps}_n${N}_seed1000_ep50${asuf}"; ep=50; vq=0; tag=sr
      tiled=$([ "$KIND" = search ] && echo 8 || echo 1)   # BC non-tiled (batching untested)
    fi
    sm=$([ "$SEL_MODE" = argmax ] && echo a || echo s)
    jn="d${tag}${sm}_${KIND:0:1}${folder:0:2}${mk}_${steps}_n${N}"
    { [ "$VIZ" = 0 ] && [ -s "$out/eval_log.json" ]; } && { sk=$((sk+1)); continue; }
    { [ "$VIZ" = 1 ] && ls "$out"/search_q/videos/*.mp4 >/dev/null 2>&1; } && { sk=$((sk+1)); continue; }
    grep -qxF "$jn" <<<"$QN" && { sk=$((sk+1)); continue; }
    IFS='|' read -r a p g <<< "${ROUTES[$((i%${#ROUTES[@]}))]}"; i=$((i+1))
    if [ "$KIND" = search ]; then
      exp="CHECKPOINT=$ck,N_SAMPLES=$N,MODE=$SEL_MODE,SOFTMAX_TEMP=1.0,NUM_EPISODES=$ep,START_SEED=1000,DQC_RESTORE_EPOCH=100000,TASK=$tenv,MAX_STEPS=$steps,TILED_ENVS=$tiled,TILED_SEED_PER_ENV=$PER_ENV,VIZ_Q=$vq,VIZ_Q_EPISODES=10,RENDER_Q=$vq,SAVE_VIDEOS=0,VIDEO_FPS=10,OUT_DIR=$out"
    else
      exp="CHECKPOINT=$ck,N_SAMPLES=$N,NUM_EPISODES=$ep,START_SEED=1000,BC_SAMPLER=ddpm,MODE=$SEL_MODE,SOFTMAX_TEMP=1.0,MAX_STEPS=$steps,TILED_ENVS=$tiled,VIZ_Q=$vq,VIZ_Q_EPISODES=10,RENDER_Q=$vq,SAVE_VIDEOS=0,VIDEO_FPS=10,TASK=$tenv,DQC_TEXT_EMB=$emb,VERIFIER_INSTRUCTION=$instr,OUT_DIR=$out"
    fi
    [ "$DRY" = 1 ] && { echo "WOULD $jn ($folder $mk st$steps N=$N sel=$SEL_MODE viz=$vq tiled=$tiled -> $a/$p${PIN_NODE:+ @$PIN_NODE})"; nsub=$((nsub+1)); continue; }
    NL=(); [ -n "$PIN_NODE" ] && NL=(--nodelist="$PIN_NODE")
    jid=$(sbatch --parsable --job-name="$jn" --account="$a" --partition="$p" --gres="$g" \
          "${NL[@]}" --cpus-per-task=6 --mem=56G --time=12:00:00 --export=ALL,"$exp" "$SB")
    echo "sub $jn -> $jid ($a/$p${PIN_NODE:+ @$PIN_NODE})"; nsub=$((nsub+1))
  done
done
echo "=== det eval2 [$KIND $STEPS_MODE viz=$VIZ]: submitted=$nsub skipped=$sk ==="
