#!/bin/bash
# -----------------------------------------------------------------------------
# One-shot: re-run the sub50 BC + SEARCH policies for all 3 tasks at MAX_STEPS=500
# (vs the default 120), N=1,8,64, softmax, 50 episodes, seeds 1000-1049, with
# action+Q overlay videos (VIZ_Q + RENDER_Q). Search renders 10 videos/N, BC 5/N.
#
#   SEARCH sub50 milestones (baked DQC-chunk verifier, TASK-correct MUSE emb):
#     spoon @30k  stack @20k  eggplant @70k
#   BC sub50 (best n=1 ckpt, DQC-chunk verifier via VERIFIER_TYPE=dqc):
#     stack step_0030002(30k)  spoon step_0010001(10k)  eggplant latest.ckpt(50k)
#
# Cells stage under data/eval/_500step_runs/; collect_500step_videos.sh then copies
# the rendered videos into each category's <ck>_500steps/ (search) and
# bc/<bcck>_500steps/ (BC) subfolder. Idempotent: a cell with >=50 ep or already
# queued is skipped.  DRY=1 to preview.
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
SBS=scriptsv2/eval_search/eval_search_dqc_chunk.sbatch
SBB=scriptsv2/eval_search/eval_bc_dqc_chunk.sbatch
MT=/gscratch/robotics/harine/RoboMonkey_data/train/rmse_multitask
FL=/gscratch/robotics/harine/RoboMonkey_data/train/rmse_filtered
BCD=/mmfs1/home/harine/RoboMonkey/diffusion_policy/data/train/bc_denoiser
Q=/mmfs1/home/harine/RoboMonkey/scriptsv2/q_fn_eval
STAGE=data/eval/_500step_runs
MAXST=500
DRY="${DRY:-0}"
mkdir -p "$STAGE"
# Search N-grids match the 120-step ("before") evals per task: stack/spoon 1..64,
# eggplant 1..512. BC stays 1,8,64 (as requested for the video renders).
NG_LO="1 2 4 8 16 32 64"          # stack/spoon search
NG_HI="1 2 4 8 16 32 64 128 256 512"   # eggplant search (matches its before-grid)
NG_BC="1 8 64"
# Free GPUs (hyakalloc 2026-07-07): cse l40s x4, weirdlab l40 x6 / l40s x7, weirdlab a40 x3, cse a100 x3.
ROUTES=( "weirdlab|gpu-l40|gpu:l40:1" "weirdlab|gpu-l40s|gpu:l40s:1" "cse|gpu-l40s|gpu:l40s:1" "weirdlab|gpu-a40|gpu:a40:1" "cse|gpu-a100|gpu:a100:1" )
# High-N (>=128) at 500 steps = 10-40h jobs -> weirdlab 7-day partitions only (cse
# caps at 1 day). Long wall so they are not killed mid-run.
HIROUTES=( "weirdlab|gpu-l40|gpu:l40:1" "weirdlab|gpu-l40s|gpu:l40s:1" "weirdlab|gpu-a40|gpu:a40:1" )

# kind|task|cktag|ckpt|TASK_ENV|EMB|INSTR|NLIST   (EMB/INSTR only used by BC)
SPECS=(
  "search|spoon|30k|$MT/diff_search_rmse_spoon_sub50_diverse/checkpoints/step_0030001.ckpt|widowx_spoon_on_towel|||$NG_LO"
  "search|stack|20k|$MT/diff_search_rmse_stack_sub50_diverse/checkpoints/step_0020001.ckpt|widowx_stack_cube|||$NG_LO"
  "search|eggplant|70k|$FL/diff_search_rmse_sub50_diverse_data/checkpoints/step_0070002.ckpt|widowx_put_eggplant_in_basket|||$NG_HI"
  "bc|stack|30k|$BCD/bc_unet_h8_50k_stack_sub50_nobs1/checkpoints/step_0030002.ckpt|widowx_stack_cube|$Q/task_emb_stack_use3.npy|stack the green block on the yellow block|$NG_BC"
  "bc|spoon|10k|$BCD/bc_unet_h8_50k_spoon_sub50_nobs1/checkpoints/step_0010001.ckpt|widowx_spoon_on_towel|$Q/task_emb_spoon_use3.npy|put the spoon on the towel|$NG_BC"
  "bc|eggplant|50k|$BCD/eggplant_bc_unet_h8_50k_sub50_diverse_nobs1/checkpoints/latest.ckpt|widowx_put_eggplant_in_basket|$Q/task_emb_use3.npy|put the eggplant in the basket|$NG_BC"
)

QNAMES="$(squeue -u "$USER" -h -o '%j')"
i=0; sub=0; sd=0; sq=0
done_cell () {  # $1=out_dir
  python3 - "$1" <<'PY' 2>/dev/null
import json,sys
try:
    d=json.load(open(f"{sys.argv[1]}/eval_log.json")); sys.exit(0 if int(d.get("num_episodes",0))>=50 else 1)
except Exception: sys.exit(1)
PY
}
i=0; hi=0
for spec in "${SPECS[@]}"; do
  IFS='|' read -r kind task ck ckpt tenv emb instr nlist <<< "$spec"
  [ -f "$ckpt" ] || { echo "MISSING ckpt: $ckpt -- skip $kind $task $ck"; continue; }
  for N in $nlist; do
    out="$STAGE/${kind}_${task}_${ck}_n${N}"
    jn="${kind:0:1}500_${task:0:2}${ck}_n${N}"
    if done_cell "$out"; then sd=$((sd+1)); continue; fi
    grep -qxF "$jn" <<< "$QNAMES" && { sq=$((sq+1)); continue; }
    # N>=128: heavy (10-40h) -> weirdlab 7-day partitions + long wall; else mixed + 20h.
    if [ "$N" -ge 128 ]; then
      IFS='|' read -r a p g <<< "${HIROUTES[$((hi % ${#HIROUTES[@]}))]}"; hi=$((hi+1)); wall=72:00:00; mem=48G
    else
      IFS='|' read -r a p g <<< "${ROUTES[$((i % ${#ROUTES[@]}))]}"; i=$((i+1)); wall=20:00:00
      mem=16G; [ "$N" -ge 64 ] && mem=32G
    fi
    if [ "$kind" = "search" ]; then
      viz=10; SB="$SBS"
      exp="CHECKPOINT=$ckpt,N_SAMPLES=$N,MODE=softmax,SOFTMAX_TEMP=1.0,NUM_EPISODES=50,START_SEED=1000,DQC_RESTORE_EPOCH=100000,TASK=$tenv,MAX_STEPS=$MAXST,SAVE_VIDEOS=0,VIDEO_FPS=10,VIZ_Q=1,VIZ_Q_EPISODES=$viz,RENDER_Q=1,OUT_DIR=$out"
    else
      viz=5; SB="$SBB"
      exp="CHECKPOINT=$ckpt,N_SAMPLES=$N,NUM_EPISODES=50,START_SEED=1000,BC_SAMPLER=ddpm,MODE=softmax,SOFTMAX_TEMP=1.0,MAX_STEPS=$MAXST,SAVE_VIDEOS=0,VIDEO_FPS=10,VIZ_Q=1,VIZ_Q_EPISODES=$viz,RENDER_Q=1,TASK=$tenv,DQC_TEXT_EMB=$emb,VERIFIER_INSTRUCTION=$instr,OUT_DIR=$out"
    fi
    if [ "$DRY" = 1 ]; then echo "WOULD $jn ($kind $task $ck N=$N 500st viz=$viz -> $a/$p $mem ${wall})"; sub=$((sub+1)); continue; fi
    jid=$(sbatch --parsable --job-name="$jn" --account="$a" --partition="$p" --gres="$g" \
          --cpus-per-task=2 --mem="$mem" --time="$wall" \
          --export=ALL,"$exp" "$SB")
    echo "submitted $jn -> $jid ($kind $task $ck N=$N, $a/$p, $wall)"; sub=$((sub+1))
  done
done
echo "=== 500-step sub50 evals: submitted=$sub skip_done=$sd skip_queued=$sq ==="
