#!/bin/bash
# -----------------------------------------------------------------------------
# Diagnostic VIDEO pass for the finetuned-verifier BoN drop: for each cell
# (v2_all@30k, v3_dist@30k) x (bc_h8, diff10k, diff33k, gauss20k) x N=1..64, run
# the FIRST 10 eval seeds (1000..1009) and save BOTH:
#   * rollout MP4s            (SAVE_VIDEOS=10)              -> shows the behavior
#   * Q-overlay MP4s          (VIZ_Q + render_q)           -> candidate trajectories
#     colored by verifier-Q rank + the picked one -> shows WHY the verifier chose.
# Same verifier re-point / config as launch_ft_verifier_bon.sh (apples-to-apples
# with the success-rate cells; these reuse the same seeds, episode i = 1000+i).
# 1 GPU / cell, cpus=2, ckpt-all scavenger (infinite walltime, resumable).
#
#   bash scriptsv2/eval_search/launch_ft_verifier_videos.sh        # submit gaps
#   DRY=1 ...                                                      # report only
#   N_LIST="8 16 32" ...                                          # subset of N
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
REPO="$PWD"
SB_SEARCH=scriptsv2/eval_search/eval_search_dqc_chunk.sbatch
SB_BC=scriptsv2/eval_search/eval_bc_dqc_chunk.sbatch
EDIR=/gscratch/robotics/harine/RoboMonkey_data/eval_ckpts
GDIR=/gscratch/robotics/harine/RoboMonkey_data/train/dqc_chunk/eggplant_gaussian_dqc_chunk8_hyak_predhorizon8_rollout4/checkpoints
BC_CKPT=diffusion_policy/data/outputs/2026.06.12/14.35.54_robomonkey_eggplant_bc_robomonkey_eggplant_state/checkpoints/best.ckpt
TEMB="$REPO/scriptsv2/q_fn_eval/task_emb_use3.npy"
N_LIST="${N_LIST:-1 2 4 8 16 32 64}"
EP="${EP:-10}"            # episodes (=seeds 1000..1000+EP-1) per cell
DRY="${DRY:-0}"
VROOT=data/eval/_ft_bon_videos

VERIFIERS=(
  "v2all30k|/gscratch/robotics/harine/RoboMonkey_data/viz_q/ft/dqc_finetune/v2_all/v2_all|30000"
  "v3dist30k|/gscratch/robotics/harine/RoboMonkey_data/viz_q/ft/dqc_finetune/v3_dist/v3_dist_b256_a100|30000"
)
SEARCH_POLICIES=(
  "diff10k|$EDIR/diffusion_dqc_h8_roll4_step_0010000_10k_pathfix.ckpt"
  "diff33k|$EDIR/diffusion_dqc_h8_roll4_step33001_pathfix.ckpt"
  "gauss20k|$GDIR/step_0020000.ckpt"
)
ROUTES=( "weirdlab-ckpt|ckpt-all|gpu:l40s:1|ckpt-gpu" "weirdlab-ckpt|ckpt-all|gpu:l40:1|ckpt-gpu" \
         "weirdlab-ckpt|ckpt-all|gpu:a40:1|ckpt-gpu" "weirdlab-ckpt|ckpt-all|gpu:a100:1|ckpt-gpu" )
VID="SAVE_VIDEOS=${EP},VIDEO_FPS=10,VIZ_Q=1,VIZ_Q_EPISODES=${EP},RENDER_Q=1"

QNAMES="$(squeue -u "$USER" -h -o '%j' 2>/dev/null)"
i=0; sub=0; skip_done=0; skip_q=0
done_cell () { python3 -c "import json,sys;d=json.load(open('$1/eval_log.json'));sys.exit(0 if int(d.get('num_episodes',0))>=$EP else 1)" 2>/dev/null; }
submit () {  # $1=jobname $2=sbatch $3=exports
  IFS='|' read -r acct part gres qos <<< "${ROUTES[$((i % ${#ROUTES[@]}))]}"; i=$((i+1))
  local qf=(); [[ -n "${qos:-}" ]] && qf=(--qos="$qos")
  if [ "$DRY" = "1" ]; then printf 'WOULD %-28s %s/%s\n' "$1" "$part" "$gres"; sub=$((sub+1)); return; fi
  jid=$(sbatch --parsable --job-name="$1" --account="$acct" --partition="$part" --gres="$gres" "${qf[@]}" \
        --cpus-per-task=2 --mem=16G --time=1-00:00:00 --export=ALL,"$3" "$2")
  printf 'submit %-28s %-14s -> %s\n' "$1" "$part/$gres" "$jid"; sub=$((sub+1))
}

for vf in "${VERIFIERS[@]}"; do
  IFS='|' read -r vtag vdir vep <<< "$vf"
  for pol in "${SEARCH_POLICIES[@]}"; do
    IFS='|' read -r pname pckpt <<< "$pol"
    for N in $N_LIST; do
      out="$VROOT/${pname}_${vtag}_n${N}_seed1000_ep${EP}"
      jn="vid_${vtag}_${pname}_n${N}"
      if done_cell "$out"; then skip_done=$((skip_done+1)); continue; fi
      grep -qxF "$jn" <<< "$QNAMES" && { skip_q=$((skip_q+1)); continue; }
      submit "$jn" "$SB_SEARCH" \
        "CHECKPOINT=$pckpt,N_SAMPLES=$N,MODE=softmax,SOFTMAX_TEMP=1.0,NUM_EPISODES=$EP,START_SEED=1000,OUT_DIR=$out,DQC_VERIFIER_CKPT=$vdir,DQC_VERIFIER_EPOCH=$vep,DQC_TEXT_EMB=$TEMB,$VID"
    done
  done
  for N in $N_LIST; do
    out="$VROOT/bcH8_${vtag}_n${N}_seed1000_ep${EP}"
    jn="vid_${vtag}_bcH8_n${N}"
    if done_cell "$out"; then skip_done=$((skip_done+1)); continue; fi
    grep -qxF "$jn" <<< "$QNAMES" && { skip_q=$((skip_q+1)); continue; }
    submit "$jn" "$SB_BC" \
      "CHECKPOINT=$BC_CKPT,BC_SAMPLER=ddpm,N_SAMPLES=$N,MODE=softmax,SOFTMAX_TEMP=1.0,NUM_EPISODES=$EP,START_SEED=1000,OUT_DIR=$out,DQC_CKPT_DIR=$vdir,DQC_RESTORE_EPOCH=$vep,DQC_TEXT_EMB=$TEMB,$VID"
  done
done
echo "=== submitted=$sub skip_done=$skip_done skip_queued=$skip_q  -> videos in $VROOT/<cell>/{videos,search_q/videos}/ ==="
