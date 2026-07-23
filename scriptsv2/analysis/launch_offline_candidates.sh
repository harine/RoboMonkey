#!/bin/bash
# -----------------------------------------------------------------------------
# Launch the offline candidate-sampling matrix:
#   3 tasks x {train,test} x {search,bc} x N in {1,4,16,64}  = 48 cells
# Each cell: dataset states -> N candidates -> Q per candidate -> Q(a*) + a*_rank -> npz
#
# Routes to robotics / cse / ckpt only (no weirdlab). Idempotent: a cell with npz is skipped.
# Vars are EXPORTED into the environment and passed with plain `--export=ALL`, because
# instruction strings contain spaces (and --export is comma-delimited).
#
#   bash scriptsv2/analysis/launch_offline_candidates.sh              # all 48
#   TASKS=spoon N_LIST=16 DRY=1 bash scriptsv2/analysis/launch_offline_candidates.sh
# -----------------------------------------------------------------------------
set -uo pipefail
cd /mmfs1/home/harine/RoboMonkey
REPO="$PWD"
SB=scriptsv2/analysis/offline_candidates.sbatch
Q="$REPO/scriptsv2/q_fn_eval"
MT=/gscratch/robotics/harine/RoboMonkey_data/train/rmse_multitask
FL=/gscratch/robotics/harine/RoboMonkey_data/train/rmse_filtered
BCD=/mmfs1/home/harine/RoboMonkey/diffusion_policy/data/train/bc_denoiser
DATA=/gscratch/robotics/harine/data

TASKS="${TASKS:-eggplant spoon stack_cube}"
SPLITS="${SPLITS:-train test}"
KINDS="${KINDS:-search bc}"
N_LIST="${N_LIST:-1 4 16 64}"
NUM_EPISODES="${NUM_EPISODES:-50}"
DRY="${DRY:-0}"
OUT_ROOT="${OUT_ROOT:-data/eval/train_test_analysis}"

ROUTES=( "robotics|ckpt-all|gpu:a40:1" "cse|gpu-l40s|gpu:l40s:1" "cse|gpu-a100|gpu:a100:1" \
         "robotics|gpu-a40|gpu:a40:1" "robotics|ckpt-all|gpu:a100:1" )

find_ckpt(){ ls "$1"/checkpoints/step_*.ckpt 2>/dev/null | awk -v M=$(($2*1000)) \
  '{n=$0;sub(/.*step_0*/,"",n);sub(/\.ckpt/,"",n);n+=0; if(n>=M-200&&n<=M+1200){d=(n>M)?n-M:M-n; if(b==""||d<bd){b=$0;bd=d}}} END{if(b)print b}'; }

# task_key|env_task|search_dir|search_mk|bc_ckpt|text_emb|train_ds|test_ds|instruction
POLS=(
"eggplant|widowx_put_eggplant_in_basket|$FL/diff_search_rmse_sub50_diverse_data|90|$BCD/eggplant_bc_unet_h8_50k_sub50_diverse_nobs1/checkpoints/latest.ckpt|$Q/task_emb_use3.npy|strict_distance_success_sub50_diverse_start|strict_distance_success_testset_not_in_sub100|put the eggplant in the basket"
"spoon|widowx_spoon_on_towel|$MT/diff_search_rmse_spoon_sub50_diverse|90|$BCD/bc_unet_h8_50k_spoon_sub50_nobs1/checkpoints/step_0010001.ckpt|$Q/task_emb_spoon_use3.npy|spoon_on_towel_strict_sub50_diverse_start|spoon_on_towel_strict_testset_not_in_sub100|put the spoon on the towel"
"stack_cube|widowx_stack_cube|$MT/diff_search_rmse_stack_sub50_diverse|60|$BCD/bc_unet_h8_50k_stack_sub50_nobs1/checkpoints/step_0030002.ckpt|$Q/task_emb_stack_use3.npy|stack_cube_strict_sub50_diverse_start|stack_cube_strict_testset_not_in_sub100|stack the green block on the yellow block"
)

mkdir -p "$OUT_ROOT/_logs"
i=0; nsub=0; sk=0
for spec in "${POLS[@]}"; do
  IFS='|' read -r tkey tenv sdir smk bcck temb trds teds instr <<< "$spec"
  grep -qw "$tkey" <<<"$TASKS" || continue
  sck=$(find_ckpt "$sdir" "$smk")
  [ -f "$sck" ] || { echo "NO search ckpt for $tkey ($sdir @${smk}k)"; continue; }
  [ -f "$bcck" ] || { echo "NO bc ckpt for $tkey"; continue; }

  for split in $SPLITS; do
    case "$split" in
      train) ds="$trds";;
      test)  ds="$teds";;
    esac
    zarr="$DATA/$ds/$ds.zarr"
    [ -d "$zarr" ] || { echo "NO zarr $zarr"; continue; }

    for kind in $KINDS; do
      case "$kind" in search) ck="$sck";; bc) ck="$bcck";; esac
      for N in $N_LIST; do
        out="$OUT_ROOT/$tkey/offline/${kind}_${split}_N${N}"
        ls "$out"/search_q/*.npz >/dev/null 2>&1 && { sk=$((sk+1)); continue; }

        IFS='|' read -r acct part gres <<< "${ROUTES[$((i % ${#ROUTES[@]}))]}"; i=$((i+1))
        jn="oc_${tkey:0:3}${kind:0:1}${split:0:2}N${N}"

        export CKPT="$ck" ZARR="$zarr" POLICY_KIND="$kind" SPLIT="$split" \
               TASK="$tenv" INSTRUCTION="$instr" TEXT_EMB="$temb" \
               N_SAMPLES="$N" OUT_DIR="$out" NUM_EPISODES="$NUM_EPISODES" \
               STRIDE=4 N_ACTION_STEPS=4

        if [ "$DRY" = 1 ]; then
          echo "WOULD $jn -> $acct/$part  out=$out"
          echo "        ckpt=$(basename "$ck")  emb=$(basename "$temb")"
          nsub=$((nsub+1)); continue
        fi
        jid=$(sbatch --parsable --job-name="$jn" --account="$acct" --partition="$part" \
              --gres="$gres" --export=ALL "$SB")
        echo "sub $jn -> $jid ($acct/$part)"; nsub=$((nsub+1))
      done
    done
  done
done
echo "=== offline candidates: submitted=$nsub skipped=$sk ==="
