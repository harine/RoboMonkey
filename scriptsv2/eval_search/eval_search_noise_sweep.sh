#!/bin/bash
# Softmax-selection N-sample sweep over the three noise-comparison search
# policies (in-process RoboMonkey verifier, SimplerEnv eggplant task):
#
#   noise0.4 : tunable_corruption @ obs_noise_level=0.4   (unconditioned)
#   noise0.8 : tunable_corruption @ obs_noise_level=0.8   (unconditioned)
#   tmrl     : noise_conditioned  (model is told a noise level tcont in [0,1];
#              we sweep tcont over TCONT_LIST. tcont=0 -> clean obs.)
#
# For every (policy, tcont) we sweep N_SAMPLES in {1,2,4,8,16,32,64,128}. Each
# replan draws N autoregressive candidates, the verifier scores all N, and the
# executed candidate is sampled from softmax(values / SOFTMAX_TEMP) over ALL N
# (MODE=softmax). Each cell runs NUM_EPISODES episodes from START_SEED.
#
# Cells: noise0.4 (|N|) + noise0.8 (|N|) + tmrl (|TCONT_LIST| x |N|).
# tcont only applies to the noise-conditioned (tmrl) policy; the
# tunable_corruption policies don't take a noise-level input, so they run once
# per N with no tcont.
#
# Parallelism: NUM_WORKERS GPU workers pull cells from a shared queue, each
# pinned to one GPU (CUDA_VISIBLE_DEVICES = worker index).
#
# Usage: bash scriptsv2/eval_search/eval_search_noise_sweep.sh
# Overrides (env):
#   N_LIST="1 2 4 8 16 32 64 128"   sample-count sweep
#   TCONT_LIST="0.0 0.4 0.8"        tmrl noise-level sweep (tmrl only)
#   POLICIES="noise0.4 noise0.8 tmrl"  subset of policies to run
#   NUM_EPISODES=50                 episodes per cell
#   START_SEED=1000                 first episode seed (episode i -> seed+i)
#   SOFTMAX_TEMP=1.0                softmax selection temperature
#   NUM_WORKERS=4                   parallel GPU workers (<= #visible GPUs)
#   MAX_STEPS=120                   env steps per episode
#   TASK=widowx_put_eggplant_in_basket
#   CONDA_ENV=monkey-verifier       env with the in-process verifier deps
#   FORCE=1                         re-run cells whose eval_log.json exists

set -euo pipefail

repo_root="$(cd "$(dirname "$(realpath "$0")")/../.." && pwd)"
cd "$repo_root"

# eval_search.sh falls back to a hardcoded /home/harine path that does not
# exist on klone; pin the real diffusion_policy root so PYTHONPATH is correct.
export DIFFUSION_POLICY_ROOT="${DIFFUSION_POLICY_ROOT:-${repo_root}/diffusion_policy}"
export MONKEY_VERIFIER_SRC="${MONKEY_VERIFIER_SRC:-${repo_root}/monkey-verifier/src}"
export MODEL_DIR="${MODEL_DIR:-${repo_root}/monkey-verifier/model_dir}"

# In-process verifier speedup — same as training used: the image-feature /
# processed-image caches in the default score_paired path (the image+prefix is
# encoded once per frame and reused across the N candidate scorings).
# KV_PREFIX is a SEPARATE, faster scorer (score_paired_kvprefix) that training
# never enabled and that currently crashes in this env (_slice_kv hits a None
# layer in past_key_values), so it stays OFF. Set ROBOMONKEY_KV_PREFIX=1 only
# after that verifier bug is fixed.
export ROBOMONKEY_KV_PREFIX="${ROBOMONKEY_KV_PREFIX:-0}"
export ROBOMONKEY_KV_CHUNK="${ROBOMONKEY_KV_CHUNK:-16}"
export ROBOMONKEY_IMAGE_FEAT_CACHE_SIZE="${ROBOMONKEY_IMAGE_FEAT_CACHE_SIZE:-512}"
export ROBOMONKEY_PROC_IMAGE_CACHE_SIZE="${ROBOMONKEY_PROC_IMAGE_CACHE_SIZE:-512}"

OUT_ROOT="${repo_root}/diffusion_policy/data/outputs"
declare -A CKPTS=(
    [noise0.0]="${OUT_ROOT}/2026.06.05/14.35.14_robomonkey_eggplant_search_tunable_corruption_noise0.0_robomonkey_eggplant_state/checkpoints/latest.ckpt"
    [noise0.4]="${OUT_ROOT}/2026.06.03/18.32.58_robomonkey_eggplant_search_tunable_corruption_noise0.4_robomonkey_eggplant_state/checkpoints/latest.ckpt"
    [noise0.8]="${OUT_ROOT}/2026.06.03/18.25.46_robomonkey_eggplant_search_tunable_corruption_noise0.8_robomonkey_eggplant_state/checkpoints/latest.ckpt"
    [tmrl]="${OUT_ROOT}/2026.06.03/18.29.43_robomonkey_eggplant_search_noise_conditioned_robomonkey_eggplant_state/checkpoints/latest.ckpt"
)
# CKPT_DIR override: if set, every policy's checkpoint is taken from
# ${CKPT_DIR}/<policy>/checkpoints/latest.ckpt instead of the hardcoded paths
# above. Used to eval freshly-trained chunk-sum checkpoints (data/train/chunksum).
if [[ -n "${CKPT_DIR:-}" ]]; then
    for pol in "${!CKPTS[@]}"; do
        CKPTS[$pol]="${CKPT_DIR}/${pol}/checkpoints/latest.ckpt"
    done
fi
# Policies that take a tcont noise-level input (only the noise-conditioned one).
declare -A TCONT_POLICY=( [tmrl]=1 )

read -ra POLICY_ARR <<< "${POLICIES:-noise0.4 noise0.8 tmrl}"
read -ra N_ARR      <<< "${N_LIST:-1 2 4 8 16 32 64 128 256 512 1024}"
read -ra TCONT_ARR  <<< "${TCONT_LIST:-0.4 0.8 0.0}"
NUM_EPISODES="${NUM_EPISODES:-50}"
START_SEED="${START_SEED:-1000}"
SOFTMAX_TEMP="${SOFTMAX_TEMP:-1.0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-120}"
TASK="${TASK:-widowx_put_eggplant_in_basket}"
CONDA_ENV="${CONDA_ENV:-monkey-verifier}"
# Render data: save per-replan candidate actions + q-values + frames + camera
# transforms (search_q/*.npz) and an MP4 for the first VIZ_Q_EPISODES episodes.
# Episode i uses seed START_SEED+i, so these are the SAME fixed seeds across
# every N cell -> directly comparable renders per sample count. Full SR is still
# computed over all NUM_EPISODES.
VIZ_Q="${VIZ_Q:-1}"
VIZ_Q_EPISODES="${VIZ_Q_EPISODES:-10}"
SAVE_VIDEOS="${SAVE_VIDEOS:-10}"
VIDEO_FPS="${VIDEO_FPS:-10}"

# Verify every requested checkpoint exists up front.
for pol in "${POLICY_ARR[@]}"; do
    ckpt="${CKPTS[$pol]:-}"
    if [[ -z "$ckpt" ]]; then
        echo "ERROR: unknown policy '$pol' (known: ${!CKPTS[*]})" >&2; exit 1
    fi
    if [[ ! -f "$ckpt" ]]; then
        echo "ERROR: checkpoint for '$pol' not found: $ckpt" >&2; exit 1
    fi
done

SUMMARY_DIR="data/eval/search_noise_sweep/_summaries"
mkdir -p "$SUMMARY_DIR"
# Timestamp from the environment (date is fine in a shell launcher).
TS="$(date +%Y%m%d_%H%M%S)"
RESULTS_TXT="${SUMMARY_DIR}/softmax_n_sweep_${TS}.txt"
WORK_DIR="${SUMMARY_DIR}/.sweep_${TS}"
mkdir -p "$WORK_DIR"
QUEUE_FILE="$WORK_DIR/queue.txt"
LOCK_FILE="$WORK_DIR/queue.lock"
: > "$QUEUE_FILE"
: > "$LOCK_FILE"

# Row keys = "pol|tcont" (tcont="-" for policies with no noise-level input).
# Used for both the queue and the final table.
ROW_KEYS=()
for pol in "${POLICY_ARR[@]}"; do
    if [[ -n "${TCONT_POLICY[$pol]:-}" ]]; then
        for tc in "${TCONT_ARR[@]}"; do ROW_KEYS+=("${pol}|${tc}"); done
    else
        ROW_KEYS+=("${pol}|-")
    fi
done

{
    echo "search-policy softmax N-sample sweep"
    echo "policies       : ${POLICY_ARR[*]}"
    echo "N_SAMPLES      : ${N_ARR[*]}"
    echo "tcont (tmrl)   : ${TCONT_ARR[*]}"
    echo "mode           : softmax (temp=${SOFTMAX_TEMP})"
    echo "num_episodes   : $NUM_EPISODES per cell"
    echo "start_seed     : $START_SEED"
    echo "num_workers    : $NUM_WORKERS  (1 GPU per worker)"
    echo "task           : $TASK"
    echo "conda_env      : $CONDA_ENV"
    echo "started        : $(date)"
    echo "============================================================"
} | tee "$RESULTS_TXT"

# Build the work queue: one line per (policy, n, tcont) cell.
for key in "${ROW_KEYS[@]}"; do
    pol="${key%%|*}"; tc="${key##*|}"
    for n in "${N_ARR[@]}"; do
        echo "$pol $n $tc" >> "$QUEUE_FILE"
    done
done

# A short display label for a (pol, tcont) row, e.g. "tmrl@0.4" or "noise0.4".
row_label() {
    local pol="$1" tc="$2"
    if [[ "$tc" == "-" ]]; then echo "$pol"; else echo "${pol}@${tc}"; fi
}

cell_out_dir() {
    local pol="$1" n="$2" tc="$3"
    local base="data/eval/search_noise_sweep/${pol}/softmaxT${SOFTMAX_TEMP}"
    if [[ "$tc" == "-" ]]; then
        echo "${base}_n${n}_seed${START_SEED}_ep${NUM_EPISODES}"
    else
        echo "${base}_tcont${tc}_n${n}_seed${START_SEED}_ep${NUM_EPISODES}"
    fi
}

append_result_row() {
    local out="$1" label="$2" n="$3" tag="$4"
    python - "$out" "$label" "$n" "$RESULTS_TXT" "$tag" <<'PY'
import json, sys
out, label, n, rt, tag = sys.argv[1:6]
try:
    with open(f"{out}/eval_log.json") as f:
        log = json.load(f)
except FileNotFoundError:
    line = f"[{tag}] {label:>12} n={n:>3}  eval_log.json missing"
    print(line); open(rt, "a").write(line + "\n"); sys.exit(0)
ns = int(log.get("num_successes", 0)); ne = int(log.get("num_episodes", 0))
sr = float(log.get("success_rate", 0.0)); tt = log.get("total_time_s")
t_str = f"{tt:.0f}s" if tt is not None else "?"
line = f"[{tag}] {label:>12} n={n:>3}  succ={ns:>3}/{ne}={sr:.3f}  ({t_str})"
print(line); open(rt, "a").write(line + "\n")
PY
}

run_cell() {
    local gpu_id="$1" pol="$2" n="$3" tc="$4"
    local ckpt="${CKPTS[$pol]}"
    local label; label="$(row_label "$pol" "$tc")"
    local out; out="$(cell_out_dir "$pol" "$n" "$tc")"
    local log_file="$out/eval_log.json"
    local tctag=""; [[ "$tc" != "-" ]] && tctag="_tcont${tc}"
    local cell_log="$WORK_DIR/${pol}${tctag}_n${n}_gpu${gpu_id}.log"

    if [[ -f "$log_file" && "${FORCE:-0}" != "1" ]]; then
        echo "[gpu=$gpu_id reuse ] $label n=$n -> $log_file" | tee -a "$RESULTS_TXT"
    else
        echo "[gpu=$gpu_id start ] $label n=$n -> $out" | tee -a "$RESULTS_TXT"
        mkdir -p "$out"
        # TCONT only set for policies that accept a noise-level input.
        local tcont_env=()
        [[ "$tc" != "-" ]] && tcont_env=(TCONT="$tc")
        if ! env CUDA_VISIBLE_DEVICES="$gpu_id" \
                DEVICE="cuda:0" \
                CONDA_ENV="$CONDA_ENV" \
                MODE="softmax" \
                SOFTMAX_TEMP="$SOFTMAX_TEMP" \
                N_SAMPLES="$n" \
                START_SEED="$START_SEED" \
                MAX_STEPS="$MAX_STEPS" \
                USE_EMA=1 \
                TASK="$TASK" \
                VIZ_Q="$VIZ_Q" \
                VIZ_Q_EPISODES="$VIZ_Q_EPISODES" \
                SAVE_VIDEOS="$SAVE_VIDEOS" \
                VIDEO_FPS="$VIDEO_FPS" \
                "${tcont_env[@]}" \
                bash scriptsv2/eval_search/eval_search.sh \
                    "$ckpt" "$NUM_EPISODES" "$out" \
                    > "$cell_log" 2>&1; then
            echo "[gpu=$gpu_id ERROR ] $label n=$n (log: $cell_log)" | tee -a "$RESULTS_TXT"
            return 0
        fi
        echo "[gpu=$gpu_id done  ] $label n=$n" | tee -a "$RESULTS_TXT"
    fi
    append_result_row "$out" "$label" "$n" "result"
}

worker() {
    local gpu_id="$1"
    while :; do
        local job=""
        {
            flock 9
            if [[ -s "$QUEUE_FILE" ]]; then
                job="$(head -n1 "$QUEUE_FILE")"
                sed -i '1d' "$QUEUE_FILE"
            fi
        } 9>"$LOCK_FILE"
        [[ -z "$job" ]] && break
        local pol n tc
        read -r pol n tc <<< "$job"
        run_cell "$gpu_id" "$pol" "$n" "$tc"
    done
}

pids=()
for ((w=0; w<NUM_WORKERS; w++)); do
    worker "$w" &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

# ---------------- Final table: success_rate vs N per (policy, tcont) --------
{
    echo
    echo "============================================================"
    echo "FINAL TABLE — success_rate (softmax, temp=${SOFTMAX_TEMP}, ${NUM_EPISODES} ep)"
    printf "%-12s" "policy"; for n in "${N_ARR[@]}"; do printf " %7s" "n=$n"; done; echo
    echo "------------------------------------------------------------"
} | tee -a "$RESULTS_TXT"

for key in "${ROW_KEYS[@]}"; do
    pol="${key%%|*}"; tc="${key##*|}"
    label="$(row_label "$pol" "$tc")"
    row="$(printf "%-12s" "$label")"
    for n in "${N_ARR[@]}"; do
        out="$(cell_out_dir "$pol" "$n" "$tc")"
        sr="$(python - "$out" <<'PY'
import json, sys
try:
    with open(f"{sys.argv[1]}/eval_log.json") as f:
        print(f"{json.load(f).get('success_rate', float('nan')):.3f}")
except Exception:
    print("  ----")
PY
)"
        row+="$(printf " %7s" "$sr")"
    done
    echo "$row" | tee -a "$RESULTS_TXT"
done

{
    echo "------------------------------------------------------------"
    echo "finished      : $(date)"
} | tee -a "$RESULTS_TXT"

echo
echo "[eval_search_noise_sweep] full results -> $RESULTS_TXT"
echo "[eval_search_noise_sweep] per-cell logs -> $WORK_DIR/"
