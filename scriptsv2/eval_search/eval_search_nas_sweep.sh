#!/bin/bash
# Softmax-selection N-sample sweep for a trained search policy, evaluated at a
# chosen number of EXECUTED action steps per replan (N_ACTION_STEPS).
#
# This is a sibling of eval_search_noise_sweep.sh used to ablate the executed
# action horizon: the policy ALWAYS predicts the full horizon (16), but
# n_action_steps controls how many steps are executed per replan (and, with the
# default SCORE_WINDOW=executed, how many steps the verifier scores). The
# N_ACTION_STEPS override is applied in eval_search.py at load time.
#
# Outputs live under data/eval/search_nas_sweep/ (tagged nas<N>) so they never
# collide with the n_action_steps=8 runs in data/eval/search_noise_sweep/.
#
# Usage: bash scriptsv2/eval_search/eval_search_nas_sweep.sh
# Overrides (env):
#   N_ACTION_STEPS=4                executed steps/replan (default 4)
#   POLICIES="noise0.0"            subset of policies to run
#   N_LIST="1 2 4 8 16 32 64 128"  sample-count sweep
#   TCONT_LIST="0.0 0.4 0.8"       tmrl noise-level sweep (tmrl only)
#   NUM_EPISODES=50  START_SEED=1000  SOFTMAX_TEMP=1.0  NUM_WORKERS=4
#   MAX_STEPS=120  TASK=...  CONDA_ENV=monkey-verifier  FORCE=1

set -euo pipefail

repo_root="$(cd "$(dirname "$(realpath "$0")")/../.." && pwd)"
cd "$repo_root"

export DIFFUSION_POLICY_ROOT="${DIFFUSION_POLICY_ROOT:-${repo_root}/diffusion_policy}"
export MONKEY_VERIFIER_SRC="${MONKEY_VERIFIER_SRC:-${repo_root}/monkey-verifier/src}"
export MODEL_DIR="${MODEL_DIR:-${repo_root}/monkey-verifier/model_dir}"

export ROBOMONKEY_KV_PREFIX="${ROBOMONKEY_KV_PREFIX:-0}"
export ROBOMONKEY_KV_CHUNK="${ROBOMONKEY_KV_CHUNK:-16}"
export ROBOMONKEY_IMAGE_FEAT_CACHE_SIZE="${ROBOMONKEY_IMAGE_FEAT_CACHE_SIZE:-512}"
export ROBOMONKEY_PROC_IMAGE_CACHE_SIZE="${ROBOMONKEY_PROC_IMAGE_CACHE_SIZE:-512}"

# Executed action steps per replan (read by eval_search.py via os.environ).
NAS="${N_ACTION_STEPS:-4}"
export N_ACTION_STEPS="$NAS"

# Optional run label to keep distinct evals (e.g. different verifier
# checkpoints) from sharing cell dirs. Output: .../<pol>/<RUN_TAG>_nas<N>_...
RUN_TAG="${RUN_TAG:-}"
TAG_PREFIX="${RUN_TAG:+${RUN_TAG}_}"

OUT_ROOT="${repo_root}/diffusion_policy/data/outputs"
declare -A CKPTS=(
    [noise0.0]="${OUT_ROOT}/2026.06.05/14.35.14_robomonkey_eggplant_search_tunable_corruption_noise0.0_robomonkey_eggplant_state/checkpoints/latest.ckpt"
    [noise0.4]="${OUT_ROOT}/2026.06.03/18.32.58_robomonkey_eggplant_search_tunable_corruption_noise0.4_robomonkey_eggplant_state/checkpoints/latest.ckpt"
    [noise0.8]="${OUT_ROOT}/2026.06.03/18.25.46_robomonkey_eggplant_search_tunable_corruption_noise0.8_robomonkey_eggplant_state/checkpoints/latest.ckpt"
    [tmrl]="${OUT_ROOT}/2026.06.03/18.29.43_robomonkey_eggplant_search_noise_conditioned_robomonkey_eggplant_state/checkpoints/latest.ckpt"
)
if [[ -n "${CKPT_DIR:-}" ]]; then
    for pol in "${!CKPTS[@]}"; do
        CKPTS[$pol]="${CKPT_DIR}/${pol}/checkpoints/latest.ckpt"
    done
fi
declare -A TCONT_POLICY=( [tmrl]=1 )

read -ra POLICY_ARR <<< "${POLICIES:-noise0.0}"
read -ra N_ARR      <<< "${N_LIST:-1 2 4 8 16 32 64 128}"
read -ra TCONT_ARR  <<< "${TCONT_LIST:-0.4 0.8 0.0}"
NUM_EPISODES="${NUM_EPISODES:-50}"
START_SEED="${START_SEED:-1000}"
SOFTMAX_TEMP="${SOFTMAX_TEMP:-1.0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-120}"
TASK="${TASK:-widowx_put_eggplant_in_basket}"
CONDA_ENV="${CONDA_ENV:-monkey-verifier}"
VIZ_Q="${VIZ_Q:-1}"
VIZ_Q_EPISODES="${VIZ_Q_EPISODES:-10}"
SAVE_VIDEOS="${SAVE_VIDEOS:-10}"
VIDEO_FPS="${VIDEO_FPS:-10}"

for pol in "${POLICY_ARR[@]}"; do
    ckpt="${CKPTS[$pol]:-}"
    if [[ -z "$ckpt" ]]; then
        echo "ERROR: unknown policy '$pol' (known: ${!CKPTS[*]})" >&2; exit 1
    fi
    if [[ ! -f "$ckpt" ]]; then
        echo "ERROR: checkpoint for '$pol' not found: $ckpt" >&2; exit 1
    fi
done

SUMMARY_DIR="data/eval/search_nas_sweep/_summaries"
mkdir -p "$SUMMARY_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
RESULTS_TXT="${SUMMARY_DIR}/${TAG_PREFIX}nas${NAS}_softmax_n_sweep_${TS}.txt"
WORK_DIR="${SUMMARY_DIR}/.sweep_${TAG_PREFIX}nas${NAS}_${TS}"
mkdir -p "$WORK_DIR"
QUEUE_FILE="$WORK_DIR/queue.txt"
LOCK_FILE="$WORK_DIR/queue.lock"
: > "$QUEUE_FILE"
: > "$LOCK_FILE"

ROW_KEYS=()
for pol in "${POLICY_ARR[@]}"; do
    if [[ -n "${TCONT_POLICY[$pol]:-}" ]]; then
        for tc in "${TCONT_ARR[@]}"; do ROW_KEYS+=("${pol}|${tc}"); done
    else
        ROW_KEYS+=("${pol}|-")
    fi
done

{
    echo "search-policy softmax N-sample sweep  (executed action steps = ${NAS})"
    echo "policies       : ${POLICY_ARR[*]}"
    echo "N_ACTION_STEPS : ${NAS}  (steps executed per replan; horizon predicted = 16)"
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

for key in "${ROW_KEYS[@]}"; do
    pol="${key%%|*}"; tc="${key##*|}"
    for n in "${N_ARR[@]}"; do
        echo "$pol $n $tc" >> "$QUEUE_FILE"
    done
done

row_label() {
    local pol="$1" tc="$2"
    if [[ "$tc" == "-" ]]; then echo "$pol"; else echo "${pol}@${tc}"; fi
}

cell_out_dir() {
    local pol="$1" n="$2" tc="$3"
    local base="data/eval/search_nas_sweep/${pol}/${TAG_PREFIX}nas${NAS}_softmaxT${SOFTMAX_TEMP}"
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
        echo "[gpu=$gpu_id start ] $label n=$n (nas=${NAS}) -> $out" | tee -a "$RESULTS_TXT"
        mkdir -p "$out"
        local tcont_env=()
        [[ "$tc" != "-" ]] && tcont_env=(TCONT="$tc")
        if ! env CUDA_VISIBLE_DEVICES="$gpu_id" \
                DEVICE="cuda:0" \
                CONDA_ENV="$CONDA_ENV" \
                MODE="softmax" \
                SOFTMAX_TEMP="$SOFTMAX_TEMP" \
                N_SAMPLES="$n" \
                N_ACTION_STEPS="$NAS" \
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

{
    echo
    echo "============================================================"
    echo "FINAL TABLE — success_rate (nas=${NAS}, softmax temp=${SOFTMAX_TEMP}, ${NUM_EPISODES} ep)"
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
echo "[eval_search_nas_sweep] full results -> $RESULTS_TXT"
echo "[eval_search_nas_sweep] per-cell logs -> $WORK_DIR/"
