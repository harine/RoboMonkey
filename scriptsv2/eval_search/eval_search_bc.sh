#!/bin/bash
# Softmax N-sample sweep for the BC diffusion UNet baseline on the eggplant task.
#
# Loads the BC checkpoint (DiffusionUnetImagePolicy), wraps it at eval time with
# BCSearchWrapper + the in-process RoboMonkey verifier, then sweeps N_SAMPLES in
# {1,2,4,8,16,32,64,128}. Each replan draws N independent BC samples, verifier-
# scores all N, and the executed sample is chosen by softmax over all N scores.
#
# Usage: bash scriptsv2/eval_search/eval_search_bc.sh
# Overrides (env):
#   CHECKPOINT       path to bc latest.ckpt (set automatically by sbatch)
#   N_LIST           (default: 1 2 4 8 16 32 64 128)
#   NUM_EPISODES     (default: 50)
#   START_SEED       (default: 1000)
#   SOFTMAX_TEMP     (default: 1.0)
#   NUM_WORKERS      (default: 4)  parallel GPU workers
#   MAX_STEPS        (default: 120)
#   TASK             (default: widowx_put_eggplant_in_basket)
#   CONDA_ENV        (default: monkey-verifier)
#   FORCE            (default: 0) set 1 to re-run completed cells

set -euo pipefail

repo_root="$(cd "$(dirname "$(realpath "$0")")/../.." && pwd)"
cd "$repo_root"

export DIFFUSION_POLICY_ROOT="${DIFFUSION_POLICY_ROOT:-${repo_root}/diffusion_policy}"
export MONKEY_VERIFIER_SRC="${MONKEY_VERIFIER_SRC:-${repo_root}/monkey-verifier/src}"
export MODEL_DIR="${MODEL_DIR:-${repo_root}/monkey-verifier/model_dir}"

# KV_PREFIX stays OFF (same as search sweep — see eval_search_noise_sweep.sh).
export ROBOMONKEY_KV_PREFIX="${ROBOMONKEY_KV_PREFIX:-0}"
export ROBOMONKEY_KV_CHUNK="${ROBOMONKEY_KV_CHUNK:-16}"
export ROBOMONKEY_IMAGE_FEAT_CACHE_SIZE="${ROBOMONKEY_IMAGE_FEAT_CACHE_SIZE:-512}"
export ROBOMONKEY_PROC_IMAGE_CACHE_SIZE="${ROBOMONKEY_PROC_IMAGE_CACHE_SIZE:-512}"

# Verifier config for the BC wrapper (eval_search.py reads these env vars).
export VERIFIER_INSTRUCTION="${VERIFIER_INSTRUCTION:-put the eggplant in the basket}"
export VERIFIER_IMAGE_KEY="${VERIFIER_IMAGE_KEY:-agentview_image}"

read -ra N_ARR <<< "${N_LIST:-1 2 4 8 16 32 64 128 256 512 1024}"
START_SEED="${START_SEED:-1000}"
# Selection mode: "softmax" (sample from softmax over candidate q-values) or
# "argmax" (Best-of-N: execute the single highest-q-value candidate).
MODE="${MODE:-softmax}"
# BC inference sampler: ddpm = ancestral/stochastic (diverse candidates),
# ddim = trained deterministic sampler (control). Read by eval_search.py.
BC_SAMPLER="${BC_SAMPLER:-ddim}"
VIDEO_FPS="${VIDEO_FPS:-10}"
# VIZ_Q gates the eval style:
#   VIZ_Q=1 (default) -> "viz" run: 10 fixed seeds shared across EVERY N so the
#     saved render data (frames + candidate actions + q-values + camera
#     transforms cam_K/cam_T_wc/base_pose_world -> <out>/search_q/*.npz) and the
#     per-episode MP4s are directly comparable cell-to-cell.
#   VIZ_Q=0 -> "SR" run (e.g. Best-of-N success-rate sweep): NUM_EPISODES rollouts
#     from START_SEED, no fixed-seed list, no per-replan dumps, no videos.
VIZ_Q="${VIZ_Q:-1}"
if [[ "$VIZ_Q" == "1" ]]; then
    NUM_SEEDS="${NUM_SEEDS:-10}"
    if [[ -z "${SEEDS:-}" ]]; then
        SEEDS="$(python -c "print(','.join(str($START_SEED+i) for i in range($NUM_SEEDS)))")"
    fi
    # episode count = number of fixed seeds (used only for output-dir naming).
    NUM_EPISODES="$(echo "$SEEDS" | awk -F, '{print NF}')"
    SAVE_VIDEOS="${SAVE_VIDEOS:-$NUM_EPISODES}"
else
    NUM_EPISODES="${NUM_EPISODES:-50}"
    SEEDS=""
    SAVE_VIDEOS="${SAVE_VIDEOS:-0}"
fi
SOFTMAX_TEMP="${SOFTMAX_TEMP:-1.0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-120}"
TASK="${TASK:-widowx_put_eggplant_in_basket}"
CONDA_ENV="${CONDA_ENV:-monkey-verifier}"

# CHECKPOINT: required. Set via sbatch --export or env override.
if [[ -z "${CHECKPOINT:-}" ]]; then
    echo "ERROR: CHECKPOINT env var must be set to the BC latest.ckpt path." >&2
    exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

SUMMARY_DIR="data/eval/search_bc_sweep/_summaries"
mkdir -p "$SUMMARY_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
RESULTS_TXT="${SUMMARY_DIR}/softmax_n_sweep_bc_${TS}.txt"
WORK_DIR="${SUMMARY_DIR}/.sweep_${TS}"
mkdir -p "$WORK_DIR"
QUEUE_FILE="$WORK_DIR/queue.txt"
LOCK_FILE="$WORK_DIR/queue.lock"
: > "$QUEUE_FILE"
: > "$LOCK_FILE"

{
    echo "BC diffusion UNet N-sample sweep"
    echo "checkpoint     : $CHECKPOINT"
    echo "N_SAMPLES      : ${N_ARR[*]}"
    if [[ "$MODE" == "argmax" ]]; then
        echo "mode           : argmax (Best-of-N)"
    else
        echo "mode           : softmax (temp=${SOFTMAX_TEMP})"
    fi
    echo "bc_sampler     : $BC_SAMPLER  (ddpm=stochastic ancestral, ddim=deterministic)"
    if [[ -n "$SEEDS" ]]; then
        echo "seeds          : $SEEDS  ($NUM_EPISODES fixed seeds, shared across all N)"
    else
        echo "episodes       : $NUM_EPISODES from start_seed=$START_SEED"
    fi
    echo "viz_q          : $VIZ_Q  (1 = save candidate actions + q-values + frames + transforms)"
    echo "save_videos    : $SAVE_VIDEOS  (fps=$VIDEO_FPS)"
    echo "num_workers    : $NUM_WORKERS  (1 GPU per worker)"
    echo "task           : $TASK"
    echo "conda_env      : $CONDA_ENV"
    echo "started        : $(date)"
    echo "============================================================"
} | tee "$RESULTS_TXT"

for n in "${N_ARR[@]}"; do
    echo "$n" >> "$QUEUE_FILE"
done

cell_out_dir() {
    local n="$1"
    local tag="softmaxT${SOFTMAX_TEMP}"
    [[ "$MODE" == "argmax" ]] && tag="argmax"
    echo "data/eval/search_bc_sweep/bc/${tag}_${BC_SAMPLER}_n${n}_seed${START_SEED}_ep${NUM_EPISODES}"
}

append_result_row() {
    local out="$1" n="$2" tag="$3"
    python - "$out" "$n" "$RESULTS_TXT" "$tag" <<'PY'
import json, sys
out, n, rt, tag = sys.argv[1:5]
try:
    with open(f"{out}/eval_log.json") as f:
        log = json.load(f)
except FileNotFoundError:
    line = f"[{tag}] bc n={n:>3}  eval_log.json missing"
    print(line); open(rt, "a").write(line + "\n"); sys.exit(0)
ns = int(log.get("num_successes", 0)); ne = int(log.get("num_episodes", 0))
sr = float(log.get("success_rate", 0.0)); tt = log.get("total_time_s")
t_str = f"{tt:.0f}s" if tt is not None else "?"
line = f"[{tag}]           bc n={n:>3}  succ={ns:>3}/{ne}={sr:.3f}  ({t_str})"
print(line); open(rt, "a").write(line + "\n")
PY
}

run_cell() {
    local gpu_id="$1" n="$2"
    local out; out="$(cell_out_dir "$n")"
    local log_file="$out/eval_log.json"
    local cell_log="$WORK_DIR/bc_n${n}_gpu${gpu_id}.log"

    if [[ -f "$log_file" && "${FORCE:-0}" != "1" ]]; then
        echo "[gpu=$gpu_id reuse ] bc n=$n -> $log_file" | tee -a "$RESULTS_TXT"
    else
        echo "[gpu=$gpu_id start ] bc n=$n -> $out" | tee -a "$RESULTS_TXT"
        mkdir -p "$out"
        if ! env CUDA_VISIBLE_DEVICES="$gpu_id" \
                DEVICE="cuda:0" \
                CONDA_ENV="$CONDA_ENV" \
                MODE="$MODE" \
                BC_SAMPLER="$BC_SAMPLER" \
                SOFTMAX_TEMP="$SOFTMAX_TEMP" \
                N_SAMPLES="$n" \
                START_SEED="$START_SEED" \
                SEEDS="$SEEDS" \
                MAX_STEPS="$MAX_STEPS" \
                USE_EMA=1 \
                TASK="$TASK" \
                VIZ_Q="$VIZ_Q" \
                SAVE_VIDEOS="$SAVE_VIDEOS" \
                VIDEO_FPS="$VIDEO_FPS" \
                VERIFIER_INSTRUCTION="$VERIFIER_INSTRUCTION" \
                VERIFIER_IMAGE_KEY="$VERIFIER_IMAGE_KEY" \
                bash scriptsv2/eval_search/eval_search.sh \
                    "$CHECKPOINT" "$NUM_EPISODES" "$out" \
                    > "$cell_log" 2>&1; then
            echo "[gpu=$gpu_id ERROR ] bc n=$n (log: $cell_log)" | tee -a "$RESULTS_TXT"
            return 0
        fi
        echo "[gpu=$gpu_id done  ] bc n=$n" | tee -a "$RESULTS_TXT"
    fi
    append_result_row "$out" "$n" "result"
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
        run_cell "$gpu_id" "$job"
    done
}

pids=()
for ((w=0; w<NUM_WORKERS; w++)); do
    worker "$w" &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

# ---------------- Final table ------------------------------------------------
{
    echo
    echo "============================================================"
    echo "FINAL TABLE — success_rate (BC softmax, temp=${SOFTMAX_TEMP}, ${NUM_EPISODES} ep)"
    printf "%-12s" "policy"
    for n in "${N_ARR[@]}"; do printf " %7s" "n=$n"; done
    echo
    echo "------------------------------------------------------------"
} | tee -a "$RESULTS_TXT"

row="$(printf "%-12s" "bc")"
for n in "${N_ARR[@]}"; do
    out="$(cell_out_dir "$n")"
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

{
    echo "------------------------------------------------------------"
    echo "finished      : $(date)"
} | tee -a "$RESULTS_TXT"

echo
echo "[eval_search_bc] full results -> $RESULTS_TXT"
echo "[eval_search_bc] per-cell logs -> $WORK_DIR/"
