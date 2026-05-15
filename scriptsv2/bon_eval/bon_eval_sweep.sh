#!/bin/bash
# Parallel BoN sweep with the in-process improved verifier across the
# (start_seed x k) grid. Default: 3 start seeds x k in {1,2,4,8,16,32,64}
# (2^0..2^6). Each cell runs NUM_EPISODES episodes. The script reports
# num_successes at the first 50 and at all 100 episodes, plus a final
# seed-aggregated table. Streams ongoing results to a txt file.
#
# Parallelism: NUM_WORKERS GPU workers pull cells from a shared queue. Each
# worker pins itself to one GPU (CUDA_VISIBLE_DEVICES = worker index). When
# launched from the sbatch (which reserves 2 a40s + all 52 node CPUs),
# NUM_WORKERS=2 is the natural default.
#
# Usage: bash scriptsv2/bon_eval/bon_eval_sweep.sh
# Overrides:
#   KS="1 2 4 8 16 32 64"        k sweep
#   START_SEEDS="17 38 99"       starting seeds (vary asset positions)
#   NUM_EPISODES=100             episodes per cell
#   NUM_WORKERS=2                parallel workers (<= #visible GPUs)
#   FORCE=1                      re-run cells whose eval_log.json exists

set -euo pipefail

export TASK="widowx_put_eggplant_in_basket"
UNET_CKPT="/mmfs1/home/harine/RoboMonkey/data/models/data/outputs/2026.04.29/16.45.01_train_diffusion_unet_eggplant_in_basket_lowdim_eggplant_in_basket_lowdim/checkpoints/latest.ckpt"
RUN_NAME="$(basename "$(dirname "$(dirname "$UNET_CKPT")")")"

read -ra START_SEEDS <<< "${START_SEEDS:-17 38 99}"
read -ra KS          <<< "${KS:-1 2 4 8 16 32 64}"
NUM_EPISODES="${NUM_EPISODES:-100}"
NUM_WORKERS="${NUM_WORKERS:-2}"
REPLAN_N=4
SCORE_N=4

repo_root="$(cd "$(dirname "$(realpath "$0")")/../.." && pwd)"
cd "$repo_root"

SUMMARY_DIR="data/eval/bon/_summaries"
mkdir -p "$SUMMARY_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
RESULTS_TXT="${SUMMARY_DIR}/improved_verifier_sweep_${TS}.txt"
WORK_DIR="${SUMMARY_DIR}/.bon_sweep_${TS}"
mkdir -p "$WORK_DIR"
QUEUE_FILE="$WORK_DIR/queue.txt"
LOCK_FILE="$WORK_DIR/queue.lock"
: > "$QUEUE_FILE"
: > "$LOCK_FILE"

{
    echo "BoN sweep — improved verifier — eggplant U-Net (parallel)"
    echo "checkpoint    : $UNET_CKPT"
    echo "task          : $TASK"
    echo "start_seeds   : ${START_SEEDS[*]}"
    echo "k values      : ${KS[*]}"
    echo "num_episodes  : $NUM_EPISODES per cell"
    echo "num_workers   : $NUM_WORKERS  (1 GPU per worker)"
    echo "replan=$REPLAN_N  score=$SCORE_N"
    echo "started       : $(date)"
    echo "============================================================"
} | tee "$RESULTS_TXT"

# Build work queue: one line per (seed, k) cell.
for SEED in "${START_SEEDS[@]}"; do
    for K in "${KS[@]}"; do
        echo "$SEED $K" >> "$QUEUE_FILE"
    done
done

append_result_row() {
    local out="$1" seed="$2" k="$3" tag="$4"
    python - "$out" "$seed" "$k" "$RESULTS_TXT" "$tag" <<'PY'
import json, sys
out, seed, k, rt, tag = sys.argv[1:6]
try:
    with open(f"{out}/episodes.jsonl") as f:
        eps = [json.loads(l) for l in f if l.strip()]
except FileNotFoundError:
    line = f"[{tag}] seed={seed:>3} k={k:>3}  episodes.jsonl missing"
    print(line); open(rt, "a").write(line + "\n"); sys.exit(0)
total_t = None
try:
    with open(f"{out}/eval_log.json") as f:
        total_t = json.load(f).get("total_time_s")
except FileNotFoundError:
    pass
s50  = sum(int(e.get("success", False)) for e in eps[:50])
s100 = sum(int(e.get("success", False)) for e in eps[:100])
t_str = f"{total_t:.0f}s" if total_t is not None else "?"
line = (f"[{tag}] seed={seed:>3} k={k:>3}  "
        f"succ@50={s50:>3}/50={s50/50:.2f}  "
        f"succ@100={s100:>3}/100={s100/100:.2f}  ({t_str})")
print(line); open(rt, "a").write(line + "\n")
PY
}

run_cell() {
    local gpu_id="$1" seed="$2" k="$3"
    local fix_tag=""
    if [[ "${FIX_SEED:-0}" == "1" ]]; then
        fix_tag="_fixed"
    fi
    local out="data/eval/bon/${RUN_NAME}/replan${REPLAN_N}_k${k}_score${SCORE_N}_startseed${seed}${fix_tag}"
    local log_file="$out/eval_log.json"
    local cell_log="$WORK_DIR/seed${seed}_k${k}_gpu${gpu_id}.log"

    if [[ -f "$log_file" && "${FORCE:-0}" != "1" ]]; then
        echo "[gpu=$gpu_id reuse ] seed=$seed k=$k -> $log_file" | tee -a "$RESULTS_TXT"
    else
        echo "[gpu=$gpu_id start ] seed=$seed k=$k -> $out" | tee -a "$RESULTS_TXT"
        if ! CUDA_VISIBLE_DEVICES="$gpu_id" \
                DEVICE="cuda:0" \
                START_SEED="$seed" \
                BON_K="$k" \
                BON_REPLAN_EVERY_N_STEPS="$REPLAN_N" \
                BON_SCORE_NUM_ACTIONS="$SCORE_N" \
                USE_EMA=1 \
                REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-0}" \
                FIX_SEED="${FIX_SEED:-0}" \
                VIZ_Q="${VIZ_Q:-0}" \
                bash scriptsv2/eval_diffusion/eval_diffusion.sh \
                    "$UNET_CKPT" "$NUM_EPISODES" "$out" \
                    > "$cell_log" 2>&1; then
            echo "[gpu=$gpu_id ERROR ] seed=$seed k=$k (log: $cell_log)" | tee -a "$RESULTS_TXT"
            return 0
        fi
        echo "[gpu=$gpu_id done  ] seed=$seed k=$k" | tee -a "$RESULTS_TXT"
    fi
    append_result_row "$out" "$seed" "$k" "result"
}

worker() {
    local gpu_id="$1"
    while :; do
        local job=""
        # Atomic pop from queue under a file lock.
        {
            flock 9
            if [[ -s "$QUEUE_FILE" ]]; then
                job="$(head -n1 "$QUEUE_FILE")"
                sed -i '1d' "$QUEUE_FILE"
            fi
        } 9>"$LOCK_FILE"
        [[ -z "$job" ]] && break
        local seed k
        read -r seed k <<< "$job"
        run_cell "$gpu_id" "$seed" "$k"
    done
}

pids=()
for ((w=0; w<NUM_WORKERS; w++)); do
    worker "$w" &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

# ---------------- Final tables ----------------
{
    echo
    echo "============================================================"
    echo "FINAL PER-CELL TABLE"
    printf "%-6s %-4s  %-16s %-16s %-8s\n" "seed" "k" "succ@50" "succ@100" "time"
    echo "------------------------------------------------------------"
} | tee -a "$RESULTS_TXT"

for SEED in "${START_SEEDS[@]}"; do
    for K in "${KS[@]}"; do
        OUT="data/eval/bon/${RUN_NAME}/replan${REPLAN_N}_k${K}_score${SCORE_N}_startseed${SEED}"
        python - "$OUT" "$SEED" "$K" "$RESULTS_TXT" <<'PY'
import json, sys
out, seed, k, rt = sys.argv[1:5]
try:
    with open(f"{out}/episodes.jsonl") as f:
        eps = [json.loads(l) for l in f if l.strip()]
except FileNotFoundError:
    line = f"{seed:>6} {k:>4}  MISSING"
    print(line); open(rt, "a").write(line + "\n"); sys.exit(0)
total_t = None
try:
    with open(f"{out}/eval_log.json") as f:
        total_t = json.load(f).get("total_time_s")
except FileNotFoundError:
    pass
s50  = sum(int(e.get("success", False)) for e in eps[:50])
s100 = sum(int(e.get("success", False)) for e in eps[:100])
t_str = f"{total_t:.0f}s" if total_t is not None else "?"
line = (f"{seed:>6} {k:>4}  "
        f"{s50:>3}/50={s50/50:.2f}    "
        f"{s100:>3}/100={s100/100:.2f}   {t_str}")
print(line); open(rt, "a").write(line + "\n")
PY
    done
done

{
    echo
    echo "============================================================"
    echo "AGGREGATED ACROSS SEEDS (mean over seeds)"
    printf "%-4s  %-14s  %-14s  %-7s\n" "k" "mean_succ@50" "mean_succ@100" "n_seeds"
    echo "------------------------------------------------------------"
} | tee -a "$RESULTS_TXT"

python - "$RUN_NAME" "$REPLAN_N" "$SCORE_N" "$RESULTS_TXT" "${START_SEEDS[*]}" "${KS[*]}" <<'PY'
import json, sys
run_name, replan_n, score_n, rt = sys.argv[1:5]
seeds = sys.argv[5].split()
ks    = sys.argv[6].split()
for k in ks:
    s50s, s100s = [], []
    for s in seeds:
        out = f"data/eval/bon/{run_name}/replan{replan_n}_k{k}_score{score_n}_startseed{s}"
        try:
            with open(f"{out}/episodes.jsonl") as f:
                eps = [json.loads(l) for l in f if l.strip()]
        except FileNotFoundError:
            continue
        if len(eps) >= 50:
            s50s.append(sum(int(e.get("success", False)) for e in eps[:50]) / 50)
        if len(eps) >= 100:
            s100s.append(sum(int(e.get("success", False)) for e in eps[:100]) / 100)
    if s50s:
        m50  = sum(s50s) / len(s50s)
        m100 = sum(s100s) / len(s100s) if s100s else float("nan")
        line = f"{k:>4}  {m50:>14.3f}  {m100:>14.3f}  {len(s50s):>7}"
    else:
        line = f"{k:>4}  {'MISSING':>14}  {'MISSING':>14}  {0:>7}"
    print(line); open(rt, "a").write(line + "\n")
PY

{
    echo "------------------------------------------------------------"
    echo "finished      : $(date)"
} | tee -a "$RESULTS_TXT"

echo
echo "[bon_eval_sweep] full results -> $RESULTS_TXT"
echo "[bon_eval_sweep] per-cell logs -> $WORK_DIR/"
