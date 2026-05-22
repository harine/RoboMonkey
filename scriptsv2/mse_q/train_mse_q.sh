#!/bin/bash
# Train the mse_q calibrator: learn g(robomonkey_reward) -> MSEVerifier-scale
# score, so the RoboMonkey verifier can stand in for MSEVerifier when
# evaluating the MSE-trained search policy.
#
# Env split (no single conda env has both dependency sets):
#   * infer_server.py  -> `monkey-verifier` env (reward-model / llava stack)
#   * train_mse_q.py   -> `simpler_env`     env (diffusion_policy / zarr stack)
# They talk over HTTP. This wrapper auto-starts infer_server.py if it is not
# already running, and stops it again on exit.
#
# Usage
# -----
#   bash scriptsv2/mse_q/train_mse_q.sh                    # defaults
#   MAX_SAMPLES=2000 EPOCHS=50 bash scriptsv2/mse_q/train_mse_q.sh   # quick run
#   SERVER_URL=http://127.0.0.1:3100 bash scriptsv2/mse_q/train_mse_q.sh
#
# Env vars
# --------
#   OUTPUT_DIR     artifacts dir          (default data/mse_q/eggplant)
#   DATASET_DIR    offline dataset dir    (default ~/data/eggplant_in_basket)
#   SERVER_URL     verifier HTTP url      (default http://127.0.0.1:3100)
#   NOISE_LEVELS   sigma grid             (default "0.0 0.25 0.5 1.0 1.5 2.5")
#   MAX_SAMPLES    dataset windows cap    (default 20000)
#   EPOCHS         MLP training epochs    (default 300)
#   DEVICE         (default cuda:0)
#   TRAIN_ENV      env for train_mse_q.py    (default simpler_env)
#   VERIFIER_ENV   env for infer_server.py   (default monkey-verifier)

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-data/mse_q/eggplant}"
DATASET_DIR="${DATASET_DIR:-$HOME/data/eggplant_in_basket}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:3100}"
NOISE_LEVELS="${NOISE_LEVELS:-0.0 0.25 0.5 1.0 1.5 2.5}"
MAX_SAMPLES="${MAX_SAMPLES:-20000}"
EPOCHS="${EPOCHS:-300}"
DEVICE="${DEVICE:-cuda:0}"
TRAIN_ENV="${TRAIN_ENV:-simpler_env}"
VERIFIER_ENV="${VERIFIER_ENV:-monkey-verifier}"

full_path="$(realpath "$0")"
dir_path="$(dirname "$full_path")"
repo_root="$(cd "$dir_path/../.." && pwd)"   # .../RoboMonkey
cd "$repo_root"

mkdir -p "$OUTPUT_DIR"
export MONKEY_VERIFIER_SRC="${MONKEY_VERIFIER_SRC:-$repo_root/monkey-verifier/src}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"

# --- ensure the RoboMonkey verifier server is reachable ---------------------
STARTED_SERVER=0
SERVER_PID=""
SERVER_LOG="$(cd "$OUTPUT_DIR" && pwd)/infer_server.log"

cleanup() {
    if [[ "$STARTED_SERVER" == "1" && -n "$SERVER_PID" ]]; then
        echo "[mse_q] stopping infer_server.py (pid $SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ "$SERVER_URL" == "in_process" ]]; then
    echo "[mse_q] SERVER_URL=in_process — reward-model deps must be in $TRAIN_ENV"
else
    if curl -sf "$SERVER_URL/" >/dev/null 2>&1; then
        echo "[mse_q] verifier server already up at $SERVER_URL"
    else
        echo "[mse_q] starting infer_server.py in '$VERIFIER_ENV' (log: $SERVER_LOG)"
        (
            source "$HOME/miniconda3/etc/profile.d/conda.sh"
            conda activate "$VERIFIER_ENV"
            cd "$repo_root/monkey-verifier/src"
            exec python infer_server.py
        ) >"$SERVER_LOG" 2>&1 &
        SERVER_PID=$!
        STARTED_SERVER=1
        echo "[mse_q] infer_server.py pid=$SERVER_PID — waiting for health ..."
        for i in $(seq 1 90); do
            if curl -sf "$SERVER_URL/" >/dev/null 2>&1; then
                echo "[mse_q] verifier server ready after ${i}0s"
                break
            fi
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                echo "[mse_q] infer_server.py died on startup — see $SERVER_LOG"
                tail -n 30 "$SERVER_LOG" || true
                exit 1
            fi
            sleep 10
        done
        if ! curl -sf "$SERVER_URL/" >/dev/null 2>&1; then
            echo "[mse_q] verifier server did not become healthy — see $SERVER_LOG"
            exit 1
        fi
    fi
fi

# --- run the calibrator training -------------------------------------------
conda activate "$TRAIN_ENV"
export PYTHONPATH="$repo_root/diffusion_policy:$repo_root/monkey-verifier/src:${PYTHONPATH:-}"

echo "============================================================"
echo "  mse_q calibrator training"
echo "  trainer env  : $TRAIN_ENV"
echo "  dataset      : $DATASET_DIR"
echo "  verifier     : $SERVER_URL"
echo "  noise levels : $NOISE_LEVELS"
echo "  max samples  : $MAX_SAMPLES"
echo "  epochs       : $EPOCHS"
echo "  output dir   : $OUTPUT_DIR"
echo "============================================================"

python "$dir_path/train_mse_q.py" \
    --output-dir "$OUTPUT_DIR" \
    --dataset-dir "$DATASET_DIR" \
    --server-url "$SERVER_URL" \
    --noise-levels $NOISE_LEVELS \
    --max-samples "$MAX_SAMPLES" \
    --epochs "$EPOCHS" \
    --device "$DEVICE"
