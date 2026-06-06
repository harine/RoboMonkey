#!/bin/bash
#SBATCH --job-name=det_smoke
#SBATCH --account=socialrl
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:40:00
#SBATCH --output=data/eval/_smoke/slurm-%j.out
#SBATCH --error=data/eval/_smoke/slurm-%j.out
#
# Reproducibility smoke test for the deterministic candidate-sampling change.
# Runs eval_search twice in MODE=argmax (deterministic selection) with the new
# default seeded sampling. If seeding works end-to-end the two runs must be
# bit-identical (same success, same per-replan q-values, same selected idx).
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
mkdir -p data/eval/_smoke

export DIFFUSION_POLICY_ROOT="$SLURM_SUBMIT_DIR/diffusion_policy"
export MONKEY_VERIFIER_SRC="$SLURM_SUBMIT_DIR/monkey-verifier/src"
export MODEL_DIR="$SLURM_SUBMIT_DIR/monkey-verifier/model_dir"
export ROBOMONKEY_KV_PREFIX=0

CKPT="$SLURM_SUBMIT_DIR/diffusion_policy/data/outputs/2026.06.03/18.32.58_robomonkey_eggplant_search_tunable_corruption_noise0.4_robomonkey_eggplant_state/checkpoints/latest.ckpt"

run_one() {
    local out="$1"
    rm -rf "$out"
    env DEVICE="cuda:0" CONDA_ENV="monkey-verifier" \
        MODE="argmax" N_SAMPLES="2" START_SEED="1000" \
        MAX_STEPS="15" USE_EMA=1 \
        bash scriptsv2/eval_search/eval_search.sh "$CKPT" 2 "$out"
}

run_one data/eval/_smoke/run_a
run_one data/eval/_smoke/run_b

echo "=================== DIFF ==================="
python - data/eval/_smoke/run_a data/eval/_smoke/run_b <<'PY'
import json, sys
a_dir, b_dir = sys.argv[1], sys.argv[2]
def load(d):
    rows = []
    with open(f"{d}/episodes.jsonl") as f:
        for line in f:
            r = json.loads(line)
            r.pop("duration_s", None)  # timing varies; ignore
            rows.append(r)
    return rows
a, b = load(a_dir), load(b_dir)
ok = (a == b)
print(f"episodes match (ignoring duration): {ok}")
if not ok:
    for i, (ra, rb) in enumerate(zip(a, b)):
        if ra != rb:
            print(f"  --- mismatch ep {i} ---")
            print(f"  A success={ra.get('success')} q0={ (ra.get('q_values') or [[None]])[0][:4] }")
            print(f"  B success={rb.get('success')} q0={ (rb.get('q_values') or [[None]])[0][:4] }")
print("SMOKE RESULT:", "PASS" if ok else "FAIL")
PY
