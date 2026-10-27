#!/bin/bash
#SBATCH --job-name=det_smoke
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mmfs1/home/harine/RoboMonkey/data/eval/_smoke_det/slurm-%j.out
#
# Determinism smoke test for the search-policy eval (single seeding path).
#   A1, A2  TILED_ENVS=1, softmax, obs as trained, SPLIT=test -> must be identical
#   B       TILED_ENVS=5                            -> matches A up to batch-shape float
#                                                      drift (reported, not asserted)
#   C       SPLIT=train                             -> different seeds, candidates differ
#   D1, D2  OBS_CORRUPTION=clean                    -> identical to each other
#   sbatch [--export=ALL,CHECKPOINT=<ckpt>] scriptsv2/eval_search/_smoke_determinism.sbatch
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/RoboMonkey}"
OUT=data/eval/_smoke_det
mkdir -p "$OUT"

CHECKPOINT="${CHECKPOINT:-/gscratch/robotics/harine/RoboMonkey_data/train/outer_inner/oi_eggplant_ie4/checkpoints/latest.ckpt}"
export DIFFUSION_POLICY_ROOT="$PWD/diffusion_policy"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35

run_one() {
    local out="$1"; shift
    rm -rf "$out"
    env "$@" CONDA_ENV=dqc_jax DEVICE=cuda:0 MODE=softmax SOFTMAX_TEMP=1.0 \
        SCORE_WINDOW=last N_SAMPLES=16 MAX_STEPS=40 \
        bash scriptsv2/eval_search/eval_search.sh "$CHECKPOINT" 5 "$out"
}

run_one "$OUT/A1" TILED_ENVS=1
run_one "$OUT/A2" TILED_ENVS=1
run_one "$OUT/B"  TILED_ENVS=5
run_one "$OUT/C"  TILED_ENVS=1 SPLIT=train
run_one "$OUT/D1" TILED_ENVS=1 OBS_CORRUPTION=clean
run_one "$OUT/D2" TILED_ENVS=1 OBS_CORRUPTION=clean

echo "=================== COMPARE ==================="
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate dqc_jax
python - "$OUT" <<'PY'
import json, sys
import numpy as np
root = sys.argv[1]
IGNORE = {"duration_s", "tiled_envs", "obs_corruption"}
def load(name):
    with open(f"{root}/{name}/episodes.jsonl") as f:
        return [{k: v for k, v in json.loads(l).items() if k not in IGNORE} for l in f]
def first_q(rows):
    return [np.asarray(r["q_values"][0], dtype=np.float64) for r in rows]
A1, A2, B, C, D1, D2 = (load(n) for n in ("A1", "A2", "B", "C", "D1", "D2"))
ok = True
def report(name, cond):
    global ok
    print(f"{'PASS' if cond else 'FAIL'}  {name}")
    ok &= bool(cond)
report("A1 == A2 (rerun, TILED_ENVS=1)", A1 == A2)
report("D1 == D2 (rerun, clean obs)", D1 == D2)
# The tiled path logs no per-replan q_values, so B is compared on outcomes; it matches A
# only up to batch-shape float drift, which can flip near-tied softmax picks.
qa, qc, qd = first_q(A1), first_q(C), first_q(D1)
dmin = [abs(a["obj_goal_dist_min"] - b["obj_goal_dist_min"]) for a, b in zip(A1, B)]
print(f"      A vs B (TILED 1 vs 5): max |d obj_goal_dist_min| = {max(dmin):.4f}; "
      f"successes A={[r['success'] for r in A1]} B={[r['success'] for r in B]}")
report("A vs C differ (train split = different seeds)",
       [r["seed"] for r in A1] != [r["seed"] for r in C]
       and any(not np.allclose(x, y) for x, y in zip(qa, qc)))
print(f"      A vs D (corrupted vs clean obs) first-replan differ: "
      f"{any(not np.allclose(x, y) for x, y in zip(qa, qd))}")
print("SMOKE RESULT:", "PASS" if ok else "FAIL")
PY
