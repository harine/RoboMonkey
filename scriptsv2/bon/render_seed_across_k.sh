#!/bin/bash
# Render BoN branching MP4s for one (start_seed, replan) cell across a set of
# k values and episode indices, with a flat per-run folder layout:
#
#   data/eval/bon/_summaries/bon_viz/<YYYYMMDD_HHMMSS>/seed<S>_replan<R>/
#       k2_ep000.mp4
#       k2_ep001.mp4
#       k4_ep000.mp4
#       k4_ep001.mp4
#       ...
#
# Usage:
#   bash scriptsv2/bon/render_seed_across_k.sh [start_seed] [k_list] [n_eps] [replan] [run_name]
#
# Examples:
#   # default: start_seed=17, replan=4, k=2..32, first 5 episodes
#   bash scriptsv2/bon/render_seed_across_k.sh
#
#   # one episode per k:
#   bash scriptsv2/bon/render_seed_across_k.sh 17 "2 4 8 16 32" 1
#
#   # different start seed cell, episodes 0..9, only k=8 and 16:
#   bash scriptsv2/bon/render_seed_across_k.sh 38 "8 16" 10
#
# Env vars:
#   FPS         : 2     (mp4 frames-per-second)
#   CONDA_ENV   : simpler_env
#   TASK        : widowx_put_eggplant_in_basket
#   FIXED       : 0     (set 1 to read from *_startseed<S>_fixed/ cells, i.e.
#                        sweeps run with FIX_SEED=1 — every episode same seed)

set -euo pipefail

START_SEED="${1:-17}"
KS="${2:-2 4 8 16 32}"
N_EPS="${3:-5}"
REPLAN="${4:-4}"
RUN_NAME="${5:-16.45.01_train_diffusion_unet_eggplant_in_basket_lowdim_eggplant_in_basket_lowdim}"
FPS="${FPS:-2}"
CONDA_ENV="${CONDA_ENV:-simpler_env}"
TASK="${TASK:-widowx_put_eggplant_in_basket}"
FIXED="${FIXED:-0}"
FIX_TAG=""
[[ "$FIXED" == "1" ]] && FIX_TAG="_fixed"
# Read from top3-weighted cells (sweeps run with BON_SELECT=top3_weighted).
TOP3W="${TOP3W:-0}"
SEL_TAG=""
[[ "$TOP3W" == "1" ]] && SEL_TAG="_top3w"

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo_root"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
export DISPLAY=""

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="data/eval/bon/_summaries/bon_viz/${TS}/seed${START_SEED}_replan${REPLAN}${FIX_TAG}${SEL_TAG}"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "  render_seed_across_k"
echo "  start_seed : $START_SEED"
echo "  replan     : $REPLAN"
echo "  ks         : $KS"
echo "  n_eps      : $N_EPS  (ep000..ep$(printf '%03d' $((N_EPS - 1))))"
echo "  run_name   : $RUN_NAME"
echo "  fps        : $FPS"
echo "  out_dir    : $OUT_DIR"
echo "============================================================"

for K in $KS; do
    cell_dir="data/eval/bon/${RUN_NAME}/replan${REPLAN}_k${K}_score4_startseed${START_SEED}${FIX_TAG}${SEL_TAG}"
    bon_q_dir="${cell_dir}/bon_q"
    if [[ ! -d "$bon_q_dir" ]]; then
        echo "[skip] k=$K  no $bon_q_dir"
        continue
    fi

    # Collect the first N_EPS raw npz files for this k (sorted by ep index).
    mapfile -t raw_files < <(
        ls "${bon_q_dir}"/ep*_seed*.npz 2>/dev/null | grep -v "_aug.npz" | sort | head -n "$N_EPS"
    )
    if (( ${#raw_files[@]} == 0 )); then
        echo "[skip] k=$K  no ep*.npz in $bon_q_dir"
        continue
    fi

    # Augment any that don't already have a sibling _aug.npz.
    to_augment=()
    for raw in "${raw_files[@]}"; do
        aug="${raw%.npz}_aug.npz"
        [[ -f "$aug" ]] || to_augment+=("$raw")
    done
    if (( ${#to_augment[@]} > 0 )); then
        echo "[aug ] k=$K  augmenting ${#to_augment[@]} files"
        python scriptsv2/viz_bon/augment_bon_q.py "${to_augment[@]}" --task "$TASK" >/dev/null
    fi

    # Render each ep -> one MP4. viz_action_branches.py writes to
    # <out_dir>/<stem>/<stem>.mp4; we stage under a temp dir per K then move
    # to k<K>_ep<NNN>.mp4 in the flat seed_replan folder.
    tmp_out="${OUT_DIR}/.tmp_k${K}"
    mkdir -p "$tmp_out"
    aug_files=()
    for raw in "${raw_files[@]}"; do
        aug_files+=("${raw%.npz}_aug.npz")
    done
    python scriptsv2/viz_bon/viz_action_branches.py "${aug_files[@]}" \
        --out-dir "$tmp_out" --video --fps "$FPS"

    for raw in "${raw_files[@]}"; do
        base="$(basename "${raw%.npz}")"          # ep000_seed17
        ep_idx="${base%%_*}"                      # ep000
        meta="$(python - "${raw%.npz}_aug.npz" <<'PY'
import sys, numpy as np
with np.load(sys.argv[1], allow_pickle=False) as z:
    s = bool(int(z["success"])); tr = bool(int(z["truncated"]))
    sel = str(z["bon_select"]) if "bon_select" in z.files else "argmax"
status = "success" if s else ("truncated" if tr else "fail")
sel_tag = "argmax" if sel == "argmax" else "top3w"
print(f"{status} {sel_tag}")
PY
)"
        status="${meta%% *}"
        sel_tag="${meta##* }"
        mv "${tmp_out}/${base}/${base}.mp4" \
           "${OUT_DIR}/k${K}_${ep_idx}_${sel_tag}_${status}.mp4"
    done
    rm -rf "$tmp_out"
    echo "[ok  ] k=$K  -> ${#raw_files[@]} mp4s"
done

echo
echo "[render_seed_across_k] done -> $OUT_DIR"
ls -la "$OUT_DIR" | head -50
