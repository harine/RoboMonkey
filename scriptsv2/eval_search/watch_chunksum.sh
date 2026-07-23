#!/bin/bash
# Monitor the chunk-sum retrain + eval runs (trainings, policy evals, BC evals).
# One snapshot per call; wrap in `watch` for live updates:
#   watch -n 30 bash scriptsv2/eval_search/watch_chunksum.sh
set -uo pipefail
cd "$(cd "$(dirname "$(realpath "$0")")/../.." && pwd)"

echo "===== $(date '+%Y-%m-%d %H:%M:%S')  chunk-sum runs ====="
squeue -u "$USER" -o "%.10i %.12j %.9P %.7T %.7M %.4D %R" 2>/dev/null \
    | grep -E "JOBID|tr_|ev_|bc_|rmky" || echo "(no chunk-sum jobs in queue)"

echo
echo "----- TRAINING (step / checkpoint) -----"
for pol in noise0.0 noise0.4 noise0.8 tmrl; do
    f=$(grep -l "POLICY=$pol\b" data/train/chunksum/slurm-*.out 2>/dev/null | xargs -r ls -t 2>/dev/null | head -1)
    ck="diffusion_policy/data/train/chunksum/$pol/checkpoints/latest.ckpt"
    prog="<no log>"
    if [[ -n "${f:-}" ]]; then
        prog=$(grep -oE "Training epoch [0-9]+:[^]]*[0-9]+/[0-9]+|global_step[^0-9]*[0-9]+" "$f" 2>/dev/null | tail -1)
        [[ -z "$prog" ]] && prog=$(grep -iE "Traceback|CUDA error|uncorrectable ECC|InstantiationException|Error in call to target|RuntimeError" "$f" 2>/dev/null | tail -1)
        [[ -z "$prog" ]] && { grep -q "verifier ready\|Loaded model" "$f" 2>/dev/null && prog="<verifier loaded, starting>" || prog="<loading verifier>"; }
    fi
    ckst="no"; [[ -f "$ck" ]] && ckst="YES ($(date -r "$ck" '+%H:%M' 2>/dev/null))"
    printf "  %-9s ckpt:%-14s %s\n" "$pol" "$ckst" "$prog"
done

echo
echo "----- EVAL cells done -----"
for pol in noise0.0 noise0.4 noise0.8 tmrl; do
    n=$(find "data/eval/search_noise_sweep/$pol" -name eval_log.json 2>/dev/null | wc -l)
    tot=11; [[ "$pol" == tmrl ]] && tot=33
    printf "  %-9s %2s/%-2s cells\n" "$pol" "$n" "$tot"
done
for s in ddim ddpm; do
    n=$(find data/eval/search_bc_sweep/bc -name eval_log.json -path "*argmax_${s}_*" 2>/dev/null | wc -l)
    printf "  bc-%-6s %2s/11 cells\n" "$s" "$n"
done

echo
echo "----- BC BoN  success_rate vs N (argmax, chunk scoring) -----"
BC_N_LIST="${BC_N_LIST:-1 2 4 8 16 32 64 128 256 512 1024}"
for s in ddim ddpm; do
    row="  bc-$(printf '%-4s' "$s")"
    for nn in $BC_N_LIST; do
        # cells are saved as bc/argmax_<sampler>_n<N>_seed<seed>_ep<E>/eval_log.json
        f=$(ls -d data/eval/search_bc_sweep/bc/argmax_${s}_n${nn}_seed*_ep*/eval_log.json 2>/dev/null | head -1)
        if [[ -n "${f:-}" && -f "$f" ]]; then
            sr=$(grep -oE '"success_rate"[^,]*' "$f" 2>/dev/null | grep -oE '[0-9.]+' | head -1)
            [[ -z "$sr" ]] && sr="?"
        else
            sr="-"
        fi
        row+=$(printf " %5s" "$sr")
    done
    echo "$row"
done
printf "  %-7s" "N="; for nn in $BC_N_LIST; do printf " %5s" "$nn"; done; echo

echo
echo "----- RoboMonkey BoN  (VLA policy + in-process verifier) success_rate / episodes vs N -----"
RM_PROG="data/eval/robomonkey_repro/bon_sweep/progress.json"
RM_N_LIST="${RM_N_LIST:-1 2 4 8 16 32 64 128 256 512 1024}"
if [[ -f "$RM_PROG" ]]; then
    PYBIN="$(command -v python3 || command -v python || echo /gscratch/robotics/harine/miniconda3/envs/monkey-verifier/bin/python)"
    "$PYBIN" - "$RM_PROG" "$RM_N_LIST" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
ns = sys.argv[2].split()
sr = ["  rmky"]; ep = ["  eps "]
for n in ns:
    c = d.get(n) or {}
    e = c.get("episodes", 0); s = c.get("success", 0)
    sr.append(f"{(100*s/e):5.0f}" if e else f"{'-':>5}")
    ep.append(f"{e:5d}")
print("".join(x if i == 0 else " " + x for i, x in enumerate(sr)))
print("".join(x if i == 0 else " " + x for i, x in enumerate(ep)))
PY
else
    echo "  (no progress.json yet at $RM_PROG)"
fi
printf "  %-7s" "N="; for nn in $RM_N_LIST; do printf " %5s" "$nn"; done; echo

echo
echo "----- EVAL jobs (state / waits-on) -----"
squeue -u "$USER" -h -o "%i|%j|%P|%T|%E|%R" 2>/dev/null | grep -E "ev_|bc_|rmky" \
| while IFS='|' read -r id name part state dep reason; do
    dep="${dep#afterok:}"; dep="${dep#afterany:}"; dep="${dep%%(*}"
    waits=""; [[ -n "$dep" && "$dep" != "(null)" ]] && waits="waits:$dep"
    printf "  %-11s %-9s %-8s %-14s %s\n" "$name" "$part" "$state" "$waits" "$reason"
done
[[ -z "$(squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -E 'ev_|bc_|rmky')" ]] && echo "  (none in queue)"

echo
echo "----- recent errors (if any) -----"
grep -ilE "Traceback|CUDA error|ECC|InstantiationException" \
    data/train/chunksum/slurm-*.out \
    data/eval/search_noise_sweep/_summaries/slurm-*.out \
    data/eval/search_bc_sweep/_summaries/slurm-*.out \
    data/eval/robomonkey_repro/slurm-*.out 2>/dev/null | sed 's/^/  ERR: /' | tail -6 || true
