#!/bin/bash
# Autofill §1 (RoboMonkey reproduction) DQC seeds 2/3 as the SLURM runs produce data.
# Re-runs the comparison generator whenever bon_sweep_dqc_c1_seed{2,3}/progress.json
# changes; exits once both seeds have the N=32 cell at >=50 episodes (or after the
# safety cap). Meant to be launched in the background.
set -u

PY=/gscratch/robotics/harine/miniconda3/envs/monkey-verifier/bin/python
GEN=/mmfs1/home/harine/RoboMonkey/scriptsv2/analysis/aggregate_verifier_results.py
BASE=/mmfs1/home/harine/RoboMonkey/data/eval/robomonkey_repro
S2="$BASE/bon_sweep_dqc_c1_seed2/progress.json"
S3="$BASE/bon_sweep_dqc_c1_seed3/progress.json"
LOG=/mmfs1/home/harine/RoboMonkey/data/eval/_verifier_comparison/_autofill_section1.log

INTERVAL=300     # seconds between polls
MAX_ITERS=72     # 72 * 300s = 6h safety cap

complete() {  # both seeds have N=32 cell with >=50 episodes?
  "$PY" - "$S2" "$S3" <<'PYEOF'
import json, sys
def done(p):
    try:
        return json.load(open(p)).get("32", {}).get("episodes", 0) >= 50
    except Exception:
        return False
sys.exit(0 if (done(sys.argv[1]) and done(sys.argv[2])) else 1)
PYEOF
}

echo "=== autofill §1 started $(date) (interval ${INTERVAL}s, cap ${MAX_ITERS}) ===" >> "$LOG"
prev=""
for i in $(seq 1 "$MAX_ITERS"); do
  cur="$(cat "$S2" "$S3" 2>/dev/null | md5sum | cut -d' ' -f1)"
  if [ "$cur" != "$prev" ]; then
    echo "--- $(date) iter $i: progress changed ($cur); regenerating ---" >> "$LOG"
    "$PY" "$GEN" >> "$LOG" 2>&1
    prev="$cur"
  fi
  if complete; then
    echo "=== $(date) iter $i: both seeds complete (>=50 ep @ N=32); final regen + exit ===" >> "$LOG"
    "$PY" "$GEN" >> "$LOG" 2>&1
    exit 0
  fi
  sleep "$INTERVAL"
done
echo "=== $(date) hit safety cap ($MAX_ITERS iters) without both seeds complete; exiting ===" >> "$LOG"
exit 0
