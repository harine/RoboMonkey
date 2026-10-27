"""Per-cell results + Gemini cost for the undertrained-BC Veritas BoN sweep.

Writes <ROOT>/results.csv (one row per task x arm x N, partial cells included) and prints it.
Paired columns compare each cell with the same task's N=1 baseline on shared seeds.
Run: python scriptsv2/eval_search/veritas_undertrained_csv.py
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'diffusion_policy'))
from diffusion_policy.veritas.plan_store import cost_usd  # noqa: E402

ROOT = Path('/gscratch/robotics/harine/RoboMonkey_data/experiments/veritas_undertrained_bon_20261006')
ARMS = ['n1', 'veritas', 'veritas_distance', 'sim_veritas', 'sim_veritas_distance']
CKPT = {'eggplant': 'eggplant_50_bc_h16 step 20k', 'stack': 'stack_50_bc_h16 step 30k'}


def episodes(cell):
    f = cell / 'episodes.jsonl'
    if not f.exists():
        return {}
    out = {}
    for line in f.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not e.get('error'):
            out[e['seed']] = bool(e['success'])
    return out


def cost(cell):
    total, calls = 0.0, 0
    for p in (cell / 'plans').glob('seed*.json') if (cell / 'plans').exists() else []:
        if p.name.endswith('_raw.json'):
            continue
        try:
            r = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        c = cost_usd(r.get('usage'), r.get('model'))
        if c is not None:
            total += c
            calls += 1
    return total, calls


rows = []
for task in ('eggplant', 'stack'):
    base = episodes(ROOT / 'eval' / task / 'n1' / 'n1')
    for arm in ARMS:
        for n in (1, 4, 16, 64):
            cell = ROOT / 'eval' / task / arm / f'n{n}'
            if not cell.exists():
                continue
            eps = episodes(cell)
            usd, calls = cost(cell)
            shared = [s for s in eps if s in base]
            rows.append(dict(
                task=task, checkpoint=CKPT[task], arm=arm, N=n, successes=sum(eps.values()),
                episodes=len(eps), success_rate=round(sum(eps.values()) / len(eps), 3) if eps else '',
                done=int(len(eps) >= 30),
                paired_n1_successes=sum(base[s] for s in shared),
                wins_vs_n1=sum(eps[s] and not base[s] for s in shared),
                losses_vs_n1=sum(base[s] and not eps[s] for s in shared),
                gemini_calls=calls, gemini_usd=round(usd, 4), path=str(cell)))

out = ROOT / 'results.csv'
with out.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(f"{r['task']:8s} {r['arm']:21s} N={r['N']:<3d} {r['successes']:>2d}/{r['episodes']:<2d}"
          f"  vs N1 {r['paired_n1_successes']:>2d} (+{r['wins_vs_n1']}/-{r['losses_vs_n1']})"
          f"  gemini {r['gemini_calls']} calls ${r['gemini_usd']:.3f}")
print('total gemini $%.3f' % sum(r['gemini_usd'] for r in rows), '->', out)
