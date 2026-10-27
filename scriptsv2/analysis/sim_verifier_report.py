"""Aggregate matched simulator evaluations; success means environment benchmark success.
Filter distances, contact gaps, and reopening counts are diagnostics only.
"""
import json,csv,os
from pathlib import Path
ROOT=Path('/gscratch/robotics/harine/RoboMonkey_data/experiments/metric_verifier_pilot_20261004')
rows=[]
REVISIONS=['subtask_two_full_trial','pixel_sim_distance_trial','pixel_sim_full_trial']
for revision in REVISIONS:
 for n_bc in [16]:
  p=ROOT/f'eval_{revision}_30ep/eggplant'/f'n{n_bc}'
  f=p/'episodes.jsonl'
  if not f.exists():continue
  for line in f.read_text().splitlines():
   try:e=json.loads(line)
   except json.JSONDecodeError:continue
   seed=e['seed'];o=p/f'seed{seed}_full_filter_outcome.json'
   if not o.exists() or e.get('error'):continue
   d=json.loads(o.read_text());r=d['filter']
   rows.append(dict(verifier=revision,n_bc=n_bc,n_candidates=3*n_bc,seed=seed,benchmark_success=int(e['success']),final_distance_m=r['final_distance'],backtrack_m=r['backtrack_m'],carry_max_m=r['carry_max_m'],contact_gap=r['max_contact_gap'],reopen_runs=r['reopen_runs'],max_replay_position_variation_m=max(d['restoration_errors'],default=0),max_replay_score_variation=max(d.get('score_variation',[]),default=0)))
summary={}
for revision in REVISIONS:
 summary[revision]={}
 for n_bc in [16]:
  selected=[r for r in rows if r['verifier']==revision and r['n_bc']==n_bc]
  n=len(selected)
  summary[revision][str(n_bc)]=dict(n_bc=n_bc,n_candidates=3*n_bc,completed=n,requested=30,benchmark_successes=sum(r['benchmark_success'] for r in selected),success_rate=sum(r['benchmark_success'] for r in selected)/n if n else None,mean_final_distance_m=sum(r['final_distance_m'] for r in selected)/n if n else None)
baseline_path=ROOT/'eval/eggplant/n1/episodes.jsonl'
if baseline_path.exists():
 baseline={x['seed']:bool(x['success']) for x in map(json.loads,baseline_path.read_text().splitlines()) if not x.get('error')}
 for revision in REVISIONS:
  paired=[x for x in rows if x['verifier']==revision and x['seed'] in baseline]
  summary[revision]['16']['matched_bc_n1_comparison']=dict(episodes=len(paired),
    baseline_successes=sum(baseline[x['seed']] for x in paired),
    wins=sum(bool(x['benchmark_success']) and not baseline[x['seed']] for x in paired),
    losses=sum(not bool(x['benchmark_success']) and baseline[x['seed']] for x in paired))
report=ROOT/'sim_lookahead_30ep_report.json';tmp=report.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(summary,indent=2));tmp.replace(report)
if rows:
 path=ROOT/'sim_lookahead_30ep_episodes.csv';tmp=path.with_suffix(f'.{os.getpid()}.tmp')
 with tmp.open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 tmp.replace(path)
print(json.dumps(summary))
