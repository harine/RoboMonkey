"""Paired pilot gate and candidate diagnostics; never starts training on missing data."""
import csv,json,math,os,sys
from collections import Counter
from pathlib import Path
ROOT=Path('/gscratch/robotics/harine/RoboMonkey_data/experiments/metric_verifier_pilot_20261004')

def report(revision='v1'):
    pixel=revision.startswith('pixel_w')
    name='pilot_pixel_gripper.json' if pixel else ('pilot.json' if revision=='v1' else 'pilot_v2.json')
    manifest=json.loads((ROOT/name).read_text());out={};rows=[]
    group='eval' if revision=='v1' else 'eval_'+revision
    output=ROOT if revision=='v1' else ROOT/group
    for task in ('stack','eggplant'):
        episodes={};results={};diagnostics={}
        for n in (1,4,16):
            d=ROOT/group/task/f'n{n}';p=d/'episodes.jsonl'
            rec=[json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
            eps={r['seed']:r for r in rec};episodes[n]=eps
            complete=set(eps)==set(manifest['specs'][task]['seeds']) and not any(r.get('error') for r in eps.values())
            wins=sum(bool(r['success']) for r in eps.values())
            results[str(n)]=dict(completed=len(eps),successes=wins,rate=wins/len(eps) if eps else None,complete=complete)
            if complete:rows.append(dict(task=task,n=n,episodes=30,successes=wins,success_rate=wins/30,
                checkpoint=manifest['specs'][task]['checkpoint'],verifier='unused_N1' if n==1 else manifest['verifier'],
                **({'gripper_weight':float(revision.split('_w')[1].replace('p','.'))} if pixel else {})))
            phases=Counter();selected=Counter();grip_override=0;count=0;no_grasp=0
            for p in d.glob('seed*.jsonl'):
                traces=[json.loads(x) for x in p.read_text().splitlines()]
                no_grasp+=bool(traces) and not any(x['metrics']['held'] for x in traces)
                for x in traces:
                    phases[x['metrics']['phase']]+=1;selected[str(x['selected_index'])]+=1
                    c=x['components'];j=x['selected_index']
                    geom=c.get('geometry',[-v for v in c.get('pixel_cost',[])])
                    best=max(range(len(geom)),key=lambda i:geom[i])
                    grip_override+=j!=best and c['gripper_cost'][j]<c['gripper_cost'][best];count+=1
            diagnostics[str(n)]=dict(replans=count,phase_counts=dict(phases),selected_indices=dict(selected),
                never_held_episodes=None if pixel else no_grasp,gripper_penalty_overrode_geometric_best=grip_override)
        gate=False;paired=None
        if all(v['complete'] for v in results.values()):
            wins=sum(episodes[16][s]['success'] and not episodes[1][s]['success'] for s in episodes[1])
            losses=sum(episodes[1][s]['success'] and not episodes[16][s]['success'] for s in episodes[1])
            k=wins+losses;p=sum(math.comb(k,i) for i in range(wins,k+1))/2**k if k else 1.
            gate=wins-losses>=3 and results['4']['successes']>=results['1']['successes'] and p<=.10
            paired=dict(gains=wins,losses=losses,net=wins-losses,one_sided_exact_p=p)
        out[task]=dict(results=results,paired=paired,passes_training_gate=gate,diagnostics=diagnostics)
    ledger=ROOT/'gemini_cost.json';cost=json.loads(ledger.read_text()) if ledger.exists() else {'requests':[]}
    out['gemini']=dict(requests=len(cost['requests']),charged_or_reserved_usd=sum(r['charged_or_reserved_usd'] for r in cost['requests']),cap_usd=100)
    out['both_tasks_pass']=all(out[t]['passes_training_gate'] for t in ('stack','eggplant'))
    out['revision']=revision
    if pixel:out['gripper_weight']=float(revision.split('_w')[1].replace('p','.'))
    tmp=output/f'report.{os.getpid()}.tmp';tmp.write_text(json.dumps(out,indent=2));tmp.replace(output/'report.json')
    if rows:
        tmp=output/f'pilot_results.{os.getpid()}.tmp'
        with tmp.open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        tmp.replace(output/'pilot_results.csv')
    print(json.dumps(out,indent=2))
if __name__=='__main__':report(sys.argv[1] if len(sys.argv)>1 else 'v1')
