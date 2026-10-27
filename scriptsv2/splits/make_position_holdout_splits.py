"""Deterministic joint source/target position holdouts; nested 50/100 train, 20 val, 50 test.
Uses only position/provenance metadata, never policy results. Existing splits stay immutable.
"""
import collections,hashlib,itertools,json,random
from pathlib import Path
SPLITS=Path(__file__).resolve().parents[2]/'diffusion_policy/diffusion_policy/config/splits/simpler'
OUT=Path('/gscratch/robotics/harine/RoboMonkey_data/experiments/position_holdout_20261009/splits')
def cell(r,k):return tuple(r[k])
def balanced(ids,rows,count,reserve_coverage=False):
    rng=random.Random(42);ids=list(ids);rng.shuffle(ids);out=[];sc=collections.Counter();gc=collections.Counter()
    totals={k:collections.Counter(cell(rows[i],k) for i in ids) for k in ('start_cell','goal_cell')}
    for _ in range(count):
        eligible=[i for i in ids if not reserve_coverage or all(totals[k][cell(rows[i],k)]>1 for k in totals)]
        if not eligible:raise ValueError('Insufficient validation episodes while preserving training coverage')
        best=min(eligible,key=lambda i:(sc[cell(rows[i],'start_cell')]+gc[cell(rows[i],'goal_cell')],max(sc[cell(rows[i],'start_cell')],gc[cell(rows[i],'goal_cell')])))
        ids.remove(best);out.append(best);sc[cell(rows[best],'start_cell')]+=1;gc[cell(rows[best],'goal_cell')]+=1
        for k in totals:totals[k][cell(rows[best],k)]-=1
    return out

def build(task):
    base=json.loads((SPLITS/task/('random_train50_val70.json' if task=='stack' else 'random_train100.json')).read_text());rows={int(i):r for i,r in base['episodes'].items() if r['seed_gate']};sources=sorted({cell(r,'start_cell') for r in rows.values()});targets=sorted({cell(r,'goal_cell') for r in rows.values()})
    candidates=[]
    for n in (1,2):
      for hs in itertools.combinations(sources,n):
       for ht in ([(t,) for t in targets] if len(targets)>1 else [()]):
        pool=[i for i,r in rows.items() if cell(r,'start_cell') not in hs and cell(r,'goal_cell') not in ht]
        test=[i for i,r in rows.items() if i not in pool and r['seed_gate']]
        if len(pool)<120 or len(test)<50:continue
        if any(sum(cell(rows[i],k)==c for i in test)<10 for k,held in [('start_cell',hs),('goal_cell',ht)] for c in held):continue
        if {cell(rows[i],'start_cell') for i in pool}!=set(sources)-set(hs) or {cell(rows[i],'goal_cell') for i in pool}!=set(targets)-set(ht):continue
        counts=collections.Counter((k,cell(rows[i],k)) for i in pool for k in ('start_cell','goal_cell'))
        # Prefer one held-out source cell; largest weakest-cell coverage, then more training data.
        candidates.append(((n,-min(counts.values()),-len(pool),hs,ht),hs,ht,pool,test))
    if not candidates:raise ValueError(f'{task}: no feasible split')
    _,hs,ht,pool,testpool=min(candidates,key=lambda x:x[0]);val=balanced(pool,rows,20,reserve_coverage=True);remaining=set(pool)-set(val)
    # Explicit set cover ensures even rare non-held-out positions occur in train50.
    selected=[];missing={('start_cell',s) for s in sources if s not in hs}|{('goal_cell',t) for t in targets if t not in ht}
    while missing:
        i=max(sorted(remaining),key=lambda i:sum((k,cell(rows[i],k)) in missing for k in ('start_cell','goal_cell')))
        selected.append(i);remaining.remove(i);missing-={(k,cell(rows[i],k)) for k in ('start_cell','goal_cell')}
    selected+=balanced(remaining,rows,100-len(selected));test=[]
    for k,held in [('start_cell',hs),('goal_cell',ht)]:
        for c in held:
            need=max(0,10-sum(cell(rows[i],k)==c for i in test))
            test+=balanced([i for i in testpool if i not in test and cell(rows[i],k)==c],rows,need)
    test+=balanced([i for i in testpool if i not in test],rows,50-len(test))
    for n in (50,100):
        m=dict(base,train=sorted(selected[:n]),val=sorted(val),test=sorted(test));m['checksum']=hashlib.md5(json.dumps({k:m[k] for k in ('train','val','test')},sort_keys=True).encode()).hexdigest()
        m['derivation']=dict(rule='position_holdout',seed=42,held_source_positions=hs,held_target_positions=ht,n_train=n,n_val=20,n_test=50,nested=True,selection='seed-gate-passing episodes only; position metadata only; joint source/target coverage',target_limitation='single fixed basket position; source-only holdout' if len(targets)==1 else None)
        assert not(set(m['train'])&set(val) or set(m['train'])&set(test) or set(val)&set(test))
        for k,held in [('start_cell',hs),('goal_cell',ht)]:
            assert {cell(rows[i],k) for i in m['train']}=={cell(r,k) for r in rows.values()}-set(held)
            assert not {cell(rows[i],k) for i in m['train']+val}&set(held)
        assert all(rows[i]['seed_gate'] for i in test)
        m['position_counts']={part:{k:{str(c):v for c,v in collections.Counter(cell(rows[i],k) for i in m[part]).items()} for k in ('start_cell','goal_cell')} for part in ('train','val','test')}
        out=OUT/task;out.mkdir(parents=True,exist_ok=True);p=out/f'position_holdout_train{n}.json';p.write_text(json.dumps(m,indent=1));print(p,'held source',hs,'target',ht,'train/val/test',n,20,50)
if __name__=='__main__':
 for t in ('stack','eggplant'):build(t)
