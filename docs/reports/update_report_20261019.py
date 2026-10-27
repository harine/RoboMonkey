"""Rebuild concise Simpler report from completed, valid campaign results."""
import csv,json,dataclasses,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
import wandb_workspaces.reports.v2 as wr
ROOT=Path('/gscratch/robotics/harine/RoboMonkey_data/experiments')
OUT=ROOT/'report_revision_20261019';OUT.mkdir(exist_ok=True)
DOC=Path('/mmfs1/home/harine/RoboMonkey/docs/reports')
SPLITS=Path('/mmfs1/home/harine/RoboMonkey/diffusion_policy/diffusion_policy/config/splits/simpler')
URL=json.loads((DOC/'simpler_verifier_wandb_media_manifest.json').read_text())['report_url']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
COLORS=['#000000','#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9']
NS=[1,2,4,8,16,32,64]
assets=[]
def save(fig,key,title):
 p=OUT/(key.replace('/','_')+'.png');fig.savefig(p,dpi=170);fig.savefig(p.with_suffix('.pdf'));plt.close(fig);assets.append(dict(key=key,path=str(p),title=title));return key

def campaign(task):
 name='stack_veritas_20261006' if task=='stack' else 'eggplant_veritas_resume_20261008'
 return [r for r in csv.DictReader((ROOT/name/'results.csv').open()) if not r['alias_of']]
rows={t:campaign(t) for t in ('eggplant','stack')}
# BC checkpoint matrix: identical datasets/test seeds, including archived valid sweeps.
for task,root,run_name in [('eggplant','veritas_h16_20261002','eggplant_100_bc_h16_s8_e8_seed42'),('stack','stack_veritas_20261006','stack_random50_bc')]:
 for p in sorted((ROOT/root/'eval'/run_name).glob('step_*/argmax/**/eval_log.json')):
  step=int(p.relative_to(ROOT/root/'eval'/run_name).parts[0].split('_')[1]);n=int(p.parent.name[1:]);d=json.loads(p.read_text())
  if step not in (5000,10000,20000,30000):continue
  if any(r['model']=='bc' and r['split']==('random100' if task=='eggplant' else 'random50') and int(r['checkpoint_step'])==step and int(r['N'])==n for r in rows[task]):continue
  rows[task].append(dict(model='bc',split='random100' if task=='eggplant' else 'random50',checkpoint_step=str(step),observations='clean',selection='argmax',N=str(n),successes=str(d['num_successes']),episodes=str(d['num_episodes']),success_rate=str(d['success_rate'])))
selected=[]
# Fixed named checkpoints; never choose a different checkpoint for each N.
configs={
 'eggplant':[('bc',20000,'clean','BC 20k',0),('uniform',30000,'clean','ST uniform 30k',1),('flat10',10000,'clean','ST flat10 10k · clean',2),('flat10',10000,'conditional','ST flat10 10k · trained noise',2),('ramp2to10',10000,'clean','ST ramp2→10 10k · clean',3),('ramp2to10',10000,'conditional','ST ramp2→10 10k · trained noise',3)],
 'stack':[('bc',20000,'clean','BC 20k',0),('uniform',20000,'clean','ST uniform 20k',1),('flat100',10000,'clean','ST flat100 10k · clean',2),('flat100',10000,'conditional','ST flat100 10k · trained noise',2),('ramp25to100',10000,'clean','ST ramp25→100 10k · clean',3),('ramp25to100',10000,'conditional','ST ramp25→100 10k · trained noise',3)]}
plot_keys={}
for task in rows:
 for mode in ('argmax','final'):
  fig,ax=plt.subplots(figsize=(6.2,5.1));fig.subplots_adjust(bottom=.32,top=.9,left=.12,right=.97)
  for kind,step,obs,label,col in configs[task]:
   eligible=[r for r in rows[task] if r['split']==('random50' if task=='stack' else 'random100') and r['model']==kind and r['observations']==obs and (r['N']=='1' if kind=='bc' else r['selection']==mode and int(r['N'])>1)]
   if eligible:
    best=max(eligible,key=lambda r:(float(r['success_rate']),int(r['checkpoint_step'])==20000,-int(r['checkpoint_step'])))
    old_step=step;step=int(best['checkpoint_step']);label=label.replace(f'{old_step//1000}k',f'{step//1000}k')
   selected.append(dict(task=task,model=kind,checkpoint=step,observations=obs,selection=mode))
   vals={int(r['N']):100*float(r['success_rate']) for r in rows[task] if r['split']==('random50' if task=='stack' else 'random100') and r['model']==kind and int(r['checkpoint_step'])==step and r['observations']==obs and r['selection']==('argmax' if kind=='bc' else mode)}
   if mode=='final' and kind!='bc':
    n1=[r for r in rows[task] if r['split']==('random50' if task=='stack' else 'random100') and r['model']==kind and int(r['checkpoint_step'])==step and r['observations']==obs and r['selection']=='argmax' and r['N']=='1']
    if n1:vals[1]=100*float(n1[0]['success_rate'])
   if not vals:continue
   xs=sorted(vals);yy=[vals[n] for n in xs]
   selected[-1]['completed_N']=xs
   ax.plot(xs,yy,color=COLORS[col],marker='s' if obs=='conditional' else 'o',ls='--' if obs=='conditional' else '-',lw=1.7,ms=5,label=label)
  ax.set_xscale('log',base=2);ax.set_xticks(NS,labels=NS);ax.set_ylim(0,100);ax.set_ylabel('Benchmark success (%)');ax.set_xlabel('Candidates N');ax.grid(alpha=.2)
  ax.set_title(f'{task.title()} · {mode}\n'+('100 demos' if task=='eggplant' else 'random50 split')+' · 50 episodes',fontsize=11)
  ax.legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=2,fontsize=7.7,frameon=False,columnspacing=1)
  plot_keys[(task,mode)]=save(fig,f'plots/{task}_{mode}',f'{task.title()} · {mode}')
# Position plots: exact manifest positions, no artificial shifts or random jitter.
positions=[]
for task,filename,title in [('eggplant','random_train100.json','Eggplant · random100'),('eggplant','quadrant_train100.json','Eggplant · quadrant100'),('stack','random_train50_val70.json','Stack · random50'),('stack','quadrant_train50.json','Stack · quadrant50')]:
 d=json.loads((SPLITS/task/filename).read_text());fig,axes=plt.subplots(1,2,figsize=(6.2,4.3));fig.subplots_adjust(wspace=.4,bottom=.18,top=.79)
 for ax,key,entity in zip(axes,['start_xy','goal_xy'],['Source object','Target object']):
  allxy=np.array([e[key] for e in d['episodes'].values()]);
  for split,color,marker in [('train','#0072B2','o'),('test','#D55E00','x')]:
   xy=np.array([d['episodes'][str(i)][key] for i in d[split]])
   ax.scatter(xy[:,0],xy[:,1],c=color,marker=marker,s=25,alpha=.7,label=f'{split} ({len(xy)})')
  span=max(np.ptp(allxy[:,0]),np.ptp(allxy[:,1]),.04);center=np.mean([allxy.min(axis=0),allxy.max(axis=0)],axis=0)
  ax.set_xlim(center[0]-.6*span,center[0]+.6*span);ax.set_ylim(center[1]-.6*span,center[1]+.6*span)
  ax.set_aspect('equal');ax.set_xlabel('World x (m)');ax.set_ylabel('World y (m)');ax.set_title(entity,fontsize=10);ax.grid(alpha=.2)
  if task=='stack' and 'quadrant' in filename and key=='start_xy':ax.axhline(0,color='#666666',ls=':',lw=1)
 axes[0].legend(fontsize=8);fig.suptitle(title,fontsize=11)
 positions.append(save(fig,'positions/'+task+'_'+filename.split('.')[0],title))
# Actual Veritas waypoints over the matching scene; no standalone query-image panels.
import matplotlib.patheffects as pe
queries=json.loads((DOC/'simpler_verifier_gemini_query_images_2026-10-05.json').read_text());trajectories=[]
for task in ('eggplant','stack'):
 x=next(x for x in queries if x['task']==task and x['verifier']=='Pixel Veritas')
 plan=json.loads(Path(x['plan_path']).read_text())['plan'];wp=plan['waypoints'];xy=np.array([w['uv'] for w in wp])
 fig,ax=plt.subplots(figsize=(6.2,4.8));fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.89)
 ax.imshow(plt.imread(x['path']));line,=ax.plot(xy[:,0],xy[:,1],color='#FFFF00',lw=2.5,marker='o',ms=5);line.set_path_effects([pe.Stroke(linewidth=4,foreground='black'),pe.Normal()])
 grouped={}
 for i,(u,v) in enumerate(xy):grouped.setdefault((round(u,1),round(v,1)),[]).append(str(i+1))
 for (u,v),nums in grouped.items():ax.annotate('/'.join(nums),(u,v),xytext=(7,-6),textcoords='offset points',color='white',fontsize=10,path_effects=[pe.Stroke(linewidth=3,foreground='black'),pe.Normal()])
 ax.set_axis_off();ax.set_title(f'{task.title()} · Veritas trajectory · seed {x["seed"]}',fontsize=11)
 trajectories.append(save(fig,'trajectory/'+task,f'{task.title()} · Veritas trajectory overlay'))
# Re-render all-checkpoint uniform plots as a compact pair.
uniform_keys=[]
for mode in ('argmax','final'):
 fig,ax=plt.subplots(figsize=(6.2,4.7));fig.subplots_adjust(bottom=.22,left=.12,right=.97,top=.88)
 for i,step in enumerate((10000,20000,30000,40000,50000)):
  vals={int(r['N']):100*float(r['success_rate']) for r in rows['eggplant'] if r['model']=='uniform' and int(r['checkpoint_step'])==step and r['selection']==mode}
  if mode=='final':
   n1=next(r for r in rows['eggplant'] if r['model']=='uniform' and int(r['checkpoint_step'])==step and r['selection']=='argmax' and r['N']=='1');vals[1]=100*float(n1['success_rate'])
  ax.plot(NS,[vals.get(n,np.nan) for n in NS],label=f'ST uniform · {step//1000}k',color=COLORS[i+1],marker=['o','s','^','D','v'][i],ms=4)
 ax.set_xscale('log',base=2);ax.set_xticks(NS,labels=NS);ax.set_ylim(0,100);ax.set_xlabel('Candidates N');ax.set_ylabel('Benchmark success (%)');ax.set_title(f'Eggplant · {mode} · clean');ax.grid(alpha=.2);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=3,frameon=False,fontsize=8)
 uniform_keys.append(save(fig,'checkpoints/eggplant_'+mode,f'Eggplant · uniform checkpoints · {mode}'))
# Upload native W&B media; every artifact remains in gscratch.
os.environ['WANDB_MODE']='online';os.environ['WANDB_SILENT']='true'
run=wandb.init(entity='l2sml',project='simpler_veritas_h16',name='concise-report-20261019',job_type='report',dir=str(OUT))
for a in assets:run.log({a['key']:wandb.Image(a['path'],caption=a['title'])})
artifact=wandb.Artifact('simpler-concise-report-20261019',type='report-assets')
for a in assets:artifact.add_file(a['path']);artifact.add_file(str(Path(a['path']).with_suffix('.pdf')))
for t in rows:artifact.add_file(str(ROOT/('stack_veritas_20261006' if t=='stack' else 'eggplant_veritas_resume_20261008')/'results.csv'),name=t+'_results.csv')
run.log_artifact(artifact);rid=run.id;run.finish()
(OUT/'media_manifest.json').write_text(json.dumps(dict(run=rid,assets=assets),indent=2))
r=wr.Report.from_url(URL)
(OUT/'report_before_publish.json').write_text(json.dumps(dataclasses.asdict(r),default=str))
def md(s):return wr.MarkdownBlock(text=s)
def grid(keys,run_id=rid,height=11):
 return wr.PanelGrid(hide_run_sets=True,runsets=[wr.Runset(entity='l2sml',project='simpler_veritas_h16',filters=f'ID == "{run_id}"')],panels=[wr.MediaBrowser(title=next((a['title'] for a in assets if a['key']==key),key),media_keys=[key],num_columns=1,mode='gallery',gallery_axis='run',layout=wr.Layout(x=(i%2)*12,y=(i//2)*height,w=12,h=height)) for i,key in enumerate(keys)])
# Preserve pilot result table with checkpoint identities.
pilot='''## Verifier trials
BC100: eggplant **10k**, stack **40k**. 30 paired episodes. Benchmark success.

| Verifier | Egg N4 | Egg N16 | Stack N4 | Stack N16 |
|---|---:|---:|---:|---:|
| Metric v1 | 40% | 40% | 53% | 57% |
| Metric v2 | 37% | 40% | 47% | 67% |
| Pixel Veritas | 63% | 37% | 50% | 63% |
| Pixel + gripper 0.1 | 63% | 60% | 50% | 60% |
| Two subtasks + sim full filter* | — | 57% | — | — |
| Sim Veritas + distance* | — | 53% | — | — |
| Sim Veritas + full filter* | — | 40% | — | — |

N1: egg **53%**, stack **60%**. *16 samples → 48 gripper choices. Full-filter pass is diagnostic.
'''
under='''## Earlier BC checkpoints
50 demos; 30 paired episodes. Eggplant **20k**, stack **30k**.

| Verifier | Egg N4 / N16 / N64 | Stack N4 / N16 / N64 |
|---|---|---|
| Pixel Veritas | 57 / 60 / 50% | 60 / 67 / 60% |
| + distance | 40 / 63 / 57% | 53 / 63 / 53% |
| Sim Veritas | 50 / 47 / 53% | 53 / 63 / 63% |
| Sim + distance | 40 / 60 / 57% | 60 / 53 / 63% |

N1: egg **53%**, stack **37%**. Stack scales here; the stronger 40k pilot had smaller gains.
'''
def bc_table(task):
 lines=[f'### {task.title()} BC checkpoints','', '| Checkpoint | N1 | N4 | N16 | N64 |','|---|---:|---:|---:|---:|']
 for step in (5000,10000,20000,30000):
  cells=[]
  for n in (1,4,16,64):
   rr=[r for r in rows[task] if r['model']=='bc' and r['split']==('random100' if task=='eggplant' else 'random50') and int(r['checkpoint_step'])==step and int(r['N'])==n]
   cells.append(f"{100*float(rr[0]['success_rate']):.0f}%" if rr else 'Not completed')
  lines.append('| '+str(step//1000)+'k | '+' | '.join(cells)+' |')
 return md('\n'.join(lines))
blocks=[md('''# Simpler runs and results
Updated October 19, 2026 (Pacific). Latest available campaign exports checked against HYAK.

| Task / split | Best completed BC | Best completed ST |
|---|---|---|
| Eggplant, random100 | 60% (30/50), 10k, N64 | 72% (36/50), ramp2to10 50k, clean, argmax, N32 |
| Stack, random50 | 66% (33/50), 20k, N4 | 48% (24/50), uniform 40k, clean, argmax, N16 |
| Stack, quadrant50 holdout | 0% | At most 2% (1/50) |

These are maxima across completed sweeps, not fresh confirmation trials. Candidate scaling is nonmonotonic. The eggplant 72% result was checked against its raw evaluation log.

## Run status
No Simpler training or evaluation jobs are currently running. Stack monitor 41644137 is held with `user env retrieval failed requeued held`. Missing evaluations are unfinished, not necessarily scheduled. The latest comparison export is dated October 14; it contains 157 rows. The stack and resumed eggplant exports contain 272 and 434 rows respectively, including aliases; these are not independent-trial counts.

A failed BC comparison evaluation reports `--obs-corruption clean: DiffusionUnetImagePolicy has no corrupt_obs_eval switch`. This explains at least some missing BC comparison cells; no runs were relaunched for this report update.

## Position splits'''),grid(positions[:2]),grid(positions[2:]),md('Actual source/target positions. Blue circles: train; orange crosses: test. Eggplant basket position is fixed.'),md('## Verifier feedback\nYellow: returned Veritas waypoints used to score candidate endpoints.'),grid(trajectories,height=10)]
for task in ('eggplant','stack'):
 best=max((r for r in rows[task] if r['model']!='bc' and r['split']==('random100' if task=='eggplant' else 'random50')),key=lambda r:float(r['success_rate']))
 blocks.append(md(f"## {task.title()}\n50 test episodes. N=1/2/4/8/16/32/64 candidates; score8 / execute8. BC checkpoint selected by N1; each ST curve uses one checkpoint selected by its best completed sweep success.\n\nBest completed ST: **{100*float(best['success_rate']):.0f}%**, {best['model']} **{int(best['checkpoint_step'])//1000}k**, {best['selection']}, **N={best['N']}**, {best['observations']}. Selection uses these test results; fresh confirmation is pending.\n\nMarkers are completed N values; lines connect them. Missing evaluations remain unfinished."))
 blocks.append(grid([plot_keys[(task,'argmax')],plot_keys[(task,'final')]]))
 missing=[f"{x['model']} {x['checkpoint']//1000}k / {x['observations']} / {x['selection']}: "+','.join(str(n) for n in NS if n not in x['completed_N']) for x in selected if x['task']==task and len(x.get('completed_N',[]))<7]
 if missing:blocks.append(md('Unfinished N: '+'; '.join(missing)+'.'))
 blocks.append(bc_table(task))
blocks +=[md(pilot),md(under),md('## Feedback examples\nHigher score wins; scales differ. BC checkpoints: egg10k / stack40k.')]
(OUT/'selected_curves.json').write_text(json.dumps(selected,indent=2))
feedback=json.loads((DOC/'simpler_verifier_feedback_examples_2026-10-05.json').read_text())
for task in ('eggplant','stack'):
 lines=[f'**{task.title()}**','', '| Verifier | Seed / step | Choice | Score |','|---|---|---:|---:|']
 for x in feedback:
  if x['task']==task:lines.append(f"| {x['verifier']} | {x['seed']} / {x['t']} | {x['candidate']} | {x['score']:.3f} |")
 blocks.append(md('\n'.join(lines)))
blocks.append(md('''## Gripper and drift
| Issue | Tried | Remaining |
|---|---|---|
| Pixel score ignores grasp | Scaled gripper penalty | Wrong open/close choices remain |
| Kinematic drift | Calibrated motion; physics rollout | Simulation alone did not improve scaling |
| Stage switching | Continuous carry; position gates | Stalls / phase errors remain |
| Carry / placement | Full-filter metrics | Wrong trajectory can outweigh task progress |
| Transient success | Placement / release checks | Object can be carried away afterward |

## Rollout examples
Cropped videos; selected candidate and verifier score overlays.
'''))
manifest=json.loads((DOC/'simpler_verifier_wandb_media_manifest.json').read_text())
for x in manifest['media']:
 task='stack' if x['key'].split('/')[-1].startswith('stack') else 'eggplant'
 # Existing pilot ST media encode checkpoint in their original caption.
 caption=x['caption']
 if '30k' not in caption:caption+=f" · BC {'40k' if task=='stack' else '10k'}"
 blocks.append(wr.PanelGrid(hide_run_sets=True,runsets=[wr.Runset(entity='l2sml',project='simpler_veritas_h16',filters=f'ID == "{x["run"]}"')],panels=[wr.MediaBrowser(title=caption,media_keys=[x['key']],num_columns=1,mode='gallery',gallery_axis='run',layout=wr.Layout(w=24,h=13))]))
blocks +=[md('## Eggplant uniform: all checkpoints\nClean observations; 50 episodes/cell.'),grid(uniform_keys),md('[Download report data](https://wandb.ai/l2sml/simpler_veritas_h16/artifacts/report-assets/simpler-concise-report-20261019/latest).')]
r.blocks=blocks;r.width='fixed';r.description='Checkpoint scaling, split positions, verifier trajectories, and rollout failures.';r.save()
check=wr.Report.from_url(URL)
text='\n'.join(b.text for b in check.blocks if isinstance(b,wr.MarkdownBlock));assert 'Images sent to Gemini' not in text
panels=[p for b in check.blocks if isinstance(b,wr.PanelGrid) for p in b.panels]
assert not any(any(str(k).startswith('query_image/') for k in getattr(p,'media_keys',[])) for p in panels)
(OUT/'report_after.json').write_text(json.dumps(dataclasses.asdict(check),default=str))
(OUT/'report.md').write_text(text)
(DOC/'simpler_verifier_trials_2026-10-04.md').write_text(text+'\n')
print('PUBLISHED',URL,'run',rid,'blocks',len(check.blocks),'panels',len(panels),flush=True)
