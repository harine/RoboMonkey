"""Refresh BC tables and fixed-checkpoint train20/test50 report curves as results land."""
import csv,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
import wandb_workspaces.reports.v2 as wr
BASE=Path('/gscratch/robotics/harine/RoboMonkey_data/experiments');ROOT=BASE/'report_comparison_20261009';OUT=ROOT/'report';OUT.mkdir(exist_ok=True)
URL='https://wandb.ai/l2sml/simpler_veritas_h16/reports/Simpler-verifier-trials-eggplant-and-stack-2026-10-04--VmlldzoxODA1NTg1Mg'
COLORS=['#000000','#0072B2','#D55E00','#009E73'];NS=[1,2,4,8,16,32,64]
def readcsv(p):return list(csv.DictReader(p.open())) if p.exists() else []
def old_result(task,model,step,obs,mode,n):
 source=BASE/('stack_veritas_20261006' if task=='stack' else 'veritas_h16_20261002');name='stack_random50_'+model if task=='stack' else f'eggplant_100_{model}_h16_s8_e8_seed42'
 # Current campaign CSV carries clean/as-trained identities; use it first.
 campaign=BASE/('stack_veritas_20261006' if task=='stack' else 'eggplant_veritas_resume_20261008')/'results.csv'
 matches=[r for r in readcsv(campaign) if r['split']==('random50' if task=='stack' else 'random100') and r['model']==model and int(r['checkpoint_step'])==step and r['observations']==obs and r['selection']==mode and int(r['N'])==n]
 if matches:return float(matches[0]['success_rate'])*100
 if model=='bc':
  for p in [(source/'eval'/name/f'step_{step:07d}'/'argmax'/f'n{n}/eval_log.json'),(source/'eval'/name/f'step_{step:07d}'/'argmax'/'clean'/f'n{n}/eval_log.json')]:
   if p.exists():return json.loads(p.read_text())['success_rate']*100
 return None

def update():
 cells=readcsv(ROOT/'results.csv');selected=json.loads((BASE/'report_revision_20261009/selected_curves.json').read_text());plots=[];choices=[]
 for task in ('eggplant','stack'):
  for mode in ('argmax','final'):
   fig,ax=plt.subplots(figsize=(6.2,5.4));fig.subplots_adjust(bottom=.35,left=.12,right=.97,top=.9)
   groups={}
   for s in selected:
    if s['task']!=task or s['selection']!=mode:continue
    vals={n:old_result(task,s['model'],s['checkpoint'],s['observations'],'argmax' if n==1 or s['model']=='bc' else mode,n) for n in NS};vals={n:v for n,v in vals.items() if v is not None}
    if not vals:continue
    if s['model'] not in groups or max(vals.values())>max(groups[s['model']][1].values()):groups[s['model']]=(s,vals)
   for idx,(model,(s,test)) in enumerate(groups.items()):
    train={int(r['n']):100*float(r['success_rate']) for r in cells if r['task']==task and r['partition']=='train20' and r['model']==model and int(r['step'])==s['checkpoint'] and r['obs']==s['observations'] and (r['mode']==('argmax' if model=='bc' else mode) or r['n']=='1')}
    label=f"{'BC' if model=='bc' else model} {s['checkpoint']//1000}k"+(' · noisy' if s['observations']=='conditional' else ' · clean')
    ax.plot(sorted(test),[test[n] for n in sorted(test)],color=COLORS[idx],marker='s',lw=1.7,ms=4,label=label+' test50')
    ax.plot(sorted(train),[train[n] for n in sorted(train)],color=COLORS[idx],marker='o',ls='--',lw=1.7,ms=4,label=label+(' train20' if train else ' train20 pending'))
    choices.append(dict(task=task,mode=mode,**{k:s[k] for k in ('model','checkpoint','observations')},train_completed=sorted(train),test_completed=sorted(test)))
   ax.set_xscale('log',base=2);ax.set_xticks(NS,labels=NS);ax.set_ylim(0,100);ax.set_xlabel('Candidates N');ax.set_ylabel('Benchmark success (%)');ax.set_title(f'{task.title()} · {mode} · train vs test',fontsize=11);ax.grid(alpha=.2);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=2,fontsize=7.4,frameon=False)
   p=OUT/f'{task}_{mode}.png';fig.savefig(p,dpi=170);fig.savefig(p.with_suffix('.pdf'));plt.close(fig);plots.append((f'train_test/{task}_{mode}',p))
 (OUT/'curves.json').write_text(json.dumps(choices,indent=2))
 state=json.loads((OUT/'media_run.json').read_text()) if (OUT/'media_run.json').exists() else {}
 run=wandb.init(entity='l2sml',project='simpler_veritas_h16',name='train-test-progress-20261009',id=state.get('id'),resume='allow',dir=str(OUT))
 for key,p in plots:run.log({key:wandb.Image(str(p))})
 rid=run.id;run.finish();(OUT/'media_run.json').write_text(json.dumps(dict(id=rid)))
 report=wr.Report.from_url(URL)
 # Replace just this section; preserve all other report edits and media.
 blocks=[];skipping=False
 for b in report.blocks:
  if isinstance(b,wr.MarkdownBlock) and b.text.startswith('## Train-subset versus test'):
   skipping=True;continue
  if skipping and isinstance(b,wr.MarkdownBlock) and b.text.startswith('<!-- end train-test -->'):
   skipping=False;continue
  if not skipping:blocks.append(b)
 for idx,b in enumerate(blocks):
  if isinstance(b,wr.MarkdownBlock):
   for task in ('eggplant','stack'):
    if b.text.startswith(f'### {task.title()} BC checkpoints'):
     lines=[f'### {task.title()} BC checkpoints','','| Checkpoint | N1 | N4 | N16 | N64 |','|---|---:|---:|---:|---:|']
     for step in (5000,10000,20000,30000):
      vals=[]
      for n in (1,4,16,64):
       rr=[r for r in cells if r['task']==task and r['partition']=='test50' and int(r['step'])==step and int(r['n'])==n]
       v=100*float(rr[0]['success_rate']) if rr else old_result(task,'bc',step,'clean','argmax',n)
       vals.append(f'{v:.0f}%' if v is not None else 'Pending')
      lines.append('| '+str(step//1000)+'k | '+' | '.join(vals)+' |')
     if task=='stack':lines+=['','5k is regenerated with the original configuration/seed.']
     blocks[idx]=wr.MarkdownBlock(text='\n'.join(lines))
 section=[wr.MarkdownBlock(text='## Train-subset versus test\nFixed 20 training episodes versus 50 held-out episodes. Same checkpoint and N on both; dashed = train, solid = test. Best reported observation setting per policy. Pending train curves appear as results arrive.')]
 for task in ('eggplant','stack'):
  section.append(wr.PanelGrid(hide_run_sets=True,runsets=[wr.Runset(entity='l2sml',project='simpler_veritas_h16',filters=f'ID == "{rid}"')],panels=[wr.MediaBrowser(title=f'{task.title()} · {mode}',media_keys=[f'train_test/{task}_{mode}'],num_columns=1,mode='gallery',gallery_axis='run',layout=wr.Layout(x=i*12,w=12,h=12)) for i,mode in enumerate(('argmax','final'))]))
 section.append(wr.MarkdownBlock(text='<!-- end train-test -->'))
 # Insert before historical verifier trials.
 index=next((i for i,b in enumerate(blocks) if isinstance(b,wr.MarkdownBlock) and b.text.startswith('## Verifier trials')),len(blocks))
 report.blocks=blocks[:index]+section+blocks[index:];report.width='fixed';report.save();print('Updated train/test plots and BC tables',len(cells),'completed cells',flush=True)
if __name__=='__main__':
 previous=None
 while True:
  signature=hashlib.sha256((ROOT/'results.csv').read_bytes() if (ROOT/'results.csv').exists() else b'').hexdigest()
  if signature!=previous:
   update();previous=signature
  if '--watch' not in sys.argv:break
  time.sleep(600)
