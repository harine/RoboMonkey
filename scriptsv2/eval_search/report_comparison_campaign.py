"""Fill BC checkpoint table and train20 curves for the currently reported policies."""
import csv,fcntl,json,os,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO/'scriptsv2/train_search'))
import stack_veritas_campaign as m
ROOT=Path('/gscratch/robotics/harine/RoboMonkey_data/experiments/report_comparison_20261009');BASE=ROOT.parent
TASKS={'stack':'widowx_stack_cube','eggplant':'widowx_put_eggplant_in_basket'}
def source(task):return BASE/('stack_veritas_20261006' if task=='stack' else 'veritas_h16_20261002')
def runname(task,model):return 'stack_random50_'+model if task=='stack' else f'eggplant_100_{model}_h16_s8_e8_seed42'
def manifest(task):return REPO/'diffusion_policy/diffusion_policy/config/splits/simpler'/task/('random_train50_val70.json' if task=='stack' else 'random_train100.json')
def ckpt(task,model,step):
 paths=sorted((source(task)/'train'/runname(task,model)/'checkpoints').glob('step_*.ckpt'))
 if task=='stack' and model=='bc' and step==5000:paths=sorted((ROOT/'train/stack_bc5k_replay/checkpoints').glob('step_*.ckpt'))
 return next((p for p in paths if step<=int(p.stem.split('_')[1])<step+1000),None)
def out(c):return ROOT/'eval'/c['task']/c['partition']/c['model']/str(c['step'])/c['obs']/c['mode']/f'n{c["n"]}'
def identity(c):return '_'.join(str(c[k]) for k in ('task','partition','model','step','obs','mode','n'))
def init():
 for d in ('eval','logs','cells','train'): (ROOT/d).mkdir(parents=True,exist_ok=True)
 if (ROOT/'cells.json').exists():return
 selected=m.read(BASE/'report_revision_20261009/selected_curves.json');cells={}
 for x in selected:
  for n in m.NS:
   c=dict(task=x['task'],partition='train20',model=x['model'],step=x['checkpoint'],obs=x['observations'],mode='argmax' if n==1 or x['model']=='bc' else x['selection'],n=n)
   cells[identity(c)]=c
 for task in TASKS:
  for step in (5000,10000,20000,30000):
   for n in (1,4,16,64):
    c=dict(task=task,partition='test50',model='bc',step=step,obs='clean',mode='argmax',n=n)
    old=source(task)/'eval'/runname(task,'bc')/f'step_{step:07d}'/'argmax'
    existing=next((p for p in (old/f'n{n}/eval_log.json',old/f'clean/n{n}/eval_log.json') if p.exists()),None)
    if existing:c['existing']=str(existing)
    cells[identity(c)]=c
 m.write(ROOT/'cells.json',cells)
 sys.path.insert(0,str(REPO/'scriptsv2/splits'));from make_position_holdout_splits import balanced
 seeds={}
 for task in TASKS:
  d=m.read(manifest(task));eps={int(i):r for i,r in d['episodes'].items()};ids=[i for i in d['train'] if eps[i]['seed_gate']];ids=balanced(ids,eps,20);seeds[task]=[eps[i]['seed'] for i in ids]
 m.write(ROOT/'train_seeds.json',seeds)

def submit(name,mode,item='',cpu=False):
 cmd=['sbatch','--parsable','--job-name='+name,'--account=ckpt-robotics','--partition=ckpt','--requeue','--chdir='+str(REPO),'--cpus-per-task='+('1' if cpu else '6'),'--mem='+('2G' if cpu else '48G'),'--time='+('7-00:00:00' if cpu else '24:00:00'),'--output='+str(ROOT/'logs/%x-%j.out'),'--export=NONE']
 if cpu:cmd+=['--signal=B:USR1@600','--open-mode=append']
 else:cmd+=['--qos=ckpt-gpu','--gres=gpu:1','--constraint=a40|l40s|l40|a100']
 cmd+=[str(REPO/'scriptsv2/eval_search/report_comparison_job.sh'),mode,str(item)]
 return subprocess.check_output(cmd,text=True).strip().split(';')[0]

def evaluate(key):
 c=m.read(ROOT/'cells.json')[key];checkpoint=ckpt(c['task'],c['model'],c['step']);assert checkpoint
 d=out(c);d.mkdir(parents=True,exist_ok=True);task=c['task'];plans=source(task)/'plans'/('random50' if task=='stack' else 'eggplant')
 env=os.environ.copy();env.update(VERIFIER_TYPE='veritas',VERITAS_STACK_POSITION='0',VERITAS_CAMERA_FILE=str(REPO/f'diffusion_policy/diffusion_policy/config/veritas/camera_{TASKS[task]}.json'),VERITAS_PLAN_DIR=str(plans),VERITAS_ALLOW_LIVE='0',VERITAS_HORIZON='8',VERITAS_PLAN_DONE_MODE='zero',VERITAS_ACTION_POS_SCALE='null',N_ACTION_STEPS='8',BC_SAMPLER='ddim',BC_VERIFIER_SCORE='full',BC_BATCH_CANDIDATES='1',BC_GRIPPER_AUG='0',TASK=TASKS[task],MODE=c['mode'],N_SAMPLES=str(c['n']),OBS_CORRUPTION='clean' if c['obs']=='clean' else 'as_trained',EVAL_OBS_NOISE_LEVEL='',USE_EMA='1',RAND_AUG_K='0',EVAL_CHUNK_SELECT='0',TILED_ENVS='1',SAVE_VIDEOS='2',MAX_STEPS='120',CONDA_ENV='simpler_env')
 if c['partition']=='train20':
  env['SEEDS']=','.join(map(str,m.read(ROOT/'train_seeds.json')[task]))
  if task=='stack' and c['n']>1:env.update(VERITAS_ALLOW_LIVE='1',VERITAS_PLAN_DIR=str(d/'plans'))
 else:
  env.update(SPLIT_FILE=str(manifest(task)),SPLIT_NAME='test')
  if c['n']>1 and task=='stack':env.update(VERITAS_ALLOW_LIVE='1',VERITAS_PLAN_DIR=str(d/'plans'))
 subprocess.run(['bash',str(REPO/'scriptsv2/eval_search/eval_search.sh'),str(checkpoint),'all',str(d)],env=env,cwd=REPO,check=True)

def replay():
 spec=dict(name='stack_bc5k_replay',kind='bc',split='random50');cmd=m.train_command(ROOT,spec)
 cmd=[x.replace('+training.max_gradient_steps=50000','+training.max_gradient_steps=5000') for x in cmd]
 m.write(ROOT/'train/replay_command.json',cmd);subprocess.run(cmd,cwd=m.DP,check=True)

def export(cells):
 rows=[]
 for c in cells.values():
  p=Path(c.get('existing',str(out(c)/'eval_log.json')))
  if p.exists():
   d=m.read(p);rows.append(dict(**{k:c[k] for k in ('task','partition','model','step','obs','mode','n')},successes=d['num_successes'],episodes=d['num_episodes'],success_rate=d['success_rate'],source=str(p)))
 if rows:
  temp=ROOT/'results.tmp'
  with temp.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  temp.replace(ROOT/'results.csv')
 return len(rows)

def watch():
 with (ROOT/'watcher.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  while True:
   cells=m.read(ROOT/'cells.json');done=export(cells)
   active=dict(line.split('|',1) for line in subprocess.check_output(['squeue','-h','-u','harine','-o','%i|%j'],text=True).splitlines())
   live=sum(name.startswith('ev_report_') for name in active.values())
   for key,c in sorted(cells.items(),key=lambda kv:(kv[1]['partition']!='test50',kv[1]['n'],kv[0])):
    if Path(c.get('existing',str(out(c)/'eval_log.json'))).exists() or not ckpt(c['task'],c['model'],c['step']):continue
    if c.get('job') in active:continue
    # Adopt any accepted-but-unrecorded submission before attempting a retry.
    matching=[j for j,n in active.items() if n=='ev_report_'+key]
    if matching:c['job']=matching[0];continue
    if c.get('attempts',0)>=3 or live>=4:continue
    c['attempts']=c.get('attempts',0)+1
    try:c['job']=submit('ev_report_'+key,'eval',key);live+=1
    except subprocess.CalledProcessError:pass
    m.write(ROOT/'cells.json',cells)
   print('completed',done,'/',len(cells),'live',live,flush=True)
   if done==len(cells):return
   time.sleep(60)
if __name__=='__main__':
 mode=sys.argv[1];init()
 if mode=='init':print('cells',len(m.read(ROOT/'cells.json')))
 elif mode=='eval':evaluate(sys.argv[2])
 elif mode=='replay':replay()
 elif mode=='watch':watch()
 elif mode=='launch':
  state=m.read(ROOT/'jobs.json') if (ROOT/'jobs.json').exists() else {}
  if not state.get('replay'):state['replay']=submit('tr_stack_bc5k_replay','replay')
  if not state.get('watcher'):state['watcher']=submit('watch_report_comparison','watch',cpu=True)
  m.write(ROOT/'jobs.json',state);print(state)
