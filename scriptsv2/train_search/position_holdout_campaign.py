"""50-demo position-holdout campaigns; resumable 10k checkpoint and train20/test50 sweeps."""
import json,os,shutil,subprocess,sys,time
from pathlib import Path
import stack_veritas_campaign as m
BASE=Path('/gscratch/robotics/harine/RoboMonkey_data/experiments/position_holdout_20261009')
TASKS={'stack':'widowx_stack_cube','eggplant':'widowx_put_eggplant_in_basket'}

def demos(root):return 100 if root.name.endswith('100') else 50

def configure(root):
 task=root.name.removesuffix('100');m.TASK=TASKS[task];m.CAMERA=m.DP/f'diffusion_policy/config/veritas/camera_{m.TASK}.json'
 mf=BASE/'splits'/task/f'position_holdout_train{demos(root)}.json';m.split_file=lambda split:mf
 m.NOISE={'flat100':[100]*16,'ramp25to100':[round(25+75*i/15) for i in range(16)],'flat10':[10]*16,'ramp2to10':[round(2+8*i/15) for i in range(16)]}
 def submit(root,name,mode,item,*,dependency=None,cpu=False,hours=None):
  cmd=['sbatch','--parsable','--job-name='+name,'--chdir='+str(m.REPO),'--account=ckpt-robotics','--partition=ckpt','--requeue','--cpus-per-task='+('1' if cpu else '6'),'--mem='+('2G' if cpu else '48G'),'--time='+('7-00:00:00' if cpu else '24:00:00'),'--output='+str(root/'logs/%x-%j.out'),'--export=NONE']
  if mode in ('train','watch'):cmd+=['--signal=B:USR1@600','--open-mode=append']
  if not cpu:cmd+=['--qos=ckpt-gpu','--gres=gpu:1','--constraint=a40|l40s|l40|a100']
  if dependency:cmd+=['--dependency=afterok:'+':'.join(map(str,dependency))]
  cmd+=[str(m.REPO/'scriptsv2/train_search/position_holdout_job.sh'),str(root),mode,str(item)]
  # A failed submission can have been accepted: adopt by exact job name before retry.
  for attempt in range(3):
   try:return subprocess.check_output(cmd,text=True).strip().split(';')[0]
   except subprocess.CalledProcessError:
    existing=[line.split('|')[0] for line in subprocess.check_output(['squeue','-h','-u','harine','-o','%i|%j'],text=True).splitlines() if line.split('|')[1]==name]
    if len(existing)==1:return existing[0]
    if len(existing)>1:raise RuntimeError('Ambiguous accepted job '+name)
    time.sleep(10)
  raise RuntimeError('Slurm submission failed after retries')
 m.submit=submit
 m.outstanding_evals=lambda:sum(x.startswith('ev_sv_ph_'+task+'_') and not x.endswith('_n1') for x in subprocess.check_output(['squeue','-h','-u','harine','-o','%j'],text=True).splitlines())
 saved_run=m.run
 def run(cmd,**kwargs):
  if 'env' in kwargs:
   env=dict(kwargs['env']);env['VERITAS_STACK_POSITION']='1' if task=='stack' else '0'
   # Train20 and test50 both plan freshly after reset; N1 can reuse a plan.
   if '_train20/' in str(cmd[-1]):
    env.update(SEEDS=','.join(map(str,m.read(root/'train_eval_seeds.json'))),SPLIT_NAME='train')
    if env['N_SAMPLES']=='1':env.update(VERITAS_ALLOW_LIVE='0',VERITAS_PLAN_DIR=str(root/f'plans/holdout{demos(root)}'))
   kwargs['env']=env
  return saved_run(cmd,**kwargs)
 m.run=run
 return mf

def prep(root):
 task=root.name.removesuffix('100');mf=configure(root);data=m.read(mf);out=root/f'plans/holdout{demos(root)}';out.mkdir(parents=True,exist_ok=True)
 dirs=[BASE/'eggplant100/plans/holdout100'] if task=='eggplant' and demos(root)==50 else []
 dirs +=[BASE.parent/'veritas_h16_20261002/plans'/task,Path('/gscratch/robotics/harine/data/veritas_plans')/task/'v1_upstream']
 if task=='stack':dirs +=[BASE.parent/'stack_veritas_20261006/plans/random50',BASE.parent/'stack_veritas_20261006/plans/quadrant50']
 missing=[]
 for i in data['train']+data['val']+data['test']:
  seed=data['episodes'][str(i)]['seed'];dest=out/f'ep{i}.json'
  for source in [dest]+[d/f for d in dirs for f in (f'seed{seed}.json',f'ep{i}.json')]:
   if not source.exists():continue
   rec=m.read(source)
   if rec.get('seed')==seed and rec.get('plan'):
    if source!=dest:shutil.copy2(source,dest)
    link=out/f'seed{seed}.json'
    if not link.exists():link.symlink_to(dest.name)
    break
  else:missing.append(i)
 m.write(root/'missing_plans.json',missing)
 if missing:m.run([sys.executable,m.REPO/'scriptsv2/veritas/plan_simpler_episodes.py','--task',m.TASK,'--split-file',mf,'--outdir',out,'--model','gemini-3.1-flash-lite','--skip-planned','--episodes',*map(str,missing)])
 cmd=[sys.executable,m.REPO/'scriptsv2/veritas/build_veritas_targets.py','--task',m.TASK,'--split-file',mf,'--plan-dir',out,'--horizon','8','--require-splits','train','val','test']
 if task=='stack':cmd+=['--stack-position']
 m.run(cmd)

def train(root,name):
 configure(root);spec=m.read(root/'campaign.json')['runs'][name];cmd=m.train_command(root,spec)
 cmd=[x.replace('logging.project=stack_veritas','logging.project=simpler_veritas_h16').replace(f'logging.group=holdout{demos(root)}','logging.group=position_holdout_20261009_'+root.name) for x in cmd]
 if spec['kind']=='bc':cmd=[x.replace('+training.max_gradient_steps=50000','+training.max_gradient_steps=20000') for x in cmd]
 elif root.name=='stack':
  meta=m.read(root/f'plans/holdout{demos(root)}/veritas_targets.json');assert meta['verifier_version']=='veritas_stack_position_v1'
  cmd+=['policy.verifier._target_=diffusion_policy.policy.veritas_stack_position_verifier.VeritasStackPositionVerifier']
 m.write(root/'train'/f'{name}_command.json',cmd)
 m.run(cmd,cwd=m.DP)

def initialize(root):
 task=root.name.removesuffix('100');configure(root)
 for d in ('train','eval','plans','logs','cells'): (root/d).mkdir(parents=True,exist_ok=True)
 if (root/'campaign.json').exists():return
 kinds=('bc','uniform','flat100','ramp25to100') if task=='stack' else ('bc','uniform','flat10','ramp2to10')
 runs={}
 for kind in kinds:
  name=f'ph_{task}_{demos(root)}_{kind}';spec=dict(name=name,kind=kind,split=f'holdout{demos(root)}',obs_noise_timesteps=m.NOISE.get(kind,[]));runs[name]=spec
  runs[name+'_train20']=dict(spec,name=name+'_train20',split='train20',eval_only=True)
  (root/'train'/name).mkdir(exist_ok=True);(root/'train'/(name+'_train20')).symlink_to(name,target_is_directory=True)
 data=m.read(m.split_file('holdout50'))
 sys.path.insert(0,str(m.REPO/'scriptsv2/splits'));from make_position_holdout_splits import balanced
 subset=balanced(data['train'],{int(i):r for i,r in data['episodes'].items()},20)
 m.write(root/'train_eval_seeds.json',[data['episodes'][str(i)]['seed'] for i in subset])
 m.write(root/'campaign.json',dict(task=task,verifier='veritas_stack_position_v1' if task=='stack' else 'veritas_pixel',runs=runs,eval=dict(ns=list(m.NS),milestones=list(m.MILESTONES),bc_baseline_step=20000,max_live=4,observations=['clean','conditional'],modes=['argmax','final'],episodes='train20 / test50')))
 m.write(root/'evaluations.json',{});m.write(root/'eval_queue.json',[])

def main():
 root=Path(sys.argv[1]).resolve();mode=sys.argv[2];item=sys.argv[3] if len(sys.argv)>3 else '';configure(root)
 if mode=='init':initialize(root)
 elif mode=='prep':prep(root)
 elif mode=='train':train(root,item)
 elif mode=='eval':m.evaluate(root,m.read(item))
 elif mode=='watch':m.watch(root)
 elif mode=='launch':
  cfg=m.read(root/'campaign.json')
  if not cfg.get('prep_job'):cfg['prep_job']=m.submit(root,'prep_ph_'+root.name,'prep','',dependency=cfg.get('start_after_jobs'))
  for name,spec in cfg['runs'].items():
   if not spec.get('eval_only') and not spec.get('job'):spec['job']=m.submit(root,'tr_'+name,'train',name,dependency=[cfg['prep_job']]);m.write(root/'campaign.json',cfg)
  if not cfg.get('watcher'):cfg['watcher']=m.submit(root,'watch_ph_'+root.name,'watch','',cpu=True,dependency=cfg.get('start_after_jobs'))
  m.write(root/'campaign.json',cfg);print(json.dumps(dict(prep=cfg['prep_job'],train={n:s.get('job') for n,s in cfg['runs'].items() if not s.get('eval_only')},watcher=cfg['watcher'])))
 else:raise ValueError(mode)
if __name__=='__main__':main()
