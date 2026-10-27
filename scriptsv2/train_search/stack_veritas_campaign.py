#!/usr/bin/env python3
"""Stack Veritas campaign: BC + ST (uniform / flat100 / ramp25to100) on two 50-demo splits.

Splits: random50 (random_train50_val70: 50 train / 70 val / 50 test) and quadrant50
(quadrant_train50: train from green-cube start y<=0, val+test from y>0). Verifier: pixel
Veritas, cached plans for training, a fresh Gemini plan at every eval reset. Eval: every
10k ST checkpoint, all 50 test episodes, N=1/2/4/8/16/32/64, argmax/final,
clean/conditional observations. BC is fixed at 20k. At most
MAX_LIVE live cells at once. wandb project stack_veritas.

  python stack_veritas_campaign.py <root> init      # write campaign.json, no jobs
  python stack_veritas_campaign.py <root> prep <split>
  python stack_veritas_campaign.py <root> dry       # print training commands
  python stack_veritas_campaign.py <root> launch    # submit trainings + watcher
  python stack_veritas_campaign.py <root> watch
"""
import json
import csv
import fcntl
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import zipfile

REPO = Path(__file__).resolve().parents[2]
DP = REPO / 'diffusion_policy'
TASK = 'widowx_stack_cube'
SPLITS = dict(random50='random_train50_val70.json', quadrant50='quadrant_train50.json')
KINDS = ('bc', 'uniform', 'flat100', 'ramp25to100')
NOISE = dict(flat100=[100] * 16, ramp25to100=[round(25 + 75 * i / 15) for i in range(16)])
MILESTONES = (10000, 20000, 30000, 40000, 50000)
NS = (1, 2, 4, 8, 16, 32, 64)
MAX_LIVE = 4
MAX_RETRIES = 3
OLD_PLANS = Path('/gscratch/robotics/harine/RoboMonkey_data/experiments/veritas_h16_20261002/plans/stack')
CACHE = Path('/gscratch/robotics/harine/data/veritas_plans/stack/v1_upstream')
CAMERA = DP / f'diffusion_policy/config/veritas/camera_{TASK}.json'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def run(cmd, **kwargs):
    print(shlex.join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, **kwargs)


def split_file(split):
    return DP / 'diffusion_policy/config/splits/simpler/stack' / SPLITS[split]


def train_command(root, spec, smoke=False):
    split, kind = spec['split'], spec['kind']
    mf = split_file(split)
    out = root / 'train' / spec['name']
    common = [f'hydra.run.dir={out}', f'task.dataset.dataset_dir={read(mf)["dataset_dir"]}',
              f'task.dataset.split_file={mf}', 'task.dataset.use_cache=true',
              'training.seed=42', 'dataloader.num_workers=4', 'val_dataloader.num_workers=2',
              f'logging.name={spec["name"]}', f'logging.id={spec["name"]}',
              'logging.project=stack_veritas', f'logging.group={split}',
              '+logging.entity=l2sml']
    if kind == 'bc':
        cmd = [sys.executable, 'diffusion_policy/workspace/train_diffusion_unet_image_workspace.py',
               '--config-name=robomonkey_eggplant_bc', 'horizon=16', 'n_obs_steps=1',
               'training.num_epochs=100000', '+training.max_gradient_steps=50000',
               'training.lr_scheduler=cosine_then_constant', '+training.lr_decay_steps=40000',
               '+training.lr_min_ratio=0.1', 'training.lr_warmup_steps=500',
               'training.checkpoint_every=1000', 'training.resume=true']
    else:
        plans = root / 'plans' / split
        meta = read(plans / 'veritas_targets.json')
        from diffusion_policy.common.veritas_calibration import validate_calibration
        validate_calibration(meta, read(mf), 8, None, str(CAMERA))
        cmd = [sys.executable, 'diffusion_policy/workspace/train_search_outer_inner_robomonkey_workspace.py',
               '--config-name=robomonkey_veritas_h16',
               'training.max_gradient_steps=50000', 'training.resume=true',
               f'policy.verifier.camera_file={CAMERA}',
               f'policy.verifier.value_mean={meta["value_mean"]}',
               f'policy.verifier.value_std={meta["value_std"]}',
               f'task.dataset.extra_obs_arrays.veritas_targets={plans / "veritas_targets.npy"}']
        if kind != 'uniform':
            ts = NOISE[kind]
            assert len(ts) == 16 and all(0 <= t < 1000 for t in ts)
            cmd += ['policy.slot_obs_noise.mode=list',
                    '+policy.slot_obs_noise.timesteps=' + json.dumps(ts, separators=(',', ':'))]
    if smoke:
        common += ['--cfg', 'job', '--resolve']
    return cmd + common


def prep(root, split):
    """Per-split plan dir: reuse every existing plan (old campaign dir, then the offline
    cache); plan the rest live; then build this split's veritas_targets."""
    plans = root / 'plans' / split
    plans.mkdir(parents=True, exist_ok=True)
    mf = read(split_file(split))
    missing = []
    for i in mf['train'] + mf['val'] + mf['test']:
        dst = plans / f'ep{i}.json'
        seed = mf['episodes'][str(i)]['seed']
        for src in (OLD_PLANS / f'ep{i}.json', CACHE / f'ep{i}.json'):
            if dst.exists():
                break
            if src.exists():
                record = read(src)
                if record.get('plan') and record.get('seed') == seed:
                    shutil.copy2(src, dst)
        if not dst.exists() or not read(dst).get('plan'):
            missing.append(i)
        else:
            link = plans / f'seed{seed}.json'
            if not link.exists():
                link.symlink_to(dst.name)
    print(f'[prep] {split}: {len(missing)} episodes need a plan', flush=True)
    if missing:
        run([sys.executable, REPO / 'scriptsv2/veritas/plan_simpler_episodes.py',
             '--task', TASK, '--split-file', split_file(split), '--outdir', plans,
             '--skip-planned', '--episodes', *missing])
    run([sys.executable, REPO / 'scriptsv2/veritas/build_veritas_targets.py',
         '--task', TASK, '--split-file', split_file(split), '--plan-dir', plans,
         '--require-splits', 'train', 'val', 'test', '--horizon', '8'])


def cell_dir(root, cell):
    base = root / 'eval' / cell['run'] / f'step_{cell["step"]:07d}' / cell['mode']
    if 'obs' in cell and not cell.get('legacy_output', False):
        base = base / cell['obs']
    return base / f'n{cell["n"]}'


def evaluate(root, cell):
    checkpoint = Path(cell['checkpoint'])
    if not checkpoint.name.startswith('step_') or checkpoint.resolve().name == 'latest.ckpt':
        raise ValueError(f'Evaluation requires an immutable step checkpoint: {checkpoint}')
    spec = read(root / 'campaign.json')['runs'][cell['run']]
    out = cell_dir(root, cell)
    (out / 'plans').mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(DIFFUSION_POLICY_ROOT=str(DP), VERIFIER_TYPE='veritas',
               VERITAS_CAMERA_FILE=str(CAMERA), VERITAS_PLAN_DIR=str(out / 'plans'),
               VERITAS_ALLOW_LIVE='1', VERITAS_PLAN_DONE_MODE='zero', VERITAS_HORIZON='8',
               VERITAS_ACTION_POS_SCALE='null', N_ACTION_STEPS='8',
               BC_SAMPLER='ddim', BC_VERIFIER_SCORE='full', BC_BATCH_CANDIDATES='1', BC_GRIPPER_AUG='0',
               SPLIT_FILE=str(split_file(spec['split'])), SPLIT_NAME='test',
               N_SAMPLES=str(cell['n']), MODE=cell['mode'], TASK=TASK,
               CONDA_ENV='simpler_env', MAX_STEPS='120', SAVE_VIDEOS='5',
               USE_EMA='1', OBS_CORRUPTION='as_trained', EVAL_OBS_NOISE_LEVEL='',
               RAND_AUG_K='0', EVAL_CHUNK_SELECT='0', TILED_ENVS='1', VIZ_Q='0',
               # Gemini 503 storms can outlast the 1 h default; a cell resumes per episode anyway.
               VERITAS_PLAN_MAX_WAIT_S='21600')
    if 'obs' in cell and not cell.get('legacy_output', False):
        env['OBS_CORRUPTION'] = 'clean' if cell['obs'] == 'clean' and spec['kind'] != 'bc' else 'as_trained'
    if cell['n'] == 1:
        # One candidate: the verifier never changes the executed action, so the plan is
        # irrelevant to the outcome. Use the split's cached plans (zero Gemini, no 503 waits).
        env.update(VERITAS_PLAN_DIR=str(root / 'plans' / spec['split']), VERITAS_ALLOW_LIVE='0')
    run(['bash', REPO / 'scriptsv2/eval_search/eval_search.sh', cell['checkpoint'], 'all', out], env=env)


def submit(root, name, mode, item, *, dependency=None, cpu=False, hours=None):
    cmd = ['sbatch', '--parsable', '--job-name=' + name, '--chdir=' + str(REPO),
           '--account=ckpt-robotics', '--partition=ckpt', '--requeue',
           '--cpus-per-task=' + ('1' if cpu else '6'), '--mem=' + ('2G' if cpu else '48G'),
           '--time=' + (f'{hours}:00:00' if hours else ('7-00:00:00' if cpu else '9:00:00')),
           '--output=' + str(root / 'logs' / '%x-%j.out')]
    if mode in ('train', 'watch'):
        cmd += ['--signal=B:USR1@600', '--open-mode=append']
    if not cpu:
        cmd += ['--qos=ckpt-gpu', '--gres=gpu:1', '--constraint=a40|l40s|l40|a100']
    if dependency:
        cmd += ['--dependency=afterok:' + ':'.join(map(str, dependency))]
    cmd += ['--export=NONE', str(REPO / 'scriptsv2/train_search/stack_veritas_job.sh'),
            str(root), mode, str(item)]
    result = subprocess.check_output(cmd, text=True).strip().split(';')[0]
    print(name, result, flush=True)
    return result


def outstanding_evals():
    out = subprocess.check_output(['squeue', '-h', '-u', os.environ.get('USER', 'harine'),
                                   '-o', '%j'], text=True)
    # N=1 cells plan from cache and are not Gemini-bound, so they do not count toward MAX_LIVE.
    return sum(1 for line in out.splitlines() if line.startswith('ev_sv_') and not line.endswith('_n1'))


def observation_for_legacy(spec):
    # Uniform and BC historically used as_trained, which is effectively clean.
    return 'clean' if spec['kind'] in ('bc', 'uniform') else 'conditional'


def cell_slot(cell):
    return f"{cell['obs']}_{cell['mode']}_n{cell['n']}"


def bc_steps(campaign, name):
    cfg = campaign['eval']
    steps = [cfg['bc_baseline_step']] + cfg.get('bc_extra_sweep_steps', [])
    if cfg.get('bc_select_best_n1'):
        if campaign['runs'][name].get('eval_only'):
            selected = cfg.get('bc_selected_steps', {}).get(name.removesuffix('_train20'))
            if selected is not None:
                steps.append(selected)
        else:
            steps += cfg.get('bc_candidate_steps', {}).get(name, [])
    return sorted(set(steps))


def select_bc_checkpoint(root, campaign, submitted):
    cfg = campaign['eval']
    if not cfg.get('bc_select_best_n1'):
        return
    for name, steps in cfg.get('bc_candidate_steps', {}).items():
        scores = []
        for step in steps:
            entry = submitted.get(f'{name}_{step}', {})
            cell = entry.get('cells', {}).get('clean_argmax_n1')
            if not cell:
                break
            path = cell_dir(root, cell) / 'eval_log.json'
            episodes = path.parent / 'episodes.jsonl'
            if not path.exists() or not episodes.exists():
                break
            records = [json.loads(line) for line in episodes.read_text().splitlines() if line.strip()]
            result = read(path)
            if len(records) != 50 or any(x.get('error') for x in records) or result['num_episodes'] != 50:
                break
            scores.append((result['num_successes'], -step))
        else:
            if scores:
                selected = -max(scores)[1]  # Earliest checkpoint breaks ties.
                cfg.setdefault('bc_selected_steps', {})[name] = selected
                write(root / 'bc_selection.json', dict(selected_steps=cfg['bc_selected_steps'],
                      criterion='maximum test50 N=1 successes; earliest checkpoint breaks ties',
                      note='Test-selected: independent confirmation required'))
    write(root / 'campaign.json', campaign)


def desired_cells(campaign, name, checkpoint, step):
    spec = campaign['runs'][name]
    if spec['kind'] == 'bc' and step not in bc_steps(campaign, name):
        return []
    observations = ('clean',) if spec['kind'] == 'bc' else ('clean', 'conditional')
    modes = ('argmax',) if spec['kind'] == 'bc' else ('argmax', 'final')
    return [dict(key=f'{name}_{step}', run=name, step=step, checkpoint=str(checkpoint),
                 obs=obs, mode=mode, n=n)
            for obs in observations for n in campaign['eval']['ns'] for mode in modes
            if spec['kind'] != 'bc' or step == campaign['eval']['bc_baseline_step']
            or step in campaign['eval'].get('bc_extra_sweep_steps', [])
            or step == campaign['eval'].get('bc_selected_steps', {}).get(name.removesuffix('_train20')) or n == 1]


def reconcile(root, campaign, submitted, queue):
    # Canonicalize legacy identities without moving results or disrupting active evals.
    for key, entry in submitted.items():
        name, step = key.rsplit('_', 1)
        spec = campaign['runs'][name]
        for oldslot in list(entry.get('jobs', {})):
            if oldslot.startswith(('clean_', 'conditional_')):
                continue
            mode, n = oldslot.rsplit('_n', 1)
            cell = dict(key=key, run=name, step=int(step), checkpoint=entry['checkpoint'],
                        mode=mode, n=int(n), obs=observation_for_legacy(spec), legacy_output=True)
            slot = cell_slot(cell)
            entry['jobs'][slot] = entry['jobs'].pop(oldslot)
            entry.setdefault('cells', {})[slot] = cell
            if oldslot in entry.get('retries', {}):
                entry['retries'][slot] = entry['retries'].pop(oldslot)
    normalized = {}
    for cell in queue:
        cell = dict(cell)
        if 'obs' not in cell:
            cell.update(obs=observation_for_legacy(campaign['runs'][cell['run']]), legacy_output=True)
        if campaign['runs'][cell['run']]['kind'] == 'bc' and cell['step'] not in bc_steps(campaign, cell['run']):
            continue
        if cell['n'] == 1 and cell['mode'] == 'final':
            continue
        normalized[(cell['key'], cell_slot(cell))] = cell
    for key, entry in submitted.items():
        name, step = key.rsplit('_', 1)
        for cell in desired_cells(campaign, name, entry['checkpoint'], int(step)):
            slot = cell_slot(cell)
            cell = entry.setdefault('cells', {}).setdefault(slot, cell)
            if campaign['runs'][name]['kind'] == 'uniform' and cell['obs'] == 'conditional':
                source = dict(cell, obs='clean', mode='argmax' if cell['n']==1 else cell['mode'])
                entry.setdefault('aliases', {})[slot] = cell_slot(source)
                normalized.pop((key,slot), None)
                continue
            if cell['n'] == 1 and cell['mode'] == 'final':
                source = dict(cell, mode='argmax')
                entry.setdefault('aliases', {})[slot] = cell_slot(source)
                normalized.pop((key,slot), None)
                continue
            if (cell_dir(root, cell) / 'eval_log.json').exists() or slot in entry.get('jobs', {}):
                normalized.pop((key, slot), None)
                continue
            normalized[(key, slot)] = cell
    return list(normalized.values())


def save_results(root, campaign, submitted):
    rows = []
    for key, entry in submitted.items():
        for slot, cell in entry.get('cells', {}).items():
            if campaign['runs'][cell['run']]['kind'] == 'bc' and cell['step'] not in bc_steps(campaign, cell['run']):
                continue
            alias = entry.get('aliases', {}).get(slot)
            source = entry['cells'][alias] if alias else cell
            path = cell_dir(root, source) / 'eval_log.json'
            if not path.exists():
                continue
            result = read(path)
            spec = campaign['runs'][cell['run']]
            expected_episodes = 20 if spec.get('eval_only') and spec['split'] == 'train20' else 50
            if result['num_episodes'] != expected_episodes:
                continue
            episode_path = path.parent / 'episodes.jsonl'
            if episode_path.exists() and any(json.loads(line).get('error') for line in episode_path.read_text().splitlines() if line.strip()):
                continue
            rows.append(dict(split=spec['split'], model=spec['kind'], checkpoint_step=cell['step'],
                             observations=cell['obs'], selection=cell['mode'], N=cell['n'],
                             successes=result['num_successes'], episodes=result['num_episodes'],
                             success_rate=result['success_rate'],
                             noise_schedule=(spec.get('obs_noise_timesteps') or []) if cell['obs']=='conditional' else '',
                             alias_of=alias or '', source=str(path)))
    rows.sort(key=lambda x:(x['split'], x['model'], x['checkpoint_step'], x['observations'], x['selection'], x['N']))
    if rows:
        tmp=root/'results.csv.tmp'
        with tmp.open('w', newline='') as f:
            writer=csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        tmp.replace(root/'results.csv')
    return len(rows)


def watch(root, once=False):
    # Manual reconciliation and Slurm requeues must never submit the same cell twice.
    with (root / 'watcher.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another watcher owns this campaign; leaving its queue unchanged', flush=True)
            return
        _watch(root, once=once)


def _watch(root, once=False):
    deadline = time.time() + 6.5 * 86400
    sizes = {}
    while time.time() < deadline:
        campaign = read(root / 'campaign.json')
        submitted = read(root / 'evaluations.json') if (root / 'evaluations.json').exists() else {}
        queue = read(root / 'eval_queue.json') if (root / 'eval_queue.json').exists() else []
        for name, spec in campaign['runs'].items():
            ckdir = root / 'train' / name / 'checkpoints'
            milestones = bc_steps(campaign, name) if spec['kind']=='bc' else campaign['eval']['milestones']
            for step in milestones:
                key=f'{name}_{step}'
                if key in submitted:
                    continue
                matches=sorted(p for p in ckdir.glob('step_*.ckpt') if step <= int(p.stem.split('_')[1]) < step+1000)
                if not matches:
                    continue  # Never evaluate latest.ckpt; wait for an immutable saved checkpoint.
                ckpt=matches[0];size=ckpt.stat().st_size;previous=sizes.get(str(ckpt));sizes[str(ckpt)]=size
                if previous != size or not zipfile.is_zipfile(ckpt):
                    continue
                submitted[key]=dict(checkpoint=str(ckpt), jobs={})
        select_bc_checkpoint(root, campaign, submitted)
        queue=reconcile(root, campaign, submitted, queue)
        job_ids=[j for entry in submitted.values() for j in entry.get('jobs', {}).values()]
        if job_ids:
            states=dict(line.split('|')[:2] for line in subprocess.check_output(
                ['sacct','-n','-X','-P','-j',','.join(job_ids),'--format=JobID,State'],text=True).splitlines() if line.strip())
            for key, entry in submitted.items():
                for slot, job in list(entry.get('jobs', {}).items()):
                    cell=entry.get('cells',{}).get(slot)
                    if not cell or slot in entry.get('aliases',{}) or (campaign['runs'][cell['run']]['kind']=='bc' and cell['step'] not in bc_steps(campaign, cell['run'])):
                        continue
                    if (cell_dir(root,cell)/'eval_log.json').exists():
                        continue
                    state=states.get(job,'')
                    if not any(s in state for s in ('FAILED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','CANCELLED')):
                        continue
                    retries=entry.setdefault('retries',{})
                    if retries.get(slot,0)>=MAX_RETRIES:
                        continue
                    retries[slot]=retries.get(slot,0)+1
                    entry.setdefault('failed_jobs',[]).append(job);del entry['jobs'][slot]
                    queue.append(cell)
                    print(f'requeue {key} {slot} after {state}',flush=True)
        # One cell per identity, N1 first, then fixed BC baseline and breadth over ST checkpoints.
        queue=list({(c['key'],cell_slot(c)):c for c in queue}.values())
        queue.sort(key=lambda c:(c['n']!=1, campaign['runs'][c['run']]['kind']!='bc',
                                 {1:0,16:1,4:2,64:3,8:4,32:5,2:6}[c['n']],
                                 c['step'], c['run'], c['obs'], c['mode']))
        write(root/'eval_queue.json',queue);write(root/'evaluations.json',submitted)
        # N1 does not call Gemini; allow a bounded independent lane for those jobs.
        names=subprocess.check_output(['squeue','-h','-u',os.environ.get('USER','harine'),'-o','%j'],text=True).splitlines()
        n1_live=sum(name.startswith('ev_sv_') and name.endswith('_n1') for name in names)
        while queue:
            live=outstanding_evals()
            idx=next((i for i,c in enumerate(queue) if c['n']==1 and n1_live<4),None)
            if idx is None:
                if live>=campaign['eval']['max_live']:
                    break
                idx=next((i for i,c in enumerate(queue) if c['n']!=1),None)
                if idx is None:
                    break
            cell=queue.pop(idx);slot=cell_slot(cell)
            cells_file=root/'cells'/f'{cell["key"]}_{slot}.json';write(cells_file,cell)
            job=submit(root,f'ev_sv_{cell["run"]}_{cell["step"]}_{slot}','eval',cells_file,hours=24)
            submitted[cell['key']]['jobs'][slot]=job
            if cell['n']==1:n1_live+=1
            write(root/'eval_queue.json',queue);write(root/'evaluations.json',submitted)
        count=save_results(root,campaign,submitted)
        print(f'[watch] {count} result rows; {len(queue)} pending cells',flush=True)
        if once:
            return
        expected={f'{name}_{step}' for name,spec in campaign['runs'].items()
                  for step in (bc_steps(campaign, name) if spec['kind']=='bc' else campaign['eval']['milestones'])}
        if not queue and expected.issubset(submitted) and outstanding_evals()==0 and n1_live==0:
            failed=[(key,slot) for key in expected for slot,cell in submitted[key]['cells'].items()
                    if slot not in submitted[key].get('aliases',{}) and not (cell_dir(root,cell)/'eval_log.json').exists()]
            if failed:raise RuntimeError(f'Evaluation cells exhausted retries: {failed}')
            return
        time.sleep(60)
    raise RuntimeError('Watcher timed out; resubmit to resume the same cell identities')


def main():
    root = Path(sys.argv[1]).resolve()
    mode = sys.argv[2]
    item = sys.argv[3] if len(sys.argv) > 3 else ''
    if mode == 'init':
        if (root / 'campaign.json').exists():
            raise RuntimeError('Campaign already exists')
        for folder in ('logs', 'cells', 'train', 'plans', 'eval'):
            (root / folder).mkdir(parents=True, exist_ok=True)
        runs = {}
        for split in SPLITS:
            for kind in KINDS:
                name = f'stack_{split}_{kind}'
                runs[name] = dict(name=name, split=split, kind=kind, task='stack',
                                  obs_noise_timesteps=NOISE.get(kind))
        write(root / 'campaign.json', dict(
            project='stack_veritas', horizon=16, score_horizon=8, execute=8, seed=42,
            eval=dict(episodes='all test (50)', ns=list(NS), plans='live Gemini at every reset',
                      modes=['argmax', 'final'], observations=['clean', 'conditional'],
                      bc_baseline_step=20000, uniform_conditional='same clean observations as training',
                      milestones=list(MILESTONES), max_live=MAX_LIVE),
            splits={k: str(split_file(k)) for k in SPLITS}, runs=runs))
    elif mode == 'prep':
        prep(root, item)
    elif mode == 'dry':
        for spec in read(root / 'campaign.json')['runs'].values():
            print(spec['name'], '\n  ', shlex.join(map(str, train_command(root, spec))))
    elif mode == 'launch':
        campaign = read(root / 'campaign.json')
        for name, spec in campaign['runs'].items():
            if spec.get('job'):
                continue
            spec['job'] = submit(root, 'tr_sv_' + name, 'train', name)
            write(root / 'campaign.json', campaign)
        campaign['watcher'] = submit(root, 'watch_sv', 'watch', '', cpu=True)
        write(root / 'campaign.json', campaign)
    elif mode == 'train':
        spec = read(root / 'campaign.json')['runs'][item]
        cmd = train_command(root, spec)
        write(root / 'train' / f'{item}_command.json', cmd)
        run(cmd, cwd=DP)
    elif mode == 'eval':
        evaluate(root, read(item))
    elif mode == 'watch':
        watch(root, once=item == 'once')
    else:
        raise ValueError(mode)


if __name__ == '__main__':
    main()
