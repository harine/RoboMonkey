#!/usr/bin/env python3
"""Submit/execute the matched H16 / score8 / execute8 Simpler experiment.

The CPU watcher submits evaluations as each 10k milestone is saved. All state,
plans, checkpoints, resolved configurations and evaluation outputs live in ROOT.
"""
import argparse
import json
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
TASKS = dict(eggplant='widowx_put_eggplant_in_basket', stack='widowx_stack_cube')


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


def split(task, n):
    return DP / f'diffusion_policy/config/splits/simpler/{task}/random_train{n}.json'


def camera(task):
    return DP / f'diffusion_policy/config/veritas/camera_{TASKS[task]}.json'


def train_command(root, spec, smoke=False):
    task, kind, n = spec['task'], spec['kind'], spec['demos']
    mf = split(task, n)
    out = root / 'train' / spec['name']
    common = [f'hydra.run.dir={out}', f'task.dataset.dataset_dir={read(mf)["dataset_dir"]}',
              f'task.dataset.split_file={mf}', 'task.dataset.use_cache=true',
              'training.seed=42', 'dataloader.num_workers=4', 'val_dataloader.num_workers=2',
              f'logging.name={spec["name"]}', f'logging.id={spec["name"]}',
              'logging.project=simpler_veritas_h16', 'logging.group=veritas_h16_s8_e8',
              '+logging.entity=l2sml']
    if kind == 'bc':
        cmd = [sys.executable, 'diffusion_policy/workspace/train_diffusion_unet_image_workspace.py',
               '--config-name=robomonkey_eggplant_bc', 'horizon=16', 'n_obs_steps=1',
               'training.num_epochs=100000', f'+training.max_gradient_steps={spec.get("max_gradient_steps", 50000)}',
               'training.lr_scheduler=cosine_then_constant', '+training.lr_decay_steps=40000',
               '+training.lr_min_ratio=0.1', 'training.lr_warmup_steps=500',
               'training.checkpoint_every=1000', 'training.resume=true']
    else:
        plans = root / 'plans' / task
        meta = read(plans / 'veritas_targets.json')
        from diffusion_policy.common.veritas_calibration import validate_calibration
        validate_calibration(meta, read(mf), 8, None, str(camera(task)))
        cmd = [sys.executable, 'diffusion_policy/workspace/train_search_outer_inner_robomonkey_workspace.py',
               '--config-name=robomonkey_veritas_h16',
               f'training.max_gradient_steps={spec.get("max_gradient_steps", 50000)}',
               'training.resume=true',
               f'policy.verifier.camera_file={camera(task)}',
               f'policy.verifier.value_mean={meta["value_mean"]}',
               f'policy.verifier.value_std={meta["value_std"]}',
               f'task.dataset.extra_obs_arrays.veritas_targets={plans / "veritas_targets.npy"}']
        if kind != 'uniform':
            ts = spec['obs_noise_timesteps']
            if len(ts) != 16 or any(not isinstance(t, int) or not 0 <= t < 1000 for t in ts):
                raise ValueError('obs_noise_timesteps must contain 16 DDPM indices in [0, 999]')
            cmd += ['policy.slot_obs_noise.mode=list',
                    '+policy.slot_obs_noise.timesteps=' + json.dumps(ts, separators=(',', ':'))]
    if smoke:
        common += ['--cfg', 'job', '--resolve']
    return cmd + common


def prep(root, task):
    plans = root / 'plans' / task
    plans.mkdir(parents=True, exist_ok=True)
    cache = Path('/gscratch/robotics/harine/data/veritas_plans') / task / 'v1_upstream'
    mf = read(split(task, 100))
    missing = []
    for i in mf['train'] + mf['val'] + mf['test']:
        dst, src = plans / f'ep{i}.json', cache / f'ep{i}.json'
        seed = mf['episodes'][str(i)]['seed']
        if not dst.exists() and src.exists():
            record = read(src)
            if record.get('plan') and record.get('seed') == seed:
                shutil.copy2(src, dst)
        if not dst.exists() or not read(dst).get('plan'):
            missing.append(i)
        else:
            link = plans / f'seed{seed}.json'
            if not link.exists():
                link.symlink_to(dst.name)
    if missing:
        key_file = Path.home() / '.config/gemini/api_key'
        if not os.environ.get('GEMINI_API_KEY') and key_file.exists():
            os.environ['GEMINI_API_KEY'] = key_file.read_text().strip()
        run([sys.executable, REPO / 'scriptsv2/veritas/plan_simpler_episodes.py',
             '--task', TASKS[task], '--split-file', split(task, 100), '--outdir', plans,
             '--skip-planned', '--episodes', *missing])
    run([sys.executable, REPO / 'scriptsv2/veritas/build_veritas_targets.py',
         '--task', TASKS[task], '--split-file', split(task, 100), '--plan-dir', plans,
         '--require-splits', 'train', 'val', 'test', '--horizon', '8'])


def evaluate(root, cell):
    checkpoint = Path(cell['checkpoint'])
    if not checkpoint.name.startswith('step_') or checkpoint.resolve().name == 'latest.ckpt':
        raise ValueError(f'Evaluation requires an immutable step checkpoint: {checkpoint}')
    spec = read(root / 'campaign.json')['runs'][cell['run']]
    task = spec['task']
    env = os.environ.copy()
    env.update(DIFFUSION_POLICY_ROOT=str(DP), VERIFIER_TYPE='veritas',
               VERITAS_CAMERA_FILE=str(camera(task)), VERITAS_PLAN_DIR=str(root / 'plans' / task),
               VERITAS_ALLOW_LIVE='0', VERITAS_PLAN_DONE_MODE='zero', VERITAS_HORIZON='8',
               VERITAS_ACTION_POS_SCALE='null', N_ACTION_STEPS='8',
               BC_SAMPLER='ddim', BC_VERIFIER_SCORE='full', BC_BATCH_CANDIDATES='1',
               SPLIT_FILE=str(split(task, spec['demos'])), SPLIT_NAME='test',
               N_SAMPLES=str(cell['n']), MODE=cell['mode'], TASK=TASKS[task],
               CONDA_ENV='simpler_env', MAX_STEPS='120', SAVE_VIDEOS='5',
               USE_EMA='1', OBS_CORRUPTION=cell.get('obs_corruption', 'as_trained'), EVAL_OBS_NOISE_LEVEL='',
               RAND_AUG_K='0', EVAL_CHUNK_SELECT='0',
               VIZ_Q=str(int(cell.get('viz_q', False))), VIZ_Q_EPISODES='5')
    out = root / cell.get('output_group', 'eval') / cell['run'] / f'step_{cell["step"]:07d}' / cell['mode'] / f'n{cell["n"]}'
    run(['bash', REPO / 'scriptsv2/eval_search/eval_search.sh', cell['checkpoint'],
         str(cell.get('num_episodes', 'all')), out], env=env)
    if cell.get('viz_q'):
        run([sys.executable, REPO / 'scriptsv2/eval_search/render_q.py', out / 'search_q',
             '--out-dir', out / 'overlays', '--limit', '5', '--action-steps', '8'], env=env)
        run([sys.executable, REPO / 'scriptsv2/eval_search/analyze_candidate_diversity.py',
             out / 'search_q'], env=env)
        run([sys.executable, REPO / 'scriptsv2/eval_search/feedback_sensitivity.py',
             cell['checkpoint'], out / 'search_q'], env=env)
        links = root / 'diagnostic_videos'
        links.mkdir(exist_ok=True)
        link = links / f'{cell["run"]}_step{cell["step"]}_{cell["mode"]}_n{cell["n"]}'
        if not link.exists() and not link.is_symlink():
            link.symlink_to(out / 'overlays', target_is_directory=True)


def submit(root, name, mode, item, *, dependency=None, array=None, cpu=False):
    cmd = ['sbatch', '--parsable', '--job-name=' + name, '--chdir=' + str(REPO),
           '--account=ckpt-robotics', '--partition=ckpt', '--requeue',
           '--cpus-per-task=' + ('1' if cpu else '6'), '--mem=' + ('2G' if cpu else '48G'),
           '--time=' + ('7-00:00:00' if cpu else ('9:00:00' if mode == 'train' else '12:00:00')),
           '--output=' + str(root / 'logs' / '%x-%A_%a.out')]
    if mode == 'train':
        cmd += ['--signal=B:USR1@600', '--open-mode=append']
    if not cpu:
        cmd += ['--qos=ckpt-gpu', '--gres=gpu:1', '--constraint=a40|l40s|l40|a100']
    if dependency:
        cmd += ['--dependency=afterok:' + ':'.join(map(str, dependency))]
    if array:
        cmd += ['--array=' + array]
    # Export only scheduling/runtime essentials, never inherited API keys.
    cmd += ['--export=NONE', str(REPO / 'scriptsv2/train_search/veritas_h16_job.sh'),
            str(root), mode, str(item)]
    result = subprocess.check_output(cmd, text=True).strip().split(';')[0]
    print(name, result, flush=True)
    return result


def watch(root):
    campaign = read(root / 'campaign.json')
    deadline = time.time() + 6.5 * 86400
    sizes = {}
    while time.time() < deadline:
        submitted = read(root / 'evaluations.json') if (root / 'evaluations.json').exists() else {}
        prep_ok = all((root / 'plans' / t / 'veritas_targets.json').exists() for t in TASKS)
        for name, spec in campaign['runs'].items():
            ckdir = root / 'train' / name / 'checkpoints'
            for step in range(10000, spec.get('max_gradient_steps', 50000) + 1, 10000):
                key = f'{name}_{step}'
                if key in submitted or not prep_ok:
                    continue
                matches = sorted(p for p in ckdir.glob('step_*.ckpt')
                                 if step <= int(p.stem.split('_')[1]) < step + 1000)
                if matches:
                    ckpt = matches[0]
                elif step == spec.get('max_gradient_steps', 50000) and (ckdir / 'latest.ckpt').exists():
                    status = subprocess.check_output(['sacct', '-n', '-X', '-j', spec['job'],
                                                      '--format=State'], text=True).strip()
                    if status != 'COMPLETED':
                        continue
                    # Pin final weights before a later extension overwrites latest.
                    ckpt = ckdir / f'step_{step:07d}.ckpt'
                    if not ckpt.exists():
                        os.link(ckdir / 'latest.ckpt', ckpt)
                else:
                    continue
                size = ckpt.stat().st_size
                previous = sizes.get(str(ckpt))
                sizes[str(ckpt)] = size
                if previous != size or not zipfile.is_zipfile(ckpt):
                    continue
                modes = ('argmax',) if spec['kind'] == 'bc' else ('argmax', 'final')
                cells = [dict(run=name, step=step, checkpoint=str(ckpt), n=n, mode=mode)
                         for n in (1, 2, 4, 8, 16) for mode in modes
                         if n != 1 or mode == 'argmax']
                if spec.get('ema_buffers_fixed'):
                    for cell in cells:
                        cell['viz_q'] = cell['n'] == 16 and cell['mode'] == 'argmax'
                cells_file = root / 'cells' / f'{key}.json'
                write(cells_file, cells)
                job = submit(root, f'ev_h16_{key}', 'eval', cells_file,
                             array=f'0-{len(cells)-1}%2')
                submitted[key] = dict(job=job, checkpoint=str(ckpt), cells=str(cells_file))
                write(root / 'evaluations.json', submitted)
        if all(f'{name}_{step}' in submitted
               for name, spec in campaign['runs'].items()
               for step in range(10000, spec.get('max_gradient_steps', 50000) + 1, 10000)):
            return
        # Stop visibly if a prerequisite failed; do not silently wait a week for files.
        job_ids = list(campaign['prep'].values()) + [s['job'] for s in campaign['runs'].values()]
        accounting = subprocess.check_output(
            ['sacct', '-n', '-X', '-P', '-j', ','.join(job_ids), '--format=JobID,State'], text=True)
        failures = [line for line in accounting.splitlines()
                    if any(state in line for state in ('FAILED', 'CANCELLED', 'OUT_OF_MEMORY', 'TIMEOUT'))]
        if failures:
            raise RuntimeError('Campaign job failed; resolve and restart watcher: ' + '; '.join(failures))
        time.sleep(60)
    raise RuntimeError('Watcher timed out; resubmit watch to continue from evaluations.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    ap.add_argument('mode', choices=['init', 'prep', 'train', 'eval', 'watch', 'extend', 'clean'])
    ap.add_argument('item', nargs='?', default='')
    a = ap.parse_args()
    root = a.root.resolve()
    if a.mode == 'init':
        if (root / 'campaign.json').exists():
            raise RuntimeError('Campaign already exists; refusing duplicate submission')
        for folder in ('logs', 'cells', 'train', 'plans', 'eval'):
            (root / folder).mkdir(parents=True, exist_ok=True)
        campaign = dict(horizon=16, score_horizon=8, execute=8, seed=42, runs={}, prep={})
        for task in TASKS:
            campaign['prep'][task] = submit(root, f'prep_h16_{task}', 'prep', task)
        for task in TASKS:
            for kind, demos in [('bc', 50), ('bc', 100), ('uniform', 100), ('flat10', 100), ('ramp2to10', 100)]:
                name = f'{task}_{demos}_{kind}_h16_s8_e8_seed42'
                campaign['runs'][name] = dict(name=name, task=task, kind=kind, demos=demos)
                if kind in ('flat10', 'ramp2to10'):
                    campaign['runs'][name]['obs_noise_timesteps'] = (
                        [10] * 16 if kind == 'flat10' else [round(2 + 8 * i / 15) for i in range(16)])
        write(root / 'campaign.json', campaign)
        for name, spec in campaign['runs'].items():
            deps = None if spec['kind'] == 'bc' else [campaign['prep'][spec['task']]]
            spec['job'] = submit(root, 'tr_' + name, 'train', name, dependency=deps)
            write(root / 'campaign.json', campaign)
        campaign['watcher'] = submit(root, 'watch_veritas_h16', 'watch', '', cpu=True)
        write(root / 'campaign.json', campaign)
    elif a.mode == 'prep':
        prep(root, a.item)
    elif a.mode == 'train':
        spec = read(root / 'campaign.json')['runs'][a.item]
        if spec.get('resume_required'):
            import torch
            import dill
            checkpoint = root / 'train' / a.item / 'checkpoints' / 'latest.ckpt'
            payload = torch.load(checkpoint, map_location='cpu', pickle_module=dill)
            step = int(dill.loads(payload['pickles']['global_step']))
            assert step >= spec['resume_min_step'], (step, spec['resume_min_step'])
            assert all(k in payload['state_dicts'] for k in ('model', 'ema_model', 'optimizer'))
            assert payload['state_dicts']['optimizer']['state']
            print(f'Validated continuation from step {step}: model, EMA and optimizer restored', flush=True)
            del payload
        cmd = train_command(root, spec)
        write(root / 'train' / f'{a.item}_command.json', cmd)
        run(cmd, cwd=DP)
    elif a.mode == 'eval':
        evaluate(root, read(a.item)[int(os.environ['SLURM_ARRAY_TASK_ID'])])
    elif a.mode == 'clean':
        from veritas_h16_clean_evals import tick
        while not tick(root):
            time.sleep(60)
    elif a.mode == 'extend':
        from veritas_h16_extensions import tick
        while not tick(root):
            time.sleep(60)
    else:
        watch(root)


if __name__ == '__main__':
    main()
