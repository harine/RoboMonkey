#!/usr/bin/env python3
"""Resume eggplant evaluation with the original cached-plan pixel Veritas verifier.

Uses the same resumable seven-width, clean/as-trained sweep controller as stack.
Training resumes separately through veritas_h16_campaign.py in the original run dirs.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import stack_veritas_campaign as campaign
import veritas_h16_campaign as original


def configure(root):
    source = Path(campaign.read(root / 'campaign.json')['source_campaign'])
    campaign.TASK = original.TASKS['eggplant']
    campaign.CAMERA = original.camera('eggplant')
    campaign.split_file = lambda split: original.split('eggplant', int(split.replace('random', '')))
    def outstanding_cached_evals():
        names = subprocess.check_output(
            ['squeue', '-h', '-u', os.environ.get('USER', 'harine'), '-o', '%j'], text=True)
        return sum(name.startswith('ev_sv_eggplant_') and not name.endswith('_n1')
                   for name in names.splitlines())
    # Cached-plan cells use no Gemini quota; give this campaign its own bounded lane.
    campaign.outstanding_evals = outstanding_cached_evals
    saved_run = campaign.run
    def cached_run(cmd, **kwargs):
        if 'env' in kwargs:
            kwargs['env'] = dict(kwargs['env'], VERITAS_ALLOW_LIVE='0',
                                 VERITAS_PLAN_DIR=str(source / 'plans' / 'eggplant'))
        return saved_run(cmd, **kwargs)
    campaign.run = cached_run
    def submit(root, name, mode, item, *, dependency=None, cpu=False, hours=None):
        cmd = ['sbatch', '--parsable', '--job-name='+name, '--chdir='+str(campaign.REPO),
               '--account=ckpt-robotics', '--partition=ckpt', '--requeue',
               '--cpus-per-task='+('1' if cpu else '6'), '--mem='+('2G' if cpu else '48G'),
               '--time='+(f'{hours}:00:00' if hours else ('7-00:00:00' if cpu else '24:00:00')),
               '--output='+str(root/'logs'/'%x-%j.out')]
        if mode=='watch':cmd += ['--signal=B:USR1@600','--open-mode=append']
        if not cpu:cmd += ['--qos=ckpt-gpu','--gres=gpu:1','--constraint=a40|l40s|l40|a100']
        if dependency:cmd += ['--dependency=afterok:'+':'.join(map(str,dependency))]
        cmd += ['--export=NONE',str(campaign.REPO/'scriptsv2/train_search/eggplant_veritas_resume.sh'),str(root),mode,str(item)]
        job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
        print(name,job,flush=True);return job
    campaign.submit = submit


def main():
    root=Path(sys.argv[1]).resolve();mode=sys.argv[2];item=sys.argv[3] if len(sys.argv)>3 else ''
    configure(root)
    if mode=='watch':campaign.watch(root,once=item=='once')
    elif mode=='eval':campaign.evaluate(root,campaign.read(item))
    elif mode=='launch':
        cfg=campaign.read(root/'campaign.json');cfg['watcher']=campaign.submit(root,'watch_sv_eggplant','watch','',cpu=True);campaign.write(root/'campaign.json',cfg)
    else:raise ValueError(mode)

if __name__=='__main__':main()
