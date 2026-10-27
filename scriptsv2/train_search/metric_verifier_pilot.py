"""Small, versioned fixed-checkpoint BC scaling pilot; no implicit training launch."""
import json,os,subprocess,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
ROOT=Path('/gscratch/robotics/harine/RoboMonkey_data/experiments/metric_verifier_pilot_20261004')
OLD=Path('/gscratch/robotics/harine/RoboMonkey_data/experiments/veritas_h16_20261002')
TASKS={'stack':'widowx_stack_cube','eggplant':'widowx_put_eggplant_in_basket'}

def setup():
    ROOT.mkdir(exist_ok=True,parents=True);(ROOT/'logs').mkdir(exist_ok=True)
    specs={}
    for task,step in [('stack',40000),('eggplant',10000)]:
        old=OLD/'eval'/f'{task}_100_bc_h16_s8_e8_seed42'/f'step_{step:07d}'/'argmax'/'n1'
        d=json.loads((old/'eval_log.json').read_text())
        specs[task]=dict(checkpoint=d['checkpoint'],seeds=d['seeds'][:30],step=step)
    manifest=dict(verifier='veritas_metric_v1',specs=specs,ns=[1,4,16],episodes=30,
        pool_size=16,gemini_cap_usd=100,per_rollout_request_cap=3,
        gate='Per task: N16 at least 3 net successes above N1, N4 >= N1, one-sided paired exact p <= 0.10; zero infrastructure failures. Both tasks must pass before launching both-task ST campaign.',
        training_if_pass=dict(demos=50,variants=['uniform','flat10','ramp2to10'],max_steps=100000,
            eval_selection=['argmax','final'],eval_ns=[1,2,4,8,16]))
    (ROOT/'pilot.json').write_text(json.dumps(manifest,indent=2))

def evaluate(task,n,smoke=False,revision='v1'):
    trial=revision in ('subtask_trial','subtask_two_trial','subtask_two_grip_trial','subtask_two_strict_trial','subtask_two_full_trial','pixel_sim_distance_trial','pixel_sim_full_trial')
    trial_episodes=int(os.environ.get('TRIAL_EPISODES','1')) if trial else 30
    if not 1<=trial_episodes<=30:raise ValueError('trial episodes must be 1..30')
    pixel=revision.startswith('pixel_w') or trial
    manifest='pilot_pixel_gripper.json' if pixel else ('pilot.json' if revision=='v1' else 'pilot_v2.json')
    d=json.loads((ROOT/manifest).read_text());spec=d['specs'][task]
    group='eval' if revision=='v1' else 'eval_'+revision
    if trial and trial_episodes>1:group+=f'_{trial_episodes}ep'
    out=ROOT/(('smoke3' if revision=='v1' else 'smoke_v2') if smoke else group)/task/f'n{n}';out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(DIFFUSION_POLICY_ROOT=str(REPO/'diffusion_policy'),
        VERIFIER_TYPE='veritas' if n==1 else 'veritas_metric',
        VERITAS_CAMERA_FILE=str(REPO/f'diffusion_policy/diffusion_policy/config/veritas/camera_{TASKS[task]}.json'),
        VERITAS_PLAN_DIR=str(OLD/'plans'/task),VERITAS_ALLOW_LIVE='0',VERITAS_PLAN_DONE_MODE='zero',
        VERITAS_HORIZON='8',VERITAS_ACTION_POS_SCALE='null',N_ACTION_STEPS='8',
        BC_SAMPLER='ddim',BC_VERIFIER_SCORE='full',BC_BATCH_CANDIDATES='1',BC_CANDIDATE_POOL='16',
        N_SAMPLES=str(n),MODE='argmax',TASK=TASKS[task],CONDA_ENV='simpler_env',MAX_STEPS='120',
        SAVE_VIDEOS='5',USE_EMA='1',OBS_CORRUPTION='as_trained',RAND_AUG_K='0',EVAL_CHUNK_SELECT='0',
        VIZ_Q='1',VIZ_Q_EPISODES='5',TILED_ENVS='1',METRIC_PILOT_ROOT=str(ROOT),METRIC_TRACE_DIR=str(out),
        SEEDS=','.join(map(str,spec['seeds'][2:3] if smoke else spec['seeds'])),
        EVAL_OBS_NOISE_LEVEL='')
    if revision=='v2':
        env.update(METRIC_VERIFIER_REVISION='v2',METRIC_CALIBRATION=str(ROOT/f'calibration_{task}.json'))
    if pixel:
        env.update(VERIFIER_TYPE='veritas_pixel_gripper',
            VERITAS_GRIPPER_WEIGHT=('0.1' if trial else revision.split('_w')[1].replace('p','.')),METRIC_VERIFIER_REVISION='v1',
            METRIC_CALIBRATION='',VERITAS_LOOKAHEAD_K='3',VERITAS_LOOKAHEAD_ALPHA='0.5')
    if trial:
        env.update(VERIFIER_TYPE='veritas_subtask',SEEDS=','.join(map(str,spec['seeds'][:trial_episodes])),
            SUBTASK_RECIPE='two_physical_v2' if revision.startswith('subtask_two') else 'four_waypoint_v1')
    if revision in ('subtask_two_grip_trial','subtask_two_strict_trial','subtask_two_full_trial','pixel_sim_distance_trial','pixel_sim_full_trial'):env['BC_GRIPPER_AUG']='1'
    if revision=='subtask_two_full_trial':env.update(FULL_FILTER_LOOKAHEAD='1',STRICT_TASK_WEIGHT='0')
    if revision=='subtask_two_strict_trial':env['STRICT_TASK_WEIGHT']='1'
    if revision in ('pixel_sim_distance_trial','pixel_sim_full_trial'):
        env.update(VERIFIER_TYPE='veritas_pixel_gripper',VERITAS_GRIPPER_WEIGHT='0',
            FULL_FILTER_LOOKAHEAD='1',SIM_LOOKAHEAD_MODE=('trajectory_full_filter' if revision=='pixel_sim_full_trial' else 'trajectory_distance'),STRICT_TASK_WEIGHT='0')
    subprocess.run(['bash',str(REPO/'scriptsv2/eval_search/eval_search.sh'),spec['checkpoint'],
        '1' if smoke else str(trial_episodes),str(out)],env=env,check=True,cwd=REPO)
    if trial and trial_episodes>1:
        subprocess.run([sys.executable,str(REPO/'scriptsv2/analysis/sim_verifier_report.py')],check=True,cwd=REPO)
    if not trial:subprocess.run([sys.executable,str(REPO/'scriptsv2/train_search/metric_pilot_report.py'),revision],check=True,cwd=REPO)
    subprocess.run([sys.executable,str(REPO/'scriptsv2/eval_search/render_q.py'),str(out/'search_q'),
        '--out-dir',str(out/'overlays'),'--limit','5','--action-steps','8'],check=True,cwd=REPO)

if __name__=='__main__':
    if sys.argv[1]=='setup':setup()
    else:
        revision=next((x for x in sys.argv if x.startswith('pixel_w') or x in ('subtask_trial','subtask_two_trial','subtask_two_grip_trial','subtask_two_strict_trial','subtask_two_full_trial','pixel_sim_distance_trial','pixel_sim_full_trial')),'v2' if '--v2' in sys.argv else 'v1')
        evaluate(sys.argv[1],int(sys.argv[2]),'--smoke' in sys.argv,revision)
