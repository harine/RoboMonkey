"""Veritas BoN on undertrained BC (eggplant train50@20k, stack train50@30k), 30 report seeds.

Usage (inside the sbatch): python veritas_undertrained_bon.py <task> <arm> <n>
Arms: n1 (single candidate; cached plans, zero Gemini) and the BON_SCORE arms
veritas | veritas_distance | sim_veritas | sim_veritas_distance (live Gemini plan at every reset,
saved in <cell>/plans). SMOKE=1 runs 1 episode into smoke/.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/gscratch/robotics/harine/RoboMonkey_data/experiments/veritas_undertrained_bon_20261006')
OLD = Path('/gscratch/robotics/harine/RoboMonkey_data/experiments/veritas_h16_20261002')
PILOT = Path('/gscratch/robotics/harine/RoboMonkey_data/experiments/metric_verifier_pilot_20261004/pilot.json')
TASKS = {'stack': 'widowx_stack_cube', 'eggplant': 'widowx_put_eggplant_in_basket'}
CKPTS = {
    'eggplant': OLD / 'train/eggplant_50_bc_h16_s8_e8_seed42/checkpoints/step_0020001.ckpt',
    'stack': OLD / 'train/stack_50_bc_h16_s8_e8_seed42/checkpoints/step_0030001.ckpt',
}


def evaluate(task, arm, n):
    seeds = json.loads(PILOT.read_text())['specs'][task]['seeds']
    smoke = os.environ.get('SMOKE') == '1'
    if smoke:
        seeds = seeds[:1]
    out = ROOT / (os.environ.get('SMOKE_DIR', 'smoke') if smoke else 'eval') / task / arm / f'n{n}'
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        DIFFUSION_POLICY_ROOT=str(REPO / 'diffusion_policy'), VERIFIER_TYPE='veritas',
        VERITAS_CAMERA_FILE=str(REPO / f'diffusion_policy/diffusion_policy/config/veritas/camera_{TASKS[task]}.json'),
        VERITAS_PLAN_DIR=str(OLD / 'plans' / task), VERITAS_ALLOW_LIVE='0', VERITAS_PLAN_DONE_MODE='zero',
        VERITAS_HORIZON='8', VERITAS_ACTION_POS_SCALE='null', N_ACTION_STEPS='8',
        BC_SAMPLER='ddim', BC_VERIFIER_SCORE='full', BC_BATCH_CANDIDATES='1', BC_GRIPPER_AUG='0',
        N_SAMPLES=str(n), MODE='argmax', TASK=TASKS[task], CONDA_ENV='simpler_env', MAX_STEPS='120',
        SAVE_VIDEOS='5', USE_EMA='1', OBS_CORRUPTION='as_trained', RAND_AUG_K='0', EVAL_CHUNK_SELECT='0',
        TILED_ENVS='1', SEEDS=','.join(map(str, seeds)), EVAL_OBS_NOISE_LEVEL='')
    if arm != 'n1':
        if arm not in ('veritas', 'veritas_distance', 'sim_veritas', 'sim_veritas_distance'):
            raise ValueError(arm)
        (out / 'plans').mkdir(exist_ok=True)
        (out / 'bon_trace').mkdir(exist_ok=True)
        env.update(VERITAS_PLAN_DIR=str(out / 'plans'), VERITAS_ALLOW_LIVE='1', BON_SCORE=arm,
                   BON_TRACE_DIR=str(out / 'bon_trace'))
    subprocess.run(['bash', str(REPO / 'scriptsv2/eval_search/eval_search.sh'), str(CKPTS[task]),
                    str(len(seeds)), str(out)], env=env, check=True, cwd=REPO)


if __name__ == '__main__':
    evaluate(sys.argv[1], sys.argv[2], int(sys.argv[3]))
