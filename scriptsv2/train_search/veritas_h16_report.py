#!/usr/bin/env python3
"""Write campaign train/eval status for the long-running Slurm report job."""
import json
from pathlib import Path
import subprocess
import time
import os
import sys
import csv
import statistics


def states(job_ids):
    if not job_ids:
        return {}
    raw = subprocess.check_output(
        ['sacct', '-n', '-X', '-P', '-j', ','.join(map(str, job_ids)),
         '--format=JobID,State,Elapsed'], text=True)
    result = {}
    for row in raw.splitlines():
        fields = row.split('|')
        if len(fields) >= 3:
            result[fields[0]] = (fields[1].split()[0], fields[2])
    for job in map(str, job_ids):
        children = [v for k, v in result.items() if k.startswith(job + '_')]
        if children and job not in result:
            names = {v[0] for v in children}
            state = next((s for s in ('RUNNING', 'PENDING', 'FAILED', 'OUT_OF_MEMORY',
                                     'TIMEOUT', 'CANCELLED') if s in names), 'COMPLETED')
            result[job] = (state, '')
    return result


def snapshot(root, wb_run=None):
    campaign = json.loads((root / 'campaign.json').read_text())
    submitted = json.loads((root / 'evaluations.json').read_text()) if (root / 'evaluations.json').exists() else {}
    extra = {}
    for filename in ('extended_evaluations.json', 'diagnostic_evaluations.json', 'clean_evaluations.json'):
        path = root / filename
        if path.exists():
            extra.update({filename + ':' + k: v for k, v in json.loads(path.read_text()).items()})
    ids = list(campaign['prep'].values()) + [x['job'] for x in campaign['runs'].values()]
    ids += [x['job'] for x in submitted.values()]
    ids += [x['job'] for x in extra.values()]
    status = states(ids)
    lines = ['# VERITAS H16 campaign progress', '',
             f'Updated: {time.strftime("%Y-%m-%d %H:%M:%S %Z")}',
             'Horizons: predict 16 / score 8 / execute 8', '',
             '| Run | Job state | Steps | 10k eval sweeps submitted |',
             '|---|---|---:|---:|']
    all_training_done = True
    metrics = {'campaign/eval_arrays_submitted': len(submitted)}
    eval_rows = []
    paths = [p for group in ('eval', 'eval_clean', 'eval_repaired',
                              'diagnostics_as_trained', 'diagnostics_clean', 'diagnostics_repaired')
             for p in (root / group).glob('*/*/*/*/eval_log.json')]
    for path in sorted(paths):
        try:
            record = json.loads(path.read_text())
            group = path.relative_to(root).parts[0]
            rel = path.relative_to(root / group).parts
            run, step, mode, n = rel[0], rel[1], rel[2], rel[3]
            rate = float(record['success_rate'])
            eval_rows.append((group, run, step, mode, n, rate, int(record.get('num_successes', round(rate * record['num_episodes']))), int(record['num_episodes'])))
            prefix = 'eval_result' if group == 'eval' else group
            key = f'{prefix}/{run}/{step}/{mode}/{n}'
            metrics[key] = rate
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
    for name, spec in campaign['runs'].items():
        ck = root / 'train' / name / 'checkpoints'
        steps = sorted(int(p.stem.split('_')[1]) for p in ck.glob('step_*.ckpt'))
        latest = max(steps, default=0)
        log = root / 'train' / name / 'logs.json.txt'
        if log.exists():
            with log.open('rb') as f:
                f.seek(max(0, log.stat().st_size - 32768))
                for line in reversed(f.read().splitlines()):
                    try:
                        record = json.loads(line)
                        # BC logs the update's zero-based index; ST logs the count.
                        latest = max(latest, int(record['global_step']) + int(spec['kind'] == 'bc'))
                        break
                    except (ValueError, KeyError, json.JSONDecodeError):
                        continue
        state = status.get(str(spec['job']), ('unknown', ''))[0]
        all_training_done &= state in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY')
        n_eval = sum(f'{name}_{s}' in submitted for s in range(10000, spec.get('max_gradient_steps', 50000) + 1, 10000))
        metrics[f'train/{name}/global_step'] = latest
        metrics[f'train/{name}/state_code'] = float(state == 'RUNNING')
        metrics[f'train/{name}/eval_sweeps_submitted'] = n_eval
        lines.append(f'| {name} | {state} | {latest:,} | {n_eval}/{spec.get("max_gradient_steps", 50000) // 10000} |')
    lines += ['', f'Preparation jobs: eggplant {status.get(str(campaign["prep"]["eggplant"]), ("unknown",))[0]}, '
              f'stack {status.get(str(campaign["prep"]["stack"]), ("unknown",))[0]}',
              f'Evaluation array jobs submitted: {len(submitted)}/{sum(s.get("max_gradient_steps", 50000) // 10000 for s in campaign["runs"].values())}', '',
              '| Evaluation array | State |', '|---|---|']
    all_eval_done = len(submitted) == sum(s.get('max_gradient_steps', 50000) // 10000 for s in campaign['runs'].values())
    all_eval_done &= sum(k.startswith('extended_evaluations.json:') for k in extra) == sum(s.get('max_gradient_steps', 50000) // 10000 for s in campaign['runs'].values() if s['kind'] in ('bc', 'uniform'))
    if (root / 'clean_eval_watcher.json').exists():
        expected_clean = sum(s.get('max_gradient_steps', 50000) // 10000
                             for s in campaign['runs'].values() if s['kind'] in ('flat10', 'ramp2to10'))
        all_eval_done &= sum(k.startswith('clean_evaluations.json:') for k in extra) == expected_clean
    for key, value in sorted({**submitted, **extra}.items()):
        eval_state = status.get(str(value['job']), ('unknown',))[0]
        metrics[f'eval/{key}/state_code'] = float(eval_state == 'COMPLETED')
        lines.append(f'| {key} ({value["job"]}) | {eval_state} |')
        all_eval_done &= eval_state in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY')
    lines += ['', 'Completed sweep results (successes / episodes):', '',
              '| Condition | Run | Step | Selection | N | Success |', '|---|---|---:|---|---:|---:|']
    for group, run, step, mode, n, rate, successes, episodes in sorted(eval_rows):
        lines.append(f'| {group} | {run} | {step.replace("step_", "")} | {mode} | {n[1:] if n.startswith("n") else n} | {successes}/{episodes} ({rate:.1%}) |')
    lines.append(f'\nCompleted sweep cells: {len(eval_rows)}')
    # User's comparison rule: choose checkpoints independently. BC is selected at
    # N=1; search can choose N and selection mode from its completed sweep.
    selected = []
    lines += ['', 'Best observed sweep scores (checkpoint selection; not a fresh holdout):', '',
              '| Task | Demos | Model | Step | Selection | N | Success |',
              '|---|---:|---|---:|---|---:|---:|']
    for task in ('eggplant', 'stack'):
        for demos in (50, 100):
            for model in ('bc', 'search'):
                eligible = [r for r in eval_rows if r[0] == 'eval'
                            and r[1].startswith(f'{task}_{demos}_')
                            and (('_bc_' in r[1] and int(r[4].lstrip('n')) == 1)
                                 if model == 'bc' else '_bc_' not in r[1])]
                if not eligible:
                    continue
                # Resolve ties deterministically: fewer candidates, earlier step.
                best = max(eligible, key=lambda r: (r[5], -int(r[4].lstrip('n')),
                                                    -int(r[2].replace('step_', ''))))
                _, run, step, mode, n, rate, successes, episodes = best
                record = dict(task=task, demos=demos, model=model, run=run, step=step,
                              selection=mode, n=int(n.lstrip('n')), success_rate=rate,
                              successes=successes, episodes=episodes)
                selected.append(record)
                lines.append(f'| {task} | {demos} | {model} | {step} | {mode} | '
                             f'{record["n"]} | {successes}/{episodes} ({rate:.1%}) |')
                metrics[f'selected/{task}/{demos}/{model}/success_rate'] = rate
    scaling = []
    lines += ['', 'BC verifier scaling at the checkpoint selected by N=1:', '',
              '| Task | Demos | BC step | N | BC successes | Best ST argmax | Best ST final |',
              '|---|---:|---|---:|---:|---:|---:|']
    for baseline in (r for r in selected if r['model'] == 'bc'):
        task, demos = baseline['task'], baseline['demos']
        for n in (1, 2, 4, 8, 16, 32, 64):
            bc = [r for r in eval_rows if r[0] == 'eval' and r[1] == baseline['run']
                  and r[2] == baseline['step'] and r[3] == 'argmax' and int(r[4].lstrip('n')) == n]
            if not bc:
                continue
            record = dict(task=task, demos=demos, bc_step=baseline['step'], n=n,
                          bc_successes=bc[0][6], bc_success_rate=bc[0][5], episodes=bc[0][7])
            for mode in ('argmax', 'final'):
                candidates = [r for r in eval_rows if r[0] == 'eval'
                              and r[1].startswith(f'{task}_{demos}_') and '_bc_' not in r[1]
                              and r[3] == mode and int(r[4].lstrip('n')) == n]
                best = max(candidates, key=lambda r: (r[5], -int(r[2].replace('step_', '')))) if candidates else None
                record[f'st_{mode}_successes'] = best[6] if best else None
                record[f'st_{mode}_step'] = best[2] if best else None
                record[f'st_{mode}_run'] = best[1] if best else None
            scaling.append(record)
            lines.append(f'| {task} | {demos} | {baseline["step"]} | {n} | {record["bc_successes"]}/50 | '
                         f'{record["st_argmax_successes"]} | {record["st_final_successes"]} |')
    temp = root / 'selected_best_sweep.json.tmp'
    temp.write_text(json.dumps(dict(rule='BC: best checkpoint at N=1; search: best checkpoint, N and selection in sweep',
                                    selected=selected, bc_fixed_checkpoint_scaling=scaling), indent=2) + '\n')
    temp.replace(root / 'selected_best_sweep.json')
    diagnostics = []
    for path in sorted(root.glob('diagnostics_*/*/*/*/*/candidate_diversity.json')):
        try:
            rows = json.loads(path.read_text())
            if not rows:
                continue
            group, name = path.relative_to(root).parts[:2]
            record = dict(condition=group, run=name, episodes=len(rows),
                          successes=sum(int(x['success']) for x in rows))
            for field in ('pairwise_translation_rmse_m', 'pairwise_endpoint_distance_m',
                          'gripper_sign_disagreement', 'final_minus_first_value'):
                value = statistics.mean(x['first_decision'][field] for x in rows)
                record[field] = value
                metrics[f'diagnostic/{group}/{name}/{field}'] = value
            sensitivity = path.with_name('feedback_sensitivity.json')
            for condition in ('clean', 'noisy'):
                value = None
                if sensitivity.exists():
                    values = json.loads(sensitivity.read_text())
                    value = statistics.mean(x[condition]['normalized_action_rmse'] for x in values)
                    metrics[f'diagnostic/{group}/{name}/feedback_sensitivity_{condition}'] = value
                record[f'feedback_sensitivity_{condition}'] = value
            diagnostics.append(record)
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
    if diagnostics:
        with (root / 'diagnostics_summary.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(diagnostics[0]))
            writer.writeheader()
            writer.writerows(diagnostics)
        lines += ['', 'Matched first-decision candidate diversity (five-episode diagnostics; not the full sweep):', '',
                  '| Condition | Run | Episodes | Endpoint spread (cm) | Feedback sensitivity clean / noisy |',
                  '|---|---|---:|---:|---:|']
        for row in diagnostics:
            a, b = row['feedback_sensitivity_clean'], row['feedback_sensitivity_noisy']
            sensitivity = 'pending' if a is None else f'{a:.4f} / {b:.4f}'
            lines.append(f'| {row["condition"]} | {row["run"]} | {row["episodes"]} | '
                         f'{100 * row["pairwise_endpoint_distance_m"]:.2f} | {sensitivity} |')
    tmp = root / 'progress.md.tmp'
    tmp.write_text('\n'.join(lines) + '\n')
    tmp.replace(root / 'progress.md')
    if wb_run is not None:
        wb_run.log(metrics)
    return all_training_done and all_eval_done


def main():
    root = Path(sys.argv[1])
    import wandb
    campaign = json.loads((root / 'campaign.json').read_text())
    wb_run = wandb.init(entity='l2sml', project='simpler_veritas_h16',
                        name='h16_campaign_progress', id=campaign.get('monitor_run_id', 'h16_campaign_20261002_progress'),
                        group='veritas_h16_s8_e8', tags=['simpler_veritas', 'progress'],
                        resume='allow')
    try:
        while True:
            done = snapshot(root, wb_run)
            if done:
                return
            time.sleep(10 * 60)
    finally:
        wb_run.finish()


if __name__ == '__main__':
    main()
