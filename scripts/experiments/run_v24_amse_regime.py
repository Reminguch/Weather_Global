"""Bounded, frozen AMSE joint-RMSE/power campaign with measured Slurm gates.

Selection uses the exploratory eight starts; confirmation dates are frozen
before training and are not read by selection. All later jobs are submitted
only by successful, provenance-checked preceding stages.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import resource
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.frozen_runtime import freeze_source, verify_source, file_sha256

DEFAULT_EXPERIMENT = 'artifacts/checkpoints/v24_Ilya/res1_amse_joint_regime_20261004'
HISTORICAL = ('res1_amse_pair_20260919', 'res1_amse_blend_lambda0p1_20260924')
PILOT_ARMS = (('mse', None), ('spectral_mse', 0.), ('amse_l025', .25), ('amse_l050', .5))
LABELS = {'mse': 'MSE', 'spectral_mse': 'Spectral MSE control',
          'amse_l025': 'AMSE λ=0.25', 'amse_l050': 'AMSE λ=0.5'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def pinned(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': file_sha256(path)}


def check(record):
    if file_sha256(record['path']) != record['sha256']:
        raise ValueError(f"Pinned file changed: {record['path']}")


def call(args):
    args = list(map(str, args))
    print(shlex.join(args), flush=True)
    subprocess.run(args, check=True)


def confirmation_dates(pilot_dates, year=2023):
    """Sixteen dates per meteorological season, outside pilot forecast windows.

    Ten-day forecasts overlap within this confirmation sample; inference must
    use time blocks, never treat wavenumbers or adjacent dates as independent.
    """
    import numpy as np
    excluded = [datetime.fromisoformat(x.replace('Z', '+00:00')).date() for x in pilot_dates]
    seasons = ((12, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11))
    selected = []
    for months in seasons:
        start = datetime(year, 1, 2).date()
        end = datetime(year, 12, 22).date()
        candidates = []
        day = start
        while day < end:
            if day.month in months and all(abs((day - old).days) > 10 for old in excluded):
                candidates.append(day)
            day += timedelta(days=1)
        if len(candidates) < 16:
            raise ValueError('Insufficient confirmation dates outside pilot forecast windows')
        selected.extend(candidates[int(x)] for x in np.linspace(0, len(candidates) - 1, 16))
    if len(set(selected)) != 64:
        raise ValueError('Confirmation dates are duplicated')
    return [x.isoformat() + 'T00:00:00Z' for x in sorted(selected)]


def make_config(template, experiment, name, *, amse_lambda=None, steps=500, seed=22,
                bptt=24, feedback='closed_loop_sg', ramp=100, multipliers=None, degrees=None):
    config = copy.deepcopy(template)
    config['output'] = {'output_root': str(experiment / 'runs'), 'run_name': name}
    config['optimizer'].update(max_steps=steps, checkpoint_every=min(100, steps),
                               learning_rate=1e-5, end_learning_rate=1e-6,
                               warmup_steps=min(50, max(0, steps // 5)), seed=seed,
                               mamba_lr_multiplier=3., spatial_lr_multiplier=1.)
    config['sequence'].update(bptt_steps=bptt, ar_tail_k=20 if bptt == 24 else bptt - 1,
                              feedback_mode=feedback, segment_steps=96)
    config['memory']['bptt_backend'] = ('explicit_weather_reverse_vjp' if feedback == 'closed_loop'
                                         else 'explicit_reverse_vjp')
    config['objective'] = {'loss_mode': 'all_steps', 'loss_kind': 'mse_fp32'}
    if amse_lambda is not None:
        config['objective'].update(loss_kind='amse', amse_lmax=179, amse_lambda=amse_lambda,
                                    amse_power_epsilon=1e-12,
                                    amse_lambda_ramp_steps=ramp if amse_lambda else 0,
                                    amse_lambda_start=0.)
        if multipliers:
            config['objective']['amse_variable_lambda_multipliers'] = multipliers
        if degrees:
            config['objective']['amse_degree_lambda_multipliers'] = degrees
    config['validation']['enabled'] = False  # Common external exact/spectral protocol.
    return config


def config_record(experiment, name, config, init_from):
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    path = experiment / 'configs' / f'{name}.json'
    if path.exists() and read(path) != config:
        raise ValueError(f'Refusing to replace a campaign configuration: {path}')
    write(path, config)
    parsed = load_training_config(path)
    return {'name': name, 'config': pinned(path), 'run_dir': str(experiment / 'runs' / name),
            'init_from': str(init_from), 'steps': parsed.max_steps,
            'initial_checkpoint': pinned(init_from) if Path(init_from).is_file() else None,
            'checkpoint': str(experiment / 'runs' / name / 'checkpoints' /
                              f'checkpoint_step{parsed.max_steps:08d}.pkl')}


def evaluator_args(stage, manifest, *extra):
    return [sys.executable, '-u', ROOT / 'scripts/analyze_models/evaluate_v24_amse_regime.py',
            stage, '--manifest', manifest, *extra]


def prepare_eval(experiment, phase, tasks, *, dates=None):
    tasks_path = experiment / phase / 'tasks.json'
    write(tasks_path, tasks)
    args = [sys.executable, '-u', ROOT / 'scripts/analyze_models/evaluate_v24_amse_regime.py',
            'prepare', '--tasks-json', tasks_path, '--out-dir', experiment / phase / 'evaluation',
            '--reference-experiment', read(experiment / 'manifest.json')['reference_experiment'],
            '--workspace', read(experiment / 'manifest.json')['workspace'],
            '--mse-control-id', 'mature_mse', '--observer-factory',
            'src.models.mamba.v24_Ilya.spherical_observer:make_observer']
    if dates:
        args.extend(['--spectral-initializations', dates, '--role', 'confirmation'])
    call(args)
    return experiment / phase / 'evaluation' / 'manifest.json'


def control_task(manifest, *, reuse=True):
    init = manifest['initialization']
    task = {'id': 'mature_mse', 'label': 'MSE', 'role': 'mse_control', 'family': 'mature_mse',
            'step': 2000, 'checkpoint': init['checkpoint']['path'],
            'source_run_dir': init['source_run_dir']}
    if reuse:
        # Recompute spectra to obtain per-initialization RMSE and spherical
        # diagnostics needed by paired uncertainty and the degree ablation.
        task['reuse'] = {'exact': init['reuse']['exact']}
    return task


def evaluation_task(entry, step=None, *, family=None):
    step = entry['steps'] if step is None else step
    name = entry['name']
    return {'id': f'{name}_step{step:05d}', 'label': LABELS.get(name, name.replace('_', ' ')),
            'role': 'candidate', 'family': family or name, 'step': step,
            'checkpoint': str(Path(entry['run_dir']) / 'checkpoints' / f'checkpoint_step{step:08d}.pkl'),
            'source_run_dir': entry['run_dir'], 'planned_config': entry['config']['path']}


def prepare(experiment, workspace):
    if (experiment / 'manifest.json').exists():
        raise FileExistsError(f'Campaign already prepared: {experiment}')
    experiment.mkdir(parents=True, exist_ok=True)
    historical = [workspace / 'artifacts/checkpoints/v24_Ilya' / name for name in HISTORICAL]
    # Initialization selection is strictly 2022 exact32, never the 2023 spectra.
    from scripts.analyze_models.evaluate_v24_amse_regime import select_initialization
    selection = select_initialization(historical, experiment / 'initialization_selection.json')
    chosen = selection['selected']
    init = {'checkpoint': {'path': chosen['checkpoint'], 'sha256': chosen['checkpoint_sha256']},
            'source_run_dir': chosen['source_run_dir'], 'exact32_loss': chosen['exact32_rollout_loss'],
            'selection_record': pinned(experiment / 'initialization_selection.json'),
            'reuse': {stage: str(Path(chosen['experiment']) / 'eval' / stage / 'mse_fp32' / chosen['tag'])
                      for stage in ('exact', 'spectral')}}
    template = read(historical[1] / 'configs/mse_fp32.json')
    pilot_entries = []
    smoke_entries = []
    for name, value in PILOT_ARMS:
        config = make_config(template, experiment, name, amse_lambda=value)
        pilot_entries.append(config_record(experiment, name, config, init['checkpoint']['path']))
        smoke_name = 'smoke_' + name
        smoke = make_config(template, experiment, smoke_name, amse_lambda=value, steps=32, ramp=16)
        smoke_entries.append(config_record(experiment, smoke_name, smoke, init['checkpoint']['path']))
    for bptt in (2, 4, 8):
        name = f'smoke_feedback_b{bptt}'
        config = make_config(template, experiment, name, amse_lambda=.5, steps=8, bptt=bptt,
                             feedback='closed_loop', ramp=4)
        smoke_entries.append(config_record(experiment, name, config, init['checkpoint']['path']))
    pilot_protocol = read(workspace / 'plots/analyze_models/data/resolution_eval/res1_spectral_pilot_2023_20260917/protocol.json')
    dates = confirmation_dates(pilot_protocol['initialization_times'])
    write(experiment / 'confirmation_dates.json', {'initialization_times': dates,
                                                 'role': 'confirmation only; not a selection input',
                                                 'exclusion': 'More than ten days from each exploratory initialization',
                                                 'uncertainty': 'Use paired temporal-block bootstrap; forecast windows overlap'})
    manifest = {'format': 'v24_amse_joint_regime_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'workspace': str(workspace), 'experiment': str(experiment), 'source_root': str(experiment / 'source'),
                'reference_experiment': str(historical[0]), 'initialization': init, 'template': template,
                'pilot_entries': pilot_entries, 'smoke_entries': smoke_entries,
                'confirmation_dates': pinned(experiment / 'confirmation_dates.json'),
                'policy': {'pilot_steps': 500, 'checkpoint_every': 100, 'lambda_ramp_updates': 100,
                           'feedback_windows': [2, 4, 8], 'feedback_updates_per_window': 100,
                           'adaptive_updates': 500, 'confirmation_updates': 2000,
                           'confirmation_seeds': [22, 23, 24], 'confirmation_recipes': 1,
                           'maximum_phases': ['pilot', 'feedback', 'adaptive', 'confirmation'],
                           'no_forecast_blending': True, 'automatic_retries': 0},
                'resources': {'historical_training_jobs': ['14147422', '14147424', '14397797', '14397799'],
                              'historical_peak_rss_gib': 245602608 / 1024**2,
                              'historical_seconds_per_update': 29.,
                              'smoke_memory_gib': 256, 'feedback_smoke_memory_gib': 320, 'smoke_minutes': 90,
                              'historical_64_update_preflight_peak_rss_gib': 162924480 / 1024**2,
                              'eval_memory_gib': 16, 'eval_minutes': 150}}
    write(experiment / 'manifest.json', manifest)
    history_tasks = [control_task(manifest)]
    for campaign, arm, family, label in ((historical[1], 'mse_fp32', 'mse_trajectory', 'MSE'),
                                        (historical[0], 'amse', 'full_amse', 'Full AMSE'),
                                        (historical[1], 'amse', 'amse_lambda0p1', 'AMSE λ=0.1')):
        for step in (500, 1000, 1500, 2000):
            row = {'id': f'{family}_step{step:05d}', 'label': label, 'role': 'candidate',
                   'family': family, 'step': step, 'source_run_dir': str(campaign / 'runs' / arm),
                   'checkpoint': str(campaign / 'runs' / arm / 'checkpoints' / f'checkpoint_step{step:08d}.pkl')}
            if step == 2000:
                row['reuse'] = {'exact': str(campaign / 'eval/exact' / arm / 'step2000')}
            history_tasks.append(row)
        if family != 'mse_trajectory':
            history_tasks.append({'id': f'{family}_swa', 'label': label, 'role': 'candidate', 'family': family,
                                  'step': 2000, 'source_run_dir': str(campaign / 'runs' / arm),
                                  'checkpoint': str(campaign / 'runs' / arm / 'swa/swa_step00500-02000.pkl'),
                                  'reuse': {'exact': str(campaign / 'eval/exact' / arm / 'swa_step00500-02000')}})
    prepare_eval(experiment, 'trajectory', history_tasks)
    prepare_eval(experiment, 'pilot', [control_task(manifest)] + [evaluation_task(entry, step)
                 for entry in pilot_entries for step in (100, 200, 300, 400, 500)])
    source = freeze_source(workspace, experiment / 'source')
    manifest['source_digest'] = source['source_digest']
    manifest['prepared_evaluations'] = {phase: pinned(experiment / phase / 'evaluation/manifest.json')
                                        for phase in ('trajectory', 'pilot')}
    write(experiment / 'manifest.json', manifest)
    (experiment / 'manifest.sha256').write_text(file_sha256(experiment / 'manifest.json') + '\n')
    print(experiment / 'manifest.json')


def verify(experiment, require_snapshot=False):
    if file_sha256(experiment / 'manifest.json') != (experiment / 'manifest.sha256').read_text().strip():
        raise ValueError('Campaign manifest changed after preparation')
    manifest = read(experiment / 'manifest.json')
    source = verify_source(manifest['source_root'])
    if source['source_digest'] != manifest['source_digest']:
        raise ValueError('Campaign source differs')
    if require_snapshot and ROOT != Path(manifest['source_root']):
        raise ValueError('Execute from the campaign snapshot')
    for entry in manifest['pilot_entries'] + manifest['smoke_entries']:
        check(entry['config'])
    for record in (manifest['initialization']['checkpoint'], manifest['initialization']['selection_record'], manifest['confirmation_dates'],
                   *manifest['prepared_evaluations'].values()):
        check(record)
    from scripts.analyze_models.evaluate_v24_amse_regime import load_manifest
    for record in manifest['prepared_evaluations'].values():
        load_manifest(record['path'])
    selection = read(manifest['initialization']['selection_record']['path'])
    for candidate in selection['candidates']:
        for key in ('evaluation', 'specification', 'run_config'):
            check(candidate[key])
    return manifest


def submit_job(experiment, name, args):
    from scripts.experiments.run_v24_amse_pair import submit_job as submit_once
    directory = experiment / 'submissions'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f'{name}.lock').open('a') as lock:
        # Concurrent report/release retries must never create duplicate jobs.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return submit_once(experiment, name, args)


def sbatch(experiment, manifest, stage, *, phase='pilot', memory=8, minutes=20,
           gpu=False, array=None, dependency=None, task=None):
    if not 1 <= memory <= 512 or not 1 <= minutes <= 6 * 24 * 60:
        raise ValueError('Slurm request is outside the campaign resource bounds')
    if array is not None and (not array or '%' in str(array)):
        raise ValueError('Arrays must be nonempty and unthrottled')
    args = ['sbatch', '--parsable', '--account=dabanin', f"--chdir={manifest['workspace']}",
            '--cpus-per-task=8' if gpu else '--cpus-per-task=2', f'--mem={memory}G', f'--time={minutes}',
            f'--job-name=amse-{phase}-{stage}', f'--output={experiment}/logs/{phase}_{stage}_%A_%a.out',
            f'--error={experiment}/logs/{phase}_{stage}_%A_%a.err']
    if gpu:
        args += ['--gres=gpu:1', '--constraint=gpu80',
                 '--qos=' + ('gpu-short' if minutes <= 480 else 'gpu-medium' if minutes <= 4320 else 'gpu-long')]
    if array is not None:
        args.append(f'--array={array}')
    if dependency:
        args.append(f'--dependency=afterok:{dependency}')
    args += [str(Path(manifest['source_root']) / 'scripts/experiments/v24_amse_regime.slurm'),
             stage, str(experiment), phase]
    if task is not None:
        args.append(str(task))
    return args


def train_entry(experiment, manifest, entry):
    import numpy as np
    from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_training_checkpoint
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    check(entry['config'])
    config = load_training_config(Path(entry['config']['path']))
    if entry.get('initial_checkpoint'):
        check(entry['initial_checkpoint'])
    else:
        prior_name = Path(entry['init_from']).parent.parent.name
        prior = read(experiment / 'checks' / f'{prior_name}_complete.json')
        if prior['checkpoint']['path'] != entry['init_from'] or prior['source_digest'] != manifest['source_digest']:
            raise ValueError('Preceding curriculum checkpoint provenance differs')
        check(prior['checkpoint'])
    directory = Path(entry['run_dir'])
    proof = experiment / 'checks' / f"{entry['name']}_complete.json"
    if proof.exists():
        record = read(proof)
        check(record['checkpoint'])
        if record['source_digest'] != manifest['source_digest'] or record['config'] != entry['config']:
            raise ValueError('Completed training provenance differs')
        return record
    if directory.exists():
        raise FileExistsError(f'Partial run requires explicit diagnosis, not an automatic restart: {directory}')
    start = time.monotonic()
    call([sys.executable, '-c', 'import jax; assert any(d.platform == "gpu" for d in jax.devices()), "GPU allocation required"'])
    call([sys.executable, '-u', ROOT / 'scripts/training/train_v24_Ilya.py', '--config', entry['config']['path'],
          '--init-from', entry['init_from'], '--zero-state-on-init'])
    checkpoint = load_v24_Ilya_training_checkpoint(Path(entry['checkpoint']))
    records = [json.loads(x) for x in (directory / 'train_metrics.jsonl').read_text().splitlines()]
    if [x['step'] for x in records] != list(range(1, entry['steps'] + 1)):
        raise ValueError('Missing or duplicated training updates')
    if not all(math.isfinite(x[k]) for x in records for k in ('loss', 'gradient_norm', 'step_seconds')):
        raise ValueError('Nonfinite training metrics')
    if not any(x['gradient_norm'] > 0 for x in records):
        raise ValueError('All training gradients are zero')
    if checkpoint.completed_step != entry['steps'] or checkpoint.resolved_training_config != config.to_dict():
        raise ValueError('Training checkpoint configuration differs')
    if checkpoint.baseline_checkpoint_fingerprint != file_sha256(config.baseline_checkpoint):
        raise ValueError('Frozen GraphCast baseline fingerprint differs')
    if not all(np.isfinite(x).all() for layer in checkpoint.residual_params.values() for x in layer.values()):
        raise ValueError('Nonfinite trained parameters')
    if config.loss_kind == 'amse':
        from src.models.mamba.v24_Ilya.training.spectral_loss import scheduled_amse_lambda
        expected = [float(scheduled_amse_lambda(config.amse_lambda, config.amse_lambda_ramp_steps, x['step'] - 1,
                                                start=config.amse_lambda_start)) for x in records]
        np.testing.assert_allclose([x['amse_lambda'] for x in records], expected, atol=1e-7)
    elapsed = time.monotonic() - start
    record = {'passed': True, 'source_digest': manifest['source_digest'], 'config': entry['config'],
              'checkpoint': pinned(entry['checkpoint']), 'initial_checkpoint': pinned(entry['init_from']),
              'elapsed_seconds': elapsed, 'peak_rss_gib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2,
              'steady_p90_seconds': float(np.percentile([x.get('end_to_end_seconds', x['step_seconds']) for x in records[3:]], 90))}
    write(proof, record)
    return record


def resource_budget(manifest, reports, steps):
    peak = max(manifest['resources']['historical_peak_rss_gib'], *(x['peak_rss_gib'] for x in reports))
    memory = math.ceil(peak * 1.35 / 16) * 16
    per_step = max(manifest['resources']['historical_seconds_per_update'], *(x['steady_p90_seconds'] for x in reports))
    minutes = max(90, math.ceil((1200 + per_step * steps) * 1.35 / 1800) * 30)
    if minutes > 6 * 24 * 60 or memory > 512:
        raise ValueError('Measured resources exceed campaign bounds; revise the design')
    return memory, minutes


def schedule_evaluation(experiment, manifest, phase, dependency=None, *, confirmation=False):
    evaluation = read(experiment / phase / 'evaluation/manifest.json')
    n = len(evaluation['task_specs'])
    job = submit_job(experiment, phase + '_eval', sbatch(experiment, manifest, 'evaluate', phase=phase,
                     memory=16, minutes=480 if confirmation else 150, gpu=True, array=f'0-{n-1}', dependency=dependency))
    report = submit_job(experiment, phase + '_report', sbatch(experiment, manifest, 'report', phase=phase,
                        memory=12, minutes=30, dependency=job))
    return report


def submit(experiment):
    manifest = verify(experiment)
    (experiment / 'logs').mkdir(exist_ok=True)
    trajectory = schedule_evaluation(experiment, manifest, 'trajectory')
    smoke = submit_job(experiment, 'smoke', sbatch(experiment, manifest, 'smoke', memory=256, minutes=90,
                      gpu=True, array='0-3'))
    feedback_smoke = submit_job(experiment, 'feedback_smoke', sbatch(experiment, manifest, 'smoke', memory=320,
                               minutes=90, gpu=True, array='4-6'))
    release = submit_job(experiment, 'pilot_release', sbatch(experiment, manifest, 'release', dependency=smoke))
    write(experiment / 'status.json', {'status': 'submitted', 'trajectory_report_job': trajectory,
                                      'preflight_job': smoke, 'feedback_preflight_job': feedback_smoke,
                                      'pilot_release_job': release})


def release(experiment, phase):
    manifest = verify(experiment, require_snapshot=True)
    if phase != 'pilot':
        raise ValueError('Only the pilot uses the initial release gate')
    reports = [read(experiment / 'checks' / f"{x['name']}_complete.json") for x in manifest['smoke_entries'][:4]]
    for entry, report in zip(manifest['smoke_entries'][:4], reports, strict=True):
        if report['passed'] is not True or report['source_digest'] != manifest['source_digest']:
            raise ValueError('Preflight did not pass with the frozen source')
        if report['config'] != entry['config'] or report['checkpoint']['path'] != entry['checkpoint']:
            raise ValueError('Preflight report belongs to a different task')
        check(report['config'])
        check(report['checkpoint'])
    memory, minutes = resource_budget(manifest, reports[:4], 500)
    write(experiment / 'pilot/resources.json', {'memory_gib': memory, 'time_minutes': minutes, 'evidence': reports[:4]})
    train = submit_job(experiment, 'pilot_train', sbatch(experiment, manifest, 'train', memory=memory,
                      minutes=minutes, gpu=True, array='0-3'))
    schedule_evaluation(experiment, manifest, 'pilot', train)


def read_plan(experiment, phase):
    from scripts.experiments.run_v24_spectral_pilot import protocol_digest
    plan = read(experiment / phase / 'plan.json')
    if plan['plan_hash'] != protocol_digest({k: v for k, v in plan.items() if k != 'plan_hash'}):
        raise ValueError('Continuation plan changed')
    for record in plan.get('evidence', []):
        check(record)
    for entry in plan['entries']:
        for stage in entry.get('stages', [entry]):
            check(stage['config'])
    return plan


def launch_phase(experiment, manifest, phase, entries, tasks, evidence, decision, *, dates=None):
    """Write once, then submit an idempotent bounded continuation."""
    from scripts.experiments.run_v24_spectral_pilot import protocol_digest
    plan = {'phase': phase, 'entries': entries, 'evidence': evidence, 'decision': decision,
            'source_digest': manifest['source_digest']}
    plan['plan_hash'] = protocol_digest(plan)
    path = experiment / phase / 'plan.json'
    if path.exists():
        if read_plan(experiment, phase) != plan:
            raise ValueError('Continuation plan already exists with different decisions')
    else:
        write(path, plan)
    eval_path = experiment / phase / 'evaluation/manifest.json'
    if not eval_path.exists():
        prepare_eval(experiment, phase, tasks, dates=dates)
    else:
        from scripts.analyze_models.evaluate_v24_amse_regime import load_manifest
        load_manifest(eval_path)
        if read(experiment / phase / 'tasks.json') != tasks:
            raise ValueError('Continuation evaluation task list changed')
    needs_feedback = any(read(stage['config']['path'])['sequence']['feedback_mode'] == 'closed_loop'
                         for entry in entries for stage in entry.get('stages', [entry]))
    smoke_entries = manifest['smoke_entries'] if needs_feedback else manifest['smoke_entries'][:4]
    reports = [read(experiment / 'checks' / f"{x['name']}_complete.json") for x in smoke_entries]
    for entry, report in zip(smoke_entries, reports, strict=True):
        if (report.get('passed') is not True or report['source_digest'] != manifest['source_digest']
                or report['config'] != entry['config'] or report['checkpoint']['path'] != entry['checkpoint']):
            raise ValueError('Continuation preflight provenance differs')
        check(report['checkpoint'])
    steps = max(sum(stage['steps'] for stage in x.get('stages', [x])) for x in entries)
    memory, minutes = resource_budget(manifest, reports, steps)
    write(experiment / phase / 'resources.json', {'memory_gib': memory, 'time_minutes': minutes,
                                                'max_updates_per_task': steps, 'preflight_evidence': reports})
    train = submit_job(experiment, phase + '_train', sbatch(experiment, manifest, 'train', phase=phase,
                      memory=memory, minutes=minutes, gpu=True, array=f'0-{len(entries) - 1}'))
    schedule_evaluation(experiment, manifest, phase, train, confirmation=phase == 'confirmation')
    write(experiment / 'status.json', {'status': 'running', 'phase': phase, 'decision': decision,
                                      'training_job': train, 'confirmation_has_not_been_used_for_selection': True})


def amse_candidates(experiment, phases):
    """Rank positive-lambda recipes using development evidence only."""
    candidates = []
    for phase in phases:
        path = experiment / phase / 'evaluation/report/joint_selection.json'
        if not path.exists():
            continue
        report = read(path)
        if report['status'] != 'complete' or report['evaluation_role'] != 'development':
            raise ValueError('Recipe selection requires complete development evaluation')
        check(report['manifest'])
        for record in report['sources']:
            check(record)
        for row in report['records']:
            if row['role'] != 'candidate' or not row.get('planned_config'):
                continue
            check(row['planned_config_record'])
            spec_path = experiment / phase / 'evaluation/results' / row['id'] / 'spectral/specification.json'
            spec = read(spec_path)
            check({'path': spec['checkpoint'], 'sha256': spec['checkpoint_sha256']})
            config = read(row['planned_config'])
            if config['objective']['loss_kind'] == 'amse' and config['objective'].get('amse_lambda', 1) > 0:
                candidates.append({**row, 'phase': phase, 'config_value': config, 'report_evidence': pinned(path)})
    if not candidates:
        raise ValueError('No evaluated positive-lambda AMSE candidate')
    return sorted(candidates, key=lambda row: (
        not row['score']['eligible'], not row.get('pareto_front', False),
        -row['score']['joint_wins_vs_gc'], row['score']['mean_rmse_ratio_vs_gc'],
        row['score']['mean_relative_power_error_pct'], row['id']))


def degree_policy(experiment, selected):
    """Derive fixed degree weights from global spherical development coherence.

    Unitless coherence is pooled equally over nine primitive fields and the
    four declared leads. Low coherence degrees get weaker amplitude pressure;
    weights never remove the underlying spectral MSE supervision.
    """
    import numpy as np
    from scripts.analyze_models.report_v24_spectral_pilot import _time
    directory = experiment / selected['phase'] / 'evaluation/results' / selected['id'] / 'spectral/spherical'
    spec = read(directory.parent / 'specification.json')
    expected_dates = {_time(stamp) for stamp in spec['initialization_times']}
    files = sorted(directory.glob('*.npz'))
    if len(files) != 8:
        raise ValueError('Adaptive degree policy requires all eight global spherical diagnostics')
    records = []
    observed_dates = set()
    variables = ['T2m', 'U10', 'V10', 'MSLP', 'TP6', 'T850', 'Q700', 'U850', 'V850']
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata']))
            if metadata['model_id'] != selected['id'] or metadata['status'] != 'complete':
                raise ValueError('Spherical policy input identity differs')
            if metadata['checkpoint_sha256'] != file_sha256(selected['checkpoint']):
                raise ValueError('Spherical policy checkpoint differs')
            stamp = _time(metadata['initialization_time'])
            if metadata['protocol_hash'] != spec['protocol_hash'] or stamp not in expected_dates or stamp in observed_dates:
                raise ValueError('Spherical policy protocol/dates differ')
            observed_dates.add(stamp)
            np.testing.assert_array_equal(data['degree'], np.arange(180))
            np.testing.assert_array_equal(data['lead_hours'], spec['lead_hours'])
            feature_indices = [list(data['variables']).index(name) for name in variables]
            records.append({key: data[key][:, feature_indices, :].copy()
                            for key in ('truth_power', 'full_power', 'full_cross_power')})
    if observed_dates != expected_dates:
        raise ValueError('Spherical policy initialization coverage differs')
    pooled = {key: np.mean([record[key] for record in records], axis=0) for key in records[0]}
    power, truth, cross = (pooled[key][[3, 11, 27, 39], :9, :] for key in ('full_power', 'truth_power', 'full_cross_power'))
    coherence = np.divide(cross, np.sqrt(power * truth), out=np.zeros_like(cross), where=(power * truth) > 0)
    mean = np.clip(coherence, -1, 1).mean(axis=(0, 1))
    multipliers = .2 + .8 * np.clip((mean - .3) / .4, 0, 1)
    multipliers[0] = 0.  # Retain spectral MSE on the global mean.
    record = {'sources': [pinned(path) for path in files], 'selected_checkpoint': pinned(selected['checkpoint']),
              'mean_coherence_by_degree': mean.tolist(), 'lambda_multipliers': multipliers.tolist(),
              'rule': '0.2 below coherence0.3; linear taper to1 at0.7; degree0 multiplier0',
              'scope': 'Fixed validation-derived degree weights, nine primitive fields and days1/3/7/10',
              'interpretation': 'A hypothesis ablation; zero lambda means spectral MSE, not exact grid MSE.'}
    write(experiment / 'adaptive/degree_policy.json', record)
    return multipliers.tolist(), pinned(experiment / 'adaptive/degree_policy.json')


def prepare_feedback(experiment, manifest, selected):
    entries = []
    tasks = [control_task(manifest)]
    value = selected['config_value']['objective']['amse_lambda']
    for mode in ('closed_loop_sg', 'closed_loop'):
        name = 'feedback_sg' if mode == 'closed_loop_sg' else 'feedback_differentiable'
        stages, init = [], selected['checkpoint']
        for bptt in (2, 4, 8):
            stage_name = f'{name}_b{bptt}'
            config = make_config(manifest['template'], experiment, stage_name, amse_lambda=value,
                                 steps=100, bptt=bptt, feedback=mode, ramp=0)
            stage = config_record(experiment, stage_name, config, init)
            stages.append(stage)
            init = stage['checkpoint']
        entries.append({'name': name, 'stages': stages})
        tasks.append(evaluation_task(stages[-1], family=name))
    launch_phase(experiment, manifest, 'feedback', entries, tasks, [selected['report_evidence']],
                 {'trigger': 'No qualified AMSE recipe or coherence loss exceeds0.001',
                  'selected_development_candidate': selected['id'], 'initial_checkpoint': pinned(selected['checkpoint']),
                  'matched_comparison': 'Same starting checkpoint, lambda, batch4, seeds, windows, update counts and optimizer resets; only feedback derivatives differ'})


def prepare_adaptive(experiment, manifest, selected):
    degrees, evidence = degree_policy(experiment, selected)
    template = selected['config_value']
    value, bptt = template['objective']['amse_lambda'], template['sequence']['bptt_steps']
    mode = template['sequence']['feedback_mode']
    entries = []
    tasks = [control_task(manifest)]
    for name, variables, weights in (('adaptive_uniform', None, None),
                                      ('adaptive_precip', {'total_precipitation_6hr': 0.}, None),
                                      ('adaptive_degrees', None, degrees)):
        config = make_config(manifest['template'], experiment, name, amse_lambda=value,
                             steps=500, bptt=bptt, feedback=mode, ramp=0,
                             multipliers=variables, degrees=weights)
        entry = config_record(experiment, name, config, selected['checkpoint'])
        entries.append(entry)
        for step in (100, 300, 500):
            tasks.append(evaluation_task(entry, step))
    launch_phase(experiment, manifest, 'adaptive', entries, tasks,
                 [selected['report_evidence'], evidence],
                 {'selected_development_candidate': selected['id'], 'initial_checkpoint': pinned(selected['checkpoint']),
                  'comparison': 'Uniform continuation versus precipitation spectral-MSE protection versus fixed spherical-degree lambda taper; no forecast averaging'})


def prepare_confirmation(experiment, manifest, selected):
    config = selected['config_value']
    entries, tasks = [], [control_task(manifest, reuse=False)]
    for seed in manifest['policy']['confirmation_seeds']:
        for arm in ('mse', 'amse'):
            name = f'confirmation_{arm}_seed{seed}'
            value = config['objective']['amse_lambda'] if arm == 'amse' else None
            mode, final_bptt = config['sequence']['feedback_mode'], config['sequence']['bptt_steps']
            stages, init = [], manifest['initialization']['checkpoint']['path']
            # Short-window recipes are re-created from the common mature MSE
            # initialization for every seed and its MSE control.
            schedule = [(2, 100), (4, 100)] if final_bptt == 8 else []
            schedule += [(final_bptt, manifest['policy']['confirmation_updates'])]
            for index, (bptt, steps) in enumerate(schedule):
                stage_name = name if index == len(schedule) - 1 else f'{name}_b{bptt}'
                cfg = make_config(manifest['template'], experiment, stage_name, amse_lambda=value,
                                  steps=steps, seed=seed, bptt=bptt, feedback=mode,
                                  ramp=100 if index == 0 else 0,
                                  multipliers=config['objective'].get('amse_variable_lambda_multipliers') if arm == 'amse' else None,
                                  degrees=config['objective'].get('amse_degree_lambda_multipliers') if arm == 'amse' else None)
                stage = config_record(experiment, stage_name, cfg, init)
                stages.append(stage)
                init = stage['checkpoint']
            entries.append({'name': name, 'stages': stages})
            task = evaluation_task(stages[-1], family=f'confirmation_{arm}')
            task.update(recipe_id=f'confirmation_{arm}', seed=seed,
                        paired_mse_id=f'confirmation_mse_seed{seed}_step02000')
            tasks.append(task)
    launch_phase(experiment, manifest, 'confirmation', entries, tasks,
                 [selected['report_evidence'], manifest['confirmation_dates']],
                 {'selected_development_candidate': selected['id'], 'selected_recipe': config,
                  'qualified_on_development': selected['score']['eligible'],
                  'selection_locked_before_confirmation': True,
                  'protocol': 'Retrain chosen recipe and matched MSE from common mature weights, three fine-tuning seeds; score fixed final2000 checkpoint on64 fresh seasonal starts; no selection on confirmation results'},
                 dates=manifest['confirmation_dates']['path'])


def advance(experiment, phase):
    manifest = verify(experiment, require_snapshot=True)
    if phase not in manifest['policy']['maximum_phases']:
        raise ValueError('Unknown continuation phase')
    path = experiment / phase / 'evaluation/report/joint_selection.json'
    report = read(path)
    if report['status'] != 'complete':
        raise ValueError('Cannot advance an incomplete evaluation')
    check(report['manifest'])
    if phase != 'pilot':
        read_plan(experiment, phase)
    if phase == 'confirmation':
        if report['evaluation_role'] != 'confirmation':
            raise ValueError('Final report is not a confirmation evaluation')
        write(experiment / 'status.json', {'status': 'complete', 'confirmation_report': pinned(path),
                                          'meaning': 'Experiment workflow complete; scientific success depends on the reported paired metrics and uncertainty, not job completion'})
        return
    if report['evaluation_role'] != 'development':
        raise ValueError('Confirmation results cannot select or modify a recipe')
    phases = ['pilot'] if phase == 'pilot' else ['pilot', 'feedback'] if phase == 'feedback' else ['pilot', 'feedback', 'adaptive']
    selected = amse_candidates(experiment, phases)[0]
    if phase == 'pilot' and (not selected['score']['eligible'] or
                             selected['score']['mean_coherence_change_vs_mse'] < -.001):
        feedback_ready = all((experiment / 'checks' / f"{x['name']}_complete.json").exists()
                             for x in manifest['smoke_entries'][4:])
        if not feedback_ready:
            smoke_job = read(experiment / 'submissions/feedback_smoke.json')['job_id']
            gate = submit_job(experiment, 'feedback_gate', sbatch(experiment, manifest, 'advance', phase='pilot',
                              dependency=smoke_job))
            write(experiment / 'status.json', {'status': 'waiting_for_feedback_preflight', 'gate_job': gate})
            return
        prepare_feedback(experiment, manifest, selected)
    elif phase in ('pilot', 'feedback'):
        prepare_adaptive(experiment, manifest, selected)
    else:
        prepare_confirmation(experiment, manifest, selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'verify', 'submit', 'smoke', 'release', 'train', 'evaluate', 'report', 'advance'))
    parser.add_argument('--experiment', type=Path, default=ROOT / DEFAULT_EXPERIMENT)
    parser.add_argument('--workspace', type=Path, default=ROOT)
    parser.add_argument('--phase', default='pilot')
    parser.add_argument('--task', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)))
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    if args.stage == 'prepare':
        prepare(experiment, args.workspace.resolve())
    elif args.stage == 'verify':
        verify(experiment)
    elif args.stage == 'submit':
        submit(experiment)
    elif args.stage == 'release':
        release(experiment, args.phase)
    elif args.stage == 'smoke':
        manifest = verify(experiment, require_snapshot=True)
        train_entry(experiment, manifest, manifest['smoke_entries'][args.task])
    elif args.stage == 'train':
        manifest = verify(experiment, require_snapshot=True)
        entries = manifest['pilot_entries'] if args.phase == 'pilot' else read_plan(experiment, args.phase)['entries']
        for entry in entries[args.task].get('stages', [entries[args.task]]):
            train_entry(experiment, manifest, entry)
    elif args.stage == 'evaluate':
        verify(experiment, require_snapshot=True)
        call(evaluator_args('evaluate', experiment / args.phase / 'evaluation/manifest.json', '--task', args.task,
                            '--stage', 'spectral' if args.phase == 'confirmation' else 'both'))
    elif args.stage == 'report':
        verify(experiment, require_snapshot=True)
        call(evaluator_args('report', experiment / args.phase / 'evaluation/manifest.json'))
        call([sys.executable, '-u', ROOT / 'scripts/analyze_models/plot_v24_amse_regime.py',
              '--report', experiment / args.phase / 'evaluation/report/joint_selection.json',
              '--out-dir', Path(read(experiment / 'manifest.json')['workspace']) /
              'plots/analyze_models/images/resolution_eval' / experiment.name / args.phase])
        if args.phase != 'trajectory':
            advance(experiment, args.phase)
    else:
        advance(experiment, args.phase)


if __name__ == '__main__':
    main()
