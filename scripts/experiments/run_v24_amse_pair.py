"""Freeze, validate and submit the matched res1 full-field MSE/AMSE trial."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ARMS = ('mse_fp32', 'amse')
REFERENCE = 'artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908'
DEFAULT_EXPERIMENT = 'artifacts/checkpoints/v24_Ilya/res1_amse_pair_20260919'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def call(args):
    args = list(map(str, args))
    print(shlex.join(args), flush=True)
    subprocess.run(args, check=True)


def freeze(workspace, destination):
    destination.mkdir(parents=True, exist_ok=False)
    files = {}
    for folder in ('src', 'third_party/graphcast', 'scripts', 'tests/v24_Ilya'):
        for source in sorted((workspace / folder).rglob('*')):
            if not source.is_file() or source.suffix not in ('.py', '.sh', '.slurm'):
                continue
            relative = source.relative_to(workspace)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            files[str(relative)] = sha(target)
    write(destination / 'snapshot.json', {'files': files})
    return sha(destination / 'snapshot.json')


def prepare(experiment, workspace):
    if (experiment / 'manifest.json').exists():
        raise FileExistsError('Experiment already frozen')
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    template = read(workspace / REFERENCE / 'configs/di16_fast_mamba.json')
    start = workspace / REFERENCE / 'seeds/di16_fast_mamba_warmup/checkpoints/checkpoint_step00000500.pkl'
    start_record = read(workspace / REFERENCE / 'checks/start_di16_fast_mamba.json')
    if Path(start_record['checkpoint']).resolve() != start.resolve() or start_record['completed_step'] != 500:
        raise ValueError('Reference warm-up identity differs')
    for key in ('prepared_root', 'anchor_manifest_root', 'baseline_checkpoint', 'stats_dir'):
        template['data'][key] = str((workspace / template['data'][key]).resolve())
    entries = []
    for name in ARMS:
        config = copy.deepcopy(template)
        config['objective']['loss_kind'] = name
        config['output'] = {'output_root': str(experiment / 'runs'), 'run_name': name}
        config_path = experiment / 'configs' / f'{name}.json'
        write(config_path, config)
        load_training_config(config_path)
        smoke = copy.deepcopy(config)
        smoke['optimizer'].update(max_steps=64, checkpoint_every=64)
        smoke['output']['output_root'] = str(experiment / 'preflight')
        smoke_path = experiment / 'configs' / f'{name}_preflight.json'
        write(smoke_path, smoke)
        load_training_config(smoke_path)
        entries.append({'name': name, 'config': str(config_path), 'config_sha256': sha(config_path),
                        'preflight_config': str(smoke_path), 'preflight_config_sha256': sha(smoke_path),
                        'run_dir': str(experiment / 'runs' / name)})
    baseline = Path(template['data']['baseline_checkpoint'])
    baseline_sha256 = sha(baseline)
    source_digest = freeze(workspace, experiment / 'source')
    manifest = {
        'format': 'v24_res1_amse_pair_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'workspace': str(workspace), 'experiment': str(experiment), 'source_root': str(experiment / 'source'),
        'source_digest': source_digest, 'entries': entries,
        'init_from': str(start), 'init_sha256': sha(start), 'baseline_path': str(baseline),
        'baseline_sha256': baseline_sha256, 'reference_warmup': start_record,
        'primary_swa_steps': [500, 1000, 1500, 2000],
        'evaluation_reference': str(workspace / REFERENCE / 'runs/di16_fast_mamba/eval/cold_full_zero_exact_gc/matched32/swa_step00500-02000.json'),
        'spectral_protocol': str(workspace / 'plots/analyze_models/data/resolution_eval/res1_spectral_pilot_2023_20260917/protocol.json'),
        'comparison': 'Same starting residual weights, data, state policy, seed, optimizer and 2000-update budget. Full-field FP32 scoring in both arms; original legacy MSE ran inside BF16.',
        'preflight': {'updates': 64, 'memory_gib': 320, 'time_minutes': 120,
                      'schedule_note': 'Independent short measurement run; same warmup50, shortened cosine. Production restarts from the common500-update checkpoint.'},
        'historical_resources': {'training': ['13740084_6', '13807466_0', '13807466_1', '13807466_2'],
                                 'max_rss_gib': 228.8589, 'steady_seconds_approx': 29,
                                 'evaluation_job': '13740086_6', 'evaluation_seconds': 11187,
                                 'evaluation_rss_gib': 9.48},
    }
    write(experiment / 'manifest.json', manifest)
    print(experiment / 'manifest.json')


def verify(experiment, *, require_snapshot=False):
    m = read(experiment / 'manifest.json')
    source = Path(m['source_root'])
    if sha(source / 'snapshot.json') != m['source_digest']:
        raise ValueError('Source manifest changed')
    if require_snapshot and ROOT != source:
        raise ValueError('Execute from frozen source')
    for relative, expected in read(source / 'snapshot.json')['files'].items():
        if sha(source / relative) != expected:
            raise ValueError(f'Frozen source changed: {relative}')
    for path, expected in ((m['init_from'], m['init_sha256']), (m['baseline_path'], m['baseline_sha256'])):
        if sha(path) != expected:
            raise ValueError(f'Frozen input changed: {path}')
    for entry in m['entries']:
        for key in ('config', 'preflight_config'):
            if sha(entry[key]) != entry[key + '_sha256']:
                raise ValueError(f'Configuration changed: {entry[key]}')
    return m


def validate_training(directory, config_path, steps, manifest):
    import numpy as np
    from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_training_checkpoint
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    records = [json.loads(line) for line in (directory / 'train_metrics.jsonl').read_text().splitlines()]
    if [r['step'] for r in records] != list(range(1, steps + 1)):
        raise ValueError('Missing or duplicate training updates')
    if not all(math.isfinite(r[key]) for r in records for key in ('loss', 'gradient_norm', 'step_seconds')):
        raise ValueError('Non-finite training loss/gradient/timing')
    if not any(r['gradient_norm'] > 0 for r in records):
        raise ValueError('All gradients are zero')
    path = directory / f'checkpoints/checkpoint_step{steps:08d}.pkl'
    ckpt = load_v24_Ilya_training_checkpoint(path)
    if ckpt.completed_step != steps or ckpt.resolved_training_config != load_training_config(Path(config_path)).to_dict():
        raise ValueError('Checkpoint/config mismatch')
    if ckpt.baseline_checkpoint_fingerprint != manifest['baseline_sha256']:
        raise ValueError('Frozen baseline mismatch')
    if not all(np.isfinite(value).all() for module in ckpt.residual_params.values() for value in module.values()):
        raise ValueError('Non-finite checkpoint parameters')
    start = load_v24_Ilya_training_checkpoint(Path(manifest['init_from']))
    if not any(not np.array_equal(value, start.residual_params[module][name])
               for module, values in ckpt.residual_params.items() for name, value in values.items()):
        raise ValueError('Training did not change residual weights')
    return records, path


def preflight(experiment, task):
    m = verify(experiment, require_snapshot=True)
    entry = m['entries'][task]
    # Fail on CPU fallback before a costly model build.
    call([sys.executable, '-c', 'import jax; assert any(d.platform == "gpu" for d in jax.devices()), "GPU required"'])
    directory = experiment / 'preflight' / entry['name']
    if directory.exists():
        raise FileExistsError(f'Preflight output exists: {directory}')
    started = time.monotonic()
    call([sys.executable, '-u', ROOT / 'scripts/training/train_v24_Ilya.py', '--config',
          entry['preflight_config'], '--init-from', m['init_from']])
    elapsed = time.monotonic() - started
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2
    records, checkpoint = validate_training(directory, entry['preflight_config'], 64, m)
    call([sys.executable, '-u', ROOT / 'scripts/analyze_models/evaluate_v24_amse_pair.py',
          'smoke', '--experiment', experiment, '--task', task, '--checkpoint', checkpoint])
    roundtrip = experiment / 'checks' / f"{entry['name']}_roundtrip.json"
    if not roundtrip.is_file():
        raise ValueError('Checkpoint roundtrip report missing')
    import numpy as np
    seconds = [r.get('end_to_end_seconds', r['step_seconds']) for r in records[4:]]
    write(experiment / 'checks' / f"{entry['name']}_preflight.json", {
        'passed': True, 'source_digest': m['source_digest'], 'init_sha256': m['init_sha256'],
        'config_sha256': entry['config_sha256'], 'completed_steps': 64,
        'elapsed_seconds': elapsed, 'peak_rss_gib': peak,
        'steady_p90_seconds': float(np.percentile(seconds, 90)),
        'startup_seconds': max(0., elapsed - sum(seconds)),
        'checkpoint_sha256': sha(checkpoint), 'roundtrip_sha256': sha(roundtrip),
    })


def budget(report, history):
    if report.get('passed') is not True or report['completed_steps'] != 64:
        raise ValueError('Preflight is incomplete')
    for key in ('peak_rss_gib', 'steady_p90_seconds', 'elapsed_seconds'):
        if not math.isfinite(report[key]) or report[key] <= 0:
            raise ValueError(f'Invalid resource measurement: {key}')
    memory = math.ceil(max(report['peak_rss_gib'], history['max_rss_gib']) * 1.4 / 16) * 16
    seconds = (report['startup_seconds'] + report['steady_p90_seconds'] * 2000 + 1800) * 1.35
    minutes = max(62, math.ceil(seconds / 1800) * 30)
    if minutes > 6 * 24 * 60:
        raise ValueError('Measured training exceeds six days; split/resume design needed before launch')
    return {'memory_gib': memory, 'time_minutes': minutes}


def submit_job(experiment, record_name, args):
    path = experiment / 'submissions' / f'{record_name}.json'
    if path.exists():
        record = read(path)
        if record['command'] != list(map(str, args)):
            raise ValueError('Existing submission command differs')
        if not record.get('job_id'):
            raise RuntimeError(f'Unresolved submission intent; check Slurm before retry: {path}')
        return record['job_id']
    record = {'command': list(map(str, args)), 'created_utc': datetime.now(timezone.utc).isoformat()}
    write(path, record)
    print(shlex.join(record['command']), flush=True)
    result = subprocess.run(record['command'], check=True, text=True, capture_output=True)
    job = result.stdout.strip().split(';')[0]
    if not job.isdigit():
        raise ValueError(f'Unexpected sbatch response: {result.stdout}')
    record.update(job_id=job, stderr=result.stderr)
    write(path, record)
    return job


def sbatch(experiment, manifest, stage, *, memory, minutes, gpu=True, dependency=None, task=None, array=None):
    args = ['sbatch', '--parsable', '--account=dabanin', f"--chdir={manifest['workspace']}", '--cpus-per-task=8' if gpu else '--cpus-per-task=2',
            f'--mem={memory}G', f'--time={minutes}', f'--job-name=v24-amse-{stage}',
            f'--output={experiment}/logs/{stage}_%A_%a.out', f'--error={experiment}/logs/{stage}_%A_%a.err']
    if gpu:
        args += ['--gres=gpu:1', '--constraint=gpu80', '--qos=' + ('gpu-short' if minutes <= 480 else 'gpu-medium' if minutes <= 4320 else 'gpu-long')]
    if dependency:
        args += [f'--dependency=afterok:{dependency}']
    if array is not None:
        args += [f'--array={array}']
    args += [str(Path(manifest['source_root']) / 'scripts/experiments/v24_amse_pair.slurm'), stage, str(experiment)]
    if task is not None:
        args.append(str(task))
    return args


def submit_preflight(experiment):
    m = verify(experiment)
    (experiment / 'logs').mkdir(exist_ok=True)
    job = submit_job(experiment, 'preflight', sbatch(experiment, m, 'preflight', memory=320, minutes=120, array='0-1'))
    release = submit_job(experiment, 'release', sbatch(experiment, m, 'release', memory=4, minutes=15, gpu=False, dependency=job))
    print(json.dumps({'preflight': job, 'release': release}))


def release(experiment):
    m = verify(experiment, require_snapshot=True)
    evaluations = []
    for task, entry in enumerate(m['entries']):
        report = read(experiment / 'checks' / f"{entry['name']}_preflight.json")
        if any(report[key] != expected for key, expected in (
            ('source_digest', m['source_digest']), ('init_sha256', m['init_sha256']), ('config_sha256', entry['config_sha256']))):
            raise ValueError('Preflight provenance mismatch')
        roundtrip = experiment / 'checks' / f"{entry['name']}_roundtrip.json"
        if sha(roundtrip) != report['roundtrip_sha256']:
            raise ValueError('Roundtrip validation changed')
        resources = budget(report, m['historical_resources'])
        write(experiment / 'checks' / f"{entry['name']}_budget.json", resources)
        train = submit_job(experiment, f"train_{entry['name']}", sbatch(experiment, m, 'train',
                           memory=resources['memory_gib'], minutes=resources['time_minutes'], task=task))
        evaluation = submit_job(experiment, f"eval_{entry['name']}", sbatch(experiment, m, 'evaluate',
                                memory=16, minutes=240, dependency=train, task=task))
        evaluations.append(evaluation)
    submit_job(experiment, 'report', sbatch(experiment, m, 'report', memory=8, minutes=30,
                                          gpu=False, dependency=':'.join(evaluations)))


def train(experiment, task):
    m = verify(experiment, require_snapshot=True)
    entry = m['entries'][task]
    if read(experiment / 'checks' / f"{entry['name']}_preflight.json").get('passed') is not True:
        raise ValueError('Preflight missing')
    if Path(entry['run_dir']).exists():
        raise FileExistsError('Production run already exists; exact resume must be explicit')
    call([sys.executable, '-u', ROOT / 'scripts/training/train_v24_Ilya.py', '--config', entry['config'],
          '--init-from', m['init_from']])
    _, checkpoint = validate_training(Path(entry['run_dir']), entry['config'], 2000, m)
    write(experiment / 'checks' / f"{entry['name']}_completed.json", {'passed': True, 'checkpoint': str(checkpoint), 'sha256': sha(checkpoint)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'verify', 'submit', 'preflight', 'release', 'train'))
    parser.add_argument('--experiment', type=Path, default=ROOT / DEFAULT_EXPERIMENT)
    parser.add_argument('--workspace', type=Path, default=ROOT)
    parser.add_argument('--task', type=int, choices=(0, 1), default=int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)))
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    if args.stage == 'prepare':
        prepare(experiment, args.workspace.resolve())
    elif args.stage == 'verify':
        verify(experiment)
    elif args.stage == 'submit':
        submit_preflight(experiment)
    elif args.stage == 'release':
        release(experiment)
    else:
        {'preflight': preflight, 'train': train}[args.stage](experiment, args.task)


if __name__ == '__main__':
    main()
