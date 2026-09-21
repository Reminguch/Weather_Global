#!/usr/bin/env python3
"""Measured, dependency-gated 2.8-degree cache and four-arm training pipeline."""
from pathlib import Path
import argparse
import fcntl
import json
import math
import os
import resource
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/neuralgcm'))
from src.models.neuralgcm_residual.numerics import configure_environment
configure_environment()
from src.models.neuralgcm_residual.io import read_json, write_json, sha256, digest
from src.models.neuralgcm_residual.launcher import load_experiment, submit, resolve_resources


def once(root, key, make_plan):
    """Persist submission intent before sbatch; never retry an ambiguous result."""
    path = root / 'pipeline' / 'jobs' / (key + '.json')
    if path.exists():
        previous = read_json(path)
        if not previous.get('job_id'):
            raise RuntimeError('Reconcile ambiguous submission before retrying: ' + str(path))
        return previous['job_id']
    plan = make_plan()
    write_json(path, dict(plan, submission_intent=time.time()), immutable=True)
    result = subprocess.run(plan['sbatch'], input=plan['script'], capture_output=True,
                            text=True, check=True, timeout=60)
    job = result.stdout.strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError('Unexpected scheduler response: ' + result.stdout)
    write_json(path, dict(plan, job_id=job, submitted_at=time.time()))
    print(json.dumps({'submitted': key, 'job_id': job}), flush=True)
    return job


def control(root, manifest, cfg, action, *, after=(), gpu=False, key=None,
            memory_gb=8, hours=1, extra=()):
    key = key or action
    command = [manifest['python'], str(Path(manifest['source_root']) / 'scripts/experiments/launch_neuralgcm_2p8_pipeline.py'),
               action, '--experiment-root', str(root), *map(str, extra)]
    cpus = 8 if gpu else 2
    batch = ['sbatch', '--parsable', '--job-name', 'ngcm2p8-' + key, '--cpus-per-task', str(cpus),
             '--mem', f'{memory_gb}G', '--time', f'{hours}:00:00',
             '--output', str(root / 'logs' / (key + '-%j.out')),
             '--error', str(root / 'logs' / (key + '-%j.err'))]
    if gpu:
        batch += ['--gres', 'gpu:1', '--constraint', cfg['constraint']]
    if gpu and action == 'bootstrap':
        if hours > 1:
            raise ValueError('GPU bootstrap smoke must fit one hour')
        batch += ['--qos', 'gpu-test']
    if after:
        batch += ['--dependency', 'afterok:' + ':'.join(after)]
    script = ('#!/usr/bin/env bash\nset -euo pipefail\nulimit -c 0\n'
              'unset PYTHONPATH PYTHONHOME\nexport PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1\n'
              'export XLA_PYTHON_CLIENT_PREALLOCATE=false\n'
              f'export JAX_PLATFORMS={"cuda" if gpu else "cpu"}\n'
              f'export OMP_NUM_THREADS={cpus} OPENBLAS_NUM_THREADS={cpus} MKL_NUM_THREADS={cpus}\n'
              'cd ' + shlex.quote(manifest['source_root']) + '\nexec ' + shlex.join(command) + '\n')
    return once(root, key, lambda: {'sbatch': batch, 'script': script, 'afterok': list(after),
                                   'source_id': manifest['source_id']})


def stage(root, cfg, key, name, *, after=(), memory_gb=32, hours=2, **kwargs):
    def plan():
        plans = submit(root, name, resolution='res2p8', dry_run=True, after=after,
                       memory_gb=memory_gb, hours=hours, constraint=cfg['constraint'], **kwargs)
        if len(plans) != 1:
            raise ValueError('One durable submission intent must describe exactly one job')
        return plans[0]
    return once(root, key, plan)


def check_data(manifest, cfg):
    verification = read_json(cfg['verification'])
    prepared = Path(manifest['resources']['res2p8']['prepared'])
    if (not verification['passed'] or verification['frames'] != 13148
            or verification['manifest_sha256'] != sha256(prepared / 'manifest.json')
            or verification['dataset_id'] != digest(read_json(prepared / 'manifest.json'))):
        raise ValueError('Full-data verification does not match this experiment')
    return verification


def bootstrap(root, manifest, cfg):
    check_data(manifest, cfg)
    resources = resolve_resources(manifest, 'res2p8')
    smoke = root / 'checks' / 'full_data_smoke'
    subprocess.run([manifest['python'], '-u', str(ROOT / 'scripts/training/smoke_neuralgcm_training.py'),
                    '--checkpoint', resources['checkpoint'], '--prepared', resources['prepared'],
                    '--output', str(smoke), '--resolution', '2.8'], check=True)
    report = read_json(smoke / 'report.json')
    if not report['passed'] or not report['frozen_parameters_unchanged'] or report['dataset_id'] != check_data(manifest, cfg)['dataset_id']:
        raise ValueError('Full-data GPU smoke did not pass')
    from src.models.neuralgcm_residual.worker import execute
    execute(root, manifest, 'preflight', resolution='res2p8', phase='inspect')
    execute(root, manifest, 'cache', resolution='res2p8', shard_id='plan')
    cache = read_json(Path(resources['cache_root']) / 'manifest.json')
    samples = []
    for sid in (cache['shards'][0]['id'], cache['shards'][-1]['id']):
        receipt = execute(root, manifest, 'cache', resolution='res2p8', shard_id=sid)
        samples.extend(receipt['result']['completed_shards'])
    import jax
    value = {'passed': True, 'source_id': manifest['source_id'], 'samples': samples,
             'records': cache['record_count'], 'shards': len(cache['shards']),
             'peak_cpu_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
             'device_kinds': [d.device_kind for d in jax.devices()],
             'gpu_measured': all(d.platform == 'gpu' for d in jax.devices())}
    write_json(root / 'checks' / 'cache_pilot.json', value, immutable=True)


def release_cache(root, manifest, cfg):
    pilot = read_json(root / 'checks/cache_pilot.json')
    if not pilot['passed'] or not pilot['gpu_measured'] or pilot['source_id'] != manifest['source_id']:
        raise ValueError('Missing measured cache pilot')
    per_record = max(s['seconds'] / s['records'] for s in pilot['samples'])
    bytes_per_record = max(s['stored_bytes'] / s['records'] for s in pilot['samples'])
    cache_root = Path(manifest['resources']['res2p8']['cache_root'])
    estimated_bytes = math.ceil(bytes_per_record * pilot['records'] * 1.3)
    if shutil.disk_usage(cache_root).free < estimated_bytes * 2:
        raise RuntimeError('Insufficient scratch headroom for cache and training artifacts')
    memory = max(16, math.ceil(pilot['peak_cpu_rss_bytes'] * 2 / 2**30))
    hours = max(1, math.ceil((per_record * 24 * math.ceil(pilot['shards'] / cfg['lanes']) * 2 + 900) / 3600))
    write_json(root / 'pipeline/cache_resources.json', dict(memory_gb=memory, hours=hours,
               seconds_per_record_upper=per_record, estimated_cache_bytes=estimated_bytes,
               lanes=cfg['lanes'], pilot_sha256=sha256(root / 'checks/cache_pilot.json')), immutable=True)
    jobs = [stage(root, cfg, f'cache-lane-{i}', 'cache', memory_gb=memory, hours=hours,
                  shard_id=f'lane-{i}-{cfg["lanes"]}') for i in range(cfg['lanes'])]
    # Statistics additionally encode every next truth state and stream all train frames.
    verify_hours = min(12, max(2, math.ceil((per_record * pilot['records'] * 3 + 3600) / 3600)))
    verify = stage(root, cfg, 'verify-cache', 'verify-cache', after=jobs,
                   memory_gb=max(32, memory), hours=verify_hours)
    numerical = stage(root, cfg, 'numerical', 'preflight', phase='numerical', after=[verify], memory_gb=64, hours=1)
    profile = stage(root, cfg, 'profile', 'preflight', phase='profile', after=[verify], memory_gb=64, hours=1)
    release = control(root, manifest, cfg, 'release-training', after=[numerical, profile])
    write_json(root / 'pipeline/cache_chain.json', dict(cache_jobs=jobs, verification=verify,
               numerical=numerical, profile=profile, training_release=release), immutable=True)


def training_resources(profile, cache):
    from src.models.neuralgcm_residual.data import complete_segments
    origins = [t for s in cache['shards'] if s['split'] == 'train' for t in s['origins']]
    val_records = sum(len(s['origins']) for s in cache['shards'] if s['split'] == 'val')
    segments, omitted = complete_segments(origins)
    updates = len(segments) * 4 * 20
    validation = 32 * sum(profile['validation20_seconds'].values())
    pretrain = (updates * max(profile['cached_update_seconds']) +
                21 * val_records * profile['one_step_validation_seconds'] + 5 * validation)
    finetune = 2000 * max(profile['live20_seconds']) * 1.2 + 10 * validation
    memory = max(32, math.ceil(profile['peak_cpu_rss_bytes'] * 2 / 2**30))
    return {'memory_gb': memory, 'pretrain_estimated_seconds': pretrain,
            'finetune_estimated_seconds': finetune, 'updates_per_arm': updates,
            'records_per_pass': len(segments) * 96, 'omitted_records': len(omitted),
            'pretrain_hours': min(24, max(2, math.ceil((pretrain * 2 + 1800) / 3600))),
            'finetune_hours': min(24, max(2, math.ceil((finetune * 2 + 1800) / 3600))),
            'walltime_policy': '24h maximum per job; explicit checkpoint resume until full budget completes'}


def release_training(root, manifest, cfg):
    from src.models.neuralgcm_residual.worker import require_receipt
    require_receipt(root, 'verify-cache', 'res2p8', manifest['source_id'])
    numerical = read_json(root / 'checks/numerical_res2p8.json')
    profile = read_json(root / 'checks/profile_res2p8.json')
    if (not numerical['passed'] or not profile['measured'] or not profile['gpu_measured']
            or any(r['identities']['source'] != manifest['source_id'] for r in (numerical, profile))):
        raise ValueError('Numerical/profile gates are incomplete')
    cache = read_json(Path(manifest['resources']['res2p8']['cache_root']) / 'manifest.json')
    resources = training_resources(profile, cache)
    resources['profile_sha256'] = sha256(root / 'checks/profile_res2p8.json')
    write_json(root / 'pipeline/training_resources.json', resources, immutable=True)
    jobs = {run: training_job(root, manifest, cfg, resources, run, 'pretrain', 0)
            for run in manifest['configs']}
    write_json(root / 'pipeline/training_chain.json', {'pretrain_jobs': jobs}, immutable=True)


def training_job(root, manifest, cfg, resources, run, stage_name, attempt, after=()):
    return control(root, manifest, cfg, 'train-slice', gpu=True, after=after,
                   key=f'{stage_name}-{run}-{attempt:03d}', memory_gb=resources['memory_gb'],
                   hours=resources[stage_name + '_hours'],
                   extra=('--run-id', run, '--training-stage', stage_name, '--attempt', attempt))


def latest_checkpoint(output, stage_name):
    candidates = []
    for path in output.glob('checkpoint_*.pkl'):
        meta = path.with_suffix('.json')
        if meta.exists():
            value = read_json(meta)
            if value['stage'] == stage_name and value['sha256'] == sha256(path):
                candidates.append((value['cursor']['update'], path))
    return max(candidates, key=lambda x: x[0]) if candidates else (0, None)


def train_slice(root, manifest, cfg, run, stage_name, attempt):
    resources = read_json(root / 'pipeline/training_resources.json')
    output = root / 'runs' / run / 'seed22' / stage_name
    previous_update, checkpoint = latest_checkpoint(output, stage_name)
    command = [manifest['python'], '-u', str(ROOT / 'scripts/experiments/run_neuralgcm_residual.py'),
               'execute', '--experiment-root', str(root), '--stage', stage_name, '--run-id', run]
    if checkpoint:
        command += ['--resume', str(checkpoint)]
    try:
        subprocess.run(command, check=True, timeout=resources[stage_name + '_hours'] * 3600 - 900)
    except subprocess.TimeoutExpired:
        next_update, saved = latest_checkpoint(output, stage_name)
        if saved is None or next_update <= previous_update:
            raise RuntimeError('Training slice made no checkpointed progress; inspect rather than resubmit')
        training_job(root, manifest, cfg, resources, run, stage_name, attempt + 1,
                     after=[os.environ['SLURM_JOB_ID']])
        return
    from src.models.neuralgcm_residual.worker import require_receipt
    require_receipt(root, stage_name, run, manifest['source_id'])
    if stage_name == 'pretrain':
        training_job(root, manifest, cfg, resources, run, 'finetune', 0,
                     after=[os.environ['SLURM_JOB_ID']])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('start', 'bootstrap', 'release-cache', 'release-training', 'train-slice'))
    p.add_argument('--experiment-root', required=True, type=Path)
    p.add_argument('--verification', type=Path)
    p.add_argument('--lanes', type=int, default=16)
    p.add_argument('--constraint', default='a100&gpu80')
    p.add_argument('--run-id')
    p.add_argument('--training-stage', choices=('pretrain', 'finetune'))
    p.add_argument('--attempt', type=int, default=0)
    a = p.parse_args()
    root = a.experiment_root.resolve()
    manifest = load_experiment(root)
    if set(manifest['resources']) != {'res2p8'}:
        raise ValueError('This launcher requires an explicit 2.8-degree-only experiment')
    if ROOT != Path(manifest['source_root']):
        os.execv(manifest['python'], [manifest['python'], str(Path(manifest['source_root']) /
                  'scripts/experiments/launch_neuralgcm_2p8_pipeline.py'), *sys.argv[1:]])
    (root / 'pipeline').mkdir(exist_ok=True)
    lock = (root / 'pipeline' / (a.action + '-' + str(a.run_id) + '.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config_path = root / 'pipeline/config.json'
    if a.action == 'start':
        if a.verification is None or not 1 <= a.lanes <= 32:
            p.error('start requires a full-data verification report and 1..32 cache lanes')
        cfg = dict(verification=str(a.verification.resolve()), lanes=a.lanes,
                   constraint=a.constraint, source_id=manifest['source_id'])
        write_json(config_path, cfg, immutable=True)
        check_data(manifest, cfg)
        pilot = control(root, manifest, cfg, 'bootstrap', gpu=True, memory_gb=32, hours=1)
        release = control(root, manifest, cfg, 'release-cache', after=[pilot])
        print(json.dumps({'bootstrap': pilot, 'cache_release': release}), flush=True)
        return
    cfg = read_json(config_path)
    if cfg['source_id'] != manifest['source_id']:
        raise ValueError('Pipeline source identity changed')
    actions = {'bootstrap': bootstrap, 'release-cache': release_cache, 'release-training': release_training}
    if a.action == 'train-slice':
        if a.run_id not in manifest['configs'] or a.training_stage is None:
            p.error('train-slice requires an active run and training stage')
        train_slice(root, manifest, cfg, a.run_id, a.training_stage, a.attempt)
    else:
        actions[a.action](root, manifest, cfg)


if __name__ == '__main__':
    main()
