#!/usr/bin/env python3
"""Run frozen training with an explicit, content-bound user startup policy.

The model, optimizers, validation, checkpointing and numerical execution policy
come from the unchanged experiment snapshot. Only the requirement to wait for
additional full-statistics numerical/profile reports is replaced here. Full
cache verification and full training statistics are still mandatory.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys


def read(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_policy(root, expected_hash, runner_path=None):
    root = Path(root).resolve()
    path = root / 'pipeline/authorized_training_policy.json'
    if file_hash(path) != expected_hash:
        raise ValueError('Authorized startup policy changed')
    policy, manifest = read(path), read(root / 'manifest.json')
    runner_path = Path(runner_path or __file__).resolve()
    if (policy['schema'] != 'neuralgcm_user_authorized_start_v1'
            or policy['experiment_root'] != str(root)
            or policy['source_id'] != manifest['source_id']
            or policy['runner']['path'] != str(runner_path)
            or policy['runner']['sha256'] != file_hash(runner_path)
            or not policy['authorization_text']):
        raise ValueError('Startup authorization identity differs')
    if set(policy['evidence']) != set(manifest['configs']):
        raise ValueError('Startup evidence must cover every configured run')
    for run, evidence in policy['evidence'].items():
        for kind in ('detailed', 'resume'):
            entry = evidence[kind]
            report = read(entry['path'])
            if (file_hash(entry['path']) != entry['sha256']
                    or not report['passed']
                    or report['source_id'] != manifest['source_id']
                    or report['config_sha256'] != manifest['configs'][run]['sha256']
                    or report['run_id'] != run):
                raise ValueError('Completed smoke evidence changed: ' + run + '/' + kind)
            if kind == 'detailed' and (len(report['checks']) != 15 or
                    not all(c['passed'] for c in report['checks'].values())):
                raise ValueError('Incomplete detailed smoke evidence')
            if kind == 'resume' and not all(report[k] for k in (
                    'exact_checkpoint_equality', 'exact_metrics_equality',
                    'independent_processes', 'interrupted_log_recovery')):
                raise ValueError('Incomplete exact process-resume evidence')
    if (policy['constraint'] != 'a100&gpu80&pcie'
            or policy['resources'] != dict(memory_gb=64, pretrain_hours=24, finetune_hours=24)):
        raise ValueError('Unexpected authorized training reservation')
    return policy, manifest


def load_backend(manifest):
    source = Path(manifest['source_root'])
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / 'third_party/neuralgcm'))
    spec = importlib.util.spec_from_file_location('frozen_training_pipeline',
            source / 'scripts/experiments/launch_neuralgcm_2p8_pipeline.py')
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    return backend


def data_gate(root, manifest, resolution, runtime, worker, backend):
    receipt = worker.require_receipt(root, 'verify-cache', resolution, manifest['source_id'])
    resources = backend.resolve_resources(manifest, resolution)
    stats_path = Path(resources['statistics'])
    statistics = read(stats_path)
    cache_root = Path(resources['cache_root'])
    cache = read(cache_root / 'manifest.json')
    origins = [t for shard in cache['shards'] if shard['split'] == 'train'
               for t in shard['origins']]
    if (receipt['result']['statistics_sha256'] != file_hash(stats_path)
            or receipt['result']['ready_sha256'] != file_hash(cache_root / 'READY.json')
            or receipt['result']['producer_id'] != cache['producer_id']
            or statistics['origin_count'] != len(origins)
            or statistics['origins_sha256'] != backend.digest(origins)
            or statistics['split'] != 'train'
            or statistics['dataset_id'] != runtime.store.identity
            or statistics['native_schema_id'] != runtime.adapter.identity
            or backend.digest(statistics) != runtime.normalization.identity):
        raise ValueError('Full production data/statistics do not match runtime')
    return receipt


def job_plan(root, manifest, policy, policy_hash, run, stage, attempt, after):
    key = f'authorized-{stage}-{run}-{attempt:03d}'
    command = [manifest['python'], policy['runner']['path'], 'train-slice',
               '--experiment-root', str(root), '--policy-sha256', policy_hash,
               '--run-id', run, '--training-stage', stage, '--attempt', str(attempt)]
    batch = ['sbatch', '--parsable', '--job-name', 'ngcm2p8-' + key,
             '--cpus-per-task', '8', '--mem', '64G', '--time', '24:00:00',
             '--gres', 'gpu:1', '--constraint', policy['constraint'], '--qos', 'gpu-short',
             '--output', str(root / 'logs' / (key + '-%j.out')),
             '--error', str(root / 'logs' / (key + '-%j.err'))]
    if after:
        batch += ['--dependency', 'afterok:' + ':'.join(after)]
    script = ('#!/usr/bin/env bash\nset -euo pipefail\nulimit -c 0\n'
              'unset PYTHONPATH PYTHONHOME\n'
              'export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1\n'
              'export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda\n'
              'export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8\n'
              'cd ' + shlex.quote(manifest['source_root']) + '\nexec ' + shlex.join(command) + '\n')
    return key, dict(sbatch=batch, script=script, afterok=list(after),
                     source_id=manifest['source_id'], startup_policy_sha256=policy_hash,
                     operational_runner_sha256=policy['runner']['sha256'])


def submit_training(root, manifest, policy, policy_hash, backend, run, stage, attempt, after):
    key, plan = job_plan(root, manifest, policy, policy_hash, run, stage, attempt, after)
    return backend.once(root, key, lambda: plan)


def train_slice(root, manifest, policy, policy_hash, backend, run, stage, attempt):
    output = root / 'runs' / run / 'seed22' / stage
    previous_update, checkpoint = backend.latest_checkpoint(output, stage)
    command = [manifest['python'], '-u', policy['runner']['path'], 'execute',
               '--experiment-root', str(root), '--policy-sha256', policy_hash,
               '--run-id', run, '--training-stage', stage]
    if checkpoint:
        command += ['--resume', str(checkpoint)]
    try:
        subprocess.run(command, check=True,
                       timeout=policy['resources'][stage + '_hours'] * 3600 - 900)
    except subprocess.TimeoutExpired:
        next_update, saved = backend.latest_checkpoint(output, stage)
        if saved is None or next_update <= previous_update:
            raise RuntimeError('Training slice made no checkpointed progress')
        submit_training(root, manifest, policy, policy_hash, backend, run, stage,
                        attempt + 1, [os.environ['SLURM_JOB_ID']])
        return
    from src.models.neuralgcm_residual.worker import require_receipt
    require_receipt(root, stage, run, manifest['source_id'])
    if stage == 'pretrain':
        submit_training(root, manifest, policy, policy_hash, backend, run, 'finetune',
                        0, [os.environ['SLURM_JOB_ID']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('validate-policy', 'train-slice', 'execute'))
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--policy-sha256', required=True)
    parser.add_argument('--run-id')
    parser.add_argument('--training-stage', choices=('pretrain', 'finetune'))
    parser.add_argument('--attempt', type=int, default=0)
    parser.add_argument('--resume')
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    policy, manifest = validate_policy(root, args.policy_sha256)
    backend = load_backend(manifest)
    manifest = backend.load_experiment(root)
    backend.check_data(manifest, backend.read_json(root / 'pipeline/config.json'))
    if args.action == 'validate-policy':
        print(json.dumps({'policy_valid': True, 'source_id': manifest['source_id'],
                          'full_statistics_still_required': True}))
        return
    if args.run_id not in manifest['configs'] or not args.training_stage:
        parser.error('Training requires a configured run and stage')
    lock = (root / 'pipeline' / f'authorized-{args.action}-{args.run_id}-{args.training_stage}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.action == 'train-slice':
        train_slice(root, manifest, policy, args.policy_sha256, backend,
                    args.run_id, args.training_stage, args.attempt)
        return
    from src.models.neuralgcm_residual import worker
    original_gate = worker.require_production_gates
    def authorized_gate(gate_root, gate_manifest, resolution, runtime):
        if Path(gate_root).resolve() != root or gate_manifest != manifest:
            raise ValueError('Authorization used outside its experiment')
        receipt = data_gate(root, manifest, resolution, runtime, worker, backend)
        backend.write_json(root / 'runs' / args.run_id / 'seed22' / args.training_stage /
                ('startup_policy_' + os.environ['SLURM_JOB_ID'] + '.json'),
                dict(startup_basis='explicit_user_authorization_after_completed_pilot_tests',
                     policy_sha256=args.policy_sha256,
                     operational_runner_sha256=policy['runner']['sha256'],
                     deferred_checks=policy['deferred_checks'],
                     production_statistics_sha256=receipt['result']['statistics_sha256'],
                     source_id=manifest['source_id']), immutable=True)
    worker.require_production_gates = authorized_gate
    try:
        worker.execute(root, manifest, args.training_stage, run_id=args.run_id, resume=args.resume)
    finally:
        worker.require_production_gates = original_gate


if __name__ == '__main__':
    main()
