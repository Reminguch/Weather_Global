#!/usr/bin/env python3
"""User-authorized production start while full feedback smokes run separately."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def policy(root, manifest):
    value = read(root / 'pipeline/early_start_v1/policy.json')
    if (value['source_id'] != manifest['source_id'] or value['experiment_root'] != str(root)
            or value['runner']['path'] != str(Path(__file__).resolve())
            or value['runner']['sha256'] != sha(__file__)
            or value['wait_for_full_smokes'] is not False or not value['authorization_text']):
        raise ValueError('Explicit early-start authorization identity differs')
    pilot = Path(manifest['pilot_root']) / 'outputs/report.json'
    if sha(pilot) != value['independent_smoke_sha256'] or not read(pilot)['passed']:
        raise ValueError('Independent smoke evidence changed')
    return value


def submit_one(root, manifest, frozen, backend, run, stage='pretrain', attempt=0, after=()):
    _, plan = frozen.job_plan(root, manifest, 'slice', run, stage=stage, attempt=attempt, after=after)
    key = f'early-{stage}-{run}-{attempt:03d}'
    command = [manifest['python'], '-u', str(Path(__file__).resolve()), 'slice',
               '--experiment-root', str(root), '--run-id', run, '--stage', stage,
               '--attempt', str(attempt)]
    batch = plan['sbatch']
    batch[batch.index('--job-name') + 1] = 'ngcm-v2-' + key
    for option, suffix in (('--output', '.out'), ('--error', '.err')):
        batch[batch.index(option) + 1] = str(root / 'logs' / (key + '-%j' + suffix))
    lines = plan['script'].splitlines()
    lines[-1] = 'exec ' + shlex.join(command)
    plan.update(script='\n'.join(lines) + '\n', early_start_policy_sha256=sha(root / 'pipeline/early_start_v1/policy.json'),
                operational_runner_sha256=sha(__file__))
    return backend.once(root, key, lambda: plan)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('publish', 'submit', 'slice', 'execute'))
    p.add_argument('--experiment-root', type=Path, required=True)
    p.add_argument('--run-id')
    p.add_argument('--stage', choices=('pretrain', 'finetune'), default='pretrain')
    p.add_argument('--attempt', type=int, default=0)
    p.add_argument('--resume')
    args = p.parse_args(); root = args.experiment_root.resolve(); args.experiment_root = root
    manifest = read(root / 'manifest.json')
    if args.action == 'publish':
        version = root / 'pipeline/early_start_v1'
        version.mkdir(exist_ok=False)
        runner = version / Path(__file__).name
        shutil.copyfile(__file__, runner)
        value = dict(experiment_root=str(root), source_id=manifest['source_id'],
            authorization_text='2026-09-24 user: 你的正式任务不需要等smoketets',
            wait_for_full_smokes=False, numerical_source_unchanged=True,
            runner=dict(path=str(runner), sha256=sha(runner)),
            independent_smoke_sha256=sha(Path(manifest['pilot_root']) / 'outputs/report.json'))
        (version / 'policy.json').write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        runner.chmod(0o444)
        print(json.dumps(value, ensure_ascii=False), flush=True)
        return
    authorization = policy(root, manifest)
    source = Path(manifest['source_root'])
    frozen = module(source / 'scripts/experiments/restart_neuralgcm_feedback.py', 'frozen_feedback_restart')
    frozen.initialize(root)
    from src.models.neuralgcm_residual.launcher import load_experiment
    from src.models.neuralgcm_residual.io import write_json, versions
    from src.models.neuralgcm_residual.worker import require_receipt
    manifest = load_experiment(root)
    if versions() != manifest['libraries']: raise ValueError('Numerical libraries changed')
    frozen.verify_reuse(root, manifest)
    backend = module(source / 'scripts/experiments/launch_neuralgcm_2p8_pipeline.py', 'early_submission')
    if args.action == 'submit':
        jobs = {run: submit_one(root, manifest, frozen, backend, run) for run in manifest['configs']}
        previous = read(root / 'pipeline/restart_chain.json')
        write_json(root / 'pipeline/restart_chain_with_smoke_dependencies.json', previous, immutable=True)
        chain = dict(previous, pretrain_jobs=jobs, production_afterok=[],
            startup_basis='explicit_user_authorization_without_waiting_for_full_smokes',
            early_start_policy_sha256=sha(root / 'pipeline/early_start_v1/policy.json'))
        write_json(root / 'pipeline/early_start_chain.json', chain, immutable=True)
        write_json(root / 'pipeline/restart_chain.json', chain)
        print(json.dumps({'pretrain_jobs': jobs, 'smoke_dependencies': []}), flush=True)
        return
    if args.run_id not in manifest['configs']: raise ValueError('Unknown run')
    if args.action == 'execute':
        def authorized_start(gate_root, gate_manifest):
            if Path(gate_root).resolve() != root or gate_manifest != manifest:
                raise ValueError('Authorization used outside its experiment')
            return dict(startup_basis='explicit_user_authorization_without_waiting_for_full_smokes',
                policy_sha256=sha(root / 'pipeline/early_start_v1/policy.json'),
                full_smokes_are_not_a_startup_gate=True,
                independent_smoke_sha256=authorization['independent_smoke_sha256'])
        frozen.validate_smoke_reports = authorized_start
        args.smoke_mode = False
        frozen.execute(args, manifest)
        return
    lock = (root / 'pipeline' / f'{args.run_id}-{args.stage}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output = root / 'runs' / args.run_id / 'seed22' / args.stage
    previous, checkpoint = backend.latest_checkpoint(output, args.stage)
    command = [manifest['python'], '-u', __file__, 'execute', '--experiment-root', str(root),
               '--run-id', args.run_id, '--stage', args.stage]
    if checkpoint: command += ['--resume', str(checkpoint)]
    try:
        subprocess.run(command, check=True, timeout=24 * 3600 - 900)
    except subprocess.TimeoutExpired:
        update, saved = backend.latest_checkpoint(output, args.stage)
        if saved is None or update <= previous:
            raise RuntimeError('Slice made no checkpointed progress')
        submit_one(root, manifest, frozen, backend, args.run_id, args.stage, args.attempt + 1,
                   after=[os.environ['SLURM_JOB_ID']])
        return
    require_receipt(root, args.stage, args.run_id, manifest['source_id'])
    if args.stage == 'pretrain':
        submit_one(root, manifest, frozen, backend, args.run_id, 'finetune', after=[os.environ['SLURM_JOB_ID']])


if __name__ == '__main__':
    main()
