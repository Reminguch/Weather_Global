#!/usr/bin/env python3
"""Fresh, versioned four-arm training, gated by real GPU transition smokes."""
from pathlib import Path
import argparse
import fcntl
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[2]


def initialize(root=None):
    raw = json.loads((root / 'manifest.json').read_text()) if root else None
    source = Path(raw['source_root']) if raw else WORKSPACE
    for path in (source / 'third_party/graphcast', source / 'third_party/neuralgcm', source):
        sys.path.insert(0, str(path))
    from src.models.neuralgcm_residual.numerics import configure_environment
    configure_environment()
    return source


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(args):
    from dataclasses import replace
    from src.models.neuralgcm_residual.launcher import source_snapshot, load_experiment
    from src.models.neuralgcm_residual.config import load_config, BALANCED_LOSS_NAME
    from src.models.neuralgcm_residual.io import read_json, write_json, sha256, versions
    root, parent = args.experiment_root, args.parent_root.resolve()
    if root.exists():
        raise FileExistsError('Use a new experiment root; old runs are preserved')
    previous = load_experiment(parent)
    if previous['libraries'] != versions():
        raise ValueError('Library versions changed')
    snapshot = source_snapshot(WORKSPACE, root / 'source')
    resources = previous['resources']
    if set(resources) != {'res2p8'}:
        raise ValueError('Restart is limited to the four 2.8-degree arms')
    configurations = {}
    for run, entry in previous['configs'].items():
        config = replace(load_config(entry['path']), correction_policy='no_pressure_zero_mean_v2',
                         loss=BALANCED_LOSS_NAME)
        path = root / 'configs' / (run + '.json')
        write_json(path, config.to_dict(), immutable=True)
        configurations[run] = dict(path=str(path), sha256=sha256(path), resolution=config.resolution_id)
    manifest = dict(previous, experiment_id=root.name, source_id=snapshot['source_id'],
        source_root=str(root / 'source'), workspace=str(WORKSPACE), configs=configurations,
        production_status='fresh_feedback_v2_requires_restart_smokes',
        restart_parent=str(parent), restart_parent_manifest_sha256=sha256(parent / 'manifest.json'),
        restart_from='fresh_zero_head_seed22', pilot_root=str(args.pilot_root.resolve()))
    write_json(root / 'manifest.json', manifest, immutable=True)
    r = resources['res2p8']
    original_receipt = read_json(parent / 'checks/verify-cache_res2p8.json')
    if not original_receipt['passed'] or original_receipt['source_id'] != previous['source_id']:
        raise ValueError('Parent full-cache verification is not valid')
    cache = read_json(Path(r['cache_root']) / 'manifest.json')
    for relative, expected in cache['producer']['code'].items():
        if sha256(root / 'source/src/models/neuralgcm_residual' / relative) != expected:
            raise ValueError('Baseline cache producer changed: ' + relative)
    stats = read_json(r['statistics'])
    pipeline = read_json(parent / 'pipeline/config.json')
    paths = [parent / 'manifest.json', parent / 'checks/verify-cache_res2p8.json',
        Path(pipeline['verification']), Path(r['prepared']) / 'manifest.json',
        Path(r['cache_root']) / 'manifest.json', Path(r['cache_root']) / 'READY.json',
        Path(r['statistics']), Path(r['statistics']).parent / stats['climatology_file'],
        Path(r['checkpoint_index']), Path(r['validation_origins']), Path(r['test_origins'])]
    audit = dict(kind='verified_read_only_baseline_cache_and_training_statistics_reuse',
        files={str(p): sha256(p) for p in paths}, parent_source_id=previous['source_id'],
        source_id=manifest['source_id'], producer_id=cache['producer_id'],
        reason='Native features and baseline cache producer unchanged; correction and loss versioned separately',
        shard_verification='Each CacheReader rechecks shard bytes against READY hashes on first access')
    write_json(root / 'pipeline/reuse.json', audit, immutable=True)
    verify_reuse(root, manifest)
    write_json(root / 'checks/verify-cache_res2p8.json', dict(original_receipt,
        source_id=manifest['source_id'], reuse_audit_sha256=sha256(root / 'pipeline/reuse.json'),
        operation='verify_existing_immutable_artifacts'), immutable=True)
    for name in ('logs', 'submissions', 'baselines', 'reports'):
        (root / name).mkdir(exist_ok=True)
    print(json.dumps(dict(prepared=str(root), source_id=manifest['source_id'],
        runs=list(configurations), fresh_start=True)), flush=True)


def verify_reuse(root, manifest, runtime=None):
    from src.models.neuralgcm_residual.io import read_json, sha256, digest
    audit = read_json(root / 'pipeline/reuse.json')
    if audit['source_id'] != manifest['source_id']:
        raise ValueError('Reuse audit source differs')
    for path, expected in audit['files'].items():
        if sha256(path) != expected:
            raise ValueError('Reused artifact changed: ' + path)
    if runtime is not None:
        r = manifest['resources']['res2p8']
        stats = read_json(r['statistics'])
        cache = read_json(Path(r['cache_root']) / 'manifest.json')
        origins = [t for s in cache['shards'] if s['split'] == 'train' for t in s['origins']]
        if (stats['split'] != 'train' or stats['origin_count'] != len(origins)
                or stats['origins_sha256'] != digest(origins)
                or stats['native_schema_id'] != runtime.adapter.identity
                or stats['dataset_id'] != runtime.store.identity
                or digest(stats) != runtime.normalization.identity):
            raise ValueError('Runtime no longer matches full training statistics')


def validate_smoke_reports(root, manifest):
    from src.models.neuralgcm_residual.io import read_json, sha256
    reports = {}
    for run, entry in manifest['configs'].items():
        path = root / 'checks/restart_smoke' / run / 'PASSED.json'
        report = read_json(path)
        required = dict(passed=True, source_id=manifest['source_id'], run_id=run,
            config_sha256=entry['sha256'], runner_sha256=sha256(__file__), gpu_measured=True)
        if any(report.get(k) != v for k, v in required.items()):
            raise ValueError('Missing, failed or stale transition smoke: ' + run)
        for file, expected in report['evidence'].items():
            if sha256(file) != expected:
                raise ValueError('Smoke evidence changed: ' + file)
        reports[run] = dict(path=str(path), sha256=sha256(path))
    pilot = read_json(Path(manifest['pilot_root']) / 'outputs/report.json')
    if not pilot['passed'] or not pilot['frozen_parameters_unchanged']:
        raise ValueError('Independent representative-data smoke failed')
    return reports


def job_plan(root, manifest, action, run, *, after=(), attempt=0, stage='pretrain'):
    runner = Path(manifest['source_root']) / 'scripts/experiments/restart_neuralgcm_feedback.py'
    key = f'{action}-{run}-{stage}-{attempt:03d}'
    command = [manifest['python'], '-u', str(runner), action, '--experiment-root', str(root),
               '--run-id', run, '--stage', stage, '--attempt', str(attempt)]
    smoke = action == 'smoke'
    batch = ['sbatch', '--parsable', '--job-name', 'ngcm-v2-' + key,
        '--cpus-per-task', '8', '--mem', '64G', '--gres', 'gpu:1',
        '--constraint', 'a100&gpu80', '--qos', 'gpu-test' if smoke else 'gpu-short',
        '--time', '01:00:00' if smoke else '24:00:00',
        '--output', str(root / 'logs' / (key + '-%j.out')),
        '--error', str(root / 'logs' / (key + '-%j.err'))]
    if after:
        batch += ['--dependency', 'afterok:' + ':'.join(after)]
    script = '\n'.join(['#!/usr/bin/env bash', 'set -euo pipefail', 'ulimit -c 0',
        'unset PYTHONPATH PYTHONHOME',
        'export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1',
        'export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda',
        'export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8',
        'cd ' + shlex.quote(manifest['source_root']), 'exec ' + shlex.join(command), ''])
    return key, dict(sbatch=batch, script=script, afterok=list(after), source_id=manifest['source_id'])


def submit_one(root, manifest, action, run, **kwargs):
    pipeline = load_module(Path(manifest['source_root']) /
        'scripts/experiments/launch_neuralgcm_2p8_pipeline.py', 'restart_submission')
    key, plan = job_plan(root, manifest, action, run, **kwargs)
    return pipeline.once(root, key, lambda: plan)


def submit_all(args, manifest):
    from src.models.neuralgcm_residual.io import read_json, write_json, sha256
    root = args.experiment_root
    if not args.pilot_job or not args.pilot_job.isdigit():
        raise ValueError('Supply the independent gpu-test smoke job ID')
    verify_reuse(root, manifest)
    smokes = {run: submit_one(root, manifest, 'smoke', run) for run in manifest['configs']}
    pilot_path = Path(manifest['pilot_root']) / 'outputs/report.json'
    pilot_done = pilot_path.is_file() and read_json(pilot_path).get('passed') is True
    # Slurm removes completed jobs from its dependency table after MinJobAge.
    # Keep completed independent-smoke evidence by hash, not an expired job ID.
    dependencies = ([] if pilot_done else [args.pilot_job]) + list(smokes.values())
    training = {run: submit_one(root, manifest, 'slice', run, after=dependencies)
                for run in manifest['configs']}
    write_json(root / 'pipeline/restart_chain.json', dict(independent_smoke=args.pilot_job,
        smoke_jobs=smokes, pretrain_jobs=training, production_afterok=dependencies,
        independent_smoke_report_sha256=sha256(pilot_path) if pilot_done else None,
        source_id=manifest['source_id']), immutable=True)
    print(json.dumps(dict(smoke_jobs=smokes, pretrain_jobs=training)), flush=True)


def execute(args, manifest):
    from types import SimpleNamespace
    from src.models.neuralgcm_residual import worker
    from src.models.neuralgcm_residual.io import read_json, write_json
    root = args.experiment_root
    if args.smoke_mode and manifest.get('smoke_only') is not True:
        raise ValueError('Shortened execution is restricted to an isolated smoke root')
    original_gate, original_reader, original_config = worker.require_production_gates, worker.CacheReader, worker.load_config
    def gate(gate_root, gate_manifest, resolution, runtime):
        if Path(gate_root).resolve() != root or gate_manifest != manifest:
            raise ValueError('Gate used outside its experiment')
        verify_reuse(root, manifest, runtime)
        reports = {} if args.smoke_mode else validate_smoke_reports(root, manifest)
        write_json(root / 'pipeline/startup' / args.run_id / args.stage /
            (os.environ.get('SLURM_JOB_ID', 'manual') + '.json'),
            dict(source_id=manifest['source_id'], reports=reports, smoke_only=args.smoke_mode))
    worker.require_production_gates = gate
    if args.smoke_mode:
        class BoundedReader:
            def __init__(self, cache, store, split):
                self.reader = original_reader(cache, store, split)
                self.times = self.reader.times[:96 if split == 'train' else 24]
                self.manifest = self.reader.manifest
            def __len__(self): return len(self.times)
            def __getitem__(self, index):
                if not 0 <= index < len(self): raise IndexError(index)
                return self.reader[index]
        def short_config(path):
            cfg = original_config(path)
            return SimpleNamespace(**{**vars(cfg), 'pretrain_epochs': 1,
                'finetune_updates': 2, 'finetune_validate_every': 1},
                resolution_id=cfg.resolution_id, identity=cfg.identity, to_dict=cfg.to_dict)
        worker.CacheReader, worker.load_config = BoundedReader, short_config
    try:
        worker.execute(root, manifest, args.stage, run_id=args.run_id, resume=args.resume)
    finally:
        worker.require_production_gates, worker.CacheReader, worker.load_config = original_gate, original_reader, original_config


def train_slice(args, manifest):
    from src.models.neuralgcm_residual.worker import require_receipt
    pipeline = load_module(Path(manifest['source_root']) /
        'scripts/experiments/launch_neuralgcm_2p8_pipeline.py', 'restart_slice')
    root = args.experiment_root
    lock = (root / 'pipeline' / f'{args.run_id}-{args.stage}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_reuse(root, manifest)
    validate_smoke_reports(root, manifest)
    output = root / 'runs' / args.run_id / 'seed22' / args.stage
    previous, checkpoint = pipeline.latest_checkpoint(output, args.stage)
    command = [manifest['python'], '-u', __file__, 'execute', '--experiment-root', str(root),
               '--run-id', args.run_id, '--stage', args.stage]
    if checkpoint: command += ['--resume', str(checkpoint)]
    try:
        subprocess.run(command, check=True, timeout=24 * 3600 - 900)
    except subprocess.TimeoutExpired:
        update, saved = pipeline.latest_checkpoint(output, args.stage)
        if saved is None or update <= previous:
            raise RuntimeError('Timed-out slice made no checkpointed progress')
        submit_one(root, manifest, 'slice', args.run_id, stage=args.stage,
                   attempt=args.attempt + 1, after=[os.environ['SLURM_JOB_ID']])
        return
    require_receipt(root, args.stage, args.run_id, manifest['source_id'])
    if args.stage == 'pretrain':
        submit_one(root, manifest, 'slice', args.run_id, stage='finetune',
                   after=[os.environ['SLURM_JOB_ID']])


def k2_smoke(args, manifest):
    from types import SimpleNamespace
    import jax
    import jax.numpy as jnp
    import numpy as np
    from src.models.neuralgcm_residual.launcher import resolve_resources
    from src.models.neuralgcm_residual.config import load_config
    from src.models.neuralgcm_residual.runtime import build_runtime, runtime_identities
    from src.models.neuralgcm_residual.cache import CacheReader
    from src.models.neuralgcm_residual.finetune import run_finetune
    from src.models.neuralgcm_residual.native_state import physical_state
    from src.models.neuralgcm_residual.io import write_json
    if any(d.platform != 'gpu' for d in jax.devices()):
        raise RuntimeError('K=2 transition requires a real GPU')
    config = load_config(manifest['configs'][args.run_id]['path'])
    resources = resolve_resources(manifest, config.resolution_id)
    runtime = build_runtime(config, resources, stage='finetune')
    verify_reuse(args.experiment_root, manifest, runtime)
    reader = CacheReader(resources['cache_root'], runtime.store, 'train')
    ids = runtime_identities(runtime, config, manifest['source_id'], reader.manifest['producer_id'])
    ids['smoke_rollout_steps'] = 2
    arm = args.experiment_root / 'checks/restart_smoke' / args.run_id
    parent = arm / 'detailed' / args.run_id / 'pretrain_full/checkpoint_pass_01.pkl'
    output = arm / ('k2_resumed' if args.resume else 'k2_full')
    short = SimpleNamespace(**{**vars(config), 'rollout_steps': 2,
                              'finetune_updates': 2, 'finetune_validate_every': 1})
    if args.resume:
        output.mkdir(exist_ok=False)
        # Preserve the committed prefix before exercising actual log recovery.
        first_line = (arm / 'k2_full/metrics.jsonl').read_text().splitlines()[0]
        (output / 'metrics.jsonl').write_text(first_line + '\n')
    else:
        record = reader[0]
        state = record['baseline_state']
        delta = jnp.ones(tuple(runtime.adapter.grid.nodal_shape) + (runtime.adapter.output_size,))
        corrected = runtime.adapter.apply_increment(state, runtime.normalization.increment(delta))
        old, new = physical_state(state), physical_state(corrected)
        np.testing.assert_array_equal(new.log_surface_pressure, old.log_surface_pressure)
        for key in ('divergence', 'vorticity'):
            np.testing.assert_array_equal(getattr(new, key)[..., 0], getattr(old, key)[..., 0])
        fields = []
        # Training-only representative samples audit weights without fitting to validation.
        for index in np.linspace(0, len(reader) - 1, 8).astype(int):
            r = reader[int(index)]
            decoded = runtime.backbone.decode(r['baseline_state'], r['forcing'])
            fields.append({k: float(v) for k, v in
                runtime.trainer.weather_loss.field_scores(decoded, r['target']).items()})
        means = {k: float(np.mean([f[k] for f in fields])) for k in fields[0]}
        fraction = sum(v for k, v in means.items() if k.startswith('specific_cloud_')) / sum(means.values())
        if not np.isfinite(list(means.values())).all() or fraction >= .9:
            raise AssertionError(f'Cloud normalization remains pathological: {fraction}')
        write_json(arm / 'constraint_and_loss_audit.json', dict(passed=True,
            constraints_exact=True, training_samples=8, fields_mean=means,
            cloud_fraction=fraction, loss_name=runtime.trainer.weather_loss.name), immutable=True)
    run_finetune(runtime, short, output, ids, parent, resume=args.resume)


def smoke(args, manifest):
    from src.models.neuralgcm_residual.io import read_json, write_json, sha256
    from src.models.neuralgcm_residual.checkpoint import load_checkpoint
    from src.models.neuralgcm_residual.checks import tree_comparison
    import jax
    started = time.perf_counter()
    root = args.experiment_root
    arm = root / 'checks/restart_smoke' / args.run_id
    arm.mkdir(parents=True, exist_ok=False)
    if any(d.platform != 'gpu' for d in jax.devices()):
        raise RuntimeError('Smoke requires an actual GPU')
    # This controller performs no numerical work. Release its GPU allocation to
    # independent child interpreters instead of initializing large model arrays.
    def child(command):
        print(json.dumps({'child_command': command}), flush=True)
        subprocess.run(command, check=True, timeout=2400)
    source = Path(manifest['source_root'])
    detailed = source / 'scripts/training/smoke_neuralgcm_detailed.py'
    child([manifest['python'], '-u', str(detailed), '--experiment-root', str(root),
           '--run-id', args.run_id, '--output', str(arm / 'detailed')])
    # Exercise the production CLI, gate and checkpoint-selection handoff in an
    # isolated root, with 96 K=1 records and two genuine K=20 fine-tuning updates.
    isolated = arm / 'cli'
    resources = {k: dict(v) for k, v in manifest['resources'].items()}
    val = read_json(resources['res2p8']['validation_origins'])
    val_path = isolated / 'manifests/val_origins.json'
    write_json(val_path, dict(val, origins=val['origins'][:1]), immutable=True)
    resources['res2p8']['validation_origins'] = str(val_path)
    child_manifest = dict(manifest, experiment_id=isolated.name, smoke_only=True, resources=resources)
    write_json(isolated / 'manifest.json', child_manifest, immutable=True)
    for relative in ('pipeline/reuse.json', 'checks/verify-cache_res2p8.json'):
        write_json(isolated / relative, read_json(root / relative), immutable=True)
    for stage in ('pretrain', 'finetune'):
        child([manifest['python'], '-u', __file__, 'execute', '--experiment-root', str(isolated),
               '--run-id', args.run_id, '--stage', stage, '--smoke-mode'])
    for resume in (None, arm / 'k2_full/checkpoint_000001.pkl'):
        command = [manifest['python'], '-u', __file__, 'k2-smoke', '--experiment-root', str(root),
                   '--run-id', args.run_id]
        if resume: command += ['--resume', str(resume)]
        child(command)
    final_states = []
    for name in ('k2_full', 'k2_resumed'):
        path = arm / name / 'checkpoint_000002.pkl'
        meta = read_json(path.with_suffix('.json'))
        final_states.append(load_checkpoint(path, stage='finetune', identities=meta['identities']))
    comparison = tree_comparison(tuple(final_states[0][k] for k in ('params', 'optimizer', 'memory', 'rng')),
                                 tuple(final_states[1][k] for k in ('params', 'optimizer', 'memory', 'rng')), exact=True)
    if final_states[0]['cursor'] != final_states[1]['cursor']:
        raise AssertionError('Independent-process K=2 resume cursor differs')
    metrics = []
    for name in ('k2_full', 'k2_resumed'):
        rows = [json.loads(line) for line in (arm / name / 'metrics.jsonl').read_text().splitlines()]
        metrics.append([{k: v for k, v in r.items() if k != 'seconds'} for r in rows])
    if metrics[0] != metrics[1]:
        raise AssertionError('Independent-process K=2 resume metrics differ')
    detailed_report = arm / 'detailed' / args.run_id / 'report.json'
    report = read_json(detailed_report)
    if not report['passed'] or len(report['checks']) != 15 or not all(v['passed'] for v in report['checks'].values()):
        raise AssertionError('Detailed production-shape checks incomplete')
    cli_base = isolated / 'runs' / args.run_id / 'seed22'
    pretrain = read_json(cli_base / 'pretrain/selection_locked.json')
    fine_parent = read_json(cli_base / 'finetune/parent.json')
    if pretrain['sha256'] != fine_parent['sha256']:
        raise AssertionError('Actual CLI handoff selected a different parent')
    evidence_paths = [detailed_report, arm / 'constraint_and_loss_audit.json',
        cli_base / 'pretrain/selection_locked.json', cli_base / 'finetune/selection_locked.json',
        cli_base / 'finetune/parent.json', arm / 'k2_full/checkpoint_000002.json',
        arm / 'k2_resumed/checkpoint_000002.json', arm / 'k2_full/metrics.jsonl', arm / 'k2_resumed/metrics.jsonl']
    write_json(arm / 'PASSED.json', dict(passed=True, source_id=manifest['source_id'],
        run_id=args.run_id, config_sha256=manifest['configs'][args.run_id]['sha256'],
        runner_sha256=sha256(__file__), gpu_measured=True,
        job_id=os.environ.get('SLURM_JOB_ID'), devices=[d.device_kind for d in jax.devices()],
        detailed_checks=15, actual_cli_pretrain_to_k20=True, k1_to_k2_updates=2,
        independent_process_k2_resume_exact=comparison, checkpoint_cursor_exact=True,
        checkpoint_metrics_exact=True, evidence={str(p): sha256(p) for p in evidence_paths},
        seconds=time.perf_counter() - started), immutable=True)
    print(json.dumps({'smoke_passed': args.run_id, 'seconds': time.perf_counter() - started}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'submit', 'smoke', 'execute', 'slice', 'k2-smoke'))
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--parent-root', type=Path)
    parser.add_argument('--pilot-root', type=Path)
    parser.add_argument('--pilot-job')
    parser.add_argument('--run-id')
    parser.add_argument('--stage', choices=('pretrain', 'finetune'), default='pretrain')
    parser.add_argument('--attempt', type=int, default=0)
    parser.add_argument('--resume')
    parser.add_argument('--smoke-mode', action='store_true')
    args = parser.parse_args()
    args.experiment_root = args.experiment_root.resolve()
    initialize(None if args.action == 'prepare' else args.experiment_root)
    if args.action == 'prepare':
        if not args.parent_root or not args.pilot_root: parser.error('prepare requires parent and pilot roots')
        prepare(args)
        return
    from src.models.neuralgcm_residual.launcher import load_experiment
    from src.models.neuralgcm_residual.io import versions
    manifest = load_experiment(args.experiment_root)
    if versions() != manifest['libraries']: raise ValueError('Numerical libraries changed')
    if args.action != 'submit' and args.run_id not in manifest['configs']:
        parser.error('Select a configured run')
    actions = {'submit': submit_all, 'execute': execute, 'slice': train_slice,
               'smoke': smoke, 'k2-smoke': k2_smoke}
    actions[args.action](args, manifest)


if __name__ == '__main__':
    main()
