#!/usr/bin/env python3
"""Real-process resume smoke: 100 cached updates and 6 closed-loop k=20 episodes.

Each phase is a separate Slurm process, limited to one hour on gpu-test. Uses
isolated 96-record pilot statistics and never writes production checkpoints.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

PHASES = ('pretrain-reference', 'pretrain-resume', 'finetune-reference', 'finetune-resume')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def metrics(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def compare_metrics(reference, resumed, updates):
    """Ignore wall-clock timing only; all losses, sample order and cursors agree."""
    expected = list(range(1, updates + 1))
    if [x['update'] for x in reference] != expected or [x['update'] for x in resumed] != expected:
        raise AssertionError('Missing, duplicate, or out-of-order resumed updates')
    clean = lambda rows: [{k: v for k, v in row.items() if k != 'seconds'} for row in rows]
    if clean(reference) != clean(resumed):
        raise AssertionError('Resumed metrics, data order, or cursor differs')


def seed_interrupted_log(reference, resumed, boundary):
    """Model a completed checkpoint plus a valid and a torn uncommitted log row."""
    rows = metrics(reference / 'metrics.jsonl')
    resumed.mkdir(parents=True, exist_ok=False)
    prefix = ''.join(json.dumps(row) + '\n' for row in rows if row['update'] <= boundary)
    tail = json.dumps(dict(update=boundary + 1, loss=-999, uncommitted=True)) + '\n{"update":'
    (resumed / 'metrics.jsonl').write_text(prefix + tail)
    return tail


def aggregate_reports(manifest, output, runner_hash):
    reports = {}
    for run, entry in manifest['configs'].items():
        path = output / run / 'report.json'
        report = json.loads(path.read_text())
        expected = dict(passed=True, run_id=run, source_id=manifest['source_id'],
                        config_sha256=entry['sha256'], runner_sha256=runner_hash,
                        independent_processes=True, pretrain_updates=100,
                        pretrain_resume_update=50, finetune_updates=6,
                        finetune_resume_update=3, closed_loop_steps=120,
                        normalization_scope='96_training_records')
        for key, value in expected.items():
            if report.get(key) != value:
                raise ValueError(f'{run}: invalid resume report {key}')
        if not report.get('exact_checkpoint_equality') or not report.get('exact_metrics_equality'):
            raise ValueError(f'{run}: resume mismatch')
        reports[run] = {'sha256': file_hash(path)}
    return reports


def run_phase(args, manifest, config, resources):
    import jax
    import numpy as np
    from src.models.neuralgcm_residual.cache import CacheReader
    from src.models.neuralgcm_residual.checkpoint import save_checkpoint, load_checkpoint
    from src.models.neuralgcm_residual.checks import backbone_parameter_digest
    from src.models.neuralgcm_residual.runtime import build_runtime, runtime_identities
    from src.models.neuralgcm_residual.pretrain import run_pretrain
    from src.models.neuralgcm_residual.finetune import run_finetune
    from src.models.neuralgcm_residual.io import read_json, write_json
    root = args.output / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    is_pretrain = args.phase.startswith('pretrain-')
    resumed = args.phase.endswith('-resume')
    runtime = build_runtime(config, resources, stage='pretrain' if is_pretrain else 'finetune')
    if not all(d.platform == 'gpu' for d in jax.devices()):
        raise RuntimeError('Resume phases require an actual GPU')
    reader = CacheReader(resources['cache_root'], runtime.store, 'train')
    if len(reader) != 96:
        raise ValueError('Expected the isolated 96-record pilot cache')
    ids = runtime_identities(runtime, config, manifest['source_id'], reader.manifest['producer_id'])
    before = backbone_parameter_digest(runtime.backbone.model)
    short = SimpleNamespace(**vars(config))
    short.pretrain_epochs = 25  # 96/24 * 25 = 100 production optimizer updates.
    short.finetune_updates = 6  # 120 live autoregressive steps, alternating cold/warm.
    short.finetune_validate_every = 1
    started = time.perf_counter()
    midpoint = root / 'pretrain_midpoint_50.pkl'
    output = root / args.phase
    expected_tail = None
    if resumed:
        reference = root / args.phase.replace('-resume', '-reference')
        expected_tail = seed_interrupted_log(reference, output, 50 if is_pretrain else 3)
    if is_pretrain:
        if resumed:
            saved = load_checkpoint(midpoint, stage='pretrain', identities=ids)
            if saved['cursor']['update'] != 50 or saved['cursor']['chunk'] != 2:
                raise AssertionError('Pretraining restart must be inside a 96-record segment')
            if not any(np.any(np.asarray(x) != 0) for x in jax.tree_util.tree_leaves(saved['memory'])):
                raise AssertionError('Mid-segment checkpoint must carry nonzero Mamba memory')
            run_pretrain(runtime, reader, short, output, ids, resume=midpoint)
        else:
            trainer = runtime.trainer
            original_gradients, original_update = trainer.cached_gradients, trainer.checked_update
            last_optimizer = None
            updates = 0
            def capture(params, memory, rng, records):
                if updates == 50:
                    cursor = metrics(output / 'metrics.jsonl')[-1]
                    cursor = {k: cursor[k] for k in ('epoch', 'segment', 'chunk', 'update', 'processed_timestamps')}
                    save_checkpoint(midpoint, stage='pretrain', identities=ids, params=params,
                                    optimizer=last_optimizer, memory=memory, rng=rng, cursor=cursor)
                    print(json.dumps({'checkpoint_saved_at_update':50,'cursor':cursor}), flush=True)
                return original_gradients(params, memory, rng, records)
            def update(params, optimizer, gradients):
                nonlocal updates, last_optimizer
                result = original_update(params, optimizer, gradients)
                last_optimizer = result[1]
                updates += 1
                if updates % 10 == 0:
                    print(json.dumps({'optimizer_updates':updates,'seconds':time.perf_counter()-started}), flush=True)
                return result
            trainer.cached_gradients, trainer.checked_update = capture, update
            try:
                run_pretrain(runtime, reader, short, output, ids)
            finally:
                trainer.cached_gradients, trainer.checked_update = original_gradients, original_update
            if updates != 100 or not midpoint.exists():
                raise AssertionError('Incomplete pretraining reference')
    else:
        parent = root / 'pretrain-reference/checkpoint_pass_25.pkl'
        resume = root / 'finetune-reference/checkpoint_000003.pkl' if resumed else None
        run_finetune(runtime, short, output, ids, parent, resume=resume)
    if backbone_parameter_digest(runtime.backbone.model) != before:
        raise AssertionError('Frozen NeuralGCM weights changed')
    if expected_tail is not None:
        boundary = 50 if is_pretrain else 3
        archive = output / f'metrics.jsonl.after_{boundary}.archive'
        if archive.read_text() != expected_tail:
            raise AssertionError('Interrupted metric tail was not archived exactly')
    write_json(root / (args.phase + '.json'), dict(passed=True, phase=args.phase,
        source_id=manifest['source_id'], runner_sha256=file_hash(__file__),
        job_id=os.environ['SLURM_JOB_ID'], pid=os.getpid(), gpu_measured=True,
        devices=[d.device_kind for d in jax.devices()], seconds=time.perf_counter()-started,
        frozen_backbone_sha256=before), immutable=True)
    print(json.dumps({'phase_passed':args.phase,'seconds':time.perf_counter()-started}), flush=True)


def check_run(args, manifest):
    from src.models.neuralgcm_residual.checkpoint import load_checkpoint
    from src.models.neuralgcm_residual.checks import tree_comparison
    from src.models.neuralgcm_residual.io import read_json, write_json
    root = args.output / args.run_id
    receipts = [read_json(root / (phase+'.json')) for phase in PHASES]
    for receipt in receipts:
        if (not receipt['passed'] or not receipt['gpu_measured'] or receipt['source_id'] != manifest['source_id']
                or receipt['runner_sha256'] != file_hash(__file__)):
            raise ValueError('Missing/stale GPU phase receipt')
    if len({receipt['job_id'] for receipt in receipts}) != 4:
        raise ValueError('Resume tests must restart in distinct Slurm processes')
    if len({receipt['frozen_backbone_sha256'] for receipt in receipts}) != 1:
        raise ValueError('Frozen backbone differs between processes')
    comparisons = {}
    for stage, name, count in [('pretrain','checkpoint_pass_25.pkl',100),('finetune','checkpoint_000006.pkl',6)]:
        paths = [root / (stage+suffix) / name for suffix in ('-reference','-resume')]
        identities = read_json(paths[0].with_suffix('.json'))['identities']
        states = [load_checkpoint(path, stage=stage, identities=identities) for path in paths]
        if states[0]['cursor'] != states[1]['cursor'] or states[0]['cursor']['update'] != count:
            raise AssertionError('Final resume cursor mismatch')
        comparisons[stage] = tree_comparison(tuple(states[0][k] for k in ('params','optimizer','memory','rng')),
                                            tuple(states[1][k] for k in ('params','optimizer','memory','rng')), exact=True)
        compare_metrics(metrics(paths[0].parent/'metrics.jsonl'), metrics(paths[1].parent/'metrics.jsonl'), count)
    report = dict(passed=True, run_id=args.run_id, source_id=manifest['source_id'],
        config_sha256=manifest['configs'][args.run_id]['sha256'], runner_sha256=file_hash(__file__),
        independent_processes=True, pretrain_updates=100, pretrain_resume_update=50,
        pretrain_processed_records=2400, bptt_steps=24, finetune_updates=6,
        finetune_resume_update=3, closed_loop_steps=120, rollout_steps=20,
        normalization_scope='96_training_records', production_statistics_required_separately=True,
        exact_checkpoint_equality=True, exact_metrics_equality=True, comparisons=comparisons,
        interrupted_log_recovery=True, phases=receipts)
    write_json(root/'report.json',report,immutable=True)
    print(json.dumps(report),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root',type=Path,required=True)
    parser.add_argument('--pilot-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run-id')
    parser.add_argument('--phase',choices=(*PHASES,'check','aggregate'),required=True)
    args = parser.parse_args()
    if args.phase != 'aggregate' and not args.run_id:
        parser.error('--run-id is required')
    raw = json.loads((args.experiment_root/'manifest.json').read_text());source=Path(raw['source_root'])
    for path in (source/'third_party/graphcast',source/'third_party/neuralgcm',source):sys.path.insert(0,str(path))
    from src.models.neuralgcm_residual.numerics import configure_environment
    configure_environment()
    from src.models.neuralgcm_residual.launcher import load_experiment,resolve_resources
    from src.models.neuralgcm_residual.config import load_config
    from src.models.neuralgcm_residual.io import read_json,write_json,sha256,versions
    manifest = load_experiment(args.experiment_root)
    if versions() != manifest['libraries']:raise ValueError('Dependency versions changed')
    if args.phase == 'aggregate':
        reports = aggregate_reports(manifest,args.output,file_hash(__file__))
        write_json(args.output/'PASSED.json',dict(passed=True,source_id=manifest['source_id'],reports=reports,
            runner_sha256=file_hash(__file__),production_statistics_required_separately=True),immutable=True)
        return
    if args.phase == 'check':
        check_run(args,manifest)
        return
    config = load_config(manifest['configs'][args.run_id]['path'])
    resources = resolve_resources(manifest,config.resolution_id)
    pilot=args.pilot_root/'pilot'/args.run_id
    quick_path = args.pilot_root/args.run_id/'report.json'
    if not quick_path.exists():
        quick_path = args.pilot_root/args.run_id/'in_progress.json'
    quick = read_json(quick_path)
    # Long restart checks can run independently once their cache/live inputs
    # and recurrent forward carry have been verified. The full detailed suite
    # remains a separate mandatory production gate.
    prerequisite_checks = ('zero_residual_40', 'cache_live_24', 'chunk_carry')
    if (quick.get('not_a_production_gate') is not True or not quick.get('gpu_measured')
            or quick['source_id'] != manifest['source_id']
            or quick['config_sha256'] != manifest['configs'][args.run_id]['sha256']
            or any(quick.get('checks', {}).get(name, {}).get('passed') is not True
                   for name in prerequisite_checks)):
        raise ValueError('Verified real cache/live/carry checks are required before resume testing')
    if sha256(pilot/'statistics.json') != read_json(pilot/'READY.json')['statistics_sha256']:
        raise ValueError('Pilot statistics changed')
    resources.update(cache_root=str(pilot/'cache'),statistics=str(pilot/'statistics.json'),
                     validation_origins=str(pilot/'validation_origins.json'))
    run_phase(args,manifest,config,resources)


if __name__ == '__main__':main()
