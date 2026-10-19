#!/usr/bin/env python3
"""Evaluate the matched reset-trained checkpoint using the frozen 2023 evaluator.

This is an extension of the original campaign, which is never modified. Its
M1 and M1_reset results supply the fixed comparison arms and canonical baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def record(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha(path)}


def checked(item):
    path = Path(item['path'])
    if sha(path) != item['sha256']:
        raise ValueError('Frozen file identity changed: ' + str(path))
    return path


def digest(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_new(path, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(body, stream, indent=2, allow_nan=False)
        stream.write('\n')


def prepare(workflow, workspace):
    if workflow.exists():
        raise FileExistsError(workflow)
    reference = workspace / 'artifacts/evaluations/v24_res1_2023_stride3_memory_recovery_20261009'
    ref = read(reference / 'protocol.json')
    if digest({k: v for k, v in ref.items() if k != 'protocol_hash'}) != ref['protocol_hash']:
        raise ValueError('Invalid reference protocol')
    training = workspace / 'artifacts/checkpoints/v24_Ilya/res1_selected_matched_reset_20261015'
    complete = read(training / 'checks/training_complete.json')
    if complete['completed_updates'] != {'branch': 2000, 'warmup': 500}:
        raise ValueError('Matched reset training is incomplete')
    checkpoint = record(complete['swa_checkpoint'])
    if checkpoint['sha256'] != complete['swa_sha256']:
        raise ValueError('Reset-trained SWA differs from completion record')
    reset_config = read(training / 'runs/branch/run_config.json')
    selected = next(m for m in ref['models'] if m['model_id'] == 'M1')
    if reset_config['architecture'] != selected['architecture']:
        raise ValueError('Reset-trained architecture differs from M1')
    if reset_config['sequence']['temporal_state_policy'] != 'reset_every_anchor':
        raise ValueError('Checkpoint was not trained with per-anchor reset')
    if ref['anchor_shard_count'] != 8 or len(ref['initialization_times']) != 122:
        raise ValueError('Unexpected reference coverage')
    for model_id in ('M1', 'M1_reset'):
        if not all((reference / 'shards' / f'{model_id}_{i:02d}.json').is_file() for i in range(8)):
            raise ValueError('Reference comparison arm is incomplete: ' + model_id)
    source = workspace / 'scripts'
    targets = {
        'run_v24_matched_reset_eval.py': source / 'experiments/run_v24_matched_reset_eval.py',
        'report_v24_matched_reset_2023.py': source / 'analyze_models/report_v24_matched_reset_2023.py',
        'v24_matched_reset_eval_gpu.slurm': source / 'experiments/v24_matched_reset_eval_gpu.slurm',
        'v24_matched_reset_eval_merge.slurm': source / 'experiments/v24_matched_reset_eval_merge.slurm',
    }
    for path in targets.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    (workflow / 'control').mkdir(parents=True)
    for name, path in targets.items():
        shutil.copy2(path, workflow / 'control' / name)
    for directory in ('shards', 'logs', 'submissions', 'report'):
        (workflow / directory).mkdir()
    body = {
        'schema_version': 1, 'workspace': str(workspace), 'model_id': 'R_matched',
        'reference_campaign': str(reference), 'reference_protocol': record(reference / 'protocol.json'),
        'reference_protocol_hash': ref['protocol_hash'], 'source_root': ref['source_root'],
        'source_hashes': ref['source_hashes'], 'checkpoint': checkpoint,
        'architecture': selected['architecture'], 'baseline': ref['baseline'], 'stats_dir': ref['stats_dir'],
        'forecast': ref['forecast'], 'data_path': ref['data_path'],
        'initialization_manifest': ref['initialization_manifest'],
        'initialization_times': ref['initialization_times'], 'target_steps': 40, 'anchor_shard_count': 8,
        'training_state_policy': 'reset_every_anchor', 'reset_state_every_step': True,
        'training_source_limit': read(training / 'source_provenance.json')['historical_limit'],
        'bootstrap': ref['bootstrap'], 'initialization_stride_days': 3,
        'prior_exposure': ref['prior_exposure'], 'numerical_policy': ref['numerical_policy'],
        'numerics': ref['numerics'], 'packages': ref['packages'],
        'training_provenance': [record(training / path) for path in (
            'manifest.json', 'checks/training_complete.json', 'checks/eval_preflight_20261018.json', 'source_provenance.json',
            'runs/warmup/run_config.json', 'runs/branch/run_config.json')],
        'source_files': {str((workflow / 'control' / name).resolve()): sha(workflow / 'control' / name)
                         for name in targets},
        'analysis': {
            'primary_endpoint': 'M1_vs_R_matched_rollout',
            'secondary_endpoint': 'M1_vs_R_matched_day10',
            'interpretation': 'New planned paired contrast; not a member of the original confirmatory endpoint family.',
            'canonical_baseline': 'Original M1 execution of GraphCast',
        },
        'resources': {'gpu': {'count': 1, 'cpu_count': 8, 'mem_gib': 12, 'walltime_seconds': 3720},
                      'evidence': '15297331_24..31: 22m12s–23m26s and 7.42–7.68 GiB MaxRSS; identical architecture/reset-inference workload.'},
    }
    body['protocol_hash'] = digest(body)
    write_new(workflow / 'protocol.json', body)
    print('Prepared ' + str(workflow), flush=True)


def verify(workflow):
    protocol = read(workflow / 'protocol.json')
    if digest({k: v for k, v in protocol.items() if k != 'protocol_hash'}) != protocol['protocol_hash']:
        raise ValueError('Extension protocol hash mismatch')
    if (protocol['model_id'] != 'R_matched' or protocol['training_state_policy'] != 'reset_every_anchor'
            or protocol['reset_state_every_step'] is not True):
        raise ValueError('Extension must evaluate reset-trained/reset-forecast policy')
    checked(protocol['reference_protocol'])
    checked(protocol['checkpoint'])
    checked(protocol['initialization_manifest'])
    for item in protocol['training_provenance']:
        checked(item)
    for path, expected in protocol['source_files'].items():
        checked({'path': path, 'sha256': expected})
    ref = read(protocol['reference_protocol']['path'])
    if (ref['protocol_hash'] != protocol['reference_protocol_hash']
            or protocol['initialization_manifest'] != ref['initialization_manifest']
            or protocol['initialization_times'] != ref['initialization_times']):
        raise ValueError('Extension does not use the original frozen start manifest')
    source = Path(ref['source_root'])
    sys.path[:0] = [str(source), str(source / 'third_party/graphcast')]
    spec = importlib.util.spec_from_file_location('matched_reset_frozen_workflow',
                                                 source / 'scripts/experiments/run_v24_2023_eval.py')
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    frozen.verify_protocol(Path(protocol['reference_campaign']))
    return protocol, ref, frozen


def evaluate(workflow, shard):
    protocol, ref, frozen = verify(workflow)
    if not 0 <= shard < protocol['anchor_shard_count']:
        raise ValueError('Invalid shard index')
    frozen.validate_runtime(ref)
    import jax
    from src.models.mamba.v24_Ilya.config import V24IlyaEvalConfig
    from src.models.mamba.v24_Ilya.evaluation import evaluate_v24_Ilya
    if len(jax.devices()) != 1 or jax.devices()[0].platform != 'gpu':
        raise ValueError('Evaluation requires one allocated GPU')
    gate = frozen.require_production_gate(Path(protocol['reference_campaign']), ref)
    hardware = frozen.capture_runtime_hardware(ref)
    if any(hardware[key] != gate[key] for key in ('gpu_kind', 'driver_version')):
        raise ValueError('GPU kind/driver differs from the qualified original campaign')
    output = workflow / 'shards' / f'R_matched_{shard:02d}.json'
    unpublished = output.with_name(output.stem + '.unpublished.json')
    if output.exists() or unpublished.exists():
        raise FileExistsError('Existing evaluation output: ' + str(output))
    config = V24IlyaEvalConfig(
        ckpt=Path(protocol['checkpoint']['path']), out_json=unpublished,
        ckpt_in=Path(ref['baseline']['path']), stats_dir=Path(ref['stats_dir']),
        data_path=ref['data_path'], val_year=2023, train_start_year=2015, train_end_year=2021,
        target_steps=40, n_samples=122, omit_rms_bias=True,
        initialization_manifest=Path(protocol['initialization_manifest']['path']),
        model_id='R_matched', anchor_shard_count=8, anchor_shard_index=shard,
        reset_state_every_step=True, **ref['forecast'], **protocol['architecture'])
    started = time.monotonic()
    result = evaluate_v24_Ilya(config)
    expected = protocol['initialization_times'][shard::8]
    rows = result.get('original_graphcast_loss_per_initialization', [])
    if (not result.get('metrics_complete') or result.get('evaluation_status') != 'complete'
            or result.get('rs_reset_every_step') is not True
            or [r['initialization_time'] for r in rows] != expected):
        raise ValueError('Evaluation output coverage or reset policy is incomplete')
    result.update(
        model_id='R_matched', protocol_hash=protocol['protocol_hash'],
        reference_protocol_hash=ref['protocol_hash'],
        checkpoint_sha256=protocol['checkpoint']['sha256'],
        baseline_checkpoint_sha256=ref['baseline']['checkpoint_sha256'],
        training_state_policy='reset_every_anchor', eval_mode='cold_full',
        residual_state_init_resolved='zero', warmup_steps=0,
        evaluator_source_digest=ref['source_hashes']['sha256'], evaluator_family='v24_Ilya',
        elapsed_seconds=time.monotonic() - started,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        devices=[{'platform': d.platform, 'kind': d.device_kind} for d in jax.devices()],
        slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    result.update(hardware)
    write_new(output, result)
    unpublished.unlink()
    print('Completed ' + str(output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'verify', 'evaluate'))
    parser.add_argument('--workflow', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    parser.add_argument('--shard-index', type=int, default=0)
    args = parser.parse_args()
    workflow = args.workflow.resolve()
    if args.phase == 'prepare':
        prepare(workflow, args.workspace.resolve())
    elif args.phase == 'verify':
        p, _, _ = verify(workflow)
        print('Verified ' + p['protocol_hash'])
    else:
        evaluate(workflow, args.shard_index)


if __name__ == '__main__':
    main()
