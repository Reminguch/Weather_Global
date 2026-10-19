"""Frozen amplitude/pattern MGDA, di16 and di32, 2,000 canonical updates.

Two calibrations precede two independent 100-update qualifications. Successful
technical qualification releases exactly 1,900 more updates per width, followed
by the declared evaluations and a terminal report. No skill selection or retry.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import pickle
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.frozen_runtime import freeze_source, verify_source
from scripts.experiments.run_v24_mgda import (
    call, check, compare_native, file_sha256, gpu_required, pinned, read,
    seal, serialization_check, submit_once, unseal, write, write_once,
)

DEFAULT_EXPERIMENT = 'artifacts/checkpoints/v24_Ilya/res1_amse_split_mgda_di16_di32_20261012'
INITIAL_ROOT = 'artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908'
REFERENCE = 'artifacts/checkpoints/v24_Ilya/res1_amse_pair_20260919'
ARMS = ('di16', 'di32')
INITIAL_HASHES = {
    'di16': '25a0c980a3ae2f64d45bdfab95c876738dd9e8817edccfef3fc810a2dd47545b',
    'di32': '46220eda60f80b22fccc4a205a345ada4d9afda399c410bb1d9354c0392659f9',
}
EVAL_STEPS = (0, 500, 2000)
FORMAT = 'v24_amse_split_mgda_2k_v1'
WRAPPER = 'scripts/experiments/v24_amse_split_mgda.slurm'
DRIVER = 'scripts/experiments/run_v24_amse_split_mgda.py'
# Numerical warning thresholds, fixed before qualification. Historical BF16 replay
# measured parameter relative-L2 3.38e-5 and state 2.09e-2. Moment comparisons
# exclude the integer optimizer count, which would otherwise hide drift.
NUMERICAL_ENVELOPES = {
    'residual_params': {'relative': 1e-3, 'absolute': 1e-5},
    'second_moment': {'relative': 1e-3, 'absolute': 1e-5},
    'residual_state': {'relative': 0.1, 'absolute': 1e-5},
    'trajectory_loss': {'relative': 0.1, 'absolute': 1e-5},
    'trajectory_update_norm': {'relative': 0.25, 'absolute': 1e-7},
}


def make_config(template, experiment, arm, calibration=None, *, bootstrap=False, shadow=False):
    if arm not in ARMS or bootstrap and shadow:
        raise ValueError('Invalid arm/configuration role')
    config = copy.deepcopy(template)
    config['objective'] = {'loss_mode': 'all_steps', 'loss_kind': 'amse',
        'objective_pair': 'amse_split', 'amse_lmax': 179, 'amse_lambda': 1.,
        'amse_power_epsilon': 1e-12}
    if not bootstrap:
        if calibration is None:
            raise ValueError('Production needs the common frozen calibration')
        config['objective']['amse_split_calibration'] = copy.deepcopy(calibration)
    config['optimizer'].update(optimizer_kind='adamw' if bootstrap else 'mgda_rms',
        max_steps=2000, checkpoint_every=50, learning_rate=1e-5,
        warmup_start_learning_rate=1e-6, end_learning_rate=3e-6,
        learning_rate_schedule='warmup_cosine_constant', warmup_steps=100,
        decay_end_step=1000, rms_decay=.98, rms_epsilon=1e-8, weight_decay=0.,
        seed=22, spatial_lr_multiplier=1., mamba_lr_multiplier=3., grad_clip=1.)
    config['validation']['enabled'] = False
    folder = 'calibration/forward' if bootstrap else 'shadow' if shadow else 'runs'
    config['output'] = {'output_root': str(Path(experiment) / folder), 'run_name': arm}
    return config


def tasks_for(manifest, experiment):
    tasks = []
    for arm in ARMS:
        init = manifest['initializations'][arm]
        for step in EVAL_STEPS:
            run = Path(init['source_run_dir']) if step == 0 else experiment / 'runs' / arm
            task = {'id': f'{arm}_step{step:05d}', 'label': f'{arm} ' + ('initialization' if not step else f'update {step}'),
                'role': ('mse_control' if arm == 'di16' else 'reference') if step == 0 else 'candidate',
                'family': arm, 'step': step, 'seed': 22, 'architecture': init['architecture'],
                'checkpoint': init['checkpoint']['path'] if not step else str(run / f'checkpoints/checkpoint_step{step:08d}.pkl'),
                'source_run_dir': str(run), 'paired_initialization_id': f'{arm}_step00000',
                'endpoint': 'primary' if step == 2000 else 'initialization' if not step else 'diagnostic'}
            if step:
                task['planned_config'] = str(experiment / 'configs' / f'{arm}.json')
            tasks.append(task)
    return tasks


def coordinate_coverage(spec):
    """Read only coordinates; ensure both inputs and every scored lead exist."""
    import numpy as np
    import xarray as xr
    with xr.open_zarr(spec['data_path'], consolidated=True, chunks=None) as dataset:
        times = np.asarray(dataset.time.values, dtype='datetime64[ns]')
    if times.ndim != 1 or len(times) < 42 or not np.all(np.diff(times) == np.timedelta64(6, 'h')):
        raise ValueError('Evaluation time coordinate must be complete six-hourly data')
    if spec['stage'] == 'spectral':
        starts = np.array([np.datetime64(value.replace('Z', ''), 'ns') for value in spec['initialization_times']])
    else:
        selected = times[(times >= np.datetime64(f"{spec['evaluation_year']}-01-01")) &
                         (times < np.datetime64(f"{spec['evaluation_year'] + 1}-01-01"))]
        indices = np.asarray(spec['chosen_idx'], dtype=int)
        if len(indices) != spec['n_samples'] or np.any(indices < 1) or np.any(indices + spec['target_steps'] >= len(selected)):
            raise ValueError('Exact cohort indices lack input/forecast coverage')
        starts = selected[indices]
    required = starts[:, None] + np.arange(-1, spec['target_steps'] + 1)[None, :] * np.timedelta64(6, 'h')
    if not np.isin(required, times).all():
        raise ValueError('Evaluation coordinate coverage is incomplete')
    return {'stage': spec['stage'], 'samples': len(starts), 'required_coordinate_count': int(required.size),
            'first': str(times[0]), 'last': str(times[-1]), 'passed': True}


def prepare(experiment, workspace):
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    from src.data_operations.loaders.zarr_metadata import verify_frozen_zarr_metadata
    experiment, workspace = Path(experiment).resolve(), Path(workspace).resolve()
    if experiment.exists():
        raise FileExistsError(f'Use a fresh campaign root: {experiment}')
    templates, initializations, inputs = {}, {}, []
    for arm in ARMS:
        run = workspace / INITIAL_ROOT / 'runs' / f'{arm}_joint'
        checkpoint, source_config = pinned(run / 'swa/swa_step00500-02000.pkl'), pinned(run / 'run_config.json')
        if checkpoint['sha256'] != INITIAL_HASHES[arm]:
            raise ValueError('Prescribed matched initialization hash differs')
        template_path = workspace / INITIAL_ROOT / 'configs' / f'{arm}_joint.json'
        template = read(template_path)
        for key in ('prepared_root', 'anchor_manifest_root', 'baseline_checkpoint', 'stats_dir'):
            template['data'][key] = str((workspace / template['data'][key]).resolve())
        architecture = read(source_config['path'])['architecture']
        if architecture['temporal_d_inner'] != int(arm[2:]):
            raise ValueError('Initialization width mismatch')
        initializations[arm] = {'checkpoint': checkpoint, 'run_config': source_config,
                               'source_run_dir': str(run), 'architecture': architecture}
        templates[arm] = template
        inputs += [checkpoint, source_config, pinned(template_path)]
    left, right = (copy.deepcopy(initializations[arm]['architecture']) for arm in ARMS)
    left.pop('temporal_d_inner'); right.pop('temporal_d_inner')
    if left != right:
        raise ValueError('Matched initializations differ beyond temporal width')
    data = templates['di16']['data']
    if data != templates['di32']['data']:
        raise ValueError('Both widths must use identical training inputs')
    inputs += [pinned(data['baseline_checkpoint']), pinned(Path(data['prepared_root']) / 'metadata.json')]
    inputs += [pinned(Path(data['anchor_manifest_root']) / path) for path in
               ('metadata.json', 'anchors/anchor_indices.npy', 'anchors/anchor_times.npy', 'anchors/split_train.npy', 'anchors/split_val.npy')]
    inputs += [pinned(Path(data['stats_dir']) / name) for name in
               ('stddev_by_level.nc', 'mean_by_level.nc', 'diffs_stddev_by_level.nc')]
    evaluation_data, coverage = [], []
    for stage in ('exact', 'spectral'):
        path = workspace / REFERENCE / 'eval' / stage / 'mse_fp32/swa_step00500-02000/specification.json'
        inputs.append(pinned(path))
        spec = read(path)
        evaluation_data.append({'root': spec['data_path'], 'metadata': verify_frozen_zarr_metadata(spec['data_path'], spec['data_metadata'])})
        for field in ('initialization_manifest', 'evaluation_reference', 'source_spectral_protocol'):
            if field in spec:
                check(spec[field]); inputs.append(spec[field])
        coverage.append(coordinate_coverage(spec))
    experiment.mkdir(parents=True)
    for arm, template in templates.items():
        path = experiment / 'templates' / f'{arm}.json'
        write(path, template); inputs.append(pinned(path))
        path = experiment / 'configs' / f'calibration_{arm}.json'
        write(path, make_config(template, experiment, arm, bootstrap=True))
        load_training_config(path); inputs.append(pinned(path))
    write(experiment / 'checks/coordinate_coverage.json', coverage)
    inputs.append(pinned(experiment / 'checks/coordinate_coverage.json'))
    source = freeze_source(workspace, experiment / 'source')
    manifest = {'format': FORMAT, 'workspace': str(workspace), 'experiment': str(experiment),
        'source_root': str(experiment / 'source'), 'source_digest': source['source_digest'],
        'inputs': inputs, 'initializations': initializations, 'evaluation_data': evaluation_data,
        'reference_experiment': str(workspace / REFERENCE), 'numerical_envelopes': NUMERICAL_ENVELOPES,
        'numerical_policy': 'advisory',
        'policy': {'arms': list(ARMS), 'seed': 22, 'updates': 2000, 'qualification_updates': 100,
                   'shadow_updates': 50, 'continuation_updates': 1900, 'calibration_batches_per_width': 32,
                   'checkpoint_every': 50, 'evaluation_steps': list(EVAL_STEPS), 'primary': 'native_2000',
                   'swa': False, 'automatic_retries': 0, 'forecast_skill_gate': False, 'terminal_stage': 'report'},
        'resources': {'historical_peak_rss_gib': 234.225,
            'calibration': {'di16': {'memory': 128, 'minutes': 90}, 'di32': {'memory': 160, 'minutes': 90}},
            'qualification': {'di16': {'memory': 320, 'minutes': 210}, 'di32': {'memory': 384, 'minutes': 240}}},
        'runtime': {'xla_flags': '--xla_gpu_enable_triton_gemm=false', 'python_safe_path': True}}
    seal(experiment / 'manifest.json', manifest)
    return verify(experiment)


def verify(experiment, *, frozen=False, resolved=False):
    experiment = Path(experiment).resolve()
    manifest = unseal(experiment / 'manifest.json')
    if manifest['format'] != FORMAT or manifest['experiment'] != str(experiment):
        raise ValueError('Unknown/moved campaign')
    source = Path(manifest['source_root'])
    if frozen and ROOT != source:
        raise ValueError('Execute this stage from the frozen source')
    if verify_source(source)['source_digest'] != manifest['source_digest']:
        raise ValueError('Frozen source changed')
    for record in manifest['inputs']:
        check(record)
    from src.data_operations.loaders.zarr_metadata import verify_frozen_zarr_metadata
    for data in manifest['evaluation_data']:
        verify_frozen_zarr_metadata(data['root'], data['metadata'])
    if resolved:
        record = unseal(experiment / 'resolved.json')
        if record['manifest_hash'] != manifest['record_hash']:
            raise ValueError('Resolved configuration belongs to another campaign')
        for item in record['artifacts']:
            check(item)
    return manifest


def calibrate(experiment, task):
    m = verify(experiment, frozen=True)
    gpu_required()
    arm = ARMS[task]
    destination = experiment / 'calibration' / f'{arm}.json'
    if destination.exists():
        raise FileExistsError('Calibration output exists; no implicit retry')
    from src.models.mamba.v24_Ilya.training.runner import calibrate_training
    calibrate_training(experiment / 'configs' / f'calibration_{arm}.json',
                       Path(m['initializations'][arm]['checkpoint']['path']), destination, batches=32)


def combine(experiment):
    m = verify(experiment, frozen=True)
    if (experiment / 'resolved.json').exists():
        return verify(experiment, frozen=True, resolved=True)
    from src.models.mamba.v24_Ilya.training.amse_split_loss import combine_amse_split_calibrations
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    from scripts.analyze_models.evaluate_v24_amse_regime import prepare_evaluation
    raw_paths = [experiment / 'calibration' / f'{arm}.json' for arm in ARMS]
    records = [read(path) for path in raw_paths]
    for arm, record in zip(ARMS, records, strict=True):
        config_path = experiment / 'configs' / f'calibration_{arm}.json'
        if record.get('batches') != 32 or record.get('init_from_sha256') != m['initializations'][arm]['checkpoint']['sha256']:
            raise ValueError('Calibration count or prescribed initialization differs')
        if record.get('source_config_sha256') != file_sha256(config_path):
            raise ValueError('Calibration source configuration hash differs')
        if record.get('source_config') != load_training_config(config_path).to_dict():
            raise ValueError('Calibration resolved source configuration differs')
    calibration = combine_amse_split_calibrations(records)
    common = experiment / 'calibration/common.json'
    write_once(common, calibration)
    artifacts = [pinned(path) for path in [*raw_paths, common]]
    for arm in ARMS:
        template = read(experiment / 'templates' / f'{arm}.json')
        for shadow in (False, True):
            path = experiment / 'configs' / f'{arm}{"_shadow" if shadow else ""}.json'
            write_once(path, make_config(template, experiment, arm, calibration, shadow=shadow))
            load_training_config(path); artifacts.append(pinned(path))
    prepare_evaluation(tasks_for(m, experiment), experiment / 'evaluation', 'di16_step00000',
        reference_experiment=m['reference_experiment'], workspace=m['workspace'],
        observer_factory='src.models.mamba.v24_Ilya.spherical_observer:make_full_field_observer',
        policy={'qualification': 'No forecast-skill gate or checkpoint selection; native2000 primary.'})
    artifacts.append(pinned(experiment / 'evaluation/manifest.json'))
    return seal(experiment / 'resolved.json', {'manifest_hash': m['record_hash'],
                'common_scale': calibration['common_scale'], 'artifacts': artifacts})


def training_command(experiment, manifest, arm, *, resume=None, stop_after=None, shadow=False):
    args = [sys.executable, '-u', Path(manifest['source_root']) / 'scripts/training/train_v24_Ilya.py',
            '--config', experiment / 'configs' / f'{arm}{"_shadow" if shadow else ""}.json']
    args += ['--resume', resume] if resume else ['--init-from', manifest['initializations'][arm]['checkpoint']['path'], '--zero-state-on-init']
    if stop_after is not None:
        args += ['--stop-after-steps', str(stop_after)]
    return args


def validate_records(records, steps, *, first=1):
    if [row.get('step') for row in records] != list(range(first, steps + 1)):
        raise ValueError('Missing or duplicated training updates')
    required = ('loss', 'gradient_norm', 'step_seconds', 'gc_loss', 'amplitude_loss', 'pattern_loss',
                'amplitude_loss_normalized', 'pattern_loss_normalized', 'amplitude_gradient_norm',
                'pattern_gradient_norm', 'combined_gradient_norm', 'clip_factor', 'direction_norm',
                'update_norm', 'gradient_cosine', 'mgda_alpha', 'amplitude_direction_dot',
                'pattern_direction_dot', 'amplitude_displacement_dot', 'pattern_displacement_dot',
                'metric_amplitude_amplitude', 'metric_amplitude_pattern', 'metric_pattern_pattern')
    for row in records:
        if row.get('objective_pair') != 'amse_split':
            raise ValueError('Missing or incorrect objective identities')
        for key in required:
            if key not in row or not isinstance(row[key], (int, float)) or not math.isfinite(row[key]):
                raise ValueError(f'Missing/nonfinite diagnostic {key}')
        if row.get('finite') is not True or row.get('descent_valid') is not True:
            raise ValueError('Optimizer finite/common-descent check failed')
        if not 0 <= row['mgda_alpha'] <= 1 or not 0 <= row['clip_factor'] <= 1:
            raise ValueError('Invalid MGDA coefficient or clip factor')
        if any(key in row for key in ('power_loss', 'gc_loss_normalized', 'metric_gc_power')):
            raise ValueError('Split objective mislabeled as GC/power')
        for objective in ('amplitude', 'pattern'):
            for dot, norm in (('direction_dot', 'direction_norm'), ('displacement_dot', 'update_norm')):
                if row[f'{objective}_{dot}'] < -1e-6 * row[f'{objective}_gradient_norm'] * row[norm]:
                    raise ValueError('Proposed/actual update violates common descent')


def validate_run(experiment, arm, steps, *, shadow=False):
    import jax
    import numpy as np
    from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_training_checkpoint
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    config = load_training_config(experiment / 'configs' / f'{arm}{"_shadow" if shadow else ""}.json')
    run = experiment / ('shadow' if shadow else 'runs') / arm
    records = [json.loads(line) for line in (run / 'train_metrics.jsonl').read_text().splitlines()]
    validate_records(records, steps, first=51 if shadow else 1)
    path = run / f'checkpoints/checkpoint_step{steps:08d}.pkl'
    checkpoint = load_v24_Ilya_training_checkpoint(path)
    if checkpoint.completed_step != steps or checkpoint.resolved_training_config != config.to_dict():
        raise ValueError('Checkpoint/config/update index mismatch')
    if int(checkpoint.optimizer_state.count) != steps:
        raise ValueError('RMS moment update count differs')
    if checkpoint.baseline_checkpoint_fingerprint != file_sha256(config.baseline_checkpoint):
        raise ValueError('Frozen baseline identity changed')
    for tree in (checkpoint.residual_params, checkpoint.residual_state, checkpoint.optimizer_state):
        if any(not np.isfinite(np.asarray(value)).all() for value in jax.tree_util.tree_leaves(tree)):
            raise ValueError('Nonfinite checkpoint state')
    return records, path


def relocated_checkpoint(source, destination, output):
    """Copy native state exactly; change only the operational output destination."""
    from src.models.mamba.v24_Ilya.checkpoint import atomic_pickle_dump
    with Path(source).open('rb') as handle:
        payload = pickle.load(handle)
    payload['resolved_training_config'] = copy.deepcopy(payload['resolved_training_config'])
    payload['resolved_training_config']['output'] = copy.deepcopy(output)
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    atomic_pickle_dump(payload, Path(destination))


def prepare_shadow_run(canonical_run, shadow_run, midpoint, shadow_config_path):
    """Build the separate resume directory, including required run metadata."""
    if shadow_run.exists():
        raise FileExistsError('Shadow directory exists; never overwrite restart evidence')
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    shadow_config = load_training_config(shadow_config_path).to_dict()
    metadata = read(canonical_run / 'run_config.json')
    for key, value in shadow_config.items():
        if key != 'output' and metadata.get(key) != value:
            raise ValueError(f'Shadow run metadata would change scientific section {key}')
    metadata['output'] = copy.deepcopy(shadow_config['output'])
    metadata['config_source'] = str(shadow_config_path)
    destination = shadow_run / 'checkpoints' / midpoint.name
    relocated_checkpoint(midpoint, destination, shadow_config['output'])
    write_once(shadow_run / 'run_config.json', metadata)
    return destination


def compare_shadow(canonical, shadow, canonical_config, shadow_config, envelopes):
    """Restore only output metadata in a temporary copy, then check all state."""
    import numpy as np
    from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_training_checkpoint
    from src.models.mamba.v24_Ilya.training.cached_diagnostics import tree_comparison
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    a, b = (load_v24_Ilya_training_checkpoint(Path(path)) for path in (canonical, shadow))
    ca, cb = (load_training_config(Path(path)).to_dict() for path in (canonical_config, shadow_config))
    if a.resolved_training_config != ca or b.resolved_training_config != cb:
        raise ValueError('Replay config differs from its pinned configuration')
    aa, bb = copy.deepcopy(ca), copy.deepcopy(cb)
    aa.pop('output'); bb.pop('output')
    if aa != bb:
        raise ValueError('Shadow changes scientific configuration')
    with tempfile.TemporaryDirectory(prefix='split-mgda-shadow-') as temporary:
        normalized = Path(temporary) / 'normalized.pkl'
        relocated_checkpoint(shadow, normalized, ca['output'])
        result = compare_native(canonical, normalized, policy='advisory')
    groups = {'residual_params': (a.residual_params, b.residual_params),
              'residual_state': (a.residual_state, b.residual_state),
              'second_moment': (a.optimizer_state.second_moment, b.optimizer_state.second_moment)}
    result['gross_envelopes'] = copy.deepcopy(envelopes)
    result['gross_comparisons'] = {}
    result['warnings'] = []
    for name, (expected, actual) in groups.items():
        limits = envelopes[name]
        value = tree_comparison(actual, expected, **limits)
        if 'reason' in value:
            raise ValueError(f'Invalid restart comparison: {name}: {value}')
        # Absolute tolerance handles small moments without using optimizer count
        # in the denominator; retain both relative and absolute diagnostics.
        acceptable = value.get('passed') is True or value.get('max_abs', np.inf) <= limits['absolute']
        result['gross_comparisons'][name] = {**value, 'within_envelope': acceptable}
        if not acceptable:
            message = f'Restart drift exceeds numerical warning threshold: {name}: {value}'
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            result['warnings'].append(message)
            result['status'] = 'complete_with_warnings'
            result['numerical_status'] = 'warning'
    result['output_relocation_only'] = True
    return result


def compare_trajectory(canonical, shadow, envelopes):
    import numpy as np
    result = {}
    paired = canonical[50:100]
    if [row['step'] for row in paired] != [row['step'] for row in shadow]:
        raise ValueError('Replay trajectory step identities differ')
    for key in ('amplitude_loss', 'pattern_loss', 'gc_loss', 'update_norm'):
        left = np.asarray([row[key] for row in paired], np.float64)
        right = np.asarray([row[key] for row in shadow], np.float64)
        if not (np.isfinite(left).all() and np.isfinite(right).all()):
            raise ValueError(f'Nonfinite replay trajectory: {key}')
        absolute = float(np.max(np.abs(left - right)))
        scale = float(np.linalg.norm(left))
        relative = float(np.linalg.norm(left - right) / scale) if scale else (0. if absolute == 0 else None)
        limit = envelopes['trajectory_update_norm' if key == 'update_norm' else 'trajectory_loss']
        passed = absolute <= limit['absolute'] or relative is not None and relative <= limit['relative']
        result[key] = {'relative_l2': relative, 'max_abs': absolute, 'within_envelope': passed, 'limits': limit}
        if not passed:
            message = f'Replayed trajectory exceeds numerical warning threshold: {key}'
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            result[key]['warning'] = message
    return result


def steady_timings(run, records):
    """Include data waits and amortized checkpoint I/O; exclude initial compile."""
    checkpoints = run / 'checkpoint_metrics.jsonl'
    overhead = {}
    if checkpoints.exists():
        overhead = {row['step']: row['checkpoint_seconds'] for row in
                    [json.loads(line) for line in checkpoints.read_text().splitlines()]}
    return [row.get('end_to_end_seconds', row['step_seconds']) + overhead.get(row['step'], 0.) for row in records[4:]]


def proof(experiment, manifest, arm, *, qualification):
    steps = 100 if qualification else 2000
    stage = 'qualification' if qualification else 'train'
    result = unseal(experiment / 'checks' / f'{arm}_{stage}.json')
    expected = {'passed': True, 'arm': arm, 'completed_steps': steps,
                'manifest_hash': manifest['record_hash'], 'source_digest': manifest['source_digest'],
                'integrity_passed': True, 'serialization_checked': True}
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f'Completion proof differs: {key}')
    if qualification and result.get('resume_checked') is not True:
        raise ValueError('Qualification lacks shadow restart proof')
    for key, path in {'checkpoint': experiment / 'runs' / arm / f'checkpoints/checkpoint_step{steps:08d}.pkl',
                      'config': experiment / 'configs' / f'{arm}.json', 'resolved': experiment / 'resolved.json'}.items():
        if result.get(key, {}).get('path') != str(path):
            raise ValueError('Completion proof refers to another run')
        check(result[key])
    for item in result.get('retained_evidence', []):
        check(item)
    return result


def train(experiment, task, *, qualification=False):
    m = verify(experiment, frozen=True, resolved=True)
    gpu_required()
    from src.models.mamba.v24_Ilya.training.cached_diagnostics import require_matched_gpu_environment
    environment = require_matched_gpu_environment('advisory')
    arm, steps = ARMS[task], 100 if qualification else 2000
    stage = 'qualification' if qualification else 'train'
    proof_path = experiment / 'checks' / f'{arm}_{stage}.json'
    if proof_path.exists():
        return proof(experiment, m, arm, qualification=qualification)
    run = experiment / 'runs' / arm
    if qualification and run.exists():
        raise FileExistsError('Partial qualification exists; diagnosis required, no implicit retry')
    resume = None
    if not qualification:
        initial = proof(experiment, m, arm, qualification=True)
        resume = Path(initial['checkpoint']['path'])
        # A continuation may not silently restart an already partially advanced run.
        existing = [json.loads(line)['step'] for line in (run / 'train_metrics.jsonl').read_text().splitlines()]
        if existing != list(range(1, 101)):
            raise ValueError('Partial continuation requires explicit diagnosis; automatic replay refused')
    start = time.monotonic()
    call(training_command(experiment, m, arm, resume=resume, stop_after=100 if qualification else None))
    records, checkpoint = validate_run(experiment, arm, steps)
    retained = []
    replay, trajectory = None, None
    timings = steady_timings(run, records if qualification else records[100:])
    if qualification:
        canonical_pin = pinned(checkpoint)
        midpoint = run / 'checkpoints/checkpoint_step00000050.pkl'
        midpoint_serialization = serialization_check(midpoint)
        shadow_run = experiment / 'shadow' / arm
        shadow50 = prepare_shadow_run(run, shadow_run, midpoint,
                                     experiment / 'configs' / f'{arm}_shadow.json')
        # Check relocation preserved every numerical and discrete leaf before use.
        relocation = compare_shadow(midpoint, shadow50, experiment / 'configs' / f'{arm}.json',
                                    experiment / 'configs' / f'{arm}_shadow.json', m['numerical_envelopes'])
        if not relocation['bitwise_equal']:
            raise ValueError('Shadow relocation changed serialized numerical state')
        call(training_command(experiment, m, arm, resume=shadow50, stop_after=100, shadow=True))
        shadow_records, shadow100 = validate_run(experiment, arm, 100, shadow=True)
        replay = compare_shadow(checkpoint, shadow100, experiment / 'configs' / f'{arm}.json',
                                experiment / 'configs' / f'{arm}_shadow.json', m['numerical_envelopes'])
        trajectory = compare_trajectory(records, shadow_records, m['numerical_envelopes'])
        check(canonical_pin)
        timings += steady_timings(shadow_run, shadow_records)
        retained += [pinned(midpoint), pinned(shadow50), pinned(shadow100),
                     pinned(shadow_run / 'train_metrics.jsonl'), pinned(shadow_run / 'run_config.json')]
    serialization = serialization_check(checkpoint)
    import numpy as np
    report = {'passed': True, 'integrity_passed': True, 'serialization_checked': True,
        'serialization': serialization, 'numerical_comparison': replay, 'trajectory_comparison': trajectory,
        'midpoint_serialization': midpoint_serialization if qualification else None,
        'status': ('complete_with_warnings' if replay and
                   (replay['status'] == 'complete_with_warnings' or
                    any('warning' in value for value in trajectory.values())) else 'complete'),
        'numerical_policy': 'advisory', 'arm': arm, 'completed_steps': steps,
        'manifest_hash': m['record_hash'], 'source_digest': m['source_digest'],
        'config': pinned(experiment / 'configs' / f'{arm}.json'), 'resolved': pinned(experiment / 'resolved.json'),
        'checkpoint': pinned(checkpoint), 'retained_evidence': retained, 'environment': environment,
        'elapsed_seconds': time.monotonic() - start,
        'peak_rss_gib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2,
        'steady_p90_seconds': float(np.percentile(timings, 90)), 'resume_checked': qualification,
        'zero_update_count': sum(row['update_norm'] == 0 for row in records)}
    return seal(proof_path, report)


def resource_budget(report, historical_peak=234.225):
    if report.get('passed') is not True or report.get('completed_steps') != 100:
        raise ValueError('Successful 100-update qualification required')
    peak, seconds = report['peak_rss_gib'], report['steady_p90_seconds']
    if not all(math.isfinite(value) and value > 0 for value in (peak, seconds)):
        raise ValueError('Invalid qualification measurement')
    memory = math.ceil(1.35 * max(historical_peak, peak) / 16) * 16
    minutes = math.ceil(1.35 * (1900 * seconds + 1200) / 1800) * 30
    if memory > 512 or minutes > 4320:
        raise ValueError('Measured continuation exceeds frozen memory/gpu-medium bounds')
    return {'memory_gib': memory, 'minutes': minutes}


def recover_qualification(experiment, task):
    """Reassess preserved 100-step trajectories; never execute optimizer updates."""
    import numpy as np
    m = verify(experiment, frozen=True, resolved=True)
    recovery = m['recovery']
    arm = ARMS[task]
    proof_path = experiment / 'checks' / f'{arm}_qualification.json'
    if proof_path.exists():
        return proof(experiment, m, arm, qualification=True)
    parent = Path(recovery['parent'])
    # Verify the original frozen campaign and every imported artifact first.
    original = verify(parent, resolved=True)
    if original['record_hash'] != recovery['parent_manifest_hash']:
        raise ValueError('Recovery parent identity differs')
    for item in recovery['evidence']:
        check(item)
    records, checkpoint = validate_run(experiment, arm, 100)
    shadow_records, shadow100 = validate_run(experiment, arm, 100, shadow=True)
    run, shadow_run = experiment / 'runs' / arm, experiment / 'shadow' / arm
    config, shadow_config = (experiment / 'configs' / f'{arm}{suffix}.json' for suffix in ('', '_shadow'))
    retained = []
    # Imported checkpoints may change only output destinations, not saved state.
    for kind, target_config in (('runs', config), ('shadow', shadow_config)):
        for step in (50, 100):
            old = parent / kind / arm / f'checkpoints/checkpoint_step{step:08d}.pkl'
            new = experiment / kind / arm / f'checkpoints/checkpoint_step{step:08d}.pkl'
            comparison = compare_shadow(old, new, parent / 'configs' / target_config.name,
                                        target_config, m['numerical_envelopes'])
            if not comparison['bitwise_equal']:
                raise ValueError('Recovery import changed serialized state')
            retained += [pinned(old), pinned(new)]
    midpoint = run / 'checkpoints/checkpoint_step00000050.pkl'
    shadow50 = shadow_run / 'checkpoints/checkpoint_step00000050.pkl'
    relocation = compare_shadow(midpoint, shadow50, config, shadow_config, m['numerical_envelopes'])
    if not relocation['bitwise_equal']:
        raise ValueError('Shadow midpoint changed serialized state')
    replay = compare_shadow(checkpoint, shadow100, config, shadow_config, m['numerical_envelopes'])
    trajectory = compare_trajectory(records, shadow_records, m['numerical_envelopes'])
    timings = steady_timings(run, records) + steady_timings(shadow_run, shadow_records)
    for directory in (run, shadow_run):
        retained += [pinned(directory / name) for name in ('train_metrics.jsonl', 'run_config.json')]
    report = {'passed': True, 'integrity_passed': True, 'serialization_checked': True,
        'serialization': serialization_check(checkpoint), 'midpoint_serialization': serialization_check(midpoint),
        'numerical_comparison': replay, 'trajectory_comparison': trajectory,
        'status': ('complete_with_warnings' if replay['status'] == 'complete_with_warnings' or
                   any('warning' in value for value in trajectory.values()) else 'complete'),
        'numerical_policy': 'advisory', 'arm': arm, 'completed_steps': 100,
        'manifest_hash': m['record_hash'], 'source_digest': m['source_digest'],
        'config': pinned(config), 'resolved': pinned(experiment / 'resolved.json'),
        'checkpoint': pinned(checkpoint), 'retained_evidence': retained,
        'recovery': recovery, 'peak_rss_gib': recovery['peak_rss_gib'][arm],
        'steady_p90_seconds': float(np.percentile(timings, 90)), 'resume_checked': True,
        'zero_update_count': sum(row['update_norm'] == 0 for row in records)}
    return seal(proof_path, report)


def sbatch(experiment, manifest, stage, *, memory, minutes, gpu=False, task=0, array=None, dependency=None):
    if not 1 <= memory <= 512 or not 1 <= minutes <= 4320:
        raise ValueError('Resource request outside bounded campaign')
    if array is not None and (not str(array) or '%' in str(array)):
        raise ValueError('Arrays must be unthrottled')
    args = ['sbatch', '--parsable', '--account=dabanin', f"--chdir={manifest['workspace']}",
            '--kill-on-invalid-dep=yes', '--cpus-per-task=8' if gpu else '--cpus-per-task=2',
            f'--mem={memory}G', f'--time={minutes}', f'--job-name=split-{stage}',
            f'--output={experiment}/logs/{stage}_%A_%a.out', f'--error={experiment}/logs/{stage}_%A_%a.err']
    if gpu:
        args += ['--gres=gpu:1', '--constraint=gpu80', '--qos=' + ('gpu-short' if minutes <= 480 else 'gpu-medium')]
    if dependency:
        args.append(f'--dependency=afterok:{dependency}')
    if array is not None:
        args.append(f'--array={array}')
    args += [str(Path(manifest['source_root']) / WRAPPER), stage, str(experiment)]
    if array is None:
        args.append(str(task))
    return args


def submit(experiment):
    m = verify(experiment)
    calibrations = [submit_once(experiment, f'calibrate_{arm}', sbatch(experiment, m, 'calibrate',
        **m['resources']['calibration'][arm], gpu=True, task=index)) for index, arm in enumerate(ARMS)]
    combined = submit_once(experiment, 'combine', sbatch(experiment, m, 'combine', memory=8, minutes=20,
        dependency=':'.join(calibrations)))
    qualifiers = [submit_once(experiment, f'qualify_{arm}', sbatch(experiment, m, 'qualify',
        **m['resources']['qualification'][arm], gpu=True, task=index, dependency=combined)) for index, arm in enumerate(ARMS)]
    return submit_once(experiment, 'release', sbatch(experiment, m, 'release', memory=8, minutes=20,
        dependency=':'.join(qualifiers)))


def release(experiment):
    m = verify(experiment, frozen=True, resolved=True)
    # Validate both widths before submitting either continuation.
    proofs = {arm: proof(experiment, m, arm, qualification=True) for arm in ARMS}
    budgets = {arm: resource_budget(proofs[arm], m['resources']['historical_peak_rss_gib']) for arm in ARMS}
    jobs = []
    for index, arm in enumerate(ARMS):
        budget = budgets[arm]
        seal(experiment / 'resources' / f'{arm}.json', {**budget, 'remaining_updates': 1900,
             'evidence': pinned(experiment / 'checks' / f'{arm}_qualification.json')})
        jobs.append(submit_once(experiment, f'train_{arm}', sbatch(experiment, m, 'train',
            memory=budget['memory_gib'], minutes=budget['minutes'], gpu=True, task=index,
            dependency=os.environ.get('SLURM_JOB_ID'))))
    evaluations = []
    for stage, minutes in (('exact', 90), ('spectral', 60)):
        evaluations.append(submit_once(experiment, f'evaluate_{stage}', sbatch(experiment, m,
            f'evaluate-{stage}', memory=16, minutes=minutes, gpu=True, array='0-5', dependency=':'.join(jobs))))
    return submit_once(experiment, 'report', sbatch(experiment, m, 'report', memory=12, minutes=30,
        dependency=':'.join(evaluations)))


def evaluate(experiment, task, stage):
    m = verify(experiment, frozen=True, resolved=True)
    gpu_required()
    from scripts.analyze_models import evaluate_v24_amse_regime as evaluator
    path = experiment / 'evaluation/manifest.json'
    manifest = evaluator.load_manifest(path)
    if not 0 <= task < 6 or stage not in ('exact', 'spectral'):
        raise ValueError('Invalid evaluation task/stage')
    item = manifest['task_specs'][task]
    proof(experiment, m, item['family'], qualification=False)
    destination = evaluator.evaluation_directory(manifest, item, stage)
    if destination.exists():
        if not (destination / 'evaluation.json').exists():
            raise FileExistsError('Partial evaluation requires diagnosis; no implicit retry')
        evaluator.resolve_result(path, task, stage)
        return
    evaluator.evaluate_worker(path, task, stage)
    directory, specification, result, _ = evaluator.resolve_result(path, task, stage)
    if stage == 'spectral':
        spectral = evaluator.load_spectral_result(directory, specification, result)
        if spectral['global_mse'] is None:
            raise ValueError('Physical-field spectral observer outputs missing')


def report(experiment):
    m = verify(experiment, frozen=True, resolved=True)
    for arm in ARMS:
        proof(experiment, m, arm, qualification=False)
    from scripts.analyze_models.report_v24_amse_split_mgda import build_report
    images = Path(m['workspace']) / 'plots/analyze_models/images/resolution_eval' / experiment.name
    output = build_report(experiment / 'evaluation/manifest.json', image_dir=images, training_root=experiment / 'runs')
    return seal(experiment / 'complete.json', {'manifest_hash': m['record_hash'],
        'terminal_stage': 'report', 'report': pinned(output / 'split_mgda_report.json'),
        'primary_endpoints': [f'{arm}_step02000' for arm in ARMS], 'integrity_passed': True,
        'status': 'complete', 'scientific_success_is_separate': True})


def dry_run(experiment):
    m = verify(experiment)
    nodes = []
    def add(name, stage, resources, dependencies=(), **kwargs):
        nodes.append({'id': name, 'depends_on': list(dependencies), 'resources': resources,
                      'command': sbatch(experiment, m, stage, **resources, **kwargs,
                        dependency=':'.join(f'<{value}>' for value in dependencies) or None)})
    for index, arm in enumerate(ARMS):
        add(f'calibrate_{arm}', 'calibrate', {**m['resources']['calibration'][arm], 'gpu': True}, task=index)
    add('combine', 'combine', {'memory': 8, 'minutes': 20}, [f'calibrate_{arm}' for arm in ARMS])
    for index, arm in enumerate(ARMS):
        add(f'qualify_{arm}', 'qualify', {**m['resources']['qualification'][arm], 'gpu': True}, ['combine'], task=index)
    add('release', 'release', {'memory': 8, 'minutes': 20}, [f'qualify_{arm}' for arm in ARMS])
    for index, arm in enumerate(ARMS):
        command = sbatch(experiment, m, 'train', memory=320, minutes=1920, gpu=True, task=index)
        command = ['--mem=<measured_GiB>G' if item.startswith('--mem=') else
                   '--time=<measured_minutes>' if item.startswith('--time=') else
                   '--qos=<measured_qos>' if item.startswith('--qos=') else item for item in command]
        nodes.append({'id': f'train_{arm}', 'depends_on': ['release'], 'command_template': command,
            'remaining_updates': 1900, 'resume_step': 100,
            'resources': {'measured_from': f'checks/{arm}_qualification.json',
                'memory_formula': 'ceil(1.35*max(234.225, peak_GiB)/16)*16',
                'minutes_formula': 'ceil(1.35*(1900*p90_seconds+1200)/1800)*30'}})
    for stage, minutes in (('exact', 90), ('spectral', 60)):
        add(f'evaluate_{stage}', f'evaluate-{stage}', {'memory': 16, 'minutes': minutes, 'gpu': True},
            [f'train_{arm}' for arm in ARMS], array='0-5')
    add('report', 'report', {'memory': 12, 'minutes': 30}, ['evaluate_exact', 'evaluate_spectral'])
    return {'submitted': False, 'manifest': pinned(experiment / 'manifest.json'),
            'source_digest': m['source_digest'], 'nodes': nodes, 'terminal_stage': 'report'}


def status(experiment):
    m = verify(experiment)
    expected = [experiment / 'calibration' / f'{arm}.json' for arm in ARMS]
    expected += [experiment / 'resolved.json']
    expected += [experiment / 'checks' / f'{arm}_{stage}.json' for arm in ARMS for stage in ('qualification', 'train')]
    expected += [experiment / 'evaluation/results' / f'{arm}_step{step:05d}' / stage / 'evaluation.json'
                 for arm in ARMS for step in EVAL_STEPS for stage in ('exact', 'spectral')]
    expected.append(experiment / 'complete.json')
    submissions = [{'name': path.stem, **read(path)} for path in sorted((experiment / 'submissions').glob('*.json'))]
    jobs = [row['job_id'] for row in submissions if row.get('job_id')]
    accounting, queue = [], []
    if jobs:
        for command, target in [(['sacct', '-j', ','.join(jobs), '--array', '-Pn', '--format=JobID,State,ExitCode,Elapsed,MaxRSS'], accounting),
                                (['squeue', '-j', ','.join(jobs), '-h', '-o', '%i|%T|%r'], queue)]:
            result = subprocess.run(command, text=True, capture_output=True)
            target.extend((result.stdout if result.returncode == 0 else result.stderr).splitlines())
    return {'campaign': m['experiment'], 'complete': (experiment / 'complete.json').exists(),
            'missing_outputs': [str(path.relative_to(experiment)) for path in expected if not path.exists()],
            'submissions': submissions, 'accounting': accounting, 'queue': queue, 'terminal_stage': 'report'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'dry-run', 'submit', 'calibrate', 'combine', 'qualify', 'release', 'recover-qualification',
                                         'train', 'evaluate-exact', 'evaluate-spectral', 'report', 'status'))
    parser.add_argument('--experiment', type=Path, default=Path(DEFAULT_EXPERIMENT))
    parser.add_argument('--workspace', type=Path, default=Path(os.environ.get('WEATHER_WORKSPACE', ROOT)))
    parser.add_argument('--task', type=int, default=0)
    args = parser.parse_args(argv)
    experiment = args.experiment.resolve()
    try:
        if args.stage in ('calibrate', 'qualify', 'train', 'recover-qualification') and args.task not in (0, 1):
            parser.error('--task must identify di16 (0) or di32 (1)')
        if args.stage == 'prepare':
            prepare(experiment, args.workspace)
            result = dry_run(experiment)
        elif args.stage == 'calibrate':
            result = calibrate(experiment, args.task)
        elif args.stage in ('qualify', 'train'):
            result = train(experiment, args.task, qualification=args.stage == 'qualify')
        elif args.stage == 'recover-qualification':
            result = recover_qualification(experiment, args.task)
        elif args.stage.startswith('evaluate-'):
            result = evaluate(experiment, args.task, args.stage.split('-', 1)[1])
        else:
            result = globals()[args.stage.replace('-', '_')](experiment)
        if result is not None:
            print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        if experiment.exists() and args.stage not in ('prepare', 'dry-run', 'submit', 'status'):
            write(experiment / 'failures' / f'{args.stage}_{args.task}_{os.environ.get("SLURM_JOB_ID", "local")}.json',
                  {'stage': args.stage, 'task': args.task, 'error_type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
