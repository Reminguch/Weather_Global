"""Frozen 200-update GC-MSE/excess-AMSE MGDA pilot with matched controls.

Qualification never advances production parameters, memory, optimizer, RNG or
cursor. It gates three fresh trajectories; no 2k/di32 expansion is automatic.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.frozen_runtime import (
    FORMAT as SOURCE_FORMAT, _digest, _source_files, file_sha256, verify_source,
)
from scripts.experiments.run_v24_mgda import (
    call, check, gpu_required, pinned, read, seal, serialization_check,
    submit_once, unseal, write, write_once,
)

DEFAULT_EXPERIMENT = 'artifacts/checkpoints/v24_Ilya/res1_mse_amse_excess_pilot_20261018'
INITIAL_ROOT = 'artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908'
INITIAL_SHA256 = '25a0c980a3ae2f64d45bdfab95c876738dd9e8817edccfef3fc810a2dd47545b'
REFERENCE = 'artifacts/checkpoints/v24_Ilya/res1_amse_pair_20260919'
ARMS = ('mgda', 'mse_control', 'fixed_sum')
FORMAT = 'v24_mse_amse_excess_pilot_v1'
WRAPPER = 'scripts/experiments/v24_mse_amse_excess_pilot.slurm'
DRIVER = 'scripts/experiments/run_v24_mse_amse_excess_pilot.py'
DATES = tuple(f'2022-{month:02d}-15T00:00:00Z' for month in (1, 2, 4, 5, 7, 8, 10, 11))
BACKTRACKING = dict(max_halvings=8, shrink=.5, absolute_floor=1e-8,
                    relative_floor=1e-7, max_relative_noise=1e-3,
                    repeatability_repeats=3)
PREPARED_LOADER = 'src/models/graphcast/training/core/prepared_array.py'


def make_config(template, experiment, arm, *, qualification=False):
    if arm not in ARMS:
        raise ValueError('Unknown pilot arm')
    config = copy.deepcopy(template)
    config['objective'] = dict(loss_mode='all_steps', loss_kind='amse',
        objective_pair='mse_amse_excess', amse_lmax=179, amse_lambda=1., amse_power_epsilon=1e-12)
    config['optimizer'].update(optimizer_kind='mgda_rms', mgda_direction=arm,
        mgda_backtracking=copy.deepcopy(BACKTRACKING), max_steps=200, checkpoint_every=50,
        learning_rate=1e-6, warmup_start_learning_rate=1e-6, end_learning_rate=1e-6,
        learning_rate_schedule='warmup_cosine_constant', warmup_steps=2, decay_end_step=3,
        rms_decay=.98, rms_epsilon=1e-8, weight_decay=0., seed=22,
        spatial_lr_multiplier=1., mamba_lr_multiplier=3., grad_clip=1.)
    config['validation']['enabled'] = False
    config['output'] = {'output_root': str(Path(experiment) / ('qualification' if qualification else 'runs')),
                        'run_name': arm}
    return config


def tasks_for(manifest, experiment):
    init = manifest['initialization']
    tasks = [{'id': 'parent', 'label': 'Original MSE parent', 'role': 'mse_control',
        'family': 'parent', 'step': 0, 'checkpoint': init['checkpoint']['path'],
        'source_run_dir': init['source_run_dir'], 'architecture': init['architecture'],
        'paired_initialization_id': 'parent', 'endpoint': 'initialization'}]
    for arm in ARMS:
        run = Path(experiment) / 'runs' / arm
        tasks.append({'id': f'{arm}_step00200', 'label': f'{arm} update200', 'role': 'candidate',
            'family': arm, 'step': 200, 'seed': 22, 'architecture': init['architecture'],
            'checkpoint': str(run / 'checkpoints/checkpoint_step00000200.pkl'),
            'source_run_dir': str(run), 'planned_config': str(Path(experiment) / 'configs' / f'{arm}.json'),
            'paired_initialization_id': 'parent', 'endpoint': 'primary'})
    return tasks


def freeze_runtime(workspace, destination, legacy):
    """Freeze current implementation plus the original prepared-array dependency."""
    if destination.exists():
        raise FileExistsError(f'Never refresh an existing snapshot: {destination}')
    for source in _source_files(workspace):
        relative = source.relative_to(workspace)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = file_sha256(source)
        shutil.copy2(source, target)
        if file_sha256(source) != before or file_sha256(target) != before:
            raise ValueError(f'Source changed while copying: {relative}')
    revision = (legacy / 'git_commit.txt').read_text().strip()
    historical = subprocess.check_output(['git', 'show', f'{revision}:{PREPARED_LOADER}'], cwd=workspace)
    (destination / PREPARED_LOADER).write_bytes(historical)
    files = {str(path.relative_to(destination)): file_sha256(path) for path in _source_files(destination)}
    snapshot = dict(format=SOURCE_FORMAT, workspace=str(workspace), files=files, source_digest=_digest(files))
    write(destination / 'snapshot.json', snapshot)
    verify_source(destination)
    return snapshot, {'historical_revision': revision, 'restored_dependency': PREPARED_LOADER,
        'restored_sha256': files[PREPARED_LOADER], 'other_source': 'current workspace frozen at preparation',
        'historical_limit': 'Original source archive omitted dependencies/environment; bitwise historical reproduction is not claimed.'}


def prepare_evaluation(manifest, experiment):
    """Freeze the same eight seasonal 2022 origins for both evaluation stages."""
    import numpy as np
    import xarray as xr
    from scripts.analyze_models import evaluate_v24_amse_regime as evaluator
    from scripts.experiments.run_v24_spectral_pilot import protocol_digest
    from src.data_operations.loaders.zarr_metadata import verify_frozen_zarr_metadata

    reference = Path(manifest['workspace']) / REFERENCE
    inputs, prototypes = [], {}
    for stage in ('exact', 'spectral'):
        path = reference / 'eval' / stage / 'mse_fp32/swa_step00500-02000/specification.json'
        inputs.append(pinned(path))
        prototypes[stage] = read(path)
    exact = prototypes['exact']
    metadata = verify_frozen_zarr_metadata(exact['data_path'], exact['data_metadata'])
    with xr.open_zarr(exact['data_path'], consolidated=True, chunks=None) as dataset:
        times = np.asarray(dataset.time.values, dtype='datetime64[ns]')
    if times.ndim != 1 or not np.all(np.diff(times) == np.timedelta64(6, 'h')):
        raise ValueError('Evaluation data must have a complete six-hourly coordinate')
    year = times[(times >= np.datetime64('2022-01-01')) & (times < np.datetime64('2023-01-01'))]
    starts = np.asarray([np.datetime64(stamp.replace('Z', ''), 'ns') for stamp in DATES])
    required = starts[:, None] + np.arange(-1, 41)[None, :] * np.timedelta64(6, 'h')
    if not np.isin(required, year).all():
        raise ValueError('Seasonal evaluation inputs/forecast leads are not covered')
    # Explicit-manifest evaluation bounds its dataset at the first origin's
    # preceding input frame. chosen_idx is relative to that bounded coordinate.
    bounded = year[(year >= required[0, 0]) & (year <= required[-1, -1])]
    indices = [int(np.flatnonzero(bounded == stamp)[0]) for stamp in starts]
    out = experiment / 'evaluation'
    dates = out / 'initializations.json'
    write(dates, {'schema_version': 1, 'initializations': [{'initialization_time': stamp} for stamp in DATES]})
    date_record = pinned(dates)
    protocols = {}
    excluded = {'protocol_hash', 'model_id', 'checkpoint', 'checkpoint_sha256', 'run_config',
                'experiment_manifest', 'arm', 'tag', 'evaluation_reference', 'source_spectral_protocol'}
    for stage in ('exact', 'spectral'):
        spec = {key: value for key, value in prototypes[stage].items() if key not in excluded}
        spec.update(data_path=exact['data_path'], data_metadata=metadata, evaluation_year=2022,
            n_samples=8, warmup_steps=0, initialization_times=list(DATES),
            initialization_manifest=date_record, initialization_manifest_sha256=date_record['sha256'])
        if stage == 'exact':
            spec['chosen_idx'] = indices
        protocols[stage] = spec
    value = {'schema_version': 1, 'format': 'v24_amse_regime_evaluation_v1',
        'workspace': manifest['workspace'], 'output_dir': str(out),
        'task_specs': tasks_for(manifest, experiment), 'canonical_mse_id': 'parent',
        'protocols': protocols, 'preparation_sources': inputs + [date_record],
        'selection_policy': {'selection_allowed': False, 'automatic_extension': False},
        'evaluation_role': 'development',
        'bootstrap_grouping': [{'initialization_time': stamp,
            'group': evaluator.bootstrap_identity(stamp)[0], 'stratum': evaluator.bootstrap_identity(stamp)[1]} for stamp in DATES],
        'observer_factory': 'src.models.mamba.v24_Ilya.spherical_observer:make_full_field_observer',
        'scope': 'Eight matched seasonal2022 starts,40 six-hour leads; exact GC, power and coherence. Pilot screen, not independent confirmation.',
        'limitations': 'Small development cohort; no selection or automatic extension.'}
    # Freeze planned configurations and all already-existing parent inputs.
    for task in value['task_specs']:
        task['run_config'] = str(Path(task['source_run_dir']) / 'run_config.json')
        if task.get('planned_config'):
            task['planned_config_record'] = pinned(task['planned_config'])
        for field in ('checkpoint', 'run_config'):
            if Path(task[field]).exists():
                task[f'{field}_record'] = pinned(task[field])
    value['manifest_hash'] = protocol_digest(value)
    write(out / 'manifest.json', value)
    evaluator.load_manifest(out / 'manifest.json')
    return {'root': exact['data_path'], 'metadata': metadata}, {
        'passed': True, 'samples': 8, 'target_steps': 40, 'chosen_idx': indices,
        'initialization_times': list(DATES), 'required_coordinate_count': int(required.size)}


def prepare(experiment, workspace):
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    experiment, workspace = Path(experiment).resolve(), Path(workspace).resolve()
    if experiment.exists():
        raise FileExistsError(f'Use a fresh campaign root: {experiment}')
    parent = workspace / INITIAL_ROOT
    run = parent / 'runs/di16_joint'
    checkpoint = pinned(run / 'swa/swa_step00500-02000.pkl')
    if checkpoint['sha256'] != INITIAL_SHA256:
        raise ValueError('Original prescribed MSE parent differs')
    source_config = pinned(run / 'run_config.json')
    template_path = parent / 'configs/di16_joint.json'
    template = read(template_path)
    for key in ('prepared_root', 'anchor_manifest_root', 'baseline_checkpoint', 'stats_dir'):
        template['data'][key] = str((workspace / template['data'][key]).resolve())
    architecture = read(source_config['path'])['architecture']
    if architecture['width'] != 512 or architecture['temporal_d_inner'] != 16:
        raise ValueError('Pilot must use width512/di16')
    experiment.mkdir(parents=True)
    for name in ('logs', 'checks', 'submissions'):
        (experiment / name).mkdir()
    inputs = [checkpoint, source_config, pinned(template_path), pinned(parent / 'source/git_commit.txt')]
    write(experiment / 'templates/di16.json', template)
    inputs.append(pinned(experiment / 'templates/di16.json'))
    for arm in ARMS:
        path = experiment / 'configs' / f'{arm}.json'
        write(path, make_config(template, experiment, arm))
        load_training_config(path)
        inputs.append(pinned(path))
    qualifier = experiment / 'configs/qualification.json'
    write(qualifier, make_config(template, experiment, 'mgda', qualification=True))
    load_training_config(qualifier)
    inputs.append(pinned(qualifier))
    data = template['data']
    inputs += [pinned(data['baseline_checkpoint']), pinned(Path(data['prepared_root']) / 'metadata.json')]
    for relative in ('metadata.json', 'anchors/anchor_indices.npy', 'anchors/anchor_times.npy', 'anchors/split_train.npy', 'anchors/split_val.npy'):
        inputs.append(pinned(Path(data['anchor_manifest_root']) / relative))
    for name in ('stddev_by_level.nc', 'mean_by_level.nc', 'diffs_stddev_by_level.nc'):
        inputs.append(pinned(Path(data['stats_dir']) / name))
    initialization = {'checkpoint': checkpoint, 'run_config': source_config,
                      'source_run_dir': str(run), 'architecture': architecture}
    m = {'workspace': str(workspace), 'initialization': initialization}
    evaluation_data, coverage = prepare_evaluation(m, experiment)
    write(experiment / 'checks/coordinate_coverage.json', coverage)
    inputs += [pinned(experiment / 'evaluation/manifest.json'), pinned(experiment / 'checks/coordinate_coverage.json')]
    source, provenance = freeze_runtime(workspace, experiment / 'source', parent / 'source')
    write(experiment / 'source_provenance.json', provenance)
    inputs.append(pinned(experiment / 'source_provenance.json'))
    manifest = {'format': FORMAT, **m, 'experiment': str(experiment),
        'source_root': str(experiment / 'source'), 'source_digest': source['source_digest'],
        'inputs': inputs, 'evaluation_data': [evaluation_data],
        'policy': {'arms': list(ARMS), 'seed': 22, 'updates': 200, 'qualification_batches': 4,
            'fresh_optimizer_rng_cursor': True, 'zero_state_on_init': True,
            'checkpoint_every': 50, 'swa': False, 'calibration': False,
            'automatic_retries': 0, 'automatic_extension': False, 'terminal_stage': 'report',
            'objective_pair': ['exact_gc_grid_mse', 'spectral_amse_minus_spectral_mse'],
            'qualification_advances_production': False, 'backtracking': BACKTRACKING},
        'resources': {'qualification': {'memory': 320, 'minutes': 240},
            'historical_peak_rss_gib': 234.71, 'max_train_minutes': 2880,
            'evidence': {'training_job': '15515799', 'elapsed': '22:53:09/1900updates',
                'peak_rss_kib': 246115856, 'evaluation_jobs': ['15515801', '15515802']}},
        'runtime': {'xla_flags': '--xla_gpu_enable_triton_gemm=false', 'python_safe_path': True}}
    seal(experiment / 'manifest.json', manifest)
    return verify(experiment)


def verify(experiment, *, frozen=False):
    experiment = Path(experiment).resolve()
    manifest = unseal(experiment / 'manifest.json')
    if manifest['format'] != FORMAT or manifest['experiment'] != str(experiment):
        raise ValueError('Unknown or moved pilot')
    if frozen and ROOT != Path(manifest['source_root']):
        raise ValueError('Execute this stage from the frozen source')
    if verify_source(manifest['source_root'])['source_digest'] != manifest['source_digest']:
        raise ValueError('Frozen source changed')
    for record in manifest['inputs']:
        check(record)
    from src.data_operations.loaders.zarr_metadata import verify_frozen_zarr_metadata
    for data in manifest['evaluation_data']:
        verify_frozen_zarr_metadata(data['root'], data['metadata'])
    return manifest


def qualification_proof(experiment, manifest):
    result = unseal(experiment / 'checks/qualification.json')
    for key, expected in {'passed': True, 'batches': 4, 'manifest_hash': manifest['record_hash'],
        'source_digest': manifest['source_digest'], 'production_updates': 0}.items():
        if result.get(key) != expected:
            raise ValueError(f'Qualification proof differs: {key}')
    check(result['raw_qualification'])
    return result


def qualify(experiment):
    manifest = verify(experiment, frozen=True)
    gpu_required()
    raw = experiment / 'qualification/qualification.json'
    if raw.exists() or (experiment / 'qualification/mgda').exists():
        raise FileExistsError('Partial qualification requires diagnosis; no automatic retry')
    from src.models.mamba.v24_Ilya.training.runner import qualify_mgda_training
    qualify_mgda_training(experiment / 'configs/qualification.json',
        Path(manifest['initialization']['checkpoint']['path']), raw, batches=4)
    result = read(raw)
    if (result.get('passed') is not True or result.get('batches') != 4
            or result.get('parameters_updated') is not False or result.get('optimizer_updates') != 0):
        raise ValueError('Fixed-batch numerical/step qualification did not pass')
    if len(result.get('records', [])) != 4:
        raise ValueError('Qualification lacks the four fixed-batch records')
    for row in result['records']:
        for feedback in ('baseline', 'closed_loop_sg'):
            validate_update(row[feedback]['diagnostics'], 'mgda')
            validate_acceptance_history(row[feedback]['acceptance_record'], row[feedback]['diagnostics'])
    # These measurements must include regenerated-rollout checks, not gradient time alone.
    for key in ('peak_rss_gib', 'steady_p90_seconds'):
        if not math.isfinite(result.get(key, float('nan'))) or result[key] <= 0:
            raise ValueError(f'Invalid qualification measurement: {key}')
    return seal(experiment / 'checks/qualification.json', {
        **result, 'manifest_hash': manifest['record_hash'], 'source_digest': manifest['source_digest'],
        'production_updates': 0, 'raw_qualification': pinned(raw)})


def resource_budget(report, historical_peak=234.71):
    if report.get('passed') is not True or report.get('batches') != 4:
        raise ValueError('Successful four-batch qualification required')
    peak, seconds = report['peak_rss_gib'], report['steady_p90_seconds']
    if not all(math.isfinite(value) and value > 0 for value in (peak, seconds)):
        raise ValueError('Invalid qualification measurement')
    memory = math.ceil(1.35 * max(historical_peak, peak) / 16) * 16
    minutes = max(60, math.ceil((200 * seconds * 1.4 + 900) / 1800) * 30)
    if memory > 512 or minutes > 2880:
        raise ValueError('Measured pilot exceeds frozen512GiB/48h limit; diagnosis required')
    return {'memory': memory, 'minutes': minutes}


def training_command(experiment, manifest, arm):
    return [sys.executable, '-u', Path(manifest['source_root']) / 'scripts/training/train_v24_Ilya.py',
        '--config', experiment / 'configs' / f'{arm}.json',
        '--init-from', manifest['initialization']['checkpoint']['path'], '--zero-state-on-init']


def validate_update(row, arm):
    if (row.get('objective_pair') != 'mse_amse_excess' or row.get('mgda_direction') != arm
            or row.get('finite') is not True or row.get('descent_valid') is not True
            or row.get('backtracking_accepted') is not True):
        raise ValueError('Wrong, nonfinite or unaccepted training update')
    keys = ('gc_loss', 'amse_excess_loss', 'update_norm', 'learning_rate',
            'objective_1_before', 'objective_2_before', 'objective_1_after', 'objective_2_after',
            'acceptance_threshold_1', 'acceptance_threshold_2',
            'repeatability_span_1', 'repeatability_span_2')
    if not all(key in row and math.isfinite(row[key]) and row[key] >= 0 for key in keys):
        raise ValueError('Missing/nonfinite/negative accepted-update diagnostic')
    if not (0 < row['learning_rate'] <= 1e-6 and row['update_norm'] > 0):
        raise ValueError('Accepted step violates the LR cap or has zero displacement')
    trials, halvings = row.get('backtracking_trials'), row.get('backtracking_halvings')
    if (type(trials) is not int or type(halvings) is not int
            or not 1 <= trials <= 9 or halvings != trials - 1
            or row.get('backtracking_baseline_repeats') != 3):
        raise ValueError('Accepted update lacks bounded backtracking/repeatability proof')
    decreases = [row[f'objective_{i}_before'] - row[f'objective_{i}_after'] for i in (1, 2)]
    floors = [row[f'acceptance_threshold_{i}'] for i in (1, 2)]
    if arm == 'mgda':
        accepted = all(delta > floor for delta, floor in zip(decreases, floors, strict=True))
    elif arm == 'mse_control':
        accepted = decreases[0] > floors[0]
    elif arm == 'fixed_sum':
        accepted = math.fsum(decreases) > math.fsum(floors)
    else:
        raise ValueError('Unknown pilot arm')
    if not accepted:
        raise ValueError('Saved before/after losses violate the arm acceptance criterion')


def validate_acceptance_history(history, diagnostics):
    """Retain every rejected candidate and reconcile it with the committed step."""
    acceptance = history.get('acceptance', {})
    trials = acceptance.get('trials', [])
    if (acceptance.get('accepted') is not True or acceptance.get('mode') != diagnostics['mgda_direction']
            or len(trials) != diagnostics['backtracking_trials']
            or not trials or trials[-1].get('accepted') is not True
            or any(trial.get('accepted') is not False for trial in trials[:-1])):
        raise ValueError('Candidate history lacks the unique final accepted step')
    first_lr = trials[0].get('learning_rate', float('nan'))
    if not math.isclose(first_lr, 1e-6, rel_tol=1e-7, abs_tol=0):
        raise ValueError('Candidate history changed the fixed initial LR cap')
    for index, trial in enumerate(trials):
        if trial.get('index') != index or trial.get('learning_rate') != first_lr * .5 ** index:
            raise ValueError('Candidate history changed the fixed halving schedule')
    expected = {
        'before_losses': [diagnostics[f'objective_{i}_before'] for i in (1, 2)],
        'losses': [diagnostics[f'objective_{i}_after'] for i in (1, 2)],
        'thresholds': [diagnostics[f'acceptance_threshold_{i}'] for i in (1, 2)],
        'learning_rate': diagnostics['learning_rate'],
    }
    if any(acceptance.get(key) != value for key, value in expected.items()):
        raise ValueError('Candidate history disagrees with committed update diagnostics')
    if trials[-1].get('losses') != expected['losses']:
        raise ValueError('Final trial losses disagree with the committed candidate')
    if trials[-1].get('learning_rate') != expected['learning_rate']:
        raise ValueError('Final trial learning rate differs from the committed candidate')
    repeatability = history.get('repeatability', {})
    if repeatability.get('repeats') != 3:
        raise ValueError('Candidate history lacks three unchanged forward repeats')


def validate_run(experiment, arm):
    import jax
    import numpy as np
    from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_training_checkpoint
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    run = experiment / 'runs' / arm
    records = [json.loads(line) for line in (run / 'train_metrics.jsonl').read_text().splitlines() if line.strip()]
    if [row.get('step') for row in records] != list(range(1, 201)):
        raise ValueError('Expected exactly200 canonical updates')
    for row in records:
        validate_update(row, arm)
        if not all(math.isfinite(row[key]) for key in ('loss', 'step_seconds')):
            raise ValueError('Missing/nonfinite training metrics')
    histories = [json.loads(line) for line in (run / 'backtracking_trials.jsonl').read_text().splitlines() if line.strip()]
    if [row.get('step') for row in histories] != list(range(1, 201)):
        raise ValueError('Expected the complete200-update candidate history')
    for history, row in zip(histories, records, strict=True):
        validate_acceptance_history(history, row)
    path = run / 'checkpoints/checkpoint_step00000200.pkl'
    checkpoint = load_v24_Ilya_training_checkpoint(path)
    config = load_training_config(experiment / 'configs' / f'{arm}.json')
    if checkpoint.completed_step != 200 or checkpoint.resolved_training_config != config.to_dict():
        raise ValueError('Checkpoint/config/update index mismatch')
    if int(checkpoint.optimizer_state.count) != 200:
        raise ValueError('Pair-RMS moment count differs')
    for tree in (checkpoint.residual_params, checkpoint.residual_state, checkpoint.optimizer_state):
        if any(not np.isfinite(np.asarray(value)).all() for value in jax.tree_util.tree_leaves(tree)):
            raise ValueError('Nonfinite final checkpoint')
    if checkpoint.baseline_checkpoint_fingerprint != file_sha256(config.baseline_checkpoint):
        raise ValueError('Frozen GC baseline changed')
    return records, path


def train_proof(experiment, manifest, arm):
    result = unseal(experiment / 'checks' / f'{arm}_train.json')
    for key, expected in {'passed': True, 'arm': arm, 'completed_steps': 200,
        'manifest_hash': manifest['record_hash'], 'source_digest': manifest['source_digest']}.items():
        if result.get(key) != expected:
            raise ValueError(f'Training proof differs: {key}')
    check(result['checkpoint'])
    check(result['metrics'])
    check(result['backtracking_history'])
    return result


def train(experiment, task):
    manifest = verify(experiment, frozen=True)
    gpu_required()
    qualification_proof(experiment, manifest)
    arm = ARMS[task]
    run = experiment / 'runs' / arm
    if run.exists():
        raise FileExistsError('Existing training output requires diagnosis; no automatic retry')
    started = time.monotonic()
    call(training_command(experiment, manifest, arm))
    records, checkpoint = validate_run(experiment, arm)
    return seal(experiment / 'checks' / f'{arm}_train.json', {
        'passed': True, 'arm': arm, 'completed_steps': 200,
        'manifest_hash': manifest['record_hash'], 'source_digest': manifest['source_digest'],
        'checkpoint': pinned(checkpoint), 'metrics': pinned(run / 'train_metrics.jsonl'),
        'backtracking_history': pinned(run / 'backtracking_trials.jsonl'),
        'serialization': serialization_check(checkpoint),
        'elapsed_seconds': time.monotonic() - started,
        'peak_rss_gib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2,
        'zero_update_count': sum(row['update_norm'] == 0 for row in records)})


def sbatch(experiment, manifest, stage, *, memory, minutes, gpu=False, array=None, dependency=None):
    if not 1 <= memory <= 512 or not 1 <= minutes <= 2880:
        raise ValueError('Resource request outside bounded pilot')
    if array is not None and (not str(array) or '%' in str(array)):
        raise ValueError('Arrays must be unthrottled')
    args = ['sbatch', '--parsable', '--account=dabanin', f"--chdir={manifest['workspace']}",
        '--kill-on-invalid-dep=yes', '--cpus-per-task=8' if gpu else '--cpus-per-task=2',
        f'--mem={memory}G', f'--time={minutes}', f'--job-name=excess-{stage}',
        f'--output={experiment}/logs/{stage}_%A_%a.out', f'--error={experiment}/logs/{stage}_%A_%a.err']
    if gpu:
        args += ['--gres=gpu:1', '--constraint=gpu80', '--qos=' + ('gpu-short' if minutes <= 480 else 'gpu-medium')]
    if array is not None:
        args.append(f'--array={array}')
    if dependency:
        args.append(f'--dependency=afterok:{dependency}')
    return args + [str(Path(manifest['source_root']) / WRAPPER), stage, str(experiment)]


def submit(experiment):
    manifest = verify(experiment)
    qualifier = submit_once(experiment, 'qualify', sbatch(experiment, manifest, 'qualify',
        **manifest['resources']['qualification'], gpu=True))
    return submit_once(experiment, 'release', sbatch(experiment, manifest, 'release',
        memory=8, minutes=15, dependency=qualifier))


def release(experiment):
    manifest = verify(experiment, frozen=True)
    qualified = qualification_proof(experiment, manifest)
    budget = resource_budget(qualified, manifest['resources']['historical_peak_rss_gib'])
    seal(experiment / 'resources/production.json', {**budget, 'updates_per_arm': 200,
        'measured_from': pinned(experiment / 'checks/qualification.json'),
        'memory_formula': 'ceil(1.35*max(234.71,peak_GiB)/16)*16',
        'minutes_formula': 'max(60,ceil((200*p90_seconds*1.4+900)/1800)*30)'})
    training = submit_once(experiment, 'train', sbatch(experiment, manifest, 'train',
        **budget, gpu=True, array='0-2', dependency=os.environ.get('SLURM_JOB_ID')))
    evaluations = [submit_once(experiment, f'evaluate_{stage}', sbatch(experiment, manifest,
        f'evaluate-{stage}', memory=16, minutes=62, gpu=True, array='0-3', dependency=training))
        for stage in ('exact', 'spectral')]
    return submit_once(experiment, 'report', sbatch(experiment, manifest, 'report',
        memory=12, minutes=30, dependency=':'.join(evaluations)))


def evaluate(experiment, task, stage):
    """Use explicit origins for exact and spectral stages, with identical forecasts."""
    import jax
    from scripts.analyze_models import evaluate_v24_amse_regime as evaluator
    from src.models.mamba.v24_Ilya.config import V24IlyaEvalConfig
    from src.models.mamba.v24_Ilya.evaluation import evaluate_v24_Ilya
    from src.models.mamba.v24_Ilya.spectral_observer import SpectralObserver
    import importlib
    manifest = verify(experiment, frozen=True)
    gpu_required()
    for arm in ARMS:
        train_proof(experiment, manifest, arm)
    path = experiment / 'evaluation/manifest.json'
    evaluation_manifest = evaluator.load_manifest(path)
    item = evaluation_manifest['task_specs'][task]
    spec = evaluator.make_specification(path, task, stage)
    destination = evaluator.evaluation_directory(evaluation_manifest, item, stage)
    if destination.exists():
        raise FileExistsError('Existing evaluation requires diagnosis; no automatic retry')
    write(destination / 'specification.json', spec)
    identity = {key: spec[key] for key in ('protocol_hash', 'model_id', 'checkpoint_sha256')}
    identity['baseline_checkpoint_sha256'] = spec['baseline']['checkpoint_sha256']
    observer = None
    if stage == 'spectral':
        module, function = evaluation_manifest['observer_factory'].split(':', 1)
        observer = evaluator.CompositeObserver([
            SpectralObserver(destination, identity, target_steps=40, export_fields=False),
            evaluator.GlobalFieldObserver(destination, identity, target_steps=40),
            getattr(importlib.import_module(module), function)(spec, destination, identity)])
    cfg = V24IlyaEvalConfig(ckpt=Path(spec['checkpoint']), out_json=destination / 'evaluation.unpublished.json',
        ckpt_in=Path(spec['baseline']['path']), stats_dir=Path(spec['stats_dir']),
        data_path=spec['data_path'], val_year=2022, train_start_year=2015, train_end_year=2021,
        target_steps=40, n_samples=8, warmup_steps=0, omit_rms_bias=True,
        initialization_manifest=Path(spec['initialization_manifest']['path']),
        model_id=spec['model_id'], **spec['forecast'], **spec['architecture'])
    started, status = time.monotonic(), 'failed'
    try:
        result = evaluate_v24_Ilya(cfg, prediction_observer=observer)
        result.update(identity)
        evaluator.validate_result(result, spec)
        write(destination / 'evaluation.json', result)
        (destination / 'evaluation.unpublished.json').unlink(missing_ok=True)
        if stage == 'spectral':
            evaluator.load_spectral_result(destination, spec, result)
        status = 'complete'
    finally:
        write(destination / 'resources.json', {**identity, 'status': status,
            'elapsed_seconds': time.monotonic() - started,
            'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2})


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def report(experiment):
    import numpy as np
    from scripts.analyze_models import evaluate_v24_amse_regime as evaluator
    manifest = verify(experiment, frozen=True)
    qualification_proof(experiment, manifest)
    training = {arm: train_proof(experiment, manifest, arm) for arm in ARMS}
    path = experiment / 'evaluation/manifest.json'
    evaluation_manifest = evaluator.load_manifest(path)
    records = {}
    for index, item in enumerate(evaluation_manifest['task_specs']):
        record = {}
        for stage in ('exact', 'spectral'):
            directory, spec, result, reused = evaluator.resolve_result(path, index, stage)
            record[stage] = {'directory': directory, 'specification': spec, 'result': result}
            if stage == 'spectral':
                spectral = evaluator.load_spectral_result(directory, spec, result)
                record['features'] = evaluator.feature_rows(spectral)
        records[item['id']] = record
    parent = records['parent']
    canonical_gc = parent['exact']['result']['original_graphcast_loss']
    parent_loss = canonical_gc['full_rollout']
    summaries, features = [], []
    for item in evaluation_manifest['task_specs']:
        record = records[item['id']]
        loss = record['exact']['result']['original_graphcast_loss']
        own_gc = np.asarray(loss['baseline_per_step'], float)
        canonical = np.asarray(canonical_gc['baseline_per_step'], float)
        power = np.mean([row['mean_relative_power_error_pct'] for row in record['features']])
        coherence = np.mean([row['mean_coherence'] for row in record['features']])
        summaries.append({'id': item['id'], 'arm': item['family'],
            'exact_gc_loss': loss['full_rollout'],
            'exact_gc_gain_pct': 100 * (1 - loss['full_rollout'] / canonical_gc['baseline_rollout']),
            'loss_change_vs_parent_pct': 100 * (loss['full_rollout'] / parent_loss - 1),
            'own_baseline_gain_pct': loss['improvement_pct_rollout'],
            'baseline_max_abs_drift': float(np.max(np.abs(own_gc - canonical))),
            'baseline_relative_l2_drift': float(np.linalg.norm(own_gc - canonical) / np.linalg.norm(canonical)),
            'power_error_pct_feature_day_mean': float(power),
            'coherence_feature_day_mean': float(coherence)})
        for row in record['features']:
            features.append({'id': item['id'], **row})
    out = experiment / 'evaluation/report'
    write_csv(out / 'summary.csv', summaries)
    write_csv(out / 'spectral_features.csv', features)
    output = {'format': FORMAT, 'status': 'complete', 'selection_allowed': False,
        'automatic_extension': False, 'canonical_gc_task': 'parent',
        'cohort': {'initialization_times': list(DATES), 'leads': 40, 'year': 2022, 'role': 'development'},
        'summary': summaries, 'qualification': pinned(experiment / 'checks/qualification.json'),
        'training': training,
        'metric_notes': {'exact_gc_gain_pct': 'Aggregate raw losses first, then100*(1-model/canonical_parent_GC).',
            'power_coherence': 'Secondary diagnostics: equal mean across declared feature/day1,3,7,10 cells; not GC loss.'}}
    write(out / 'pilot_report.json', output)
    lines = ['# GC-MSE / excess-AMSE pilot', '',
        'Eight matched seasonal2022 starts ×40 six-hour leads. Final200-update checkpoints; development screen.', '',
        '| Model | Exact GC loss | Gain vs GC (%) | Loss change vs parent (%) |',
        '|---|---:|---:|---:|']
    lines += [f"| {row['id']} | {row['exact_gc_loss']:.6f} | {row['exact_gc_gain_pct']:.3f} | {row['loss_change_vs_parent_pct']:.3f} |" for row in summaries]
    lines += ['', 'GC gain uses the single original-parent GC execution. Own-baseline sensitivity and drift remain in summary.csv.',
        'Power/coherence diagnostics are in spectral_features.csv. Qualification and train metrics retain rejected trials and numerical floors.',
        'No automatic checkpoint selection, 2,000-update continuation or di32 expansion.']
    (out / 'README.md').write_text('\n'.join(lines) + '\n')
    return seal(experiment / 'complete.json', {'status': 'complete', 'manifest_hash': manifest['record_hash'],
        'terminal_stage': 'report', 'report': pinned(out / 'pilot_report.json'),
        'scientific_success_is_separate': True})


def dry_run(experiment):
    manifest = verify(experiment)
    nodes = []
    def add(name, stage, resources, dependencies=(), **kwargs):
        nodes.append({'id': name, 'depends_on': list(dependencies), 'resources': resources,
            'command': sbatch(experiment, manifest, stage, **resources, **kwargs,
                dependency=':'.join(f'<{value}>' for value in dependencies) or None)})
    add('qualify', 'qualify', {**manifest['resources']['qualification'], 'gpu': True})
    add('release', 'release', {'memory': 8, 'minutes': 15}, ['qualify'])
    nodes.append({'id': 'train', 'depends_on': ['release'], 'array': '0-2', 'updates_per_arm': 200,
        'resources': {'measured_from': 'checks/qualification.json',
            'memory_formula': 'ceil(1.35*max(234.71,peak_GiB)/16)*16',
            'minutes_formula': 'max(60,ceil((200*p90_seconds*1.4+900)/1800)*30)'}})
    for stage in ('exact', 'spectral'):
        add(f'evaluate_{stage}', f'evaluate-{stage}', {'memory': 16, 'minutes': 62, 'gpu': True}, ['train'], array='0-3')
    add('report', 'report', {'memory': 12, 'minutes': 30}, ['evaluate_exact', 'evaluate_spectral'])
    return {'submitted': False, 'nodes': nodes, 'source_digest': manifest['source_digest'],
        'terminal_stage': 'report', 'automatic_extension': False}


def status(experiment):
    manifest = verify(experiment)
    expected = ['checks/qualification.json', *[f'checks/{arm}_train.json' for arm in ARMS],
        *[f'evaluation/results/{item["id"]}/{stage}/evaluation.json'
          for item in tasks_for(manifest, experiment) for stage in ('exact', 'spectral')], 'complete.json']
    return {'campaign': str(experiment), 'complete': (experiment / 'complete.json').exists(),
        'missing_outputs': [path for path in expected if not (experiment / path).exists()],
        'submissions': [{'name': path.stem, **read(path)} for path in sorted((experiment / 'submissions').glob('*.json'))],
        'failures': [read(path) for path in sorted((experiment / 'failures').glob('*.json'))]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'dry-run', 'submit', 'qualify', 'release', 'train',
        'evaluate-exact', 'evaluate-spectral', 'report', 'status'))
    parser.add_argument('--experiment', type=Path, default=Path(DEFAULT_EXPERIMENT))
    parser.add_argument('--workspace', type=Path, default=Path(os.environ.get('WEATHER_WORKSPACE', ROOT)))
    parser.add_argument('--task', type=int, default=0)
    args = parser.parse_args(argv)
    experiment = args.experiment.resolve()
    try:
        if args.stage == 'train' and args.task not in range(3):
            parser.error('--task must identify one of three pilot arms')
        if args.stage.startswith('evaluate-') and args.task not in range(4):
            parser.error('--task must identify parent or one of three final checkpoints')
        if args.stage == 'prepare':
            prepare(experiment, args.workspace)
            result = dry_run(experiment)
        elif args.stage == 'train':
            result = train(experiment, args.task)
        elif args.stage.startswith('evaluate-'):
            result = evaluate(experiment, args.task, args.stage.split('-', 1)[1])
        else:
            result = globals()[args.stage.replace('-', '_')](experiment)
        if result is not None:
            print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        if experiment.exists() and args.stage not in ('prepare', 'dry-run', 'submit', 'status'):
            write(experiment / 'failures' / f'{args.stage}_{args.task}_{os.environ.get("SLURM_JOB_ID", "local")}.json',
                {'stage': args.stage, 'task': args.task, 'error_type': type(error).__name__, 'error': str(error),
                 'diagnostics': getattr(error, 'diagnostics', None)})
        raise


if __name__ == '__main__':
    main()
