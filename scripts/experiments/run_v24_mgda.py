"""Bounded, frozen 2,000-update MGDA/MSE comparison. Submission is explicit.

The final report is terminal. This driver never reads confirmation scores or
releases additional seeds, AMSE continuations, or automatic failure retries.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
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
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.frozen_runtime import file_sha256, freeze_source, verify_source

DEFAULT_EXPERIMENT = 'artifacts/checkpoints/v24_Ilya/res1_mgda_gc_power_2k_20261005'
PRIOR = 'artifacts/checkpoints/v24_Ilya/res1_amse_joint_regime_20261004'
REFERENCE = 'artifacts/checkpoints/v24_Ilya/res1_amse_pair_20260919'
INITIALIZATION = 'artifacts/checkpoints/v24_Ilya/res1_amse_blend_lambda0p1_20260924/runs/mse_fp32/swa/swa_step00500-02000.pkl'
INITIALIZATION_SHA256 = 'aa8043a6f3c03c7d994a9aba0db06dfd93ba5a7072743ce99475fc73f087cd98'
ARMS = ('mgda', 'mse')
SWA_STEPS = tuple(range(1100, 2001, 100))
EVAL_STEPS = (500, 1000, 1500, 2000)
RECOVERY_FILES = ('scripts/experiments/run_v24_mgda.py', 'scripts/experiments/v24_mgda.slurm')


def numerical_policy(manifest):
    policy = manifest.get('numerical_policy', 'strict')
    if policy not in ('strict', 'advisory'):
        raise ValueError('Unknown numerical policy')
    return policy


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def pinned(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': file_sha256(path)}


def check(record):
    if file_sha256(record['path']) != record['sha256']:
        raise ValueError(f"Pinned input changed: {record['path']}")


def write_once(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError(f'Immutable artifact differs: {path}')
    else:
        write(path, value)


def seal(path, value):
    value = copy.deepcopy(value)
    value['record_hash'] = digest(value)
    write_once(path, value)
    return value


def unseal(path):
    value = read(path)
    if value.get('record_hash') != digest({k: v for k, v in value.items() if k != 'record_hash'}):
        raise ValueError(f'Manifest or stage record changed: {path}')
    return value


def call(args):
    args = list(map(str, args))
    print(shlex.join(args), flush=True)
    subprocess.run(args, check=True)


def make_config(template, experiment, arm, calibration, *, smoke=False):
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    config = copy.deepcopy(template)
    config['objective'] = {'loss_mode': 'all_steps', 'loss_kind': 'mse_fp32',
                           'power_calibration': copy.deepcopy(calibration)}
    config['optimizer'].update(optimizer_kind='mgda_rms' if arm == 'mgda' else 'rms_mse',
        max_steps=100 if smoke else 2000, checkpoint_every=50 if smoke else 100,
        learning_rate=1e-5, warmup_start_learning_rate=1e-6, end_learning_rate=3e-6,
        learning_rate_schedule='warmup_cosine_constant', warmup_steps=100, decay_end_step=1000,
        rms_decay=.98, rms_epsilon=1e-8, weight_decay=0., seed=22,
        spatial_lr_multiplier=1., mamba_lr_multiplier=3., grad_clip=1.)
    config['validation']['enabled'] = False
    config['output'] = {'output_root': str(experiment / ('smoke' if smoke else 'runs')), 'run_name': arm}
    return config


def tasks_for(manifest, experiment):
    tasks = [{'id': 'mature_mse', 'label': 'Mature MSE initialization', 'role': 'mse_control',
              'family': 'initialization', 'checkpoint': manifest['initialization']['path'],
              'source_run_dir': manifest['initial_source_run_dir'], 'step': 0}]
    for arm in ARMS:
        for tag, step in [(f'step{x:05d}', x) for x in EVAL_STEPS] + [('swa', 2000)]:
            run = experiment / 'runs' / arm
            checkpoint = run / ('swa/swa_step01100-02000.pkl' if tag == 'swa'
                                else f'checkpoints/checkpoint_step{step:08d}.pkl')
            tasks.append({'id': f'{arm}_{tag}', 'label': f'{arm.upper()} ' + ('late SWA' if tag == 'swa' else f'update {step}'),
                'role': 'candidate', 'family': arm, 'step': step, 'seed': 22,
                'checkpoint': str(checkpoint), 'source_run_dir': str(run),
                'planned_config': str(experiment / 'configs' / f'{arm}.json'),
                'paired_mse_id': f'mse_{tag}',
                'endpoint': 'primary' if tag == 'swa' else 'secondary' if step == 2000 else 'diagnostic'})
    return tasks


def prepare(experiment, workspace):
    """Metadata-only preparation; neither calibrates on GPU nor submits jobs."""
    experiment, workspace = Path(experiment).resolve(), Path(workspace).resolve()
    if experiment.exists():
        raise FileExistsError(f'Use a new campaign directory: {experiment}')
    initialization = pinned(workspace / INITIALIZATION)
    if initialization['sha256'] != INITIALIZATION_SHA256:
        raise ValueError('The prescribed September 24 initialization differs')
    selection = pinned(workspace / PRIOR / 'initialization_selection.json')
    chosen = read(selection['path'])['selected']
    if chosen['checkpoint_sha256'] != initialization['sha256']:
        raise ValueError('Existing initialization selection disagrees')
    template_path = workspace / PRIOR / 'configs/mse.json'
    template = read(template_path)
    inputs = [initialization, selection, pinned(template_path)]
    data = template['data']
    for key in ('prepared_root', 'anchor_manifest_root', 'baseline_checkpoint', 'stats_dir'):
        data[key] = str((workspace / data[key]).resolve())
    inputs += [pinned(data['baseline_checkpoint']), pinned(Path(data['prepared_root']) / 'metadata.json')]
    anchor = Path(data['anchor_manifest_root'])
    inputs += [pinned(anchor / relative) for relative in ('metadata.json', 'anchors/anchor_indices.npy',
               'anchors/anchor_times.npy', 'anchors/split_train.npy', 'anchors/split_val.npy')]
    inputs += [pinned(Path(data['stats_dir']) / name) for name in ('stddev_by_level.nc', 'mean_by_level.nc', 'diffs_stddev_by_level.nc')]
    from src.data_operations.loaders.zarr_metadata import verify_frozen_zarr_metadata
    evaluation_data = []
    for stage in ('exact', 'spectral'):
        path = workspace / REFERENCE / 'eval' / stage / 'mse_fp32/swa_step00500-02000/specification.json'
        inputs.append(pinned(path))
        spec = read(path)
        metadata = verify_frozen_zarr_metadata(spec['data_path'], spec['data_metadata'])
        evaluation_data.append({'root': spec['data_path'], 'metadata': metadata})
        inputs += [spec['initialization_manifest']] if spec.get('initialization_manifest') else []
        if stage == 'spectral':
            dates = spec['initialization_times']
            if len(dates) != 8:
                raise ValueError('Expected eight development starts')
    experiment.mkdir(parents=True)
    # Calibration executes the ordinary FP32 loss path without optimizer updates.
    bootstrap = copy.deepcopy(template)
    bootstrap['optimizer']['max_steps'] = 2000
    bootstrap['output'] = {'output_root': str(experiment / 'calibration'), 'run_name': 'forward_only'}
    write(experiment / 'templates/base.json', template)
    write(experiment / 'configs/calibration.json', bootstrap)
    inputs += [pinned(experiment / 'templates/base.json'), pinned(experiment / 'configs/calibration.json')]
    source = freeze_source(workspace, experiment / 'source')
    manifest = {'format': 'v24_mgda_2k_v1', 'workspace': str(workspace), 'experiment': str(experiment),
        'source_root': str(experiment / 'source'), 'source_digest': source['source_digest'],
        'inputs': inputs, 'evaluation_data': evaluation_data, 'initialization': initialization,
        'initial_source_run_dir': str(Path(initialization['path']).parent.parent),
        'reference_experiment': str(workspace / REFERENCE), 'development_dates': dates,
        'policy': {'arms': list(ARMS), 'seed': 22, 'updates': 2000, 'calibration_batches': 32,
                   'smoke_updates': 100, 'swa_steps': list(SWA_STEPS), 'primary': 'late_swa',
                   'secondary': 'native_2000', 'automatic_retries': 0, 'terminal_stage': 'report',
                   'confirmation': False, 'repeat_seeds': [], 'frozen_feedback': 'closed_loop_sg'},
        'resources': {'historical_peak_rss_gib': 245602608 / 1024**2,
                      'historical_jobs': ['14147422', '14147424', '14397797', '14397799'],
                      'smoke_memory_gib': 384, 'smoke_minutes': 240, 'eval_memory_gib': 16, 'eval_minutes': 150,
                      'smoke_note': '100 uninterrupted updates plus resume50-to100 proof and two compilations'}}
    seal(experiment / 'manifest.json', manifest)
    return verify(experiment)


def verify(experiment, *, frozen=False, resolved=False):
    m = unseal(experiment / 'manifest.json')
    if m['format'] != 'v24_mgda_2k_v1':
        raise ValueError('Unknown campaign format')
    numerical_policy(m)
    source = Path(m['source_root'])
    if frozen and ROOT != source:
        raise ValueError('Execute GPU/campaign stages from the frozen source')
    if verify_source(source)['source_digest'] != m['source_digest']:
        raise ValueError('Frozen source digest differs')
    for record in m['inputs']:
        check(record)
    from src.data_operations.loaders.zarr_metadata import verify_frozen_zarr_metadata
    for data in m['evaluation_data']:
        verify_frozen_zarr_metadata(data['root'], data['metadata'])
    if resolved:
        record = unseal(experiment / 'resolved.json')
        if record['manifest_hash'] != m['record_hash']:
            raise ValueError('Resolved stage belongs to another campaign')
        for artifact in record['artifacts']:
            check(artifact)
    return m


def prepare_recovery(experiment, previous, workspace):
    """Reuse scientifically identical calibration/smokes with new release evidence."""
    import numpy as np
    from scripts.experiments.frozen_runtime import SOURCE_DIRS
    experiment, previous, workspace = (Path(p).resolve() for p in (experiment, previous, workspace))
    if experiment.exists():
        raise FileExistsError(f'Use a new campaign directory: {experiment}')
    old = verify(previous, resolved=True)
    if old['initialization']['sha256'] != INITIALIZATION_SHA256:
        raise ValueError('Recovery initialization differs from prescribed mature MSE')
    # Validate the completed historical training before preparing a new release.
    historical = {arm: validate_run(previous, old, arm, smoke=True) for arm in ARMS}
    experiment.mkdir(parents=True)
    for name in ('logs', 'checks', 'calibration', 'templates'):
        (experiment / name).mkdir()
    shutil.copy2(previous / 'templates/base.json', experiment / 'templates/base.json')
    shutil.copy2(previous / 'calibration/calibration.json', experiment / 'calibration/calibration.json')
    with tempfile.TemporaryDirectory(prefix='mgda-recovery-source-') as temporary:
        staging = Path(temporary)
        for relative in SOURCE_DIRS:
            shutil.copytree(previous / 'source' / relative, staging / relative)
        for relative in RECOVERY_FILES:
            shutil.copy2(ROOT / relative, staging / relative)
        snapshot = freeze_source(staging, experiment / 'source')
        snapshot['workspace'] = str(workspace)
        write(experiment / 'source/snapshot.json', snapshot)
    m = {key: copy.deepcopy(value) for key, value in old.items() if key != 'record_hash'}
    m.update(experiment=str(experiment), workspace=str(workspace), source_root=str(experiment / 'source'),
             source_digest=snapshot['source_digest'], numerical_policy='advisory',
             runtime={'xla_flags': '--xla_gpu_enable_triton_gemm=false',
                      'xla_python_client_preallocate': 'false', 'xla_python_client_allocator': 'platform'},
             reused_smoke={'experiment': str(previous), 'manifest': pinned(previous / 'manifest.json'),
                          'resolved': pinned(previous / 'resolved.json'),
                          'source_digest': old['source_digest'], 'patched_files': list(RECOVERY_FILES)},
             recovery_reason='Finite 100-update trajectories completed; bitwise recomputed trajectories differ.')
    m['inputs'] += [pinned(previous / 'manifest.json'), pinned(previous / 'resolved.json'),
                    pinned(previous / 'calibration/calibration.json'), pinned(experiment / 'templates/base.json')]
    m['resources']['recovery_budgets'] = {'mgda': {'memory_gib': 320, 'minutes': 2040},
                                       'mse': {'memory_gib': 320, 'minutes': 1380}}
    # Accounting measures the complete 100-update + replay processes, not a
    # startup failure. Store its exact provenance alongside the reused metrics.
    m['resources']['reused_accounting'] = {
        'mgda': {'job_id': '15069233_0', 'peak_rss_kib': 234003620, 'elapsed_seconds': 8152},
        'mse': {'job_id': '15069233_1', 'peak_rss_kib': 234102500, 'elapsed_seconds': 5884}}
    m = seal(experiment / 'manifest.json', m)
    resolve_configs(experiment, m, experiment / 'calibration/calibration.json')
    for arm in ARMS:
        records, checkpoint = historical[arm]
        old_config = read(previous / 'configs' / f'{arm}_smoke.json')
        new_config = read(experiment / 'configs' / f'{arm}_smoke.json')
        if {k: v for k, v in old_config.items() if k != 'output'} != {
                k: v for k, v in new_config.items() if k != 'output'}:
            raise ValueError('Recovered smoke scientific configuration differs')
        reference = previous / 'checks' / f'{arm}_uninterrupted100.pkl'
        numerical = compare_native(reference, checkpoint, policy='advisory')
        serialization = serialization_check(checkpoint)
        # Test both the initial smoke and its restarted checkpoint at the native
        # format boundary. This is exact disk I/O, with no model computation.
        midpoint = previous / 'smoke' / arm / 'checkpoints/checkpoint_step00000050.pkl'
        midpoint_serialization = serialization_check(midpoint)
        accounting = m['resources']['reused_accounting'][arm]
        proof = {'passed': True, 'integrity_passed': True, 'arm': arm, 'completed_steps': 100,
            'manifest_hash': m['record_hash'], 'source_digest': m['source_digest'],
            'config': pinned(experiment / 'configs' / f'{arm}_smoke.json'),
            'resolved': pinned(experiment / 'resolved.json'), 'checkpoint': pinned(checkpoint),
            'resume_checked': True, 'serialization_checked': True, 'serialization': serialization,
            'midpoint_serialization': midpoint_serialization, 'numerical_comparison': numerical,
            'numerical_policy': 'advisory', 'status': numerical['status'],
            'reuse_manifest': m['reused_smoke']['manifest'],
            'reused_artifacts': [pinned(path) for path in (reference, midpoint,
                previous / 'configs' / f'{arm}_smoke.json', previous / 'smoke' / arm / 'train_metrics.jsonl')],
            'elapsed_seconds': accounting['elapsed_seconds'], 'peak_rss_gib': accounting['peak_rss_kib'] / 1024**2,
            'steady_p90_seconds': float(np.percentile(
                [r.get('end_to_end_seconds', r['step_seconds']) for r in records[4:]], 90))}
        measured = resource_budget(proof, m['resources']['historical_peak_rss_gib'])
        budget = m['resources']['recovery_budgets'][arm]
        if any(measured[key] > budget[key] for key in ('memory_gib', 'minutes')):
            raise ValueError('Recovered smoke evidence exceeds planned resources')
        seal(experiment / 'checks' / f'{arm}_smoke.json', proof)
        validate_completed_proof(experiment, m, arm, smoke=True)
    return verify(experiment, resolved=True)


def gpu_required():
    import jax
    if not any(d.platform == 'gpu' for d in jax.devices()):
        raise RuntimeError('A GPU allocation is required; no CPU fallback')


def calibrate(experiment):
    m = verify(experiment, frozen=True)
    gpu_required()
    if (experiment / 'resolved.json').exists():
        verify(experiment, frozen=True, resolved=True)
        return
    output = experiment / 'calibration/calibration.json'
    if output.exists():
        raise FileExistsError('Partial calibration requires diagnosis; no implicit retry')
    from src.models.mamba.v24_Ilya.training.runner import calibrate_training
    calibrate_training(experiment / 'configs/calibration.json', Path(m['initialization']['path']), output, batches=32)
    resolve_configs(experiment, m, output)


def resolve_configs(experiment, manifest, calibration_path):
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    from scripts.analyze_models.evaluate_v24_amse_regime import prepare_evaluation
    calibration = read(calibration_path)
    template = read(experiment / 'templates/base.json')
    artifacts = [pinned(calibration_path)]
    for arm in ARMS:
        for smoke in (False, True):
            path = experiment / 'configs' / f'{arm}{"_smoke" if smoke else ""}.json'
            write_once(path, make_config(template, experiment, arm, calibration, smoke=smoke))
            load_training_config(path)
            artifacts.append(pinned(path))
    destination = experiment / 'evaluation'
    prepare_evaluation(tasks_for(manifest, experiment), destination, 'mature_mse',
                       reference_experiment=manifest['reference_experiment'], workspace=manifest['workspace'],
                       observer_factory='src.models.mamba.v24_Ilya.spherical_observer:make_observer')
    artifacts.append(pinned(destination / 'manifest.json'))
    seal(experiment / 'resolved.json', {'manifest_hash': manifest['record_hash'], 'artifacts': artifacts})


def resource_budget(report, historical_peak):
    if report.get('passed') is not True or report.get('completed_steps') != 100:
        raise ValueError('Valid 100-update smoke proof required')
    peak, seconds = report['peak_rss_gib'], report['steady_p90_seconds']
    if not all(math.isfinite(x) and x > 0 for x in (peak, seconds)):
        raise ValueError('Invalid resource measurement')
    memory = math.ceil(1.35 * max(historical_peak, peak) / 16) * 16
    minutes = math.ceil(1.35 * (2000 * seconds + 1200) / 1800) * 30
    if memory > 512 or minutes > 6 * 24 * 60:
        raise ValueError('Measured resources exceed campaign bounds')
    return {'memory_gib': memory, 'minutes': minutes}


def validate_run(experiment, m, arm, *, smoke=False):
    import numpy as np
    from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_training_checkpoint
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    config_path = experiment / 'configs' / f'{arm}{"_smoke" if smoke else ""}.json'
    config = load_training_config(config_path)
    directory = experiment / ('smoke' if smoke else 'runs') / arm
    records = [json.loads(line) for line in (directory / 'train_metrics.jsonl').read_text().splitlines()]
    steps = 100 if smoke else 2000
    if [row['step'] for row in records] != list(range(1, steps + 1)):
        raise ValueError('Missing or duplicated training updates')
    for row in records:
        required = ['loss', 'gradient_norm', 'step_seconds', 'gc_loss', 'power_loss',
                    'gc_loss_normalized', 'power_loss_normalized', 'gc_gradient_norm',
                    'combined_gradient_norm', 'clip_factor', 'direction_norm', 'update_norm',
                    'gc_direction_dot', 'gc_displacement_dot', 'mgda_alpha']
        if arm == 'mgda':
            required += ['power_gradient_norm', 'gradient_cosine', 'power_direction_dot',
                         'power_displacement_dot', 'metric_gc_gc', 'metric_gc_power', 'metric_power_power']
        for key in required:
            if key not in row or not math.isfinite(row[key]):
                raise ValueError(f'Nonfinite training diagnostic: {key}')
        if row.get('finite') is not True or row.get('descent_valid') is not True:
            raise ValueError('Missing or failed optimizer descent proof')
        if not 0 <= row['mgda_alpha'] <= 1 or not 0 <= row['clip_factor'] <= 1:
            raise ValueError('Invalid MGDA weight or clipping factor')
        for objective in (('gc', 'power') if arm == 'mgda' else ('gc',)):
            for dot, norm in (('direction_dot', 'direction_norm'), ('displacement_dot', 'update_norm')):
                if row[f'{objective}_{dot}'] < -1e-6 * row[f'{objective}_gradient_norm'] * row[norm]:
                    raise ValueError('Recorded actual/proposed update violates descent')
    path = directory / f'checkpoints/checkpoint_step{steps:08d}.pkl'
    ckpt = load_v24_Ilya_training_checkpoint(path)
    if ckpt.completed_step != steps or ckpt.resolved_training_config != config.to_dict():
        raise ValueError('Native checkpoint/config mismatch')
    if int(ckpt.optimizer_state.count) != steps:
        raise ValueError('Optimizer count differs from completed updates')
    if ckpt.baseline_checkpoint_fingerprint != file_sha256(config.baseline_checkpoint):
        raise ValueError('Frozen baseline fingerprint mismatch')
    if any(not np.all(np.isfinite(value)) for group in ckpt.residual_params.values() for value in group.values()):
        raise ValueError('Nonfinite checkpoint weights')
    return records, path


def training_command(experiment, m, arm, *, smoke=False, resume=None):
    args = [sys.executable, '-u', Path(m['source_root']) / 'scripts/training/train_v24_Ilya.py',
            '--config', experiment / 'configs' / f'{arm}{"_smoke" if smoke else ""}.json']
    if resume:
        args += ['--resume', resume]
    else:
        args += ['--init-from', m['initialization']['path'], '--zero-state-on-init']
    return args


def compare_native(left, right, *, policy='strict'):
    """Check integrity exactly; quantify optional post-compute trajectory drift.

    Serialization always uses strict. Advisory is only for separately computed
    trajectories: floating leaves may differ, integer state/RNG/metadata may not.
    """
    import jax
    import numpy as np
    from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_training_checkpoint
    a, b = (load_v24_Ilya_training_checkpoint(Path(p)) for p in (left, right))
    if policy not in ('strict', 'advisory'):
        raise ValueError('Unknown numerical policy')
    measurements = {}
    for key in ('residual_params', 'optimizer_state', 'residual_state', 'rng_key'):
        x, y = getattr(a, key), getattr(b, key)
        if jax.tree_util.tree_structure(x) != jax.tree_util.tree_structure(y):
            raise ValueError(f'Resume tree differs: {key}')
        equal = True
        for u, v in zip(jax.tree_util.tree_leaves(x), jax.tree_util.tree_leaves(y), strict=True):
            u, v = np.asarray(u), np.asarray(v)
            if u.shape != v.shape or u.dtype != v.dtype:
                raise ValueError(f'Resume leaf shape or dtype differs: {key}')
            if np.issubdtype(u.dtype, np.inexact) and not (np.isfinite(u).all() and np.isfinite(v).all()):
                raise ValueError(f'Nonfinite resume state: {key}')
            if u.dtype.hasobject or np.ascontiguousarray(u).tobytes() != np.ascontiguousarray(v).tobytes():
                equal = False
                if policy == 'strict' or key == 'rng_key' or not np.issubdtype(u.dtype, np.inexact):
                    raise ValueError(f'Resume leaf differs bitwise: {key}')
        from src.models.mamba.v24_Ilya.training.cached_diagnostics import tree_comparison
        measurements[key] = {**tree_comparison(y, x), 'bitwise_equal': equal}
    for key in ('completed_step', 'training_cursor', 'resolved_training_config',
                'baseline_checkpoint_path', 'baseline_checkpoint_fingerprint',
                'anchor_manifest_fingerprint', 'parameter_overlay_metadata'):
        if getattr(a, key) != getattr(b, key):
            raise ValueError(f'Resume metadata differs: {key}')
    identical = all(item['bitwise_equal'] for item in measurements.values())
    return {'integrity_passed': True, 'numerical_policy': policy,
            'numerical_status': 'pass' if identical else 'warning', 'bitwise_equal': identical,
            'status': 'complete' if identical else 'complete_with_warnings', 'measurements': measurements}


def serialization_check(checkpoint):
    """Exercise disk serialization without advancing computation or training."""
    import pickle
    from src.models.mamba.v24_Ilya.checkpoint import atomic_pickle_dump
    with Path(checkpoint).open('rb') as handle:
        payload = pickle.load(handle)
    with tempfile.TemporaryDirectory(prefix='mgda-serialization-') as directory:
        target = Path(directory) / 'checkpoint.pkl'
        atomic_pickle_dump(payload, target)
        result = compare_native(checkpoint, target, policy='strict')
    return {**result, 'kind': 'exact_save_load', 'checkpoint': pinned(checkpoint)}


def validate_completed_proof(experiment, manifest, arm, *, smoke=False):
    """Bind a successful completion proof to this exact campaign and arm."""
    experiment = Path(experiment).resolve()
    if arm not in ARMS:
        raise ValueError('Unknown arm in completion proof')
    steps = 100 if smoke else 2000
    proof = unseal(experiment / 'checks' / f'{arm}_{"smoke" if smoke else "train"}.json')
    if proof.get('passed') is not True or type(proof.get('completed_steps')) is not int:
        raise ValueError('Completion proof has invalid success or update-count types')
    expected = {'passed': True, 'arm': arm, 'completed_steps': steps,
                'manifest_hash': manifest['record_hash'], 'source_digest': manifest['source_digest']}
    for key, value in expected.items():
        if proof.get(key) != value:
            raise ValueError(f'Completion proof {key} differs from requested task')
    if smoke and proof.get('resume_checked') is not True:
        raise ValueError('Smoke completion proof lacks native resume check')
    if numerical_policy(manifest) == 'advisory':
        if proof.get('integrity_passed') is not True or proof.get('serialization_checked') is not True:
            raise ValueError('Completion proof lacks exact serialization/integrity check')
    directory = experiment / ('smoke' if smoke else 'runs') / arm
    expected_paths = {
        'config': experiment / 'configs' / f'{arm}{"_smoke" if smoke else ""}.json',
        'resolved': experiment / 'resolved.json',
        'checkpoint': directory / f'checkpoints/checkpoint_step{steps:08d}.pkl',
    }
    if smoke and manifest.get('reused_smoke'):
        source = manifest['reused_smoke']['experiment']
        expected_paths['checkpoint'] = Path(source) / 'smoke' / arm / 'checkpoints/checkpoint_step00000100.pkl'
        for record in proof.get('reused_artifacts', []):
            check(record)
        if not proof.get('reused_artifacts') or proof.get('reuse_manifest') != manifest['reused_smoke']['manifest']:
            raise ValueError('Completion proof lacks frozen reused-smoke evidence')
        check(proof['reuse_manifest'])
    for key, path in expected_paths.items():
        record = proof.get(key)
        if not isinstance(record, dict) or record.get('path') != str(path.resolve()):
            raise ValueError(f'Completion proof {key} path differs from requested task')
        check(record)
    return proof


def train(experiment, task, *, smoke=False, resume=None):
    m = verify(experiment, frozen=True, resolved=True)
    gpu_required()
    arm = ARMS[task]
    directory = experiment / ('smoke' if smoke else 'runs') / arm
    proof_path = experiment / 'checks' / f'{arm}_{"smoke" if smoke else "train"}.json'
    if proof_path.exists():
        return validate_completed_proof(experiment, m, arm, smoke=smoke)
    if directory.exists() and resume is None:
        raise FileExistsError('Partial training requires explicit resume')
    if resume is not None and Path(resume).resolve().parent.parent != directory.resolve():
        raise ValueError('Resume checkpoint belongs to another run')
    start = time.monotonic()
    call(training_command(experiment, m, arm, smoke=smoke, resume=resume))
    elapsed = time.monotonic() - start
    records, checkpoint = validate_run(experiment, m, arm, smoke=smoke)
    # Smoke exercises a real process restart: rerun updates 51..100 from the
    # serialized midpoint and compare all state against uninterrupted update100.
    numerical = None
    if smoke:
        reference = experiment / 'checks' / f'{arm}_uninterrupted100.pkl'
        reference.parent.mkdir(exist_ok=True)
        shutil.copy2(checkpoint, reference)
        call(training_command(experiment, m, arm, smoke=True,
                              resume=directory / 'checkpoints/checkpoint_step00000050.pkl'))
        numerical = compare_native(reference, checkpoint, policy=numerical_policy(m))
    serialization = serialization_check(checkpoint)
    import numpy as np
    report = {'passed': True, 'integrity_passed': True, 'serialization_checked': True,
        'serialization': serialization, 'numerical_policy': numerical_policy(m),
        'numerical_comparison': numerical,
        'status': numerical['status'] if numerical else 'complete',
        'arm': arm, 'completed_steps': 100 if smoke else 2000,
        'manifest_hash': m['record_hash'], 'source_digest': m['source_digest'],
        'config': pinned(experiment / 'configs' / f'{arm}{"_smoke" if smoke else ""}.json'),
        'resolved': pinned(experiment / 'resolved.json'), 'checkpoint': pinned(checkpoint),
        'elapsed_seconds': elapsed, 'peak_rss_gib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2,
        'steady_p90_seconds': float(np.percentile([r.get('end_to_end_seconds', r['step_seconds']) for r in records[4:]], 90)),
        'resume_checked': smoke}
    return seal(proof_path, report)


def sbatch(experiment, m, stage, *, memory, minutes, gpu=False, task=0, array=None, dependency=None):
    if not 1 <= memory <= 512 or not 1 <= minutes <= 8640:
        raise ValueError('Resource request outside bounds')
    if array is not None and ('%' in str(array) or not str(array)):
        raise ValueError('Array must be unthrottled')
    args = ['sbatch', '--parsable', '--account=dabanin', f"--chdir={m['workspace']}",
            '--cpus-per-task=8' if gpu else '--cpus-per-task=2', f'--mem={memory}G', f'--time={minutes}',
            f'--job-name=mgda-{stage}', f'--output={experiment}/logs/{stage}_%A_%a.out',
            f'--error={experiment}/logs/{stage}_%A_%a.err']
    if gpu:
        args += ['--gres=gpu:1', '--constraint=gpu80', '--qos=' + ('gpu-short' if minutes <= 480 else 'gpu-medium' if minutes <= 4320 else 'gpu-long')]
    if dependency:
        args.append(f'--dependency=afterok:{dependency}')
    if array is not None:
        args.append(f'--array={array}')
    args += [str(Path(m['source_root']) / 'scripts/experiments/v24_mgda.slurm'), stage, str(experiment)]
    if array is None:
        args.append(str(task))
    return args


def submit_once(experiment, name, args):
    from scripts.experiments.run_v24_amse_pair import submit_job
    directory = experiment / 'submissions'
    directory.mkdir(parents=True, exist_ok=True)
    (experiment / 'logs').mkdir(exist_ok=True)
    with (directory / f'{name}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return submit_job(experiment, name, args)


def dry_run(experiment):
    m = verify(experiment)
    nodes = []
    def add(name, stage, resources, dependencies=(), **kwargs):
        nodes.append({'id': name, 'depends_on': list(dependencies), 'resources': resources,
                      'command': sbatch(experiment, m, stage, **resources, **kwargs,
                                        dependency=':'.join(f'<{x}>' for x in dependencies) or None)})
    if not m.get('reused_smoke'):
        add('calibrate', 'calibrate', {'memory': 384, 'minutes': 180, 'gpu': True})
        add('smoke', 'smoke', {'memory': 384, 'minutes': 240, 'gpu': True}, ['calibrate'], array='0-1')
    add('release', 'release', {'memory': 8, 'minutes': 20}, [] if m.get('reused_smoke') else ['smoke'])
    for index, arm in enumerate(ARMS):
        if m.get('reused_smoke'):
            budget = m['resources']['recovery_budgets'][arm]
            add(f'train_{arm}', 'train', {'memory': budget['memory_gib'], 'minutes': budget['minutes'], 'gpu': True},
                ['release'], task=index)
            continue
        command = sbatch(experiment, m, 'train', memory=384, minutes=1440, gpu=True, task=index)
        command = ['--mem=<measured_GiB>G' if arg.startswith('--mem=') else
                   '--time=<measured_minutes>' if arg.startswith('--time=') else
                   '--qos=<selected_from_measured_time>' if arg.startswith('--qos=') else arg for arg in command]
        nodes.append({'id': f'train_{arm}', 'depends_on': ['release'], 'task': index,
                      'resources': {'measured_from': f'checks/{arm}_smoke.json',
                                    'memory_formula': 'ceil(1.35*max(234.2249, peak_GiB)/16)*16',
                                    'minutes_formula': 'ceil(1.35*(2000*p90_seconds+1200)/1800)*30',
                                    'bounds': {'memory_GiB': 512, 'minutes': 8640}},
                      'command_template': command})
    add('swa', 'swa', {'memory': 16, 'minutes': 20}, ['train_mgda', 'train_mse'], array='0-1')
    add('evaluate', 'evaluate', {'memory': 16, 'minutes': 150, 'gpu': True}, ['swa'], array='0-10')
    add('report', 'report', {'memory': 12, 'minutes': 30}, ['evaluate'])
    return {'manifest': pinned(experiment / 'manifest.json'), 'source_digest': m['source_digest'],
            'submitted': False, 'terminal_stage': 'report', 'nodes': nodes,
            'calibration_pending': not (experiment / 'resolved.json').exists()}


def submit(experiment):
    m = verify(experiment)
    if m.get('reused_smoke'):
        for arm in ARMS:
            validate_completed_proof(experiment, m, arm, smoke=True)
        return submit_once(experiment, 'release', sbatch(experiment, m, 'release', memory=8, minutes=20))
    calibration = submit_once(experiment, 'calibrate', sbatch(experiment, m, 'calibrate', memory=384, minutes=180, gpu=True))
    smoke = submit_once(experiment, 'smoke', sbatch(experiment, m, 'smoke', memory=384, minutes=240,
                          gpu=True, array='0-1', dependency=calibration))
    submit_once(experiment, 'release', sbatch(experiment, m, 'release', memory=8, minutes=20, dependency=smoke))


def release(experiment):
    m = verify(experiment, frozen=True, resolved=True)
    jobs = []
    for index, arm in enumerate(ARMS):
        proof = validate_completed_proof(experiment, m, arm, smoke=True)
        budget = resource_budget(proof, m['resources']['historical_peak_rss_gib'])
        if m.get('reused_smoke'):
            planned = m['resources']['recovery_budgets'][arm]
            if any(budget[key] > planned[key] for key in ('memory_gib', 'minutes')):
                raise ValueError('Measured resources exceed frozen recovery budget')
            budget = planned
        seal(experiment / 'resources' / f'{arm}.json', {**budget, 'evidence': pinned(experiment / 'checks' / f'{arm}_smoke.json')})
        jobs.append(submit_once(experiment, f'train_{arm}', sbatch(experiment, m, 'train', memory=budget['memory_gib'], minutes=budget['minutes'], gpu=True, task=index)))
    swa_job = submit_once(experiment, 'swa', sbatch(experiment, m, 'swa', memory=16, minutes=20, array='0-1', dependency=':'.join(jobs)))
    evaluation = submit_once(experiment, 'evaluate', sbatch(experiment, m, 'evaluate', memory=16, minutes=150, gpu=True, array='0-10', dependency=swa_job))
    submit_once(experiment, 'report', sbatch(experiment, m, 'report', memory=12, minutes=30, dependency=evaluation))


def swa(experiment, task):
    m = verify(experiment, frozen=True, resolved=True)
    arm = ARMS[task]
    validate_completed_proof(experiment, m, arm)
    from src.models.mamba.v24_Ilya.checkpoint import build_v24_Ilya_swa
    run = experiment / 'runs' / arm
    paths = [run / f'checkpoints/checkpoint_step{x:08d}.pkl' for x in SWA_STEPS]
    destination = run / 'swa/swa_step01100-02000.pkl'
    proof_path = experiment / 'checks' / f'{arm}_swa.json'
    if destination.exists():
        if not proof_path.exists():
            raise FileExistsError('Partial SWA needs diagnosis; no implicit overwrite')
        saved = unseal(proof_path)
        if saved['manifest_hash'] != m['record_hash'] or saved['state_policy'] != 'zero':
            raise ValueError('SWA provenance mismatch')
        for artifact in [saved['checkpoint'], *saved['sources']]:
            check(artifact)
        if saved['sources'] != [pinned(p) for p in paths]:
            raise ValueError('SWA source list differs')
        return
    build_v24_Ilya_swa(paths, list(SWA_STEPS), destination, residual_state_policy='zero')
    seal(proof_path, {'manifest_hash': m['record_hash'],
        'sources': [pinned(p) for p in paths], 'checkpoint': pinned(destination), 'state_policy': 'zero'})


def evaluate_worker(experiment, task, attempt, stage):
    verify(experiment, frozen=True, resolved=True)
    gpu_required()
    from scripts.analyze_models import evaluate_v24_amse_regime as evaluator
    if stage not in ('exact', 'spectral') or not attempt.replace('-', '').replace('_', '').isalnum():
        raise ValueError('Invalid evaluation stage or attempt')
    manifest_path = experiment / 'evaluation/manifest.json'
    manifest = evaluator.load_manifest(manifest_path)
    item = manifest['task_specs'][task]
    destination = experiment / 'evaluation/attempts' / f'{item["id"]}_{attempt}'
    original = evaluator.evaluation_directory
    evaluator.evaluation_directory = lambda m, t, selected: destination / selected
    try:
        evaluator.evaluate_worker(manifest_path, task, stage)
    finally:
        evaluator.evaluation_directory = original


def evaluate(experiment, task, attempt='initial'):
    """Each explicit retry uses isolated output; only complete tasks are promoted."""
    m = verify(experiment, frozen=True, resolved=True)
    from scripts.analyze_models import evaluate_v24_amse_regime as evaluator
    if not attempt.replace('-', '').replace('_', '').isalnum():
        raise ValueError('Unsafe attempt name')
    manifest_path = experiment / 'evaluation/manifest.json'
    manifest = evaluator.load_manifest(manifest_path)
    item = manifest['task_specs'][task]
    target = experiment / 'evaluation/results' / item['id']
    if target.exists():
        for stage in ('exact', 'spectral'):
            evaluator.resolve_result(manifest_path, task, stage)
        return
    destination = experiment / 'evaluation/attempts' / f'{item["id"]}_{attempt}'
    if destination.exists():
        raise FileExistsError('Attempt already exists; diagnose and use a new explicit attempt')
    # Keep specifications tied to the canonical frozen manifest while redirecting
    # only output placement, never protocol/checkpoint identity.
    original = evaluator.evaluation_directory
    evaluator.evaluation_directory = lambda m, t, stage: destination / stage
    try:
        for stage in ('exact', 'spectral'):
            call([sys.executable, '-u', Path(m['source_root']) / 'scripts/experiments/run_v24_mgda.py',
                  'evaluate-worker', '--experiment', experiment, '--task', task,
                  '--attempt', attempt, '--eval-stage', stage])
            directory, spec, result, _ = evaluator.resolve_result(manifest_path, task, stage)
            if stage == 'spectral':
                spectra = evaluator.load_spectral_result(directory, spec, result)
                if spectra['global_mse'] is None:
                    raise ValueError('Missing physical-field observer outputs')
    finally:
        evaluator.evaluation_directory = original
    target.parent.mkdir(parents=True, exist_ok=True)
    destination.rename(target)


def report(experiment):
    m = verify(experiment, frozen=True, resolved=True)
    from scripts.analyze_models.report_v24_mgda import build_report
    images = Path(m['workspace']) / 'plots/analyze_models/images/resolution_eval' / experiment.name
    result = build_report(experiment / 'evaluation/manifest.json', image_dir=images,
                          training_runs={arm: experiment / 'runs' / arm for arm in ARMS})
    smoke_proofs = {arm: validate_completed_proof(experiment, m, arm, smoke=True) for arm in ARMS}
    numerical_warnings = [arm for arm, proof in smoke_proofs.items() if proof.get('status') == 'complete_with_warnings']
    seal(experiment / 'complete.json', {'terminal_stage': 'report', 'manifest_hash': m['record_hash'],
         'report': pinned(experiment / 'evaluation/report/mgda_report.json'), 'workflow_status': result.get('status'),
         'status': 'complete_with_warnings' if numerical_warnings else 'complete',
         'integrity_passed': True, 'numerical_policy': numerical_policy(m),
         'smoke_numerical_warning_arms': numerical_warnings,
         'smoke_proofs': {arm: pinned(experiment / 'checks' / f'{arm}_smoke.json') for arm in ARMS},
         'primary_endpoint': 'mgda_swa',
         'primary_outcomes': next(row['outcomes'] for row in result['records'] if row['id'] == 'mgda_swa')})


def status(experiment):
    m = verify(experiment)
    submissions = []
    for path in sorted((experiment / 'submissions').glob('*.json')):
        row = read(path)
        submissions.append({'stage': path.stem, 'job_id': row.get('job_id'),
                            'state': 'submitted' if row.get('job_id') else 'unknown_submission_outcome'})
    jobs = [row['job_id'] for row in submissions if row['job_id']]
    if jobs:
        query = subprocess.run(['sacct', '-j', ','.join(jobs), '-X', '--array', '-Pn',
                                '--format=JobID,State,ExitCode'], text=True, capture_output=True)
        queue = subprocess.run(['squeue', '-j', ','.join(jobs), '-h', '-o', '%i|%T|%r'], text=True, capture_output=True)
        accounting = query.stdout.splitlines() if query.returncode == 0 else [query.stderr.strip()]
        live = queue.stdout.splitlines() if queue.returncode == 0 else [queue.stderr.strip()]
    else:
        accounting, live = [], []
    return {'campaign': m['experiment'], 'complete': (experiment / 'complete.json').exists(),
            'calibrated': (experiment / 'resolved.json').exists(), 'submissions': submissions,
            'accounting': accounting, 'queue_with_dependency_reasons': live,
            'partial_training': [arm for arm in ARMS if (experiment / 'runs' / arm).exists()
                                 and not (experiment / 'checks' / f'{arm}_train.json').exists()],
            'terminal_stage': 'report'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'prepare-recovery', 'dry-run', 'submit', 'calibrate', 'smoke', 'release', 'train', 'resume', 'swa', 'evaluate', 'evaluate-worker', 'report', 'status'))
    parser.add_argument('--experiment', type=Path, default=Path(DEFAULT_EXPERIMENT))
    parser.add_argument('--workspace', type=Path, default=Path(os.environ.get('WEATHER_WORKSPACE', ROOT)))
    parser.add_argument('--from-experiment', type=Path, default=Path(DEFAULT_EXPERIMENT))
    parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--attempt', default='initial')
    parser.add_argument('--eval-stage', choices=('exact', 'spectral'))
    args = parser.parse_args(argv)
    experiment = args.experiment.resolve()
    if args.stage in ('train', 'smoke', 'resume', 'swa') and args.task not in (0, 1):
        parser.error('--task must be 0 (MGDA) or 1 (MSE)')
    if args.stage == 'prepare-recovery':
        prepare_recovery(experiment, args.from_experiment, args.workspace)
        print(json.dumps(dry_run(experiment), indent=2))
    elif args.stage == 'prepare':
        prepare(experiment, args.workspace.resolve())
        print(json.dumps(dry_run(experiment), indent=2))
    elif args.stage == 'dry-run':
        print(json.dumps(dry_run(experiment), indent=2))
    elif args.stage == 'status':
        print(json.dumps(status(experiment), indent=2))
    elif args.stage == 'resume':
        if not args.checkpoint:
            parser.error('Explicit resume requires --checkpoint')
        train(experiment, args.task, resume=args.checkpoint)
    elif args.stage in ('train', 'smoke'):
        train(experiment, args.task, smoke=args.stage == 'smoke')
    elif args.stage == 'swa':
        swa(experiment, args.task)
    elif args.stage == 'evaluate':
        evaluate(experiment, args.task, args.attempt)
    elif args.stage == 'evaluate-worker':
        if args.eval_stage is None:
            parser.error('evaluate-worker requires --eval-stage')
        evaluate_worker(experiment, args.task, args.attempt, args.eval_stage)
    else:
        globals()[args.stage](experiment)


if __name__ == '__main__':
    main()
