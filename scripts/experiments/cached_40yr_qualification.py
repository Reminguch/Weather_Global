"""Measured GPU qualification and matched evaluations for the cached 40-year campaign.

The campaign driver schedules these functions; this module never submits jobs.
GPU probes run in separate processes so allocator peaks and failures are isolated.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The agreed V22-matched 2022 exact32 protocol, expressed as dates so offsets do
# not change when the downloaded store grows from eight to forty-one years.
EXACT32_INDICES = (4, 23, 47, 57, 104, 241, 246, 369, 384, 422, 548, 693, 699,
                   751, 771, 774, 836, 870, 873, 893, 930, 1006, 1017, 1065,
                   1117, 1130, 1161, 1181, 1191, 1256, 1294, 1338)


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    from src.models.mamba.v24_Ilya.checkpoint import atomic_json_dump
    atomic_json_dump(value, Path(path))


def _write_once(path, value):
    path = Path(path)
    if path.exists():
        if _read(path) != value:
            raise ValueError(f"Conflicting pinned qualification artifact: {path}")
    else:
        _write(path, value)


def _report(campaign, name, report):
    path = Path(campaign['root']) / 'reports' / 'qualification' / f'{name}.json'
    _write(path, report)
    return {**report, 'report_path': str(path)}


def _run(campaign, arguments, log):
    from scripts.experiments.frozen_runtime import frozen_command
    command, kwargs = frozen_command(Path(campaign['source']), Path(campaign['workspace']),
                                     'scripts/experiments/cached_40yr_qualification.py', arguments)
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    if arguments[0] == 'cache-worker':
        kwargs['env']['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false --xla_gpu_deterministic_ops=true'
    with Path(log).open('a') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, **kwargs)
    if result.returncode:
        tail = Path(log).read_text(errors='replace')[-12000:]
        resource = any(word in tail.lower() for word in ('out of memory', 'resource_exhausted',
                       'cuda_error_out_of_memory', 'no visible gpu', 'exactly one allocated gpu',
                       'gpu memory measurements unavailable', 'gpu identity unavailable',
                       'allocated gpu does not match'))
        raise WorkerFailure(result.returncode, str(log), 'resource' if resource else 'integrity', tail[-2000:])


class WorkerFailure(RuntimeError):
    def __init__(self, code, log, kind, tail):
        self.failure_kind = kind
        super().__init__(f'Qualification worker exited {code}; log={log}\n{tail}')


def execute_cache_pilot(campaign: dict, gpu_kind: str) -> dict:
    """Generate/replay the first/last train and validation chunks on this GPU."""
    pilot = Path(campaign['root']) / 'qualification' / f'cache_{gpu_kind}'
    pilot.mkdir(parents=True, exist_ok=True)
    payload = pilot / 'campaign.json'
    _write_once(payload, campaign)
    try:
        _run(campaign, ['cache-worker', '--campaign', str(payload), '--gpu-kind', gpu_kind], pilot / 'worker.log')
        report = _read(pilot / 'pilot_report.json')
        parity_passed = report.get('passed') is True
        peak = float(report.get('peak_gpu_gib', float('nan')))
        fits = math.isfinite(peak) and 0 < peak < 32
        report.update(gpu_kind=gpu_kind, integrity_passed=parity_passed,
                      parity_passed=parity_passed, gpu_memory_passed=fits, passed=parity_passed and fits,
                      failure_kind='integrity' if not parity_passed else None if fits else 'resource')
    except WorkerFailure as error:
        report = {'passed': False, 'integrity_passed': False,
                  'failure_kind': error.failure_kind, 'error': str(error), 'gpu_kind': gpu_kind}
    return _report(campaign, f'cache_{gpu_kind}', report)


def _cache_worker(campaign, gpu_kind):
    from src.models.mamba.v24_Ilya.baseline_cache import build_manifest, generate_shard, load_context
    from src.models.mamba.v24_Ilya.training.cached_diagnostics import require_matched_gpu_environment, memory_snapshot
    runtime = require_matched_gpu_environment('strict')
    runtime['allocation'] = validate_allocated_gpu(gpu_kind)
    pilot = Path(campaign['root']) / 'qualification' / f'cache_{gpu_kind}'
    prior = _read(pilot / 'pilot_report.json') if (pilot / 'pilot_report.json').exists() else None
    if prior is not None and 'peak_gpu_gib' not in prior:
        raise RuntimeError('GPU memory measurements unavailable in existing cache pilot; use a fresh measured pilot')
    context = load_context(Path(campaign['paths']['cache_config']))
    manifest = build_manifest(context, 1, 4)
    generate_shard(context, manifest, pilot, 0, resume=True, parity=True)
    snapshot = memory_snapshot()
    report = _read(pilot / 'pilot_report.json')
    report.update(peak_gpu_gib=max(_gpu_peak(snapshot), float(prior['peak_gpu_gib']) if prior else 0),
                  runtime=runtime, requested_gpu_kind=gpu_kind,
                  gpu_memory_measurement='max(JAX allocator peak, per-process nvidia-smi, exact allocated NVML instance)',
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  slurm_job_gpus=os.environ.get('SLURM_JOB_GPUS'))
    _write(pilot / 'pilot_report.json', report)


_PROCESS_GPU_PEAK = None
_ALLOCATED_GPU_MEMORY = None


def _cuda_nvml_identity():
    """Resolve CUDA ordinal zero to its exact full-GPU or MIG NVML handle.

    cuDeviceGetUuid_v2 identifies the visible MIG compute instance, unlike PCI
    bus IDs or physical-GPU UUIDs which could select a neighboring MIG tenant.
    """
    import ctypes as ct
    import uuid
    cuda, nvml = ct.CDLL('libcuda.so.1'), ct.CDLL('libnvidia-ml.so.1')
    class UUID(ct.Structure):
        _fields_ = [('bytes', ct.c_ubyte * 16)]
    class Memory(ct.Structure):
        _fields_ = [('total', ct.c_ulonglong), ('free', ct.c_ulonglong), ('used', ct.c_ulonglong)]
    def call(library, name, types, *args):
        function = getattr(library, name)
        function.argtypes, function.restype = types, ct.c_int
        result = function(*args)
        if result != 0:
            raise RuntimeError(f'GPU identity unavailable: {name} returned {result}')
    call(cuda, 'cuInit', [ct.c_uint], 0)
    device, identifier = ct.c_int(), UUID()
    call(cuda, 'cuDeviceGet', [ct.POINTER(ct.c_int), ct.c_int], ct.byref(device), 0)
    call(cuda, 'cuDeviceGetUuid_v2', [ct.POINTER(UUID), ct.c_int], ct.byref(identifier), device)
    suffix = str(uuid.UUID(bytes=bytes(identifier.bytes)))
    call(nvml, 'nvmlInit_v2', [])
    handle = ct.c_void_p()
    get_handle = nvml.nvmlDeviceGetHandleByUUID
    get_handle.argtypes, get_handle.restype = [ct.c_char_p, ct.POINTER(ct.c_void_p)], ct.c_int
    matched = None
    for prefix in ('MIG-', 'GPU-'):
        candidate = prefix + suffix
        if get_handle(candidate.encode(), ct.byref(handle)) == 0:
            matched = candidate
            break
    if matched is None:
        raise RuntimeError('GPU identity unavailable: CUDA UUID does not resolve to an NVML device')
    is_mig, name = ct.c_uint(), ct.create_string_buffer(128)
    call(nvml, 'nvmlDeviceIsMigDeviceHandle', [ct.c_void_p, ct.POINTER(ct.c_uint)], handle, ct.byref(is_mig))
    call(nvml, 'nvmlDeviceGetName', [ct.c_void_p, ct.c_char_p, ct.c_uint], handle, name, len(name))
    def memory():
        value = Memory()
        call(nvml, 'nvmlDeviceGetMemoryInfo', [ct.c_void_p, ct.POINTER(Memory)], handle, ct.byref(value))
        return value
    value = memory()
    return ({'uuid': matched, 'name': name.value.decode(), 'is_mig': bool(is_mig.value),
             'total_memory_gib': value.total / 2**30}, lambda: memory().used / 2**30)


def validate_allocated_gpu(expected_kind):
    global _ALLOCATED_GPU_MEMORY
    try:
        identity, memory_reader = _cuda_nvml_identity()
    except (OSError, AttributeError) as error:
        raise RuntimeError(f'GPU identity unavailable: {error}') from error
    actual_kind = 'mig' if identity['is_mig'] else 'full'
    if (expected_kind != actual_kind or 'A100' not in identity['name']
            or not 36 <= identity['total_memory_gib'] <= 42):
        raise RuntimeError(f'Allocated GPU does not match requested {expected_kind} A100 40 GB: {identity}')
    # This handle is scoped to the exact allocated instance; it never includes
    # neighboring tenants on the MIG parent. Also covers nvidia-smi N/A on MIG.
    _ALLOCATED_GPU_MEMORY = memory_reader
    return identity


def parse_process_gpu_memory(output, pid):
    values = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(',')]
        if len(fields) == 2 and fields[0] == str(pid):
            try:
                values.append(float(fields[1]) / 1024)
            except ValueError:
                pass
    return max(values, default=None)


class GpuSampler:
    """Sample only this process, avoiding other tenants of a MIG parent GPU."""
    def __enter__(self):
        self.stop = threading.Event()
        def poll():
            global _PROCESS_GPU_PEAK
            while not self.stop.is_set():
                try:
                    if _ALLOCATED_GPU_MEMORY is not None:
                        value = _ALLOCATED_GPU_MEMORY()
                        _PROCESS_GPU_PEAK = max(_PROCESS_GPU_PEAK or 0, value)
                    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                             '--format=csv,noheader,nounits'],
                                            capture_output=True, text=True, timeout=3)
                    value = parse_process_gpu_memory(result.stdout, os.getpid())
                    if value is not None:
                        _PROCESS_GPU_PEAK = max(_PROCESS_GPU_PEAK or 0, value)
                except (OSError, subprocess.TimeoutExpired, RuntimeError):
                    pass
                self.stop.wait(.25)
        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *unused):
        self.stop.set()
        self.thread.join(timeout=4)


def _gpu_peak(snapshot):
    stats = snapshot.get('gpu_memory', {})
    values = [stats[key] for key in ('peak_bytes_in_use', 'bytes_in_use') if key in stats]
    gib = [value / 2**30 for value in values]
    if _PROCESS_GPU_PEAK is not None:
        gib.append(_PROCESS_GPU_PEAK)
    if not gib:
        raise RuntimeError('GPU memory measurements unavailable; qualification cannot approve 40 GB')
    return max(gib)


def bounded_memory_report(snapshots):
    """Two repeated passes over early/middle/late year chunks must plateau."""
    if len(snapshots) != 6:
        raise ValueError('Bounded-memory probe requires two passes across three years')
    first = max(float(row['VmRSS_gib']) for row in snapshots[:3])
    second = max(float(row['VmRSS_gib']) for row in snapshots[3:])
    return {'passed': second <= first + max(4.0, first * .1),
            'first_pass_peak_gib': first, 'second_pass_peak_gib': second,
            'allowed_growth_gib': max(4.0, first * .1), 'snapshots': snapshots}


def _sample_years(cached):
    import numpy as np
    from src.models.mamba.v24_Ilya.training.cached_diagnostics import memory_snapshot
    train = [row for row in cached.manifest['chunks'] if row['split'] == 'train']
    items = [train[index] for index in (0, len(train) // 2, len(train) - 1)]
    snapshots = []
    for _ in range(2):
        for row in items:
            batch = cached._materialize(split='train', segment_id=row['segment_id'],
                                        segment_offset=row['segment_offset'], next_cursor=None,
                                        include_validation_fields=False)
            # Force actual reads rather than only constructing mmap-backed views.
            for dataset in (*batch.inputs, *batch.targets, *batch.forcings):
                if not all(np.isfinite(variable.values).all() for variable in dataset.data_vars.values()):
                    raise ValueError('Nonfinite representative-year training input')
            del batch
            gc.collect()
            snapshots.append({**memory_snapshot(), 'target_timestamp': row['target_timestamps'][0]})
    return bounded_memory_report(snapshots)


def training_initialization_checks(source, config, init_from=None):
    from src.models.mamba.v24_Ilya.training.cached_parity import _initialization_checks
    if init_from is None:
        return _initialization_checks(source, config)
    import jax
    import numpy as np
    from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_training_checkpoint, file_sha256
    from src.models.mamba.v24_Ilya.training.cached_diagnostics import tree_digest
    from src.models.mamba.v24_Ilya.training.endpoint_step import build_optimizer
    from src.models.mamba.v24_Ilya.training.data import TrainingCursor
    parent = load_v24_Ilya_training_checkpoint(Path(init_from))
    optimizer, _ = build_optimizer(config)
    meta = source.parameter_overlay_metadata
    checks = {
        'parent_identity': meta.get('init_from_sha256') == file_sha256(Path(init_from)),
        'all_parent_weights': tree_digest(source.residual_params) == tree_digest(parent.residual_params),
        'same_architecture': parent.resolved_training_config['architecture'] == config.to_dict()['architecture'],
        'same_baseline': source.baseline_checkpoint_fingerprint == parent.baseline_checkpoint_fingerprint,
        'zero_recurrent_state': all(np.count_nonzero(v) == 0 for v in jax.tree_util.tree_leaves(source.residual_state)),
        'fresh_optimizer': tree_digest(source.optimizer_state) == tree_digest(optimizer.init(source.residual_params)),
        'fresh_cursor': source.training_cursor == TrainingCursor().to_dict(),
        'step_zero': source.completed_step == 0,
        'finite_parameters': all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves(source.residual_params)),
    }
    return {'passed': all(checks.values()), 'checks': checks, 'init_from': str(init_from)}


def _worker_training(campaign, width, gpu_kind, batch_size, *, resume_check=False):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_training_checkpoint
    from src.models.mamba.v24_Ilya.model import build_model_configs
    from src.models.mamba.v24_Ilya.training.cached_batching import compare_batch_lanes
    from src.models.mamba.v24_Ilya.training.cached_checkpoint import (
        build_provenance, validate_cached_checkpoint, write_cached_checkpoint)
    from src.models.mamba.v24_Ilya.training.cached_config import load_cached_training_config
    from src.models.mamba.v24_Ilya.training.cached_data import open_cached_training_data
    from src.models.mamba.v24_Ilya.training.cached_diagnostics import (
        memory_snapshot, require_matched_gpu_environment, tree_digest, tree_comparison, comparison_integrity)
    from src.models.mamba.v24_Ilya.training.cached_parity import (
        _forbid_gc, _initialization_checks, _keys, _nonzero, _raw_integrity)
    from src.models.mamba.v24_Ilya.training.cached_runner import prepare_shared_initialization, execution_metadata
    from src.models.mamba.v24_Ilya.training.cached_stepwise import make_cached_stepwise_train_step
    from src.models.mamba.v24_Ilya.training.cached_validation import run_cached_validation
    from src.models.mamba.v24_Ilya.training.data import TrainingCursor
    from src.models.mamba.v24_Ilya.training.endpoint_step import build_optimizer

    started = time.monotonic()
    runtime = require_matched_gpu_environment(campaign['numerical_policy'])
    runtime['allocation'] = validate_allocated_gpu(gpu_kind)
    probe = Path(campaign['root']) / 'qualification' / f'w{width}_{gpu_kind}' / f'batch{batch_size}'
    probe.mkdir(parents=True, exist_ok=True)
    production = Path(campaign['configs'][str(width)])
    payload = _read(production)
    payload['distributed']['per_device_batch_size'] = batch_size
    payload['output'].update(output_root=str(probe), run_name='run')
    config_path = probe / 'config.json'
    _write_once(config_path, payload)
    shared_dir = (Path(campaign['root']) / 'training' / f'w{width}' / 'shared'
                  if batch_size == 4 else probe / 'shared')
    shared_dir.parent.mkdir(parents=True, exist_ok=True)
    # Full and MIG qualifiers must share precisely the same production init.
    with shared_dir.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        shared = prepare_shared_initialization(production if batch_size == 4 else config_path,
                    shared_dir, numerical_policy=campaign['numerical_policy'],
                    init_from=Path(campaign['init_from']) if campaign.get('init_from') else None)
    cache_root = Path(campaign['paths']['cache_root'])
    cache_hash = _read(cache_root / 'manifest.json')['manifest_sha256']
    resolved = load_cached_training_config(config_path, cache_root=cache_root,
                                           prefetch_batches=0, expected_manifest_sha256=cache_hash)
    config = resolved.common
    cached = open_cached_training_data(resolved, numerical_policy=campaign['numerical_policy'])
    source = load_v24_Ilya_training_checkpoint(shared)
    initialization = training_initialization_checks(source, config, campaign.get('init_from'))
    if not initialization['passed']:
        raise ValueError('Requested training initialization failed its integrity checks')
    execution = execution_metadata(resolved.execution)
    provenance = build_provenance(config, shared, Path(campaign['paths']['cache_root']), execution=execution)
    if resume_check:
        source = validate_cached_checkpoint(probe / 'run/checkpoints/checkpoint_step00000002.pkl', provenance, execution)
    models = build_model_configs(cached.context[1].model_config, config.architecture)
    optimizer, _ = build_optimizer(config)
    trainer = make_cached_stepwise_train_step(config=config, model_config=models.residual,
                                              task_config=cached.context[2], stats=cached.context[3], optimizer=optimizer)
    params, state, opt, rng = source.residual_params, source.residual_state, source.optimizer_state, source.rng_key
    cursor = TrainingCursor.from_mapping(source.training_cursor)
    zero = jax.tree_util.tree_map(jnp.zeros_like, state)
    context = ({}, resolved, cached, source, models, optimizer, trainer)
    memory = _sample_years(cached) if batch_size == 4 and not resume_check else None
    raw = _raw_integrity(context) if batch_size == 4 and not resume_check else None
    if raw is not None and not raw['integrity_passed']:
        raise ValueError('Raw weather/cache source integrity failed')
    setup = time.monotonic() - started
    limit = 3 if resume_check or batch_size != 4 else 70
    records, checkpoints, parity = [], [], None
    iterator = cached.iter_chunks(cursor, prefetch_batches=0, max_chunks=limit-source.completed_step)
    with _forbid_gc():
        try:
            for step in range(source.completed_step + 1, limit + 1):
                tick = time.monotonic()
                if cursor.segment_offset == 0:
                    state = zero
                batch = next(iterator)
                rng, keys = _keys(rng, config.bptt_steps)
                result = trainer(params, state, opt, keys, batch.inputs, batch.targets, batch.forcings)
                jax.block_until_ready(result)
                params, state, opt, loss, grad_norm, components = result
                if not np.isfinite(np.asarray([loss, grad_norm])).all() or not np.isfinite(np.asarray(components)).all():
                    raise ValueError('Nonfinite cached batch training result')
                cursor = batch.next_cursor
                records.append({'step': step, 'seconds': time.monotonic()-tick, 'memory': memory_snapshot()})
                if step in (2, 3) and not resume_check:
                    tick = time.monotonic()
                    checkpoint = probe / 'run/checkpoints' / f'checkpoint_step{step:08d}.pkl'
                    write_cached_checkpoint(checkpoint, config=config, completed_step=step,
                        residual_params=params, residual_state=state, optimizer_state=opt, rng_key=rng, cursor=cursor,
                        baseline_fingerprint=source.baseline_checkpoint_fingerprint,
                        anchor_fingerprint=source.anchor_manifest_fingerprint,
                        overlay_metadata=source.parameter_overlay_metadata, provenance=provenance, execution=execution)
                    saved = validate_cached_checkpoint(checkpoint, provenance, execution)
                    if any(tree_digest(getattr(saved, name)) != tree_digest(value) for name, value in
                            (('residual_params', params), ('residual_state', state), ('optimizer_state', opt), ('rng_key', rng))):
                        raise ValueError('Checkpoint serialization changed numerical state')
                    if saved.training_cursor != cursor.to_dict():
                        raise ValueError('Checkpoint serialization changed lane cursor')
                    checkpoints.append(time.monotonic()-tick)
                if step == 5 and batch_size == 4:
                    test_params, test_state, _ = _nonzero(params, state)
                    parity = compare_batch_lanes(trainer=trainer, params=test_params, state=test_state,
                                                keys=keys, batch=batch, batch_size=batch_size)
                    _write(probe / 'batch_parity.json', parity)
                    if not parity['passed'] or not parity['integrity_passed']:
                        raise ValueError('New batched implementation differs from independent lanes')
                _write(probe / 'updates.json', records)
                del batch, result
        finally:
            iterator.close()
        if resume_check:
            expected = validate_cached_checkpoint(probe / 'run/checkpoints/checkpoint_step00000003.pkl', provenance, execution)
            checks = {name: tree_comparison(actual, getattr(expected, name)) for name, actual in
                      (('residual_params', params), ('residual_state', state), ('optimizer_state', opt), ('rng_key', rng))}
            integrity = all(comparison_integrity(value) for value in checks.values()) and expected.training_cursor == cursor.to_dict()
            _write(probe / 'resume.json', {'integrity_passed': integrity,
                'passed': integrity and all(value['passed'] for value in checks.values()), 'checks': checks,
                'cursor_equal': expected.training_cursor == cursor.to_dict(),
                'peak_host_gib': memory_snapshot()['host_peak_gib'],
                'peak_gpu_gib': _gpu_peak(memory_snapshot())})
            if not integrity:
                raise ValueError('Checkpoint continuation structural integrity failed')
            return
        validation, unchanged = None, True
        if batch_size == 4:
            before = tree_digest((params, state, opt, rng, cursor.to_dict()))
            validation = run_cached_validation(train_step=trainer, cached_data=cached, config=config,
                                               params=params, zero_state=zero, step=limit)
            unchanged = before == tree_digest((params, state, opt, rng, cursor.to_dict()))
            _write(probe / 'validation.json', validation)
            if not unchanged or validation['selected_segments'] != validation['available_segments']:
                raise ValueError('Validation mutated training state or omitted held-out segments')
    snapshots = [record['memory'] for record in records] + [memory_snapshot()]
    _write(probe / 'benchmark.json', {'completed_steps': limit, 'batch_size': batch_size,
        'shared_init': str(shared), 'runtime': runtime, 'initialization': initialization, 'raw_integrity': raw,
        'batch_parity': parity, 'bounded_memory': memory, 'validation_unchanged': unchanged,
        'full_validation_completed': bool(validation),
        'validation_seconds': validation['duration_seconds'] if validation else 0,
        'startup_seconds': setup + records[0]['seconds'],
        'checkpoint_seconds': max(checkpoints, default=0),
        'p90_end_to_end_seconds': float(np.percentile([row['seconds'] for row in records[20:] or records[1:]], 90)),
        'peak_host_gib': max(row['host_peak_gib'] for row in snapshots),
        'peak_gpu_gib': max(_gpu_peak(row) for row in snapshots), 'provenance': provenance})


def execute_training_pilot(campaign: dict, width: int, gpu_kind: str) -> dict:
    root = Path(campaign['root'])
    probe = root / 'qualification' / f'w{width}_{gpu_kind}'
    probe.mkdir(parents=True, exist_ok=True)
    payload = probe / 'campaign.json'
    _write_once(payload, campaign)
    reports = {}
    try:
        for batch_size in (1, 2, 4):
            arguments = ['train-worker', '--campaign', str(payload), '--width', str(width),
                         '--gpu-kind', gpu_kind, '--batch-size', str(batch_size)]
            _run(campaign, arguments, probe / f'batch{batch_size}.log')
            reports[str(batch_size)] = _read(probe / f'batch{batch_size}' / 'benchmark.json')
        _run(campaign, [*arguments, '--resume-check'], probe / 'resume.log')
        resume = _read(probe / 'batch4/resume.json')
        benchmark = reports['4']
        benchmark['peak_gpu_gib'] = max(benchmark['peak_gpu_gib'], resume['peak_gpu_gib'])
        benchmark['peak_host_gib'] = max(benchmark['peak_host_gib'], resume['peak_host_gib'])
        checks = {**benchmark['initialization']['checks'],
                  'raw_cache_integrity': benchmark['raw_integrity']['integrity_passed'],
                  'independent_batch_lanes': benchmark['batch_parity']['passed'],
                  'checkpoint_resume_integrity': resume['integrity_passed'],
                  'validation_preserves_training': benchmark['validation_unchanged'],
                  'full_validation': benchmark['full_validation_completed']}
        bounded = benchmark['bounded_memory']['passed']
        fits = benchmark['peak_gpu_gib'] < 32
        passed = all(checks.values()) and bounded and fits
        integrity_path = probe / 'production_integrity.json'
        if passed:
            _write(integrity_path, {'kind': 'cached_integrity', 'numerical_policy': campaign['numerical_policy'],
                'integrity_passed': True, 'checks': checks, 'provenance': benchmark['provenance'],
                'batch_size': 4, 'numerical_repeatability': resume})
        report = {**benchmark, 'passed': passed, 'integrity_passed': all(checks.values()),
                  'bounded_memory_passed': bounded, 'gpu_memory_passed': fits,
                  'gpu_kind': gpu_kind, 'width': width, 'batch_probes': reports,
                  'integrity_report': str(integrity_path), 'resume': resume,
                  'numerical_warnings': [] if resume['passed'] else ['GPU checkpoint replay tolerance drift'],
                  'failure_kind': None if passed else 'resource' if all(checks.values()) else 'integrity'}
    except WorkerFailure as error:
        report = {'passed': False, 'integrity_passed': False, 'failure_kind': error.failure_kind,
                  'error': str(error), 'gpu_kind': gpu_kind, 'width': width, 'batch_probes': reports}
    return _report(campaign, f'training_w{width}_{gpu_kind}', report)


def select_exact_checkpoint(run_dir):
    """Select aggregate physical GraphCast loss, never legacy BF16 step loss."""
    run_dir = Path(run_dir)
    records = [json.loads(line) for line in (run_dir / 'validation_metrics.jsonl').read_text().splitlines() if line]
    candidates = [row for row in records if row.get('role') == 'fixed_checkpoint'
                  and row.get('selected_segments', 0) == row.get('available_segments', -1)
                  and row.get('selected_segments', 0) > 0]
    if not candidates:
        raise ValueError('No complete held-out 2022 validation checkpoint')
    for row in candidates:
        metric = float(row['original_graphcast_loss']['full_rollout'])
        if not math.isfinite(metric):
            raise ValueError('Nonfinite exact validation loss')
    row = min(candidates, key=lambda value: (float(value['original_graphcast_loss']['full_rollout']), int(value['step'])))
    checkpoint = run_dir / 'checkpoints' / f"checkpoint_step{int(row['step']):08d}.pkl"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return {'step': int(row['step']), 'checkpoint': str(checkpoint),
            'selection_loss': row['original_graphcast_loss']['full_rollout'],
            'selection_metric': 'exact_original_graphcast_loss', 'validation': row}


def exact_initializations():
    origin = dt.datetime(2022, 1, 1, tzinfo=dt.timezone.utc)
    return {'schema_version': 1, 'initializations': [(origin + dt.timedelta(hours=6*index)).isoformat().replace('+00:00', 'Z')
                                                    for index in EXACT32_INDICES]}


def evaluation_arguments(campaign, width, selected, mode, manifest, output):
    config = _read(campaign['configs'][str(width)])
    architecture = config['architecture']
    arguments = ['--ckpt', selected['checkpoint'], '--data-path', campaign['paths']['source_zarr'],
                 '--ckpt-in', campaign['paths']['baseline_checkpoint'], '--stats-dir', campaign['paths']['stats_dir'],
                 '--eval-mode', mode, '--initialization-manifest', str(manifest), '--target-steps', '40',
                 '--n-samples', '32', '--warmup-steps', '0', '--residual-state-init', 'zero', '--seed', '0',
                 '--stream-block-steps', '4', '--omit-rms-bias', '--out-json', str(output)]
    for key, value in architecture.items():
        if isinstance(value, bool):
            if key == 'temporal_conv_bias':
                if not value: arguments.append('--no-temporal-conv-bias')
            elif key == 'temporal_zero_init_out':
                if not value: arguments.append('--no-zero-init-out')
            elif key == 'temporal_stateful':
                arguments.append('--temporal-stateful' if value else '--no-temporal-stateful')
            elif key == 'temporal_bias' and value:
                arguments.append('--temporal-bias')
        elif value is not None:
            arguments += ['--' + key.replace('_', '-'), str(value)]
    return arguments


def execute_evaluation(campaign: dict, width: int) -> dict:
    from scripts.experiments.frozen_runtime import frozen_command
    from src.models.mamba.v24_Ilya.checkpoint import file_sha256
    root = Path(campaign['root'])
    config = _read(campaign['configs'][str(width)])
    selected = select_exact_checkpoint(Path(config['output']['output_root']) / config['output']['run_name'])
    output_root = root / 'evaluation'
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / 'initializations.json'
    _write_once(manifest, exact_initializations())
    results = {}
    for mode in ('cold_bp', 'cold_full'):
        output = output_root / f'w{width}_{mode}.json'
        identity = {'checkpoint_sha256': file_sha256(Path(selected['checkpoint'])),
                    'source_digest': campaign['source_digest'], 'mode': mode,
                    'config_sha256': file_sha256(Path(campaign['configs'][str(width)])),
                    'initializations_sha256': file_sha256(manifest)}
        _write_once(output.with_suffix('.identity.json'), identity)
        if not output.exists():
            command, kwargs = frozen_command(Path(campaign['source']), Path(campaign['workspace']),
                'scripts/analyze_models/eval_v24_Ilya.py',
                evaluation_arguments(campaign, width, selected, mode, manifest, output))
            subprocess.run(command, check=True, **kwargs)
        value = _read(output)
        metric = value['original_graphcast_loss']
        if (value.get('evaluated_samples') != 32 or value['target_steps'] != 40
                or value['ckpt'] != selected['checkpoint'] or len(metric['full_per_step']) != 40):
            raise ValueError('Evaluation output does not match selected checkpoint and exact32 protocol')
        # Take the ratio after aggregation over samples and leads.
        baseline, full = sum(metric['baseline_per_step']), sum(metric['full_per_step'])
        if not math.isfinite(baseline) or not math.isfinite(full) or baseline <= 0:
            raise ValueError('Nonfinite or invalid exact evaluation loss')
        if value.get('initialization_manifest_sha256') != identity['initializations_sha256']:
            raise ValueError('Evaluation used different initialization timestamps')
        results[mode] = {'path': str(output), 'original_graphcast_loss': metric,
                         'improvement_pct': 100*(1-full/baseline)}
    return _report(campaign, f'evaluation_w{width}', {'passed': True, 'width': width,
        'selection': selected, 'protocol': {'starts': 32, 'leads': 40, 'initializations': str(manifest)}, 'modes': results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('worker', choices=('cache-worker', 'train-worker'))
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--gpu-kind', choices=('full', 'mig'), required=True)
    parser.add_argument('--width', type=int, choices=(128, 512))
    parser.add_argument('--batch-size', type=int, choices=(1, 2, 4), default=4)
    parser.add_argument('--resume-check', action='store_true')
    args = parser.parse_args()
    campaign = _read(args.campaign)
    if args.worker == 'cache-worker':
        with GpuSampler():
            _cache_worker(campaign, args.gpu_kind)
    else:
        with GpuSampler():
            _worker_training(campaign, args.width, args.gpu_kind, args.batch_size, resume_check=args.resume_check)


if __name__ == '__main__':
    main()
