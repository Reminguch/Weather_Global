#!/usr/bin/env python3
"""Rerun frozen forecasts and score the same fields against two truth versions.

The forecast implementation, checkpoint, input data and numerical policy come
from the original frozen comparison. Only the observer is extended. It retains
common-grid predictions at every lead and native 1-degree fields at repaired
verification times. Original production artifacts are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time

import numpy as np
import xarray as xr


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_module(path):
    spec = importlib.util.spec_from_file_location('frozen_common_grid_repair_reference', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_repair(args):
    protocol = read(args.repair_protocol)
    body = {key: value for key, value in protocol.items() if key != 'protocol_hash'}
    expected = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if protocol.get('protocol_hash') != expected:
        raise ValueError('Repair protocol self-hash differs')
    for key, actual in [('reference_workflow', args.reference_workflow),
                        ('initialization_manifest', args.initializations),
                        ('corrected_common_truth', args.corrected_truth)]:
        if Path(protocol[key]).resolve() != actual.resolve():
            raise ValueError(f'CLI path differs from repair protocol: {key}')
    for key, path in [('reference_protocol_sha256', args.reference_workflow / 'protocol.json'),
                      ('initialization_manifest_sha256', args.initializations),
                      ('wrapper_sha256', Path(__file__))]:
        if protocol[key] != sha256(path):
            raise ValueError(f'File differs from repair protocol: {key}')
    if 'recovery_report' in protocol:
        record = protocol['recovery_report']
        if sha256(Path(record['path'])) != record['sha256']:
            raise ValueError('Independent-source recovery report differs from frozen identity')
    times = protocol['repaired_valid_times']
    if not times or times != sorted(set(times)):
        raise ValueError('Repaired valid times must be unique and sorted')
    return protocol


def validate_frozen_truth(repair, overlay, corrected_directory, evaluator):
    """Bind live receipt checks to the independently frozen repair decision."""
    entries = repair['truth_validation']
    if (len(entries) != len(repair['repaired_valid_times'])
            or sorted(row['valid_time'] for row in entries) != repair['repaired_valid_times']):
        raise ValueError('Frozen truth validation does not cover exactly the declared repaired times')
    expected = {row['valid_time']: {'original_sha256': row['old_sha256'],
                                    'corrected_sha256': row['new_sha256']}
                for row in entries}
    if overlay.expected != expected:
        raise ValueError('Live original/corrected truth differs from frozen repair hashes')
    for valid_time, identity in expected.items():
        path = Path(corrected_directory) / (evaluator.timestamp_id(valid_time) + '.nc')
        receipt = evaluator.verify_ready(path, valid_time, 'common_truth')
        if receipt.get('original_truth_sha256') != identity['original_sha256']:
            raise ValueError(f'Repaired truth receipt does not identify the frozen original: {valid_time}')


def publish_dataset(path, dataset, attributes):
    """Lossless atomic publication, refusing existing completed fields."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Materialize device-backed arrays before serialization, retaining dtypes
    # and coordinates. Do not mutate the forecast arrays used for feedback.
    materialized = dataset.copy(deep=False).load().assign_attrs(attributes)
    encoding = {key: {'zlib': True, 'complevel': 1, 'shuffle': True}
                for key in materialized.data_vars}
    descriptor, temporary_name = tempfile.mkstemp(prefix='.truth-repair-', suffix='.nc', dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        materialized.to_netcdf(temporary, engine='netcdf4', encoding=encoding)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {'path': str(path.resolve()), 'sha256': sha256(path), 'size_bytes': path.stat().st_size}


class OverlayTruth:
    def __init__(self, evaluator, original, corrected_directory, repaired_times):
        self.original = original
        self.corrected = evaluator.CommonTruth(corrected_directory)
        self.repaired_times = set(repaired_times)
        self.expected = {}
        for valid_time in repaired_times:
            old, old_hash = self.original.at(valid_time)
            new, new_hash = self.corrected.at(valid_time)
            if old_hash == new_hash:
                raise ValueError(f'Repaired truth is unchanged: {valid_time}')
            if set(old.data_vars) != set(new.data_vars):
                raise ValueError(f'Repaired truth diagnostics differ: {valid_time}')
            for coordinate in ('lat', 'lon'):
                np.testing.assert_array_equal(old[coordinate], new[coordinate])
            if any(not np.isfinite(np.asarray(new[key])).all() for key in new):
                raise ValueError(f'Nonfinite repaired truth: {valid_time}')
            self.expected[valid_time] = {'original_sha256': old_hash, 'corrected_sha256': new_hash}

    def at(self, valid_time):
        if valid_time in self.repaired_times:
            result, digest = self.corrected.at(valid_time)
            if digest != self.expected[valid_time]['corrected_sha256']:
                raise ValueError(f'Repaired truth changed during inference: {valid_time}')
            return result, digest
        return self.original.at(valid_time)


class RepairObserver:
    def __init__(self, evaluator, original_truth, repaired_truth, helper, steps, output_dir):
        self.evaluator = evaluator
        self.old = evaluator.CommonGridObserver(original_truth, helper, steps)
        self.new = evaluator.CommonGridObserver(repaired_truth, helper, steps)
        self.helper = helper
        self.repaired_times = repaired_truth.repaired_times
        self.output_dir = Path(output_dir)
        self.common_fields = []
        self.native_fields = []

    @property
    def records(self):
        return self.new.records

    def begin_sample(self, metadata):
        self.old.begin_sample(metadata)
        self.new.begin_sample(metadata)

    def score(self, index, predictions):
        # Both calls use the original scoring implementation on the identical
        # native prediction, so regridding and reduction order are preserved.
        self.old.score(index, predictions)
        self.new.score(index, predictions)
        row = self.new.current
        start, valid = row['initialization_time'], row['valid_times'][index]
        lead = row['lead_hours'][index]
        tag = self.evaluator.timestamp_id(start)
        for model, prediction in predictions.items():
            common = self.helper.regrid_diagnostics(prediction)
            path = self.output_dir / 'common_fields' / model / tag / f'lead_{lead:03d}.nc'
            receipt = publish_dataset(path, common, {
                'initialization_time': start, 'valid_time': valid, 'lead_hours': lead,
                'model': model, 'purpose': 'Retained prediction for independent truth rescoring',
            })
            self.common_fields.append({'model': model, 'initialization_time': start,
                                       'valid_time': valid, 'lead_hours': lead, **receipt})

    def update_step(self, index, native_truth, baseline, full):
        self.score(index, {'gc1': baseline, 'm1': full})
        row = self.new.current
        valid = row['valid_times'][index]
        if valid not in self.repaired_times:
            return
        start, lead = row['initialization_time'], row['lead_hours'][index]
        for model, fields in [('gc1', baseline), ('m1', full), ('original_native_truth', native_truth)]:
            path = (self.output_dir / 'native_repair_fields' / model
                    / self.evaluator.timestamp_id(start) / f'lead_{lead:03d}.nc')
            receipt = publish_dataset(path, fields, {
                'initialization_time': start, 'valid_time': valid, 'lead_hours': lead,
                'model': model,
                'purpose': 'Native 1-degree fields retained for later exact GraphCast loss rescoring',
                'truth_status': 'Original native truth is known invalid at this time; not corrected in this rerun',
            })
            self.native_fields.append({'model': model, 'initialization_time': start,
                                       'valid_time': valid, 'lead_hours': lead, **receipt})

    def finish_sample(self):
        self.old.finish_sample()
        self.new.finish_sample()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repair-protocol', type=Path, required=True)
    parser.add_argument('--reference-workflow', type=Path, required=True)
    parser.add_argument('--initializations', type=Path, required=True)
    parser.add_argument('--corrected-truth', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', choices=['m1', 'gc025'], required=True)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    args = parser.parse_args(argv)
    if not 0 <= args.shard_index < args.shard_count:
        parser.error('Require 0 <= shard-index < shard-count')
    args.output_dir = args.output_dir.resolve()
    out = args.output_dir / f'{args.model}_{args.shard_index:02d}.json'
    if out.exists():
        raise FileExistsError(out)
    repair = validate_repair(args)
    reference_path = args.reference_workflow / 'protocol.json'
    reference = read(reference_path)
    evaluator_path = Path(reference['source_root']) / 'scripts/analyze_models/eval_common_grid_2023.py'
    evaluator = load_module(evaluator_path)
    evaluator.validate_comparison_protocol(reference, reference_path)
    original = evaluator.configure_frozen_runtime(reference)
    records = evaluator.records_from_manifest(args.initializations, reference.get('target_steps', 40))
    original_records = evaluator.records_from_manifest(Path(reference['initialization_manifest']))
    if not {row['initialization_time'] for row in records} <= {row['initialization_time'] for row in original_records}:
        raise ValueError('Repair manifest is not a subset of the original evaluation')
    selected = records[args.shard_index::args.shard_count]
    if not selected:
        raise ValueError('Empty repair shard')
    helper = evaluator.load_helper()
    original_truth = evaluator.CommonTruth(evaluator.path_value(reference['common_truth']))
    repaired_truth = OverlayTruth(evaluator, original_truth, args.corrected_truth, repair['repaired_valid_times'])
    validate_frozen_truth(repair, repaired_truth, args.corrected_truth, evaluator)
    observer = RepairObserver(evaluator, original_truth, repaired_truth, helper,
                              reference.get('target_steps', 40), args.output_dir)
    import jax
    if len(jax.devices()) != 1 or jax.devices()[0].platform != 'gpu':
        raise RuntimeError('Repair forecast evaluation requires exactly one allocated GPU')
    started = time.monotonic()
    if args.model == 'm1':
        evidence = evaluator.run_m1(args, reference, original, observer, args.initializations)
    else:
        evidence = evaluator.run_gc025(args, reference, observer, selected)
    expected_starts = [row['initialization_time'] for row in selected]
    if ([row['initialization_time'] for row in observer.records] != expected_starts
            or [row['initialization_time'] for row in observer.old.records] != expected_starts):
        raise ValueError('Repair output coverage differs from its manifest')
    branches = 2 if args.model == 'm1' else 1
    if len(observer.common_fields) != len(selected) * reference.get('target_steps', 40) * branches:
        raise ValueError('Retained common prediction fields are incomplete')
    evaluator.publish_new(out, {
        'schema_version': 1, 'status': 'complete', 'model': args.model,
        'protocol_hash': repair['protocol_hash'], 'protocol_sha256': sha256(args.repair_protocol),
        'reference_protocol_hash': reference['protocol_hash'], 'reference_protocol_sha256': sha256(reference_path),
        'original_protocol_hash': original['protocol_hash'],
        'initialization_manifest_sha256': sha256(args.initializations),
        'wrapper_sha256': sha256(Path(__file__)), 'evaluator_sha256': sha256(evaluator_path),
        'shard_index': args.shard_index, 'shard_count': args.shard_count,
        'per_initialization': observer.records,
        'original_truth_per_initialization': observer.old.records,
        'truth_replacements': repaired_truth.expected,
        'common_prediction_fields': observer.common_fields,
        'native_repair_fields': observer.native_fields,
        'native_loss_status': 'Bridge against original native truth; known November 10 truth defect remains. Not corrected native-loss estimates.',
        'evidence': evidence, 'elapsed_seconds': time.monotonic() - started,
        'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'devices': [str(device) for device in jax.devices()],
        'gpu_kind': jax.devices()[0].device_kind,
    })
    print(f'[truth-repair] completed {out}', flush=True)


if __name__ == '__main__':
    main()
