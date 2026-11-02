#!/usr/bin/env python3
"""Freeze the three-start rerun after independently checking repaired truth."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
STARTS = ['2023-11-03T00:00:00Z', '2023-11-06T00:00:00Z', '2023-11-09T00:00:00Z']
TIMES = [f'2023-11-10T{h}:00:00Z' for h in ('06', '12', '18')]
FIELDS = ['Z500', 'Z850', 'T500', 'Q700', 'Q850']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workflow', type=Path, required=True)
    parser.add_argument('--recovery-report', type=Path, required=True)
    args = parser.parse_args()
    workflow = args.workflow.resolve()
    if (workflow / 'protocol.json').exists():
        raise FileExistsError('A frozen repair protocol already exists')
    parent = ROOT / 'artifacts/evaluations/graphcast_common_grid_2023_20261015'
    parent_path = parent / 'protocol.json'
    protocol = json.loads(parent_path.read_text())
    module_path = Path(protocol['source_root']) / 'scripts/analyze_models/eval_common_grid_2023.py'
    spec = importlib.util.spec_from_file_location('repair_parent_evaluator', module_path)
    evaluator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = evaluator
    spec.loader.exec_module(evaluator)
    evaluator.validate_comparison_protocol(protocol, parent_path)
    report_path = args.recovery_report.resolve()
    report = json.loads(report_path.read_text())
    if report.get('status') != 'complete':
        raise ValueError('Independent-source recovery is not complete')
    overlay = workflow / 'data/common_truth'
    checks = []
    for valid_time in TIMES:
        name = evaluator.timestamp_id(valid_time) + '.nc'
        old_path = Path(protocol['common_truth']) / name
        new_path = overlay / name
        old_receipt = evaluator.verify_ready(old_path, valid_time, 'common_truth')
        new_receipt = evaluator.verify_ready(new_path, valid_time, 'common_truth')
        with xr.open_dataset(old_path) as a, xr.open_dataset(new_path) as b:
            if set(a.data_vars) != set(b.data_vars):
                raise ValueError('Replacement diagnostic coverage changed')
            np.testing.assert_array_equal(a.lat, b.lat)
            np.testing.assert_array_equal(a.lon, b.lon)
            if b.attrs.get('valid_time') != valid_time:
                raise ValueError('Replacement time differs from its file identity')
            changes = {}
            for field in a.data_vars:
                old, new = a[field].values, b[field].values
                if not np.isfinite(new).all() or old.shape != new.shape:
                    raise ValueError(f'Invalid replacement field: {valid_time}/{field}')
                if field in FIELDS:
                    if np.array_equal(old, new):
                        raise ValueError(f'Repair left stale field unchanged: {valid_time}/{field}')
                    changes[field] = float(np.sqrt(np.mean((new - old)**2)))
                else:
                    np.testing.assert_array_equal(old, new)
            checks.append({'valid_time': valid_time, 'old_sha256': old_receipt['sha256'],
                           'new_sha256': new_receipt['sha256'], 'unweighted_field_change_rmse': changes,
                           'unchanged_diagnostics': sorted(set(a.data_vars) - set(FIELDS))})
    initializations = workflow / 'initializations.json'
    write_new(initializations, {'schema_version': 1, 'initializations': STARTS})
    source_files = {}
    for relative in ('scripts/analyze_models/rerun_common_grid_truth_repair.py',
                     'scripts/preprocessing/repair_common_grid_2023_truth_ncar.py',
                     'scripts/experiments/common_grid_truth_repair_gpu.slurm',
                     'scripts/experiments/prepare_common_grid_truth_repair.py'):
        source, target = ROOT / relative, workflow / 'source' / relative
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_files[relative] = sha(target)
    for directory in ('logs', 'shards', 'submissions', 'report'):
        (workflow / directory).mkdir(parents=True, exist_ok=True)
    body = {
        'schema_version': 1, 'experiment': 'common_grid_2023_three_start_truth_repair',
        'prepared_date': '2026-10-18', 'reference_workflow': str(parent),
        'reference_protocol_sha256': sha(parent_path),
        'reference_protocol_hash': protocol['protocol_hash'],
        'initialization_manifest': str(initializations),
        'initialization_manifest_sha256': sha(initializations),
        'initialization_times': STARTS, 'repaired_valid_times': TIMES,
        'repaired_diagnostics': FIELDS, 'corrected_common_truth': str(overlay),
        'recovery_report': {'path': str(report_path), 'sha256': sha(report_path)},
        'truth_validation': checks, 'wrapper_sha256': source_files[
            'scripts/analyze_models/rerun_common_grid_truth_repair.py'],
        'source_files': source_files,
        'models': {'m1': protocol['m1'], 'gc1': protocol['gc1'],
                   'gc025': {'path': protocol['gc025_checkpoint'],
                             'sha256': protocol['gc025_checkpoint_sha256']}},
        'numerics': protocol['numerics'], 'bootstrap': protocol['bootstrap'],
        'resources': {
            'gc025': {'mem': '40G', 'time': '00:45:00', 'starts_per_job': 3,
                      'evidence': '15634902_0: 32m14s/26.45GiB single start; 15642244: 36m20s-38m39s/30.3-31.0GiB for 7-8 starts'},
            'm1': {'mem': '16G', 'time': '00:20:00', 'starts_per_job': 3,
                   'evidence': '15634901_0: 8m57s/7.06GiB single start; 15642243: 23m20s-24m42s/7.42-7.72GiB for 15-16 starts'},
        },
        'analysis_policy': 'Reuse all original records for 119 starts; replace entire records for the 3 declared starts. Same rerun forecasts scored against original and repaired truth to isolate truth changes from inference variation. No missing cells or exclusions in the final 122-start comparison.',
        'native_loss_policy': 'Original native-loss output is a numerical bridge using original truth. Saved native fields permit separate corrected-loss analysis; no automatic replacement of native manuscript values.',
    }
    body['protocol_hash'] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    write_new(workflow / 'protocol.json', body)
    print(json.dumps({'workflow': str(workflow), 'protocol_hash': body['protocol_hash'],
                      'truth_validation': checks}, indent=2))


if __name__ == '__main__':
    main()
