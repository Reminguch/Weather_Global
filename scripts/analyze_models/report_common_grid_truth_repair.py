#!/usr/bin/env python3
"""Merge three repaired reruns with 119 untouched starts and regenerate figures."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analyze_models.plot_public_checkpoints_2023_qc import make_plot

REPAIRED_TIMES = [f'2023-11-10T{hour}:00:00Z' for hour in ('06', '12', '18')]
REPAIRED_DIAGNOSTICS = ['Z500', 'Z850', 'T500', 'Q700', 'Q850']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(body, indent=2, allow_nan=False) + '\n')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_repair_inputs(workflow, repair, protocol, evaluator, old_truth):
    """Bind the merge to the same frozen recovery decision used by both jobs."""
    if (repair.get('schema_version') != 1
            or repair['reference_protocol_hash'] != protocol['protocol_hash']
            or repair['repaired_valid_times'] != REPAIRED_TIMES
            or repair['repaired_diagnostics'] != REPAIRED_DIAGNOSTICS
            or repair['numerics'] != protocol['numerics']
            or repair['bootstrap'] != protocol['bootstrap']):
        raise ValueError('Repair protocol changes the declared scope or analysis policy')
    expected_models = {'m1': protocol['m1'], 'gc1': protocol['gc1'],
                       'gc025': {'path': protocol['gc025_checkpoint'],
                                 'sha256': protocol['gc025_checkpoint_sha256']}}
    if repair['models'] != expected_models:
        raise ValueError('Repair protocol changes the fixed model identities')
    manifest = Path(repair['initialization_manifest'])
    if (sha(manifest) != repair['initialization_manifest_sha256']
            or read(manifest) != {'schema_version': 1, 'initializations': repair['initialization_times']}):
        raise ValueError('Repair initialization manifest changed')
    source_root = (workflow / 'source').resolve()
    for relative, digest in repair['source_files'].items():
        path = (source_root / relative).resolve()
        if not path.is_relative_to(source_root) or sha(path) != digest:
            raise ValueError(f'Frozen repair source changed: {relative}')
    wrapper = 'scripts/analyze_models/rerun_common_grid_truth_repair.py'
    if repair['wrapper_sha256'] != repair['source_files'][wrapper]:
        raise ValueError('Repair wrapper identity differs from its source snapshot')
    recovery_path = Path(repair['recovery_report']['path'])
    if sha(recovery_path) != repair['recovery_report']['sha256']:
        raise ValueError('Frozen recovery report changed')
    recovery = read(recovery_path)
    if (recovery.get('status') != 'complete'
            or recovery['repaired_valid_times'] != REPAIRED_TIMES
            or Path(recovery['corrected_common_truth_root']).resolve()
            != Path(repair['corrected_common_truth']).resolve()):
        raise ValueError('Recovery report does not describe the frozen replacement truth')
    entries = repair['truth_validation']
    recovered = recovery['outputs']
    if ([r['valid_time'] for r in entries] != REPAIRED_TIMES
            or [r['timestamp'] for r in recovered] != REPAIRED_TIMES):
        raise ValueError('Repair truth validation coverage must be exact and unique')
    expected = {}
    for entry, recovered_entry in zip(entries, recovered, strict=True):
        valid = entry['valid_time']
        name = evaluator.timestamp_id(valid) + '.nc'
        old_path = evaluator.path_value(protocol['common_truth']) / name
        new_path = Path(repair['corrected_common_truth']) / name
        old_receipt = evaluator.verify_ready(old_path, valid, 'common_truth')
        new_receipt = evaluator.verify_ready(new_path, valid, 'common_truth')
        if (entry['old_sha256'] != old_truth[valid]
                or old_receipt['sha256'] != entry['old_sha256']
                or new_receipt['sha256'] != entry['new_sha256']
                or new_receipt.get('original_truth_sha256') != entry['old_sha256']
                or recovered_entry['sha256'] != entry['new_sha256']
                or Path(recovered_entry['path']).resolve() != new_path.resolve()
                or sha(str(new_path) + '.READY.json') != recovered_entry['ready_receipt_sha256']):
            raise ValueError(f'Repaired truth identity disagrees with its frozen provenance: {valid}')
        with xr.open_dataset(old_path) as old, xr.open_dataset(new_path) as new:
            if (set(old.data_vars) != set(protocol['verification']['diagnostics'])
                    or set(new.data_vars) != set(old.data_vars)
                    or new.attrs.get('valid_time') != valid):
                raise ValueError(f'Repaired truth diagnostic/time coordinates changed: {valid}')
            for coordinate in ('lat', 'lon'):
                np.testing.assert_array_equal(old[coordinate], new[coordinate])
            for diagnostic in old.data_vars:
                a, b = old[diagnostic].values, new[diagnostic].values
                if a.shape != b.shape or not np.isfinite(b).all():
                    raise ValueError(f'Invalid repaired truth: {valid}/{diagnostic}')
                if diagnostic in REPAIRED_DIAGNOSTICS:
                    if np.array_equal(a, b):
                        raise ValueError(f'Declared truth repair is unchanged: {valid}/{diagnostic}')
                else:
                    np.testing.assert_array_equal(a, b)
        expected[valid] = {'original_sha256': entry['old_sha256'],
                           'corrected_sha256': entry['new_sha256']}
    return expected


def validate_retained_fields(records, expected, diagnostics=None):
    """Require complete field coverage, including the timestamp stored in each file."""
    keys = [(r['model'], r['initialization_time'], r['lead_hours'], r['valid_time']) for r in records]
    if set(keys) != expected or len(keys) != len(expected):
        raise ValueError('Retained forecast field coverage mismatch')
    paths = [Path(r['path']).resolve() for r in records]
    if len(set(paths)) != len(paths):
        raise ValueError('Different retained forecast identities share a file')
    for record, path in zip(records, paths, strict=True):
        if path.stat().st_size != record['size_bytes'] or sha(path) != record['sha256']:
            raise ValueError(f'Retained forecast field changed: {path}')
        with xr.open_dataset(path) as fields:
            for key in ('model', 'initialization_time', 'lead_hours', 'valid_time'):
                if fields.attrs.get(key) != record[key]:
                    raise ValueError(f'Retained field metadata differs from its receipt: {path}/{key}')
            if diagnostics is not None and set(fields.data_vars) != set(diagnostics):
                raise ValueError(f'Retained common-grid diagnostic coverage changed: {path}')


def rmse_change_metrics(reference, candidate, indices, starts, leads, diagnostics):
    """Signed physical-RMSE changes; positive values mean larger error."""
    if np.any(reference[indices] <= 0):
        raise ValueError('Relative bridge change is undefined for a zero reference MSE')
    changes = 100 * (np.sqrt(candidate[indices] / reference[indices]) - 1)
    pooled = 100 * (np.sqrt(candidate.mean(axis=0) / reference.mean(axis=0)) - 1)
    worst = np.unravel_index(np.argmax(np.abs(changes)), changes.shape)
    return {'max_abs_per_start_rmse_change_pct': float(np.max(np.abs(changes))),
            'p95_abs_per_start_rmse_change_pct': float(np.percentile(np.abs(changes), 95)),
            'max_abs_full_sample_pooled_rmse_change_pct': float(np.max(np.abs(pooled))),
            'worst_case': {'initialization_time': starts[indices[worst[0]]],
                           'lead_hours': leads[worst[1]], 'diagnostic': diagnostics[worst[2]],
                           'signed_rmse_change_pct': float(changes[worst])},
            'per_start_rmse_change_pct': changes.tolist(),
            'pooled_rmse_change_pct': pooled.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workflow', type=Path, required=True)
    parser.add_argument('--name', default='public_checkpoints_2023_repaired_20261018')
    args = parser.parse_args()
    workflow = args.workflow.resolve()
    repair = read(workflow / 'protocol.json')
    body = {k: v for k, v in repair.items() if k != 'protocol_hash'}
    if hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest() != repair['protocol_hash']:
        raise ValueError('Repair protocol self-hash mismatch')
    parent = Path(repair['reference_workflow'])
    if sha(parent / 'protocol.json') != repair['reference_protocol_sha256']:
        raise ValueError('Parent protocol changed')
    parent_protocol = read(parent / 'protocol.json')
    source_root = Path(parent_protocol['source_root'])
    snapshot = read(source_root / 'snapshot.json')
    digest = hashlib.sha256(json.dumps(snapshot['files'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != parent_protocol['source_digest'] or digest != snapshot['source_digest']:
        raise ValueError('Parent source snapshot identity changed')
    module_path = source_root / 'scripts/experiments/run_common_grid_2023.py'
    evaluator_path = source_root / 'scripts/analyze_models/eval_common_grid_2023.py'
    for path in (module_path, evaluator_path):
        if sha(path) != snapshot['files'][str(path.relative_to(source_root))]:
            raise ValueError(f'Frozen analysis/evaluator changed: {path}')
    evaluator = load_module('frozen_repair_evaluator_check', evaluator_path)
    evaluator.validate_comparison_protocol(parent_protocol, parent / 'protocol.json')
    frozen = load_module('frozen_repair_report', module_path)
    protocol, original, sources = frozen.gather(parent)
    starts = protocol['initialization_times']
    rerun_starts = repair['initialization_times']
    if rerun_starts != ['2023-11-03T00:00:00Z', '2023-11-06T00:00:00Z', '2023-11-09T00:00:00Z']:
        raise ValueError('Unexpected repair initialization selection')
    diagnostics = protocol['verification']['diagnostics']
    old_truth = {}
    for source in sources:
        for row in read(source['path'])['per_initialization']:
            for valid, digest in zip(row['valid_times'], row['truth_sha256'], strict=True):
                if valid in old_truth and old_truth[valid] != digest:
                    raise ValueError('Original truth disagreement')
                old_truth[valid] = digest
    truth_replacements = validate_repair_inputs(workflow, repair, protocol, evaluator, old_truth)
    corrected_hash = {valid: record['corrected_sha256'] for valid, record in truth_replacements.items()}
    affected_starts = [start for start in starts if any(
        (datetime.fromisoformat(start.replace('Z', '+00:00')) + timedelta(hours=h))
        .isoformat().replace('+00:00', 'Z') in corrected_hash for h in protocol['lead_hours'])]
    if affected_starts != rerun_starts:
        raise ValueError('Rerun starts do not cover exactly all forecasts using repaired truth')
    values = {m: cube.copy() for m, cube in original.items()}
    bridge = {m: cube.copy() for m, cube in original.items()}
    rerun_sources, retained_fields, native_fields = [], [], []
    seen = set()
    for model, branches in [('m1', {'gc1', 'm1'}), ('gc025', {'gc025'})]:
        path = workflow / 'shards' / f'{model}_00.json'
        result = read(path)
        if (result.get('schema_version') != 1 or result.get('status') != 'complete' or result.get('model') != model
                or result.get('protocol_hash') != repair['protocol_hash']
                or result.get('protocol_sha256') != sha(workflow / 'protocol.json')
                or result.get('reference_protocol_hash') != protocol['protocol_hash']
                or result.get('reference_protocol_sha256') != repair['reference_protocol_sha256']
                or result.get('original_protocol_hash') != protocol['original_protocol_hash']
                or result.get('wrapper_sha256') != repair['wrapper_sha256']
                or result.get('evaluator_sha256') != sha(evaluator_path)
                or result.get('initialization_manifest_sha256') != repair['initialization_manifest_sha256']
                or result.get('truth_replacements') != truth_replacements
                or result.get('shard_count') != 1 or result.get('shard_index') != 0):
            raise ValueError(f'Rerun identity mismatch: {path}')
        expected_checkpoint = protocol['m1']['checkpoint_sha256'] if model == 'm1' else protocol['gc025_checkpoint_sha256']
        if result['evidence']['checkpoint_sha256'] != expected_checkpoint:
            raise ValueError('Rerun checkpoint changed')
        if model == 'm1' and result['evidence']['baseline_checkpoint_sha256'] != protocol['gc1']['checkpoint_sha256']:
            raise ValueError('Rerun paired backbone changed')
        new_rows, old_rows = result['per_initialization'], result['original_truth_per_initialization']
        for rows in (new_rows, old_rows):
            if [r['initialization_time'] for r in rows] != rerun_starts:
                raise ValueError('Rerun initialization coverage mismatch')
        for new, old in zip(new_rows, old_rows, strict=True):
            start = new['initialization_time']
            index = starts.index(start)
            expected_times = [(datetime.fromisoformat(start.replace('Z', '+00:00')) + timedelta(hours=h))
                              .isoformat().replace('+00:00', 'Z') for h in protocol['lead_hours']]
            for record, is_new, destination in ((new, True, values), (old, False, bridge)):
                if (record['valid_times'] != expected_times or record['lead_hours'] != protocol['lead_hours']
                        or set(record['mse']) != branches):
                    raise ValueError('Rerun scoring coordinates changed')
                hashes = [corrected_hash.get(t, old_truth[t]) if is_new else old_truth[t] for t in expected_times]
                if record['truth_sha256'] != hashes:
                    raise ValueError('Rerun truth identity mismatch')
                for branch in branches:
                    if set(record['mse'][branch]) != set(diagnostics):
                        raise ValueError('Rerun diagnostic coverage changed')
                    array = np.asarray([record['mse'][branch][d] for d in diagnostics]).T
                    if array.shape != (40, 12) or not np.isfinite(array).all() or np.any(array < 0):
                        raise ValueError('Incomplete rerun score array')
                    destination[branch][index] = array
                    if is_new:
                        if (branch, start) in seen:
                            raise ValueError('Duplicate rerun record')
                        seen.add((branch, start))
            for branch in branches:
                for j, valid in enumerate(expected_times):
                    for k, diagnostic in enumerate(diagnostics):
                        if valid not in corrected_hash or diagnostic not in repair['repaired_diagnostics']:
                            if values[branch][index, j, k] != bridge[branch][index, j, k]:
                                raise ValueError('Truth repair changed an unrelated score')
        common = result['common_prediction_fields']
        expected_fields = {(b, row['initialization_time'], h, valid)
                           for b in branches for row in new_rows
                           for h, valid in zip(row['lead_hours'], row['valid_times'], strict=True)}
        validate_retained_fields(common, expected_fields, diagnostics)
        expected_native = {(b, start, h, valid)
                           for _, start, h, valid in expected_fields if valid in corrected_hash
                           for b in ('gc1', 'm1', 'original_native_truth')} if model == 'm1' else set()
        validate_retained_fields(result['native_repair_fields'], expected_native)
        retained_fields += common
        native_fields += result['native_repair_fields']
        rerun_sources.append(identity(path))
    if seen != {(m, t) for m in values for t in rerun_starts}:
        raise ValueError('Not every model has every rerun initialization')
    reused_indices = [i for i, start in enumerate(starts) if start not in rerun_starts]
    rerun_indices = [starts.index(start) for start in rerun_starts]
    for model, cube in values.items():
        if cube.shape != (122, 40, 12) or not np.isfinite(cube).all():
            raise ValueError('Final analysis must be complete without exclusions')
        np.testing.assert_array_equal(cube[reused_indices], original[model][reused_indices])
    bridge_metrics, truth_effect_metrics = {}, {}
    for model in values:
        bridge_metrics[model] = rmse_change_metrics(original[model], bridge[model], rerun_indices,
                                                   starts, protocol['lead_hours'], diagnostics)
        truth_effect_metrics[model] = rmse_change_metrics(bridge[model], values[model], rerun_indices,
                                                        starts, protocol['lead_hours'], diagnostics)
    rmse, analyses = frozen.analyze_values(values, starts, protocol['bootstrap'],
                                          protocol['prior_exposure']['sampled_initialization_times'])
    tables = ROOT / 'plots/analyze_models/data/resolution_eval' / args.name
    images = ROOT / 'plots/analyze_models/images/resolution_eval' / args.name
    tables.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    csv_path = tables / 'per_initialization_mse.csv'
    with csv_path.open('x') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'initialization_time', 'lead_hours', 'diagnostic', 'mse_si', 'forecast_origin'])
        for model, cube in values.items():
            for i, start in enumerate(starts):
                for j, lead in enumerate(protocol['lead_hours']):
                    for k, diagnostic in enumerate(diagnostics):
                        writer.writerow([model, start, lead, diagnostic, cube[i, j, k],
                                         'rerun_with_repaired_truth' if start in rerun_starts else 'original'])
    outputs = [identity(csv_path)]
    for filename, selected, full in [('fig5_public_checkpoints_2023_repaired', ['Z500', 'T2m', 'U850'], False),
                                      ('fig9_public_checkpoints_2023_repaired', diagnostics, True)]:
        for ext in ('pdf', 'png'):
            if (images / (filename + '.' + ext)).exists():
                raise FileExistsError('Refusing to overwrite repaired figure')
        outputs += make_plot(images, filename, selected, diagnostics,
                             np.asarray(protocol['lead_hours']) / 24, rmse, full)
    summary = {
        'schema_version': 1, 'status': 'complete', 'initializations': 122,
        'initialization_times': starts, 'lead_hours': protocol['lead_hours'], 'diagnostics': diagnostics,
        'verification': protocol['verification'], 'numerics': protocol['numerics'],
        'protocol_hash': repair['protocol_hash'], 'reference_protocol_hash': protocol['protocol_hash'],
        'rerun_initializations': rerun_starts, 'reused_initializations': 119,
        'eligible_initializations_per_lead_diagnostic': np.full((40, 12), 122).tolist(),
        'rmse': {m: v.tolist() for m, v in rmse.items()}, 'analyses': analyses,
        'bootstrap': protocol['bootstrap'], 'numerical_bridge': bridge_metrics,
        'numerical_bridge_definition': 'Rerun predictions versus original predictions, both scored against original truth. Positive percentages mean larger physical RMSE; these are not GraphCast-loss changes.',
        'truth_repair_effect': truth_effect_metrics,
        'truth_repair_effect_definition': 'Repaired versus original truth for the identical rerun predictions. Per-start arrays use rerun_initializations, lead_hours, diagnostics ordering; pooled arrays include all 122 starts.',
        'truth_replacements': repair['truth_validation'],
        'models': repair['models'], 'interpretation': protocol['interpretation'],
        'source_provenance': 'Original 119 forecast records retain their parent protocol; three complete rerun records per model use the separately frozen repair protocol. Only specified truth fields were repaired.',
        'native_loss_status': 'Common-grid repair only; native bridge still uses original truth. Saved native fields permit independent loss repair.',
        'sources': sources + rerun_sources + [identity(workflow / 'protocol.json'), identity(parent / 'protocol.json'),
                                              identity(repair['initialization_manifest']), identity(module_path), identity(evaluator_path),
                                              identity(workflow / 'source/scripts/analyze_models/rerun_common_grid_truth_repair.py'),
                                              identity(repair['recovery_report']['path']), identity(__file__),
                                              identity(ROOT / 'scripts/analyze_models/plot_public_checkpoints_2023_qc.py')],
        'outputs': outputs,
    }
    write(tables / 'summary.json', summary)
    write(workflow / 'report/summary.json', summary)
    write(workflow / 'report/retained_forecast_fields.json', {'common': retained_fields, 'native': native_fields})
    concise_bridge = {model: {key: value for key, value in metrics.items()
                              if key not in ('per_start_rmse_change_pct', 'pooled_rmse_change_pct')}
                       for model, metrics in bridge_metrics.items()}
    print(json.dumps({'summary': str(tables / 'summary.json'), 'images': str(images),
                      'numerical_bridge': concise_bridge}, indent=2))


if __name__ == '__main__':
    main()
