#!/usr/bin/env python3
"""Independently validate a complete 122-start repaired common-grid report.

No forecast or model code is imported. Scores are reconstructed directly from
the original and rerun shards, pooled independently, and recomputed from every
retained common-grid forecast using an explicit latitude-weighted MSE formula.
Successful evidence is published only after all checks finish.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import xarray as xr

MODELS = ('gc1', 'm1', 'gc025')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def valid_time(start, lead):
    return (stamp(start) + timedelta(hours=int(lead))).isoformat().replace('+00:00', 'Z')


def filename(value):
    return stamp(value).strftime('%Y%m%dT%H%M%SZ.nc')


def weighted_mse(prediction, truth, latitude):
    """Physical-unit MSE, equal longitude cells and normalized cos(latitude)."""
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    latitude = np.asarray(latitude, dtype=np.float64)
    require(prediction.shape == truth.shape and prediction.ndim == 2
            and prediction.shape[0] == latitude.size and prediction.shape[1] > 0,
            'MSE requires equal latitude-by-longitude arrays')
    require(np.isfinite(prediction).all() and np.isfinite(truth).all()
            and np.isfinite(latitude).all(), 'MSE received nonfinite values')
    weights = np.maximum(np.cos(np.deg2rad(latitude)), 0.0)
    require(weights.sum() > 0, 'Latitude weights have zero total')
    weights /= weights.sum()
    error = prediction - truth
    return float(np.dot(weights, np.mean(error * error, axis=1, dtype=np.float64)))


def close(actual, expected, message, rtol=3e-12, atol=1e-24):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=False):
        delta = float(np.max(np.abs(a - b))) if a.shape == b.shape else 'shape mismatch'
        raise ValueError(f'{message}; maximum absolute discrepancy: {delta}')


def validate_identity(record):
    path = Path(record['path']).resolve()
    require(path.is_file(), f'Missing source/output: {path}')
    require(sha(path) == record['sha256'], f'Source/output checksum changed: {path}')
    if 'size_bytes' in record:
        require(path.stat().st_size == record['size_bytes'], f'Source/output size changed: {path}')
    return path


def check_truth(path, time, expected_hash):
    path = Path(path).resolve()
    receipt = read(str(path) + '.READY.json')
    require(receipt.get('schema_version') == 1 and receipt.get('kind') == 'common_truth'
            and Path(receipt['path']).resolve() == path and receipt['timestamp'] == time
            and receipt['sha256'] == expected_hash,
            f'Truth receipt identity differs: {path}')
    validate_identity({'path': str(path), 'sha256': expected_hash, 'size_bytes': receipt['size_bytes']})
    return receipt


def validate_row(row, branches, starts, leads, diagnostics):
    start = row['initialization_time']
    require(start in starts and row['lead_hours'] == leads
            and row['valid_times'] == [valid_time(start, lead) for lead in leads]
            and len(row['truth_sha256']) == len(leads) and set(row['mse']) == set(branches),
            f'Score record coordinates differ: {start}')
    arrays = {}
    for model in branches:
        require(set(row['mse'][model]) == set(diagnostics), f'Diagnostic coverage differs: {model}/{start}')
        array = np.asarray([row['mse'][model][name] for name in diagnostics], dtype=np.float64).T
        require(array.shape == (40, 12) and np.isfinite(array).all() and np.all(array >= 0),
                f'Invalid score array: {model}/{start}')
        arrays[model] = array
    return arrays


def check_changes(reference, candidate, indices, stored, starts, leads, diagnostics):
    per_start = 100 * (np.sqrt(candidate[indices] / reference[indices]) - 1)
    pooled = 100 * (np.sqrt(candidate.mean(axis=0) / reference.mean(axis=0)) - 1)
    close(stored['per_start_rmse_change_pct'], per_start, 'Per-start change array differs', atol=1e-10)
    close(stored['pooled_rmse_change_pct'], pooled, 'Pooled change array differs', atol=1e-10)
    for key, value in [('max_abs_per_start_rmse_change_pct', np.max(np.abs(per_start))),
                       ('p95_abs_per_start_rmse_change_pct', np.percentile(np.abs(per_start), 95)),
                       ('max_abs_full_sample_pooled_rmse_change_pct', np.max(np.abs(pooled)))]:
        close(stored[key], value, f'Change statistic differs: {key}', atol=1e-10)
    worst = np.unravel_index(np.argmax(np.abs(per_start)), per_start.shape)
    record = stored['worst_case']
    require(record['initialization_time'] == starts[indices[worst[0]]]
            and record['lead_hours'] == leads[worst[1]] and record['diagnostic'] == diagnostics[worst[2]],
            'Worst-case bridge identity differs')
    close(record['signed_rmse_change_pct'], per_start[worst], 'Worst-case change differs', atol=1e-10)


def validate(workflow):
    workflow = Path(workflow).resolve()
    output = workflow / 'report/independent_validation.json'
    require(not output.exists(), f'Independent validation already exists: {output}')
    summary_path = workflow / 'report/summary.json'
    summary = read(summary_path)
    repair_path = workflow / 'protocol.json'
    repair = read(repair_path)
    body = {key: value for key, value in repair.items() if key != 'protocol_hash'}
    require(hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            == repair['protocol_hash'], 'Repair protocol self-hash differs')
    parent = Path(repair['reference_workflow'])
    require(sha(parent / 'protocol.json') == repair['reference_protocol_sha256'], 'Parent protocol changed')
    original_protocol = read(parent / 'protocol.json')
    starts, leads, diagnostics = (original_protocol['initialization_times'], original_protocol['lead_hours'],
                                  original_protocol['verification']['diagnostics'])
    rerun_starts = repair['initialization_times']
    require(rerun_starts == ['2023-11-03T00:00:00Z', '2023-11-06T00:00:00Z', '2023-11-09T00:00:00Z']
            and repair['repaired_valid_times'] == [f'2023-11-10T{hour}:00:00Z' for hour in ('06', '12', '18')]
            and set(repair['repaired_diagnostics']) == {'Z500', 'Z850', 'T500', 'Q700', 'Q850'},
            'Repair scope differs from the identified source defect')
    require(len(starts) == 122 and len(set(starts)) == 122 and leads == list(range(6, 241, 6))
            and len(diagnostics) == 12 and len(set(diagnostics)) == 12 and len(rerun_starts) == 3,
            'Unexpected experiment dimensions')
    require(summary['status'] == 'complete' and summary['initializations'] == 122
            and summary['initialization_times'] == starts and summary['lead_hours'] == leads
            and summary['diagnostics'] == diagnostics and summary['protocol_hash'] == repair['protocol_hash']
            and summary['reference_protocol_hash'] == original_protocol['protocol_hash']
            and summary['rerun_initializations'] == rerun_starts and summary['reused_initializations'] == 119,
            'Completed summary does not describe the fixed 122-start repair')
    np.testing.assert_array_equal(summary['eligible_initializations_per_lead_diagnostic'], np.full((40, 12), 122))
    require(set(summary['rmse']) == set(MODELS), 'Summary model coverage differs')
    validated_sources = [validate_identity(record) for record in summary['sources']]
    validated_outputs = [validate_identity(record) for record in summary['outputs']]
    csv_paths = [path for path in validated_outputs if path.name == 'per_initialization_mse.csv']
    require(len(csv_paths) == 1, 'Expected exactly one completed per-initialization CSV')
    require(read(csv_paths[0].with_name('summary.json')) == summary, 'Report and table summaries disagree')
    for relative, expected in repair['source_files'].items():
        path = (workflow / 'source' / relative).resolve()
        require(path.is_relative_to((workflow / 'source').resolve()) and sha(path) == expected,
                f'Frozen repair source changed: {relative}')
    validate_identity(repair['recovery_report'])
    require(sha(Path(repair['initialization_manifest'])) == repair['initialization_manifest_sha256'],
            'Repair initialization manifest changed')
    require(read(repair['initialization_manifest']) == {'schema_version': 1, 'initializations': rerun_starts},
            'Repair manifest content differs')
    source_set = set(validated_sources)

    original_rows, original_truth_hashes = {}, {}
    original_sources = []
    for group, branches in [('m1', ('gc1', 'm1')), ('gc025', ('gc025',))]:
        paths = sorted((parent / 'shards').glob(f'{group}_*.json'))
        require(len(paths) == (8 if group == 'm1' else 16), f'Original shard count differs: {group}')
        for path in paths:
            require(path.resolve() in source_set, f'Original shard omitted from summary provenance: {path}')
            shard = read(path)
            require(shard['status'] == 'complete' and shard['protocol_hash'] == original_protocol['protocol_hash'],
                    f'Original shard identity differs: {path}')
            original_sources.append(path)
            for row in shard['per_initialization']:
                arrays = validate_row(row, branches, starts, leads, diagnostics)
                start = row['initialization_time']
                for model, array in arrays.items():
                    key = model, start
                    require(key not in original_rows, f'Duplicate original forecast: {key}')
                    original_rows[key] = array
                for time, digest in zip(row['valid_times'], row['truth_sha256'], strict=True):
                    require(time not in original_truth_hashes or original_truth_hashes[time] == digest,
                            f'Original truth differs between models/starts: {time}')
                    original_truth_hashes[time] = digest
    require(set(original_rows) == {(model, start) for model in MODELS for start in starts},
            'Original forecast coverage is incomplete')
    original = {model: np.stack([original_rows[model, start] for start in starts]) for model in MODELS}
    final = {model: cube.copy() for model, cube in original.items()}
    bridge = {model: cube.copy() for model, cube in original.items()}
    original_truth_root = Path(original_protocol['common_truth'])
    for time, digest in original_truth_hashes.items():
        check_truth(original_truth_root / filename(time), time, digest)
    print(f'[independent-validation] verified {len(original_truth_hashes)} original truth files', flush=True)
    replacements = {row['valid_time']: row for row in repair['truth_validation']}
    require(sorted(replacements) == repair['repaired_valid_times'] and len(replacements) == 3
            and len(repair['truth_validation']) == 3, 'Truth replacement coverage differs')
    corrected_truth_root = Path(repair['corrected_common_truth'])
    for time, record in replacements.items():
        require(record['old_sha256'] == original_truth_hashes[time], 'Frozen original truth hash differs')
        receipt = check_truth(corrected_truth_root / filename(time), time, record['new_sha256'])
        require(receipt['original_truth_sha256'] == record['old_sha256'], 'Corrected truth linkage differs')
        with xr.open_dataset(original_truth_root / filename(time)) as old, xr.open_dataset(corrected_truth_root / filename(time)) as new:
            require(set(old.data_vars) == set(new.data_vars) == set(diagnostics), 'Repair truth diagnostic coverage differs')
            for coordinate in ('lat', 'lon'):
                np.testing.assert_array_equal(old[coordinate], new[coordinate])
            for diagnostic in diagnostics:
                a, b = old[diagnostic].values, new[diagnostic].values
                require(np.isfinite(b).all(), 'Repaired truth contains nonfinite values')
                if diagnostic in repair['repaired_diagnostics']:
                    require(not np.array_equal(a, b), 'Declared repaired diagnostic is unchanged')
                else:
                    require(a.dtype == b.dtype and np.array_equal(a, b), 'Unrelated truth diagnostic changed')
    affected = [start for start in starts if any(valid_time(start, lead) in replacements for lead in leads)]
    require(affected == rerun_starts, 'Replacement times affect starts outside the rerun selection')
    original_rerun_rows, corrected_rerun_rows, retained, native = {}, {}, [], []
    for group, branches in [('m1', ('gc1', 'm1')), ('gc025', ('gc025',))]:
        path = workflow / 'shards' / f'{group}_00.json'
        require(path.resolve() in source_set, f'Rerun shard omitted from summary provenance: {path}')
        shard = read(path)
        require(shard['status'] == 'complete' and shard['model'] == group
                and shard['protocol_hash'] == repair['protocol_hash']
                and shard['protocol_sha256'] == sha(repair_path)
                and shard['reference_protocol_sha256'] == repair['reference_protocol_sha256']
                and shard['wrapper_sha256'] == repair['wrapper_sha256']
                and shard['initialization_manifest_sha256'] == repair['initialization_manifest_sha256'],
                f'Rerun shard provenance differs: {group}')
        for collection, destination, lookup, corrected in [
                ('per_initialization', final, corrected_rerun_rows, True),
                ('original_truth_per_initialization', bridge, original_rerun_rows, False)]:
            rows = shard[collection]
            require([row['initialization_time'] for row in rows] == rerun_starts, 'Rerun date coverage differs')
            for row in rows:
                start = row['initialization_time']
                arrays = validate_row(row, branches, rerun_starts, leads, diagnostics)
                expected_hashes = [replacements[t]['new_sha256'] if corrected and t in replacements
                                   else original_truth_hashes[t] for t in row['valid_times']]
                require(row['truth_sha256'] == expected_hashes, f'Rerun truth hashes differ: {collection}/{start}')
                for model, array in arrays.items():
                    require((model, start) not in lookup, 'Duplicate rerun score record')
                    lookup[model, start] = array
                    destination[model][starts.index(start)] = array
        retained.extend(shard['common_prediction_fields'])
        native.extend(shard['native_repair_fields'])
        if group == 'm1':
            validate_identity({'path': shard['evidence']['native_result'],
                               'sha256': shard['evidence']['native_result_sha256']})
    expected_reruns = {(model, start) for model in MODELS for start in rerun_starts}
    require(set(original_rerun_rows) == expected_reruns == set(corrected_rerun_rows), 'Rerun models are incomplete')
    field_manifest = read(workflow / 'report/retained_forecast_fields.json')
    require(field_manifest == {'common': retained, 'native': native}, 'Retained field manifest differs from shards')
    keys = [(r['model'], r['initialization_time'], r['lead_hours']) for r in retained]
    expected_field_keys = {(model, start, lead) for model in MODELS for start in rerun_starts for lead in leads}
    require(len(keys) == 360 and len(set(keys)) == 360 and set(keys) == expected_field_keys,
            'Retained common fields do not cover all three models, starts and 40 leads')
    require(len({Path(r['path']).resolve() for r in retained}) == len(retained), 'Common field paths are reused')
    native_keys = [(r['model'], r['initialization_time'], r['lead_hours']) for r in native]
    expected_native = {(model, start, lead) for model in ('gc1', 'm1', 'original_native_truth')
                       for start in rerun_starts for lead in leads if valid_time(start, lead) in replacements}
    require(len(native_keys) == 27 and len(set(native_keys)) == 27 and set(native_keys) == expected_native,
            'Retained native repair fields are incomplete')
    for receipt in native:
        validate_identity(receipt)

    truth_cache = {}
    max_relative_difference = 0.0
    rescored_cells = 0
    for receipt in retained:
        path = validate_identity(receipt)
        model, start, lead = receipt['model'], receipt['initialization_time'], receipt['lead_hours']
        time = valid_time(start, lead)
        require(receipt['valid_time'] == time, 'Retained forecast receipt valid time differs')
        with xr.open_dataset(path) as source:
            prediction = source.load()
        require(set(prediction.data_vars) == set(diagnostics), 'Retained field diagnostic coverage differs')
        for key in ('model', 'initialization_time', 'lead_hours', 'valid_time'):
            require(prediction.attrs.get(key) == receipt[key], f'Retained field metadata differs: {path}/{key}')
        for corrected, lookup in [(False, original_rerun_rows), (True, corrected_rerun_rows)]:
            root = corrected_truth_root if corrected and time in replacements else original_truth_root
            truth_path = root / filename(time)
            if truth_path not in truth_cache:
                with xr.open_dataset(truth_path) as source:
                    truth_cache[truth_path] = source.load()
            truth = truth_cache[truth_path]
            require(truth.attrs.get('valid_time') == time, 'Actual truth file valid time differs')
            for coordinate in ('lat', 'lon'):
                np.testing.assert_array_equal(prediction[coordinate], truth[coordinate])
            for k, diagnostic in enumerate(diagnostics):
                actual = weighted_mse(prediction[diagnostic], truth[diagnostic], truth.lat)
                expected = lookup[model, start][leads.index(lead), k]
                close(actual, expected, f'Saved-field rescore differs: {model}/{start}/{lead}/{diagnostic}')
                max_relative_difference = max(max_relative_difference, abs(actual - expected) / max(abs(expected), 1e-30))
                rescored_cells += 1
    require(rescored_cells == 8640, 'Not every retained forecast was scored against both truth versions')
    print('[independent-validation] independently rescored 8640 old/new MSE cells', flush=True)

    csv_rows = {}
    with csv_paths[0].open(newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ['model', 'initialization_time', 'lead_hours', 'diagnostic', 'mse_si', 'forecast_origin'],
                'Completed CSV schema differs')
        for row in reader:
            key = (row['model'], row['initialization_time'], int(row['lead_hours']), row['diagnostic'])
            require(key not in csv_rows, f'Duplicate CSV score cell: {key}')
            value = float(row['mse_si'])
            require(np.isfinite(value) and value >= 0, f'Invalid CSV score cell: {key}')
            require(row['forecast_origin'] == ('rerun_with_repaired_truth' if row['initialization_time'] in rerun_starts else 'original'),
                    f'CSV forecast-origin label differs: {key}')
            csv_rows[key] = value
    expected_cells = {(model, start, lead, d) for model in MODELS for start in starts for lead in leads for d in diagnostics}
    require(len(csv_rows) == 175680 and set(csv_rows) == expected_cells, 'Final CSV has missing or extra score cells')
    reused_indices = [i for i, start in enumerate(starts) if start not in rerun_starts]
    rerun_indices = [starts.index(start) for start in rerun_starts]
    for model in MODELS:
        csv_cube = np.array([[[csv_rows[model, start, lead, d] for d in diagnostics] for lead in leads] for start in starts])
        np.testing.assert_array_equal(csv_cube, final[model])
        np.testing.assert_array_equal(csv_cube[reused_indices], original[model][reused_indices])
        rmse = np.sqrt(csv_cube.mean(axis=0))
        close(summary['rmse'][model], rmse, f'Pooled report RMSE differs: {model}')
        check_changes(original[model], bridge[model], rerun_indices, summary['numerical_bridge'][model], starts, leads, diagnostics)
        check_changes(bridge[model], final[model], rerun_indices, summary['truth_repair_effect'][model], starts, leads, diagnostics)
        for i in rerun_indices:
            for j, lead in enumerate(leads):
                for k, diagnostic in enumerate(diagnostics):
                    if valid_time(starts[i], lead) not in replacements or diagnostic not in repair['repaired_diagnostics']:
                        require(final[model][i, j, k] == bridge[model][i, j, k], 'Truth repair changed an unrelated score')
    for name, analysis in summary['analyses'].items():
        exposed = set(original_protocol['prior_exposure']['sampled_initialization_times'])
        eligible = [i for i, start in enumerate(starts) if name != 'exclude_pilot' or start not in exposed]
        require(analysis['n_initializations'] == len(eligible), f'Analysis sample count differs: {name}')
        for reference in ('gc1', 'gc025'):
            estimate = 100 * (1 - np.sqrt(final['m1'][eligible].mean(axis=0) / final[reference][eligible].mean(axis=0)))
            close(analysis['comparisons'][reference]['rmse_reduction_pct'], estimate,
                  f'Analysis point estimates differ: {name}/{reference}', atol=1e-10)
    require(summary['analyses']['primary']['n_initializations'] == 122, 'Primary analysis excludes initializations')
    evidence = {
        'schema_version': 1, 'status': 'passed', 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_hash': repair['protocol_hash'], 'summary_sha256': sha(summary_path),
        'validator_sha256': sha(Path(__file__)), 'csv_path': str(csv_paths[0]), 'csv_sha256': sha(csv_paths[0]),
        'score_cells': len(csv_rows), 'models': list(MODELS), 'initializations': 122,
        'reused_initializations_per_model': 119, 'rerun_initializations_per_model': 3,
        'all_leads_and_diagnostics_have_122_starts': True, 'excluded_primary_score_cells': 0,
        'original_truth_files_hashed': len(original_truth_hashes), 'corrected_truth_files_hashed': len(replacements),
        'common_forecast_files_rescored': len(retained), 'old_and_new_truth_score_cells_recomputed': rescored_cells,
        'max_relative_saved_field_rescore_difference': max_relative_difference,
        'rescore_tolerance': {'relative': 3e-12, 'absolute': 1e-24},
        'native_field_files_hashed': len(native), 'source_files_hashed': len(validated_sources),
        'output_files_hashed': len(validated_outputs), 'pooled_rmse_matches': True,
        'numerical_bridge_and_truth_effect_match': True, 'analysis_point_estimates_match': True,
        'limitations': 'Bootstrap intervals are not independently recomputed here. Native losses remain uncorrected bridge measurements; no inference equivalence claim is made by this validator.',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix='.independent-validation-', dir=output.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(evidence, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_name, output)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    print(json.dumps(evidence, indent=2), flush=True)


def self_test():
    latitude = np.array([-60., 0., 60.])
    prediction = np.array([[1., 1.], [2., 2.], [3., 3.]])
    truth = np.zeros_like(prediction)
    close(weighted_mse(prediction, truth, latitude), 4.5, 'Known-answer weighted MSE')
    close(weighted_mse(truth, truth, latitude), 0., 'Zero-error weighted MSE')
    close(weighted_mse(-prediction, truth, latitude), 4.5, 'Sign-invariant weighted MSE')
    for a, b in [(prediction, np.zeros((2, 2))), (prediction * np.nan, truth)]:
        try:
            weighted_mse(a, b, latitude)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid MSE input was accepted')
    print('Known-answer formula and invalid-input checks passed; no evaluation data were validated.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--workflow', type=Path, help='Completed repaired evaluation directory')
    group.add_argument('--self-test', action='store_true', help='Run known-answer formula checks only')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        validate(args.workflow)


if __name__ == '__main__':
    main()
