#!/usr/bin/env python3
"""Rescore only the available ENS cells affected by the frozen 2023 truth repair.

The original ENS score table is immutable. This downloads at most nine small
archived forecast chunks, retains their bytes and source receipts, reproduces
their original scores, and changes only available cells at the three repaired
valid times. Missing forecasts remain missing. No forecast model is run.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request

import numcodecs
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKFLOW = ROOT / 'artifacts/evaluations/graphcast_common_grid_2023_truth_repair_20261018'
ORIGINAL_ENS_SHA256 = '1f9c8b58bcbcb742ac2b678893479e74a041827a1736eecdf876ab6ac5063e14'
SOURCES = {'Z500': ('geopotential', 500), 'Z850': ('geopotential', 850),
           'T500': ('temperature', 500), 'Q700': ('specific_humidity', 700),
           'Q850': ('specific_humidity', 850)}
BLOSC = numcodecs.Blosc()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, obj):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def self_hash(protocol):
    body = {k: v for k, v in protocol.items() if k != 'protocol_hash'}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def stamp(value):
    return dt.datetime.fromisoformat(value.replace('Z', '+00:00'))


def identity(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'size_bytes': path.stat().st_size}


def cached_fetch(url, path):
    """Cache raw bytes once, including an immutable URL/checksum HTTP receipt."""
    path = Path(path)
    receipt_path = Path(str(path) + '.receipt.json')
    if path.exists() or receipt_path.exists():
        require(path.exists() and receipt_path.exists(), f'Incomplete cached object: {path}')
        receipt = read(receipt_path)
        require(receipt['url'] == url and receipt['sha256'] == sha(path)
                and receipt['size_bytes'] == path.stat().st_size, f'Changed cached object: {path}')
        return path.read_bytes(), receipt
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                raw = response.read()
                headers = {k.lower(): v for k, v in response.headers.items()
                           if k.lower() in ('etag', 'last-modified', 'content-length',
                                            'x-goog-generation', 'x-goog-metageneration', 'x-goog-hash')}
            break
        except urllib.error.HTTPError as error:
            if error.code == 404 or attempt == 3:
                raise
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    with path.open('xb') as stream:
        stream.write(raw)
    receipt = {**identity(path), 'url': url, 'http_headers': headers,
               'fetched_at_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    write_new(receipt_path, receipt)
    return raw, receipt


def weighted_mse(prediction, truth, latitude):
    prediction, truth = np.asarray(prediction, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    require(prediction.shape == truth.shape and prediction.ndim == 2, 'Wrong scoring shape')
    require(np.isfinite(prediction).all() and np.isfinite(truth).all(), 'Nonfinite scoring field')
    weights = np.maximum(np.cos(np.deg2rad(np.asarray(latitude, dtype=np.float64))), 0)
    require(weights.sum() > 0 and weights.size == truth.shape[0], 'Invalid latitude weights')
    weights /= weights.sum()
    return float(np.dot(weights, np.mean(np.square(prediction - truth), axis=1, dtype=np.float64)))


def close(actual, expected, description):
    require(np.isclose(actual, expected, rtol=3e-12, atol=1e-24),
            f'{description}: {actual} differs from {expected}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workflow', type=Path, default=DEFAULT_WORKFLOW)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        close(weighted_mse(np.array([[1.], [2.], [3.]]), np.zeros((3, 1)), [-60, 0, 60]),
              4.5, 'Known-answer latitude weighting')
        print('Known-answer weighted MSE self-test passed: 4.5')
        return

    workflow = args.workflow.resolve()
    output_dir = workflow / 'ens'
    output = output_dir / 'ens_mean_mse.json'
    audit_path = output_dir / 'repair_audit.json'
    require(not output.exists() and not audit_path.exists(), 'Completed ENS outputs already exist; refusing overwrite')
    repair_path = workflow / 'protocol.json'
    repair = read(repair_path)
    require(repair['protocol_hash'] == self_hash(repair), 'Repair protocol self-hash mismatch')
    reference_dir = Path(repair['reference_workflow'])
    reference_path = reference_dir / 'protocol.json'
    require(sha(reference_path) == repair['reference_protocol_sha256'], 'Reference protocol file changed')
    reference = read(reference_path)
    # The historical campaign uses a different protocol-hash convention. Its
    # entire file is already pinned above by the repair protocol's SHA256.
    require(reference['protocol_hash'] == repair['reference_protocol_hash'],
            'Reference protocol identity mismatch')
    recovery = repair['recovery_report']
    require(sha(recovery['path']) == recovery['sha256'], 'Recovery report changed')
    original_path = reference_dir / 'ens/ens_mean_mse.json'
    require(sha(original_path) == ORIGINAL_ENS_SHA256, 'Original ENS source differs from audited artifact')
    original = read(original_path)
    require(original['protocol_hash'] == reference['protocol_hash'], 'ENS/reference protocol mismatch')
    starts, leads = reference['initialization_times'], reference['lead_hours']
    rows = original['per_initialization']
    diagnostics = list(rows[0]['mse'])
    require([r['initialization_time'] for r in rows] == starts and len(starts) == 122 and len(leads) == 40,
            'Unexpected ENS initialization coverage')
    for row in rows:
        require(row['lead_hours'] == leads and list(row['mse']) == diagnostics,
                'ENS diagnostic/lead coverage changed')
        require(all(len(row['mse'][key]) == len(leads) for key in diagnostics), 'ENS series length differs')
    require(set(repair['repaired_diagnostics']) == set(SOURCES), 'Unexpected truth repair diagnostics')

    truth = {}
    replacements = {}
    for record in repair['truth_validation']:
        valid = record['valid_time']
        filename = stamp(valid).strftime('%Y%m%dT%H%M%SZ.nc')
        old_path = Path(reference['common_truth']) / filename
        new_path = Path(repair['corrected_common_truth']) / filename
        require(sha(old_path) == record['old_sha256'] and sha(new_path) == record['new_sha256'],
                f'Frozen truth checksums differ at {valid}')
        receipt = read(str(new_path) + '.READY.json')
        require(receipt['sha256'] == record['new_sha256'] and receipt['timestamp'] == valid
                and receipt['original_truth_sha256'] == record['old_sha256']
                and set(receipt['replaced_diagnostics']) == set(SOURCES), 'Truth READY identity mismatch')
        old_truth, new_truth = xr.load_dataset(old_path), xr.load_dataset(new_path)
        require(np.array_equal(old_truth.lat.values, new_truth.lat.values)
                and np.array_equal(old_truth.lon.values, new_truth.lon.values), 'Truth coordinates changed')
        for key in diagnostics:
            require(np.isfinite(new_truth[key].values).all(), f'Nonfinite repaired truth: {key}')
            require(np.array_equal(old_truth[key].values, new_truth[key].values) == (key not in SOURCES),
                    f'Unexpected truth change scope: {valid}/{key}')
        truth[valid] = old_truth, new_truth
        replacements[valid] = {'original': identity(old_path), 'corrected': identity(new_path),
                               'corrected_receipt': identity(str(new_path) + '.READY.json')}
    require(set(truth) == set(repair['repaired_valid_times']) and len(truth) == 3,
            'Repair valid-time coverage mismatch')

    scorer_path = ROOT / original['scorer']
    require(sha(scorer_path) == original['scorer_sha256'], 'Frozen original scorer changed')
    module_spec = importlib.util.spec_from_file_location('frozen_ens_scorer', scorer_path)
    scorer = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(scorer)
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_script = output_dir / 'source' / Path(__file__).name
    frozen_script.parent.mkdir(exist_ok=True)
    if frozen_script.exists():
        require(sha(frozen_script) == sha(__file__), 'ENS repair script changed since cache preparation')
    else:
        shutil.copyfile(__file__, frozen_script)

    remote = []
    raw, receipt = cached_fetch(original['layout_reference'] + '.zmetadata', output_dir / 'cache/reference/.zmetadata')
    remote.append(receipt)
    metadata = json.loads(raw)['metadata']
    coords = {}
    for name in ('prediction_timedelta', 'level', 'latitude', 'longitude'):
        raw, receipt = cached_fetch(original['layout_reference'] + name + '/0',
                                    output_dir / 'cache/reference' / name / '0')
        remote.append(receipt)
        spec = metadata[name + '/.zarray']
        coords[name] = np.frombuffer(BLOSC.decode(raw), dtype=spec['dtype']).copy()
    require(coords['prediction_timedelta'].tolist() == list(range(0, 361, 6))
            and coords['level'].tolist() == [500, 700, 850], 'Reference lead/level layout changed')
    sample = next(iter(truth.values()))[0]
    require(np.allclose(coords['latitude'], sample.lat.values, rtol=0, atol=1e-9)
            and np.allclose(coords['longitude'], sample.lon.values, rtol=0, atol=1e-9),
            'ENS/truth coordinates differ')
    arrays = {}
    for var, _ in SOURCES.values():
        arrays[var] = metadata[var + '/.zarray']
        spec = arrays[var]
        require(spec['chunks'] == [1, 8, 3, 240, 121] and spec['dtype'] == '<f4'
                and spec['compressor']['id'] == 'blosc'
                and metadata[var + '/.zattrs']['_ARRAY_DIMENSIONS'] ==
                ['time', 'prediction_timedelta', 'level', 'longitude', 'latitude'],
                f'ENS pressure-field layout changed: {var}')

    result = copy.deepcopy(original)
    chunks, changes, missing = {}, [], []
    changed_cells = set()
    max_reproduction_relative_error = 0.0
    for row_index, row in enumerate(rows):
        start = row['initialization_time']
        epoch_index = int((stamp(start) - dt.datetime(2016, 1, 1, tzinfo=dt.timezone.utc)).total_seconds() / 43200)
        require(epoch_index == row['store_time_index'], 'ENS time-index mismatch')
        for lead_index, lead in enumerate(leads):
            valid = (stamp(start) + dt.timedelta(hours=lead)).isoformat().replace('+00:00', 'Z')
            if valid not in truth:
                continue
            require(start in repair['initialization_times'], 'Affected start omitted from repair protocol')
            old_truth, new_truth = truth[valid]
            fields = {}
            for key, (var, level) in SOURCES.items():
                cell = {'initialization_time': start, 'lead_hours': lead, 'valid_time': valid, 'diagnostic': key}
                if row['mse'][key][lead_index] is None:
                    missing.append(cell)
                    continue
                chunk_index, offset = divmod(lead // 6, 8)
                chunk_key = f'{var}/{epoch_index}.{chunk_index}.0.0.0'
                if chunk_key not in chunks:
                    require(len(chunks) < 9, 'Repair unexpectedly requires more than nine forecast chunks')
                    raw, receipt = cached_fetch(original['store'] + chunk_key, output_dir / 'cache/forecast' / chunk_key)
                    remote.append(receipt)
                    chunks[chunk_key] = np.frombuffer(BLOSC.decode(raw), dtype='<f4').reshape(arrays[var]['chunks'])
                    print(f'Cached ENS chunk {chunk_key} ({len(raw)} bytes)', flush=True)
                plane = chunks[chunk_key][0, offset, coords['level'].tolist().index(level)].astype(np.float64).T
                require(np.isfinite(plane).all(), f'Previously available ENS field became nonfinite: {cell}')
                fields[key] = plane
            # Fill unused helper arguments only; only available affected fields
            # are scored/copied below, never synthesized as ENS predictions.
            prediction = xr.Dataset({key: (('lat', 'lon'), fields.get(key, old_truth[key].values),
                                           {'units': old_truth[key].attrs.get('units', '')})
                                     for key in diagnostics}, coords={'lat': old_truth.lat, 'lon': old_truth.lon})
            old_scores, new_scores = scorer.score_step(prediction, old_truth), scorer.score_step(prediction, new_truth)
            for key, plane in fields.items():
                old_score = row['mse'][key][lead_index]
                close(old_scores[key], old_score, f'Original ENS score reproduction: {start}/{lead}/{key}')
                close(weighted_mse(plane, old_truth[key].values, old_truth.lat.values), old_scores[key],
                      'Independent old-truth MSE formula')
                close(weighted_mse(plane, new_truth[key].values, new_truth.lat.values), new_scores[key],
                      'Independent corrected-truth MSE formula')
                max_reproduction_relative_error = max(max_reproduction_relative_error, abs(old_scores[key] / old_score - 1))
                result['per_initialization'][row_index]['mse'][key][lead_index] = new_scores[key]
                changed_cells.add((row_index, lead_index, key))
                changes.append({'initialization_time': start, 'lead_hours': lead, 'valid_time': valid,
                                'diagnostic': key, 'original_mse': old_score,
                                'reproduced_original_mse': old_scores[key], 'corrected_mse': new_scores[key]})

    require(len(changes) == 42 and len(missing) == 3 and len(chunks) == 8, 'Unexpected affected-cell/chunk coverage')
    old_mask = []
    new_mask = []
    unchanged = 0
    for i, (old_row, new_row) in enumerate(zip(rows, result['per_initialization'])):
        for key in diagnostics:
            for j, (before, after) in enumerate(zip(old_row['mse'][key], new_row['mse'][key])):
                old_mask.append(before is not None)
                new_mask.append(after is not None)
                if (i, j, key) not in changed_cells:
                    require((before is None and after is None) or
                            (before is not None and after is not None and before.hex() == after.hex()),
                            f'Unrelated ENS cell changed: {i}/{j}/{key}')
                    unchanged += 1
    require(old_mask == new_mask and sum(new_mask) == original['available_entries'] == 46930
            and len(new_mask) == original['total_entries'] == 58560, 'ENS coverage changed')
    mask = np.array([[[row['mse'][key][j] is not None for key in diagnostics]
                      for j in range(len(leads))] for row in rows])
    counts = mask.sum(axis=0)
    coverage = {'available_entries': int(mask.sum()), 'total_entries': int(mask.size),
                'complete_initializations_all_diagnostics_and_leads': int(mask.all(axis=(1, 2)).sum()),
                'minimum_starts_per_diagnostic_lead': int(counts.min()),
                'maximum_starts_per_diagnostic_lead': int(counts.max()),
                'diagnostics': diagnostics, 'lead_hours': leads, 'starts_per_lead_diagnostic': counts.tolist()}
    result.update({'repair_protocol_hash': repair['protocol_hash'],
                   'repair_protocol': identity(repair_path), 'truth_replacements': replacements,
                   'repair_audit': str(audit_path), 'original_ens_scores': identity(original_path),
                   'repair_script': identity(frozen_script), 'coverage': coverage,
                   'truth': 'Original campaign common truth, with frozen NCAR repair for exactly five diagnostics at three valid times; missing ENS forecasts unchanged.'})
    require(sha(original_path) == ORIGINAL_ENS_SHA256, 'Original ENS source changed during repair')
    write_new(output, result)
    audit = {'status': 'complete', 'schema_version': 1, 'repair_protocol_hash': repair['protocol_hash'],
             'original_ens_scores': identity(original_path), 'corrected_ens_scores': identity(output),
             'repair_script': identity(frozen_script), 'scorer': identity(scorer_path),
             'reference_protocol': identity(reference_path), 'repair_protocol': identity(repair_path),
             'recovery_report': recovery, 'truth_replacements': replacements,
             'remote_objects': remote, 'downloaded_forecast_chunks': len(chunks),
             'changed_available_cells': len(changes), 'affected_missing_cells_retained': missing,
             'unaffected_cells_bitwise_identical': unchanged,
             'original_score_max_relative_reproduction_error': max_reproduction_relative_error,
             'independent_mse_formula_checked': True, 'coverage_unchanged': True,
             'coverage': coverage, 'changes': changes,
             'limitations': ['The public ENS archive remains incomplete: no initialization has all 480 diagnostic/lead cells.',
                            'Plot every model over the identical finite ENS mask separately for each diagnostic and lead.',
                            'The 2016–2024 store lacks its own metadata; the original 71-chunk sibling-layout audit is retained, and affected old scores are reproduced here.']}
    write_new(audit_path, audit)
    print(f'Complete: repaired {len(changes)} ENS cells; {len(missing)} remain missing; '
          f'{unchanged} unchanged cells; {counts.min()}–{counts.max()} starts per curve point.', flush=True)


if __name__ == '__main__':
    main()
