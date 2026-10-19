#!/usr/bin/env python3
"""Audit exact repeated verification fields without modifying forecast scores."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import netCDF4
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workflow', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.workflow / 'protocol.json').read_text())
    root = Path(protocol['common_truth'])
    fields = protocol['verification']['diagnostics']
    hashes = {field: defaultdict(list) for field in fields}
    files = {}
    transitions = {field: [] for field in fields}
    previous = None
    for i, path in enumerate(sorted(root.glob('*.nc'))):
        receipt = json.loads(Path(str(path) + '.READY.json').read_text())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != receipt['sha256']:
            raise ValueError(f'Receipt checksum mismatch: {path}')
        with netCDF4.Dataset(path) as ds:
            stamp = ds.valid_time
            if stamp != receipt['timestamp'] or stamp in files:
                raise ValueError(f'Timestamp mismatch/duplicate: {path}')
            lat = np.asarray(ds['lat'][:])
            weights = np.cos(np.deg2rad(lat))[:, None]
            weights /= weights.mean()
            current = {}
            for field in fields:
                values = np.asarray(ds[field][:], dtype=np.float64)
                if values.shape != (121, 240) or not np.isfinite(values).all():
                    raise ValueError(f'Invalid field: {path}/{field}')
                key = hashlib.sha256(values.tobytes()).hexdigest()
                hashes[field][key].append(stamp)
                current[field] = values
                if previous is not None:
                    delta = values - previous[1][field]
                    transitions[field].append({
                        'from': previous[0], 'to': stamp,
                        'rmse_si': float(np.sqrt(np.mean(weights * delta**2))),
                    })
        files[stamp] = digest
        previous = (stamp, current)
        if (i + 1) % 200 == 0:
            print(f'Checked {i + 1} truth files', flush=True)
    repeats = {field: [{'field_sha256': key, 'valid_times': times}
                       for key, times in entries.items() if len(times) > 1]
               for field, entries in hashes.items()}
    suspicious = {stamp for groups in repeats.values() for group in groups
                  for stamp in group['valid_times']}
    affected = []
    for path in sorted((args.workflow / 'shards').glob('*.json')):
        shard = json.loads(path.read_text())
        if shard['status'] != 'complete' or shard['protocol_hash'] != protocol['protocol_hash']:
            raise ValueError(f'Invalid shard: {path}')
        for row in shard['per_initialization']:
            for valid, lead, digest in zip(row['valid_times'], row['lead_hours'],
                                           row['truth_sha256'], strict=True):
                if files[valid] != digest:
                    raise ValueError(f'Truth changed since scoring: {path}/{valid}')
                if valid in suspicious:
                    affected.append({'initialization_time': row['initialization_time'],
                                     'valid_time': valid, 'lead_hours': lead,
                                     'models': sorted(row['mse'])})
    result = {
        'status': 'complete', 'protocol_hash': protocol['protocol_hash'],
        'truth_files_checked': len(files), 'truth_file_sha256': files,
        'repeat_groups': repeats, 'potentially_affected_scores': affected,
        'largest_adjacent_changes': {field: sorted(rows, key=lambda r: r['rmse_si'],
                                                  reverse=True)[:10]
                                    for field, rows in transitions.items()},
        'interpretation': 'Exact repeats are a data-quality flag; this audit does not select replacement values or exclude any scores.',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'truth_files_checked': len(files), 'repeat_groups': repeats,
                      'affected_score_rows': len(affected)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
