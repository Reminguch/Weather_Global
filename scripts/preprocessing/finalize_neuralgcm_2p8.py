#!/usr/bin/env python3
"""Finish the immutable 2.8-degree manifest on a healthy GPFS client.

No network access, regridding or frame changes. Uses the same manifest/path
semantics as merge_prepared, without importing a training runtime. The existing
supervisor lock prevents two finalizers from writing concurrently.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
from pathlib import Path
import time
from watch_neuralgcm_smoke import expected_times, read, sha, write


def load_month(root, month):
    folder = root / 'shards' / month['id'] / 'res2p8'
    manifest = folder / 'manifest.json'
    if read(folder / 'READY.json')['manifest_sha256'] != sha(manifest):
        raise ValueError('Incomplete/corrupt month: ' + month['id'])
    data = read(manifest)
    if data['interval'] != [month['start'], month['end']]:
        raise ValueError('Unexpected monthly interval: ' + month['id'])
    records = []
    for record in data['records']:
        path = (folder / record['file']).resolve()
        if root not in path.parents or 'res2p8' not in path.parts:
            raise ValueError('Frame escaped dataset root: ' + str(path))
        records.append(dict(record, file=str(path)))
    return data, records


def merged_manifest(root):
    plan = read(root / 'download_manifest.json')
    expected_months = [f'{y}-{m:02}' for y in range(2015, 2024) for m in range(1, 13)]
    if [m['id'] for m in plan['months']] != expected_months:
        raise ValueError('Unexpected month plan')
    with ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(lambda m: load_month(root, m), plan['months']))
    merged = dict(parts[0][0])
    records = []
    for data, rows in parts:
        for field in ('data_grid', 'source', 'variables', 'units', 'dtype', 'regrid'):
            if data[field] != merged[field]:
                raise ValueError('Incompatible monthly metadata: ' + field)
        records.extend(rows)
    records.sort(key=lambda r: r['time'])
    if [r['time'] for r in records] != expected_times():
        raise ValueError('Missing/duplicate/unexpected frame timestamps')
    merged.update(records=records, interval=['2015-01-01T00', '2024-01-01T00'])
    return plan, merged


def immutable(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError('Refusing to overwrite differing immutable metadata: ' + str(path))
    else:
        write(path, value)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    root = a.root.resolve()
    lock = (root / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    plan, merged = merged_manifest(root)
    if not a.check_only:
        output = root / 'res2p8'
        output.mkdir(exist_ok=True)
        immutable(output / 'manifest.json', merged)
        immutable(output / 'READY.json', dict(manifest_sha256=sha(output / 'manifest.json')))
        immutable(root / 'READY.res2p8.json', dict(source_id=plan['source_id'], timestamps=13148, months=108))
    print(dict(frames=len(merged['records']), check_only=a.check_only,
               seconds=time.monotonic()-started, source_id=plan['source_id']), flush=True)


if __name__ == '__main__':
    main()
