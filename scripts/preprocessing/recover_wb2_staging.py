"""Repair one interrupted annual append, then resume the frozen WB2 producer.

Run on a network-enabled login host through graphcast_env.sh. The original
producer lock is held throughout; old status, metadata and stale lock are saved.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import sys
from datetime import datetime, timezone


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def write_year_bounded(year_data, output, *, mode, block_steps=8, progress=None):
    """Materialize at most one small weather block before each append."""
    import numpy as np
    import zarr
    if block_steps < 1:
        raise ValueError('block_steps must be positive')
    for offset in range(0, year_data.sizes['time'], block_steps):
        stop = min(year_data.sizes['time'], offset + block_steps)
        block = year_data.isel(time=slice(offset, stop)).load()
        if not all(np.isfinite(v.values).all() for v in block.data_vars.values()):
            raise ValueError('Nonfinite downloaded weather')
        if mode == 'w' and offset == 0:
            block.to_zarr(output, mode='w', consolidated=False, zarr_format=2)
        else:
            block.to_zarr(output, mode='a', append_dim='time', consolidated=False, zarr_format=2)
        if progress:
            progress(stop, year_data.sizes['time'])
        del block
    zarr.consolidate_metadata(str(output))


def repair_year(year_data, output, year, *, block_steps=8, progress=None):
    """Replace weather only, requiring one-timestep storage chunks and exact coordinates."""
    import numpy as np
    import pandas as pd
    import xarray as xr
    import zarr

    expected = pd.date_range(f'{year}-01-01', f'{year}-12-31T18:00', freq='6h').values
    np.testing.assert_array_equal(year_data.time.values, expected)
    with xr.open_zarr(output, consolidated=False, chunks=None) as local:
        all_times = local.time.values
        positions = np.flatnonzero((all_times >= expected[0]) & (all_times <= expected[-1]))
        np.testing.assert_array_equal(all_times[positions], expected[:len(positions)])
        if not len(positions):
            raise ValueError('Interrupted year has no stored prefix')
        start, stop = int(positions[0]), int(positions[-1]) + 1
        if stop != len(all_times):
            raise ValueError('Only the interrupted final year may be repaired')
        names = [name for name in local.data_vars if 'time' in local[name].dims]
        if set(names) != {name for name in year_data.data_vars if 'time' in year_data[name].dims}:
            raise ValueError('Weather variable coverage differs')
        group = zarr.open_group(str(output), mode='r')
        for name in names:
            arr = group[name]
            axis = local[name].dims.index('time')
            if arr.chunks[axis] != 1:
                raise ValueError('Repair requires one-timestep weather chunks')
            for dim in local[name].dims:
                if dim != 'time':
                    np.testing.assert_array_equal(local[dim].values, year_data[dim].values)
            if local[name].dtype != year_data[name].dtype:
                raise ValueError(f'Dtype differs: {name}')
        # Coordinates already passed exact checks. Excluding them avoids partial
        # rewrites of the large time-coordinate chunk and leaves prior years intact.
        if block_steps < 1:
            raise ValueError('block_steps must be positive')
        for offset in range(0, len(positions), block_steps):
            end = min(len(positions), offset + block_steps)
            block = year_data[names].isel(time=slice(offset, end)).load()
            if not all(np.isfinite(v.values).all() for v in block.data_vars.values()):
                raise ValueError('Nonfinite downloaded weather')
            payload = block.drop_vars(list(block.coords), errors='ignore')
            payload.to_zarr(output, mode='r+', region={'time': slice(start + offset, start + end)},
                            consolidated=False, zarr_format=2)
            if progress:
                progress(end, len(expected))
            del payload, block
    if len(positions) < len(expected):
        def appended(done, total):
            if progress:
                progress(len(positions) + done, len(expected))
        write_year_bounded(year_data.isel(time=slice(len(positions), None)), output,
                           mode='a', block_steps=block_steps, progress=appended)
        stop = start + len(expected)
    zarr.consolidate_metadata(str(output))
    # Read every repaired field in bounded blocks before marking the year complete.
    hashes = {}
    with xr.open_zarr(output, consolidated=True, chunks=None) as repaired:
        for name in names:
            checksum = hashlib.sha256()
            for offset in range(start, stop, 8):
                values = np.asarray(repaired[name].isel(time=slice(offset, min(stop, offset + 8))).values)
                if not np.isfinite(values).all():
                    raise ValueError(f'Nonfinite repaired values: {name}')
                checksum.update(np.ascontiguousarray(values).tobytes())
            hashes[name] = checksum.hexdigest()
    return dict(year=year, start=start, stop=stop, weather_sha256=hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staging-root', type=Path, required=True)
    parser.add_argument('--recovery-root', type=Path, required=True)
    parser.add_argument('--repair-year', type=int, required=True)
    parser.add_argument('--previous-recovery', type=Path)
    parser.add_argument('--block-steps', type=int, default=8)
    args = parser.parse_args()
    original, root = args.staging_root.resolve(), args.recovery_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / 'download_status.json'
    if status_path.exists():
        raise RuntimeError('Recovery already started; inspect its status before retrying')
    with (original / 'download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = json.loads((original / 'download_status.json').read_text())
        if args.previous_recovery:
            previous = args.previous_recovery.resolve()
            old = json.loads((previous / 'download_status.json').read_text())
        if old['hostname'] != socket.gethostname():
            raise RuntimeError('Recovery must run on the original downloader host')
        try:
            os.kill(old['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('Original PID still exists; inspect before recovery')
        source_path = original / 'stage_wb2_snapshot.py'
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != old['source_sha256']:
            raise ValueError('Original frozen producer changed')
        output = Path(old['output'])
        backup = root / 'before'
        backup.mkdir(exist_ok=False)
        shutil.copy2(original / 'download_status.json', backup / 'download_status.json')
        shutil.copy2(original / 'download.log', backup / 'download.log')
        if args.previous_recovery:
            shutil.copytree(previous, backup / 'previous_recovery')
        for path in output.rglob('.*'):
            if path.is_file() and path.name in ('.zarray', '.zattrs', '.zgroup', '.zmetadata'):
                target = backup / 'metadata' / path.relative_to(output)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        stale = output.with_name(output.name + '.lock')
        if stale.exists():
            if f'pid={old["pid"]} ' not in stale.read_text():
                raise ValueError('Unexpected staging lock owner')
            stale.rename(backup / stale.name)
        record = dict(old, status='running', pid=os.getpid(),
                      started_at=datetime.now(timezone.utc).isoformat(),
                      recovery_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      recovery_root=str(root), phase='repair', current_year=args.repair_year,
                      dask_workers=2, block_steps=args.block_steps)
        save(status_path, record)
        save(original / 'download_status.json', dict(old, status='interrupted', superseded_by=str(status_path)))
        try:
            import dask
            import xarray as xr
            module_spec = importlib.util.spec_from_file_location('frozen_wb2_recovery', source_path)
            producer = importlib.util.module_from_spec(module_spec)
            sys.modules[module_spec.name] = producer
            module_spec.loader.exec_module(producer)
            def progress(done, total):
                record.update(completed_year_steps=done, total_year_steps=total,
                              updated_at=datetime.now(timezone.utc).isoformat())
                save(status_path, record)
                print(f"{record['phase']} {record['current_year']}: {done}/{total}", flush=True)

            with dask.config.set(scheduler='threads', num_workers=2):
                with xr.open_zarr(producer.DEFAULT_URI, consolidated=True, storage_options={'token': 'anon'}) as remote:
                    remote = producer._ensure_datetime_time(producer._normalize_coords(remote))
                    names = producer._pick_vars(remote.data_vars, include_tisr=True)
                    year_data = producer._prepare_year_chunk(remote, args.repair_year, names, args.block_steps)
                    print(f'Repairing weather fields for {args.repair_year}', flush=True)
                    result = repair_year(year_data, output, args.repair_year,
                                         block_steps=args.block_steps, progress=progress)
                    save(root / 'repair.json', result)
                def tracked_write(dataset, destination, *, mode):
                    year = int(str(dataset.time.values[0])[:4])
                    record.update(phase='append', current_year=year)
                    save(status_path, record)
                    write_year_bounded(dataset, destination, mode=mode,
                                       block_steps=args.block_steps, progress=progress)
                    record['last_completed_year'] = year
                    save(status_path, record)

                producer._write_year = tracked_write
                record.update(phase='append', last_completed_year=args.repair_year)
                save(status_path, record)
                sys.argv = [str(source_path), '--start-year', '1982', '--end-year', '2022',
                            '--output', str(output), '--chunk-time', str(args.block_steps)]
                producer.main()
        except BaseException as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            save(status_path, record)
            raise
        record.update(status='complete', phase='complete', ended_at=datetime.now(timezone.utc).isoformat())
        save(status_path, record)


if __name__ == '__main__':
    main()
