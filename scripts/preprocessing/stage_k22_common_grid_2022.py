#!/usr/bin/env python3
"""Reuse verified truth and stage just the 1-degree K22 initialization fields."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts/analyze_models')]

import numpy as np
import pandas as pd
import xarray as xr
from scripts.preprocessing import stage_common_grid_2023 as stage
from scripts.analyze_models.eval_common_grid_2023 import sha256, verify_ready
from scripts.experiments.run_k22_common_grid_2022 import write

LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


def run(workflow, workers):
    p = json.loads((workflow / 'protocol.json').read_text())
    reuse = Path(p['reuse_data_root'])
    inputs, truth = Path(p['inputs1']), Path(p['common_truth'])
    inputs.mkdir(parents=True, exist_ok=True)
    truth.mkdir(parents=True, exist_ok=True)
    starts = pd.DatetimeIndex(pd.to_datetime(p['initialization_times'], utc=True)).tz_localize(None)
    valid_times = stage.verification_times(starts)
    missing_truth = []
    for stamp in valid_times:
        old = reuse / 'common_truth' / (stage.file_id(stamp)+'.nc')
        new = truth / old.name
        if old.exists():
            verify_ready(old, stage.utc_text(stamp), 'common_truth')
            if not new.exists():
                new.symlink_to(old)
        elif not new.exists():
            missing_truth.append(stamp)
    source = stage.ChunkSource(stage.DEFAULT_URI)
    source.validate_layout()
    required = pd.DatetimeIndex(sorted(set(starts) | set(starts-pd.Timedelta('6h')) | set(missing_truth)))
    hour_lookup = dict(zip(required, map(int, source.hourly_indices(required))))
    levels = source.read_vector('level')
    selected_levels = np.array([list(levels).index(v) for v in LEVELS])
    static_stamp = starts[0]-pd.Timedelta('6h')
    statics = stage.static_fields(source, hour_lookup[static_stamp])
    base_identity = {'configuration_sha256': p['protocol_hash'], 'source_uri': source.uri,
                     'source_structure_sha256': source.fingerprint,
                     'static_source_time': stage.utc_text(static_stamp)}

    def stage_input(stamp):
        path = inputs / (stage.file_id(stamp)+'.nc')
        if path.exists():
            return verify_ready(path, stage.utc_text(stamp), 'inputs1')
        old = reuse / 'inputs025' / path.name
        identity = {**base_identity, 'kind': 'inputs1', 'timestamp': stage.utc_text(stamp),
                    'grid': '1-degree subsample of native 0.25-degree ERA5; 13 pressure levels'}
        if old.exists():
            receipt = verify_ready(old, stage.utc_text(stamp), 'inputs025')
            # The installed HDF5 library is not thread-safe.
            with stage.NETCDF_LOCK:
                with xr.open_dataset(old) as ds:
                    data = ds.sel(level=LEVELS).isel(lat=slice(None, None, 4), lon=slice(None, None, 4)).load()
                temp = path.with_suffix('.partial.nc')
                data.to_netcdf(temp)
                temp.replace(path)
            return stage.publish_receipt(path, {**identity, 'reused_input_sha256': receipt['sha256']})
        times = pd.DatetimeIndex([stamp-pd.Timedelta('6h'), stamp])
        temp = path.with_suffix('.partial.nc')
        with stage.netcdf_file(temp) as handle:
            stage.configure_file(handle, {'time': 2, 'level': 13, 'lat': 181, 'lon': 360},
                                 {'source': source.uri, 'initialization_time': stage.utc_text(stamp),
                                  'configuration_sha256': p['protocol_hash']})
            stage.coordinate(handle, 'time', times.values.astype('datetime64[h]').astype(np.int64),
                             {'units': 'hours since 1970-01-01 00:00:00', 'calendar': 'proleptic_gregorian'})
            stage.coordinate(handle, 'level', np.array(LEVELS), {'units': 'hPa'})
            stage.coordinate(handle, 'lat', np.arange(-90., 91.), {'units': 'degrees_north'})
            stage.coordinate(handle, 'lon', np.arange(360.), {'units': 'degrees_east'})
            for name in stage.NATIVE_VARIABLES:
                source_name = 'total_precipitation' if name == 'total_precipitation_6hr' else name
                dims = ('time', 'level', 'lat', 'lon') if name in stage.PRESSURE_VARIABLES else ('time', 'lat', 'lon')
                variable = stage.data_variable(handle, name, dims, stage.simple_attrs(source, source_name))
                for index, when in enumerate(times):
                    hour = hour_lookup[when]
                    if name == 'total_precipitation_6hr':
                        values = np.zeros((181, 360), dtype=np.float32)
                        for h in range(hour-5, hour+1):
                            values += stage.native_field(source, 'total_precipitation', h)[::4, ::4]
                    elif name in stage.PRESSURE_VARIABLES:
                        values = stage.native_field(source, name, hour)[selected_levels, ::4, ::4].copy()
                    else:
                        values = stage.native_field(source, name, hour)[::4, ::4].copy()
                    stage.assign(variable, index, values)
            for name, values in statics.items():
                variable = stage.data_variable(handle, name, ('lat', 'lon'), stage.simple_attrs(source, name))
                stage.assign(variable, slice(None), values[::4, ::4])
        temp.replace(path)
        return stage.publish_receipt(path, identity)

    # Prioritize the pilot date; workers perform remote reads independently.
    print('staging pilot input', stage.utc_text(starts[0]), flush=True)
    stage_input(starts[0])
    print('pilot input ready', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, _ in enumerate(pool.map(stage_input, starts[1:]), 2):
            print(f'inputs {index}/{len(starts)}', flush=True)
    for stamp in missing_truth:
        path = truth / (stage.file_id(stamp)+'.nc')
        stage.write_truth(source, path, stamp, hour_lookup, levels,
                          {**base_identity, 'kind': 'common_truth', 'timestamp': stage.utc_text(stamp)})
    receipts = [verify_ready(inputs/(stage.file_id(s)+'.nc'), stage.utc_text(s), 'inputs1') for s in starts]
    hashes = {stage.utc_text(s): verify_ready(truth/(stage.file_id(s)+'.nc'), stage.utc_text(s),
                                             'common_truth')['sha256'] for s in valid_times}
    write(workflow / 'staging_complete.json', {'status': 'complete', 'protocol_hash': p['protocol_hash'],
          'inputs': len(receipts), 'truth_times': len(hashes), 'new_truth_times': len(missing_truth),
          'truth_sha256': hashes, 'input_receipts': receipts})
    print('staging complete', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workflow', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    run(args.workflow, args.workers)
