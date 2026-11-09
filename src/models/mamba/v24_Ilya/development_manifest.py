"""Resolve the established exact32 development cohort to explicit UTC dates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .data import EvalDataset, resolve_initialization_manifest
from .training.segment_order import file_sha256


def create_development_manifest(protocol, path):
    """Use the source year-sliced evaluator indices, not whole-store indices."""
    path = Path(path)
    reference = protocol['evaluation_reference']
    if file_sha256(reference['path']) != reference['sha256']:
        raise ValueError('Development evaluation-reference hash changed')
    saved = json.loads(Path(reference['path']).read_text())
    indices = np.asarray(protocol['chosen_idx'], dtype=np.int64)
    if indices.tolist() != saved['chosen_idx'] or len(indices) != protocol['n_samples']:
        raise ValueError('Development selected indices differ from the saved exact evaluation')
    with xr.open_zarr(protocol['data_path'], chunks=None) as dataset:
        times = pd.DatetimeIndex(pd.to_datetime(dataset.time.values))
    if times.has_duplicates or times.hasnans or not times.is_monotonic_increasing:
        raise ValueError('Development store has invalid timestamps')
    year_times = times[times.year == protocol['evaluation_year']]
    if np.any(indices < 1) or np.any(indices + protocol['target_steps'] >= len(year_times)):
        raise ValueError('Development indices lack complete input/target windows')
    starts = year_times[indices]
    payload = {'schema_version': 1, 'role': 'development',
               'source_evaluation': reference, 'source_data_path': protocol['data_path'],
               'source_chosen_idx': indices.tolist(),
               'index_semantics': 'index within source data restricted to evaluation_year; last input frame',
               'initializations': [{'initialization_time': stamp.tz_localize('UTC').isoformat().replace('+00:00', 'Z')}
                                   for stamp in starts]}
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f'Existing development manifest differs: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + '\n')
    records = resolve_initialization_manifest(
        EvalDataset(xr.Dataset(coords={'time': times.values}), pd.Timedelta('6h'), 2),
        path, target_steps=protocol['target_steps'], history_steps=0,
    )
    if len(records) != len(indices):
        raise ValueError('Development timestamp resolution count mismatch')
    return payload
