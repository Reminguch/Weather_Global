#!/usr/bin/env python3
"""Stream the archived K22 model and its backbone on a fixed date manifest."""
from __future__ import annotations

import argparse
import dataclasses
import gc as garbage
import json
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'third_party/graphcast'), str(Path(__file__).parent)]

import numpy as np
import pandas as pd
import xarray as xr
from eval_common_grid_2023 import (CommonGridObserver, CommonTruth, load_helper, native_forcings,
                                  native_target_template, publish_new, read_json,
                                  records_from_manifest, sha256, timestamp_id, verify_file, verify_ready)


def protocol(workflow):
    import hashlib
    from scripts.experiments.frozen_runtime import verify_source
    p = read_json(workflow / 'protocol.json')
    body = {k: v for k, v in p.items() if k != 'protocol_hash'}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != p['protocol_hash']:
        raise ValueError('Protocol changed')
    snapshot = verify_source(p['source_root'])
    if snapshot['source_digest'] != p['source_digest']:
        raise ValueError('Snapshot differs from protocol')
    verify_file({'path': p['initialization_manifest'], 'sha256': p['initialization_manifest_sha256']})
    verify_file({'path': p['development_audit'], 'sha256': p['development_audit_sha256']})
    return p


def prepare_inputs(path, record, task):
    from src.models.graphcast.training.core.dataset import prepare_dataset_for_task
    with xr.open_dataset(path) as source:
        expected = pd.date_range(pd.Timestamp(record['initialization_time'])-pd.Timedelta('6h'),
                                 periods=2, freq='6h').tz_localize(None).values
        if not np.array_equal(source.time.values, expected):
            raise ValueError('Input timestamps differ from the manifest')
        stride = 4 if source.sizes['lat'] == 721 else 1
        data = source.sel(level=list(task.pressure_levels)).isel(lat=slice(None, None, stride),
                                                                lon=slice(None, None, stride)).load()
    if not (np.array_equal(data.lat, np.arange(-90., 91.)) and
            np.array_equal(data.lon, np.arange(360.))):
        raise ValueError('Expected the historical subsampled 1-degree input grid')
    data = prepare_dataset_for_task(data, task)
    data = data[list(task.input_variables)].drop_vars('datetime', errors='ignore')
    if any(not np.isfinite(v.values).all() for v in data.data_vars.values()):
        raise ValueError('Nonfinite input')
    return data.astype(np.float32).assign_coords(time=np.array([-6, 0], dtype='timedelta64[h]')).expand_dims(batch=[0])


def run(args, p):
    import haiku as hk
    import jax
    import jax.numpy as jnp
    from graphcast import casting, graphcast as graphcast_model, normalization
    from src.models.graphcast.training.core.model import load_graphcast_checkpoint, load_stats, DirectResidualNormalizer
    from src.models.mamba.training.param_utils import overlay_matching_params
    from src.models.mamba.v22_final.checkpoint import load_v22_final_checkpoint, validate_param_tree_compatible
    from src.models.mamba.v22_final.rollout import shift_inputs_with_field
    # Exact historical factories, including the legacy temporal attachments.
    from scripts.training.full_mamba_v9.train_mz_v9 import GCResidualWithZeroHead, _attach_temporal
    if len(jax.devices()) != 1 or jax.devices()[0].platform != 'gpu':
        raise RuntimeError('One allocated GPU is required')
    manifest = args.initializations or Path(p['initialization_manifest'])
    records = records_from_manifest(manifest)
    selected = records[args.shard_index::args.shard_count]
    if not selected:
        raise ValueError('Empty shard')
    output = args.output_dir / f'k22_{args.shard_index:02d}.json'
    if output.exists():
        raise FileExistsError(output)
    checkpoint = load_graphcast_checkpoint(verify_file(p['baseline']))
    residual = load_v22_final_checkpoint(verify_file(p['checkpoint']))
    for entry in p['statistics']:
        verify_file(entry)
    stats = load_stats(Path(p['stats_dir']))
    task = dataclasses.replace(checkpoint.task_config, input_duration='12h')
    config = SimpleNamespace(**p['architecture'])
    base_config = dataclasses.replace(checkpoint.model_config, resolution=1., mesh_size=5,
                                      latent_size=512, gnn_msg_steps=16)
    residual_config = dataclasses.replace(base_config, gnn_msg_steps=2)

    def base_fn(inputs, targets, forcings):
        predictor = casting.Bfloat16Cast(graphcast_model.GraphCast(base_config, task))
        predictor = normalization.InputsAndResiduals(predictor, **stats)
        return predictor(inputs, targets_template=targets, forcings=forcings)

    def residual_fn(inputs, targets, forcings):
        predictor = GCResidualWithZeroHead(residual_config, task)
        _attach_temporal(predictor, config)
        predictor = DirectResidualNormalizer(casting.Bfloat16Cast(predictor), **stats)
        return predictor(inputs, targets_template=targets, forcings=forcings)

    base_predict, residual_predict = hk.transform_with_state(base_fn), hk.transform_with_state(residual_fn)
    base_step, residual_step = jax.jit(base_predict.apply), jax.jit(residual_predict.apply)
    @jax.jit
    def finite(fields):
        return jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in jax.tree_util.tree_leaves(fields)]))
    observer = CommonGridObserver(CommonTruth(p['common_truth']), load_helper(), 40)
    params = None
    durations, input_receipts = [], []
    started = time.monotonic()
    for record in selected:
        tick = time.monotonic()
        path = Path(p['inputs1']) / (timestamp_id(record['initialization_time'])+'.nc')
        input_receipts.append(verify_ready(path, record['initialization_time'], 'inputs1'))
        inputs = prepare_inputs(path, record, task)
        template = native_target_template(inputs, task)
        forcing = native_forcings(inputs, record['valid_times'][0], 6, task)
        if params is None:
            rng, kb, kr = jax.random.split(jax.random.PRNGKey(0), 3)
            params, base_state = base_predict.init(kb, inputs, template.assign_coords(time=forcing.time), forcing)
            params, _ = overlay_matching_params(params, checkpoint.params, strict=True)
            initialized, zero_state = residual_predict.init(kr, inputs, template.assign_coords(time=forcing.time), forcing)
            validate_param_tree_compatible(initialized, residual.residual_params)
            del initialized
        current_b, current_f = inputs, inputs
        state_b, state_f, memory = base_state, base_state, zero_state
        # No stochastic layers are active; keep a fixed, recorded key per forecast.
        rng = jax.random.PRNGKey(0)
        observer.begin_sample(record)
        for index, valid in enumerate(record['valid_times']):
            forcing = native_forcings(inputs, valid, 6*(index+1), task) if index else forcing
            target = template.assign_coords(time=forcing.time)
            rng, kb, kr = jax.random.split(rng, 3)
            baseline, state_b = base_step(params, state_b, kb, current_b, target, forcing)
            uncorrected, state_f = base_step(params, state_f, kb, current_f, target, forcing)
            correction, memory = residual_step(residual.residual_params, memory, kr, current_f, target, forcing)
            full = jax.tree_util.tree_map(lambda b, r: b+r, uncorrected, correction)
            if not bool(finite((baseline, full))):
                raise ValueError(f'Nonfinite forecast at {valid}')
            observer.score(index, {'gc1': baseline, 'k22': full})
            if index < 39:
                current_b = shift_inputs_with_field(current_b, baseline, forcing,
                                                    time_step=pd.Timedelta('6h'), input_steps=2)
                current_f = shift_inputs_with_field(current_f, full, forcing,
                                                    time_step=pd.Timedelta('6h'), input_steps=2)
            del baseline, full, uncorrected, correction, target
        observer.finish_sample()
        durations.append(time.monotonic()-tick)
        # Preserve completed per-start evidence even if a later start fails.
        publish_new(args.output_dir / f'k22_{args.shard_index:02d}_{timestamp_id(record["initialization_time"])}.start.json',
                    {'protocol_hash': p['protocol_hash'], 'per_initialization': [observer.records[-1]]})
        print(f'[k22] {record["initialization_time"]} {durations[-1]:.1f}s', flush=True)
        del inputs, template, forcing, current_b, current_f, memory, state_b, state_f
        garbage.collect()
    publish_new(output, {'schema_version': 1, 'status': 'complete', 'protocol_hash': p['protocol_hash'],
                        'initialization_manifest_sha256': sha256(manifest),
                        'shard_index': args.shard_index, 'shard_count': args.shard_count,
                        'per_initialization': observer.records, 'inputs': input_receipts,
                        'checkpoint_sha256': p['checkpoint']['sha256'],
                        'elapsed_seconds': time.monotonic()-started, 'sample_elapsed_seconds': durations,
                        'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                        'gpu_kind': jax.devices()[0].device_kind, 'job_id': os.getenv('SLURM_JOB_ID')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workflow', type=Path, required=True)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--initializations', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        parser.error('Invalid shard bounds')
    run(args, protocol(args.workflow))
