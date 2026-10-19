#!/usr/bin/env python3
"""Independent upstream GraphCast rollout on the frozen common-grid inputs.

Uses upstream model construction, direct checkpoint parameters, and upstream
chunked autoregression. It never imports the project's GraphCast/model wrappers.
The common scorer is shared deliberately: the intervention is forecast inference.
Writes a separate audit, without changing the production protocol or results.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import xarray as xr


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workflow', type=Path, required=True)
    parser.add_argument('--initialization', default='2023-01-01T00:00:00Z')
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    workflow = args.workflow.resolve()
    protocol = read(workflow / 'protocol.json')
    original = read(Path(protocol['original_campaign']) / 'protocol.json')
    source = workflow / 'audits/upstream_graphcast'
    expected_commit = '08cf73625c9d12bd9aaa038868bcb2fe488f2a22'
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != expected_commit or subprocess.check_output(
            ['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Upstream checkout identity or clean status changed')
    if not 1 <= args.steps <= 40:
        raise ValueError('Expected 1..40 leads')
    sys.path.insert(0, str(source))
    os.environ['XLA_FLAGS'] = original['numerics']['xla_flags']
    os.environ.update(original['numerics'].get('environment', {}))
    import haiku as hk
    import jax
    from graphcast import autoregressive, casting, checkpoint, data_utils
    from graphcast import graphcast as gc, normalization, rollout
    if not Path(gc.__file__).resolve().is_relative_to(source):
        raise ValueError('GraphCast did not import from isolated upstream checkout')
    if args.prepare_only:
        if any(d.platform != 'cpu' for d in jax.devices()):
            raise ValueError('Preparation audit requires explicit CPU backend')
    elif len(jax.devices()) != 1 or jax.devices()[0].platform != 'gpu':
        raise ValueError('Forecast audit requires one allocated GPU')
    evaluator = module_at('stock_audit_evaluator', workflow / 'source/scripts/analyze_models/eval_common_grid_2023.py')
    helper = evaluator.load_helper()
    ckpt_path = evaluator.verify_file({'path': protocol['gc025_checkpoint'],
                                     'sha256': protocol['gc025_checkpoint_sha256']})
    with ckpt_path.open('rb') as stream:
        ckpt = checkpoint.load(stream, gc.CheckPoint)
    task = ckpt.task_config
    stamp = pd.Timestamp(args.initialization)
    if stamp.tzinfo is None or stamp.utcoffset() != pd.Timedelta(0):
        raise ValueError('Explicit UTC initialization required')
    tag = stamp.strftime('%Y%m%dT%H%M%SZ')
    destination = workflow / 'audits' / ('stock_prepare_' + tag + '.json' if args.prepare_only else 'stock_rollout_' + tag)
    if destination.exists():
        raise FileExistsError(destination)
    input_path = Path(protocol['inputs025']) / (tag + '.nc')
    receipt = evaluator.verify_ready(input_path, args.initialization, 'inputs025')
    raw = xr.load_dataset(input_path).sel(level=list(task.pressure_levels))
    input_times = pd.date_range(stamp - pd.Timedelta('6h'), periods=2, freq='6h').tz_localize(None).values
    np.testing.assert_array_equal(raw.time.values, input_times)
    np.testing.assert_array_equal(raw.lat, np.arange(-90, 90.25, .25))
    np.testing.assert_array_equal(raw.lon, np.arange(0, 360, .25))
    times = pd.date_range(stamp - pd.Timedelta('6h'), periods=args.steps + 2, freq='6h').tz_localize(None).values
    # No meteorological targets enter this data structure. All future fields
    # here are deterministic calendar/solar forcings from the upstream API.
    generated = xr.Dataset(coords={'time': times, 'datetime': ('time', times),
                                  'lat': raw.lat.values, 'lon': raw.lon.values})
    data_utils.add_derived_vars(generated)
    data_utils.add_tisr_var(generated)
    inputs = xr.merge([raw, generated[list(task.forcing_variables)].isel(time=slice(0, 2)).drop_vars('datetime')])
    inputs = inputs[list(task.input_variables)].astype(np.float32).assign_coords(
        time=np.array([-6, 0], dtype='timedelta64[h]')).expand_dims(batch=[0])
    forcings = generated[list(task.forcing_variables)].isel(time=slice(2, None)).drop_vars('datetime')
    forcings = forcings.astype(np.float32).assign_coords(
        time=np.arange(1, args.steps + 1) * np.timedelta64(6, 'h')).expand_dims(batch=[0]).load()
    del raw, generated
    input_hashes = {}
    for name in inputs:
        a = np.ascontiguousarray(inputs[name].values)
        if not np.isfinite(a).all():
            raise ValueError('Nonfinite input: ' + name)
        input_hashes[name] = {'shape': list(a.shape), 'dims': list(inputs[name].dims),
                              'sha256': hashlib.sha256(a.tobytes()).hexdigest()}
    del a
    evidence = {'upstream_commit': commit, 'protocol_hash': protocol['protocol_hash'],
                'initialization_time': args.initialization, 'steps': args.steps,
                'checkpoint_sha256': sha(ckpt_path), 'input_receipt': receipt,
                'input_hashes': input_hashes, 'script_sha256': sha(__file__),
                'graphcast_module': gc.__file__, 'jax_version': jax.__version__,
                'devices': [str(d) for d in jax.devices()], 'slurm_job_id': os.environ.get('SLURM_JOB_ID')}
    if args.prepare_only:
        destination.write_text(json.dumps(evidence, indent=2) + '\n')
        print(destination, flush=True)
        return
    destination.mkdir()
    (destination / 'provenance.json').write_text(json.dumps(evidence, indent=2) + '\n')
    stats = {name: xr.load_dataset(Path(protocol['stats_dir']) / (name + '.nc'))
             for name in ('stddev_by_level', 'mean_by_level', 'diffs_stddev_by_level')}
    for record in protocol['statistics']:
        evaluator.verify_file(record)

    def predict(i, t, f):
        predictor = gc.GraphCast(ckpt.model_config, task)
        predictor = casting.Bfloat16Cast(predictor)
        predictor = normalization.InputsAndResiduals(predictor, **stats)
        predictor = autoregressive.Predictor(predictor, gradient_checkpointing=False)
        return predictor(i, targets_template=t, forcings=f)

    transformed = hk.transform_with_state(predict)
    # Direct checkpoint use, as in the upstream demonstration; no initialization
    # or parameter overlay and no project-specific recurrent state.
    @jax.jit
    def apply(rng, inputs, targets_template, forcings):
        return transformed.apply(ckpt.params, {}, rng, inputs, targets_template, forcings)[0]

    one = xr.zeros_like(inputs[list(task.target_variables)].isel(time=slice(-1, None))).assign_coords(
        time=np.array([6], dtype='timedelta64[h]'))
    targets = rollout.extend_targets_template(one, args.steps)
    truth_store = evaluator.CommonTruth(Path(protocol['common_truth']))
    records = []
    started = time.monotonic()
    for index, prediction in enumerate(rollout.chunked_prediction_generator(
            apply, jax.random.PRNGKey(0), inputs, targets_template=targets,
            forcings=forcings, num_steps_per_chunk=1)):
        valid = stamp + pd.Timedelta(hours=6 * (index + 1))
        truth, truth_sha = truth_store.at(valid)
        common = helper.regrid_diagnostics(prediction)
        common.to_netcdf(destination / f'common_{index + 1:02d}.nc')
        if index == 0:
            helper.select_diagnostics(prediction).to_netcdf(destination / 'native_diagnostics_01.nc')
        scores = helper.score_step(prediction, truth)
        records.append({'lead_hours': 6 * (index + 1), 'valid_time': valid.isoformat(),
                        'mse': scores, 'truth_sha256': truth_sha})
        print(json.dumps({'lead_hours': records[-1]['lead_hours'],
                          'Z500_rmse': float(np.sqrt(scores['Z500'])),
                          'elapsed_seconds': time.monotonic() - started}), flush=True)
    production = [read(p) for p in (workflow / 'shards').glob('gc025_*.json')]
    original_record = next(r for shard in production for r in shard['per_initialization']
                           if pd.Timestamp(r['initialization_time']) == stamp)
    comparison = {}
    for key in helper.DIAGNOSTICS:
        new = np.sqrt([r['mse'][key] for r in records])
        old = np.sqrt(original_record['mse']['gc025'][key][:args.steps])
        comparison[key] = {'stock_rmse': new.tolist(), 'production_rmse': old.tolist(),
                           'stock_vs_production_rmse_change_pct': (100 * (new / old - 1)).tolist()}
    evidence.update(status='complete', records=records, comparison=comparison,
                    elapsed_seconds=time.monotonic() - started,
                    peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                    limitation='One initialization; shared input data and common scorer; not an independent data-source validation.')
    (destination / 'result.json').write_text(json.dumps(evidence, indent=2, allow_nan=False) + '\n')
    print(destination / 'result.json', flush=True)


if __name__ == '__main__':
    main()
