#!/usr/bin/env python3
"""Replay frozen K=1 validation with physical-unit errors and dated predictions."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    choice = plan['runs'][args.run_id]
    root = Path(plan['experiment_root'])
    raw = json.loads((root / 'manifest.json').read_text())
    source = Path(raw['source_root'])
    for path in (source / 'third_party/graphcast', source / 'third_party/neuralgcm', source):
        sys.path.insert(0, str(path))
    from src.models.neuralgcm_residual.numerics import configure_environment
    configure_environment()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from src.models.neuralgcm_residual.cache import CacheReader, live_record
    from src.models.neuralgcm_residual.checkpoint import load_checkpoint
    from src.models.neuralgcm_residual.checks import backbone_parameter_digest
    from src.models.neuralgcm_residual.config import load_config, FIELDS
    from src.models.neuralgcm_residual.io import sha256, versions
    from src.models.neuralgcm_residual.launcher import load_experiment, resolve_resources
    from src.models.neuralgcm_residual.model import zero_memory
    from src.models.neuralgcm_residual.runtime import build_runtime, runtime_identities

    started = time.perf_counter()
    if not jax.devices() or any(d.platform != 'gpu' for d in jax.devices()):
        raise RuntimeError('This diagnostic requires a Slurm GPU allocation')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = load_experiment(root)
    if versions() != manifest['libraries']:
        raise ValueError('Library versions differ from the frozen training environment')
    config = load_config(manifest['configs'][args.run_id]['path'])
    rt = build_runtime(config, resolve_resources(manifest, config.resolution_id), stage='pretrain')
    resources = resolve_resources(manifest, config.resolution_id)
    reader = CacheReader(resources['cache_root'], rt.store, 'val')
    identities = runtime_identities(rt, config, manifest['source_id'], reader.manifest['producer_id'])
    checkpoint = Path(choice['checkpoint'])
    if sha256(checkpoint) != choice['checkpoint_sha256']:
        raise ValueError('The requested checkpoint has changed')
    saved = load_checkpoint(checkpoint, stage='pretrain', identities=identities)
    params = saved['params']
    if saved['cursor']['update'] != choice['update']:
        raise ValueError('Checkpoint update differs from evaluation plan')
    del saved
    backbone_before = backbone_parameter_digest(rt.backbone.model)
    b, trainer = rt.backbone, rt.trainer
    latitude = np.asarray(b.model.data_coords.horizontal.latitudes)
    longitude = np.asarray(b.model.data_coords.horizontal.longitudes)
    levels = np.asarray(b.model.data_coords.vertical.centers)
    nodes, area64 = np.polynomial.legendre.leggauss(len(latitude))
    np.testing.assert_allclose(nodes, np.sin(latitude), atol=2e-6)
    area64 /= area64.sum()
    area = jnp.asarray(area64, dtype=jnp.float32)
    sites = []
    for site in plan['sites']:
        lat = int(np.argmin(abs(np.rad2deg(latitude) - site['latitude'])))
        distance = abs((np.rad2deg(longitude) - site['longitude'] + 180) % 360 - 180)
        lon = int(np.argmin(distance))
        sites.append(dict(site, latitude_index=lat, longitude_index=lon,
                          grid_latitude=float(np.rad2deg(latitude[lat])),
                          grid_longitude=float((np.rad2deg(longitude[lon]) + 180) % 360 - 180)))
    unit_names = ['K', 'm2/s2', 'm/s', 'm/s', 'kg/kg', 'kg/kg', 'kg/kg']
    site_lon = np.array([s['longitude_index'] for s in sites])
    site_lat = np.array([s['latitude_index'] for s in sites])

    def reduce_fields(decoded, target):
        stats, means, points = [], [], []
        for field in FIELDS:
            p, t = decoded[field], target[field]
            error = p - t
            avg = lambda x: jnp.mean(jnp.sum(x * area, axis=-1), axis=-1)
            stats.append(jnp.stack([avg(error ** 2), avg(abs(error)), avg(error)]))
            means.append(avg(p))
            points.append(p[:, site_lon, site_lat])
        return jnp.stack(stats), jnp.stack(means), jnp.stack(points)

    @jax.jit
    def inference(p, memory, key, item):
        native, known, baseline, forcing, target = item
        delta, next_memory = rt.branch.apply(p, memory, key, native, known)
        corrected = rt.adapter.apply_increment(baseline, rt.normalization.increment(delta))
        pred, base = b.decode(corrected, forcing), b.decode(baseline, forcing)
        return next_memory, {
            'physical': jnp.stack([reduce_fields(x, target)[0] for x in (base, pred)]),
            'global_mean': jnp.stack([reduce_fields(x, target)[1] for x in (target, base, pred)]),
            'site_values': jnp.stack([reduce_fields(x, target)[2] for x in (target, base, pred)]),
            'field_scores': jnp.array([[trainer.weather_loss.field_scores(x, target)[k]
                                        for k in FIELDS] for x in (base, pred)]),
        }

    @jax.jit
    def reference(p, memory, key, item):
        loss, next_memory, corrected = trainer.forward(p, memory, key, *item)
        return loss, next_memory, b.decode(item[2], item[3]), b.decode(corrected, item[3])

    def tree_difference(x, y):
        differences = []
        for a, z in zip(jax.tree_util.tree_leaves(x), jax.tree_util.tree_leaves(y), strict=True):
            a, z = np.asarray(a), np.asarray(z)
            if a.size:
                differences.append(float(np.max(abs(a.astype(np.float64) - z.astype(np.float64)))))
        return max(differences, default=0.0)

    sample = set(np.linspace(0, len(reader) - 1, 8, dtype=int).tolist())
    indices = sorted(sample) if args.smoke else range(len(reader))
    stats_sum = np.zeros((2, len(FIELDS), 3, len(levels)), np.float64)
    scores_sum = np.zeros((2, len(FIELDS)), np.float64)
    memory, previous, since_reset = zero_memory(rt.memory), None, 0
    checks, live_checks, times, site_values, global_values, field_scores = [], [], [], [], [], []
    for count, i in enumerate(indices, 1):
        record = reader[i]
        origin = np.datetime64(record['origin'], 'h')
        valid = np.datetime64(record['target_time'], 'h')
        if valid != origin + np.timedelta64(6, 'h'):
            raise ValueError('Prediction valid time is not six hours after origin')
        if since_reset == 96 or (previous is not None and origin - previous != np.timedelta64(6, 'h')):
            memory, since_reset = zero_memory(rt.memory), 0
        item = trainer.record(record['origin_state'], record['baseline_state'], record['forcing'],
                              record['target'], rt.known(record['origin_state'], record['forcing']))
        prior_memory = memory
        memory, result = inference(params, memory, jax.random.PRNGKey(i), item)
        result = jax.device_get(result)
        if any(not np.isfinite(x).all() for x in result.values()):
            raise FloatingPointError(f'Nonfinite physical prediction at {origin}')
        if i in sample:
            ref_loss, ref_memory, base, pred = jax.device_get(reference(params, prior_memory, jax.random.PRNGKey(i), item))
            np.testing.assert_allclose(ref_loss, result['field_scores'][1].mean(), rtol=2e-5, atol=1e-7)
            memory_difference = tree_difference(memory, ref_memory)
            if memory_difference != 0:
                raise AssertionError('Diagnostic changed the production recurrent-memory update')
            max_relative_error = 0.0
            for model_index, decoded in enumerate((base, pred)):
                scores = []
                for field_index, field in enumerate(FIELDS):
                    error = np.asarray(decoded[field], np.float64) - np.asarray(record['target'][field], np.float64)
                    mse = np.mean(np.sum(error ** 2 * area64, axis=-1), axis=-1)
                    np.testing.assert_allclose(result['physical'][model_index, field_index, 0], mse,
                                               rtol=2e-4, atol=1e-12)
                    scales = np.asarray(trainer.weather_loss.scales[field], np.float64)
                    scores.append(float(np.sum(mse / scales ** 2 * np.asarray(trainer.weather_loss.level_weights))))
                np.testing.assert_allclose(result['field_scores'][model_index], scores, rtol=2e-4, atol=1e-9)
                max_relative_error = max(max_relative_error, float(np.max(abs(np.asarray(scores) - result['field_scores'][model_index]) / np.maximum(abs(np.asarray(scores)), 1e-12))))
            checks.append(dict(index=i, origin=str(origin), production_forward_loss=float(ref_loss),
                               memory_max_abs=memory_difference, numpy_score_max_relative_error=max_relative_error))
            if args.smoke:
                live = live_record(b, rt.store, record['origin'])
                difference = {k: tree_difference(live[k], record[k])
                              for k in ('origin_state', 'baseline_state', 'forcing')}
                if any(value != 0 for value in difference.values()):
                    raise AssertionError(f'Cached and live physical states differ: {difference}')
                live_checks.append(dict(origin=str(origin), max_abs=difference))
        stats_sum += result['physical']
        scores_sum += result['field_scores']
        field_scores.append(result['field_scores'])
        global_values.append(result['global_mean'])
        site_values.append(result['site_values'])
        times.append(str(valid))
        previous, since_reset = origin, since_reset + 1
        if count % 100 == 0 or count == len(indices):
            progress = dict(records=count, total=len(indices), valid_time=str(valid),
                            baseline_loss=float(scores_sum[0].mean() / count),
                            residual_loss=float(scores_sum[1].mean() / count),
                            elapsed_seconds=time.perf_counter() - started)
            print(json.dumps(progress), flush=True)
            (args.output / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')

    n = len(indices)
    scores = scores_sum / n
    if not args.smoke:
        np.testing.assert_allclose(scores[1].mean(), choice['validation_loss'], rtol=2e-5, atol=1e-7)
        np.testing.assert_allclose(scores[0].mean(), choice['baseline_loss'], rtol=2e-5, atol=1e-7)
    fields = {}
    csv_rows = []
    for j, field in enumerate(FIELDS):
        mse = stats_sum[:, j, 0] / n
        fields[field] = dict(unit=unit_names[j], objective_score_baseline=float(scores[0, j]),
                             objective_score_residual=float(scores[1, j]),
                             objective_baseline_share_pct=float(100 * scores[0, j] / scores[0].sum()))
        for k, pressure in enumerate(levels):
            csv_rows.append(dict(run_id=args.run_id, field=field, pressure_hpa=float(pressure), unit=unit_names[j],
                                 baseline_rmse=float(np.sqrt(mse[0, k])), residual_rmse=float(np.sqrt(mse[1, k])),
                                 rmse_improvement_pct=float(100 * (1 - np.sqrt(mse[1, k] / mse[0, k]))),
                                 baseline_mae=float(stats_sum[0, j, 1, k] / n), residual_mae=float(stats_sum[1, j, 1, k] / n),
                                 baseline_bias=float(stats_sum[0, j, 2, k] / n), residual_bias=float(stats_sum[1, j, 2, k] / n)))
    with (args.output / 'physical_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(csv_rows)
    np.savez_compressed(args.output / 'timeseries.npz', valid_time=np.array(times),
                        fields=np.array(FIELDS), levels_hpa=levels, models=np.array(['ERA5', 'NGCM', 'Residual NGCM']),
                        sites=np.array([s['name'] for s in sites]), site_values=np.array(site_values),
                        global_mean=np.array(global_values), field_scores=np.array(field_scores))
    if sha256(checkpoint) != choice['checkpoint_sha256'] or backbone_parameter_digest(b.model) != backbone_before:
        raise AssertionError('Evaluation changed model weights')
    report = dict(completed=True, smoke=args.smoke, run_id=args.run_id, records=n, checkpoint=choice,
                  experiment_root=str(root), source_id=manifest['source_id'], identities=identities,
                  script_sha256=sha256(__file__), plan_sha256=sha256(args.plan),
                  job_id=os.environ.get('SLURM_JOB_ID'), devices=[d.device_kind for d in jax.devices()],
                  captured_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - started,
                  prediction_horizon_hours=6, split='2022 validation', sites=sites,
                  spatial_reduction='Gaussian area weights over latitude, uniform over longitude',
                  rmse_reduction='sqrt(mean over all dates of global area-weighted squared error)',
                  physical_initialization='ERA5 encoded anew at every origin; no autoregressive physical feedback',
                  memory_policy='production validation: chronological memory, reset every 96 records or at gaps',
                  unavailable_variables={'2m_temperature': 'No 2 m temperature output or target in this checkpoint/data contract'},
                  decoded_variables=list(FIELDS), fields=fields, numpy_and_production_forward_checks=checks,
                  live_cache_checks=live_checks, baseline_loss=float(scores[0].mean()), residual_loss=float(scores[1].mean()),
                  objective_improvement_pct=float(100 * (1 - scores[1].mean() / scores[0].mean())),
                  production_checkpoint_unchanged=True, frozen_backbone_unchanged=True,
                  artifacts={name: sha256(args.output / name) for name in ('physical_metrics.csv', 'timeseries.npz')})
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'completed': True, 'report': str(args.output / 'report.json')}), flush=True)


if __name__ == '__main__':
    main()
