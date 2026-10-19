#!/usr/bin/env python3
"""Fit train-only change scales and rescore existing physical MSEs on CPU.

This is an offline audit. It does not change the active training objective,
input normalization, correction units, checkpoints, or the archived forecasts.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


FIELDS = ('temperature', 'geopotential', 'u_component_of_wind',
          'v_component_of_wind', 'specific_humidity',
          'specific_cloud_ice_water_content', 'specific_cloud_liquid_water_content')
AMPLITUDES = {k: 2.0 if k == 'geopotential' else 0.66 if k == 'specific_humidity'
              else 0.05 if k.startswith('specific_cloud_') else 1.0 for k in FIELDS}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ChangeMoments:
    """Population moments over equally weighted snapshots, including mean shifts."""

    def __init__(self, levels, latitude_weights, per_level=False):
        self.weights = np.asarray(latitude_weights, np.float64)
        self.weights = self.weights / self.weights.sum()
        self.per_level = per_level
        self.count = 0
        self.mean = np.zeros(levels if per_level else 1, np.float64)
        self.m2 = np.zeros_like(self.mean)

    def add(self, x):
        x = np.asarray(x, np.float64)
        if x.ndim != 3 or not np.isfinite(x).all():
            raise ValueError('Expected finite [level, longitude, latitude] changes')
        per_level_mean = np.mean(np.sum(x * self.weights, axis=-1), axis=-1)
        mean = per_level_mean if self.per_level else np.array([per_level_mean.mean()])
        centered = x - (mean[:, None, None] if self.per_level else mean[0])
        per_level_var = np.mean(np.sum(centered ** 2 * self.weights, axis=-1), axis=-1)
        var = per_level_var if self.per_level else np.array([per_level_var.mean()])
        self.count += 1
        delta = mean - self.mean
        self.mean += delta / self.count
        self.m2 += var + delta * (mean - self.mean)

    def std(self):
        if not self.count:
            raise ValueError('No training snapshots')
        value = np.sqrt(self.m2 / self.count)
        if not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError('Degenerate scales; no undocumented floor is applied')
        return value


def fit_scales(prepared, *, samples=60):
    manifest_path = prepared / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    ready = json.loads((prepared / 'READY.json').read_text())
    if sha(manifest_path) != ready['manifest_sha256']:
        raise ValueError('Prepared data manifest hash differs from READY')
    records = {r['time']: r for r in manifest['records']}
    times = np.array(sorted(records), dtype='datetime64[h]')
    train = times[(times >= np.datetime64('2015-01-01')) &
                  (times < np.datetime64('2022-01-01'))]
    available = set(train)
    candidates = [t for t in train if all(t + np.timedelta64(h, 'h') in available for h in (6, 24))]
    if samples < 2 or samples > len(candidates):
        raise ValueError('Invalid snapshot count')
    origins = [candidates[i] for i in np.linspace(0, len(candidates) - 1, samples, dtype=int)]
    grid = manifest['data_grid']
    levels = np.asarray(grid['levels'])
    nodes, area = np.polynomial.legendre.leggauss(len(grid['latitude']))
    np.testing.assert_allclose(nodes, np.sin(grid['latitude']), atol=2e-6)
    weights = {'uniform': np.ones_like(area), 'gaussian': area}
    moments = {(lag, weighting, field): ChangeMoments(len(levels), w, field == 'specific_humidity')
               for lag in (6, 24) for weighting, w in weights.items() for field in FIELDS}
    consumed = {}

    def frame(t):
        record = records[str(t)]
        path = prepared / record['file']
        digest = sha(path)
        if digest != record['sha256']:
            raise ValueError(f'ERA5 frame hash changed: {t}')
        consumed[str(t)] = digest
        with np.load(path, allow_pickle=False) as data:
            result = {f: np.asarray(data[f], np.float64) for f in FIELDS}
        expected = (len(levels), len(grid['longitude']), len(grid['latitude']))
        if any(v.shape != expected or not np.isfinite(v).all() for v in result.values()):
            raise ValueError(f'Invalid frame shape or values at {t}')
        return result

    for i, origin in enumerate(origins, 1):
        initial = frame(origin)
        for lag in (6, 24):
            target = frame(origin + np.timedelta64(lag, 'h'))
            for field in FIELDS:
                change = target[field] - initial[field]
                for weighting in weights:
                    moments[lag, weighting, field].add(change)
        if i % 10 == 0:
            print(json.dumps({'fitted_snapshots': i, 'total': samples}), flush=True)
    scales = {f'{lag}h_{weighting}': {field: moments[lag, weighting, field].std().tolist()
                                   for field in FIELDS}
              for lag in (6, 24) for weighting in weights}
    return dict(version='ngcm_change_scale_alignment_audit_v1', split='train',
                train_years=list(range(2015, 2022)), snapshot_count=samples,
                selection='evenly spaced indices in eligible training origins, fixed before scoring',
                origins=[str(t) for t in origins], consumed_frame_sha256=consumed,
                prepared_manifest_sha256=sha(manifest_path), pressure_hpa=levels.tolist(),
                std_definition='population std across samples, levels, longitude and latitude; humidity per level',
                latitude_weighting='both uniform and Gaussian because G.3 does not specify quadrature',
                scale_floors=None, scales=scales)


def rescore(metrics, statistics, historical, summary):
    levels = np.asarray(statistics['pressure_hpa'], np.float64)
    recipes = {
        'current_v2': ('historical', 'pressure', True, False),
        '6h_pooled_same_sample': ('6h_gaussian', 'pressure', True, False),
        '24h_pooled_same_sample': ('24h_gaussian', 'pressure', True, False),
        '24h_uniform_stats_pressure_weights': ('24h_uniform', 'pressure', True, False),
        '24h_uniform_stats_equal_levels': ('24h_uniform', 'equal', False, True),
        '24h_gaussian_stats_equal_levels': ('24h_gaussian', 'equal', False, True),
    }
    all_rows, totals = [], []
    runs = sorted({row['run_id'] for row in metrics})
    for recipe, (scale_key, vertical, average_fields, time_scale) in recipes.items():
        for run in runs:
            contributions = {model: {} for model in ('baseline', 'residual')}
            top = 0.0
            for field in FIELDS:
                rows = [r for r in metrics if r['run_id'] == run and r['field'] == field]
                rows.sort(key=lambda r: float(r['pressure_hpa']))
                np.testing.assert_array_equal([float(r['pressure_hpa']) for r in rows], levels)
                if scale_key == 'historical':
                    sigma = np.asarray(historical['loss_scales'][field], np.float64)
                    if field != 'specific_humidity':
                        sigma = np.full_like(sigma, np.sqrt(np.mean(sigma ** 2)))
                    effective = (sigma / AMPLITUDES[field]).astype(np.float32).astype(np.float64)
                else:
                    sigma = np.asarray(statistics['scales'][scale_key][field])
                    effective = sigma / AMPLITUDES[field]
                level_weight = levels / levels.sum() if vertical == 'pressure' else np.ones_like(levels) / len(levels)
                factor = (1 / 7 if average_fields else 1) * (1 / (1 + 6 / 24) if time_scale else 1)
                scores = {}
                for model in contributions:
                    mse = np.array([float(r[f'{model}_rmse']) ** 2 for r in rows])
                    scores[model] = mse / effective ** 2 * level_weight * factor
                    contributions[model][field] = float(scores[model].sum())
                if field == 'geopotential':
                    top = float(scores['baseline'][levels <= 7].sum())
                for k, pressure in enumerate(levels):
                    all_rows.append(dict(recipe=recipe, run_id=run, field=field, pressure_hpa=float(pressure),
                                         baseline_contribution=float(scores['baseline'][k]),
                                         residual_contribution=float(scores['residual'][k])))
            b, c = [sum(contributions[m].values()) for m in ('baseline', 'residual')]
            if recipe == 'current_v2':
                np.testing.assert_allclose([b, c], [summary['runs'][run]['baseline_loss'],
                                                   summary['runs'][run]['residual_loss']], rtol=2e-4)
            totals.append(dict(recipe=recipe, run_id=run, baseline_score=b, residual_score=c,
                               improvement_pct=100 * (1 - c / b),
                               baseline_geopotential_share_pct=100 * contributions['baseline']['geopotential'] / b,
                               baseline_geo_1_7_share_pct=100 * top / b, fields=contributions))
    return recipes, totals, all_rows


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--physical-results', type=Path, default=Path('plot/k1_physical_eval_20260924'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=60)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new audit output directory')
    plan = json.loads(args.plan.read_text())
    manifest = json.loads((Path(plan['experiment_root']) / 'manifest.json').read_text())
    resource = manifest['resources']['res2p8']
    historical_path = Path(resource['statistics'])
    historical = json.loads(historical_path.read_text())
    summary_path = args.physical_results / 'summary.json'
    summary = json.loads(summary_path.read_text())
    metric_path = args.physical_results / 'physical_metrics_all_levels.csv'
    with metric_path.open() as f:
        metrics = list(csv.DictReader(f))
    if summary['runs'].keys() != plan['runs'].keys():
        raise ValueError('Physical results do not match the requested checkpoint set')
    for run, report in summary['runs'].items():
        if report['checkpoint'] != plan['runs'][run] or report['records'] != 1455:
            raise ValueError('Physical results do not match the fixed full-validation audit')
    statistics = fit_scales(Path(resource['prepared']), samples=args.samples)
    recipes, totals, rows = rescore(metrics, statistics, historical, summary)
    args.output.mkdir(parents=True)
    (args.output / 'statistics.json').write_text(json.dumps(statistics, indent=2, allow_nan=False) + '\n')
    write_csv(args.output / 'contributions.csv', rows)
    report = dict(version='ngcm_simplified_mse_alignment_audit_v1',
                  captured_at_utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__),
                  plan_sha256=sha(args.plan), physical_metrics_sha256=sha(metric_path),
                  physical_summary_sha256=sha(summary_path), historical_statistics_sha256=sha(historical_path),
                  statistics_sha256=sha(args.output / 'statistics.json'),
                  contributions_sha256=sha(args.output / 'contributions.csv'),
                  forecast_lead_hours=6, evaluation_split='2022 validation', records_per_run=1455,
                  recipes={k: dict(scale=v[0], level_reduction=v[1], average_fields=v[2],
                                   apply_paper_lead_time_scaling=v[3]) for k, v in recipes.items()},
                  omitted_terms=['spatial filtering', 'model-space loss', 'spectral loss', 'bias loss'],
                  omitted_global_multiplier='lambda_data=20; multiplying the only retained term does not change ratios or shares',
                  caveats=['Reweights existing K=1 forecasts; no retraining and no K=2 forecast.',
                           'Uniform levels follow the default official reference reducer; exact paper loss bindings and masks are not supplied here.',
                           'Statistical snapshot dates and training years differ from the paper.',
                           'Uniform/Gaussian statistics sensitivity is reported; G.3 does not specify latitude quadrature.',
                           'No training-only statistics here are promoted to production automatically.'],
                  totals=totals, current_training_modified=False)
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for row in totals:
        if row['run_id'] == 'r2p8_w128_di16':
            print(json.dumps({k: v for k, v in row.items() if k != 'fields'}), flush=True)


if __name__ == '__main__':
    main()
