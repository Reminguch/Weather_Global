#!/usr/bin/env python3
"""Native exact-loss evaluation and paired reporting for fresh data-scale arms."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_operations.loaders.zarr_metadata import (
    frozen_zarr_metadata, verify_frozen_zarr_metadata,
)

EXACT32_INDICES = [4, 23, 47, 57, 104, 241, 246, 369, 384, 422, 548, 693, 699,
                   751, 771, 774, 836, 870, 873, 893, 930, 1006, 1017, 1065,
                   1117, 1130, 1161, 1181, 1191, 1256, 1294, 1338]
DEFAULT_ARMS = {(128, 7): 'w128_7y', (128, 40): 'w128_40y',
                (512, 7): 'w512_7y', (512, 40): 'w512_40y'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def pinned(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest}


def freeze_inputs(config_path, checkpoint, source, data_path, year, initialization_manifest=None):
    data = source['data']
    identity = {'config': pinned(config_path), 'checkpoint': pinned(checkpoint),
        'baseline': pinned(data['baseline_checkpoint']),
        'statistics': [pinned(Path(data['stats_dir']) / name) for name in
            ('mean_by_level.nc', 'stddev_by_level.nc', 'diffs_stddev_by_level.nc')],
        'data_metadata': frozen_zarr_metadata(data_path), 'year': year}
    if initialization_manifest is not None:
        identity['initializations'] = pinned(initialization_manifest)
    return identity


def verify_inputs(identity, data_path):
    for record in [identity['config'], identity['checkpoint'], identity['baseline'],
                   *identity['statistics'], *([identity['initializations']] if 'initializations' in identity else [])]:
        if pinned(record['path']) != record:
            raise ValueError(f"Input changed during evaluation: {record['path']}")
    # Re-select from the store: a newly added zarr.json supersedes .zmetadata.
    verify_frozen_zarr_metadata(data_path, identity['data_metadata'])


def manifest_times(path):
    from datetime import datetime, timezone
    body = read(path)
    if body.get('schema_version') != 1 or not body.get('initializations'):
        raise ValueError('Invalid initialization manifest')
    values = [v['initialization_time'] if isinstance(v, dict) else v
              for v in body['initializations']]
    dates = [datetime.fromisoformat(v.replace('Z', '+00:00')) for v in values]
    if (any(v.tzinfo is None or v.utcoffset().total_seconds() != 0 for v in dates)
            or dates != sorted(set(dates))):
        raise ValueError('Initializations must be unique, ordered UTC dates')
    return [v.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z') for v in dates]


def loss_arrays(baseline, full, shape):
    b, f = np.asarray(baseline, dtype=float), np.asarray(full, dtype=float)
    if (b.shape != shape or f.shape != shape or not np.isfinite([b, f]).all()
            or not (b > 0).all() or not (f >= 0).all()):
        raise ValueError('Invalid or nonfinite exact loss arrays')
    return b, f


def validate_result(result, *, checkpoint, baseline_checkpoint, year,
                    initialization_manifest=None, index=0, shards=1,
                    chosen_idx=EXACT32_INDICES):
    """Check completeness, forecast identity, and exact aggregation."""
    if (Path(result['ckpt']).resolve() != Path(checkpoint).resolve()
            or Path(result['ckpt_in']).resolve() != Path(baseline_checkpoint).resolve()):
        raise ValueError('Checkpoint identity mismatch')
    if (result.get('evaluation_status') != 'complete' or result.get('target_steps') != 40
            or result.get('eval_mode') != 'cold_full' or result.get('eval_feedback') != 'full'
            or result.get('residual_state_init') != 'zero'
            or result.get('rs_reset_every_step', False)
            or result.get('anchor_shard_index', 0) != index
            or result.get('anchor_shard_count', 1) != shards):
        raise ValueError('Evaluation protocol or completeness mismatch')
    g = result['original_graphcast_loss']
    b, f = loss_arrays(g['baseline_per_step'], g['full_per_step'], (40,))
    if not np.allclose([g['baseline_rollout'], g['full_rollout'], g['improvement_pct_rollout']],
                       [b.mean(), f.mean(), 100 * (1 - f.sum() / b.sum())], rtol=1e-6, atol=1e-6):
        raise ValueError('Exact loss aggregation mismatch')
    if initialization_manifest is None:
        if year != 2022 or shards != 1 or result['evaluated_samples'] != 32:
            raise ValueError('Legacy evaluation must use matched32 development starts')
        if list(result['chosen_idx']) != list(chosen_idx):
            raise ValueError('Legacy matched32 origins changed')
    else:
        expected = manifest_times(initialization_manifest)
        if year != 2023 or len(expected) != 122 or any(not v.startswith('2023-') for v in expected):
            raise ValueError('Expected the 122-start 2023 protocol')
        records = result.get('original_graphcast_loss_per_initialization', [])
        selected = expected[index::shards]
        if (result.get('initialization_manifest_sha256') != pinned(initialization_manifest)['sha256']
                or result.get('global_initialization_times') != expected
                or [r['initialization_time'] for r in records] != selected
                or result.get('evaluated_samples') != len(selected)
                or result.get('n_samples') != 122):
            raise ValueError('Manifest identity, shard coverage, or record order mismatch')
        rb, rf = loss_arrays([r['baseline_per_step'] for r in records],
                             [r['full_per_step'] for r in records], (len(selected), 40))
        if not np.allclose([rb.mean(axis=0), rf.mean(axis=0)], [b, f], rtol=1e-6, atol=1e-6):
            raise ValueError('Per-initialization losses do not reproduce pooled losses')
    return {'passed': True, 'evaluated_samples': result['evaluated_samples']}


def evaluate(config_path, checkpoint, out, *, year, data_path,
             initialization_manifest=None, index=0, shards=1,
             chosen_idx=EXACT32_INDICES, model_id=None):
    """GPU entrypoint. Freeze all four finals in the controller before 2023."""
    from src.models.mamba.v24_Ilya.config import V24IlyaEvalConfig
    from src.models.mamba.v24_Ilya.evaluation import evaluate_v24_Ilya
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    if year not in (2022, 2023) or (year == 2023) != (initialization_manifest is not None):
        raise ValueError('2022 uses legacy matched32; 2023 requires the frozen date manifest')
    source = read(config_path)
    data = source['data']
    identity = freeze_inputs(config_path, checkpoint, source, data_path, year, initialization_manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    unpublished = out.with_name(out.stem + '.unpublished.json')
    cfg = V24IlyaEvalConfig(ckpt=Path(checkpoint).resolve(), out_json=unpublished,
        ckpt_in=Path(data['baseline_checkpoint']).resolve(), stats_dir=Path(data['stats_dir']),
        data_path=str(Path(data_path).resolve()), val_year=year,
        train_start_year=2015, train_end_year=2021, input_duration='12h',
        eval_mode='cold_full', residual_state_init='zero', residual_alpha=1.,
        target_steps=40, warmup_steps=24 if year == 2022 else 0,
        n_samples=32 if year == 2022 else 122, seed=0, omit_rms_bias=True,
        initialization_manifest=Path(initialization_manifest) if initialization_manifest else None,
        anchor_shard_index=index, anchor_shard_count=shards, model_id=model_id,
        **source['architecture'])
    result = evaluate_v24_Ilya(cfg)
    proof = validate_result(result, checkpoint=checkpoint,
        baseline_checkpoint=data['baseline_checkpoint'], year=year,
        initialization_manifest=initialization_manifest, index=index, shards=shards,
        chosen_idx=chosen_idx)
    # Detect mutable checkpoint/config inputs before publishing the completed artifact.
    verify_inputs(identity, data_path)
    result.update(fresh_data_scale_identity=identity, integrity=proof)
    write(out, result)
    unpublished.unlink(missing_ok=True)
    return result


def merge(paths, expected_initmanifest, output=None, *, expected_checkpoint=None):
    """Require every shard and pool per-start losses, not averages of shard ratios."""
    paths = list(map(Path, paths))
    if not paths:
        raise ValueError('No evaluation shards')
    results = [read(p) for p in paths]
    first = results[0]
    identity = first['fresh_data_scale_identity']
    if expected_checkpoint is not None and pinned(expected_checkpoint) != identity['checkpoint']:
        raise ValueError('Unexpected final checkpoint')
    count = first['anchor_shard_count']
    if len(results) != count or sorted(r['anchor_shard_index'] for r in results) != list(range(count)):
        raise ValueError('Missing or duplicated evaluation shards')
    records = []
    for result in results:
        if result['fresh_data_scale_identity'] != identity or result.get('integrity', {}).get('passed') is not True:
            raise ValueError('Shard input identity or integrity proof differs')
        validate_result(result, checkpoint=identity['checkpoint']['path'],
            baseline_checkpoint=identity['baseline']['path'], year=2023,
            initialization_manifest=expected_initmanifest, index=result['anchor_shard_index'], shards=count)
        records.extend(result['original_graphcast_loss_per_initialization'])
    records.sort(key=lambda r: r['initialization_time'])
    times = manifest_times(expected_initmanifest)
    if [r['initialization_time'] for r in records] != times:
        raise ValueError('Incomplete or overlapping 122-start coverage')
    b, f = loss_arrays([r['baseline_per_step'] for r in records],
                       [r['full_per_step'] for r in records], (122, 40))
    report = {'status': 'complete', 'identity': identity, 'initialization_times': times,
        'baseline_loss_matrix': b.tolist(), 'full_loss_matrix': f.tolist(),
        'baseline_rollout': float(b.mean()), 'full_rollout': float(f.mean()),
        'loss_reduction_pct_own_baseline': float(100 * (1 - f.sum() / b.sum())),
        'shards': [pinned(p) for p in paths]}
    if output is not None:
        write(output, report)
    return report


def comparison_report(arm_reports, outdir, *, canonical_arm='w128_7y', arm_keys=None,
                      bootstrap_samples=10000, block_starts=5, seed=20260916):
    """Paired fixed-checkpoint inference; two primary width-specific data gains."""
    keys = DEFAULT_ARMS if arm_keys is None else arm_keys
    years = sorted({y for _, y in keys})
    if (len(keys) != 4 or len(years) != 2 or years[0] != 7 or years[1] <= 7
            or set(keys) != {(w, y) for w in (128, 512) for y in years}
            or len(set(keys.values())) != 4):
        raise ValueError('Exactly four width-by-data arms are required')
    large_years = years[1]
    reports = {key: read(v) if isinstance(v, (str, Path)) else v for key, v in arm_reports.items()}
    if set(reports) != set(keys.values()) or canonical_arm != keys[128, 7]:
        raise ValueError('Canonical baseline must be the evaluated width128 seven-year control')
    reference = reports[canonical_arm]
    n = len(reference['initialization_times'])
    if n != 122 or len(set(reference['initialization_times'])) != n:
        raise ValueError('Incomplete comparison dates')
    baseline, _ = loss_arrays(reference['baseline_loss_matrix'], reference['full_loss_matrix'], (n, 40))
    models, rows, drift = {}, {}, {}
    for key, report in reports.items():
        if report.get('status') != 'complete' or report['initialization_times'] != reference['initialization_times']:
            raise ValueError('Unmatched or incomplete arm')
        for field in ('baseline', 'statistics', 'data_metadata', 'initializations'):
            if report['identity'][field] != reference['identity'][field]:
                raise ValueError(f'Mismatched evaluation input: {field}')
        own, full = loss_arrays(report['baseline_loss_matrix'], report['full_loss_matrix'], (n, 40))
        models[key] = full
        rows[key] = {'full_rollout': float(full.mean()),
            'loss_reduction_pct': float(100 * (1 - full.sum() / baseline.sum())),
            'day10_loss_reduction_pct': float(100 * (1 - full[:, -1].sum() / baseline[:, -1].sum())),
            'checkpoint': report['identity']['checkpoint']}
        drift[key] = {'pooled_pct': float(100 * (own.sum() / baseline.sum() - 1)),
            'max_start_lead_pct': float(100 * np.max(np.abs(own / baseline - 1))),
            'outside_rtol_0p001_count': int(np.count_nonzero(np.abs(own - baseline) > 1e-6 + .001 * np.abs(baseline)))}
    if bootstrap_samples < 100 or not 1 <= block_starts <= n:
        raise ValueError('Invalid bootstrap specification')
    differences = np.stack([models[keys[w, 7]] - models[keys[w, large_years]] for w in (128, 512)])
    point = 100 * differences.sum(axis=(1, 2)) / baseline.sum()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n - block_starts + 1, (bootstrap_samples, math.ceil(n / block_starts)))
    indices = (draws[..., None] + np.arange(block_starts)).reshape(bootstrap_samples, -1)[:, :n]
    # Reduce each initialization first to keep 10k bootstrap draws memory bounded.
    draws_effect = 100 * differences.sum(axis=2)[:, indices].sum(axis=2).T / baseline.sum(axis=1)[indices].sum(axis=1)[:, None]
    radius = float(np.quantile(np.max(np.abs(draws_effect - point), axis=1), .95))
    primary = {str(w): {'gain_percentage_points': float(point[i]),
        'simultaneous_95_interval': [float(point[i] - radius), float(point[i] + radius)],
        'positive_gain_established': bool(point[i] - radius > 0)} for i, w in enumerate((128, 512))}
    interaction = draws_effect[:, 1] - draws_effect[:, 0]
    result = {'status': 'complete', 'canonical_baseline_arm': canonical_arm, 'arms': rows,
        'primary_data_gain': primary,
        'width512_minus_width128_data_gain': {'gain_percentage_points': float(point[1] - point[0]),
            'descriptive_95_interval': np.quantile(interaction, [.025, .975]).tolist()},
        'baseline_drift_advisory': drift,
        'bootstrap': {'samples': bootstrap_samples, 'block_starts': block_starts, 'seed': seed,
            'scheme': 'paired non-wrapping moving blocks; truncated final block',
            'primary_family': 'simultaneous 95% maximum absolute centered deviation; two within-width gains'},
        'training_year_counts': years,
        'interpretation': f'Fixed checkpoint/seed, weather-sample uncertainty only. 2023 has prior project exposure. Positive gain is lower loss with {large_years} training years. Day10 and width interaction are descriptive.',
        'initialization_times': reference['initialization_times']}
    outdir = Path(outdir)
    write(outdir / 'comparison.json', result)
    lines = ['# Fresh residual data-scale comparison', '',
        'Exact original GraphCast loss, pooled over 122 starts and 40 leads. Canonical baseline: matched evaluation of the width128 seven-year control.', '',
        '| Arm | Rollout loss reduction (%) | Day10 reduction (%) |', '| --- | ---: | ---: |']
    lines += [f"| {key} | {r['loss_reduction_pct']:.4f} | {r['day10_loss_reduction_pct']:.4f} |" for key, r in rows.items()]
    lines += ['', f'Primary gains from {large_years} versus seven years (percentage points; simultaneous 95% intervals):', '']
    for width, row in primary.items():
        low, high = row['simultaneous_95_interval']
        lines.append(f"- Width{width}: {row['gain_percentage_points']:.4f} [{low:.4f}, {high:.4f}].")
    lines += ['', result['interpretation'], '', 'Repeated baseline drift is advisory and retained in comparison.json. No averaging of per-lead or per-shard improvement percentages.', '']
    (outdir / 'README.md').write_text('\n'.join(lines))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--year', type=int, choices=(2022, 2023), required=True)
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--initialization-manifest', type=Path)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--model-id')
    args = vars(parser.parse_args())
    args['config_path'] = args.pop('config')
    evaluate(**args)


if __name__ == '__main__':
    main()
