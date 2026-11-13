#!/usr/bin/env python3
"""Publish the complete scalar sweep once all model smoothing reports exist.

Accuracy intervals resample paired initialization blocks. Spectral coordinates
are descriptive pooled-power scores; no spectral confidence band is implied.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import fcntl
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_models.merge_v24_2023_eval import cadence_block_counts, loss_improvement
from scripts.experiments.run_v24_mgda import check, pinned, read, seal, unseal, write
from scripts.analyze_models.comparison_policy import POLICY, partition_models, presentation_directory


def pooled_power_score(prediction, truth):
    """One macro score per variant; truth-positive mask common across methods."""
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    valid = truth > 0
    if not np.any(valid):
        return np.full(prediction.shape[1], np.nan)
    ratio = np.divide(prediction, truth[:, None], out=np.full_like(prediction, np.nan),
                      where=valid[:, None])
    return np.sum(np.where(valid[:, None], np.abs(ratio - 1), 0), axis=(0, 2, 3)) / np.sum(valid)


def build_if_complete(training_campaign):
    root = Path(training_campaign).resolve()
    selected_path = root / 'selected_arms.json'
    if not selected_path.exists():
        return {'status': 'pending', 'reason': 'both methods must finish training'}
    selected = unseal(selected_path)
    tasks, excluded = partition_models(selected['tasks'], root)
    destination = presentation_directory(root / 'report', excluded)
    destination.mkdir(exist_ok=True)
    with (destination / 'report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (destination / 'complete.json').exists():
            return unseal(destination / 'complete.json')
        if len(selected['tasks']) != 10 or len({t['logical_id'] for t in selected['tasks']}) != 10:
            raise ValueError('Expected ten distinct selected scalar arms')
        expected = [(task, root / 'smoothing' / task['logical_id']) for task in tasks]
        missing = [task['logical_id'] for task, directory in expected if not (directory / 'report.json').exists()]
        reference_path = root / 'reference_smoothing.json'
        if reference_path.exists():
            reference = Path(read(reference_path)['campaign'])
            if not (reference / 'report.json').exists():
                missing.append('M1 reference smoothing')
            else:
                expected.insert(0, ({'logical_id': 'M1', 'model_family': 'historical_gc_mamba', 'lambda': None}, reference))
        if missing:
            write(destination / 'pending.json', {'status': 'pending', 'missing_reports': missing})
            return {'status': 'pending', 'missing_reports': missing}
        return _build(root, destination, selected, expected)


def _build(root, destination, selected, expected, *, extra_metadata=None):
    included, excluded = partition_models([task for task, _ in expected], root)
    included_ids = {task['logical_id'] for task in included}
    expected = [(task, directory) for task, directory in expected if task['logical_id'] in included_ids]
    # The caller may already have filtered expected while checking readiness.
    _, selected_exclusions = partition_models(selected['tasks'], root)
    excluded = selected_exclusions or excluded
    destination = presentation_directory(destination, excluded)
    destination.mkdir(parents=True, exist_ok=True)
    records, losses, sources, warnings = [], [], [], []
    canonical, initialization_ids, canonical_power, canonical_layout = None, None, None, None
    variants = ('raw', 'gaussian', 'band')
    for task, directory in expected:
        report = unseal(directory / 'report.json')
        check(report['pooled_statistics'])
        sources.append(pinned(directory / 'report.json'))
        with np.load(report['pooled_statistics']['path'], allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata'].item()))
            ids = metadata['initialization_ids']
            values = saved['per_initialization_exact_gc_loss']
            if values.shape != (122, 40, 2, 3) or not np.all(np.isfinite(values)):
                raise ValueError('Incomplete or nonfinite scalar-sweep losses')
            truth = saved['truth_power'][..., 1:]
            powers = saved['prediction_power'][..., 1:]
            if canonical is None:
                canonical = values[:, :, 0, 0].copy()
                initialization_ids = ids
                canonical_power = truth.copy()
                canonical_layout = metadata['layout']
                gc_scores = pooled_power_score(powers[:, 0], truth)
                for variant, name in enumerate(variants):
                    losses.append(values[:, :, 0, variant].copy())
                    records.append({'model_id': 'GC', 'model_family': 'public_gc', 'lambda': None,
                                    'variant': name, 'spherical_power_error': float(gc_scores[variant])})
            elif (ids != initialization_ids or truth.shape != canonical_power.shape
                  or metadata['layout'] != canonical_layout):
                raise ValueError('Scalar sweep cohort or channel layout differs')
            else:
                delta = float(np.max(np.abs(values[:, :, 0, 0] - canonical)))
                bound = 1e-6 + 1e-6 * np.abs(canonical)
                exceeds = bool(np.any(np.abs(values[:, :, 0, 0] - canonical) > bound))
                truth_delta = float(np.max(np.abs(truth - canonical_power)))
                warnings.append({'model_id': task['logical_id'], 'baseline_max_abs_drift': delta,
                                 'baseline_reference_atol': 1e-6, 'baseline_reference_rtol': 1e-6,
                                 'exceeds_reference': exceeds, 'truth_power_max_abs_drift': truth_delta,
                                 'policy': 'warning; one canonical GC denominator used'})
            scores = pooled_power_score(powers[:, 1], canonical_power)
            for variant, name in enumerate(variants):
                losses.append(values[:, :, 1, variant].copy())
                records.append({'model_id': task['logical_id'], 'model_family': task['model_family'],
                                'lambda': task['lambda'], 'variant': name,
                                'spherical_power_error': float(scores[variant])})
    full = np.stack(losses)
    times = [datetime.strptime(value, '%Y%m%dT%H%M%SZ').isoformat() + 'Z' for value in initialization_ids]
    counts = cadence_block_counts(times, stride_days=3, block_starts=5, replicates=10000,
                                  seed=20260916, segmentation='annual')
    point = loss_improvement(canonical, full, np.ones((1, len(times))))[0]
    draws = loss_improvement(canonical, full, counts)
    lower, upper = np.quantile(draws, [.025, .975], axis=0)
    for i, row in enumerate(records):
        row.update(exact_gc_rollout_loss=float(full[i].mean()), exact_gc_day10_loss=float(full[i, :, -1].mean()),
                   improvement_pct=float(point[i, 0]), ci95_lower=float(lower[i, 0]), ci95_upper=float(upper[i, 0]),
                   day10_improvement_pct=float(point[i, -1]), day10_ci95_lower=float(lower[i, -1]),
                   day10_ci95_upper=float(upper[i, -1]))
        if not np.isfinite(row['spherical_power_error']):
            row['spherical_power_error'] = None
        task = next((t for t in selected['tasks'] if t['logical_id'] == row['model_id']), None)
        if task is None:
            row['training_stage_elapsed_gpu_hours'] = None
        else:
            proof_root = Path(task.get('training_proof_root', root / 'checks'))
            proof_pattern = task.get('training_proof_pattern', f"{task['id']}_step*.json")
            proof_paths = sorted(proof_root.glob(proof_pattern))
            proofs = [unseal(p) for p in proof_paths]
            sources.extend(pinned(p) for p in proof_paths)
            elapsed_field = task.get('training_elapsed_field', 'elapsed_seconds')
            row['training_stage_elapsed_gpu_hours'] = (
                sum(p[elapsed_field] for p in proofs) / 3600 if proofs else None)
    contrasts = paired_contrasts(records, point, draws)
    with (destination / 'tradeoff.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(destination / 'paired_accuracy.npz', baseline=canonical, full=full,
                        improvement_pct=point, lower=lower, upper=upper,
                        initialization_times=np.asarray(times))
    _plot(destination, records)
    result = seal(destination / 'complete.json', {'status': 'complete', 'sources': sources,
        'selected_arms': pinned(root / 'selected_arms.json'), 'records': records, 'contrasts': contrasts,
        'comparison_policy': POLICY if excluded else None, 'excluded_models': excluded,
        'baseline_repeatability': warnings, 'evaluation_year_reused': True,
        'bootstrap': {'replicates': 10000, 'seed': 20260916, 'block_starts': 5,
                      'method': 'paired non-wrapping moving-block percentile', 'coverage': 'pointwise exact MSE'},
        'spectral_uncertainty': 'descriptive pooled-power coordinates; per-date sufficient statistics retained for nonlinear resampling',
        'training_seed': 22, 'scientific_improvement_required': False,
        **(extra_metadata or {})})
    (destination / 'pending.json').unlink(missing_ok=True)
    return result


def paired_contrasts(records, point, draws):
    """Compare only eligible same-lambda pairs; an excluded arm is not missing data."""
    contrasts = []
    variants = ('raw', 'gaussian', 'band')
    for weight in (0., .25, .5, .75, 1.):
        for variant in variants:
            indices = [next(i for i, row in enumerate(records) if row['model_family'] == family
                            and row['lambda'] == weight and row['variant'] == variant)
                       for family in ('gc_mamba', 'full_gc')
                       if any(row['model_family'] == family and row['lambda'] == weight
                              and row['variant'] == variant for row in records)]
            if len(indices) != 2:
                continue
            a, b = indices
            difference = draws[:, a, 0] - draws[:, b, 0]
            lo, hi = np.quantile(difference, [.025, .975])
            contrasts.append({'lambda': weight, 'variant': variant,
                              'mamba_advantage_percentage_points': float(point[a, 0] - point[b, 0]),
                              'ci95_lower': float(lo), 'ci95_upper': float(hi)})
    return contrasts


def _plot(destination, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for axis, variant in zip(axes, ('raw', 'gaussian', 'band')):
        for family, color, marker in (('full_gc', 'C0', 'o'), ('gc_mamba', 'C1', 's')):
            chosen = sorted((r for r in records if r['variant'] == variant and r['model_family'] == family),
                            key=lambda r: r['lambda'])
            axis.plot([r['improvement_pct'] for r in chosen],
                      [100 * r['spherical_power_error'] if r['spherical_power_error'] is not None else np.nan for r in chosen],
                      color=color, marker=marker, label=family)
            for row in chosen:
                if row['spherical_power_error'] is not None:
                    axis.annotate(str(row['lambda']), (row['improvement_pct'], 100 * row['spherical_power_error']),
                                  xytext=(4, 4), textcoords='offset points', fontsize=8)
        for label, marker in (('GC', 'x'), ('M1', '*')):
            for row in records:
                if row['variant'] == variant and row['model_id'] == label and row['spherical_power_error'] is not None:
                    axis.scatter(row['improvement_pct'], 100 * row['spherical_power_error'], marker=marker,
                                 s=70, color='black' if label == 'GC' else 'C2', label=label)
        axis.set(title=variant, xlabel='Exact GC MSE reduction (%)', ylabel='Mean relative spherical power error (%)')
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.savefig(destination / 'lambda_tradeoff.png', dpi=180)
    fig.savefig(destination / 'lambda_tradeoff.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('training_campaign', type=Path)
    print(json.dumps(build_if_complete(parser.parse_args().training_campaign), indent=2, allow_nan=False))
