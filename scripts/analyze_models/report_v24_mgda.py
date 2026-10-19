"""Report paired MGDA continuation without selecting checkpoints or recipes.

Consumes the frozen arbitrary-checkpoint manifest used by the AMSE evaluator.
Historical AMSE qualification rules remain unchanged; this report adds the two
prespecified MGDA comparisons and plots saved observations only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_models import evaluate_v24_amse_regime as evaluation


def ratio(value, reference):
    """Do not invent a percentage improvement when a reference has zero error."""
    if reference == 0:
        return 1.0 if value == 0 else None
    return float(value / reference)


def compare(candidate, reference, *, maximum_rmse_harm_pct=2.0):
    candidate_rows = evaluation.feature_rows(candidate)
    reference_rows = evaluation.feature_rows(reference)
    fields = []
    for left, right in zip(candidate_rows, reference_rows, strict=True):
        if (left['feature'], left['day']) != (right['feature'], right['day']):
            raise ValueError('Paired field/lead coordinates differ')
        fields.append({
            'feature': left['feature'], 'day': left['day'],
            'rmse_ratio': ratio(left['rmse'], right['rmse']),
            'power_error_ratio': ratio(left['mean_relative_power_error_pct'], right['mean_relative_power_error_pct']),
            'coherence_change': left['mean_coherence'] - right['mean_coherence'],
            'rmse_guard_passed': bool(left['rmse'] <= right['rmse'] * (1 + maximum_rmse_harm_pct / 100)),
        })
    loss = candidate['result']['original_graphcast_loss']['full_rollout']
    reference_loss = reference['result']['original_graphcast_loss']['full_rollout']
    power = float(np.mean([row['mean_relative_power_error_pct'] for row in candidate_rows]))
    reference_power = float(np.mean([row['mean_relative_power_error_pct'] for row in reference_rows]))
    harms = [row for row in fields if not row['rmse_guard_passed']]
    rmse_ratios = [row['rmse_ratio'] for row in fields]
    return {
        'exact_gc_loss_ratio': ratio(loss, reference_loss),
        'exact_gc_loss_difference': float(loss - reference_loss),
        'mean_power_error_ratio': ratio(power, reference_power),
        'mean_power_error_difference_pct': float(power - reference_power),
        'mean_rmse_ratio': None if None in rmse_ratios else float(np.mean(rmse_ratios)),
        'mean_coherence_change': float(np.mean([row['coherence_change'] for row in fields])),
        'maximum_rmse_harm_pct': maximum_rmse_harm_pct,
        'safety_guard_passed': not harms, 'safety_guard_failures': harms,
        'per_feature_lead': fields,
    }


def outcomes(comparisons, intervals):
    """Keep scientific outcomes distinct from numerical training completion."""
    initial, matched = (comparisons[key] for key in ('initialization', 'matched_control'))
    guard = initial['safety_guard_passed'] and matched['safety_guard_passed']
    both = initial['exact_gc_loss_difference'] < 0 and initial['mean_power_error_difference_pct'] < 0
    beats = matched['exact_gc_loss_difference'] <= 0 and matched['mean_power_error_difference_pct'] < 0

    def supported(reference, strict):
        interval = intervals[reference]
        if interval.get('status') != 'complete':
            return None
        bounds = interval['percentile95']
        gc_upper = bounds['exact_gc_loss_ratio'][1]
        return bool(bounds['mean_power_error_ratio'][1] < 1 and
                    (gc_upper < 1 if strict else gc_upper <= 1))

    return {
        'both_improved_vs_initialization': bool(both),
        'power_improved_gc_preserved_vs_matched_control': bool(beats),
        'both_reference_safety_guards_passed': bool(guard),
        'qualified_point_estimate': bool(both and beats and guard),
        'supported_vs_initialization_95': supported('initialization', True),
        'supported_vs_matched_control_95': supported('matched_control', False),
        'uncertainty_note': 'Intervals spanning no change are inconclusive. Point estimates and pointwise safety guards are reported separately; these are not simultaneous confidence bounds.',
    }


def build_report(manifest_path, *, image_dir=None, training_runs=None):
    manifest_path = Path(manifest_path)
    manifest = evaluation.load_manifest(manifest_path)
    required = ('spectral',) if manifest['evaluation_role'] == 'confirmation' else ('exact', 'spectral')
    loaded, sources = {}, []
    for index, task in enumerate(manifest['task_specs']):
        stages = {stage: evaluation.resolve_result(manifest_path, index, stage) for stage in required}
        directory, spec, result, reused = stages['spectral']
        spectra = evaluation.load_spectral_result(directory, spec, result)
        for output, _, _, _ in stages.values():
            sources.extend(evaluation.frozen_file(output/name) for name in ('specification.json', 'evaluation.json'))
        sources.extend(spectra['sources'])
        if manifest.get('observer_factory') and not reused:
            spherical = sorted((directory/'spherical').glob('*.npz'))
            evaluation.require(len(spherical) == len(spectra['times']), 'Incomplete spherical observer outputs')
            sources.extend(evaluation.frozen_file(path) for path in spherical)
        loaded[task['id']] = {'task': task, 'spectra': spectra, 'stages': stages}

    initial_id = manifest['canonical_mse_id']
    initial = loaded[initial_id]['spectra']
    records = []
    for identity, item in loaded.items():
        task, candidate = item['task'], item['spectra']
        evaluation.require(identity == initial_id or 'paired_mse_id' in task,
                           f'Explicit matched MSE checkpoint required: {identity}')
        matched_id = task.get('paired_mse_id', initial_id)
        evaluation.require(matched_id in loaded, f'Missing matched MSE control: {matched_id}')
        np.testing.assert_array_equal(candidate['stacked']['truth_power'], initial['stacked']['truth_power'])
        comparisons, intervals = {}, {}
        for name, reference_id in (('initialization', initial_id), ('matched_control', matched_id)):
            reference = loaded[reference_id]['spectra']
            evaluation.require(candidate['times'] == reference['times'], 'Paired initialization dates differ')
            np.testing.assert_array_equal(candidate['stacked']['truth_power'], reference['stacked']['truth_power'])
            comparisons[name] = {'reference_id': reference_id, **compare(candidate, reference)}
            intervals[name] = evaluation.bootstrap_joint(candidate, reference, 'full', manifest['bootstrap_grouping'], manifest['selection_policy'])
        loss = candidate['result']['original_graphcast_loss']
        gc_loss = initial['result']['original_graphcast_loss']['baseline_rollout']
        original_context = evaluation.score_candidate(
            evaluation.feature_rows(candidate), evaluation.feature_rows(initial, 'baseline'),
            evaluation.feature_rows(loaded[matched_id]['spectra']), loss['full_rollout']/gc_loss,
            manifest['selection_policy'])
        records.append({**task, 'paired_mse_id': matched_id,
                        'comparisons': comparisons, 'intervals': intervals,
                        'outcomes': outcomes(comparisons, intervals),
                        'original_gc_context': original_context,
                        'exact32_loss': (item['stages']['exact'][2]['original_graphcast_loss'] if 'exact' in item['stages'] else None)})

    report = {
        'schema_version': 1, 'format': 'v24_mgda_report_v1', 'status': 'complete',
        'manifest': evaluation.frozen_file(manifest_path),
        'evaluation_role': manifest['evaluation_role'], 'initialization_id': initial_id,
        'primary_endpoint': 'swa_step01100-02000', 'secondary_endpoint': 'step02000',
        'records': records, 'sources': sources, 'finalists': [],
        'selection_allowed': False,
        'scope': 'Exact GC loss averages all 40 leads; physical RMSE and zonal power/coherence use nine fields at days 1, 3, 7, 10 on the same starts.',
        'power_definition': 'Pool raw powers over starts before mean absolute relative error across positive wavenumbers. Reported error is unsmoothed and unfloored; the per-example training surrogate differs.',
        'interpretation': 'Development comparisons are exploratory. Late SWA is fixed in advance; no checkpoint or recipe is selected by this report. Confirmation intervals are paired seasonal calendar-block intervals, not independent-date significance tests.',
    }
    destination = Path(manifest['output_dir'])/'report'
    evaluation.write_json(destination/'mgda_report.json', report)
    lines = ['# MGDA continuation results', '', report['scope'], '', report['interpretation'], '',
             '| Checkpoint | GC ratio vs initialization | Power ratio vs initialization | GC ratio vs matched MSE | Power ratio vs matched MSE | Both-reference guard |',
             '|---|---:|---:|---:|---:|---|']
    for row in records:
        a, b = (row['comparisons'][key] for key in ('initialization', 'matched_control'))
        fmt = lambda value: 'undefined' if value is None else f'{value:.5f}'
        lines.append(f"| {row.get('label', row['id'])} ({row['id']}) | {fmt(a['exact_gc_loss_ratio'])} | {fmt(a['mean_power_error_ratio'])} | {fmt(b['exact_gc_loss_ratio'])} | {fmt(b['mean_power_error_ratio'])} | {row['outcomes']['both_reference_safety_guards_passed']} |")
    lines += ['', 'Ratios below one favor the checkpoint. Point estimates, paired intervals, per-field safeguards and coherence are retained in `mgda_report.json`.', '', report['power_definition'], '']
    (destination/'README.md').write_text('\n'.join(lines))
    if image_dir is not None:
        render(destination/'mgda_report.json', image_dir, training_runs=training_runs)
    return report


def render(report_path, image_dir, *, training_runs=None):
    from scripts.analyze_models import plot_v24_amse_regime as plotting
    import matplotlib.pyplot as plt

    report_path, output = Path(report_path), Path(image_dir)
    report = json.loads(report_path.read_text())
    records = report['records']
    if not records:
        raise ValueError('No MGDA records to plot')
    output.mkdir(parents=True, exist_ok=True)
    plotting.style()
    manifest = {'source': plotting.identity(report_path), 'figures': [], 'no_forecast_averaging': True}
    labels = [plotting.record_label(row) for row in records]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axhline(0, color='gray', linewidth=1)
    ax.axvline(0, color='gray', linewidth=1)
    for index, row in enumerate(records):
        comparison = row['comparisons']['initialization']
        x, y = comparison['mean_rmse_ratio'], comparison['mean_power_error_ratio']
        if x is None or y is None:
            continue
        ax.scatter(100*(1-x), 100*(1-y), s=50, label=labels[index])
        interval = row['intervals']['initialization']
        if interval.get('status') == 'complete':
            bounds = interval['percentile95']
            ax.plot([100*(1-v) for v in bounds['mean_rmse_ratio']], [100*(1-y)]*2, alpha=.4, color='gray')
            ax.plot([100*(1-x)]*2, [100*(1-v) for v in bounds['mean_power_error_ratio']], alpha=.4, color='gray')
    ax.set(xlabel='Mean physical RMSE reduction vs mature initialization (%)',
           ylabel='Mean relative power-error reduction vs mature initialization (%)',
           title='MGDA continuation: paired forecast accuracy and power error')
    ax.legend(fontsize=7, loc='best')
    fig.tight_layout()
    plotting.save(fig, 'joint_rmse_power', output, manifest, reference='mature initialization', intervals='paired 95% calendar-block')

    fig, ax = plt.subplots(figsize=(11, max(5, .5*len(records))))
    for offset, reference, label in ((-.17, 'initialization', 'Mature initialization'), (.17, 'matched_control', 'Matched MSE checkpoint')):
        values = [100*(1-row['comparisons'][reference]['exact_gc_loss_ratio']) for row in records]
        ax.barh(np.arange(len(records))+offset, values, height=.32, label=label)
        for index, row in enumerate(records):
            interval = row['intervals'][reference]
            if interval.get('status') == 'complete':
                bounds = [100*(1-value) for value in interval['percentile95']['exact_gc_loss_ratio']]
                ax.plot(bounds, [index+offset]*2, color='#222222', linewidth=1)
    ax.set_yticks(np.arange(len(records)), labels, fontsize=8)
    ax.axvline(0, color='gray', linewidth=1)
    ax.set(xlabel='Exact GraphCast loss reduction (%)', title='Exact GC loss: all 40 leads pooled before taking ratios')
    ax.invert_yaxis()
    ax.legend()
    fig.tight_layout()
    plotting.save(fig, 'exact_graphcast_loss', output, manifest, references=['initialization', 'matched_control'])

    definitions = (
        ('physical_rmse', 'initialization', 'rmse_ratio', 'Physical RMSE reduction vs mature initialization (%)', lambda v: 100*(1-v)),
        ('power_error', 'initialization', 'power_error_ratio', 'Power-error reduction vs mature initialization (%)', lambda v: 100*(1-v)),
        ('coherence', 'matched_control', 'coherence_change', 'Coherence change vs matched MSE checkpoint (percentage points)', lambda v: 100*v),
    )
    for name, reference, metric, title, transform in definitions:
        coordinates = [(row['feature'], row['day']) for row in records[0]['comparisons'][reference]['per_feature_lead']]
        values = []
        for record in records:
            fields = record['comparisons'][reference]['per_feature_lead']
            if [(row['feature'], row['day']) for row in fields] != coordinates:
                raise ValueError('Heatmap coordinates differ')
            values.append([np.nan if row[metric] is None else transform(row[metric]) for row in fields])
        values = np.asarray(values)
        bound = max(float(np.nanmax(np.abs(values))), .1)
        fig, ax = plt.subplots(figsize=(17, max(5, .45*len(records))))
        im = ax.imshow(values, aspect='auto', cmap='RdBu', vmin=-bound, vmax=bound)
        ax.set_yticks(np.arange(len(records)), labels, fontsize=8)
        ax.set_xticks(np.arange(len(coordinates)), [f'{feature}\n{day}d' for feature, day in coordinates], fontsize=7)
        ax.set_title(title + '\nPositive values favor the checkpoint')
        fig.colorbar(im, ax=ax, fraction=.025, pad=.015)
        fig.tight_layout()
        plotting.save(fig, f'feature_{name}', output, manifest, reference=reference, coordinates=coordinates)

    if training_runs:
        optimizer_diagnostics(training_runs, output, manifest, plotting)
    rows = []
    for record in records:
        for reference, comparison in record['comparisons'].items():
            for field in comparison['per_feature_lead']:
                rows.append({'id': record['id'], 'reference': reference, 'reference_id': comparison['reference_id'], **field})
    with (output/'feature_lead_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    evaluation.write_json(output/'figures_manifest.json', manifest)
    return manifest


def optimizer_diagnostics(training_runs, output, manifest, plotting):
    """Missing diagnostics remain missing, never fabricated as zero curves."""
    import matplotlib.pyplot as plt

    panels = (
        ('Raw objectives', ('gc_loss', 'power_loss')),
        ('Normalized objectives', ('gc_loss_normalized', 'power_loss_normalized')),
        ('MGDA coefficient and gradient cosine', ('mgda_alpha', 'gradient_cosine')),
        ('Objective gradient norms', ('gc_gradient_norm', 'power_gradient_norm')),
        ('Actual displacement descent products', ('gc_displacement_dot', 'power_displacement_dot')),
        ('Update norm and clipping', ('update_norm', 'clip_factor')),
        ('Spatial learning rate', ('learning_rate',)),
        ('Power-floor frequency and stalled updates', ('power_floor_fraction', 'stalled')),
    )
    fig, axes = plt.subplots(4, 2, figsize=(13, 13))
    sources, missing = [], {}
    for label, directory in training_runs.items():
        path = Path(directory)
        if path.is_dir():
            path = path/'train_metrics.jsonl'
        lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        steps = [row['step'] for row in lines]
        if steps != sorted(set(steps)):
            raise ValueError(f'Duplicate or unordered training steps: {path}')
        sources.append(plotting.identity(path))
        for ax, (title, keys) in zip(axes.flat, panels, strict=True):
            ax.set_title(title)
            ax.set_xlabel('Completed optimizer updates')
            for key in keys:
                available = [(row['step'], row[key]) for row in lines if key in row]
                if not available:
                    missing.setdefault(label, []).append(key)
                    continue
                points = np.asarray(available)
                if not np.isfinite(points).all():
                    raise ValueError(f'Nonfinite optimizer diagnostic {path}/{key}')
                ax.plot(points[:, 0], points[:, 1], label=f'{label}: {key}', linewidth=.8)
    for ax in axes.flat:
        if ax.lines:
            ax.legend(fontsize=6)
        else:
            ax.text(.5, .5, 'Diagnostic unavailable', transform=ax.transAxes, ha='center')
        ax.grid(alpha=.2)
    fig.tight_layout()
    plotting.save(fig, 'optimizer_diagnostics', output, manifest, sources=sources, missing_metrics=missing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--image-dir', type=Path)
    parser.add_argument('--training-runs-json', type=Path, help='JSON mapping labels to run directories or metrics JSONL files')
    args = parser.parse_args()
    runs = json.loads(args.training_runs_json.read_text()) if args.training_runs_json else None
    build_report(args.manifest, image_dir=args.image_dir, training_runs=runs)


if __name__ == '__main__':
    main()
