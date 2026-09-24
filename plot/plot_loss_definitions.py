#!/usr/bin/env python3
"""Compare README, running v2, and 24 h simplified losses versus training step."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


FIELDS = ('temperature', 'geopotential', 'u_component_of_wind', 'v_component_of_wind',
          'specific_humidity', 'specific_cloud_ice_water_content', 'specific_cloud_liquid_water_content')
LABELS = ('Total loss', 'Temperature', 'Geopotential', 'Eastward wind', 'Northward wind',
          'Specific humidity', 'Cloud ice', 'Cloud liquid water')
RECIPES = {
    'readme_v1': 'Original README loss (6 h)',
    'current_v2': 'Current training loss v2 (6 h)',
    'normalized24_mse': '24 h simplified data MSE',
}
AMPLITUDES = {f: 2. if f == 'geopotential' else .66 if f == 'specific_humidity'
              else .05 if f.startswith('specific_cloud_') else 1. for f in FIELDS}
COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7')
MARKERS = ('o', 's', '^', 'D')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_write(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def coefficients(levels, historical, fitted):
    levels32 = np.asarray(levels, np.float32)
    pressure = (levels32 / levels32.sum()).astype(np.float64)
    result = {r: [] for r in RECIPES}
    for field in FIELDS:
        original = np.asarray(historical['loss_scales'][field], np.float32).astype(np.float64)
        pooled = np.asarray(historical['loss_scales'][field], np.float64)
        if field != 'specific_humidity':
            pooled = np.full_like(pooled, np.sqrt(np.mean(pooled ** 2)))
        pooled = (pooled / AMPLITUDES[field]).astype(np.float32).astype(np.float64)
        sigma24 = np.asarray(fitted['scales']['24h_uniform'][field], np.float64)
        result['readme_v1'].append(pressure / original ** 2 / len(FIELDS))
        result['current_v2'].append(pressure / pooled ** 2 / len(FIELDS))
        result['normalized24_mse'].append(np.ones_like(levels32, np.float64) / len(levels) *
                                         AMPLITUDES[field] ** 2 / sigma24 ** 2 / (1 + 6 / 24))
    return {recipe: np.stack(v) for recipe, v in result.items()}


def export(results, fitted_path, output):
    plan_path = results / 'evaluation_plan.json'
    plan = json.loads(plan_path.read_text())
    manifest = json.loads((Path(plan['experiment_root']) / 'manifest.json').read_text())
    historical_path = Path(manifest['resources']['res2p8']['statistics'])
    historical = json.loads(historical_path.read_text())
    fitted = json.loads(fitted_path.read_text())
    if fitted['split'] != 'train' or fitted['snapshot_count'] != 60 or fitted['train_years'] != list(range(2015, 2022)):
        raise ValueError('Expected the fixed train-only normalization audit')
    if any(not ('2015' <= t[:4] <= '2021') for t in fitted['consumed_frame_sha256']):
        raise ValueError('Normalization has data outside training years')
    smoke_reports = []
    for path in sorted(results.glob('smoke-*/report.json')):
        report = json.loads(path.read_text())
        if (report.get('completed') and report['smoke'] and report['records'] == 8
                and report['plan_sha256'] == sha(plan_path)):
            smoke_reports.append(dict(report, report_sha256=sha(path)))
    if not smoke_reports:
        raise ValueError('Missing independent representative-data GPU smoke report')
    rows, physical, reports = [], [], {}
    common_baseline = None
    for run, checkpoints in plan['runs'].items():
        candidates = sorted(results.glob(f'full-{run}-*/report.json'))
        candidates = [p for p in candidates if json.loads(p.read_text()).get('completed')]
        if not candidates:
            raise ValueError(f'Missing completed full replay: {run}')
        path = candidates[-1]
        report = json.loads(path.read_text())
        if (report['smoke'] or report['records'] != 1455 or report['checkpoints'] != checkpoints
                or report['plan_sha256'] != sha(plan_path)):
            raise ValueError(f'Replay differs from frozen plan: {run}')
        for name, digest in report['artifacts'].items():
            if sha(path.parent / name) != digest:
                raise ValueError(f'Replay artifact changed: {name}')
        with np.load(path.parent / 'physical_mse.npz', allow_pickle=False) as data:
            levels = data['pressure_hpa']
            np.testing.assert_array_equal(data['fields'], FIELDS)
            np.testing.assert_array_equal(levels, fitted['pressure_hpa'])
            np.testing.assert_array_equal(data['updates'], [c['update'] for c in checkpoints])
            np.testing.assert_array_equal(data['epochs'], [c['epoch'] for c in checkpoints])
            base, pred = data['baseline_mse'], data['residual_mse']
            if (not np.isfinite(base).all() or not np.isfinite(pred).all()
                    or np.any(base < 0) or np.any(pred < 0)):
                raise ValueError(f'Invalid physical MSE values: {run}')
            if common_baseline is None:
                common_baseline = base
            np.testing.assert_allclose(base, common_baseline, rtol=2e-5, atol=0)
            np.testing.assert_allclose(pred[0], base, rtol=2e-5, atol=0)
            coef = coefficients(levels, historical, fitted)
            for i, checkpoint in enumerate(checkpoints):
                for f, field in enumerate(FIELDS):
                    for k, level in enumerate(levels):
                        physical.append(dict(run_id=run, epoch=checkpoint['epoch'], update=checkpoint['update'],
                                             field=field, pressure_hpa=float(level),
                                             baseline_mse=float(base[f, k]), residual_mse=float(pred[i, f, k])))
                for recipe, weights in coef.items():
                    b, c = np.sum(base * weights, axis=-1), np.sum(pred[i] * weights, axis=-1)
                    if recipe == 'current_v2':
                        np.testing.assert_allclose(c.sum(), checkpoint['validation_loss'], rtol=2e-5)
                    for field, bv, cv in zip(('total',) + FIELDS, np.r_[b.sum(), b], np.r_[c.sum(), c]):
                        rows.append(dict(recipe=recipe, run_id=run, epoch=checkpoint['epoch'], update=checkpoint['update'],
                                         field=field, baseline_loss=float(bv), residual_loss=float(cv),
                                         improvement_pct=float(100 * (1 - cv / bv)),
                                         baseline_share_pct=float(100 * bv / b.sum()),
                                         residual_share_pct=float(100 * cv / c.sum())))
        reports[run] = dict(report, report_sha256=sha(path))
    csv_write(output / 'loss_curves.csv', rows)
    csv_write(output / 'physical_mse_by_step.csv', physical)
    (output / 'provenance.json').write_text(json.dumps(dict(
        captured_at_utc=datetime.now(timezone.utc).isoformat(), plan=plan, reports=reports,
        smoke_reports=smoke_reports,
        statistics24_sha256=sha(fitted_path), historical_statistics_sha256=sha(historical_path),
        statistics24_path=str(fitted_path),
        coefficient_matrices={k: v.tolist() for k, v in coef.items()},
        statistics24=fitted, historical_loss_scales=historical['loss_scales'],
        historical_statistics_metadata={k: historical[k] for k in
            ('version', 'split', 'dataset_id', 'origin_count', 'origins_sha256', 'loss_floors')},
        training_objective='neuralgcm_pooled_change_mse_v2',
        role='All three curves rescore the same v2-trained checkpoints; no retraining with other losses.',
        x_axis='training step (optimizer update count)',
        score_definition='sum of seven field contributions; lower is better',
        loss_curves_sha256=sha(output / 'loss_curves.csv'),
        physical_mse_sha256=sha(output / 'physical_mse_by_step.csv'),
    ), indent=2, allow_nan=False) + '\n')


def load_rows(output):
    with (output / 'loss_curves.csv').open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ('epoch', 'update'):
            row[key] = int(row[key])
        for key in ('baseline_loss', 'residual_loss', 'improvement_pct', 'baseline_share_pct', 'residual_share_pct'):
            row[key] = float(row[key])
    return rows


def draw(output):
    with tempfile.TemporaryDirectory(prefix='ngcm-loss-mpl-') as config_dir:
        os.environ.setdefault('MPLCONFIGDIR', config_dir)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                             'axes.spines.top': False, 'axes.spines.right': False,
                             'pdf.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'ngcm-loss-definitions'})
        rows = load_rows(output)
        runs = sorted({r['run_id'] for r in rows})

        def save(fig, stem):
            for ext in ('png', 'pdf', 'svg'):
                path = output / f'{stem}.{ext}'
                metadata = {'CreationDate': None} if ext == 'pdf' else {'Date': None} if ext == 'svg' else None
                fig.savefig(path, dpi=180, bbox_inches='tight', metadata=metadata)
                if ext == 'svg':
                    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
            plt.close(fig)

        def panel(ax, recipe, field, improvement=False):
            subset = [r for r in rows if r['recipe'] == recipe and r['field'] == field]
            ys = []
            for i, run in enumerate(runs):
                points = sorted((r for r in subset if r['run_id'] == run), key=lambda r: r['update'])
                y = [r['improvement_pct' if improvement else 'residual_loss'] for r in points]
                ys.extend(y)
                label = run.replace('r2p8_', '').replace('_', ' / ')
                ax.plot([r['update'] for r in points], y, color=COLORS[i], marker=MARKERS[i],
                        ms=3.7, lw=1.5, label=label)
            baseline = 0. if improvement else subset[0]['baseline_loss']
            ax.axhline(baseline, color='#222222', linestyle='--', lw=1.5,
                       label='Frozen NGCM baseline', zorder=1)
            ys.append(baseline)
            if improvement:
                if max(abs(v) for v in ys) > 250:
                    ax.set_yscale('symlog', linthresh=10)
                    ax.set_ylabel('MSE reduction (%) [symlog]')
                else:
                    ax.set_ylabel('MSE reduction (%)')
            else:
                positive = [v for v in ys if v > 0]
                if max(positive) / min(positive) > 100:
                    ax.set_yscale('log')
                    ax.set_ylabel('Loss contribution [log]')
                else:
                    ax.set_ylim(bottom=0)
                    ax.set_ylabel('Loss' if field == 'total' else 'Loss contribution')
            ax.set_xlabel('Training step')
            ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
            ax.set_xlim(left=0)
            ax.grid(alpha=.2, linewidth=.5)

        for recipe, name in RECIPES.items():
            (output / recipe).mkdir(exist_ok=True)
            for improvement in (False, True):
                fig, axes = plt.subplots(4, 2, figsize=(12, 14), constrained_layout=True)
                for ax, field, label in zip(axes.flat, ('total',) + FIELDS, LABELS):
                    panel(ax, recipe, field, improvement)
                    ax.set_title(label, fontsize=11)
                axes[0, 0].legend(fontsize=8, loc='best')
                fig.suptitle(name + '\nSame checkpoints, same 2022 validation; rescoring only', fontsize=13)
                save(fig, recipe + ('/improvement_by_training_step' if improvement else '/loss_by_training_step'))
            for field, label in zip(('total',) + FIELDS, LABELS):
                fig, ax = plt.subplots(figsize=(7, 4.3), constrained_layout=True)
                panel(ax, recipe, field)
                ax.set_title(f'{name} — {label}\nSame v2-trained checkpoints; 2022 validation rescoring', fontsize=10)
                ax.legend(fontsize=8, loc='best')
                save(fig, recipe + '/' + field + '_loss')
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
        for ax, (recipe, name) in zip(axes, RECIPES.items()):
            panel(ax, recipe, 'total', improvement=True)
            ax.set_title(name, fontsize=11)
        axes[0].legend(fontsize=8, loc='best')
        fig.suptitle('Same v2-trained checkpoints and baseline; only the scoring definition changes', fontsize=12)
        save(fig, 'total_improvement_comparison')


def write_readme(output):
    rows = load_rows(output)
    provenance = json.loads((output / 'provenance.json').read_text())
    runs = sorted(provenance['reports'])
    lines = ['# Three loss definitions on the same training checkpoints', '',
             '**All models here were trained with the current 6 h v2 objective.**',
             'The original README and 24 h panels rescore those same checkpoints.',
             'They are not separate training runs and are not paper-loss reproductions.', '',
             'Every curve uses **training step (optimizer updates)** on the horizontal axis.',
             'Every checkpoint is evaluated on the same **1,455 K=1, six-hour forecasts**',
             'from the 2022 validation set. Each origin uses ERA5 physical initialization;',
             'recurrent memory follows the chronological production validation policy.',
             'The black dashed line is the frozen NGCM baseline under that panel\'s metric.',
             'Colored curves distinguish the four residual configurations.', '',
             '![Total improvement comparison](total_improvement_comparison.png)', '',
             'Loss is lower-is-better. Improvement is `100 × (1 − ours / baseline)`;',
             'negative values mean worse than baseline. Loss magnitudes between different',
             'definitions are not directly comparable. Each variable panel is its contribution',
             'to the total across all 37 pressure levels, so the seven contributions sum to',
             'total loss. Very large ranges are explicitly labeled log or symlog.', '',
             '| Definition | Normalization | Vertical / field reduction | Status |',
             '| --- | --- | --- | --- |',
             '| Original README | Per-level 6 h change standard deviations and original floors; no extra variable amplitudes | Pressure-proportional levels; average seven fields | Initial project recipe |',
             '| Current v2 | RMS-pooled 6 h scales except humidity; amplitudes Z=2, Q=0.66, clouds=0.05 | Pressure-proportional levels; average seven fields | Actual objective used to train these checkpoints |',
             '| 24 h simplified MSE | Pooled 24 h standard deviations except per-level humidity; same amplitudes | Equal-level mean, field sum, six-hour lead-time factor 0.8 on squared errors | Offline candidate; omits filtering, internal-state, spectral and bias losses |', '',
             'The 24 h scales use 60 fixed training-only snapshots in 2015–2021, with',
             'uniform latitude weighting for fitting statistics; scoring uses Gaussian area',
             'weights. The equal-level and field-sum reductions follow the official reference',
             'defaults. Exact paper loss bindings, pressure masks and original statistical',
             'samples have not been reproduced. The common paper data-term multiplier 20',
             'is omitted, which does not affect improvements or shares.', '',
             'The 24 h candidate also changes the vertical and field reductions; this is',
             'not an ablation of only the normalization interval. The separate sensitivity',
             'audit contains a matched-snapshot 6 h / 24 h comparison with fixed reductions.', '',
             '[Paper G.3–G.4](https://arxiv.org/pdf/2311.07222#page=41) ·',
             '[Official reference reducers](https://github.com/neuralgcm/neuralgcm/blob/main/neuralgcm/reference_code/metrics_base.py) ·',
             '[Statistics and weighting sensitivity audit](../loss_alignment_audit_20260924/README.md)', '',
             '## Latest evaluated checkpoints', '',
             '| Configuration | Epoch | Training step | Replay job |', '| --- | ---: | ---: | --- |']
    for run in runs:
        report = provenance['reports'][run]
        checkpoint = report['checkpoints'][-1]
        lines.append(f'| {run} | {checkpoint["epoch"]} | {checkpoint["update"]:,} | {report["job_id"]} |')
    lines += ['', 'The epoch checkpoint set was frozen before replay. Each replay reproduced',
              'the original v2 validation score at every checkpoint, checked the zero-residual',
              'initialization against baseline, and independently checked physical MSE and',
              'production recurrent-memory updates. An independent representative-data GPU',
              'smoke preceded complete replays. All these GPU jobs use `gpu-test` / `gputest`',
              'with time limits at or below one hour.', '']
    for recipe, name in RECIPES.items():
        lines += [f'## {name}', '', f'![Loss curves]({recipe}/loss_by_training_step.png)', '',
                  f'[Improvement curves]({recipe}/improvement_by_training_step.png) ·',
                  f'[PDF]({recipe}/loss_by_training_step.pdf) · [SVG]({recipe}/loss_by_training_step.svg)', '',
                  '| Variable | Individual loss curve |', '| --- | --- |']
        for field, label in zip(('total',) + FIELDS, LABELS):
            lines.append(f'| {label} | [PNG]({recipe}/{field}_loss.png) · [PDF]({recipe}/{field}_loss.pdf) · [SVG]({recipe}/{field}_loss.svg) |')
        lines += ['', '| Configuration | Baseline total loss | Residual total loss | Reduction | Baseline geopotential share |',
                  '| --- | ---: | ---: | ---: | ---: |']
        for run in runs:
            point = max((r for r in rows if r['recipe'] == recipe and r['run_id'] == run and r['field'] == 'total'), key=lambda r: r['update'])
            geo = next(r for r in rows if r['recipe'] == recipe and r['run_id'] == run and r['field'] == 'geopotential' and r['update'] == point['update'])
            share = geo['baseline_share_pct']
            share_text = f'{share:.5f}' if share < .01 else f'{share:.2f}'
            lines.append(f'| {run} | {point["baseline_loss"]:.6g} | {point["residual_loss"]:.6g} | {point["improvement_pct"]:.2f}% | {share_text}% |')
        lines.append('')
    lines += ['## Interpretation and provenance', '',
              'The README recipe is explicitly a custom decoded loss, as stated in',
              '[section 9 of the original project plan](../../docs/experiments/neuralgcm_residual/README.md#9-objective-validation-and-final-report).',
              'The v2 restart subsequently pooled scales and added amplitudes. Simply replacing',
              '6 h scales with 24 h scales does not eliminate geopotential dominance; the',
              'normalization audit shows this independently of any new training.',
              'No normalization change alters the underlying physical RMSE. These charts',
              'must be read alongside the [physical-variable evaluation](../k1_physical_eval_20260924/README.md).', '',
              '[Exact formulas, README differences, diagnosed problems and proposed design](../LOSS_DEFINITION_AND_DESIGN.md)', '',
              '[Loss-curve CSV](loss_curves.csv) · [Per-level physical MSE by step](physical_mse_by_step.csv) ·',
              '[Checkpoints, coefficients, statistics and replay reports](provenance.json)', '',
              'Redraw from portable committed CSV files:', '', '```bash',
              'python plot/plot_loss_definitions.py', '```', '',
              'Refresh after completed full replays:', '', '```bash',
              'python plot/plot_loss_definitions.py --results-root logs/neuralgcm_loss_curves_20260924',
              '```', '']
    (output / 'README.md').write_text('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root', type=Path)
    p.add_argument('--statistics24', type=Path, default=Path('plot/loss_alignment_audit_20260924/statistics.json'))
    p.add_argument('--output-dir', type=Path, default=Path('plot/loss_definitions_20260924'))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.results_root:
        export(args.results_root, args.statistics24, args.output_dir)
    draw(args.output_dir)
    write_readme(args.output_dir)


if __name__ == '__main__':
    main()
