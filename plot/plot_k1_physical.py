#!/usr/bin/env python3
"""Export physical K=1 evaluation data and plot three matched forecast series."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


# Representative pressure levels are declared in advance, not chosen for skill.
SPECS = (
    ('T850', 'temperature', 850, 1.0, 'Temperature at 850 hPa', 'K'),
    ('Z500', 'geopotential', 500, 1.0, 'Geopotential at 500 hPa', r'm$^2$ s$^{-2}$'),
    ('U850', 'u_component_of_wind', 850, 1.0, 'Eastward wind at 850 hPa', r'm s$^{-1}$'),
    ('V850', 'v_component_of_wind', 850, 1.0, 'Northward wind at 850 hPa', r'm s$^{-1}$'),
    ('Q700', 'specific_humidity', 700, 1000.0, 'Specific humidity at 700 hPa', r'g kg$^{-1}$'),
    ('CI250', 'specific_cloud_ice_water_content', 250, 1000.0, 'Cloud ice at 250 hPa', r'g kg$^{-1}$'),
    ('CL850', 'specific_cloud_liquid_water_content', 850, 1000.0, 'Cloud liquid water at 850 hPa', r'g kg$^{-1}$'),
)
GEO_SPECS = tuple((f'Z{level}', 'geopotential', level, 1.0,
                  f'Geopotential at {level} hPa', r'm$^2$ s$^{-2}$')
                 for level in (1, 2, 3, 5, 7, 10, 20, 30))
SERIES = ('era5', 'baseline', 'residual')
STYLES = (('ERA5 truth', '#222222', '-'), ('Frozen NGCM', '#0072B2', '--'),
          ('Residual NGCM', '#D55E00', '-.'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def write_readme(output):
    summary = json.loads((output / 'summary.json').read_text())
    runs = summary['runs']
    representative = 'r2p8_w128_di16' if 'r2p8_w128_di16' in runs else next(iter(runs))
    report = runs[representative]
    with (output / 'physical_metrics_all_levels.csv').open() as stream:
        metrics = list(csv.DictReader(stream))
    with (output / 'objective_contributions.csv').open() as stream:
        contributions = list(csv.DictReader(stream))
    top_geopotential = sum(float(r['baseline_objective_contribution']) for r in contributions
        if r['run_id'] == representative and r['field'] == 'geopotential' and float(r['pressure_hpa']) <= 7)
    lines = [
        '# Six-hour K=1 forecasts in physical units', '',
        '> **The current training loss differs from the NeuralGCM paper.**',
        f'> Geopotential contributes **{report["fields"]["geopotential"]["objective_baseline_share_pct"]:.2f}%** of baseline custom loss;',
        f'> its 1–7 hPa levels alone contribute **{100 * top_geopotential / report["baseline_loss"]:.2f}%** of the total.',
        '> Aggregate loss improvement does not imply improvement across weather variables.',
        '> See the [loss mismatch explanation](../README.md#loss-mismatch-and-geopotential-dominance).', '',
        '**Improving upper-air results:** [geopotential at 1–30 hPa and specific humidity',
        'at 1–5 hPa, with all twelve individual comparisons and all four configurations](../upper_air_physical_eval_20260924/README.md).', '',
        f'![Seven-variable comparison]({representative}/all_variables_global_timeseries.png)', '',
        '**Three curves in every time-series panel:** ERA5 truth is black with circles,',
        'frozen NGCM baseline is blue dashed with squares, and residual NGCM is orange',
        'dash-dot with triangles. The main figure shows global Gaussian-area means.',
        'Each panel names its variable, pressure level, and physical unit.', '',
        f'The horizontal axis is **physical time in hours from a fixed start**, with',
        f'one forecast point every **6 h**. The main window starts at',
        f'`{summary["plot_start"]} UTC` and covers {summary["plot_duration_hours"]} hours.',
        'Dates are retained in the CSV metadata; the plotted ticks are 0, 6, 12, …, 72.', '',
        '**K=1 semantics:** the point at offset 12 h is the six-hour forecast initialized',
        'from ERA5 at offset 6 h. The physical state is reinitialized at each origin.',
        'These curves are successive one-step forecasts, not a free-running 72-hour rollout.',
        'Recurrent memory follows the production validation rule: chronological carry',
        'with resets every 96 records and at time gaps.', '',
        'The model has pressure-level temperature, including T850, but no direct T2m',
        'output. [NeuralGCM Fig. 4a](https://arxiv.org/pdf/2311.07222#page=12) also',
        'plots global mean temperature at **850 hPa**. Its long climate integrations',
        'are a different forecasting protocol from the six-hour evaluations here.', '',
        '## Current checkpoints', '',
        '| Configuration | Epoch | Update | Evaluation job | Validation records |',
        '| --- | ---: | ---: | --- | ---: |',
    ]
    for run, item in runs.items():
        c = item['checkpoint']
        lines.append(f'| {run} | {c["epoch"]} | {c["update"]:,} | {item["job_id"]} | {item["records"]:,} |')
    lines += ['', 'All models use the frozen `20260924_feedback_v2` source and their recorded',
              'checkpoint hashes. Checkpoint choices were frozen before physical evaluation.',
              'This is the **2022 validation set**, not the held-out 2023 test set.', '',
              '## Physical forecast errors', '',
              'Global spatial RMSE uses every grid cell and all 1,455 validation forecasts,',
              'with Gaussian latitude weights and uniform longitude weights, then takes',
              'the square root. It is not the RMSE of the global-mean time series.',
              'The three-day curves are illustrative; the table uses the full year.', '',
              '| Variable / pressure | Unit | NGCM | ' + ' | '.join(r.replace('r2p8_w', 'w').replace('_di', ' / di') for r in runs) + ' |',
              '| --- | --- | ---: | ' + ' | '.join('---:' for _ in runs) + ' |']
    for code, field, pressure, scale, title, unit in SPECS:
        selected = [next(r for r in metrics if r['run_id'] == run and r['field'] == field and float(r['pressure_hpa']) == pressure) for run in runs]
        text_unit = 'g/kg' if scale == 1000 else 'K' if field == 'temperature' else 'm²/s²' if field == 'geopotential' else 'm/s'
        vals = [float(selected[0]['baseline_rmse']) * scale] + [float(r['residual_rmse']) * scale for r in selected]
        lines.append(f'| {code} | {text_unit} | ' + ' | '.join(f'{v:.5g}' for v in vals) + ' |')
    lines += ['', '[Physical RMSE chart](physical_rmse.png) · [All 37 levels, all seven fields](physical_metrics_all_levels.csv)', '',
              '## Why the aggregate improvement is so large', '',
              f'For `{representative}`, the custom objective falls from',
              f'{report["baseline_loss"]:.6f} to {report["residual_loss"]:.6f}, a **{report["objective_improvement_pct"]:.2f}%** reduction.',
              f'Geopotential contributes **{report["fields"]["geopotential"]["objective_baseline_share_pct"]:.2f}%** of baseline loss.',
              f'The 1, 2, 3, 5, and 7 hPa geopotential levels alone contribute',
              f'**{100 * top_geopotential / report["baseline_loss"]:.2f}%** of the total baseline objective.',
              'This concentrates the headline metric on upper-atmosphere geopotential.',
              'It does not mean that temperature, wind, humidity, or clouds improve by that percentage.',
              'Use the per-variable physical errors above to assess forecast skill.', '',
              '[Objective contributions by field and pressure](objective_contributions.csv) ·',
              '[Source identities, checks, and complete reports](summary.json)', '',
              '## Individual figures and data', '',
              f'The main figure uses `{representative}`. Each configuration has the same',
              'seven individual global-mean figures and two location-specific winter/summer',
              'comparison grids. The latter use the nearest Gaussian grid cells to New York',
              '(40.464°N, 73.125°W) and Beijing (40.464°N, 115.312°E), not station data.',
              'The winter and summer windows start at 2022-01-10 00 UTC and 2022-07-10 00 UTC.', '',
              '| Variable | PNG | PDF | SVG |', '| --- | --- | --- | --- |']
    for code, _, _, _, title, _ in SPECS:
        stem = f'{representative}/{code}_global_timeseries'
        lines.append(f'| {title} | [PNG]({stem}.png) | [PDF]({stem}.pdf) | [SVG]({stem}.svg) |')
    lines += ['', '| Configuration | Global overview | New York | Beijing | Time-series CSV |',
              '| --- | --- | --- | --- | --- |']
    for run in runs:
        lines.append(f'| {run} | [Figure]({run}/all_variables_global_timeseries.png) | [Figure]({run}/all_variables_new_york.png) | [Figure]({run}/all_variables_beijing.png) | [CSV]({run}_timeseries.csv) |')
    lines += ['', 'CSV time-series columns use Kelvin, m²/s² for geopotential, m/s for wind,',
              'and g/kg for humidity/clouds. `physical_metrics_all_levels.csv` keeps the',
              'original SI units, including kg/kg for humidity/clouds. Z500 here denotes',
              'geopotential, not geopotential height; divide by 9.80665 to convert to metres.', '',
              '## Validation and reproduction', '',
              'An independent eight-date GPU preflight checked live/cache equality, production',
              'forward-pass and recurrent-memory parity, and independent NumPy float64 error',
              'reductions. Every complete replay independently reproduced its saved K=1 loss.',
              'No training or backbone weights were modified. GPU jobs used `gpu-test`,',
              '`gputest`, one A100 GPU, and a one-hour limit. Original ERA5 fields were',
              'conservatively regridded to the common 128 × 64 Gaussian verification grid.', '',
              'Redraw from the committed CSV files using Python, NumPy and Matplotlib:', '',
              '```bash', 'python plot/plot_k1_physical.py', '```', '',
              'Refresh the data from completed evaluation reports:', '', '```bash',
              'python plot/plot_k1_physical.py --results-root logs/neuralgcm_physical_eval_20260924',
              '```', '', 'The evaluation source is',
              '[evaluate_neuralgcm_k1_physical.py](../../scripts/diagnostics/evaluate_neuralgcm_k1_physical.py)',
              'and its [Slurm launcher](../../scripts/diagnostics/run_neuralgcm_k1_physical.sbatch).', '']
    geo_lines = ['## Where geopotential improves', '',
                 '![Improved geopotential levels](' + representative + '/geopotential_improved_levels.png)', '',
                 'These eight levels are the levels with improved full-year geopotential RMSE',
                 'for the representative checkpoint. They were selected after the error audit,',
                 'as requested, and are explicitly labeled; the 500 hPa comparison remains above.',
                 'The same eight levels are also plotted for every other configuration.', '',
                 '| Pressure (hPa) | Baseline RMSE (m²/s²) | Residual RMSE (m²/s²) | RMSE reduction |',
                 '| ---: | ---: | ---: | ---: |']
    for _, field, pressure, _, _, _ in GEO_SPECS:
        row = next(r for r in metrics if r['run_id'] == representative and r['field'] == field and float(r['pressure_hpa']) == pressure)
        geo_lines.append(f'| {pressure} | {float(row["baseline_rmse"]):,.2f} | {float(row["residual_rmse"]):,.2f} | {float(row["rmse_improvement_pct"]):.2f}% |')
    geo_lines += ['', f'[Loss contributions and RMSE reductions]({representative}/geopotential_loss_breakdown.png)', '',
                  f'Geopotential accounts for {100 * report["fields"]["geopotential"]["objective_score_residual"] / (7 * report["residual_loss"]):.2f}%',
                  f'of the residual model objective, compared with {report["fields"]["geopotential"]["objective_baseline_share_pct"]:.2f}% for baseline.',
                  'The stacked bars show contributions to the actual custom objective,',
                  'including its field scaling and vertical weighting; they are not physical-energy shares.', '',
                  '**This is not the original NeuralGCM training objective.** The original',
                  '[G.3–G.4 definition](https://arxiv.org/pdf/2311.07222#page=41) uses 24-hour',
                  'difference standard deviations and includes spectral, bias, and internal-state',
                  'loss terms. Our residual objective uses six-hour change statistics and decoded',
                  'weighted MSE. It borrows the amplitude factors (geopotential 2, humidity 0.66,',
                  'cloud species 0.05), but does not fully reproduce the original loss.',
                  'The reported 95% share is an empirical imbalance of this custom objective,',
                  'not a prescribed NeuralGCM weighting. The subsequent',
                  '[normalization audit](../loss_alignment_audit_20260924/README.md) finds that',
                  '24-hour scales alone do not remove this imbalance. These physical results',
                  'still describe checkpoints trained with the existing custom objective.', '']
    position = lines.index('## Individual figures and data')
    lines[position:position] = geo_lines
    (output / 'README.md').write_text('\n'.join(lines))


def export(results, output, run_id=None):
    plan_path = results / 'evaluation_plan.json'
    plan = json.loads(plan_path.read_text())
    summaries, metric_rows, contributions = {}, [], []
    manifest = json.loads((Path(plan['experiment_root']) / 'manifest.json').read_text())
    stats_path = Path(manifest['resources']['res2p8']['statistics'])
    stats = json.loads(stats_path.read_text())
    for run, checkpoint in plan['runs'].items():
        if run_id is not None and run != run_id:
            continue
        candidates = sorted(results.glob(f'full-{run}-*/report.json'))
        candidates = [p for p in candidates if json.loads(p.read_text()).get('completed')]
        if not candidates:
            raise ValueError(f'No completed full evaluation for {run}')
        path = candidates[-1]
        report = json.loads(path.read_text())
        if report['smoke'] or report['checkpoint'] != checkpoint or report['plan_sha256'] != sha256(plan_path):
            raise ValueError(f'Evaluation does not match the frozen plan: {path}')
        if report['records'] != 1455:
            raise ValueError('Full 2022 validation is required')
        for artifact, digest in report['artifacts'].items():
            if sha256(path.parent / artifact) != digest:
                raise ValueError(f'Evaluation artifact changed: {artifact}')
        summaries[run] = dict(report, report_sha256=sha256(path))
        with (path.parent / 'physical_metrics.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        metric_rows.extend(rows)
        for field, detail in report['fields'].items():
            subset = [r for r in rows if r['field'] == field]
            levels = np.array([float(r['pressure_hpa']) for r in subset], np.float32)
            weights = levels / levels.sum()
            scales = np.asarray(stats['loss_scales'][field], np.float64)
            if field != 'specific_humidity':
                scales = np.full_like(scales, np.sqrt(np.mean(scales ** 2)))
            amplitude = 0.05 if field.startswith('specific_cloud_') else 2.0 if field == 'geopotential' else 0.66 if field == 'specific_humidity' else 1.0
            scales = (scales / amplitude).astype(np.float32)
            values = {}
            for model in ('baseline', 'residual'):
                values[model] = np.array([float(r[f'{model}_rmse']) ** 2 for r in subset]) / scales.astype(np.float64) ** 2 * weights
                np.testing.assert_allclose(values[model].sum(), detail[f'objective_score_{model}'], rtol=2e-4, atol=1e-9)
            for k, pressure in enumerate(levels):
                contributions.append(dict(run_id=run, field=field, pressure_hpa=float(pressure),
                    baseline_objective_contribution=float(values['baseline'][k] / 7),
                    residual_objective_contribution=float(values['residual'][k] / 7)))
        with np.load(path.parent / 'timeseries.npz', allow_pickle=False) as data:
            fields, levels, sites = data['fields'].tolist(), data['levels_hpa'].tolist(), data['sites'].tolist()
            times = data['valid_time']
            forecasts, global_means = data['site_values'], data['global_mean']
            exported = []
            for location in ['Global mean'] + sites:
                values = global_means if location == 'Global mean' else forecasts[..., sites.index(location)]
                for i, valid in enumerate(times):
                    row = dict(valid_time=str(valid), origin_time=str(np.datetime64(valid, 'h') - np.timedelta64(6, 'h')), location=location)
                    for code, field, level, scale, _, _ in SPECS + GEO_SPECS:
                        for j, model in enumerate(SERIES):
                            row[f'{code}_{model}'] = format(float(values[i, j, fields.index(field), levels.index(level)]) * scale, '.10g')
                    exported.append(row)
            write_csv(output / f'{run}_timeseries.csv', exported)
    write_csv(output / 'physical_metrics_all_levels.csv', metric_rows)
    write_csv(output / 'objective_contributions.csv', contributions)
    (output / 'summary.json').write_text(json.dumps(dict(
        captured_at_utc=datetime.now(timezone.utc).isoformat(), plan=plan, runs=summaries,
        statistics_sha256=sha256(stats_path),
        plot_time_definition='hours since fixed t0; each marker is a fresh six-hour K=1 forecast, not autoregressive lead',
        plot_start=plan['plot_windows'][0][0] + 'T00', plot_duration_hours=72,
        plot_specs=[dict(zip(('code', 'field', 'pressure_hpa', 'display_scale', 'title', 'unit'), spec)) for spec in SPECS],
        improved_geopotential_specs=[dict(zip(('code', 'field', 'pressure_hpa', 'display_scale', 'title', 'unit'), spec)) for spec in GEO_SPECS],
    ), indent=2) + '\n')


def draw(output):
    with tempfile.TemporaryDirectory(prefix='ngcm-physical-plot-') as cache:
        os.environ.setdefault('MPLCONFIGDIR', cache)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator
        plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 10,
                             'axes.spines.top': False, 'axes.spines.right': False,
                             'axes.linewidth': 0.7, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                             'svg.hashsalt': 'neuralgcm-k1-physical'})
        summary = json.loads((output / 'summary.json').read_text())
        with (output / 'physical_metrics_all_levels.csv').open() as stream:
            metrics = list(csv.DictReader(stream))
        def save(fig, stem):
            for ext in ('png', 'pdf', 'svg'):
                metadata = {'CreationDate': None} if ext == 'pdf' else {'Date': None} if ext == 'svg' else None
                fig.savefig(str(stem) + '.' + ext, dpi=180, facecolor='white', metadata=metadata)
                if ext == 'svg':
                    path = Path(str(stem) + '.svg')
                    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
            plt.close(fig)
        def panel(ax, rows, spec, start):
            code, _, _, _, title, unit = spec
            times = [float((np.datetime64(r['valid_time'], 'h') - np.datetime64(start, 'h')) / np.timedelta64(1, 'h')) for r in rows]
            for j, (suffix, (label, color, style)) in enumerate(zip(SERIES, STYLES)):
                ax.plot(times, [float(r[f'{code}_{suffix}']) for r in rows], label=label,
                        color=color, linestyle=style, linewidth=1.15 if j else 1.5,
                        marker=('o', 's', '^')[j], markersize=3.4, markerfacecolor='white',
                        markeredgewidth=0.8, zorder=4-j, alpha=0.95)
            ax.set_ylabel(f'{title}\n({unit})')
            ax.grid(color='#e7e7e7', linewidth=0.5)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 4), useOffset=False)
            ax.xaxis.set_major_locator(MultipleLocator(6))
            ax.set_xlim(0, summary['plot_duration_hours'] + 1)
            ax.tick_params(axis='x', labelsize=8)
        def window(rows, start):
            end = np.datetime64(start, 'h') + np.timedelta64(summary['plot_duration_hours'], 'h')
            return [r for r in rows if np.datetime64(start, 'h') < np.datetime64(r['valid_time'], 'h') <= end]
        for run, report in summary['runs'].items():
            with (output / f'{run}_timeseries.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            folder = output / run
            folder.mkdir(exist_ok=True)
            start = summary['plot_start']
            global_rows = window([r for r in rows if r['location'] == 'Global mean'], start)
            for spec in SPECS + GEO_SPECS:
                fig, ax = plt.subplots(figsize=(9, 3.4), constrained_layout=True)
                panel(ax, global_rows, spec, start)
                ax.set_xlabel(r'Physical time from $t_0$ (h); points every 6 h')
                ax.legend(loc='best', frameon=False, ncol=3, fontsize=9)
                save(fig, folder / f'{spec[0]}_global_timeseries')
            fig, axes = plt.subplots(4, 2, figsize=(12, 10.4), constrained_layout=True)
            for ax, spec in zip(axes.flat, SPECS):
                panel(ax, global_rows, spec, start)
                ax.set_xlabel(r'Physical time from $t_0$ (h)')
            axes.flat[-1].axis('off')
            handles, labels = axes.flat[0].get_legend_handles_labels()
            axes.flat[-1].legend(handles, labels, loc='center', frameon=False, fontsize=13)
            axes.flat[-1].text(0.5, 0.18, f'Global area means; six-hour lead\n{run}; update {report["checkpoint"]["update"]}',
                               transform=axes.flat[-1].transAxes, ha='center', fontsize=10)
            save(fig, folder / 'all_variables_global_timeseries')
            fig, axes = plt.subplots(4, 2, figsize=(12, 10.4), constrained_layout=True)
            for ax, spec in zip(axes.flat, GEO_SPECS):
                panel(ax, global_rows, spec, start)
                ax.set_xlabel(r'Physical time from $t_0$ (h)')
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='outside upper center', ncol=3, frameon=False)
            save(fig, folder / 'geopotential_improved_levels')
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
            bottoms = np.zeros(2)
            colors = ['#0072B2', '#D55E00', '#009E73', '#56B4E9', '#CC79A7', '#F0E442', '#999999']
            labels = ['Temperature', 'Geopotential', 'Eastward wind', 'Northward wind', 'Humidity', 'Cloud ice', 'Cloud liquid']
            for (field, detail), color, label in zip(report['fields'].items(), colors, labels):
                heights = np.array([detail['objective_score_baseline'], detail['objective_score_residual']]) / 7
                axes[0].bar([0, 1], heights, bottom=bottoms, color=color, label=label, width=0.55)
                bottoms += heights
            axes[0].set_xticks([0, 1], ['Frozen NGCM', 'Residual NGCM'])
            axes[0].set_ylabel('Contribution to custom loss')
            axes[0].set_ylim(0, max(bottoms) * 1.12)
            for x, value in enumerate(bottoms):
                axes[0].text(x, value + max(bottoms) * 0.025, f'{value:.3f}', ha='center')
            axes[0].legend(frameon=False, fontsize=8, loc='upper right')
            selected = [next(r for r in metrics if r['run_id'] == run and r['field'] == 'geopotential' and float(r['pressure_hpa']) == spec[2]) for spec in GEO_SPECS]
            improvements = [float(r['rmse_improvement_pct']) for r in selected]
            axes[1].bar(range(len(selected)), improvements, color='#009E73', width=0.65)
            axes[1].set_xticks(range(len(selected)), [str(spec[2]) for spec in GEO_SPECS])
            axes[1].set_xlabel('Pressure level (hPa)')
            axes[1].set_ylabel('Geopotential RMSE reduction (%)')
            axes[1].set_ylim(0, max(105, max(improvements) + 8))
            for x, value in enumerate(improvements):
                axes[1].text(x, value + 1.5, f'{value:.1f}', ha='center', fontsize=8)
            save(fig, folder / 'geopotential_loss_breakdown')
            for site in summary['plan']['sites']:
                selected = [r for r in rows if r['location'] == site['name']]
                fig, axes = plt.subplots(7, 2, figsize=(12, 15), constrained_layout=True)
                for j, (start, end) in enumerate(summary['plan']['plot_windows']):
                    selected_window = window(selected, start)
                    if not selected_window:
                        raise ValueError(f'Empty requested time window: {start}')
                    for ax, spec in zip(axes[:, j], SPECS):
                        panel(ax, selected_window, spec, start)
                    axes[0, j].set_title(f'{site["name"]}, {"winter" if j == 0 else "summer"} window', fontsize=12)
                    axes[-1, j].set_xlabel(r'Physical time from $t_0$ (h)')
                    axes[0, j].legend(frameon=False, fontsize=8, ncol=3)
                save(fig, folder / ('all_variables_' + site['name'].lower().replace(' ', '_')))
        # Global spatial forecast error differs from error of the global mean.
        runs = list(summary['runs'])
        fig, axes = plt.subplots(4, 2, figsize=(11, 10.5), constrained_layout=True)
        for ax, spec in zip(axes.flat, SPECS):
            code, field, pressure, scale, title, unit = spec
            selected = [next(r for r in metrics if r['run_id'] == run and r['field'] == field
                             and float(r['pressure_hpa']) == pressure) for run in runs]
            bases = [float(r['baseline_rmse']) * scale for r in selected]
            np.testing.assert_allclose(bases, bases[0], rtol=1e-6)
            values = [bases[0]] + [float(r['residual_rmse']) * scale for r in selected]
            ax.bar(range(len(values)), values, color=['#555555', '#0072B2', '#D55E00', '#009E73', '#CC79A7'][:len(values)], width=0.65)
            ax.set_xticks(range(len(values)), ['NGCM'] + [run.replace('r2p8_w', '').replace('_di', '/') for run in runs])
            ax.set_ylabel(f'RMSE ({unit})')
            ax.set_title(title, fontsize=11)
            ax.grid(axis='y', color='#e7e7e7', linewidth=0.5)
            ax.set_axisbelow(True)
        axes.flat[-1].axis('off')
        axes.flat[-1].text(0.5, 0.5, 'Global spatial RMSE\n1,455 six-hour forecasts, 2022\nLower is better',
                           transform=axes.flat[-1].transAxes, ha='center', va='center', fontsize=13)
        save(fig, output / 'physical_rmse')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, help='Refresh portable data from completed GPU reports')
    parser.add_argument('--run-id', help='Export a single run; by default require all four completed runs')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'k1_physical_eval_20260924')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.results_root:
        export(args.results_root, args.output_dir, args.run_id)
    draw(args.output_dir)
    write_readme(args.output_dir)


if __name__ == '__main__':
    main()
