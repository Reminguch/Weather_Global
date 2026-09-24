#!/usr/bin/env python3
"""Publish the improving upper-air fields from the existing K=1 evaluation."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent
PHYSICAL = ROOT / 'k1_physical_eval_20260924'
REPRESENTATIVE = 'r2p8_w128_di16'
SPECS = [
    (f'Z{p}', 'geopotential', p, 1., 'Geopotential', 'm²/s²')
    for p in (1, 2, 3, 5, 7, 10, 20, 30)
] + [
    (f'Q{p}', 'specific_humidity', p, 1000., 'Specific humidity', 'g/kg')
    for p in (1, 2, 3, 5)
]
STYLES = [('era5', 'ERA5 truth', '#222222', '-', 'o'),
          ('baseline', 'Frozen NGCM', '#0072B2', '--', 's'),
          ('residual', 'Residual NGCM', '#D55E00', '-.', '^')]
START = np.datetime64('2022-01-10T00', 'h')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def export(results, output):
    summary_path = PHYSICAL / 'summary.json'
    summary = json.loads(summary_path.read_text())
    rows, sources = [], {}
    for run, saved in summary['runs'].items():
        path = results / f'full-{run}-{saved["job_id"]}' / 'report.json'
        report = json.loads(path.read_text())
        assert sha(path) == saved['report_sha256'], path
        assert report['completed'] and not report['smoke']
        assert report['records'] == 1455 and report['checkpoint'] == saved['checkpoint']
        for artifact, digest in report['artifacts'].items():
            assert sha(path.parent / artifact) == digest, artifact
        with np.load(path.parent / 'timeseries.npz', allow_pickle=False) as data:
            assert data['models'].tolist() == ['ERA5', 'NGCM', 'Residual NGCM']
            fields, levels = data['fields'].tolist(), data['levels_hpa'].tolist()
            times = data['valid_time'].astype('datetime64[h]')
            offsets = (times - START) / np.timedelta64(1, 'h')
            selection = np.flatnonzero((offsets > 0) & (offsets <= 72))
            np.testing.assert_array_equal(offsets[selection], np.arange(6, 73, 6))
            values = data['global_mean']
            for code, field, pressure, scale, _, unit in SPECS:
                for i in selection:
                    row = dict(run_id=run, variable=code, field=field,
                               pressure_hpa=pressure, unit=unit,
                               valid_time=str(times[i]),
                               origin_time=str(times[i] - np.timedelta64(6, 'h')),
                               physical_time_hours=int(offsets[i]))
                    for j, (key, *_) in enumerate(STYLES):
                        value = float(values[i, j, fields.index(field), levels.index(pressure)]) * scale
                        assert np.isfinite(value)
                        row[key] = format(value, '.12g')
                    rows.append(row)
        sources[run] = dict(checkpoint=report['checkpoint'], job_id=report['job_id'],
                            records=report['records'], report_sha256=sha(path),
                            timeseries_sha256=report['artifacts']['timeseries.npz'])
    write_csv(output / 'timeseries.csv', rows)
    provenance = dict(runs=sources, physical_summary_sha256=sha(summary_path),
                      physical_metrics_sha256=sha(PHYSICAL / 'physical_metrics_all_levels.csv'),
                      timeseries_sha256=sha(output / 'timeseries.csv'),
                      window_start=str(START), duration_hours=72,
                      forecast_lead_hours=6, statistic='global Gaussian-area mean',
                      selection='All positive full-year RMSE improvements at <=100 hPa for w128/di16; same field/levels shown for all four runs',
                      role='Existing six-hour v2-trained K=1 forecasts; no retraining or new GPU evaluation')
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


def draw_and_document(output):
    provenance = json.loads((output / 'provenance.json').read_text())
    assert sha(output / 'timeseries.csv') == provenance['timeseries_sha256']
    metrics_path = PHYSICAL / 'physical_metrics_all_levels.csv'
    assert sha(metrics_path) == provenance['physical_metrics_sha256']
    with metrics_path.open() as stream:
        metrics = list(csv.DictReader(stream))
    with (output / 'timeseries.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    selected = {(r['field'], int(float(r['pressure_hpa']))) for r in metrics
                if r['run_id'] == REPRESENTATIVE and float(r['pressure_hpa']) <= 100
                and float(r['rmse_improvement_pct']) > 0}
    assert selected == {(s[1], s[2]) for s in SPECS}, 'Update the documented selection'
    lookup = {(r['run_id'], r['field'], int(float(r['pressure_hpa']))): r for r in metrics}
    runs = provenance['runs']
    with tempfile.TemporaryDirectory(prefix='upper-air-mpl-') as cache:
        os.environ.setdefault('MPLCONFIGDIR', cache)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator
        plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10,
                             'axes.spines.top': False, 'axes.spines.right': False,
                             'pdf.fonttype': 42, 'svg.fonttype': 'none',
                             'svg.hashsalt': 'ngcm-upper-air-physical'})

        def save(fig, stem):
            for ext in ('png', 'pdf', 'svg'):
                metadata = {'CreationDate': None} if ext == 'pdf' else {'Date': None} if ext == 'svg' else None
                fig.savefig(str(stem) + '.' + ext, dpi=160, facecolor='white', metadata=metadata)
                if ext == 'svg':
                    path = Path(str(stem) + '.svg')
                    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
            plt.close(fig)

        def panel(ax, run, spec):
            code, field, pressure, _, title, unit = spec
            points = sorted((r for r in rows if r['run_id'] == run and r['variable'] == code),
                            key=lambda r: int(r['physical_time_hours']))
            assert len(points) == 12
            for key, label, color, style, marker in STYLES:
                ax.plot([int(r['physical_time_hours']) for r in points],
                        [float(r[key]) for r in points], label=label, color=color,
                        linestyle=style, marker=marker, markersize=3.5,
                        markerfacecolor='white', linewidth=1.3)
            gain = float(lookup[run, field, pressure]['rmse_improvement_pct'])
            ax.set_title(f'{title}, {pressure} hPa\nFull-year spatial RMSE reduction {gain:.1f}%', fontsize=10)
            ax.set_ylabel(unit)
            ax.set_xlabel('Physical time since 2022-01-10 00 UTC (h)', fontsize=9)
            ax.set_xlim(0, 73)
            ax.xaxis.set_major_locator(MultipleLocator(6))
            ax.tick_params(axis='x', labelsize=8)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 4), useOffset=False)
            ax.grid(color='#e7e7e7', linewidth=0.5)

        for run, source in runs.items():
            folder = output / run
            folder.mkdir(exist_ok=True)
            fig, axes = plt.subplots(6, 2, figsize=(12, 16), constrained_layout=True)
            for ax, spec in zip(axes.flat, SPECS):
                panel(ax, run, spec)
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False)
            fig.suptitle(f'{run}, update {source["checkpoint"]["update"]}\nGlobal means; successive K=1 forecasts, fixed six-hour lead', fontsize=12)
            save(fig, folder / 'upper_air_improvements')
            for spec in SPECS[8:]:
                fig, ax = plt.subplots(figsize=(9, 3.8), constrained_layout=True)
                panel(ax, run, spec)
                ax.legend(ncol=3, frameon=False, fontsize=9)
                save(fig, folder / f'{spec[0]}_global_timeseries')

    lines = ['# Upper-air physical variables with improved RMSE', '',
             '**Geopotential improves at 1, 2, 3, 5, 7, 10, 20 and 30 hPa; specific',
             'humidity improves at 1, 2, 3 and 5 hPa.** Both are displayed below.', '',
             'Selection: these are all field/level pairs at **100 hPa or lower pressure**',
             'with positive full-year physical RMSE improvement for w128/di16.',
             'The same twelve pairs are shown for all four configurations, and all improve.',
             'This is an explicitly selected view of favorable results, not an all-variable',
             'skill summary. [Standard-level results and all-level errors](../k1_physical_eval_20260924/README.md)',
             'remain available, including deteriorations.', '',
             '## Physical time and evaluation meaning', '',
             'Every panel compares **ERA5 truth (black), frozen NGCM (blue), and residual',
             'NGCM (orange)** in physical units. The curves show global Gaussian-area',
             'means from 2022-01-10 06 UTC through January 13 00 UTC. The x-axis is',
             'elapsed **physical valid time**, at 6, 12, ..., 72 h from January 10 00 UTC.',
             'Each point is a **six-hour K=1 forecast initialized from ERA5 at the preceding',
             'origin**, with chronological recurrent memory. This is not a free-running',
             '72-hour rollout. All displayed models were trained with the 6 h v2 loss.', '',
             'The RMSE reductions use **all 1,455 six-hour 2022 validation forecasts**,',
             'averaging squared errors over grid cells and dates before taking the square',
             'root. They are not errors of the global-mean curves or scores from this',
             'three-day window alone. Percentages here are **RMSE**, not MSE, reductions.', '',
             '## Representative comparison: w128/di16, update 2,968', '',
             f'![Upper-air geopotential and humidity comparison]({REPRESENTATIVE}/upper_air_improvements.png)', '',
             '| Variable | Pressure (hPa) | Unit | Baseline RMSE | Residual RMSE | RMSE reduction |',
             '| --- | ---: | --- | ---: | ---: | ---: |']
    for code, field, pressure, scale, title, unit in SPECS:
        row = lookup[REPRESENTATIVE, field, pressure]
        lines.append(f'| {title} | {pressure} | {unit} | {float(row["baseline_rmse"])*scale:.6g} | {float(row["residual_rmse"])*scale:.6g} | {float(row["rmse_improvement_pct"]):.2f}% |')
    lines += ['', '## Individual physical-time comparisons', '']
    for code, field, pressure, _, title, unit in SPECS:
        stem = (f'../k1_physical_eval_20260924/{REPRESENTATIVE}/{code}_global_timeseries'
                if field == 'geopotential' else f'{REPRESENTATIVE}/{code}_global_timeseries')
        lines += [f'### {title} at {pressure} hPa ({unit})', '',
                  f'![{code}: ERA5, baseline and residual versus physical time]({stem}.png)', '',
                  f'[PDF]({stem}.pdf) · [SVG]({stem}.svg)', '']
    lines += ['## Four-configuration RMSE reductions', '',
              '| Variable / pressure | ' + ' | '.join(run.replace('r2p8_', '') for run in runs) + ' |',
              '| --- | ' + ' | '.join('---:' for _ in runs) + ' |']
    for code, field, pressure, *_ in SPECS:
        gains = [float(lookup[run, field, pressure]['rmse_improvement_pct']) for run in runs]
        assert min(gains) > 0
        lines.append(f'| {code} | ' + ' | '.join(f'{gain:.2f}%' for gain in gains) + ' |')
    lines += ['', 'The w128 checkpoints are at update 2,968; the w256 checkpoints are at',
              'update 2,120. These comparisons do not imply matched training budgets.', '']
    for run, source in runs.items():
        lines += [f'### {run}, update {source["checkpoint"]["update"]}', '',
                  f'[PNG]({run}/upper_air_improvements.png) · [PDF]({run}/upper_air_improvements.pdf) · [SVG]({run}/upper_air_improvements.svg)', '']
        if run != REPRESENTATIVE:
            lines += [f'![{run} upper-air comparisons]({run}/upper_air_improvements.png)', '']
    lines += ['## Data and reproduction', '',
              '[Physical-time CSV](timeseries.csv) · [Checkpoint and source hashes](provenance.json) ·',
              '[All-variable, all-level physical metrics](../k1_physical_eval_20260924/physical_metrics_all_levels.csv)', '',
              'The new humidity curves are exported from the existing verified full-year',
              'evaluation arrays. No additional GPU evaluation or training was required.',
              'The geopotential individual figures reuse the earlier published physical curves.', '',
              'Redraw from committed CSV data:', '', '```bash',
              'python plot/plot_upper_air_physical.py', '```', '',
              'Refresh from the original evaluation artifacts:', '', '```bash',
              'python plot/plot_upper_air_physical.py --results-root logs/neuralgcm_physical_eval_20260924',
              '```', '']
    (output / 'README.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'upper_air_physical_eval_20260924')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.results_root:
        export(args.results_root, args.output_dir)
    draw_and_document(args.output_dir)


if __name__ == '__main__':
    main()
