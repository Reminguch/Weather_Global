#!/usr/bin/env python3
"""Plot upper-air physical time series from the saved paired K1/K2 evaluation."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

LEVELS = (1, 2, 3, 5, 7, 10, 20, 30)
HORIZONS = ((1, 6), (2, 6), (2, 12))
FIELDS = (
    ('geopotential', 'Geopotential', 1., r'm$^2$ s$^{-2}$'),
    ('temperature', 'Temperature', 1., 'K'),
    ('u_component_of_wind', 'Eastward wind', 1., r'm s$^{-1}$'),
    ('v_component_of_wind', 'Northward wind', 1., r'm s$^{-1}$'),
    ('specific_humidity', 'Specific humidity', 1e6, r'mg kg$^{-1}$'),
    ('specific_cloud_ice_water_content', 'Cloud ice', 1e6, r'mg kg$^{-1}$'),
    ('specific_cloud_liquid_water_content', 'Cloud liquid water', 1e6, r'mg kg$^{-1}$'),
)
SERIES = (('era5', 'ERA5 truth', '#222222', '-', 'o'),
          ('baseline', 'Frozen NGCM', '#0072B2', '--', 's'),
          ('residual', 'Residual NGCM', '#D55E00', '-.', '^'))


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def retained(row):
    pressure = float(row['pressure_hpa'])
    return pressure in LEVELS or (row['field'] == 'geopotential' and pressure == 500)


def load(source):
    receipt = json.loads((source / 'provenance.json').read_text())
    for name, expected in receipt['files'].items():
        if checksum(source / name) != expected:
            raise ValueError(f'Input checksum mismatch: {name}')
    times = [r for r in read_csv(source / 'timeseries.csv') if retained(r)]
    errors = [r for r in read_csv(source / 'physical_errors.csv') if retained(r)]
    summary = [r for r in read_csv(source / 'physical_rmse.csv') if retained(r)]
    plan = receipt['plan']
    index, error_index, summary_index = {}, {}, {}
    reference = {}
    for row in times:
        key = (row['run_id'], int(row['lead_hours']), row['field'],
               int(float(row['pressure_hpa'])), row['location'])
        index.setdefault(key, []).append(row)
        choice = plan['runs'][row['run_id']]
        if int(row['checkpoint_update']) != choice['update']:
            raise ValueError('Checkpoint update does not match the saved selection')
        valid = np.datetime64(row['origin'], 'h') + np.timedelta64(int(row['lead_hours']), 'h')
        hours = int((valid - np.datetime64(plan['origins'][0], 'h')) / np.timedelta64(1, 'h'))
        if valid != np.datetime64(row['valid_time'], 'h') or hours != int(row['physical_hours']):
            raise ValueError('Incorrect physical valid time')
        pair = [float(row[k]) for k in ('era5', 'baseline')]
        shared_key = tuple(row[k] for k in ('origin', 'lead_hours', 'field', 'pressure_hpa', 'location'))
        if shared_key in reference:
            np.testing.assert_array_equal(pair, reference[shared_key])
        reference[shared_key] = pair
    for row in errors:
        key = (row['run_id'], int(row['lead_hours']), row['field'], int(float(row['pressure_hpa'])))
        error_index.setdefault(key, []).append(row)
    for row in summary:
        key = (row['run_id'], int(row['lead_hours']), row['field'], int(float(row['pressure_hpa'])))
        summary_index[key] = row
    for key, rows in index.items():
        rows.sort(key=lambda r: int(r['physical_hours']))
        if len(rows) != 12 or {r['origin'] for r in rows} != set(plan['origins']):
            raise ValueError(f'Incomplete time series: {key}')
        if not np.isfinite([[float(r[c]) for c in ('era5', 'baseline', 'residual')] for r in rows]).all():
            raise ValueError('Nonfinite time series')
    for key, rows in error_index.items():
        rows.sort(key=lambda r: r['valid_time'])
        if len(rows) != 12 or {r['origin'] for r in rows} != set(plan['origins']):
            raise ValueError(f'Incomplete errors: {key}')
        for model in ('baseline', 'residual'):
            mse = np.array([float(r[f'{model}_mse']) for r in rows])
            if not np.isfinite(mse).all() or (mse < 0).any():
                raise ValueError('Invalid MSE')
            np.testing.assert_allclose(np.sqrt(mse.mean()),
                float(summary_index[key][f'{model}_rmse']), rtol=1e-12)
    for run_id, choice in plan['runs'].items():
        for lead in range(6, 6 * choice['K'] + 1, 6):
            for field, *_ in FIELDS:
                levels = LEVELS + (500,) if field == 'geopotential' else LEVELS
                for level in levels:
                    key = (run_id, lead, field, level)
                    assert key in error_index and key in summary_index
                    for location in ('Global mean', 'New York', 'Beijing'):
                        assert key + (location,) in index
    return receipt, times, errors, summary, index, error_index, summary_index


def loss_audit(root):
    """Explain loss decay on its own 16-origin validation, not the plotted window."""
    statistics = json.loads((root / 'shared_statistics_cpu_v1/statistics.json').read_text())
    folder = root / 'train_k1_w128_d16_slices_v1'
    baseline = json.loads((folder / 'baseline_validation.json').read_text())
    selected = next(json.loads(line) for line in (folder / 'validation.jsonl').read_text().splitlines()
                    if json.loads(line)['update'] == 900)
    levels = baseline['pressure_hpa']
    contributions, physical = [], []
    upper = [0., 0.]
    for field, *_ in FIELDS:
        amp = 2 if field == 'geopotential' else .66 if field == 'specific_humidity' else .05 if field.startswith('specific_cloud_') else 1
        scale = np.asarray(statistics['scales']['data'][field])
        values = []
        for i, row in enumerate((baseline, selected)):
            cells = 20 * .8 * (np.asarray(row['physical_rmse'][field][0]) * amp / scale) ** 2 / 37
            values.append(float(cells.sum()))
            if field == 'geopotential':
                upper[i] = float(cells[np.asarray(levels) <= 30].sum())
        improved = [lev for lev, a, b in zip(levels, baseline['physical_rmse'][field][0],
                                              selected['physical_rmse'][field][0]) if b < a]
        contributions.append(dict(field=field, baseline=values[0], residual=values[1],
                                  improved_levels_hpa=improved))
    for field, level in (('geopotential', 1), ('geopotential', 500), ('temperature', 850)):
        j = levels.index(level)
        physical.append(dict(field=field, pressure_hpa=level,
            baseline_rmse=baseline['physical_rmse'][field][0][j],
            residual_rmse=selected['physical_rmse'][field][0][j]))
    return dict(scope='K1/w128 update 900; original 16 fixed validation origins, not the plotted 12-origin window',
        definition='Unfiltered nodal analogue of 20*M_data; NOT the exact modal five-term loss',
        exact_loss=dict(baseline=baseline['loss'], residual=selected['loss']),
        exact_terms=dict(baseline=baseline['terms'], residual=selected['terms']),
        contributions=contributions, upper_geopotential=dict(baseline=upper[0], residual=upper[1]),
        physical_examples=physical,
        source_hashes={str(p): checksum(p) for p in (folder / 'baseline_validation.json',
            folder / 'validation.jsonl', root / 'shared_statistics_cpu_v1/statistics.json')})


def draw(output, receipt, index, error_index, summary_index):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'upper-air-physical-time'})
    plan = receipt['plan']
    saved = []

    def save(fig, stem):
        for ext in ('png', 'pdf', 'svg'):
            path = output / f'{stem}.{ext}'
            fig.savefig(path, dpi=165, bbox_inches='tight', facecolor='white')
            if ext == 'svg':
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
        plt.close(fig)
        saved.append(stem)
        print(f'Saved {stem}', flush=True)

    def axes_style(ax):
        ax.set_xlim(0, 78)
        ax.set_xticks([0, 12, 24, 36, 48, 60, 72, 78])
        ax.grid(alpha=.2)
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_powerlimits((-3, 4))
        ax.yaxis.set_major_formatter(formatter)

    def values(ax, key, scale):
        rows = index[key]
        x = [int(r['physical_hours']) for r in rows]
        for model, label, color, style, marker in SERIES:
            ax.plot(x, [float(r[model]) * scale for r in rows], label=label,
                    color=color, ls=style, marker=marker, ms=2.5, lw=1.4)

    def errors(ax, key, scale):
        rows = error_index[key]
        x = [int((np.datetime64(r['valid_time'], 'h') - np.datetime64(plan['origins'][0], 'h'))
                 / np.timedelta64(1, 'h')) for r in rows]
        for model, label, color, style, marker in SERIES[1:]:
            ax.plot(x, [np.sqrt(float(r[f'{model}_mse'])) * scale for r in rows],
                    label=label, color=color, ls=style, marker=marker, ms=2.5, lw=1.4)
        ax.set_ylim(bottom=0)

    for width in (128, 256):
        for field, title, scale, unit in FIELDS:
            levels = LEVELS + (500,) if field == 'geopotential' else LEVELS
            modes = [('Global mean', 'global', False)]
            if field == 'geopotential':
                modes += [('New York', 'new_york', False), ('Beijing', 'beijing', False),
                          ('Global gridpoint RMSE', 'rmse', True)]
            for location, slug, is_error in modes:
                fig, axes = plt.subplots(len(levels), 3, figsize=(14.5, 2.05 * len(levels) + 1.4),
                                         sharex=True, sharey='row')
                fig.subplots_adjust(left=.08, right=.99, bottom=.055, top=.905, hspace=.46, wspace=.18)
                for i, level in enumerate(levels):
                    for j, (k, lead) in enumerate(HORIZONS):
                        ax = axes[i, j]
                        key = (f'k{k}_w{width}_d16', lead, field, level)
                        if is_error:
                            errors(ax, key, scale)
                        else:
                            values(ax, key + (location,), scale)
                        axes_style(ax)
                        if j == 0:
                            ax.set_ylabel(f'{level} hPa\n{unit}')
                        if i == 0:
                            update = plan['runs'][key[0]]['update']
                            ax.set_title(f'K={k} | +{lead} h forecast | checkpoint {update}', pad=18, fontsize=10)
                        if is_error:
                            change = float(summary_index[key]['rmse_reduction_pct'])
                            text = f'Window RMSE {abs(change):.1f}% ' + ('lower' if change >= 0 else 'higher')
                            ax.text(.02, .96, text, transform=ax.transAxes, va='top', fontsize=8,
                                    bbox=dict(facecolor='white', edgecolor='none', alpha=.8))
                handles, labels = axes[0, 0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .948), ncol=3, frameon=False)
                fig.suptitle(f'{title} versus forecast valid time | w{width}, d16 | {location}\n'
                             '1-30 hPa' + ('; 500 hPa included for comparison' if field == 'geopotential' else ''),
                             y=.984, fontsize=14)
                fig.supxlabel('Forecast valid time (hours since 2022-01-10 00 UTC)\n'
                              '12 independent forecast origins; each origin restarts from ERA5', y=.012, fontsize=10)
                save(fig, f'w{width}_{field}_{slug}')

    fig, axes = plt.subplots(4, 2, figsize=(13, 11.5), sharex=True)
    fig.subplots_adjust(top=.86, bottom=.09, left=.1, right=.985, hspace=.43, wspace=.24)
    for i, level in enumerate((1, 10, 30, 500)):
        left, right = axes[i]
        key = ('k1_w128_d16', 6, 'geopotential', level)
        rows = index[key + ('Global mean',)]
        x = [int(r['physical_hours']) for r in rows]
        for model, label, color, style, marker in SERIES[:2]:
            left.plot(x, [float(r[model]) for r in rows], color=color, ls=style, marker=marker,
                      ms=3, lw=1.5, label=label)
        right.plot(x, [np.sqrt(float(r['baseline_mse'])) for r in error_index[key]],
                   color='#0072B2', ls='--', marker='s', ms=3, lw=1.5, label='Frozen NGCM')
        for width, color, marker in ((128, '#D55E00', '^'), (256, '#7B4AB5', 'D')):
            run_key = (f'k1_w{width}_d16', 6, 'geopotential', level)
            left.plot(x, [float(r['residual']) for r in index[run_key + ('Global mean',)]],
                      color=color, marker=marker, ms=3, lw=1.4, label=f'Residual w{width}')
            right.plot(x, [np.sqrt(float(r['residual_mse'])) for r in error_index[run_key]],
                       color=color, marker=marker, ms=3, lw=1.4, label=f'Residual w{width}')
        for ax in (left, right):
            axes_style(ax)
            ax.set_ylabel(f'{level} hPa\n' + r'm$^2$ s$^{-2}$')
        right.set_ylim(bottom=0)
        if i == 0:
            left.set_title('Physical geopotential: global area mean', pad=18)
            right.set_title('Gridpoint RMSE at each valid time: lower is better', pad=18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .939), ncol=4, frameon=False)
    fig.suptitle('Where geopotential improves, and where it worsens\n'
                 'K=1 | +6 h forecasts | w128 and w256 | both checkpoints at update 900',
                 fontsize=15, y=.991)
    fig.supxlabel('Forecast valid time (hours since 2022-01-10 00 UTC)\n'
                  'Successive short forecasts from 12 independent origins, not one continuous rollout',
                  fontsize=10, y=.015)
    save(fig, 'geopotential_overview')
    return saved


def report(output, receipt, metrics, audit):
    lines = ['# Upper-air physical variables versus physical time', '',
        'This report uses the same saved 12-origin paired evaluation as the earlier physical-time report.',
        'It includes **all eight levels at 1, 2, 3, 5, 7, 10, 20 and 30 hPa**, plus 500 hPa geopotential for comparison.',
        'The horizontal axis is **forecast valid time**, in hours since **2022-01-10 00 UTC**.', '',
        '![Geopotential time series and errors](geopotential_overview.png)', '',
        '[Overview PDF](geopotential_overview.pdf) · [Editable SVG](geopotential_overview.svg) ·',
        '**[中文分析：高层改善、偏差分解与 loss 下降原因](ANALYSIS.md)**', '',
        'Left: physical geopotential, with ERA5 truth, frozen NeuralGCM and both residual widths.',
        'Right: Gaussian-area gridpoint RMSE at each time, computed before spatial averaging.',
        'RMSE is not the error of the plotted global mean. Raw values are not clipped or shifted.', '',
        '## What the saved evaluation shows', '',
        'Upper-level geopotential improves while 500 hPa geopotential worsens. The earlier seven headline',
        'variable/level pairs did not include these upper levels. Their degradation does not mean that',
        'every variable at every pressure level became worse.', '',
        '| Level | Frozen +6 h | K1 w128 | K1 w256 | K2 w128 +6 h | K2 w256 +6 h | Frozen +12 h | K2 w128 +12 h | K2 w256 +12 h |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for level in LEVELS + (500,):
        a = metrics[('k1_w128_d16', 6, 'geopotential', level)]
        b = metrics[('k2_w128_d16', 12, 'geopotential', level)]
        values = [float(a['baseline_rmse'])]
        values += [float(metrics[(run, lead, 'geopotential', level)]['residual_rmse']) for run, lead in
                   (('k1_w128_d16', 6), ('k1_w256_d16', 6), ('k2_w128_d16', 6), ('k2_w256_d16', 6))]
        values += [float(b['baseline_rmse'])]
        values += [float(metrics[(run, 12, 'geopotential', level)]['residual_rmse']) for run in
                   ('k2_w128_d16', 'k2_w256_d16')]
        lines.append(f'| {level} hPa | ' + ' | '.join(f'{v:,.1f}' for v in values) + ' |')
    lines += ['', 'All table entries are window gridpoint RMSE in m²/s², lower is better.', '',
        '## All upper-level time series', '',
        'Each figure has one row per pressure level and three columns: K1/+6 h, K2/+6 h, K2/+12 h.',
        'Black is ERA5, blue is frozen NGCM, orange is the residual model.', '',
        '| Variable | w128 global mean | w256 global mean |', '| --- | --- | --- |']
    for field, title, *_ in FIELDS:
        lines.append(f'| {title} | [Figure](w128_{field}_global.png) | [Figure](w256_{field}_global.png) |')
    lines += ['', '| Geopotential detail | w128 | w256 |', '| --- | --- | --- |']
    for title, slug in (('New York gridpoint', 'new_york'), ('Beijing gridpoint', 'beijing'),
                        ('Global gridpoint RMSE over time', 'rmse')):
        lines.append(f'| {title} | [Figure](w128_geopotential_{slug}.png) | [Figure](w256_geopotential_{slug}.png) |')
    lines += ['', 'Every figure is also available as PDF and editable SVG with the same filename stem.', '',
        '## How many upper-level pairs improve?', '',
        'Counts below are across the eight levels, using window gridpoint RMSE, not global means.', '',
        '| Variable | K1 w128 +6 h | K1 w256 +6 h | K2 w128 +6 h | K2 w256 +6 h | K2 w128 +12 h | K2 w256 +12 h |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    combinations = (('k1_w128_d16', 6), ('k1_w256_d16', 6), ('k2_w128_d16', 6),
                    ('k2_w256_d16', 6), ('k2_w128_d16', 12), ('k2_w256_d16', 12))
    for field, title, *_ in FIELDS:
        counts = [sum(float(metrics[(run, lead, field, level)]['rmse_reduction_pct']) > 0 for level in LEVELS)
                  for run, lead in combinations]
        lines.append(f'| {title} | ' + ' | '.join(f'{n}/8' for n in counts) + ' |')
    if audit:
        lines += ['', '## Why the training objective can decay', '',
            'This separate diagnostic uses **K1/w128 at the selected update 900 on its original 16 validation origins**.',
            'It is not computed on the 12-origin plotting window above.', '',
            f"The exact logged five-term objective falls from **{audit['exact_loss']['baseline']:.3f}** to **{audit['exact_loss']['residual']:.3f}**.",
            'The following contributions reconstruct the normalized data-MSE component from physical RMSE.',
            '**They omit modal projection and filtering and therefore are not an exact decomposition of the full loss.**', '',
            '| Variable | Baseline contribution | Residual contribution |', '| --- | ---: | ---: |']
        for row in audit['contributions']:
            lines.append(f"| {row['field']} | {row['baseline']:.5f} | {row['residual']:.5f} |")
        base = sum(r['baseline'] for r in audit['contributions'])
        res = sum(r['residual'] for r in audit['contributions'])
        share = 100 * audit['upper_geopotential']['baseline'] / base
        lines += [f'| Total diagnostic | {base:.5f} | {res:.5f} |', '',
            f"The 1–30 hPa geopotential contribution alone falls from **{audit['upper_geopotential']['baseline']:.3f}** to **{audit['upper_geopotential']['residual']:.3f}**.",
            f'It accounts for **{share:.2f}%** of the baseline data-MSE diagnostic. This dominates the total even while other terms increase.',
            'Thus the loss decline is real in this objective, but it does not establish broad forecast improvement.',
            'This arithmetic does not establish why the baseline upper-level errors are large or validate the reconstructed objective.', '',
            '[Audit data and exact logged terms](loss_decay_audit.json)']
    lines += ['', '## Checkpoints and interpretation', '',
        '| Run | Selected checkpoint update | Completed training stage |', '| --- | ---: | ---: |']
    for name, choice in receipt['plan']['runs'].items():
        lines.append(f"| {name} | {choice['update']} | 1000 |")
    lines += ['', 'Best validation checkpoints were selected before the physical-time evaluation.',
        'There are 12 origins, six hours apart, from January 10 00 UTC through January 12 18 UTC, 2022.',
        'State and residual memory restart at each origin. K2 uses corrected-state feedback between its two leads.',
        '**These are consecutive short forecasts, not one continuous multi-day rollout.**',
        'Global means may conceal local errors; the geopotential RMSE and city panels provide additional checks.',
        'This small window does not establish full-year skill or reproduce the original paper benchmark.', '',
        'Saved input hashes, checkpoint choices, paired baselines, valid times, complete origin coverage, and',
        'agreement between per-origin errors and window RMSE are checked before plotting. No new GPU run is needed.', '',
        '[Time series](timeseries.csv) · [Per-origin gridpoint errors](physical_errors.csv) ·',
        '[Window RMSE](physical_rmse.csv) · [Provenance](provenance.json)', '',
        'Reproduce from the repository root:', '', '```bash',
        'python3 plot/plot_upper_air_physical_time.py', '```', '',
        '[Earlier headline-level plots](../paper_physical_time_20260924/README.md)', '']
    (output / 'README.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    plot_root = Path(__file__).resolve().parent
    parser.add_argument('--source', type=Path, default=plot_root / 'paper_physical_time_20260924')
    parser.add_argument('--output', type=Path, default=plot_root / 'upper_air_physical_time_20260924')
    parser.add_argument('--report-only', action='store_true', help='Recheck inputs and refresh tables without redrawing existing figures')
    args = parser.parse_args()
    receipt, times, errors, summary, index, error_index, metrics = load(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (('timeseries.csv', times), ('physical_errors.csv', errors), ('physical_rmse.csv', summary)):
        write_csv(args.output / name, rows)
    audit_path = args.output / 'loss_decay_audit.json'
    # Freeze this separate diagnostic on first export, then reproduce without live logs.
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
    else:
        logs = plot_root.parent / 'logs/ngcm_aligned_20260924'
        audit = loss_audit(logs) if logs.exists() else None
        if audit:
            audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    if args.report_only:
        figures = json.loads((args.output / 'provenance.json').read_text())['figures']
        for stem in figures:
            for ext in ('png', 'pdf', 'svg'):
                if not (args.output / f'{stem}.{ext}').is_file():
                    raise ValueError(f'Missing existing figure: {stem}.{ext}')
    else:
        with tempfile.TemporaryDirectory(prefix='ngcm-upper-air-mpl-') as cache:
            os.environ['MPLCONFIGDIR'] = cache
            figures = draw(args.output, receipt, index, error_index, metrics)
    report(args.output, receipt, metrics, audit)
    provenance = dict(exported_at=datetime.now(timezone.utc).isoformat(),
        source_directory=str(args.source), source_provenance_sha256=checksum(args.source / 'provenance.json'),
        source_files=receipt['files'], plan=receipt['plan'], upper_pressure_hpa=list(LEVELS),
        comparison_pressure_hpa=500, figures=figures,
        checks=dict(input_hashes=True, paired_baselines=True, valid_times=True, origin_coverage=True,
                    checkpoint_updates=True, rmse_reconstruction=True),
        files={name: checksum(args.output / name) for name in
               ('timeseries.csv', 'physical_errors.csv', 'physical_rmse.csv')},
        plotting_script_sha256=checksum(__file__))
    if audit:
        provenance['loss_decay_audit_sha256'] = checksum(audit_path)
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(figures=len(figures), checks=provenance['checks'], output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
