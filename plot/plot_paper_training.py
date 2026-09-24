#!/usr/bin/env python3
"""Export and plot the actual five-term training logs and paired validation.

No model replay or training mutation. --refresh snapshots complete JSONL records;
without it, figures are reproduced exclusively from the published snapshot.json.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/ngcm-paper-plot-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['geopotential', 'temperature', 'u_component_of_wind',
          'v_component_of_wind', 'specific_humidity',
          'specific_cloud_ice_water_content', 'specific_cloud_liquid_water_content']
LABELS = ['Geopotential', 'Temperature', 'Zonal wind', 'Meridional wind',
          'Specific humidity', 'Cloud ice', 'Cloud liquid']
TERMS = ['data', 'model', 'data_spectrum', 'model_spectrum', 'bias']
TERM_LABELS = ['20 × data MSE', 'Model MSE', '0.1 × data spectrum',
               '0.1 × model spectrum', '2 × bias']
COLORS = {128: '#0072B2', 256: '#D55E00'}
COEFFICIENTS = dict(data=20., model=1., data_spectrum=.1, model_spectrum=.1, bias=2.)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def snapshot(root):
    result = dict(captured_at_utc=datetime.now(timezone.utc).isoformat(),
                  experiment='ngcm_aligned_20260924/source_transfer_v2',
                  scope='16 fixed 2022 validation origins; not full-year or 2023 test',
                  sources={}, runs=[])

    def read(path, jsonl=False):
        raw = path.read_bytes()
        used = raw
        if jsonl and raw and not raw.endswith(b'\n'):
            used = raw[:raw.rfind(b'\n') + 1]
        result['sources'][str(path.relative_to(ROOT))] = dict(
            sha256=sha(raw), bytes=len(raw), complete_bytes=len(used))
        return [json.loads(x) for x in used.splitlines()] if jsonl else json.loads(raw)

    result['statistics'] = read(root / 'shared_statistics_cpu_v1/statistics.json')
    for k, width in [(1, 128), (1, 256), (2, 128), (2, 256)]:
        path = root / f'train_k{k}_w{width}_d16_slices_v1'
        run = dict(K=k, width=width, d_inner=16, training=[], validation=[])
        if not (path / 'baseline_validation.json').exists():
            run['status'] = 'No completed validation available at snapshot'
            result['runs'].append(run)
            continue
        run['config'] = read(path / 'config.json')
        run['baseline'] = read(path / 'baseline_validation.json')
        run['training'] = read(path / 'metrics.jsonl', True)
        run['validation'] = read(path / 'validation.jsonl', True) if (path / 'validation.jsonl').exists() else []
        run['milestone_1000'] = (path / 'MILESTONE_1000.json').exists()
        run['status'] = ('Stopped at 1000-update review' if run['milestone_1000']
                         else 'Partial run; latest completed validation shown')
        result['runs'].append(run)
    return result


def check(data):
    assert data['statistics']['lag_hours'] == 24 and data['statistics']['split'] == 'train'
    paired = {}
    for run in data['runs']:
        if 'baseline' not in run:
            continue
        cfg, baseline = run['config'], run['baseline']
        assert cfg['K'] == cfg['eval_K'] == run['K']
        assert cfg['architecture']['d_inner'] == 16
        assert cfg['objective']['coefficients'] == COEFFICIENTS
        assert baseline['lead_hours'] == list(range(6, 6 * run['K'] + 1, 6))
        assert len(baseline['origins']) == 16 and len(baseline['pressure_hpa']) == 37
        if run['K'] in paired:
            reference = paired[run['K']]
            assert reference['origins'] == baseline['origins']
            assert reference['physical_rmse'] == baseline['physical_rmse']
            assert reference['loss'] == baseline['loss']
        paired[run['K']] = baseline
        for rows in [run['training'], run['validation']]:
            steps = [row['update'] for row in rows]
            assert steps == sorted(set(steps)) and all(0 < x <= 1000 for x in steps)
            for row in rows:
                assert row['K'] == run['K']
                np.testing.assert_allclose(sum(row['terms'].values()), row['loss'], rtol=3e-6)
                if 'fields' in row:
                    np.testing.assert_allclose(sum(row['fields'].values()), row['loss'], rtol=3e-6)
        for row in [baseline] + run['validation']:
            for field in FIELDS:
                values = np.asarray(row['physical_rmse'][field])
                assert values.shape == (run['K'], 37)
                assert np.isfinite(values).all() and (values >= 0).all()
        if run['validation']:
            assert run['validation'][-1]['update'] <= run['training'][-1]['update']


def diagnostic(row, run, statistics):
    """Unfiltered nodal analogue of 20*M_data, not the exact modal objective."""
    levels = np.asarray(run['baseline']['pressure_hpa'])
    time = 1 / (1 + np.asarray(run['baseline']['lead_hours']) / 24)
    contribution = {}
    cells = {}
    for field in FIELDS:
        amp = 2 if field == 'geopotential' else .66 if field == 'specific_humidity' else .05 if field.startswith('specific_cloud_') else 1
        scale = np.asarray(statistics['scales']['data'][field])
        values = np.asarray(row['physical_rmse'][field]) ** 2 * (amp / scale) ** 2 * time[:, None] * 20
        cells[field] = values / values.size
        contribution[field] = float(values.mean())
    total = sum(contribution.values())
    return dict(total=total, by_field=contribution,
                geo_share_pct=100 * contribution['geopotential'] / total,
                upper_geo_share_pct=100 * float(cells['geopotential'][:, levels <= 30].sum()) / total,
                exact_logged_modal_data_term=row['terms']['data'])


def export_csv(data, out):
    with (out / 'validation_by_step.csv').open('w') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['K', 'width', 'update', 'loss', 'baseline_loss', 'objective_reduction_pct'] + TERMS)
        for run in data['runs']:
            if 'baseline' not in run:
                continue
            for row in [dict(run['baseline'], update=0)] + run['validation']:
                writer.writerow([run['K'], run['width'], row['update'], row['loss'], run['baseline']['loss'],
                                 100 * (1 - row['loss'] / run['baseline']['loss'])] + [row['terms'][t] for t in TERMS])
    with (out / 'physical_rmse_by_step.csv').open('w') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['K', 'width', 'update', 'lead_hours', 'pressure_hpa', 'field', 'rmse', 'baseline_rmse', 'rmse_ratio'])
        for run in data['runs']:
            if 'baseline' not in run:
                continue
            for row in [dict(run['baseline'], update=0)] + run['validation']:
                for field in FIELDS:
                    for i, lead in enumerate(run['baseline']['lead_hours']):
                        for j, level in enumerate(run['baseline']['pressure_hpa']):
                            base = run['baseline']['physical_rmse'][field][i][j]
                            value = row['physical_rmse'][field][i][j]
                            writer.writerow([run['K'], run['width'], row['update'], lead, level, field, value, base, value / base])


def finish(fig, out, name):
    for suffix in ['png', 'pdf', 'svg']:
        path = out / f'{name}.{suffix}'
        fig.savefig(path, dpi=180, bbox_inches='tight')
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def style(ax, title=None):
    if title:
        ax.set_title(title)
    ax.grid(alpha=.18)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('Training step (optimizer updates)')
    ax.set_xlim(0, 1000)


def moving_mean(values, window=25):
    values = np.asarray(values)
    return np.array([values[max(0, i-window+1):i+1].mean() for i in range(len(values))])


def figures(data, out):
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    available = [r for r in data['runs'] if r['validation']]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for k in [1, 2]:
        left, right = axes[k-1]
        arms = [r for r in available if r['K'] == k]
        for run in arms:
            rows = [dict(run['baseline'], update=0)] + run['validation']
            x, y = [r['update'] for r in rows], [r['loss'] for r in rows]
            color = COLORS[run['width']]
            left.plot(x, y, '-o', ms=3, color=color, label=f"w{run['width']} validation")
            train = run['training']
            left.plot([r['update'] for r in train], moving_mean([r['loss'] for r in train]),
                      color=color, alpha=.35, lw=1, label=f"w{run['width']} train, mean 25")
            right.plot(x, 100 * (1 - np.asarray(y) / run['baseline']['loss']), '-o', ms=3,
                       color=color, label=f"w{run['width']}")
        if arms:
            left.axhline(arms[0]['baseline']['loss'], color='black', ls='--', label='Frozen NGCM validation')
        left.set_yscale('log'); left.set_ylabel('Five-term loss (log scale)')
        right.axhline(0, color='black', ls='--'); right.set_ylabel('Validation objective reduction (%)')
        right.set_ylim(-3, 103)
        for ax in [left, right]:
            style(ax, f'K={k}, matched train/eval ({"6 h" if k == 1 else "6 + 12 h"})')
            ax.legend(fontsize=8, loc='best')
    fig.suptitle('New five-term training: lower objective does not establish forecast improvement\n16 fixed 2022 validation origins; d_inner=16', fontsize=13)
    finish(fig, out, 'objective_by_training_step')

    fig, axes = plt.subplots(2, 5, figsize=(17, 7), layout='constrained')
    for k in [1, 2]:
        for j, term in enumerate(TERMS):
            ax = axes[k-1, j]
            for run in [r for r in available if r['K'] == k]:
                rows = [dict(run['baseline'], update=0)] + run['validation']
                ax.plot([r['update'] for r in rows], [r['terms'][term] for r in rows], '-o', ms=3,
                        color=COLORS[run['width']], label=f"w{run['width']}")
                ax.axhline(run['baseline']['terms'][term], color='black', ls='--', lw=.8)
            ax.set_yscale('log'); style(ax, f'K={k}: {TERM_LABELS[j]}')
            if j == 0:
                ax.set_ylabel('Weighted validation contribution'); ax.legend()
    fig.suptitle('Exact logged five-term validation contributions; dashed = matched frozen NGCM', fontsize=14)
    finish(fig, out, 'loss_terms_by_training_step')

    fig, axes = plt.subplots(2, 7, figsize=(20, 7), layout='constrained')
    for k in [1, 2]:
        for j, field in enumerate(FIELDS):
            ax = axes[k-1, j]
            for run in [r for r in available if r['K'] == k]:
                train = run['training']
                ax.plot([r['update'] for r in train], moving_mean([r['fields']['data/'+field] for r in train]),
                        color=COLORS[run['width']], label=f"w{run['width']}")
            ax.set_yscale('log'); style(ax, f'K={k}: {LABELS[j]}')
            if j == 0:
                ax.set_ylabel('Training contribution, mean 25'); ax.legend()
    fig.suptitle('Exact training field contributions: data MSE + data spectrum + bias\nNative model-space terms are additional; minibatch origins change at each update', fontsize=14)
    finish(fig, out, 'training_field_contributions')

    levels_to_show = [500, 850, 850, 850, 850, 250, 850]
    units = ['m²/s²', 'K', 'm/s', 'm/s', 'g/kg', 'mg/kg', 'mg/kg']
    factors = [1, 1, 1, 1, 1e3, 1e6, 1e6]
    fig, axes = plt.subplots(2, 7, figsize=(20, 7), layout='constrained')
    for k in [1, 2]:
        for j, (field, level) in enumerate(zip(FIELDS, levels_to_show)):
            ax = axes[k-1, j]
            for run in [r for r in available if r['K'] == k]:
                idx = run['baseline']['pressure_hpa'].index(level)
                rows = [dict(run['baseline'], update=0)] + run['validation']
                y = [np.sqrt(np.mean(np.asarray(r['physical_rmse'][field])[:, idx] ** 2)) * factors[j] for r in rows]
                ax.plot([r['update'] for r in rows], y, '-o', ms=3, color=COLORS[run['width']], label=f"w{run['width']}")
                ax.axhline(y[0], color='black', ls='--', lw=.8)
            style(ax, f'K={k}: {LABELS[j]} {level} hPa'); ax.set_ylabel(f'RMSE ({units[j]})')
            if j == 0:
                ax.legend()
    fig.suptitle('Physical forecast errors on the same 16 validation origins; lower is better\nK=2 averages squared errors over 6 and 12 h before taking the square root; dashed = frozen NGCM', fontsize=14)
    finish(fig, out, 'physical_rmse_by_training_step')

    fig, axes = plt.subplots(1, 4, figsize=(17, 10), layout='constrained')
    for ax, run in zip(axes, data['runs']):
        if not run['validation']:
            ax.text(.5, .5, 'No completed validation\nat snapshot', ha='center', transform=ax.transAxes)
            ax.set_title(f"K={run['K']}, w{run['width']}, d16"); ax.set_axis_off(); continue
        last = run['validation'][-1]
        ratios = []
        for field in FIELDS:
            numerator = np.sqrt(np.mean(np.asarray(last['physical_rmse'][field]) ** 2, axis=0))
            denominator = np.sqrt(np.mean(np.asarray(run['baseline']['physical_rmse'][field]) ** 2, axis=0))
            ratios.append(numerator / denominator)
        im = ax.imshow(np.log10(np.array(ratios).T), aspect='auto', cmap='RdBu_r', norm=TwoSlopeNorm(0, vmin=-2, vmax=2))
        ax.set_xticks(range(7), ['Z', 'T', 'U', 'V', 'Q', 'Ice', 'Liquid'], rotation=45)
        ax.set_yticks(range(37), [str(int(x)) for x in run['baseline']['pressure_hpa']], fontsize=8)
        ax.set_ylabel('Pressure level (hPa)')
        ax.set_title(f"K={run['K']}, w{run['width']}, d16\nLatest validation: step {last['update']}")
    colorbar = fig.colorbar(im, ax=list(axes), shrink=.6, extend='both', ticks=[-2, -1, 0, 1, 2])
    colorbar.ax.set_yticklabels(['0.01×', '0.1×', '1×', '10×', '100×'])
    colorbar.set_label('RMSE / frozen NGCM RMSE')
    fig.suptitle('All 37 pressure levels: blue = smaller error, red = larger error\nEvery field and level is shown; K=2 uses matched 6 + 12 h lead aggregation', fontsize=14)
    finish(fig, out, 'physical_rmse_by_level')

    fig, ax = plt.subplots(figsize=(8, 4.8), layout='constrained')
    for run in [r for r in available if r['K'] == 1]:
        rows = [dict(run['baseline'], update=0)] + run['validation']
        ax.plot([r['update'] for r in rows], [100 * (1 - r['loss'] / run['baseline']['loss']) for r in rows],
                '-o', color=COLORS[run['width']], ms=4, label=f"w{run['width']} / d16")
    style(ax); ax.set_ylim(-3, 103); ax.set_ylabel('Validation objective reduction (%)')
    ax.axhline(0, color='black', ls='--', label='Frozen NGCM'); ax.legend()
    ax.set_title('K=1: new five-term objective reduction\nLoss reduction is not physical forecast skill', fontsize=12)
    finish(fig, out, 'k1_objective_reduction')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logs', type=Path, default=ROOT/'logs/ngcm_aligned_20260924')
    parser.add_argument('--output', type=Path, default=ROOT/'plot/paper_training_20260924')
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'snapshot.json'
    if args.refresh:
        data = snapshot(args.logs)
        check(data)
        write_json(path, data)
    else:
        data = json.loads(path.read_text())
        check(data)
    summary = dict(captured_at_utc=data['captured_at_utc'], snapshot_sha256=sha(path.read_bytes()),
                   diagnostic_definition='Unfiltered normalized nodal 20*M_data analogue; not exact modal or full five-term loss', runs=[])
    for run in data['runs']:
        record = {key: run[key] for key in ['K', 'width', 'd_inner', 'status']}
        if run['validation']:
            baseline, last = run['baseline'], run['validation'][-1]
            record.update(training_step=run['training'][-1]['update'], validation_step=last['update'],
                          baseline_loss=baseline['loss'], loss=last['loss'],
                          objective_reduction_pct=100 * (1 - last['loss'] / baseline['loss']),
                          baseline_diagnostic=diagnostic(baseline, run, data['statistics']),
                          latest_diagnostic=diagnostic(last, run, data['statistics']),
                          improved_level_lead_cells={field:int(np.sum(np.asarray(last['physical_rmse'][field]) < np.asarray(baseline['physical_rmse'][field]))) for field in FIELDS})
        summary['runs'].append(record)
    write_json(args.output/'summary.json', summary)
    export_csv(data, args.output)
    figures(data, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
