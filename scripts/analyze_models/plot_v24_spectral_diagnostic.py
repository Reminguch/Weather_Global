"""Plot saved pilot powers, retaining every repeated GraphCast baseline.

This diagnostic does not certify the pilot's cross-run repeatability checks.
Run with the GraphCast environment, as a module from the repository root.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.analyze_models.report_v24_spectral_pilot import (
    DISPLAY_LEADS, VARIABLE_UNITS, _load_record, _require, _time,
)


def plot_error_reductions(workflow, protocol, pooled, k, image_dir, data_dir):
    """Use each model's paired baseline; aggregate errors before ratios."""
    import csv
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    models = protocol['models']
    labels = ('di16 · faster Mamba LR', 'di32 · joint LR', 'di16 · optimizer study')
    colors = ('#0072B2', '#D55E00', '#009E73')
    variables = protocol['spectral_variables']
    days = np.asarray(protocol['lead_hours']) / 24
    outputs, rows = [], []

    def reduction(full, baseline):
        _require(np.all(baseline > 0), 'Zero spectral-error denominator')
        return 100 * (1 - full / baseline)

    def decorate(fig, axes, title):
        for ax in np.asarray(axes).ravel():
            ax.axhline(0, color='black', lw=1.2)
            ax.grid(alpha=.2)
            lo, hi = ax.get_ylim()
            ax.set_ylim(min(lo, -2), max(hi, 2))
            ax.axhspan(0, max(hi, 2), color='#009E73', alpha=.055)
            ax.axhspan(min(lo, -2), 0, color='#D55E00', alpha=.055)
        fig.suptitle(title, fontsize=17, y=.985)
        fig.text(.5, .94, 'Positive = better than original GC · Negative = worse · Zero = original GC', ha='center', fontsize=11)
        handles, names = np.asarray(axes).ravel()[0].get_legend_handles_labels()
        fig.legend(handles, names, loc='lower center', bbox_to_anchor=(.5, .065), ncol=3, frameon=False)
        fig.text(.5, .042, '1° · Eight 2023 starts · 30–60° both hemispheres · Each model uses its own GC baseline.', ha='center', fontsize=9)
        fig.text(.5, .018, 'Diagnostic: baseline repeatability unresolved; no uncertainty intervals. Spectral-error reduction is not exact GraphCast loss reduction.', ha='center', fontsize=9)
        fig.subplots_adjust(top=.87, bottom=.19, left=.085, right=.98, hspace=.4, wspace=.25)

    with PdfPages(image_dir / 'spectral_error_reduction_overview.pdf') as pdf:
        selected = ['T2m', 'WS10', 'T850', 'WS850', 'MSLP', 'TP6']
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
        for ax, variable in zip(axes.ravel(), selected):
            vi = variables.index(variable)
            for mi, model in enumerate(models):
                record = pooled[model['model_id']]
                b = record['baseline_error_power'][:, vi, 1:].sum(axis=-1)
                f = record['full_error_power'][:, vi, 1:].sum(axis=-1)
                ax.plot(days, reduction(f, b), color=colors[mi], lw=2, label=labels[mi])
            ax.set_title(variable)
            ax.set_xlabel('Forecast lead (days)')
            ax.set_ylabel('Spectral-error reduction (%)')
        decorate(fig, axes, 'Does the trained model improve accuracy? All nonzero spatial modes')
        path = image_dir / 'spectral_error_reduction_by_lead.png'
        fig.savefig(path, dpi=160); pdf.savefig(fig); plt.close(fig)
        outputs.append(str(path))

        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        x = np.arange(len(variables))
        for ax, mask, title in zip(axes, (k > 0, k >= 90),
                                   ('All nonzero modes', 'Fine scales: zonal modes ≥90')):
            for mi, model in enumerate(models):
                record = pooled[model['model_id']]
                b = record['baseline_error_power'][..., mask].sum(axis=(0, 2))
                f = record['full_error_power'][..., mask].sum(axis=(0, 2))
                values = reduction(f, b)
                bars = ax.bar(x + (mi - 1) * .25, values, width=.24, color=colors[mi], label=labels[mi])
                ax.bar_label(bars, fmt='%.1f', fontsize=8, padding=3)
                for variable, value in zip(variables, values):
                    rows.append([model['model_id'], variable, 'all_40_leads', title, float(value)])
            ax.set_xticks(x, variables)
            ax.set_title(title)
            ax.set_ylabel('Spectral-error reduction (%)')
            ax.margins(y=.2)
        decorate(fig, axes, 'Spectral-error reduction over the full 10-day rollout')
        path = image_dir / 'spectral_error_reduction_summary.png'
        fig.savefig(path, dpi=160); pdf.savefig(fig); plt.close(fig)
        outputs.append(str(path))

    with PdfPages(image_dir / 'spectral_error_reduction.pdf') as pdf:
        for vi, variable in enumerate(variables):
            fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
            for ax, lead in zip(axes.ravel(), DISPLAY_LEADS):
                step = protocol['lead_hours'].index(lead)
                for mi, model in enumerate(models):
                    record = pooled[model['model_id']]
                    b = record['baseline_error_power'][step, vi, 1:]
                    f = record['full_error_power'][step, vi, 1:]
                    values = reduction(f, b)
                    ax.plot(k[1:], values, color=colors[mi], lw=1.4, label=labels[mi])
                    for mode, value in zip(k[1:], values):
                        rows.append([model['model_id'], variable, lead, int(mode), float(value)])
                ax.set_xscale('log')
                ax.set_title(f'Day {lead // 24}')
                ax.set_xlabel('Spatial frequency: zonal wavenumber k (cycles / circle)')
                ax.set_ylabel('Spectral-error reduction (%)')
            decorate(fig, axes, f'{variable}: spectral-error reduction versus spatial frequency')
            path = image_dir / f'spectral_error_reduction_{variable}.png'
            fig.savefig(path, dpi=160); pdf.savefig(fig); plt.close(fig)
            outputs.append(str(path))
    with (data_dir / 'spectral_error_reduction.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['model_id', 'variable', 'lead_hours', 'mode_or_band', 'spectral_error_reduction_pct'])
        writer.writerows(rows)
    (data_dir / 'spectral_error_reduction.json').write_text(json.dumps({
        'status': 'diagnostic_only', 'protocol_hash': protocol['protocol_hash'],
        'formula': '100 * (1 - sum(full_error_power) / sum(paired_baseline_error_power))',
        'aggregation': 'Equal-weight mean over starts, then sum selected modes and leads before ratio.',
        'baseline': 'Each model uses the GC baseline from its own evaluation; repeatability unresolved.',
        'images': outputs, 'pdf': str(image_dir / 'spectral_error_reduction.pdf'),
    }, indent=2) + '\n')
    print(json.dumps({'error_reduction_images': outputs}, indent=2))


def plot_power_bias(protocol, pooled, k, image_dir, data_dir, sources, *, relative_error=False):
    """Signed discrepancy of pooled power spectra relative to ERA5."""
    import csv
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    models = protocol['models']
    labels = ('di16 · faster Mamba LR', 'di32 · joint LR', 'di16 · optimizer study')
    colors = ('#0072B2', '#D55E00', '#009E73')
    prefix = 'power_spectrum_relative_error' if relative_error else 'power_spectrum_bias'
    ylabel = 'Relative power-spectrum error (%)' if relative_error else 'Power discrepancy relative to ERA5 (%)'
    outputs, rows = [], []
    with PdfPages(image_dir / f'{prefix}.pdf') as pdf:
        for vi, variable in enumerate(protocol['spectral_variables']):
            fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
            for ax, lead in zip(axes.ravel(), DISPLAY_LEADS):
                step = protocol['lead_hours'].index(lead)
                truth = pooled[models[0]['model_id']]['truth_power'][step, vi, 1:]
                _require(np.all(truth > 0), 'Zero truth power denominator')
                ax.axhline(0, color='black', lw=1.5, label='ERA5: zero discrepancy')
                for mi, model in enumerate(models):
                    record = pooled[model['model_id']]
                    for branch in ('baseline', 'full'):
                        bias = 100 * (record[f'{branch}_power'][step, vi, 1:] / truth - 1)
                        if relative_error:
                            bias = np.abs(bias)
                        ax.plot(k[1:], bias,
                                color=('#555555', '#888888', '#aaaaaa')[mi] if branch == 'baseline' else colors[mi],
                                ls=('--', ':', '-.')[mi] if branch == 'baseline' else '-',
                                lw=1.3 if branch == 'baseline' else 1.7,
                                label=f'Original GC, evaluation {mi + 1}' if branch == 'baseline' else labels[mi])
                        rows.extend([model['model_id'], branch, variable, lead, int(mode), float(value)]
                                    for mode, value in zip(k[1:], bias))
                ax.set_xscale('log')
                ax.tick_params(labelbottom=True)
                ax.set_title(f'Day {lead // 24}')
                ax.set_xlabel('Spatial frequency: zonal wavenumber (cycles / circle)')
                ax.set_ylabel(ylabel)
                ax.grid(alpha=.2, which='both')
            # Set shared limits only after every lead has been drawn. Setting
            # a limit inside the loop disables autoscaling for later panels.
            values = np.concatenate([np.asarray(line.get_ydata()).ravel()
                                     for ax in axes.ravel() for line in ax.lines])
            low, high = float(values.min()), float(values.max())
            padding = .06 * max(high - low, 1.)
            axes[0, 0].set_ylim(0 if relative_error else low - padding, high + padding)
            title = 'relative power-spectrum error' if relative_error else 'power-spectrum discrepancy'
            fig.suptitle(f'{variable}: {title} versus spatial frequency', fontsize=15, y=.975)
            formula = ('100 × |forecast power − ERA5 power| / ERA5 power · Lower is better · Zero is ideal'
                       if relative_error else
                       '100 × (forecast power / ERA5 power − 1) · Zero is ideal · Negative: missing power · Positive: excess power')
            fig.text(.5, .93, formula,
                     ha='center', fontsize=10)
            handles, names = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, names, loc='lower center', bbox_to_anchor=(.5, .065), ncol=3, fontsize=9, frameon=False)
            fig.text(.5, .043, '1° · Eight 2023 starts · 30–60° both hemispheres · Cosine latitude weights · Powers averaged before ratios',
                     ha='center', fontsize=9)
            fig.text(.5, .018, 'Baseline repeatability unresolved: all three GC runs shown. Closer to zero means better power agreement.',
                     ha='center', fontsize=9)
            fig.subplots_adjust(top=.85, bottom=.24, left=.09, right=.96, hspace=.46, wspace=.24)
            path = image_dir / f'{prefix}_{variable}.png'
            fig.savefig(path, dpi=160)
            pdf.savefig(fig)
            plt.close(fig)
            outputs.append(str(path))
    with (data_dir / f'{prefix}.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['model_id', 'branch', 'variable', 'lead_hours', 'wavenumber',
                         'relative_power_error_pct' if relative_error else 'power_bias_pct'])
        writer.writerows(rows)
    summary = {'status': 'diagnostic_only', 'protocol_hash': protocol['protocol_hash'],
               'formula': ('100 * abs(mean_forecast_power - mean_ERA5_power) / mean_ERA5_power'
                           if relative_error else '100 * (mean_forecast_power / mean_ERA5_power - 1)'),
               'sources': sources, 'images': outputs, 'pdf': str(image_dir / f'{prefix}.pdf')}
    (data_dir / f'{prefix}.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(summary['pdf'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workflow', type=Path, required=True)
    parser.add_argument('--error-only', action='store_true')
    parser.add_argument('--power-bias-only', action='store_true')
    parser.add_argument('--power-relative-error-only', action='store_true')
    args = parser.parse_args()
    workflow = args.workflow.resolve()
    protocol = json.loads((workflow / 'protocol.json').read_text())
    body = {k: v for k, v in protocol.items() if k != 'protocol_hash'}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    _require(digest == protocol['protocol_hash'], 'Protocol hash mismatch')
    times = [_time(t) for t in protocol['initialization_times']]
    _require(len(times) == len(set(times)), 'Duplicate initializations')
    pooled, sources, canonical = {}, [], None
    k = None
    for model in protocol['models']:
        records = {}
        for path in sorted((workflow / 'models' / model['model_id']).glob('*.npz')):
            time, k, arrays, source = _load_record(path, protocol, model, times, k)
            _require(time not in records, 'Duplicate initialization')
            records[time] = arrays
            sources.append(source)
        _require(set(records) == set(times), 'Incomplete initializations')
        if canonical is None:
            canonical = records
        for time in times:
            _require(np.array_equal(records[time]['truth_power'], canonical[time]['truth_power']),
                     'ERA5 truth mismatch')
        pooled[model['model_id']] = {
            key: np.mean([records[t][key] for t in times], axis=0)
            for key in ('truth_power', 'baseline_power', 'full_power', 'baseline_error_power', 'full_error_power')
        }

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    image_dir = Path(protocol['output_image_dir']) / 'diagnostic'
    data_dir = workflow / 'diagnostic'
    image_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    if args.power_bias_only or args.power_relative_error_only:
        plot_power_bias(protocol, pooled, k, image_dir, data_dir, sources,
                        relative_error=args.power_relative_error_only)
        return
    plot_error_reductions(workflow, protocol, pooled, k, image_dir, data_dir)
    if args.error_only:
        return
    np.savez_compressed(data_dir / 'pooled_powers.npz', wavenumber=k,
                        lead_hours=protocol['lead_hours'], variables=protocol['spectral_variables'],
                        **{f'{m}_{key}': v for m, values in pooled.items() for key, v in values.items()})
    models = protocol['models']
    colors = ('#0072B2', '#D55E00', '#009E73')
    labels = ('di16, batch 4, faster Mamba LR (SWA 500–2000)',
              'di32, batch 4, joint LR (SWA 500–2000)',
              'di16, optimizer study (SWA 2000–8000)')
    positive = k > 0
    outputs = []
    with PdfPages(image_dir / 'spectra_all_variables.pdf') as pdf:
        for vi, variable in enumerate(protocol['spectral_variables']):
            fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True)
            for column, lead in enumerate(DISPLAY_LEADS):
                step = protocol['lead_hours'].index(lead)
                truth = pooled[models[0]['model_id']]['truth_power'][step, vi, positive]
                for row in (0, 1):
                    ax = axes[row, column]
                    denominator = np.ones_like(truth) if row == 0 else truth
                    _require(np.all(denominator > 0), 'Cannot plot zero truth power')
                    if row == 0:
                        ax.plot(k[positive], truth, color='black', lw=1.7, label='ERA5 truth')
                    else:
                        ax.axhline(1, color='black', lw=1, label='ERA5 reference')
                    for mi, model in enumerate(models):
                        power = pooled[model['model_id']]
                        ax.plot(k[positive], power['baseline_power'][step, vi, positive] / denominator,
                                color=('#555555', '#888888', '#aaaaaa')[mi],
                                ls=('-', '--', ':')[mi], lw=1.25,
                                label=f'Original GC, evaluation {mi + 1}')
                        ax.plot(k[positive], power['full_power'][step, vi, positive] / denominator,
                                color=colors[mi], lw=1.4, label=labels[mi])
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    ax.grid(alpha=.2, which='both')
                    if row == 0:
                        ax.set_title(f'Day {lead // 24}')
                    else:
                        ax.set_xlabel('Zonal wavenumber (cycles / circle)')
                    if column == 0:
                        ax.set_ylabel(f'Power ({VARIABLE_UNITS[variable]})' if row == 0 else 'Forecast / ERA5 power')
            fig.suptitle(f'{variable}: original GraphCast and trained models · 1°', fontsize=17, y=.985)
            fig.text(.5, .935, 'Eight 2023 starts · 30–60° latitude, both hemispheres · cosine weighting · powers averaged across starts',
                     ha='center', fontsize=10)
            handles, names = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, names, loc='lower center', bbox_to_anchor=(.5, .045), ncol=2, fontsize=9, frameon=False)
            fig.text(.5, .022, 'Diagnostic: baseline repeatability check failed; all three GC runs shown. Power agreement alone does not measure accuracy.',
                     ha='center', fontsize=10)
            fig.subplots_adjust(top=.88, bottom=.25, left=.07, right=.985, hspace=.27, wspace=.3)
            path = image_dir / f'spectra_{variable}.png'
            fig.savefig(path, dpi=160)
            pdf.savefig(fig)
            plt.close(fig)
            outputs.append(str(path))
    summary = {
        'status': 'diagnostic_only', 'protocol_hash': protocol['protocol_hash'],
        'checks': 'Record provenance, completeness, spectral identities and exact truth equality passed.',
        'baseline': 'All three original GC evaluations shown separately; cross-run repeatability is not certified.',
        'aggregation': 'Equal-weight mean power over eight initializations; ratios taken after averaging.',
        'models': models, 'sources': sources, 'images': outputs,
        'pdf': str(image_dir / 'spectra_all_variables.pdf'),
    }
    (data_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'images': outputs, 'pdf': summary['pdf']}, indent=2))


if __name__ == '__main__':
    main()
