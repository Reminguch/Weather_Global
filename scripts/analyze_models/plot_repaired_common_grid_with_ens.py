#!/usr/bin/env python3
"""Render repaired 2023 comparisons in the historical ENS-relative style."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS = ('m1', 'gc1', 'gc025', 'ens')
STYLES = {'m1': ('#26609f', '-', 'GraphCast–Mamba M1 (1°)'),
          'gc1': ('#8b919b', '--', 'Public GraphCast (1°)'),
          'gc025': ('#1fa450', '-', 'Public GraphCast (0.25°)')}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def write_json(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def plot_figures(images, diagnostics, leads, curves, counts):
    """Same panels, sizes, colors, line styles and ENS=100 as the 68-date plots."""
    outputs = []
    for name, selected, shape in [('fig5', ['Q700', 'T2m', 'U850'], (1, 3)),
                                   ('fig9', diagnostics, (3, 4))]:
        fig, axes = plt.subplots(*shape, figsize=(11, 3.1 if name == 'fig5' else 8), squeeze=False)
        for ax, diagnostic in zip(axes.flat, selected, strict=True):
            k = diagnostics.index(diagnostic)
            for model, (color, line, label) in STYLES.items():
                ax.plot(np.asarray(leads) / 24, curves[model][:, k], color=color,
                        ls=line, label=label, lw=1.6)
            ax.axhline(100, color='.65', lw=.7)
            ax.set(title=diagnostic, xlabel='Forecast lead (days)', xlim=(.25, 10))
            ax.grid(alpha=.2)
        for ax in axes[:, 0]:
            ax.set_ylabel('RMSE (% of matched ENS)')
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=9)
        selected_counts = counts[:, [diagnostics.index(d) for d in selected]]
        fig.suptitle(f'2023 · {selected_counts.min()}–{selected_counts.max()} matched starts per lead/diagnostic · '
                     'common 1.5° verification grid', y=.925, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .89))
        for extension in ('png', 'pdf'):
            path = images / f'{name}_public_checkpoints_2023_repaired_vs_ens.{extension}'
            if path.exists():
                raise FileExistsError(path)
            kwargs = {'metadata': {'CreationDate': None, 'ModDate': None}} if extension == 'pdf' else {}
            fig.savefig(path, dpi=220, bbox_inches='tight', **kwargs)
            outputs.append(identity(path))
        plt.close(fig)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workflow', type=Path, default=ROOT / 'artifacts/evaluations/graphcast_common_grid_2023_truth_repair_20261018')
    parser.add_argument('--name', default='public_checkpoints_2023_repaired_ens_20261018')
    args = parser.parse_args()
    workflow = args.workflow.resolve()
    summary_path = workflow / 'report/summary.json'
    summary = read(summary_path)
    validation_path = workflow / 'report/independent_validation.json'
    validation = read(validation_path)
    if validation['status'] != 'passed' or validation['summary_sha256'] != sha(summary_path):
        raise ValueError('The repaired model comparison must pass independent validation first')
    starts, leads, diagnostics = summary['initialization_times'], summary['lead_hours'], summary['diagnostics']
    if len(starts) != 122 or leads != list(range(6, 241, 6)) or len(diagnostics) != 12:
        raise ValueError('Unexpected parent evaluation dimensions')
    start_indices = {t: i for i, t in enumerate(starts)}
    lead_indices = {h: j for j, h in enumerate(leads)}
    diag_indices = {d: k for k, d in enumerate(diagnostics)}
    values = {m: np.full((122, 40, 12), np.nan) for m in MODELS}
    score_path = Path(validation['csv_path'])
    if sha(score_path) != validation['csv_sha256']:
        raise ValueError('Validated model scores changed')
    seen = set()
    with score_path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            model, t, h, d = row['model'], row['initialization_time'], int(row['lead_hours']), row['diagnostic']
            key = model, t, h, d
            if model not in STYLES or key in seen:
                raise ValueError('Unexpected or duplicate model score')
            seen.add(key)
            values[model][start_indices[t], lead_indices[h], diag_indices[d]] = float(row['mse_si'])
    if len(seen) != 175680:
        raise ValueError('Model score coverage changed')
    for model in STYLES:
        if not np.isfinite(values[model]).all() or np.any(values[model] < 0):
            raise ValueError('Invalid model scores')
        np.testing.assert_allclose(np.sqrt(values[model].mean(axis=0)), summary['rmse'][model], rtol=2e-14)
    ens_path = workflow / 'ens/ens_mean_mse.json'
    ens = read(ens_path)
    if ens.get('repair_protocol_hash') != summary['protocol_hash']:
        raise ValueError('ENS scores do not identify the same truth repair')
    ens_audit_path = Path(ens['repair_audit'])
    ens_audit = read(ens_audit_path)
    if (ens_audit['status'] != 'complete'
            or ens_audit['repair_protocol_hash'] != summary['protocol_hash']
            or ens_audit['corrected_ens_scores']['sha256'] != sha(ens_path)
            or not ens_audit['coverage_unchanged']
            or not ens_audit['independent_mse_formula_checked']):
        raise ValueError('ENS repair audit is incomplete or identifies different scores')
    if [r['initialization_time'] for r in ens['per_initialization']] != starts:
        raise ValueError('ENS initialization coverage differs from the comparison')
    for i, record in enumerate(ens['per_initialization']):
        if record['lead_hours'] != leads or set(record['mse']) != set(diagnostics):
            raise ValueError('ENS lead/diagnostic coverage changed')
        array = np.asarray([record['mse'][d] for d in diagnostics], dtype=float).T
        if array.shape != (40, 12) or np.isinf(array).any() or np.any(array[np.isfinite(array)] <= 0):
            raise ValueError('Invalid ENS MSE values')
        values['ens'][i] = array
    matched = np.isfinite(values['ens'])
    counts = matched.sum(axis=0)
    if counts.min() == 0:
        raise ValueError('ENS is unavailable for a complete lead/diagnostic point')
    pooled_mse = {model: np.where(matched, cube, 0).sum(axis=0) / counts for model, cube in values.items()}
    rmse = {model: np.sqrt(array) for model, array in pooled_mse.items()}
    relative = {model: 100 * rmse[model] / rmse['ens'] for model in STYLES}
    relative['ens'] = np.full((40, 12), 100.)
    np.testing.assert_array_equal(relative['ens'], np.full((40, 12), 100.))
    tables = ROOT / 'plots/analyze_models/data/resolution_eval' / args.name
    images = ROOT / 'plots/analyze_models/images/resolution_eval' / args.name
    tables.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    csv_path = tables / 'matched_per_initialization_mse.csv'
    with csv_path.open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'initialization_time', 'lead_hours', 'diagnostic', 'mse_si'])
        for model in MODELS:
            for i, start in enumerate(starts):
                for j, lead in enumerate(leads):
                    for k, diagnostic in enumerate(diagnostics):
                        if matched[i, j, k]:
                            writer.writerow([model, start, lead, diagnostic, values[model][i, j, k]])
    outputs = [identity(csv_path)] + plot_figures(images, diagnostics, leads, relative, counts)
    result = {
        'status': 'complete', 'schema_version': 1,
        'repair_protocol_hash': summary['protocol_hash'], 'reference_protocol_hash': summary['reference_protocol_hash'],
        'candidate_initializations': 122, 'initialization_times': starts,
        'lead_hours': leads, 'diagnostics': diagnostics, 'models': list(MODELS),
        'figure_diagnostics': {'fig5': ['Q700', 'T2m', 'U850'], 'fig9': diagnostics},
        'matched_initializations_per_lead_diagnostic': counts.tolist(),
        'matched_initializations_min_max': [int(counts.min()), int(counts.max())],
        'matched_score_cells_per_model': int(matched.sum()),
        'missing_ens_score_cells': int((~matched).sum()),
        'complete_starts_across_all_leads_and_diagnostics': int(matched.all(axis=(1, 2)).sum()),
        'aggregation': 'For each diagnostic and lead, use the identical ENS-available initialization set for every model. Pool per-start spatial MSE before square root. Plot 100 times model RMSE divided by matched ENS-mean RMSE.',
        'selection_basis': 'Only ENS forecast availability; no error-based selection or interpolation.',
        'truth': 'Same verification truth as the independently validated repaired comparison; ENS is rescored at the repaired valid times.',
        'ens_store': ens['store'], 'ens_store_note': ens['store_note'],
        'rmse': {m: v.tolist() for m, v in rmse.items()},
        'rmse_pct_of_ens': {m: v.tolist() for m, v in relative.items()},
        'style_reference': 'Same 1x3 and3x4 panels, canvas sizes, colors and line styles as fig5/fig9_k22_common_grid_68; ENS reference at100 and no shading.',
        'uncertainty': 'Point estimates only. Full122 model-only confidence intervals are not transferred to the ENS-matched subsets.',
        'sources': [identity(summary_path), identity(validation_path), identity(score_path),
                    identity(ens_path), identity(ens_audit_path), identity(__file__)],
        'outputs': outputs,
    }
    write_json(tables / 'summary.json', result)
    print(json.dumps({'tables': str(tables), 'images': str(images),
                      'matched_initializations_min_max': result['matched_initializations_min_max'],
                      'day10_percent_ens': {m: dict(zip(diagnostics, v[-1].tolist())) for m, v in relative.items()}}, indent=2))


if __name__ == '__main__':
    main()
