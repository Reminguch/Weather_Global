#!/usr/bin/env python3
"""Compare public-checkpoint and archived GraphCast on their shared 2022 starts.

Reuses saved scalar errors only. Every curve is recomputed on the intersection
with the historical 68-start comparison, with identical truth hashes and leads.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / 'artifacts/evaluations/graphcast_common_grid_2022_20261016'
MATCHED = ROOT / 'artifacts/evaluations/k22_common_grid_2022_matched68_20261018'
NAME = 'public_gc025_archive_matched24_20261018'


def main():
    sources = {}

    def read(path):
        data = path.read_bytes()
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    p, q = read(PUBLIC / 'protocol.json'), read(MATCHED / 'protocol.json')
    completed = read(MATCHED / 'report/summary.json')
    public, local = {}, {}
    for directory, pattern, expected, target in (
        (PUBLIC / 'shards', 'gc025_*.json', p['protocol_hash'], public),
        (MATCHED / 'shards', 'k22_??.json', q['score_protocol_hash'], local),
    ):
        for path in sorted(directory.glob(pattern)):
            value = read(path)
            if value['status'] != 'complete' or value['protocol_hash'] != expected:
                raise ValueError(f'Invalid shard: {path}')
            for row in value['per_initialization']:
                start = row['initialization_time']
                if start in target:
                    raise ValueError(f'Duplicate start: {start}')
                target[start] = row
    if sorted(public) != p['initialization_times']:
        raise ValueError('Public-checkpoint coverage differs from its manifest')
    starts = sorted(set(public) & set(q['initialization_times']))
    if len(starts) != 24:
        raise ValueError(f'Expected 24 shared starts, found {len(starts)}')
    leads, diagnostics = q['lead_hours'], completed['diagnostics']
    scores = {model: [] for model in ('gc1', 'public_gc025', 'archive_gc025', 'ens')}
    for start in starts:
        a, b = public[start], local[start]
        if (a['lead_hours'] != leads or b['lead_hours'] != leads
                or a['valid_times'] != b['valid_times']
                or a['truth_sha256'] != b['truth_sha256']):
            raise ValueError(f'Public/local verification mismatch: {start}')
        scores['gc1'].append(b['mse']['gc1'])
        scores['public_gc025'].append(a['mse']['gc025'])
        for source, model in (('gc025', 'archive_gc025'), ('ens', 'ens')):
            record = read(MATCHED / 'references' / f'{source}_{start[:10]}.json')
            row = record['per_initialization'][0]
            if (record['protocol_hash'] != q['score_protocol_hash']
                    or row['initialization_time'] != start or row['lead_hours'] != leads
                    or row['truth_sha256'] != b['truth_sha256']):
                raise ValueError(f'Reference verification mismatch: {model}/{start}')
            scores[model].append(row['mse'])
    cubes = {model: np.array([[[row[d][i] for d in diagnostics] for i in range(len(leads))]
                              for row in rows]) for model, rows in scores.items()}
    for model, values in cubes.items():
        if values.shape != (24, 40, 12) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f'Incomplete score cube: {model}')
    rmse = {model: np.sqrt(values.mean(axis=0)) for model, values in cubes.items()}
    curves = {model: 100 * values / rmse['ens'] for model, values in rmse.items()}
    differences = {reference: 100 * (rmse['public_gc025'] / rmse[reference] - 1)
                   for reference in ('gc1', 'archive_gc025')}
    tables = ROOT / 'plots/analyze_models/data/resolution_eval' / NAME
    images = ROOT / 'plots/analyze_models/images/resolution_eval' / NAME
    tables.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    with (tables / 'per_initialization_mse.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'initialization_time', 'lead_hours', 'diagnostic', 'mse_si'])
        for model, values in cubes.items():
            for i, start in enumerate(starts):
                for j, lead in enumerate(leads):
                    for k, diagnostic in enumerate(diagnostics):
                        writer.writerow([model, start, lead, diagnostic, values[i, j, k]])
    summary = {
        'starts': starts, 'initializations': len(starts), 'lead_hours': leads,
        'diagnostics': diagnostics, 'selection': 'Intersection of saved 122 public-checkpoint starts and fixed 68 historical-comparison starts; no error filtering.',
        'public_checkpoint': {'filename': Path(p['gc025_checkpoint']).name,
                              'sha256': p['gc025_checkpoint_sha256']},
        'archive': q['reference'], 'archive_checkpoint': 'unidentified; differences in weights and inference numerics are not isolated',
        'numerics': 'local forecasts use bfloat16 network inference; archive inference configuration is not established',
        'rmse': {k: v.tolist() for k, v in rmse.items()},
        'rmse_pct_of_ens': {k: v.tolist() for k, v in curves.items()},
        'public_rmse_difference_pct': {k: v.tolist() for k, v in differences.items()},
        'truth_and_lead_hash_checks': 'passed for every model, start and lead',
        'source_sha256': sources,
        'plotter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (tables / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    days = np.array(leads) / 24
    styles = {'gc1': ('#8b919b', '--', 'GraphCast (1°)'),
              'public_gc025': ('#c77713', '-.', 'Public checkpoint rerun (0.25°; 1979–2017)'),
              'archive_gc025': ('#1fa450', '-', 'DeepMind archive (0.25°)')}
    for suffix, variables, shape in [('three_panels', ['Z500', 'T2m', 'U850'], (1, 3)),
                                    ('all_variables', diagnostics, (3, 4))]:
        fig, axes = plt.subplots(*shape, figsize=(11, 3.2 if shape[0] == 1 else 8), squeeze=False)
        for ax, diagnostic in zip(axes.flat, variables):
            k = diagnostics.index(diagnostic)
            for model, (color, line, label) in styles.items():
                ax.plot(days, curves[model][:, k], color=color, ls=line, lw=1.6, label=label)
            ax.axhline(100, color='.65', lw=.7)
            ax.set(title=diagnostic, xlabel='Forecast lead (days)', xlim=(.25, 10))
            ax.grid(alpha=.2)
        for ax in axes[:, 0]:
            ax.set_ylabel('RMSE (% of matched ENS)')
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=9)
        fig.suptitle('24 shared 2022 starts; all curves recomputed on the same dates and 1.5° grid', y=.925, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .89))
        for extension in ('pdf', 'png'):
            fig.savefig(images / f'public_gc025_vs_archive_{suffix}.{extension}', dpi=220, bbox_inches='tight')
        plt.close(fig)
    for diagnostic in ('Z500', 'T2m', 'U850'):
        k = diagnostics.index(diagnostic)
        print(diagnostic, {str(day): {ref: round(float(values[day * 4 - 1, k]), 3)
                                     for ref, values in differences.items()} for day in (1, 3, 5, 10)})
    print(f'Validated {len(starts)} matching starts and saved {images}')


if __name__ == '__main__':
    main()
