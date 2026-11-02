#!/usr/bin/env python3
"""Score matched public references, then plot and quantify the frozen K22 comparison."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]

import numpy as np
import xarray as xr
from eval_common_grid_2023 import publish_new, read_json, sha256
from eval_k22_common_grid_2022 import protocol
from common_grid_metrics import DIAGNOSTICS


def references(workflow, p):
    import score_wb2_graphcast_archive_2022 as archive
    import score_ens_common_grid_2023 as ens
    from common_grid_metrics import score_step
    scorer = sys.modules['common_grid_metrics']
    staging = read_json(workflow / 'staging_complete.json')
    if staging['protocol_hash'] != p['protocol_hash']:
        raise ValueError('Staging belongs to another protocol')
    hashes = {}
    for s in p['initialization_times']:
        base = dt.datetime.fromisoformat(s.replace('Z', ''))
        hashes[s] = {(base+dt.timedelta(hours=h)).strftime('%Y-%m-%dT%H:%M:%SZ'):
                     staging['truth_sha256'][(base+dt.timedelta(hours=h)).strftime('%Y-%m-%dT%H:%M:%SZ')]
                     for h in p['lead_hours']}
    meta, gc_coords, gc_arrays = archive.layout()
    ens_arrays, _, ens_coords, ens_meta = ens.reference_layout()
    ens.STORE, ens.STORE_EPOCH = p['ens_reference'], ens.REFERENCE_EPOCH
    publish_new(workflow / 'references/source_metadata.json', {
        'graphcast_store': p['reference'], 'graphcast_attrs': meta['.zattrs'],
        'graphcast_metadata': meta, 'ens_store': p['ens_reference'], 'ens_metadata': ens_meta,
    })
    results = {'gc025': [], 'ens': []}

    def fetch_pair(start):
        when = dt.datetime.fromisoformat(start.replace('Z', ''))
        gc_raw = archive.fetch_start(archive.time_index(gc_coords, when), gc_arrays, p['lead_hours'])
        t = ens.time_index(ens.REFERENCE_EPOCH, when)
        raw = {}
        for var, spec in ens_arrays.items():
            for c in sorted({(h//6)//ens.LEAD_CHUNK for h in p['lead_hours']}):
                key = ens.chunk_key(var, spec, t, c)
                raw[var, c] = ens.fetch(ens.BUCKET+p['ens_reference']+key)
                if raw[var, c] is None:
                    raise ValueError(f'Missing matched ENS chunk: {key}; do not change dates based on errors')
        return gc_raw, raw

    # Bound downloaded bytes; all decoding/scoring/HDF5 runs serially.
    for offset in range(0, len(p['initialization_times']), 4):
        starts = p['initialization_times'][offset:offset+4]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(fetch_pair, s) for s in starts]
            for start, future in zip(starts, futures):
                gc_raw, ens_raw = future.result()
                for key, raw, arrays, coords, function in (
                    ('gc025', gc_raw, gc_arrays, gc_coords, archive.score_start),
                    ('ens', ens_raw, ens_arrays, ens_coords, ens.score_start),
                ):
                    row = function(start, raw, arrays=arrays, coords=coords, scorer=scorer, protocol=p, hashes=hashes)
                    row['truth_sha256'] = [hashes[start][valid] for valid in hashes[start]]
                    if any(v is None or not np.isfinite(v) for a in row['mse'].values() for v in a):
                        raise ValueError(f'Incomplete reference: {key}/{start}')
                    results[key].append(row)
                    publish_new(workflow / 'references' / f'{key}_{start[:10]}.json',
                                {'protocol_hash': p['protocol_hash'], 'per_initialization': [row]})
                print(f'references {offset+len(results["gc025"])-offset}/{len(p["initialization_times"])} {start}', flush=True)
                del gc_raw, ens_raw
    publish_new(workflow / 'references_complete.json', {
        'protocol_hash': p['protocol_hash'], 'status': 'complete',
        'initializations': len(p['initialization_times']), 'models': list(results),
    })


def blocks(starts):
    dates = np.array(starts, dtype='datetime64[s]')
    return np.split(np.arange(len(dates)), np.flatnonzero(np.diff(dates) != np.timedelta64(1, 'D'))+1)


def bootstrap_counts(starts, length, draws, seed):
    rng = np.random.default_rng(seed)
    counts = np.zeros((draws, len(starts)), dtype=np.int16)
    for period in blocks([s.replace('Z', '') for s in starts]):
        width = min(length, len(period))
        for draw in range(draws):
            offsets = rng.integers(0, len(period)-width+1, size=int(np.ceil(len(period)/width)))
            sampled = np.concatenate([period[o:o+width] for o in offsets])[:len(period)]
            counts[draw] += np.bincount(sampled, minlength=len(starts)).astype(np.int16)
    if not np.all(counts.sum(axis=1) == len(starts)):
        raise ValueError('Bootstrap changed the number of dates')
    return counts.astype(np.float64)/len(starts)


def plot_figures(summary, out):
    """Render point estimates from saved results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    count, diagnostics = summary['initializations'], summary['diagnostics']
    curves = {k: np.asarray(v) for k, v in summary['rmse_pct_of_ens'].items()}
    days = np.array(summary['lead_hours'])/24
    styles = {'k22': ('#26609f', '-', 'GraphCast–Mamba K22 (1°)'),
              'gc1': ('#8b919b', '--', 'GraphCast (1°)'), 'gc025': ('#1fa450', '-', 'DeepMind archive (0.25°)')}
    for name, variables, shape in [('fig5', ['Z500', 'T2m', 'U850'], (1,3)),
                                    ('fig9', diagnostics, (3,4))]:
        fig, axes = plt.subplots(*shape, figsize=(11, 3.1 if name=='fig5' else 8), squeeze=False)
        for ax, diagnostic in zip(axes.flat, variables):
            k = diagnostics.index(diagnostic)
            for model, (color, line, label) in styles.items():
                ax.plot(days, curves[model][:,k], color=color, ls=line, label=label, lw=1.6)
            ax.axhline(100, color='.65', lw=.7)
            ax.set(title=diagnostic, xlabel='Forecast lead (days)', xlim=(.25,10))
            ax.grid(alpha=.2)
        for ax in axes[:,0]:
            ax.set_ylabel('RMSE (% of matched ENS)')
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=9)
        fig.suptitle(f'{count} daily starts in five 2022 periods; common 1.5° verification grid', y=.925, fontsize=10)
        fig.tight_layout(rect=(0,0,1,.89))
        for extension in ('png', 'pdf'):
            fig.savefig(out / f'{name}_k22_common_grid_{count}.{extension}', dpi=220, bbox_inches='tight')
        plt.close(fig)


def report(workflow, p):
    import pandas as pd
    starts, diagnostics = p['initialization_times'], list(DIAGNOSTICS)
    count = len(starts)
    score_protocol = p.get('score_protocol_hash', p['protocol_hash'])
    scores, hashes = {}, {}
    for index in range(p['resources']['production']['shards']):
        path = workflow / 'shards' / f'k22_{index:02d}.json'
        value = read_json(path)
        if value['protocol_hash'] != score_protocol or value['status'] != 'complete':
            raise ValueError(f'Invalid production shard: {path}')
        for row in value['per_initialization']:
            start = row['initialization_time']
            if start in hashes:
                raise ValueError(f'Duplicate start: {start}')
            hashes[start] = row['truth_sha256']
            for model, mse in row['mse'].items():
                scores[model, start] = mse
    if sorted(hashes) != p.get('source_initialization_times', starts):
        raise ValueError('Production coverage differs from the frozen dates')
    for model in ('gc025', 'ens'):
        for start in starts:
            data = read_json(workflow / 'references' / f'{model}_{start[:10]}.json')
            row = data['per_initialization'][0]
            if (data['protocol_hash'] != score_protocol or row['initialization_time'] != start
                    or row['truth_sha256'] != hashes[start] or row['lead_hours'] != p['lead_hours']):
                raise ValueError(f'Reference/truth identity differs: {model}/{start}')
            scores[model, start] = row['mse']
    arrays = {model: np.array([[[scores[model, s][d][i] for d in diagnostics] for i in range(40)]
                               for s in starts]) for model in ('k22', 'gc1', 'gc025', 'ens')}
    if any(a.shape != (count, 40, 12) or not np.isfinite(a).all() or np.any(a < 0) for a in arrays.values()):
        raise ValueError('Incomplete or invalid matched score cube')
    rmse = {k: np.sqrt(v.mean(axis=0)) for k, v in arrays.items()}
    curves = {k: 100*v/rmse['ens'] for k, v in rmse.items()}
    uncertainty = {}
    for length in p['bootstrap']['block_days']:
        weights = bootstrap_counts(starts, length, p['bootstrap']['draws'], p['bootstrap']['seed'])
        sampled = {k: weights @ a.reshape(count, -1) for k, a in arrays.items()}
        reduction = 100*(1-np.sqrt(sampled['k22']/sampled['gc025']))
        relative = 100*np.sqrt(sampled['k22']/sampled['ens'])
        uncertainty[str(length)] = {
            'rmse_reduction_vs_gc025_ci95': np.quantile(reduction, [.025, .975], axis=0).reshape(2, 40, 12).tolist(),
            'k22_vs_ens_ci95': np.quantile(relative, [.025, .975], axis=0).reshape(2, 40, 12).tolist(),
        }
    summary = {'protocol_hash': p['protocol_hash'], 'score_protocol_hash': score_protocol,
               'starts': starts, 'initializations': count, 'exclusions': p.get('exclusions', []),
               'lead_hours': p['lead_hours'], 'diagnostics': diagnostics,
               'rmse': {k: a.tolist() for k, a in rmse.items()},
               'rmse_pct_of_ens': {k: a.tolist() for k, a in curves.items()},
               'uncertainty': uncertainty, 'interpretation': p['interpretation']}
    publish_new(workflow / 'report/summary.json', summary)
    out, tables = Path(p['plot_directory']), Path(p['table_directory'])
    out.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, a in arrays.items():
        for i, start in enumerate(starts):
            for j, hours in enumerate(p['lead_hours']):
                for k, diagnostic in enumerate(diagnostics):
                    rows.append((model, start, hours, diagnostic, a[i,j,k]))
    pd.DataFrame(rows, columns=['model', 'initialization_time', 'lead_hours', 'diagnostic', 'mse_si']).to_csv(
        tables / 'per_initialization_mse.csv', index=False)
    plot_figures(summary, out)
    reductions = 100*(1-rmse['k22'][-1]/rmse['gc025'][-1])
    text = ('# Matched historical K22 comparison\n\n'+p['interpretation']+'\n\n'
            f'{count} daily00UTC starts,40 six-hour leads. All four predictors use identical dates and '
            'hash-matched native-0.25° ERA5 truth conservatively mapped to1.5°. Errors pool MSE before square root. '
            'Saved uncertainty estimates are pointwise paired five-day block intervals within each of five date windows; '
            'ten-day blocks are a sensitivity. '
            'These are physical-unit RMSE diagnostics.\n\n'
            '|Diagnostic|Day10 RMSE reduction vs archived GC025 (%)|\n|---|---:|\n'+
            '\n'.join(f'|{d}|{v:.3f}|' for d,v in zip(diagnostics,reductions))+'\n')
    (workflow / 'report/report.md').write_text(text)
    publish_new(workflow / 'report_complete.json', {'status': 'complete', 'protocol_hash': p['protocol_hash'],
                                                   'figures': str(out)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['references', 'report', 'plots'])
    parser.add_argument('--workflow', type=Path, required=True)
    args = parser.parse_args()
    p = protocol(args.workflow)
    if args.phase == 'plots':
        summary_path = args.workflow / 'report/summary.json'
        summary = read_json(summary_path)
        if summary['protocol_hash'] != p['protocol_hash']:
            raise ValueError('Summary belongs to another protocol')
        plot_figures(summary, p['plot_directory'])
        (Path(p['plot_directory']) / 'plot_metadata.json').write_text(json.dumps({
            'protocol_hash': p['protocol_hash'], 'summary_sha256': sha256(summary_path),
            'plotter': str(Path(__file__).resolve()), 'plotter_sha256': sha256(__file__),
        }, indent=2)+'\n')
    else:
        (references if args.phase=='references' else report)(args.workflow, p)
