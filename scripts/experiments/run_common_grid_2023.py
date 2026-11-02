#!/usr/bin/env python3
"""Freeze and report the paired 2023 GraphCast common-grid comparison."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def prepare(workflow):
    from scripts.experiments.frozen_runtime import freeze_source
    workflow = workflow.resolve()
    if (workflow / 'protocol.json').exists():
        raise FileExistsError('Use the existing frozen protocol or a new workflow directory')
    original = ROOT / 'artifacts/evaluations/v24_res1_2023_stride3_memory_recovery_20261009'
    old = read(original / 'protocol.json')
    initializations = old['initialization_times']
    if len(initializations) != 122 or old['target_steps'] != 40:
        raise ValueError('Unexpected principal evaluation coverage')
    workflow.mkdir(parents=True, exist_ok=True)
    for name, times in [('initializations.json', initializations),
                        ('initializations_smoke.json', initializations[:1])]:
        write(workflow / name, {'schema_version': 1, 'initializations': times})
    reference = ROOT / ('data/graphcast/graphcast/params/GraphCast - ERA5 1979-2017 - '
                        'resolution 0.25 - pressure levels 37 - mesh 2to6 - '
                        'precipitation input and output.npz')
    data_root = ROOT / 'data/graphcast/graphcast/dataset/common_grid_2023_20261015'
    snapshot = freeze_source(ROOT, workflow / 'source')
    protocol = {
        'schema_version': 1, 'experiment': 'graphcast_common_grid_2023',
        'prepared_date': '2026-10-15', 'timezone': 'America/New_York',
        'workspace': str(ROOT), 'original_campaign': str(original),
        'original_protocol_sha256': digest(original / 'protocol.json'),
        'original_protocol_hash': old['protocol_hash'],
        'source_root': str(workflow / 'source'), 'source_digest': snapshot['source_digest'],
        'initialization_manifest': str(workflow / 'initializations.json'),
        'initialization_manifest_sha256': digest(workflow / 'initializations.json'),
        'initialization_times': initializations, 'target_steps': 40,
        'lead_hours': list(range(6, 241, 6)),
        'common_truth': str(data_root / 'common_truth'),
        'inputs025': str(data_root / 'inputs025'), 'data_root': str(data_root),
        'gc025_checkpoint': str(reference), 'gc025_checkpoint_sha256': digest(reference),
        'gc025_training_years': [1979, 2017], 'gc025_pressure_levels': 37,
        'stats_dir': old['stats_dir'], 'statistics': old['statistics'],
        'm1': next(m for m in old['models'] if m['model_id'] == 'M1'),
        'gc1': old['baseline'], 'numerics': old['numerics'],
        'numerical_policy': old['numerical_policy'], 'packages': old['packages'],
        'verification': {
            'source': 'ARCO ERA5 native 0.25 degree fields',
            'grid': {'latitude': [-90, 90, 1.5], 'longitude': [0, 358.5, 1.5]},
            'regridding': 'first-order conservative spherical cell overlap; periodic longitude',
            'spatial_weights': 'cosine latitude normalized over all grid points',
            'diagnostics': ['Z500','Z850','T500','T850','T2m','MSL','Q700','Q850','U850','V850','U10','V10'],
            'aggregation': 'pool physical-unit squared errors over starts before square root',
            'common_truth_required_for_all_models': True,
            'native_gc_loss': 'M1 run retained only as a numerical bridge to the principal evaluation',
        },
        'bootstrap': {'replicates': 10000, 'seed': 20260916, 'block_starts': 5,
                      'sensitivity_block_starts': [3, 7],
                      'method': 'paired non-wrapping moving-block percentile',
                      'intervals': 'pointwise; conditional on fixed checkpoints and 2023'},
        'prior_exposure': old.get('prior_exposure', {}),
        'interpretation': 'Separate fixed-model common-grid comparison; no 2023 checkpoint selection. '
                          'Does not isolate resolution from pressure levels, architecture, or training coverage.',
        'planned_shards': {'m1': 8, 'gc025': 16},
        'plot_directory': str(ROOT / 'plots/analyze_models/images/resolution_eval/common_grid_2023_20261015'),
        'table_directory': str(ROOT / 'plots/analyze_models/data/resolution_eval/common_grid_2023_20261015'),
    }
    protocol['protocol_hash'] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    write(workflow / 'protocol.json', protocol)
    print(json.dumps({'workflow': str(workflow), 'protocol_hash': protocol['protocol_hash'],
                      'data_root': str(data_root)}, indent=2))


def submit(workflow, phase, model, *, mem, walltime, shards=None):
    protocol = read(workflow / 'protocol.json')
    root = Path(protocol['workspace'])
    source = Path(protocol['source_root'])
    count = 1 if phase == 'smoke' else (shards or protocol['planned_shards'][model])
    output = workflow / ('pilot' if phase == 'smoke' else 'shards')
    output.mkdir(parents=True, exist_ok=True)
    (workflow / 'logs').mkdir(exist_ok=True)
    cmd = ['sbatch', '--parsable', f'--job-name=cgrid23-{model}-{phase}',
           f'--array=0-{count-1}', f'--mem={mem}', f'--time={walltime}',
           f'--output={workflow}/logs/{model}_{phase}_%A_%a.out',
           f'--error={workflow}/logs/{model}_{phase}_%A_%a.err',
           str(source / 'scripts/experiments/common_grid_2023_gpu.slurm'),
           str(workflow), model, str(count), str(output)]
    if phase == 'smoke':
        cmd += ['--initializations', str(workflow / 'initializations_smoke.json')]
    import shlex
    print(shlex.join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=root, text=True, capture_output=True, check=True)
    job = result.stdout.strip().split(';')[0]
    record = {'job_id': job, 'phase': phase, 'model': model, 'shards': count,
              'mem': mem, 'time': walltime, 'command': cmd, 'status_at_submission': 'submitted'}
    write(workflow / 'submissions' / f'{job}.json', record)
    print(json.dumps(record, indent=2))


def block_counts(n, length, draws, seed):
    import numpy as np
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (n, length, draws)):
        raise ValueError('Bootstrap sample count, block length, and draw count must be integers')
    if n < 1 or not 1 <= length <= n or draws < 1:
        raise ValueError('Bootstrap requires positive draws and a block length between one and sample count')
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n-length+1, size=(draws, (n+length-1)//length))
    positions = (starts[...,None] + np.arange(length)).reshape(draws, -1)[:,:n]
    counts = np.zeros((draws, n), dtype=np.int64)
    np.add.at(counts, (np.arange(draws)[:,None], positions), 1)
    return counts


def gather(workflow):
    import numpy as np
    from datetime import datetime, timedelta
    from scripts.analyze_models.common_grid_metrics import DIAGNOSTICS
    workflow = Path(workflow)
    protocol = read(workflow / 'protocol.json')
    body = {k: v for k, v in protocol.items() if k != 'protocol_hash'}
    if protocol.get('protocol_hash') != hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest():
        raise ValueError('Frozen protocol contents differ from its hash')
    times = protocol['initialization_times']
    diagnostics = protocol['verification']['diagnostics']
    if (protocol.get('schema_version') != 1 or len(times) != 122
            or times != sorted(set(times)) or protocol.get('target_steps') != 40
            or protocol['lead_hours'] != list(range(6, 241, 6))
            or diagnostics != list(DIAGNOSTICS)):
        raise ValueError('Unexpected frozen common-grid evaluation coverage')
    manifest_path = Path(protocol['initialization_manifest'])
    if digest(manifest_path) != protocol['initialization_manifest_sha256']:
        raise ValueError('Initialization manifest differs from frozen protocol')
    if read(manifest_path).get('initializations') != times:
        raise ValueError('Initialization manifest differs from protocol coverage')
    protocol_sha256 = digest(workflow / 'protocol.json')
    values = {m: np.full((len(times), 40, len(diagnostics)), np.nan) for m in ('gc1','m1','gc025')}
    time_indices = {stamp: index for index, stamp in enumerate(times)}
    truth_hashes, shard_counts, seen_shards = {}, {}, set()
    sources = []
    for path in sorted((workflow / 'shards').glob('*.json')):
        if '.partial.' in path.name or '.native.' in path.name or path.name.startswith('native_'):
            continue
        result = read(path)
        if (result.get('protocol_hash') != protocol['protocol_hash']
                or result.get('protocol_sha256') != protocol_sha256
                or result.get('original_protocol_hash') != protocol['original_protocol_hash']):
            raise ValueError(f'Wrong protocol: {path}')
        if result.get('schema_version') != 1 or result.get('status') != 'complete':
            raise ValueError(f'Incomplete shard: {path}')
        if result.get('initialization_manifest_sha256') != protocol['initialization_manifest_sha256']:
            raise ValueError(f'Wrong initialization manifest: {path}')
        model = result.get('model')
        if model not in ('m1', 'gc025'):
            raise ValueError(f'Unknown model: {model}')
        expected_branches = {'gc1', 'm1'} if model == 'm1' else {'gc025'}
        evidence = result.get('evidence', {})
        expected_checkpoint = (protocol['m1']['checkpoint_sha256'] if model == 'm1'
                               else protocol['gc025_checkpoint_sha256'])
        if evidence.get('checkpoint_sha256') != expected_checkpoint:
            raise ValueError(f'Wrong model checkpoint: {path}')
        if model == 'm1' and evidence.get('baseline_checkpoint_sha256') != protocol['gc1']['checkpoint_sha256']:
            raise ValueError(f'Wrong 1-degree backbone checkpoint: {path}')
        count, shard = result.get('shard_count'), result.get('shard_index')
        if (type(count) is not int or type(shard) is not int
                or not 1 <= count <= len(times) or not 0 <= shard < count):
            raise ValueError(f'Invalid shard ownership: {path}')
        if model in shard_counts and shard_counts[model] != count:
            raise ValueError(f'Inconsistent shard counts for {model}')
        shard_counts[model] = count
        if (model, shard) in seen_shards:
            raise ValueError(f'Duplicate shard: {model} {shard}')
        seen_shards.add((model, shard))
        records = result.get('per_initialization', [])
        if [r.get('initialization_time') for r in records] != times[shard::count]:
            raise ValueError(f'Incomplete or mismatched shard initialization coverage: {path}')
        for record in records:
            stamp = record['initialization_time']
            index = time_indices[stamp]
            hashes = record['truth_sha256']
            if len(hashes) != 40 or any(not isinstance(h,str) or len(h)!=64
                                        or any(c not in '0123456789abcdef' for c in h) for h in hashes):
                raise ValueError('Incomplete truth provenance')
            if record['lead_hours'] != protocol['lead_hours']:
                raise ValueError('Mismatched forecast leads')
            start = datetime.fromisoformat(stamp.replace('Z','+00:00'))
            expected = [(start+timedelta(hours=h)).isoformat().replace('+00:00','Z') for h in protocol['lead_hours']]
            if record['valid_times'] != expected:
                raise ValueError('Mismatched valid times')
            # The same valid time appears in overlapping forecasts as well as
            # different models. Every occurrence must use the identical truth.
            for valid_time, truth_hash in zip(expected, hashes):
                if valid_time in truth_hashes and truth_hashes[valid_time] != truth_hash:
                    raise ValueError(f'Different verification truth between models or starts: {valid_time}')
                truth_hashes[valid_time] = truth_hash
            if set(record['mse']) != expected_branches:
                raise ValueError(f'Mismatched model branches: {model}')
            for branch, mse in record['mse'].items():
                if np.any(np.isfinite(values[branch][index])):
                    raise ValueError(f'Duplicate initialization: {branch} {stamp}')
                if set(mse) != set(diagnostics):
                    raise ValueError('Mismatched diagnostics')
                array = np.asarray([mse[d] for d in diagnostics], dtype=float).T
                if array.shape != (40,len(diagnostics)) or not np.isfinite(array).all() or (array<0).any():
                    raise ValueError('Invalid MSE values')
                values[branch][index] = array
        sources.append({'path': str(path), 'sha256': digest(path)})
    if any(not np.isfinite(v).all() for v in values.values()):
        raise ValueError('All three models require all 122 starts and 40 leads before reporting')
    return protocol, values, sources


def analyze_values(values, times, settings, exposed=()):
    """Pool paired per-start MSE before sqrt, using the frozen block settings."""
    import numpy as np
    if set(values) != {'gc1', 'm1', 'gc025'}:
        raise ValueError('Exactly three paired models are required')
    expected_shape = values['m1'].shape
    if len(expected_shape) != 3 or expected_shape[0] != len(times):
        raise ValueError('MSE arrays must be indexed by initialization, lead, and diagnostic')
    if any(v.shape != expected_shape or not np.isfinite(v).all() or (v < 0).any()
           for v in values.values()):
        raise ValueError('Invalid paired MSE arrays')
    exposed = set(exposed)
    primary_length = settings['block_starts']
    cases = [('primary', primary_length, False)]
    cases += [(f'blocks{length}', length, False) for length in settings['sensitivity_block_starts']]
    cases += [('exclude_pilot', primary_length, True)]
    rmse = {model: np.sqrt(value.mean(axis=0)) for model, value in values.items()}
    analyses = {}
    for label, length, exclude in cases:
        counts = block_counts(len(times), length, settings['replicates'], settings['seed'])
        eligible = np.array([not exclude or stamp not in exposed for stamp in times])
        counts[:, ~eligible] = 0
        totals = counts.sum(axis=1)
        if not eligible.any() or (totals == 0).any():
            raise ValueError('Pilot exclusion leaves an empty analysis or bootstrap draw')
        boot = {model: np.sqrt((counts @ value.reshape(len(times), -1)
                               / totals[:, None]).reshape((-1,) + expected_shape[1:]))
                for model, value in values.items()}
        results = {}
        for reference in ('gc1', 'gc025'):
            pooled_reference = values[reference][eligible].mean(axis=0)
            if (pooled_reference <= 0).any() or (boot[reference] <= 0).any():
                raise ValueError(f'Relative RMSE reduction is undefined for zero {reference} error')
            points = 100 * (1 - np.sqrt(values['m1'][eligible].mean(axis=0) / pooled_reference))
            interval = np.percentile(100 * (1 - boot['m1'] / boot[reference]), [2.5, 97.5], axis=0)
            results[reference] = {'rmse_reduction_pct': points.tolist(),
                                  'ci95_lower': interval[0].tolist(),
                                  'ci95_upper': interval[1].tolist()}
        analyses[label] = {'block_starts': length, 'n_initializations': int(eligible.sum()),
                           'comparisons': results}
    return rmse, analyses


def plot_figures(summary, plot_dir):
    """Render saved common-grid results without recomputing the analysis."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    names = summary['diagnostics']
    rmse = {model: np.asarray(value) for model, value in summary['rmse'].items()}
    days = np.asarray(summary['lead_hours'])/24
    colors={'gc1':'#777777','m1':'#0072B2','gc025':'#009E73'}
    labels={'gc1':'GraphCast 1°','m1':'GraphCast–Mamba','gc025':'GraphCast 0.25°'}
    units=['m² s⁻²','m² s⁻²','K','K','K','Pa','kg kg⁻¹','kg kg⁻¹','m s⁻¹','m s⁻¹','m s⁻¹','m s⁻¹']
    for selected,filename in [(list(range(12)),'common_grid_all'),([0,4,8],'common_grid_selected')]:
        rows,cols=(3,4) if len(selected)==12 else (1,3)
        fig,axes=plt.subplots(rows,cols,figsize=(11,7.5) if rows==3 else (9,3),squeeze=False)
        for ax,j in zip(axes.flat,selected):
            for m in ('gc1','gc025','m1'):
                ax.plot(days,rmse[m][:,j],color=colors[m],ls='--' if m=='gc1' else '-',label=labels[m])
            ax.set(title=names[j],ylabel=f'RMSE ({units[j]})',xlabel='Forecast lead (days)'); ax.grid(alpha=.2)
        handles,legends=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,legends,loc='upper center',ncol=3,frameon=False)
        fig.tight_layout(rect=(0,0,1,.93))
        for suffix in ('pdf','png'): fig.savefig(plot_dir/f'{filename}.{suffix}',dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(3,4,figsize=(11,7.5))
    comparison=summary['analyses']['primary']['comparisons']['gc025']
    for j,ax in enumerate(axes.flat):
        ax.axhline(0,color='#888888',lw=.8)
        ax.plot(days,np.asarray(comparison['rmse_reduction_pct'])[:,j],color=colors['m1'])
        ax.set(title=names[j],xlabel='Forecast lead (days)',ylabel='RMSE reduction (%)'); ax.grid(alpha=.2)
    fig.suptitle('GraphCast–Mamba vs GraphCast 0.25°')
    fig.tight_layout()
    for suffix in ('pdf','png'): fig.savefig(plot_dir/f'common_grid_reduction.{suffix}',dpi=180)
    plt.close(fig)


def merge(workflow):
    import csv
    protocol, values, sources = gather(workflow)
    times, names = protocol['initialization_times'], protocol['verification']['diagnostics']
    settings = protocol['bootstrap']
    exposed = set(protocol.get('prior_exposure',{}).get('sampled_initialization_times',[]))
    rmse, analyses = analyze_values(values, times, settings, exposed)
    report = {'protocol_hash':protocol['protocol_hash'],'initializations':len(times),
              'lead_hours':protocol['lead_hours'],'diagnostics':names,
              'rmse':{m:v.tolist() for m,v in rmse.items()},'analyses':analyses,
              'bootstrap':settings,'sources':sources,
              'interpretation':protocol['interpretation'], 'verification':protocol['verification']}
    directory = workflow / 'report'
    write(directory / 'summary.json', report)
    table_dir, plot_dir = Path(protocol['table_directory']), Path(protocol['plot_directory'])
    table_dir.mkdir(parents=True,exist_ok=True)
    plot_dir.mkdir(parents=True,exist_ok=True)
    with (table_dir / 'rmse.csv').open('w',newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['diagnostic','lead_hours','gc1_rmse_si','m1_rmse_si','gc025_rmse_si',
                         'm1_vs_gc1_reduction_pct','m1_vs_gc025_reduction_pct','m1_vs_gc025_ci95_lower','m1_vs_gc025_ci95_upper'])
        a = analyses['primary']['comparisons']
        for k,lead in enumerate(protocol['lead_hours']):
            for j,name in enumerate(names):
                writer.writerow([name,lead,*[rmse[m][k,j] for m in ('gc1','m1','gc025')],
                    a['gc1']['rmse_reduction_pct'][k][j],a['gc025']['rmse_reduction_pct'][k][j],
                    a['gc025']['ci95_lower'][k][j],a['gc025']['ci95_upper'][k][j]])
    with (table_dir / 'per_initialization_mse.csv').open('w',newline='') as handle:
        writer=csv.writer(handle); writer.writerow(['initialization','lead_hours','diagnostic','gc1_mse_si','m1_mse_si','gc025_mse_si'])
        for i,stamp in enumerate(times):
            for k,lead in enumerate(protocol['lead_hours']):
                for j,name in enumerate(names):
                    writer.writerow([stamp,lead,name,*[values[m][i,k,j] for m in ('gc1','m1','gc025')]])
    plot_figures(report, plot_dir)
    comparison=analyses['primary']['comparisons']['gc025']
    lines=['# 2023 common-grid comparison','', 'All three fixed models: 122 identical starts, 40 six-hour leads, common native-ERA5-derived 1.5° truth.',
           '', 'RMSE pools physical squared errors across initializations. Intervals are pointwise paired five-start-block bootstrap intervals; they do not measure training uncertainty.',
           '', '| Diagnostic | GC 1° day-10 RMSE | Mamba day-10 RMSE | GC 0.25° day-10 RMSE | Mamba vs 0.25° reduction |',
           '|---|---:|---:|---:|---:|']
    for j,name in enumerate(names):
        lines.append(f"| {name} | {rmse['gc1'][-1,j]:.5g} | {rmse['m1'][-1,j]:.5g} | {rmse['gc025'][-1,j]:.5g} | {comparison['rmse_reduction_pct'][-1][j]:.2f}% |")
    lines += ['',protocol['interpretation'],'','Values use native SI units. No ENS normalization is used. Prior-pilot exclusion and block-length sensitivities are retained in summary.json.']
    (directory/'report.md').write_text('\n'.join(lines)+'\n')
    print(str(directory/'report.md'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','submit','merge'])
    parser.add_argument('--workflow',type=Path,required=True)
    parser.add_argument('--phase',choices=['smoke','production'],default='smoke')
    parser.add_argument('--model',choices=['m1','gc025'])
    parser.add_argument('--mem',default='16G')
    parser.add_argument('--time',default='01:02:00')
    parser.add_argument('--shards',type=int)
    args=parser.parse_args()
    workflow=args.workflow.resolve()
    if args.command=='prepare': prepare(workflow)
    elif args.command=='merge': merge(workflow)
    else:
        if args.model is None: parser.error('submit requires --model')
        submit(workflow,args.phase,args.model,mem=args.mem,walltime=args.time,shards=args.shards)


if __name__=='__main__': main()
