#!/usr/bin/env python3
"""Plot actual forecast valid time for the new five-term K1/K2 checkpoints."""
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

SPECS = (
    ('T850','temperature',850,1.,'Temperature at 850 hPa','K'),
    ('Z500','geopotential',500,1.,'Geopotential at 500 hPa',r'm$^2$ s$^{-2}$'),
    ('U850','u_component_of_wind',850,1.,'Eastward wind at 850 hPa',r'm s$^{-1}$'),
    ('V850','v_component_of_wind',850,1.,'Northward wind at 850 hPa',r'm s$^{-1}$'),
    ('Q700','specific_humidity',700,1000.,'Specific humidity at 700 hPa',r'g kg$^{-1}$'),
    ('CI250','specific_cloud_ice_water_content',250,1000.,'Cloud ice at 250 hPa',r'g kg$^{-1}$'),
    ('CL850','specific_cloud_liquid_water_content',850,1000.,'Cloud liquid water at 850 hPa',r'g kg$^{-1}$'),
)
SERIES = (('era5','ERA5 truth','#222222','-','o'),
          ('baseline','Frozen NGCM','#0072B2','--','s'),
          ('residual','Residual NGCM','#D55E00','-.','^'))
LOCATIONS = (('Global mean','global'),('New York','new_york'),('Beijing','beijing'))


def csv_read(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def csv_write(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(root, output, widths=(128,256)):
    plan = json.loads((root/'plan.json').read_text())
    plan['runs'] = {name:choice for name,choice in plan['runs'].items() if choice['width'] in widths}
    values, errors, reports = [], [], {}
    for name, choice in plan['runs'].items():
        folder = root/name
        report = json.loads((folder/'report.json').read_text())
        if (not report['passed'] or report['smoke'] or not report['production_eval_agreement']
                or report['checkpoint_sha256'] != choice['checkpoint_sha256']
                or report['origins'] != plan['origins']):
            raise ValueError(f'Evaluation receipt does not match frozen plan: {name}')
        for filename, digest in report['artifacts'].items():
            if checksum(folder/filename) != digest:
                raise ValueError(f'Evaluation artifact changed: {name}/{filename}')
        v, e = csv_read(folder/'timeseries.csv'), csv_read(folder/'physical_errors.csv')
        if len(v) != 12*choice['K']*7*37*3 or len(e) != 12*choice['K']*7*37:
            raise ValueError(f'Missing origin, lead, field, level or location: {name}')
        for row in v:
            if not np.isfinite([float(row[k]) for k in ('era5','baseline','residual')]).all():
                raise ValueError('Nonfinite physical time series')
            expected = np.datetime64(row['origin'],'h')+np.timedelta64(int(row['lead_hours']),'h')
            if expected != np.datetime64(row['valid_time'],'h'):
                raise ValueError('Forecast point has incorrect physical valid time')
        values.extend(v); errors.extend(e); reports[name] = report
    # A paired baseline must be identical across widths and at the common +6 h lead.
    reference = {}
    for row in values:
        key = tuple(row[k] for k in ('origin','lead_hours','field','pressure_hpa','location'))
        pair = [float(row[k]) for k in ('era5','baseline')]
        if key in reference:
            np.testing.assert_allclose(pair, reference[key], rtol=0, atol=0)
        else:
            reference[key] = pair
    csv_write(output/'timeseries.csv', values)
    csv_write(output/'physical_errors.csv', errors)
    summary = []
    for name, choice in plan['runs'].items():
        for lead in range(6,choice['K']*6+1,6):
            for field in reports[name]['variables']:
                for pressure in reports[name]['pressure_hpa']:
                    rows = [r for r in errors if r['run_id']==name and int(r['lead_hours'])==lead
                        and r['field']==field and float(r['pressure_hpa'])==pressure]
                    base = float(np.sqrt(np.mean([float(r['baseline_mse']) for r in rows])))
                    residual = float(np.sqrt(np.mean([float(r['residual_mse']) for r in rows])))
                    summary.append(dict(run_id=name,lead_hours=lead,field=field,pressure_hpa=pressure,
                        baseline_rmse=base,residual_rmse=residual,rmse_reduction_pct=100*(1-residual/base)))
    csv_write(output/'physical_rmse.csv', summary)
    provenance = dict(exported_at=datetime.now(timezone.utc).isoformat(),plan=plan,reports=reports,
        files={n:checksum(output/n) for n in ('timeseries.csv','physical_errors.csv','physical_rmse.csv')})
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


def draw(output):
    provenance = json.loads((output/'provenance.json').read_text())
    for name, digest in provenance['files'].items():
        if checksum(output/name) != digest:
            raise ValueError('Saved plotting inputs changed')
    data = csv_read(output/'timeseries.csv')
    plan = provenance['plan']
    widths = sorted({choice['width'] for choice in plan['runs'].values()})
    index = {}
    for row in data:
        key = (row['run_id'],int(row['lead_hours']),row['field'],float(row['pressure_hpa']),row['location'])
        index.setdefault(key,[]).append(row)
    with tempfile.TemporaryDirectory(prefix='ngcm-physical-time-mpl-') as cache:
        os.environ['MPLCONFIGDIR'] = cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
            'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'paper-physical-time'})
        def save(fig, stem):
            for ext in ('png','pdf','svg'):
                path = output/f'{stem}.{ext}'
                fig.savefig(path,dpi=180,bbox_inches='tight',facecolor='white')
                if ext == 'svg':
                    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
            plt.close(fig)
        def panel(ax, width, k, lead, spec, location):
            code,field,level,scale,title,unit=spec
            rows=sorted(index[(f'k{k}_w{width}_d16',lead,field,float(level),location)],key=lambda r:int(r['physical_hours']))
            x=[int(r['physical_hours']) for r in rows]
            for key,label,color,line,marker in SERIES:
                ax.plot(x,[float(r[key])*scale for r in rows],label=label,color=color,
                    linestyle=line,marker=marker,markersize=3.3,linewidth=1.5)
            ax.set_ylabel(unit); ax.grid(alpha=.2); ax.set_xlim(0,78)
            ax.set_xticks([0,12,24,36,48,60,72]); ax.ticklabel_format(axis='y',useOffset=False)
        for width in widths:
            for location,slug in LOCATIONS:
                fig,axes=plt.subplots(7,3,figsize=(15,17),sharex=True,sharey='row',layout='constrained')
                for row,spec in enumerate(SPECS):
                    for col,(k,lead) in enumerate(((1,6),(2,6),(2,12))):
                        ax=axes[row,col]; panel(ax,width,k,lead,spec,location)
                        if row==0:
                            step=plan['runs'][f'k{k}_w{width}_d16']['update']
                            ax.set_title(f'K={k} training | +{lead} h forecast\nCheckpoint update {step}',fontsize=11,pad=30)
                        if col==0:
                            ax.text(0,1.06,spec[4],transform=ax.transAxes,fontsize=10,fontweight='bold')
                        if row==6:ax.set_xlabel('Forecast valid time (hours since Jan 10, 00 UTC)')
                axes[0,2].legend(loc='best',fontsize=8)
                fig.suptitle(f'Physical variables versus forecast valid time | w{width}, d16 | {location}\n'
                    '2022-01-10 onward • new five-term training • independent origins every 6 h',fontsize=14)
                save(fig,f'w{width}_{slug}')
            for spec in SPECS:
                fig,axes=plt.subplots(1,3,figsize=(14,3.8),sharey=True,layout='constrained')
                for ax,(k,lead) in zip(axes,((1,6),(2,6),(2,12))):
                    panel(ax,width,k,lead,spec,'Global mean')
                    ax.set_title(f'K={k} | +{lead} h forecast');ax.set_xlabel('Forecast valid time (hours since Jan 10, 00 UTC)')
                axes[-1].legend(fontsize=8)
                fig.suptitle(f'{spec[4]} | Global Gaussian-area mean | w{width}, d16')
                save(fig,f'w{width}_{spec[0]}')
    lines=['# K=1/K=2 physical variables versus physical time','',
        'These figures use the new five-term training checkpoints. The horizontal axis is',
        '**forecast valid time**, measured in hours since **2022-01-10 00 UTC**.',
        'Each panel shows ERA5 truth, the official frozen deterministic 2.8° NeuralGCM,',
        'and the residual model in physical units.','','![w128 global physical time](w128_global.png)','',
        'The columns separate K=1 at +6 h, K=2 at +6 h, and K=2 at +12 h.',
        'There are 12 common origins, one every 6 h from January 10 00 UTC.',
        'The last +12 h forecast is valid on January 13 at 06 UTC (offset 78 h).',
        'State and residual memory reset at each origin. Within K=2, the corrected',
        'first forecast feeds the second step without replacing it with ERA5.',
        '**These are successive short forecasts, not a single 72-hour free-running forecast.**','',
        'The seven standard variable/level pairs were declared before evaluation. All',
        '37 levels remain available in the CSV files. Global means can hide local',
        'errors, so New York and Beijing nearest-grid-point curves are also provided.',
        'Cloud and humidity plots use g/kg; CSV values retain kg/kg. Geopotential',
        'is m²/s², not geopotential height. Outputs are shown without clipping,',
        'including negative residual cloud-water predictions.','','| Width | Global | New York | Beijing |',
        '| --- | --- | --- | --- |']
    for width in widths:
        lines.append(f'| {width} | [Figure](w{width}_global.png) | [Figure](w{width}_new_york.png) | [Figure](w{width}_beijing.png) |')
    lines+=['','## Checkpoints and scoring','',plan['selection']+'.','',
        '| Configuration | Selected update | Completed stage |','| --- | ---: | ---: |']
    for name,choice in plan['runs'].items():
        lines.append(f'| {name} | {choice["update"]} | 1000 |')
    metrics = {(r['run_id'],int(r['lead_hours']),r['field'],float(r['pressure_hpa'])):r
               for r in csv_read(output/'physical_rmse.csv')}
    for width in widths:
        lines += ['',f'### Window gridpoint RMSE, w{width}', '',
            '| Variable | Unit | Baseline +6 h | K1 +6 h | K2 +6 h | Baseline +12 h | K2 +12 h |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
        for code,field,pressure,scale,_,_ in SPECS:
            k1=metrics[(f'k1_w{width}_d16',6,field,float(pressure))]
            k2a=metrics[(f'k2_w{width}_d16',6,field,float(pressure))]
            k2b=metrics[(f'k2_w{width}_d16',12,field,float(pressure))]
            unit='K' if field=='temperature' else 'm²/s²' if field=='geopotential' else 'g/kg' if field.startswith('specific_') else 'm/s'
            values=[float(r[k])*scale for r,k in ((k1,'baseline_rmse'),(k1,'residual_rmse'),
                (k2a,'residual_rmse'),(k2b,'baseline_rmse'),(k2b,'residual_rmse'))]
            lines.append(f'| {code} | {unit} | '+' | '.join(f'{x:.4g}' for x in values)+' |')
    lines+=['','Every forecast uses the same data grid, forcing policy, verification targets',
        'and frozen backbone as training. Separate +6/+12 h errors must not be',
        'treated as scores at one common horizon. Window RMSE is computed from',
        'gridpoint squared errors before spatial/time averaging and square root;',
        'it is not the error of the plotted global-mean curves. This three-day',
        'illustration is not full-year evaluation or the paper benchmark.','',
        'The evaluation checks zero-residual equality with the official baseline,',
        'agreement with the production physical evaluator on two origins, unchanged',
        'backbone parameters, all expected fields/levels/leads, finite values, exact',
        'checkpoint/source hashes, and equal paired baselines across configurations.','',
        '[Time series](timeseries.csv) · [Per-origin gridpoint errors](physical_errors.csv) ·',
        '[Window RMSE by variable, level and lead](physical_rmse.csv) · [Provenance](provenance.json)','',
        'PNG, PDF and editable SVG files are provided for every figure.','',
        '```bash','python3 plot/plot_paper_physical_time.py','```','',
        '[Training-step curves and loss/evaluation limitations](../README.md)','']
    (output/'README.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root',type=Path)
    parser.add_argument('--widths',type=int,choices=(128,256),nargs='+',default=[128,256])
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'paper_physical_time_20260924')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.results_root:export(args.results_root,args.output,args.widths)
    draw(args.output)


if __name__=='__main__':main()
