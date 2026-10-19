#!/usr/bin/env python3
"""Compare archived raster curves with new numerical results, without editing either.

Historical values are approximate digitizations, not recovered raw observations.
The raster geometry and tick positions come from rebuild_common_grid_figures.py.
"""
from pathlib import Path
import hashlib
import io
import json
import sys

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
# Reuse the locally available PDF reader after loading the environment's PIL.
sys.path.append(str(ROOT / '.cache/presentation-python'))
import pymupdf


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    workflow = ROOT / 'artifacts/evaluations/graphcast_common_grid_2023_20261015'
    archive = ROOT / 'Overleaf/tools/figure_label_sources/Graphs/add_mamba.pdf'
    summary_path = workflow / 'report/summary.json'
    summary = json.loads(summary_path.read_text())
    with pymupdf.open(archive) as document:
        images = document[0].get_images()
        if len(images) != 1:
            raise ValueError('Expected exactly one archived bitmap')
        pixels = np.asarray(Image.open(io.BytesIO(document.extract_image(images[0][0])['image'])).convert('RGB'))
    if pixels.shape != (1540, 2520, 3):
        raise ValueError('Archived raster geometry changed')
    panels = {
        'Z500': (0, [(50,477.5),(60,429.5),(80,332.5),(100,235.5),(120,139.5)]),
        'T2m': (4, [(50,947.5),(60,899.5),(80,804.5),(100,709.5),(120,614.5)]),
        'U850': (8, [(40,1423.5),(60,1345.5),(80,1267.5),(100,1189.5),(120,1112.5)]),
    }
    colors = {'m1': np.array([43,95,158]), 'gc025': np.array([31,158,78]),
              'gc1': np.array([138,143,152])}
    days = np.asarray(summary['lead_hours']) / 24
    results = {}
    for name, (index, ticks) in panels.items():
        row, col = divmod(index, 4)
        left, top = 98 + 618 * col, 97 + 491 * row
        crop = pixels[top:top+386, left:left+536].astype(float)
        slope, intercept = np.polyfit([p for v,p in ticks], [v for v,p in ticks], 1)
        historical, trace_info = {}, {}
        for model, color in colors.items():
            # Distance <= 12 isolates the saturated curve core; gray threshold
            # is tighter to exclude axes, grid lines, and anti-aliased text.
            mask = np.linalg.norm(crop - color, axis=-1) <= (5 if model == 'gc1' else 12)
            if index == 0:
                # Exclude the original embedded legend, not the curve pixels.
                mask[max(0,105-top):179-top, max(0,105-left):431-left] = False
            xs, ys, widths = [], [], []
            for x in range(3, 533):
                candidates = np.flatnonzero(mask[:,x])
                if len(candidates):
                    if np.ptp(candidates) > 12:
                        continue  # ambiguous column, never choose a favorable branch
                    xs.append(x)
                    ys.append(float(np.median(candidates)) + top)
                    widths.append(int(np.ptp(candidates)))
            lead = (np.asarray(xs) - .5) / 534 * 10
            values = slope * np.asarray(ys) + intercept
            if len(xs) < 100 or lead.min() > .5 or lead.max() < 9.8:
                raise ValueError(f'Insufficient archived curve coverage: {name}/{model}')
            historical[model] = np.interp(days, lead, values)
            trace_info[model] = {'columns':len(xs), 'lead_range': [float(lead.min()),float(lead.max())],
                                 'max_core_thickness_pixels': max(widths)}
        j = summary['diagnostics'].index(name)
        current = {m: np.asarray(summary['rmse'][m])[:,j] for m in colors}
        old_reduction = 100*(1-historical['m1']/historical['gc025'])
        new_reduction = 100*(1-current['m1']/current['gc025'])
        old_baseline = 100*(1-historical['gc025']/historical['gc1'])
        new_baseline = 100*(1-current['gc025']/current['gc1'])
        def sustained_crossing(values):
            return next((float(days[i]) for i in range(len(days)) if np.all(values[i:]>0)), None)
        results[name] = {
            'historical_approx_ens_percent':{m:a.tolist() for m,a in historical.items()},
            'historical_trace_quality':trace_info,
            'historical_approx_m1_vs_gc025_rmse_reduction_pct':old_reduction.tolist(),
            'current_m1_vs_gc025_rmse_reduction_pct':new_reduction.tolist(),
            'historical_approx_gc025_vs_gc1_rmse_reduction_pct':old_baseline.tolist(),
            'current_gc025_vs_gc1_rmse_reduction_pct':new_baseline.tolist(),
            'historical_approx_sustained_crossing_day':sustained_crossing(old_reduction),
            'current_sustained_crossing_day':sustained_crossing(new_reduction),
        }
    audit = {'status':'comparison_complete_baseline_validation_unresolved',
             'historical_values':'Approximate curve digitization from raster; not original numerical results.',
             'historical_correction':'The curve labeled K=22 in the archived figure, not the selected M1 checkpoint.',
             'script_sha256':sha(Path(__file__)),
             'comparison_limits':'Different years, correction checkpoints, and unresolved historical reference provenance; this does not attribute the difference to one cause.',
             'sources':{str(archive.relative_to(ROOT)):sha(archive),str(summary_path.relative_to(ROOT)):sha(summary_path)},
             'lead_days':days.tolist(),'diagnostics':results}
    out = workflow / 'audits/archive_comparison.json'
    out.write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    image_dir = ROOT / 'plots/analyze_models/images/resolution_eval/common_grid_2023_20261015'
    fig, axes = plt.subplots(2,3,figsize=(11,6.8),sharex=True)
    for col,(name,record) in enumerate(results.items()):
        for row,part in enumerate(('m1_vs_gc025','gc025_vs_gc1')):
            ax=axes[row,col]
            ax.axhline(0,color='#666666',lw=.8)
            ax.plot(days,record[f'historical_approx_{part}_rmse_reduction_pct'],color='#D55E00',ls='--',label='Archived figure (approx.; labeled 2022)')
            ax.plot(days,record[f'current_{part}_rmse_reduction_pct'],color='#0072B2',label='New 2023 results')
            ax.set_title(name)
            ax.set_xlim(1,10)
            # Display days 1–10 consistently; all 40 leads remain in the audit.
            visible=days>=1
            vals=np.concatenate([np.asarray(record[f'{prefix}{part}_rmse_reduction_pct'])[visible] for prefix in ('historical_approx_','current_')])
            lo,hi=min(0,float(vals.min())),max(0,float(vals.max()))
            pad=max(1,(hi-lo)*.1)
            ax.set_ylim(lo-pad,hi+pad)
            ax.grid(alpha=.2)
            if row: ax.set_xlabel('Forecast lead (days)')
            if col==0: ax.set_ylabel(('Mamba vs GC 0.25°' if row==0 else 'GC 0.25° vs GC 1°')+'\nRMSE reduction (%)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False)
    fig.suptitle('Audit comparison: positive values favor the first model',y=.94,fontsize=11)
    fig.tight_layout(rect=(0,0,1,.91))
    for suffix in ('png','pdf'):
        fig.savefig(image_dir/f'audit_archive_comparison.{suffix}',dpi=180)
    plt.close(fig)
    for name,record in results.items():
        print(name,'day10 old/new',record['historical_approx_m1_vs_gc025_rmse_reduction_pct'][-1],record['current_m1_vs_gc025_rmse_reduction_pct'][-1],
              'sustained crossing old/new',record['historical_approx_sustained_crossing_day'],record['current_sustained_crossing_day'])
    print(out)


if __name__=='__main__':
    main()
