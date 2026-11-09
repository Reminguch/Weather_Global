"""Plot forecast-error spectral power relative to original GC error (GC=1)."""
from pathlib import Path
import json,hashlib,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'plots/analyze_models/data/resolution_eval/mgda_comparison_20261106'
IMG=ROOT/'plots/analyze_models/images/resolution_eval/mgda_comparison_20261106'
sources={}
BASE=ROOT/'artifacts/checkpoints/v24_Ilya'
def result(c,t):return BASE/c/'evaluation/results'/t/'spectral'
directories={
 'historical':[
 ROOT/'plots/analyze_models/data/resolution_eval/res1_spectral_pilot_2023_20260917/models/M1',
 result('res1_mgda_gc_power_2k_recovery_20261007','mgda_swa'),
 result('res1_amse_split_mgda_di16_di32_recovery_20261013','di16_step02000')],
 'current':[
 result('res1_mse_amse_excess_full_bptt24_20261028','parent'),
 result('res1_mse_amse_excess_full_bptt24_20261028','mgda_step00200'),
 result('res1_mse_amse_excess_advisory_20261028','mgda_step00200'),
 result('res1_mse_amse_excess_advisory_20261028','fixed_sum_step00200')]}
summary=[]
def direction_labels(ax):
    ax.axhspan(0,1,color='#009E73',alpha=.055,zorder=0)
    ax.text(.025,.98,'↑ Worse than GC (>1)',transform=ax.transAxes,
            va='top',fontsize=9,color='#A33A20',
            bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=2))
    ax.text(.025,.025,'↓ Better than GC (<1)',transform=ax.transAxes,
            va='bottom',fontsize=9,color='#007451',
            bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=2))
for group,year in [('historical',2023),('current',2022)]:
    path=DATA/f'actual_spectra_{group}.npz'
    sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path) as z:
        labels=z['labels'].tolist()[1:];fields=z['fields'].tolist();k=z['wavenumber'];days=z['lead_days']
    models=[];canonical_dates=None
    for directory in directories[group]:
        values=[];baselines=[];dates=[]
        evaluation=json.loads((directory/'evaluation.json').read_text())
        assert evaluation['evaluation_status']=='complete'
        for file in sorted(directory.glob('*.npz')):
            sources[str(file.relative_to(ROOT))]=hashlib.sha256(file.read_bytes()).hexdigest()
            with np.load(file) as z:
                meta=json.loads(str(z['metadata']))
                assert meta['checkpoint_sha256']==evaluation['checkpoint_sha256']
                dates.append(meta['initialization_time'])
                ix=[z['variables'].tolist().index(f) for f in fields]
                for branch in ('full','baseline'):
                    error=z[branch+'_error_power'][:,ix,1:]
                    reconstructed=(z[branch+'_power']+z['truth_power']-2*z[branch+'_cross_power'])[:,ix,1:]
                    np.testing.assert_allclose(error,reconstructed,rtol=1e-8,atol=1e-10)
                    assert np.isfinite(error).all() and np.all(error>=0)
                values.append(z['full_error_power'][:,ix,1:])
                baselines.append(z['baseline_error_power'][:,ix,1:])
        assert len(dates)==8 and len(set(dates))==8
        if canonical_dates is None:
            canonical_dates=dates
            models.append(np.mean(baselines,axis=0))
        assert dates==canonical_dates
        models.append(np.mean(values,axis=0))
    powers=np.stack(models)
    base=powers[labels.index('Original GraphCast')]
    assert np.all(base>0)
    ratios=powers/base[None,...]
    np.testing.assert_array_equal(ratios[labels.index('Original GraphCast')],np.ones_like(base))
    assert np.isfinite(ratios).all()
    for mi,label in enumerate(labels):
        for vi,field in enumerate(fields):
            summary.append(dict(cohort=group,model=label,field=field,
                rollout_error_reduction_pct=float(100*(1-powers[mi,:,vi].sum()/base[:,vi].sum())),
                day10_error_reduction_pct=float(100*(1-powers[mi,39,vi].sum()/base[39,vi].sum()))))
    np.savez_compressed(DATA/f'gc_relative_power_{group}.npz',ratios=ratios,labels=labels,fields=fields,wavenumber=k,lead_days=days)
    colors=['#777777','#0072B2','#009E73','#D55E00'] if group=='historical' else ['#777777','#0072B2','#CC79A7','#009E73','#D55E00']
    styles=['--']+['-']*(len(labels)-1)
    with PdfPages(IMG/f'gc_relative_power_{group}_all_fields.pdf') as pdf:
        for vi,field in enumerate(fields):
            fig,axs=plt.subplots(1,4,figsize=(15,4.7),sharex=True,sharey=True)
            for ax,day in zip(axs,[1,3,7,10]):
                for mi,label in enumerate(labels):
                    ax.semilogx(k,ratios[mi,day*4-1,vi],label=label,color=colors[mi],ls=styles[mi],lw=1.6)
                ax.set(title=f'Day {day}',xlabel='Zonal wavenumber',xlim=(1,180),ylim=(0,None));ax.grid(alpha=.2)
            axs[0].set_ylabel('Error power / original GC error power')
            axs[0].set_ylim(0,1.05*float(ratios[:,[3,11,27,39],vi,:].max()))
            for ax in axs:direction_labels(ax)
            fig.suptitle(f'{field}: forecast-error spectrum — GC = 1, lower is better\nEight {year} starts · 30–60° N/S · powers pooled before ratios',fontsize=13)
            fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=3,fontsize=9)
            fig.tight_layout(rect=(0,.18,1,.92));pdf.savefig(fig)
            if field in ('T850','U850','TP6'):fig.savefig(IMG/f'gc_relative_power_{group}_{field}.png',dpi=180)
            plt.close(fig)
    fig,axs=plt.subplots(3,3,figsize=(12,9),sharex=True)
    for vi,field in enumerate(fields):
        ax=axs.flat[vi]
        for mi,label in enumerate(labels):
            ax.semilogx(k,ratios[mi,39,vi],label=label,color=colors[mi],ls=styles[mi],lw=1.4)
        ax.set(title=field,xlim=(1,180),ylim=(0,None));ax.grid(alpha=.2)
        direction_labels(ax)
    fig.suptitle(f'Day 10 forecast-error spectrum: GC = 1, lower is better — eight {year} starts',fontsize=14)
    fig.supylabel('Error power / original GC error power')
    fig.supxlabel('Zonal wavenumber',y=.10)
    fig.legend(*axs.flat[0].get_legend_handles_labels(),loc='lower center',ncol=3,fontsize=9)
    fig.tight_layout(rect=(.02,.13,1,.95))
    for ext in ('png','pdf'):fig.savefig(IMG/f'gc_relative_power_{group}_day10.{ext}',dpi=180)
    plt.close(fig)
with (DATA/'spectral_error_reduction_by_field.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
(DATA/'gc_relative_power_provenance.json').write_text(json.dumps(dict(sources=sources,formula='mean(|FFT(forecast-truth)|^2) / mean(|FFT(GC-truth)|^2)',interpretation='GC=1; below1 means smaller spectral forecast error. Includes amplitude and phase errors. Not abs(P_forecast-P_truth)/P_truth.',validation='Saved error power agrees with P_forecast+P_truth-2*cross_power; all values finite; GC ratio exactly1.',aggregation='Pool starts before ratios; per-field rollout/day10 reductions sum positive-mode error powers before ratios. Fields are not pooled in mixed physical units.'),indent=2)+'\n')
print('Generated corrected GC=1 error-spectrum plots and per-field rollout/day10 reductions.')
