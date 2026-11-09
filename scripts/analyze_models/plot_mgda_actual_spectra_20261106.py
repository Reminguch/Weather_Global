"""Reproduce actual saved power spectra, not improvement percentages."""
from pathlib import Path
import sys,json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.analyze_models.report_v24_spectral_pilot import VARIABLE_UNITS
OUT=ROOT/'plots/analyze_models/data/resolution_eval/mgda_comparison_20261106'
IMG=ROOT/'plots/analyze_models/images/resolution_eval/mgda_comparison_20261106'
BASE=ROOT/'artifacts/checkpoints/v24_Ilya'
fields=['T2m','U10','V10','MSLP','TP6','T850','Q700','U850','V850']
days=[1,3,7,10]; sources={}
def load(p):
    es=json.loads((p/'evaluation.json').read_text());assert es['evaluation_status']=='complete'
    records=[];dates=[]
    for f in sorted(p.glob('*.npz')):
        sources[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
        with np.load(f) as z:
            meta=json.loads(str(z['metadata']));assert meta['checkpoint_sha256']==es['checkpoint_sha256']
            dates.append(meta['initialization_time'])
            ix=[z['variables'].tolist().index(v) for v in fields]
            records.append({k:z[k][:,ix,1:].copy() for k in ('truth_power','baseline_power','full_power')})
    assert len(records)==8 and len(set(dates))==8
    return {k:np.mean([r[k] for r in records],axis=0) for k in records[0]},dates
m1=ROOT/'plots/analyze_models/data/resolution_eval/res1_spectral_pilot_2023_20260917/models/M1'
def result(c,t):return BASE/c/'evaluation/results'/t/'spectral'
groups={
 'historical': [('Main-text GC-Mamba (M1)',m1,'#0072B2','-'),('Relative-power MGDA SWA',result('res1_mgda_gc_power_2k_recovery_20261007','mgda_swa'),'#009E73','-'),('Amplitude/pattern MGDA di16',result('res1_amse_split_mgda_di16_di32_recovery_20261013','di16_step02000'),'#D55E00','-')],
 'current':[('Pilot parent (di16 joint)',result('res1_mse_amse_excess_full_bptt24_20261028','parent'),'#0072B2','-'),('Full-BPTT MGDA200',result('res1_mse_amse_excess_full_bptt24_20261028','mgda_step00200'),'#CC79A7','-'),('Stopped-feedback MGDA200',result('res1_mse_amse_excess_advisory_20261028','mgda_step00200'),'#009E73','-'),('Stopped fixed sum200',result('res1_mse_amse_excess_advisory_20261028','fixed_sum_step00200'),'#D55E00',':')]}
allrows=[];audit=[];k=np.arange(1,181)
for group,models in groups.items():
    loaded=[]
    for name,path,color,style in models:
        d,t=load(path);loaded.append((name,d,color,style,t))
    dates=loaded[0][4];assert all(x[4]==dates for x in loaded)
    truth=loaded[0][1]['truth_power'];gc=loaded[0][1]['baseline_power']
    curves=[('ERA5 truth',truth,'black','-'),('Original GraphCast',gc,'#777777','--')]+[(n,d['full_power'],c,s) for n,d,c,s,_ in loaded]
    year=dates[0][:4]
    np.savez_compressed(OUT/f'actual_spectra_{group}.npz',wavenumber=k,lead_days=np.arange(1,41)/4,fields=fields,labels=[c[0] for c in curves],power=np.stack([c[1] for c in curves]))
    with PdfPages(IMG/f'actual_spectra_{group}_all_fields.pdf') as pdf:
        for vi,field in enumerate(fields):
            fig,axs=plt.subplots(2,4,figsize=(15,7.5),sharex=True)
            for j,day in enumerate(days):
                step=day*4-1
                for name,p,color,ls in curves:
                    axs[0,j].loglog(k,p[step,vi],label=name,color=color,ls=ls,lw=1.6)
                    axs[1,j].semilogx(k,p[step,vi]/truth[step,vi],label=name,color=color,ls=ls,lw=1.6)
                axs[0,j].set_title(f'Day {day}');axs[1,j].set_xlabel('Zonal wavenumber')
                for ax in axs[:,j]:ax.grid(alpha=.2);ax.set_xlim(1,180)
                axs[1,j].set_ylim(bottom=0)
            axs[0,0].set_ylabel(f'Power ({VARIABLE_UNITS[field]})');axs[1,0].set_ylabel('Forecast / ERA5 power')
            fig.suptitle(f'{field}: actual spectra through the rollout — eight {year} origins',fontsize=15)
            fig.text(.5,.91,'30–60° N/S · positive zonal modes · powers pooled over starts · no smoothing',ha='center')
            fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,fontsize=9)
            fig.tight_layout(rect=(0,.11,1,.90));pdf.savefig(fig)
            if field in ('T850','U850','TP6'):fig.savefig(IMG/f'actual_spectra_{group}_{field}.png',dpi=160)
            plt.close(fig)
    # Compact reproduction of the main-text day10 plot, with no improvement axis.
    fig,axs=plt.subplots(3,3,figsize=(11,9),sharex=True,sharey=True)
    for vi,field in enumerate(fields):
        for name,p,color,ls in curves:
            if group=='historical' and name not in ('ERA5 truth','Original GraphCast','Main-text GC-Mamba (M1)'):continue
            axs.flat[vi].semilogx(k,p[39,vi]/truth[39,vi],label=name,color=color,ls=ls,lw=1.5)
        axs.flat[vi].set_title(field);axs.flat[vi].grid(alpha=.2);axs.flat[vi].set_xlim(1,180)
    fig.suptitle(f'Day 10: forecast / ERA5 power — eight {year} starts',fontsize=15)
    fig.supylabel('Forecast / ERA5 spectral power (1 = correct power)')
    fig.supxlabel('Zonal wavenumber',y=.08)
    fig.legend(*axs.flat[0].get_legend_handles_labels(),loc='lower center',ncol=3,fontsize=9)
    fig.tight_layout(rect=(.02,.11,1,.95))
    for ext in ('png','pdf'):fig.savefig(IMG/f'power_ratio_day10_{group}.{ext}',dpi=160)
    plt.close(fig)
    for name,p,color,ls in curves[1:]:
        for day in days:
            step=4*day-1
            for vi,field in enumerate(fields):
                for ki in range(180):
                    allrows.append(dict(group=group,model=name,field=field,day=day,wavenumber=int(k[ki]),truth_power=float(truth[step,vi,ki]),forecast_power=float(p[step,vi,ki])))
            err=100*np.abs(p[step]-truth[step])/truth[step]
            audit.append(dict(group=group,model=name,day=day,mean_relative_power_error_pct=float(err.mean()),mean_error_k1_10_pct=float(err[:,:10].mean()),mean_error_k11_60_pct=float(err[:,10:60].mean()),mean_error_k61_180_pct=float(err[:,60:].mean())))
    if group=='historical':
        # Independent reproduction of the pre-existing diagnostic CSV for M1.
        prior=ROOT/'plots/analyze_models/data/resolution_eval/res1_spectral_pilot_2023_20260917/diagnostic/power_spectrum_relative_error.csv'
        vals={'baseline':[],'full':[]}
        with prior.open() as f:
            for row in csv.DictReader(f):
                if row['model_id']=='M1' and row['variable'] in fields and int(row['lead_hours']) in (24,72,168,240):vals[row['branch']].append(float(row['relative_power_error_pct']))
        for branch in vals:
            actual=np.mean([a['mean_relative_power_error_pct'] for a in audit if a['model']==('Original GraphCast' if branch=='baseline' else 'Main-text GC-Mamba (M1)')])
            np.testing.assert_allclose(actual,np.mean(vals[branch]),rtol=1e-12)
        print('Independent original diagnostic CSV reproduced:',{b:float(np.mean(v)) for b,v in vals.items()})
for name,rows in [('actual_spectra.csv',allrows),('power_metric_audit.csv',audit)]:
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(OUT/'actual_spectra_provenance.json').write_text(json.dumps(dict(sources=sources,method='Mean raw powers across eight starts; actual powers and forecast/truth ratios. No improvement percentages plotted.',validation='Historical M1 and GC aggregate relative-power errors reproduce independent original diagnostic CSV to rtol1e-12.'),indent=2)+'\n')
for a in audit:
 if a['group']=='historical' and a['model'] in ('Original GraphCast','Main-text GC-Mamba (M1)'):print(a)
