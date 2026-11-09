"""Absolute power mismatch relative to canonical vanilla GC mismatch."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'plots/analyze_models/data/resolution_eval/mgda_comparison_20261106'
IMG=ROOT/'plots/analyze_models/images/resolution_eval/mgda_comparison_20261106'
provenance={}
for group,year in [('historical',2023),('current',2022)]:
 p=DATA/f'actual_spectra_{group}.npz'
 with np.load(p) as z:
  power=z['power'];labels=z['labels'].tolist();fields=z['fields'].tolist();k=z['wavenumber']
 truth=power[0];gc=power[labels.index('Original GraphCast')]
 denominator=np.abs(gc-truth)
 numerator=np.abs(power[1:]-truth)
 ratio=np.full_like(numerator,np.nan)
 np.divide(numerator,denominator[None],out=ratio,where=denominator[None]>0)
 labels=labels[1:]
 assert np.all(np.isfinite(ratio[:,denominator>0]))
 np.testing.assert_array_equal(ratio[0][denominator>0],np.ones(np.count_nonzero(denominator>0)))
 colors=['#777777','#0072B2','#009E73','#D55E00'] if group=='historical' else ['#777777','#0072B2','#CC79A7','#009E73','#D55E00']
 # Symlog preserves zero and shows denominator-driven peaks without clipping.
 with PdfPages(IMG/f'power_mismatch_vs_gc_{group}_all_fields.pdf') as pdf:
  for vi,field in enumerate(fields):
   fig,axs=plt.subplots(1,4,figsize=(15,4.8),sharex=True,sharey=True)
   selected=[3,11,27,39]
   for ax,day,step in zip(axs,[1,3,7,10],selected):
    for mi,label in enumerate(labels):
     ax.plot(k,100*ratio[mi,step,vi],label=label,color=colors[mi],ls='--' if mi==0 else '-',lw=1.5)
    ax.set(xscale='log',yscale='symlog',title=f'Day {day}',xlabel='Zonal wavenumber',xlim=(1,180))
    ax.set_yscale('symlog',linthresh=100,linscale=1)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f'{value:,.0f}%'))
    ax.axhspan(0,100,color='#009E73',alpha=.06)
    ax.grid(alpha=.2)
    ax.text(.02,.98,'↑ Worse than GC (>100%)',transform=ax.transAxes,va='top',fontsize=9,color='#A33A20',bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
    ax.text(.02,.02,'↓ Better than GC (<100%)',transform=ax.transAxes,va='bottom',fontsize=9,color='#007451',bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
   maximum=np.nanmax(ratio[:,selected,vi])
   axs[0].set_ylim(0,100*max(1.4,maximum*1.4))
   axs[0].set_ylabel('Power mismatch (% of vanilla GC)')
   fig.suptitle(f'{field}: absolute power mismatch relative to vanilla GC — eight {year} starts',fontsize=14)
   fig.text(.5,.89,'GC = 100% · 100 × |Pmodel − Ptruth| / |PGC − Ptruth| · linear to 100%, logarithmic above · no denominator floor',ha='center',fontsize=9)
   fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=3,fontsize=9)
   fig.tight_layout(rect=(0,.17,1,.87));pdf.savefig(fig)
   fig.savefig(IMG/f'power_mismatch_vs_gc_{group}_{field}.png',dpi=170)
   plt.close(fig)
 np.savez_compressed(DATA/f'power_mismatch_vs_gc_{group}.npz',ratio=ratio,denominator=denominator,labels=labels,fields=fields,wavenumber=k,lead_days=np.arange(1,41)/4)
 provenance[group]=dict(source=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),zero_denominator_bins=int(np.count_nonzero(denominator==0)),maximum_ratio=float(np.nanmax(ratio)))
(DATA/'power_mismatch_vs_gc_provenance.json').write_text(json.dumps(dict(formula='abs(mean(P_model)-mean(P_truth))/abs(mean(P_GC)-mean(P_truth))',display='100 times ratio; vanilla GC=100%; NPZ retains unscaled ratios',cohorts=provenance,zero_policy='Exactly zero GC mismatch produces an undefined ratio, saved as NaN and omitted. No epsilon floor or clipping.',axis='Linear0–100%, logarithmic above100% to display near-zero-denominator peaks; lower is better.'),indent=2)+'\n')
print(json.dumps(provenance,indent=2))
