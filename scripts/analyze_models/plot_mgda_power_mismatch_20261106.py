"""Plot forecast/truth power and absolute relative power mismatch."""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'plots/analyze_models/data/resolution_eval/mgda_comparison_20261106'
IMG=ROOT/'plots/analyze_models/images/resolution_eval/mgda_comparison_20261106'
sources={}
for group,year in [('historical',2023),('current',2022)]:
    p=DATA/f'actual_spectra_{group}.npz'
    sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    with np.load(p) as z:
        powers=z['power'];labels=z['labels'].tolist();fields=z['fields'].tolist();k=z['wavenumber']
    assert labels[0]=='ERA5 truth' and np.all(powers[0]>0)
    ratios=powers/powers[0]
    errors=np.abs(ratios-1)
    assert np.isfinite(errors).all()
    colors=['black','#777777','#0072B2','#009E73','#D55E00'] if group=='historical' else ['black','#777777','#0072B2','#CC79A7','#009E73','#D55E00']
    styles=[':','--']+['-']*(len(labels)-2)
    selected=[3,11,27,39]
    with PdfPages(IMG/f'power_matching_{group}_all_fields.pdf') as pdf:
        for vi,field in enumerate(fields):
            fig,axs=plt.subplots(2,4,figsize=(15,7),sharex=True,sharey='row')
            for j,day in enumerate([1,3,7,10]):
                step=day*4-1
                for mi,label in enumerate(labels):
                    axs[0,j].semilogx(k,ratios[mi,step,vi],color=colors[mi],ls=styles[mi],lw=1.5,label=label)
                    axs[1,j].semilogx(k,errors[mi,step,vi]*100,color=colors[mi],ls=styles[mi],lw=1.5)
                axs[0,j].set_title(f'Day {day}')
                axs[1,j].set_xlabel('Zonal wavenumber')
                axs[0,j].text(.025,.96,'Closer to 1 is better',transform=axs[0,j].transAxes,va='top',fontsize=9,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
                axs[1,j].text(.025,.96,'Lower is better; 0 is exact match',transform=axs[1,j].transAxes,va='top',fontsize=9,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
                for ax in axs[:,j]:ax.grid(alpha=.2);ax.set_xlim(1,180)
            axs[0,0].set_ylim(0,1.15*float(ratios[:,selected,vi,:].max()))
            axs[1,0].set_ylim(0,115*float(errors[:,selected,vi,:].max()))
            axs[0,0].set_ylabel('Forecast / truth power: Pf / Pt')
            axs[1,0].set_ylabel('Relative power mismatch (%)\n100 × |Pf − Pt| / Pt')
            fig.suptitle(f'{field}: power matching through the rollout — eight {year} starts',fontsize=15)
            fig.text(.5,.915,'Powers pooled over starts · 30–60° N/S · positive zonal modes · unsmoothed diagnostic',ha='center',fontsize=10)
            fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,fontsize=9)
            fig.tight_layout(rect=(0,.13,1,.90));pdf.savefig(fig)
            fig.savefig(IMG/f'power_matching_{group}_{field}.png',dpi=160)
            plt.close(fig)
    np.savez_compressed(DATA/f'power_matching_{group}.npz',power_ratio=ratios,relative_power_mismatch=errors,labels=labels,fields=fields,wavenumber=k,lead_days=np.arange(1,41)/4)
(DATA/'power_matching_provenance.json').write_text(json.dumps(dict(sources=sources,formulas=['Pf/Pt','abs(Pf-Pt)/Pt'],aggregation='Mean powers over starts before ratios; no smoothing or floor.',training_distinction='Training averages per-example smoothed, floored relative errors; these plots show pooled unsmoothed diagnostics, not exact training loss.'),indent=2)+'\n')
print('Generated both quantities for all nine fields and both cohorts.')
