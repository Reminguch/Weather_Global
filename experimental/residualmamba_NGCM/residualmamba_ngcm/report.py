"""Render immutable evaluation arrays without selecting checkpoints on metrics."""
from pathlib import Path
import numpy as np
from .config import LEVELS
from .artifacts import load_arrays,sha256,write_json


def render(summary,output):
    data,meta=load_arrays(summary)
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Report output already exists')
    output.mkdir(parents=True,exist_ok=True)
    lines=['# Open-loop NeuralGCM residual evaluation','',
        f"Origins: {meta['origins']}. Warm-3: {meta['warm']}. Percentages below are MSE improvements.",'',
        'Positive values mean lower MSE than native NeuralGCM. Absolute RMSE is on the 2.8-degree grid.',
        '2022 has historical tuning exposure; the held-out year is 2023.','',
        '| Quantity | 120 h MSE improvement | 240 h MSE improvement |','|---|---:|---:|']
    for name,v,p in [('Z500',1,500),('T850',0,850),('U850',2,850),('V850',3,850),('Q700',4,700)]:
        x=data['mse_improvement_pct'][:,v,LEVELS.index(p)]
        lines.append(f'| {name} | {x[19]:.3f}% | {x[39]:.3f}% |')
    b,c=data['J_native'],data['J_corrected']
    lines.extend(['',f'J improvement at 120 h: {100*(1-c[19]/b[19]):.3f}%.',
        f'J improvement at 240 h: {100*(1-c[39]/b[39]):.3f}%.',
        f'Mean-over-leads J improvement: {100*(1-c.mean()/b.mean()):.3f}%.','',
        'The NPZ also contains every field, pressure level and lead, paired weekly bootstrap intervals,',
        'ACC, and Z500/T850/U850 spherical power spectra at 24/120/240 h.',''])
    (output/'REPORT.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    hours=np.arange(6,241,6)
    axes[0].plot(hours,b,label='Native NGCM');axes[0].plot(hours,c,label='Corrected')
    axes[0].set(xlabel='Lead time (hours)',ylabel='J (weighted normalized MSE)');axes[0].legend()
    improvement=100*(1-c/b)
    axes[1].plot(hours,improvement)
    lo,hi=data['J_improvement_ci95']
    axes[1].fill_between(hours,lo,hi,alpha=.2)
    axes[1].axhline(0,color='gray',linewidth=.7)
    axes[1].set(xlabel='Lead time (hours)',ylabel='MSE improvement (%)')
    fig.savefig(output/'forecast_skill.png',dpi=180)
    plt.close(fig)
    write_json(output/'manifest.json',{'stage':'report','summary_sha256':sha256(summary),
        'report_sha256':sha256(output/'REPORT.md'),'figure_sha256':sha256(output/'forecast_skill.png')},immutable=True)
