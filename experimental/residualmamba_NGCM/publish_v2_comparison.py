#!/usr/bin/env python3
"""Publish both completed v2 arms with common-validation plots and source values."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIELDS=[('temperature','Temperature'),('geopotential','Geopotential'),
        ('u_component_of_wind','Zonal wind (u)'),('v_component_of_wind','Meridional wind (v)'),
        ('specific_humidity','Specific humidity')]
COLORS={'tail20':'#2563eb','tail1':'#d97706'}
LABELS={'tail20':'Train K=20','tail1':'Train K=1'}


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def save(fig,path):
    fig.savefig(path,dpi=180,bbox_inches='tight');plt.close(fig)
def style(ax,xlabel,ylabel):
    ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.grid(alpha=.18)
def improvement(c,b):return 100*(1-np.asarray(c)/np.asarray(b))


def publish(runs,out):
    out.mkdir(parents=True,exist_ok=True)
    training={};evaluations={};checkpoints={};metadata={};time_hours={}
    for arm,root in runs.items():
        assert read(root/'report/COMPLETE.json')['step']==2000
        assert read(root/'controller/status.json')['phase']=='complete'
        plan=read(root/'plan.json');run=read(root/'checkpoints/run.json')
        assert run['identity']==plan['identity'] and run['config']['schema_version']==2
        assert run['config']['ar_tail_k']==(20 if arm=='tail20' else 1)
        raw=[json.loads(line) for line in (root/'checkpoints/training.jsonl').read_text().splitlines()]
        records={r['step']:r for r in raw};assert set(records)==set(range(1,2001))
        training[arm]=[records[i] for i in range(1,2001)]
        time_hours[arm]=np.cumsum([r['seconds'] for r in training[arm]])/3600
        assert np.isfinite([r['loss'] for r in training[arm]]).all()
        ck={read(p)['step']:read(p) for p in (root/'checkpoints').glob('step_*.json') if p.with_suffix('.pkl').exists()}
        vals=sorted([read(p) for p in (root/'eval'/arm).glob('step_*/summary.json')],key=lambda d:d['step'])
        assert set(ck)=={v['step'] for v in vals},'Every saved checkpoint must have a complete eval'
        for v in vals:
            assert v['arm']==arm and v['segments']==15 and v['chunks']==60
            assert v['identity']==plan['identity'] and v['checkpoint_sha256']==ck[v['step']]['sha256']
            assert set(v['per_field'])=={f for f,_ in FIELDS}
            np.testing.assert_allclose(sum(v['per_field'].values()),v['mean_loss'],rtol=1e-7)
        assert vals[-1]['step']==2000
        evaluations[arm]=vals;checkpoints[arm]=sorted(ck)
        metadata[arm]={'root':str(root),'source_commit':plan['repo_commit'],'identity':plan['identity'],
            'training_log_sha256':sha(root/'checkpoints/training.jsonl'),
            'plan_sha256':sha(root/'plan.json'),'checkpoints_evaluated':len(vals),
            'final_checkpoint_sha256':vals[-1]['checkpoint_sha256'],
            'stats_sha256':plan['prerequisites']['stats.npz'],
            'common_census_sha256':plan['prerequisites']['census.json']}
        shutil.copyfile(root/'eval'/arm/'step_02000/summary.json',out/f'{arm}_eval_step_02000.json')
    assert metadata['tail20']['stats_sha256']==metadata['tail1']['stats_sha256']
    assert metadata['tail20']['common_census_sha256']==metadata['tail1']['common_census_sha256']
    reference=evaluations['tail20'][0]
    for values in evaluations.values():
        for v in values:
            np.testing.assert_allclose(v['baseline_mean_loss'],reference['baseline_mean_loss'],rtol=1e-9)
            for f,_ in FIELDS:np.testing.assert_allclose(v['baseline_per_field'][f],reference['baseline_per_field'][f],rtol=1e-9)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
    # Separate axes: the two training objectives cover different forecast horizons.
    fig,axes=plt.subplots(2,1,figsize=(11,7),layout='constrained',sharex=True)
    for ax,arm in zip(axes,runs):
        loss=np.array([r['loss'] for r in training[arm]])
        rolling=np.convolve(loss,np.ones(100)/100,'valid')
        ax.plot(np.arange(1,2001),loss,color=COLORS[arm],alpha=.18,lw=.7,label='Raw updates')
        ax.plot(np.arange(100,2001),rolling,color=COLORS[arm],lw=2,label='100-update mean')
        pts=checkpoints[arm];means=[loss[max(0,s-100):s].mean() for s in pts]
        ax.scatter(pts,means,color=COLORS[arm],s=18,zorder=5,label='Saved checkpoint')
        ax.set_title(LABELS[arm],loc='left');style(ax,'Optimizer update / checkpoint','Training loss');ax.legend(loc='upper right',fontsize=9)
    fig.suptitle('Training loss through update 2,000\nDifferent training horizons; use common validation to compare forecast skill.',fontsize=13)
    save(fig,out/'train_loss_vs_checkpoint.png')
    fig,ax=plt.subplots(figsize=(10.5,5.2),layout='constrained')
    baseline=reference['baseline_mean_loss']
    ax.axhline(baseline,ls='--',color='#64748b',label=f'Official NeuralGCM baseline: {baseline:.4f}')
    for arm,values in evaluations.items():
        ax.plot([0]+[v['step'] for v in values],[baseline]+[v['mean_loss'] for v in values],
                marker='o',markersize=3,lw=2,color=COLORS[arm],label=LABELS[arm])
        v=values[-1];ax.annotate(f"{v['mean_loss']:.4f}  ({v['improvement_percent']:.2f}% lower)",
            (2000,v['mean_loss']),xytext=(-7,-19 if arm=='tail20' else -25),textcoords='offset points',ha='right',color=COLORS[arm],fontweight='bold')
    style(ax,'Checkpoint update','Normalized weighted validation MSE')
    ax.set_title('Supplementary: both models evaluated with K=20 inputs\n2022: 15 fixed segments / 60 chunks, identical tail20 inputs')
    ax.set_xlim(0,2100);ax.margins(y=.15);ax.legend(loc='lower left',fontsize=10)
    save(fig,out/'eval_loss_vs_checkpoint.png')
    fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained');axes=axes.ravel()
    for ax,(field,label) in zip(axes,FIELDS):
        baseline_field=reference['baseline_per_field'][field]
        ax.axhline(baseline_field,ls='--',lw=1,color='#64748b')
        for arm,values in evaluations.items():
            ys=[v['per_field'][field] for v in values]
            ax.plot([0]+[v['step'] for v in values],[baseline_field]+ys,marker='o',markersize=2.5,lw=1.8,color=COLORS[arm],label=LABELS[arm])
            gain=float(improvement(ys[-1],baseline_field))
            ax.text(.97,.62 if arm=='tail20' else .54,f'{LABELS[arm]}: {gain:+.2f}% improvement',
                    transform=ax.transAxes,ha='right',fontsize=9,color=COLORS[arm],bbox={'facecolor':'white','alpha':.8,'edgecolor':'none'})
        ax.set_title(label);style(ax,'Checkpoint update','Loss contribution');ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    axes[-1].axis('off');axes[0].legend(fontsize=9)
    axes[-1].text(.03,.92,'Final checkpoint: update 2,000\n\nPositive improvement = lower MSE.\nNegative improvement = higher MSE.\n\nEach variable has its own y-axis.\nCloud fields are input-only.\n\nAll 53 checkpoint evaluations are complete.\nPer-variable training histories were not logged.',va='top',linespacing=1.6)
    fig.suptitle('Supplementary per-variable evaluation: eval K=20 for both models\nTrain K=1 is a cross-horizon test; not matched short-horizon performance',fontsize=14)
    save(fig,out/'per_variable_eval_loss_vs_checkpoint.png')
    fig,ax=plt.subplots(figsize=(10.5,4.8),layout='constrained')
    for arm,values in evaluations.items():
        ax.plot([0]+[time_hours[arm][v['step']-1] for v in values],[0]+[v['improvement_percent'] for v in values],
                marker='o',markersize=3,color=COLORS[arm],label=LABELS[arm])
    style(ax,'Cumulative recorded update time (hours)','Validation MSE improvement (%)');ax.axhline(0,color='#64748b',lw=1)
    ax.set_title('Supplementary eval K=20: improvement vs recorded training time\nIncludes timed updates and batch loading; excludes queues and setup outside updates.')
    ax.legend();save(fig,out/'eval_improvement_vs_training_time.png')
    fig,ax=plt.subplots(figsize=(10.5,4.8),layout='constrained');leadrows=[]
    for arm,values in evaluations.items():
        d=values[-1];hours=np.arange(6,127,6);gains=improvement(d['per_step'][3:],d['baseline_per_step'][3:])
        ax.plot(hours,gains,marker='o',markersize=3,color=COLORS[arm],label=LABELS[arm])
        for h,g in zip(hours,gains):leadrows.append({'arm':arm,'checkpoint':2000,'lead_hours':int(h),'improvement_pct':float(g)})
    ax.axhline(0,color='#64748b',lw=1);style(ax,'Tail forecast lead (hours)','Normalized weighted MSE improvement (%)')
    ax.set_title('Supplementary eval K=20: final improvement by tail lead');ax.legend();save(fig,out/'eval_improvement_by_lead.png')
    trainrows=[];evalrows=[];finalrows=[]
    for arm,rows in training.items():
        for r in rows:trainrows.append({'arm':arm,**{k:r[k] for k in ('step','loss','lr','seconds')},'cumulative_update_hours':float(time_hours[arm][r['step']-1])})
    for arm,values in evaluations.items():
        for d in values:
            for field,label in [('total','Total')]+FIELDS:
                b=d['baseline_mean_loss'] if field=='total' else d['baseline_per_field'][field]
                c=d['mean_loss'] if field=='total' else d['per_field'][field]
                row={'arm':arm,'checkpoint_step':d['step'],'variable':field,'loss':c,'baseline':b,'improvement_pct':float(improvement(c,b)),
                     'cumulative_update_hours':float(time_hours[arm][d['step']-1])}
                evalrows.append(row)
                if d['step']==2000:finalrows.append(row)
    write_csv(out/'train_update_losses.csv',trainrows);write_csv(out/'eval_checkpoint_losses.csv',evalrows)
    write_csv(out/'final_metrics.csv',finalrows);write_csv(out/'final_lead_metrics.csv',leadrows)
    finals={a:v[-1] for a,v in evaluations.items()}
    (out/'plot-data.json').write_text(json.dumps({'runs':metadata,'evaluations':evaluations,'plot_script_sha256':sha(Path(__file__))},indent=2)+'\n')
    lines=['# Supplementary loss v2: common K20 evaluation','',
        '**This is cross-horizon evaluation for the model trained with K=1. It is not the primary training-matched comparison.** See the [primary matched K1 and K20 report](../v2_matched_20261106/README.md). The K1 result of 0.20% here does not describe its short-horizon performance.','',
        '**Both independent runs finished 2,000 updates and every saved checkpoint common-K20 evaluation.** K=20 has 26 complete evaluations; K=1 has 27. Matched K1 evaluation is tracked separately in the primary report.','',
        '## Common validation results','',
        'The fixed 2022 validation set contains 15 segments / 60 chunks. Both arms use the same tail20 native inputs, 50–1000 hPa targets, Gaussian latitude weights, pressure weights and train-only 6 h-difference normalization. Memory resets once per segment and carries across four chunks. Percentages below are reductions in normalized weighted MSE relative to the official pretrained NeuralGCM baseline.','',
        '| Quantity | Baseline | K=1 loss | K=1 improvement | K=20 loss | K=20 improvement |','|---|---:|---:|---:|---:|---:|']
    for field,label in [('total','Total')]+FIELDS:
        r={a:next(x for x in finalrows if x['arm']==a and x['variable']==field) for a in runs}
        lines.append(f"| {label} | {r['tail20']['baseline']:.6f} | {r['tail1']['loss']:.6f} | {r['tail1']['improvement_pct']:+.2f}% | {r['tail20']['loss']:.6f} | {r['tail20']['improvement_pct']:+.2f}% |")
    lines+=['','The final K=20 checkpoint improves all five variables. K=1 improves T, Z and q slightly, while both wind terms worsen slightly. The final K=20 common-validation loss is '+f"{100*(1-finals['tail20']['mean_loss']/finals['tail1']['mean_loss']):.2f}% lower than K=1.",'',
        'These are endpoint results, not validation-selected best checkpoints. Curves are not strictly monotonic. This is a single-seed 2022 monitoring comparison, not a held-out 2023 result. Changing ar_tail_k changes both training horizon and teacher forcing, so this does not isolate the causal contribution of Mamba memory. Per-variable values are weighted normalized losses, not physical-unit RMSE.','',
        '## Evaluation loss vs checkpoint','','![Common validation loss](eval_loss_vs_checkpoint.png)','',
        '## Per-variable evaluation loss','','![Per-variable validation](per_variable_eval_loss_vs_checkpoint.png)','',
        '## Training loss vs checkpoint','','The two training losses have different forecast horizons and should not be ranked directly. Markers show the preceding 100-update mean at saved checkpoints.','','![Training loss](train_loss_vs_checkpoint.png)','',
        '## Improvement vs recorded training time','','Recorded update seconds exclude queueing and setup outside the timed update loop. This is not total wall-clock completion time.','','![Improvement versus time](eval_improvement_vs_training_time.png)','',
        '## Final improvement by tail lead','','![Improvement by lead](eval_improvement_by_lead.png)','',
        '## Files and exact paths','',
        'GitHub branch: `residualmamba_NGCM_rebuild_20261104`.','',
        'Repository directory: `experimental/residualmamba_NGCM/results/v2_comparison_20261106/`.','',
        'Published local directory: `'+str(out.resolve())+'`.','']
    for arm,root in runs.items():lines.append(f"- {LABELS[arm]} original plots: `{root / 'eval/separate_plots'}`")
    lines+=['','[All eval values](eval_checkpoint_losses.csv) · [Final metrics](final_metrics.csv) · [Training values](train_update_losses.csv) · [Lead metrics](final_lead_metrics.csv) · [Source receipts and plot data](plot-data.json).','']
    (out/'README.md').write_text('\n'.join(lines))
    print(json.dumps({'output':str(out),'eval_counts':{a:len(v) for a,v in evaluations.items()},'final_improvement_pct':{a:v[-1]['improvement_percent'] for a,v in evaluations.items()}},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tail20-root',type=Path,required=True);p.add_argument('--tail1-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();publish({'tail20':a.tail20_root.resolve(),'tail1':a.tail1_root.resolve()},a.output)
