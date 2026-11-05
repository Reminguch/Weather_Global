#!/usr/bin/env python3
"""Three separate checkpoint plots, using observed train logs and complete evals."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIELDS=[('temperature','Temperature'),('geopotential','Geopotential'),
        ('u_component_of_wind','Zonal wind (u)'),('v_component_of_wind','Meridional wind (v)'),
        ('specific_humidity','Specific humidity'),('specific_cloud_ice_water_content','Cloud ice'),
        ('specific_cloud_liquid_water_content','Cloud liquid')]
COLORS={'tail20':'#2563eb','tail1':'#d97706'}
LABELS={'tail20':'ar_tail_k = 20','tail1':'ar_tail_k = 1'}


def write_csv(path,rows,fields):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)


def save(fig,path):
    tmp=path.with_name(path.stem+'.tmp.png');fig.savefig(tmp,dpi=180,bbox_inches='tight');tmp.replace(path);plt.close(fig)


def data(root):
    plan=json.loads((root/'plan.json').read_text());runs={a:Path(p) for a,p in plan['arms'].items()}
    train={};points=[];available={}
    for arm,run in runs.items():
        checkpoints=[]
        for p in sorted((run/'checkpoints').glob('step_*.json')):
            if p.with_suffix('.pkl').exists():checkpoints.append(json.loads(p.read_text())['step'])
        available[arm]=checkpoints
        if not checkpoints:continue
        text=(run/'checkpoints/training.jsonl').read_text();lines=text.splitlines()
        if text and not text.endswith('\n'):lines=lines[:-1]
        selected={r['step']:r for line in lines if (r:=json.loads(line))['step']<=max(checkpoints)}
        ordered=[selected[i] for i in sorted(selected)];train[arm]=ordered
        for step in checkpoints:
            window=[selected[i]['loss'] for i in range(max(1,step-99),step+1)]
            points.append({'arm':arm,'checkpoint_step':step,'loss_at_update':selected[step]['loss'],
                'previous_100_update_mean':float(np.mean(window)),'window_start':max(1,step-99),'window_end':step,
                'cumulative_recorded_update_seconds':sum(r['seconds'] for r in ordered if r['step']<=step)})
    evaluations=[json.loads(p.read_text()) for p in sorted(root.glob('tail*/step_*/summary.json'))]
    return train,points,available,evaluations


def draw(root,out):
    out.mkdir(parents=True,exist_ok=True);train,points,available,evals=data(root)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
    for arm,rows in train.items():
        steps=np.array([r['step'] for r in rows]);loss=np.array([r['loss'] for r in rows]);color=COLORS[arm]
        ax.plot(steps,loss,color=color,alpha=.18,lw=.8,label=LABELS[arm]+' raw training loss')
        if len(loss)>=100:ax.plot(steps[99:],np.convolve(loss,np.ones(100)/100,'valid'),color=color,lw=1.8,label=LABELS[arm]+' 100-update mean')
        p=[r for r in points if r['arm']==arm]
        ax.scatter([r['checkpoint_step'] for r in p],[r['previous_100_update_mean'] for r in p],color=color,edgecolor='white',s=65,zorder=5,label=LABELS[arm]+' saved checkpoint')
        for r in p:ax.annotate(str(r['checkpoint_step']),(r['checkpoint_step'],r['previous_100_update_mean']),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
    ax.set(title='Training loss vs checkpoint',xlabel='Optimizer update / checkpoint step',ylabel='Weighted normalized training MSE')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Markers show the preceding 100-update mean; weather samples change across updates.',fontsize=10)
    save(fig,out/'train_loss_vs_checkpoint.png')
    if not evals:raise ValueError('No completed checkpoint evaluation yet')
    baseline=evals[0]['baseline_mean_loss'];last=max(max(v,default=0) for v in available.values())
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
    ax.axhline(baseline,color='#64748b',ls='--',lw=1.2,label='Official NeuralGCM zero-head reference')
    for arm in ('tail20','tail1'):
        values=sorted([r for r in evals if r['arm']==arm],key=lambda r:r['step'])
        if values:ax.plot([0]+[r['step'] for r in values],[baseline]+[r['mean_loss'] for r in values],marker='o',color=COLORS[arm],lw=1.8,label=LABELS[arm])
        for r in values:ax.annotate(f"{r['step']}: {r['mean_loss']:.2f}",(r['step'],r['mean_loss']),xytext=(6,10),textcoords='offset points',fontsize=9)
    ax.set_xlim(0,max(last,1)*1.08);ax.set(title='Evaluation loss vs checkpoint',xlabel='Checkpoint step',ylabel='Weighted normalized validation MSE')
    ax.grid(alpha=.2);ax.legend(fontsize=9)
    pending={arm:[s for s in steps if not any(e['arm']==arm and e['step']==s for e in evals)] for arm,steps in available.items()}
    fig.suptitle('Fixed 2022 validation: 15 segments / 60 chunks. Only completed evaluations are plotted.',fontsize=10)
    save(fig,out/'eval_loss_vs_checkpoint.png')
    fig,axes=plt.subplots(4,2,figsize=(12,13),layout='constrained');axes=axes.ravel()
    for ax,(field,label) in zip(axes,FIELDS):
        reference=evals[0]['baseline_per_field'][field]
        ax.axhline(reference,color='#64748b',ls='--',lw=1.1)
        for arm in ('tail20','tail1'):
            values=sorted([r for r in evals if r['arm']==arm],key=lambda r:r['step'])
            if not values:continue
            ax.plot([0]+[r['step'] for r in values],[reference]+[r['per_field'][field] for r in values],marker='o',color=COLORS[arm],lw=1.8,label=LABELS[arm])
            latest=values[-1];change=100*(latest['per_field'][field]/reference-1)
            ax.text(.98,.96 if arm=='tail20' else .86,f"{LABELS[arm]}, step {latest['step']}: {change:+.2f}%",transform=ax.transAxes,ha='right',va='top',fontsize=9,color=COLORS[arm],bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':2})
        ax.set(title=label,xlabel='Checkpoint step',ylabel='Weighted validation loss contribution',xlim=(0,max(last,1)*1.08))
        ax.grid(alpha=.2);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    axes[-1].axis('off')
    axes[-1].text(.03,.95,'Reading these panels\n\nNegative change = lower loss.\nPositive change = higher loss.\n\nEach variable has its own y-axis scale.\nAnnotations report changes relative to the zero head.\n\nThese are normalized, weighted eval losses.\nPer-variable training loss was not logged.\n\nPending eval checkpoints:\n'+ '\n'.join(f'{LABELS[a]}: {v}' for a,v in pending.items()),va='top',fontsize=11)
    axes[0].legend(fontsize=8,loc='lower left')
    fig.suptitle('Per-variable evaluation loss vs checkpoint\nSame validation samples, normalization and loss weights for every checkpoint',fontsize=14)
    save(fig,out/'per_variable_eval_loss_vs_checkpoint.png')
    write_csv(out/'train_checkpoint_losses.csv',points,['arm','checkpoint_step','loss_at_update','previous_100_update_mean','window_start','window_end','cumulative_recorded_update_seconds'])
    raw=[{'arm':arm,**{key:r[key] for key in ('step','loss','lr','seconds')}} for arm,rows in train.items() for r in rows]
    write_csv(out/'train_update_losses.csv',raw,['arm','step','loss','lr','seconds'])
    evrows=[]
    for e in evals:
        evrows.append({'arm':e['arm'],'checkpoint_step':e['step'],'variable':'total','loss':e['mean_loss'],'baseline':e['baseline_mean_loss'],'change_percent':100*(e['mean_loss']/e['baseline_mean_loss']-1)})
        for field,_ in FIELDS:
            x,b=e['per_field'][field],e['baseline_per_field'][field]
            evrows.append({'arm':e['arm'],'checkpoint_step':e['step'],'variable':field,'loss':x,'baseline':b,'change_percent':100*(x/b-1)})
    write_csv(out/'eval_checkpoint_losses.csv',evrows,['arm','checkpoint_step','variable','loss','baseline','change_percent'])
    (out/'plot-data.json').write_text(json.dumps({'created_unix':time.time(),'checkpoints':available,'pending_evaluations':pending,
        'training_checkpoint_points':points,'evaluations':evals,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    (out/'README.md').write_text('# Loss versus checkpoint\n\n'
        'Training traces extend through the latest saved checkpoint. Checkpoint markers show the mean of the preceding 100 training updates; the CSV also contains the single-update loss. Training samples vary with update, so that curve is not a fixed validation comparison.\n\n'
        'Evaluation uses the same complete 2022 validation set (15 segments, 60 chunks) and common tail20 inputs. Only completed evaluations appear. Step zero is the official pretrained NeuralGCM zero-head reference. Lines connect observations and do not establish unmeasured intermediate values or monotonic decay.\n\n'
        'Per-variable curves are eval loss contributions, not raw physical RMSE. Per-variable training history is unavailable because it was not logged. Negative percentages in the panels mean lower loss. The side panel lists checkpoints awaiting evaluation.\n\n'
        '## Training loss\n\n![Training loss](train_loss_vs_checkpoint.png)\n\n'
        '## Evaluation loss\n\n![Evaluation loss](eval_loss_vs_checkpoint.png)\n\n'
        '## Per-variable evaluation loss\n\n![Per-variable evaluation loss](per_variable_eval_loss_vs_checkpoint.png)\n\n'
        'Source values are in [train_checkpoint_losses.csv](train_checkpoint_losses.csv), [train_update_losses.csv](train_update_losses.csv), [eval_checkpoint_losses.csv](eval_checkpoint_losses.csv) and [plot-data.json](plot-data.json).\n')
    print(json.dumps({'rendered':str(out),'checkpoints':available,'complete_evaluations':[(e['arm'],e['step']) for e in evals]}),flush=True)


def signature(root):
    plan=json.loads((root/'plan.json').read_text());paths=list(root.glob('tail*/step_*/summary.json'))
    for run in plan['arms'].values():paths.extend((Path(run)/'checkpoints').glob('step_*.json'))
    return tuple((str(p),p.stat().st_mtime_ns) for p in sorted(paths))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--watch',action='store_true');a=p.parse_args()
    if a.watch:
        import fcntl
        a.output.mkdir(parents=True,exist_ok=True)
        lock=(a.output/'watch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=None
    while True:
        current=signature(a.root)
        if current!=previous:draw(a.root,a.output);previous=current
        if not a.watch:return
        if json.loads((a.root/'controller/status.json').read_text())['phase']=='complete':return
        time.sleep(30)

if __name__=='__main__':main()
