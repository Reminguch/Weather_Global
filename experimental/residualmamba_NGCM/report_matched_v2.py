#!/usr/bin/env python3
"""Publish separate training-matched validation curves, each with its own baseline."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO=Path(__file__).resolve().parents[2]
FIELDS=[('temperature','Temperature'),('geopotential','Geopotential'),('u_component_of_wind','Zonal wind'),('v_component_of_wind','Meridional wind'),('specific_humidity','Specific humidity')]


def read(path):return json.loads(path.read_text())
def save(fig,path):fig.savefig(path,dpi=160);plt.close(fig)
def gain(value,baseline):return 100*(1-value/baseline)


def collect(matched_root):
    roots={1:REPO/'artifacts/v2_tail1_2k_20261105',20:REPO/'artifacts/v2_2k_20261105'}
    rows={};sources={}
    for k,root in roots.items():
        plan=read(root/'plan.json')
        assert read(root/'checkpoints/run.json')['config']['ar_tail_k']==k
        directory=matched_root/'tail1' if k==1 else root/'eval/tail20'
        entries=[]
        for path in sorted(directory.glob('step_*/summary.json')):
            d=read(path);step=d['step'];receipt=read(root/'checkpoints'/f'step_{step:05d}.json')
            assert d['checkpoint_sha256']==receipt['sha256'] and receipt['identity']==plan['identity']
            assert d['segments']==15 and d['chunks']==60
            if k==1:
                p=d['protocol']
                assert p['training_ar_tail_k']==p['evaluation_ar_tail_k']==1
                assert p['run_identity']==plan['identity']
                assert p['stats_sha256']==plan['prerequisites']['stats.npz']
            else:
                assert d['identity']==plan['identity'] and d['arm']=='tail20'
            np.testing.assert_allclose(sum(d['per_field'].values()),d['mean_loss'],rtol=1e-7)
            entries.append(d)
        rows[k]=sorted(entries,key=lambda x:x['step'])
        for d in entries:
            np.testing.assert_allclose(d['baseline_mean_loss'],entries[0]['baseline_mean_loss'],rtol=1e-9)
        sources[k]={'run':str(root),'source_commit':plan['repo_commit'],'identity':plan['identity'],
                    'evaluation_directory':str(directory),'training_ar_tail_k':k,'evaluation_ar_tail_k':k,
                    'source':'new matched evaluation' if k==1 else 'existing K20 evaluation already matches training'}
    return rows,sources


def draw(rows,out):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,k in zip(axes,(1,20)):
        data=rows[k];ax.set_title(f'Train K={k}, eval K={k}')
        ax.set_xlabel('Checkpoint update');ax.set_ylabel('Weighted normalized validation MSE');ax.grid(alpha=.2)
        if not data:
            ax.text(.5,.5,'Matched K=1 evaluation in progress\nNo short-horizon result claimed yet',ha='center',transform=ax.transAxes);continue
        baseline=data[0]['baseline_mean_loss']
        ax.axhline(baseline,ls='--',color='gray',label=f'Matching baseline: {baseline:.5f}')
        ax.plot([0]+[d['step'] for d in data],[baseline]+[d['mean_loss'] for d in data],'-o',ms=3,label=f'Residual trained K={k}')
        ax.legend(fontsize=9);ax.margins(y=.2)
    fig.suptitle('Primary evaluation: each model evaluated with its own training protocol\nIndependent y-axes and baselines; absolute losses across panels are not a ranking.')
    save(fig,out/'matched_eval_loss_vs_checkpoint.png')
    fig,axes=plt.subplots(5,2,figsize=(12,16),layout='constrained')
    for j,k in enumerate((1,20)):
        for i,(field,label) in enumerate(FIELDS):
            ax=axes[i,j];data=rows[k];ax.set_title(f'{label} — train K={k}, eval K={k}')
            ax.set_xlabel('Checkpoint update');ax.set_ylabel('Weighted normalized loss');ax.grid(alpha=.2)
            ax.ticklabel_format(axis='y',style='plain',useOffset=False)
            if not data:
                ax.text(.5,.5,'Matched evaluation in progress',ha='center',transform=ax.transAxes);continue
            b=data[0]['baseline_per_field'][field]
            ax.axhline(b,ls='--',color='gray',label='Matching official baseline')
            ax.plot([0]+[d['step'] for d in data],[b]+[d['per_field'][field] for d in data],'-o',ms=3,label='Residual')
            if i==0:ax.legend(fontsize=8)
    fig.suptitle('Per-variable validation: separate matched tasks, separate baselines')
    save(fig,out/'matched_per_variable_loss_vs_checkpoint.png')


def report(matched_root,out):
    out.mkdir(parents=True,exist_ok=True);rows,sources=collect(matched_root);draw(rows,out)
    flat=[]
    for k,data in rows.items():
        for d in data:
            for field,_ in [('total','Total')]+FIELDS:
                value=d['mean_loss'] if field=='total' else d['per_field'][field]
                base=d['baseline_mean_loss'] if field=='total' else d['baseline_per_field'][field]
                flat.append({'training_k':k,'evaluation_k':k,'checkpoint':d['step'],'variable':field,'loss':value,'baseline':base,'improvement_percent':gain(value,base)})
        for d in data:
            if d['step']==2000:
                (out/f'k{k}_matched_step_02000.json').write_text(json.dumps(d,indent=2)+'\n')
    with (out/'matched_eval_losses.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator='\n');w.writeheader();w.writerows(flat)
    (out/'plot-data.json').write_text(json.dumps({'sources':sources,'evaluations':rows},indent=2)+'\n')
    complete=len(rows[1])==27 and len(rows[20])==26
    lines=['# Primary loss v2 results: training-matched evaluation','',
        '**Status: '+('all 53 checkpoint evaluations complete.' if complete else f'K=1: {len(rows[1])}/27 matched evaluations complete; K=20: {len(rows[20])}/26 complete.')+'**','',
        'The primary analysis evaluates each model using the same input/origin/forecast-horizon protocol as its own training run. K=1 has its own official pretrained NeuralGCM baseline. K=20 has its own baseline. Absolute losses across these different tasks must not be ranked.','',
        'The previous [common-tail20 comparison](../v2_comparison_20261106/README.md) is supplementary cross-horizon evaluation. Its K=1 improvement of 0.20% describes train K=1 / eval K=20, not matched K=1 performance.','',
        '| Training K | Evaluation K | Update-2000 loss | Matching baseline | Improvement |','|---|---|---:|---:|---:|']
    for k in (1,20):
        final=next((d for d in rows[k] if d['step']==2000),None)
        lines.append(f"| {k} | {k} | {final['mean_loss']:.6f} | {final['baseline_mean_loss']:.6f} | {final['improvement_percent']:+.2f}% |" if final else f'| {k} | {k} | Pending | Pending | Pending |')
    lines+=['','## Final per-variable results','']
    for k in (1,20):
        d=next((v for v in rows[k] if v['step']==2000),None)
        lines += [f'### Train K={k}, eval K={k}','']
        if d is None:
            lines += ['Final matched result pending.',''];continue
        lines += ['| Variable | Residual loss | Matching baseline | Improvement |','|---|---:|---:|---:|']
        for field,label in FIELDS:
            value=d['per_field'][field];base=d['baseline_per_field'][field]
            lines.append(f'| {label} | {value:.6f} | {base:.6f} | {gain(value,base):+.2f}% |')
        lines += ['']
    lines+=['','## Matched evaluation curves','','Only completed checkpoints are shown. Lines connect observations; missing intermediate checkpoints do not imply a measured trend. Step zero is the zero-head official baseline.','','![Separate matched evaluations](matched_eval_loss_vs_checkpoint.png)','',
        '![Per-variable matched evaluations](matched_per_variable_loss_vs_checkpoint.png)','',
        '## Exact protocols','',
        '- Both use all 15 fixed 2022 validation segments / 60 chunks, train-only normalization, the five scored fields on 50–1000 hPa, Gaussian latitude weights and unchanged pressure weights. Memory resets at each segment and carries across four chunks.','- The reported total averages all 24 positions, exactly as the training objective does.','- K=1: 22 independent 6 h prefix predictions, then a tail at 6 h and 12 h. There are 23 six-hour targets and one twelve-hour target per chunk.','- K=20: 3 independent 6 h prefix predictions, then a tail at 6–126 h.','- For every prediction, the residual and its official pretrained NeuralGCM baseline use identical origins, target valid times and leads.','- K=20 results are reused because the existing K20 loader and evaluator already match its training protocol. K=1 is recomputed with the authenticated K1 cache and the original training loader.','- The new evaluator imports qualified code from each immutable training snapshot. It does not retrain either model or feed corrected weather fields back into NeuralGCM.','- Each new K1 job checks a real H200 forward pass and verifies baseline contributions against direct per-step loss computation before the full evaluation.','- These are single-seed monitoring results on 2022, not held-out 2023 skill or an isolated memory ablation.','',
        '## Training losses','','Training objectives have different horizons. The [training curves](../v2_comparison_20261106/train_loss_vs_checkpoint.png) remain in separate panels.','',
        '## Reproduction and paths','',f'Generated report: `{out.resolve()}`.',f'Raw K1 matched evaluation and submission receipts: `{matched_root.resolve()}`.','',
        'Worker: [`matched_v2_eval.py`](../../matched_v2_eval.py). Controller: [`watch_matched_v2_eval.py`](../../watch_matched_v2_eval.py). Plot/report generator: [`report_matched_v2.py`](../../report_matched_v2.py).','',
        '[All plotted values](matched_eval_losses.csv) · [Provenance and plot data](plot-data.json).','']
    (out/'README.md').write_text('\n'.join(lines))
    print(json.dumps({'matched_evals':{k:len(v) for k,v in rows.items()},'output':str(out)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matched-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report(a.matched_root.resolve(),a.output.resolve())
