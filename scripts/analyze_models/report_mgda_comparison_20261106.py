"""Compare saved MGDA evaluations, preserving cohorts and fixed denominators.
Run through scripts/graphcast_env.sh; no forecasts or training are launched.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analyze_models.amse_feature_report_data import FEATURE_SPECS
FEATURES = [x[0] for x in FEATURE_SPECS if x[2] is not None]
BASE = ROOT/'artifacts/checkpoints/v24_Ilya'
OUT = ROOT/'plots/analyze_models/data/resolution_eval/mgda_comparison_20261106'
IMG = ROOT/'plots/analyze_models/images/resolution_eval/mgda_comparison_20261106'
SOURCES = {}
WARNINGS = []
def read(path):
    path = Path(path)
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())
def evaluation(path):
    d = read(path)
    assert d['evaluation_status'] == 'complete', path
    l = d['original_graphcast_loss']
    for branch in ('full','baseline'):
        v = np.asarray(l[branch+'_per_step'])
        assert v.shape == (40,) and np.isfinite(v).all()
        np.testing.assert_allclose(v.mean(),l[branch+'_rollout'],rtol=1e-10)
    return d

def spectra(directory, result):
    records=[]; dates=[]; variables=None
    for p in sorted(directory.glob('*.npz')):
        SOURCES[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        with np.load(p) as z:
            meta=json.loads(str(z['metadata']))
            assert meta['checkpoint_sha256']==result['checkpoint_sha256']
            assert meta['baseline_checkpoint_sha256']==result['baseline_checkpoint_sha256']
            dates.append(meta['initialization_time'])
            vv=z['variables'].tolist()
            if variables is None: variables=vv
            assert variables==vv
            np.testing.assert_array_equal(z['lead_hours'],np.arange(1,41)*6)
            records.append({k:z[k].copy() for k in ('truth_power','baseline_power','full_power')})
    assert len(records)==8 and len(set(dates))==8
    pooled={k:np.mean([r[k] for r in records],axis=0) for k in records[0]}
    ix=[variables.index(f) for f in FEATURES]
    truth=pooled['truth_power'][:,ix,1:]
    assert np.all(truth>0)
    power={branch:np.mean(100*np.abs(pooled[branch+'_power'][:,ix,1:]-truth)/truth,axis=(1,2)) for branch in ('baseline','full')}
    assert all(np.isfinite(v).all() for v in power.values())
    return dict(power=power,dates=dates,truth=pooled['truth_power'],variables=variables)

campaigns=[('power','res1_mgda_gc_power_2k_recovery_20261007'),('split','res1_amse_split_mgda_di16_di32_recovery_20261013'),('stopped','res1_mse_amse_excess_advisory_20261028'),('full','res1_mse_amse_excess_full_bptt24_20261028')]
items=[]
for family,name in campaigns:
    directory=BASE/name/'evaluation'; manifest=read(directory/'manifest.json')
    for t in manifest['task_specs']:
        p=directory/'results'/t['id']
        ex=evaluation(p/'exact/evaluation.json'); sp=evaluation(p/'spectral/evaluation.json')
        assert ex['checkpoint_sha256']==sp['checkpoint_sha256']
        s=spectra(p/'spectral',sp)
        for stage,result in [('exact',ex),('spectral',sp)]:
            proto=manifest['protocols'][stage]
            assert result['n_samples']==proto['n_samples']
            assert result['baseline_checkpoint_sha256']==proto['baseline']['checkpoint_sha256']
        assert sorted(s['dates'])==sorted(manifest['protocols']['spectral']['initialization_times'])
        items.append(dict(family=family,id=t['id'],label=family+' / '+t['id'],exact=ex,spectral=sp,s=s,checkpoint=t['checkpoint']))
        print('Loaded',family,t['id'],flush=True)
pilot=ROOT/'plots/analyze_models/data/resolution_eval/res1_spectral_pilot_2023_20260917'
m1sp=evaluation(pilot/'models/M1/evaluation.json')
m1ex=evaluation(BASE/'res1_batch4_temporal_lr_20260908/runs/di16_fast_mamba/eval/cold_full_zero_exact_gc/matched32/swa_step00500-02000.json')
assert m1sp['checkpoint_sha256']==m1ex.get('checkpoint_sha256',m1sp['checkpoint_sha256'])
m1=dict(family='reference',id='M1',label='Main-text GC-Mamba (M1)',exact=m1ex,spectral=m1sp,s=spectra(pilot/'models/M1',m1sp),checkpoint=m1sp['ckpt'])
items.insert(0,m1)
newref=next(x for x in items if x['family']=='full' and x['id']=='parent')
# Old exact evaluations share the main-text model's matched32 chosen indices.
for x in items:
    ref=m1 if x['family'] in ('reference','power','split') else newref
    for key in ('chosen_idx','target_steps','eval_mode','residual_state_init'):
        assert x['exact'][key]==ref['exact'][key], (x['label'],key)
    assert x['s']['dates']==ref['s']['dates']
    drift=np.abs(x['s']['truth']-ref['s']['truth'])
    if drift.max()>0:
        WARNINGS.append(dict(model=x['label'],truth_power_max_abs_drift=float(drift.max()),truth_power_relative_l2=float(np.linalg.norm(drift)/np.linalg.norm(ref['s']['truth'])),policy='finite discrepancy retained as warning'))
    x['reference']=ref
    for stage in ('exact','spectral'):
        own=np.asarray(x[stage]['original_graphcast_loss']['baseline_per_step'])
        canonical=np.asarray(ref[stage]['original_graphcast_loss']['baseline_per_step'])
        WARNINGS.append(dict(model=x['label'],stage=stage,
            baseline_rollout_drift_pct=float(100*(own.mean()/canonical.mean()-1)),
            baseline_max_abs_drift=float(np.max(np.abs(own-canonical))),
            policy='finite baseline drift retained; common denominator used'))
rows=[];curves=[]
for x in items:
    r=x['reference']; old=r is m1
    loss=x['exact']['original_graphcast_loss']; sloss=x['spectral']['original_graphcast_loss']
    base=r['exact']['original_graphcast_loss']; sb=r['spectral']['original_graphcast_loss']
    p=x['s']['power']['full']; bp=r['s']['power']['baseline']; rp=r['s']['power']['full']
    selected=np.array([3,11,27,39])
    row=dict(family=x['family'],id=x['id'],mse_cohort='2022_32' if old else '2022_8',power_cohort='2023_8' if old else '2022_8',exact_mse=loss['full_rollout'],gc_mse=base['baseline_rollout'],mse_improvement_vs_gc_pct=100*(1-loss['full_rollout']/base['baseline_rollout']),mse_improvement_vs_reference_pct=100*(1-loss['full_rollout']/base['full_rollout']),reference='main_text_M1' if old else 'di16_joint_pilot_parent',power_error_pct=float(p[selected].mean()),gc_power_error_pct=float(bp[selected].mean()),power_error_reduction_vs_gc_pct=float(100*(1-p[selected].mean()/bp[selected].mean())),power_error_reduction_vs_reference_pct=float(100*(1-p[selected].mean()/rp[selected].mean())),spectral_cohort_mse=sloss['full_rollout'],spectral_cohort_mse_improvement_vs_gc_pct=100*(1-sloss['full_rollout']/sb['baseline_rollout']),checkpoint=x['checkpoint'])
    rows.append(row);x['row']=row
    for j in range(40):
        curves.append(dict(family=x['family'],id=x['id'],lead_day=(j+1)/4,exact_mse=loss['full_per_step'][j],exact_mse_improvement_pct=100*(1-loss['full_per_step'][j]/base['baseline_per_step'][j]),spectral_cohort_mse_improvement_pct=100*(1-sloss['full_per_step'][j]/sb['baseline_per_step'][j]),power_error_pct=float(p[j]),power_improvement_pct=float(100*(1-p[j]/bp[j]))))
    x['curves']=curves[-40:]
for name,data in [('summary.csv',rows),('rollout.csv',curves)]:
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
(OUT/'provenance.json').write_text(json.dumps(dict(sources=SOURCES,warnings=WARNINGS,features=FEATURES,power_selected_days=[1,3,7,10],power_definition='Pool powers over starts, then mean absolute relative power error over positive zonal wavenumbers and nine fields; headline mean over days 1,3,7,10. Latitude band 30–60 degrees in both hemispheres.',mse_definition='Exact original GraphCast weighted normalized MSE; mean all 40 leads before ratios. Canonical GC denominator per cohort.',main_text_reference='di16_fast_mamba, not di16_joint',limits='No matched M1 evaluation on eight 2022 pilot origins was found. Do not rank different cohorts as a controlled comparison. Finite baseline and truth-spectrum drift are warnings.'),indent=2)+'\n')
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2})
colors={'M1':'black','power':'#0072B2','split':'#D55E00','stopped':'#009E73','full':'#CC79A7'}
def label(x):
    f,i=x['family'],x['id']
    if f=='reference':return 'Main-text GC-Mamba (M1)'
    if i=='parent':return 'Pilot parent: GC-Mamba di16 joint'
    if i=='mature_mse':return 'Oct 7 mature MSE parent'
    return {'power':'Relative power','split':'Amplitude/pattern','stopped':'Stopped feedback','full':'Full BPTT'}[f]+' / '+i.replace('_step',' @ ').replace('00200','200').replace('02000','2000').replace('00500','500').replace('_',' ')
selected=[x for x in items if (x['family']=='reference' or (x['family']=='power' and x['id'] in ('mgda_swa','mgda_step02000','mse_swa','mature_mse')) or (x['family']=='split' and x['id'].endswith('02000')) or (x['family'] in ('stopped','full') and (x['id']!='parent' or x['family']=='full')))]
fig,axs=plt.subplots(2,2,figsize=(15,11))
for x in selected:
    old=x['reference'] is m1; rr=0 if old else 1
    ls='--' if 'mse' in x['id'] or 'di32' in x['id'] else '-'
    if 'fixed' in x['id']:ls=':'
    if x['id']=='parent':color='black'
    else:color=colors.get(x['family'],'black')
    if x['id']=='mature_mse':color='gray'
    if x['id']=='mgda_step02000':ls=':'
    for col,key in [(0,'spectral_cohort_mse_improvement_pct'),(1,'power_improvement_pct')]:
        if not old and col==0:key='exact_mse_improvement_pct'
        axs[rr,col].plot([c['lead_day'] for c in x['curves']],[c[key] for c in x['curves']],label=label(x),color=color,linestyle=ls,lw=2 if color=='black' else 1.5)
for rr,cohort in enumerate(['Eight 2023 origins: historical MGDA + main-text M1','Eight 2022 origins: 200-update pilots']):
    for cc,metric in enumerate(['Exact GC MSE reduction','Power-spectrum error reduction']):
        ax=axs[rr,cc];ax.axhline(0,color='gray',lw=.8);ax.set(xlabel='Forecast lead (days)',ylabel='Improvement vs original GraphCast (%)',title=cohort+'\n'+metric,xlim=(.25,10));ax.legend(fontsize=7,loc='best')
fig.suptitle('MGDA rollout comparison — positive values mean lower error\nRows use different evaluation cohorts; curves within each row share a GC denominator',fontsize=13)
fig.tight_layout(rect=(0,0,1,.94))
for ext in ('png','pdf'):fig.savefig(IMG/f'rollout_comparison.{ext}',dpi=180)
plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(15,6))
for x in items:
    if x['reference'] is not m1:continue
    if x['family']=='reference' or (x['family']=='power' and x['id'] in ('mature_mse','mgda_swa','mgda_step02000','mse_swa')) or (x['family']=='split' and x['id'].endswith('02000')):
        axs[0].plot(np.arange(1,41)/4,[c['exact_mse_improvement_pct'] for c in x['curves']],label=label(x))
for family in ('power','split'):
    for arm in (('mgda','mse') if family=='power' else ('di16','di32')):
        xx=[x for x in items if x['family']==family and x['id'].startswith(arm+'_step')]
        axs[1].plot([int(x['id'].split('step')[1]) for x in xx],[x['row']['mse_improvement_vs_gc_pct'] for x in xx],marker='o',label=family+' / '+arm)
axs[0].set(title='Matched 32-start 2022 MSE rollout',xlabel='Forecast lead (days)',ylabel='Exact GC MSE reduction vs original GC (%)')
axs[1].axhline(m1['row']['mse_improvement_vs_gc_pct'],color='black',ls='--',label='Main-text M1')
axs[1].set(title='All saved raw-checkpoint endpoints (2022 × 32)',xlabel='Additional optimizer updates',ylabel='Rollout exact GC MSE reduction (%)')
for ax in axs:ax.axhline(0,color='gray',lw=.8);ax.legend(fontsize=8)
fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(IMG/f'exact32_and_checkpoints.{ext}',dpi=180)
plt.close(fig)
print(json.dumps(rows,indent=2))
