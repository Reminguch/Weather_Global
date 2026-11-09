"""Frozen 122-start MGDA spectral evaluation and descriptive reporting."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(root):
 meta=read(root/'campaign.json')
 for rel,digest in read(root/'source_hashes.json').items():
  if sha(root/'source'/rel)!=digest:raise ValueError('Frozen source changed: '+rel)
 for record in meta['inputs']:
  if sha(record['path'])!=record['sha256']:raise ValueError('Input changed: '+record['path'])
 return meta
def ratio_mismatch(model,truth,baseline):
 denominator=np.abs(baseline-truth)
 return np.divide(np.abs(model-truth),denominator,out=np.full_like(model,np.nan),where=denominator>0)
def report(root):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.backends.backend_pdf import PdfPages
 from scripts.analyze_models import evaluate_v24_amse_regime as ev
 meta=verify(root); allmodels={};sources=[]
 for mi,model in enumerate(meta['models']):
  records={};losses={};variables=None
  for shard in range(meta['shards']):
   directory,spec,result,_=ev.resolve_result(root/f'shards/{shard:02d}/manifest.json',mi,'spectral')
   sources.extend([dict(path=str(directory/f),sha256=sha(directory/f)) for f in ('evaluation.json','specification.json')])
   for row in result['original_graphcast_loss_per_initialization']:
    stamp=row['initialization_time']
    if stamp in losses:raise ValueError('Duplicate loss origin')
    losses[stamp]=row
   for p in sorted(directory.glob('*.npz')):
    with np.load(p) as z:
     header=json.loads(str(z['metadata']));stamp=header['initialization_time']
     assert header['checkpoint_sha256']==spec['checkpoint_sha256']
     assert header['baseline_checkpoint_sha256']==spec['baseline']['checkpoint_sha256']
     assert stamp in spec['initialization_times'] and stamp not in records
     if variables is None:variables=z['variables'].tolist()
     assert variables==z['variables'].tolist()
     np.testing.assert_array_equal(z['lead_hours'],np.arange(1,41)*6)
     records[stamp]={key:z[key].copy() for key in ('full_power','truth_power','baseline_power','full_error_power','baseline_error_power')}
     assert all(v.shape==(40,len(variables),181) and np.isfinite(v).all() and (v>=0).all() for v in records[stamp].values())
    sources.append(dict(path=str(p),sha256=sha(p)))
  assert set(records)==set(meta['dates'])==set(losses) and len(records)==122
  pooled={key:np.mean([records[t][key] for t in meta['dates']],axis=0) for key in next(iter(records.values()))}
  l={b:np.mean([losses[t][b+'_per_step'] for t in meta['dates']],axis=0) for b in ('full','baseline')}
  allmodels[model['id']]=dict(power=pooled,loss=l,variables=variables)
 out=root/'report';out.mkdir(exist_ok=True)
 ref=allmodels['M1'];fields=['T2m','U10','V10','MSLP','TP6','T850','Q700','U850','V850'];ix=[ref['variables'].index(v) for v in fields]
 truth=ref['power']['truth_power'][:,ix,1:];gc=ref['power']['baseline_power'][:,ix,1:];gc_loss=ref['loss']['baseline']
 gc_mismatch=np.mean(np.abs(gc-truth)/truth,axis=(1,2));assert (truth>0).all()
 summary=[];warnings=[];curves=[];ratios={}
 for model in meta['models']:
  identity=model['id'];item=allmodels[identity];p=item['power']['full_power'][:,ix,1:];loss=item['loss']['full']
  score=np.mean(np.abs(p-truth)/truth,axis=(1,2));ratios[identity]=ratio_mismatch(p,truth,gc)
  summary.append(dict(model=identity,exact_gc_loss=float(loss.mean()),mse_reduction_rollout_pct=float(100*(1-loss.mean()/gc_loss.mean())),mse_reduction_day10_pct=float(100*(1-loss[-1]/gc_loss[-1])),mean_relative_power_mismatch=float(score.mean()),power_mismatch_reduction_rollout_pct=float(100*(1-score.mean()/gc_mismatch.mean())),power_mismatch_reduction_day10_pct=float(100*(1-score[-1]/gc_mismatch[-1]))))
  warnings.append(dict(model=identity,baseline_loss_drift_pct=float(100*(item['loss']['baseline'].mean()/gc_loss.mean()-1)),truth_power_max_abs_drift=float(np.max(np.abs(item['power']['truth_power']-ref['power']['truth_power']))),policy='Finite drift is advisory'))
  for j in range(40):curves.append(dict(model=identity,lead_days=(j+1)/4,mse_reduction_pct=float(100*(1-loss[j]/gc_loss[j])),power_mismatch_reduction_pct=float(100*(1-score[j]/gc_mismatch[j]))))
 for filename,rows in [('summary.csv',summary),('rollout.csv',curves)]:
  with (out/filename).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 np.savez_compressed(out/'pooled_spectra.npz',fields=fields,wavenumber=np.arange(1,181),truth=truth,vanilla_gc=gc,**{m['id']:allmodels[m['id']]['power']['full_power'][:,ix,1:] for m in meta['models']})
 colors=['#0072B2','#009E73','#D55E00']
 with PdfPages(out/'power_mismatch_vs_gc_all_fields.pdf') as pdf:
  for vi,field in enumerate(fields):
   fig,axs=plt.subplots(1,4,figsize=(15,4.8),sharey=True)
   for ax,day in zip(axs,[1,3,7,10]):
    ax.axhline(1,color='gray',ls='--',label='Vanilla GC')
    for i,m in enumerate(meta['models']):ax.plot(np.arange(1,181),ratios[m['id']][day*4-1,vi],label=m['label'],color=colors[i])
    ax.set_xscale('log');ax.set_yscale('symlog',linthresh=1);ax.set(title=f'Day {day}',xlabel='Zonal wavenumber',xlim=(1,180));ax.grid(alpha=.2)
    ax.axhspan(0,1,color='#009E73',alpha=.06)
    ax.text(.02,.98,'↑ Worse (>1)',transform=ax.transAxes,va='top',fontsize=9)
    ax.text(.02,.02,'↓ Better (<1)',transform=ax.transAxes,va='bottom',fontsize=9)
   ymax=max(np.nanmax(r[[3,11,27,39],vi]) for r in ratios.values())
   axs[0].set_ylim(0,max(1.4,float(ymax)*1.4));axs[0].set_ylabel('|Pmodel − Ptruth| / |PGC − Ptruth|')
   fig.suptitle(f'{field}: full 2023 evaluation — 122 starts, GC = 1')
   fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=4,fontsize=9)
   fig.tight_layout(rect=(0,.15,1,.92));pdf.savefig(fig);fig.savefig(out/f'power_mismatch_{field}.png',dpi=160);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(12,4.8))
 for mi,m in enumerate(meta['models']):
  rr=[r for r in curves if r['model']==m['id']]
  for ax,key in zip(axs,['mse_reduction_pct','power_mismatch_reduction_pct']):ax.plot([r['lead_days'] for r in rr],[r[key] for r in rr],label=m['label'],color=colors[mi])
 for ax,title in zip(axs,['Exact GC MSE reduction','Relative-power mismatch reduction']):ax.axhline(0,color='gray');ax.set(title=title,xlabel='Lead days',ylabel='Reduction vs vanilla GC (%)');ax.legend(fontsize=8);ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(out/'rollout.png',dpi=180);fig.savefig(out/'rollout.pdf');plt.close(fig)
 write(out/'report.json',dict(status='complete',starts=122,leads=40,summary=summary,warnings=warnings,sources=sources,metric='abs(mean(Pmodel)-mean(Ptruth))/abs(mean(PGC)-mean(Ptruth))',aggregate='Mean relative power mismatch over nine fields and positive modes, then pool all40 leads before ratio. Day10 uses lead40.',limits='Descriptive extension on an already explored year, not independent confirmation. No seed uncertainty claimed. Zero denominator bins remain undefined.'))
 write(root/'complete.json',dict(status='complete',models=[m['id'] for m in meta['models']],starts=122,leads=40,report_sha256=sha(out/'report.json')))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['worker','report']);p.add_argument('--root',type=Path,required=True);p.add_argument('--task',type=int,default=0);a=p.parse_args()
 if a.stage=='report':report(a.root)
 else:
  meta=verify(a.root);mi,shard=divmod(a.task,meta['shards']);assert mi<len(meta['models'])
  from scripts.analyze_models.evaluate_v24_amse_regime import evaluate
  evaluate(a.root/f'shards/{shard:02d}/manifest.json',mi,'spectral')
