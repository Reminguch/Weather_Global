"""Evaluate immutable in-training checkpoints on all ERA5 fields for 6..120 h."""
import argparse,hashlib,os,pickle,subprocess,sys,time
from pathlib import Path
from common import read,write,digest,FIELDS

class PrefixSmokeError(RuntimeError):pass

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--batch',type=Path,required=True);a=p.parse_args()
    plan=read(a.root/'plan.json');batch=read(a.batch);sys.path.insert(0,plan['source_root'])
    from src.models.neuralgcm_residual.paper_execution import configure_paper_environment
    configure_paper_environment()
    import jax
    import numpy as np
    from src.models.neuralgcm_residual.runtime import build_runtime
    from src.models.neuralgcm_residual.config import RunConfig,Architecture,BALANCED_LOSS_NAME
    from src.models.neuralgcm_residual.data import seasonal_order,eligible_origins
    from src.models.neuralgcm_residual.pressure_weighted_loss import PressureWeightedLoss
    from src.models.neuralgcm_residual.trajectory_training import TrajectoryTrainer
    from src.models.neuralgcm_residual.checks import backbone_parameter_digest
    from src.models.neuralgcm_residual.io import sha256
    from experimental.weather_only.train import source_hashes
    if not os.environ.get('SLURM_JOB_ID') or not all(x.platform=='gpu' for x in jax.devices()):raise RuntimeError('Slurm GPU required')
    started=time.monotonic();hashes=source_hashes();resources=read(plan['resources']);stats=read(plan['statistics'])
    metas=[read(a.root/'snapshots'/identity/'metadata.json') for identity in batch['snapshots']]
    if len(metas)>3:raise ValueError('At most three checkpoints per one-hour evaluation job')
    for meta in metas:
        config=meta['config']
        if config['source_hashes']!=hashes or config['statistics_sha256']!=sha256(plan['statistics']) or config['resources_sha256']!=sha256(plan['resources']):raise ValueError('Checkpoint source/statistics/resources differ')
        if config['validation_origins']!=metas[0]['config']['validation_origins']:raise ValueError('Validation origins differ')
    runtime=build_runtime(RunConfig(architecture=Architecture(width=128,d_inner=16),loss=BALANCED_LOSS_NAME,correction_policy='no_pressure_zero_mean_v2'),resources,stage='finetune')
    if stats['dataset_id']!=runtime.store.identity or stats['checkpoint_sha256']!=resources['checkpoint_sha256']:raise ValueError('Statistics provenance mismatch')
    backbone_before=backbone_parameter_digest(runtime.backbone.model)
    trainer=TrajectoryTrainer(runtime,PressureWeightedLoss(runtime.backbone.model,stats,20,upper_pressure_weight=1e-5))
    shorter=TrajectoryTrainer(runtime,PressureWeightedLoss(runtime.backbone.model,stats,2,upper_pressure_weight=1e-5))
    pool=seasonal_order(eligible_origins(runtime.store.times,'val',20,warm_hours=0))
    excluded=set(metas[0]['config']['validation_origins']);pool=[x for x in pool if x not in excluded]
    origins=pool[plan['origin_offset']:plan['origin_offset']+plan['long_origins']]
    if len(origins)!=plan['long_origins'] or excluded.intersection(origins):raise ValueError('Invalid held-out origins')
    pressure=np.asarray(runtime.backbone.model.data_coords.vertical.centers).tolist()
    signature=dict(origins=origins,pressure_hpa=pressure,source_hashes=hashes,resources_sha256=sha256(plan['resources']),statistics_sha256=sha256(plan['statistics']),eval_K=20,hardware='H200')
    identity=digest(signature)
    def evaluate(params,check_prefix=False):
        details=[]
        for i in range(0,len(origins),2):
            loss,detail,_,_=trainer.batch(params,jax.random.PRNGKey(0),origins[i:i+2],gradients=False)
            if not np.isfinite(loss) or any(not np.isfinite(detail['physical_mse'][field]).all() for field in FIELDS):raise FloatingPointError('Nonfinite forecast at origins '+str(origins[i:i+2]))
            if i==0 and check_prefix:
                _,reference,_,_=shorter.batch(params,jax.random.PRNGKey(0),origins[:2],gradients=False)
                try:
                    for field in FIELDS:np.testing.assert_allclose(detail['physical_mse'][field][:2],reference['physical_mse'][field],rtol=1e-5,atol=1e-12,err_msg=field)
                except AssertionError as error:raise PrefixSmokeError('K2/K20 physical-prefix smoke failed') from error
            details.append(detail['physical_mse'])
        return {field:np.sqrt(np.mean([d[field] for d in details],axis=0)).tolist() for field in FIELDS}
    baseline_path=a.root/'baselines'/('H200_'+identity+'.json')
    if baseline_path.exists():
        baseline=read(baseline_path)
        if baseline['signature']!=signature:raise ValueError('Baseline cache provenance mismatch')
    else:
        baseline=dict(signature=signature,physical_rmse=evaluate(runtime.params,True),lead_hours=list(range(6,121,6)),prefix_smoke_passed=True)
        write(baseline_path,baseline)
    for index,meta in enumerate(metas):
        result_path=a.root/'results'/(meta['id']+'.json')
        if result_path.exists():continue
        if time.monotonic()-started>3100:raise TimeoutError('Evaluation budget exhausted; unprocessed checkpoints retain no result')
        path=a.root/'snapshots'/meta['id']/'checkpoint.pkl'
        if sha256(path)!=meta['checkpoint_sha256']:raise ValueError('Archived checkpoint changed')
        saved=pickle.loads(path.read_bytes());params=saved['params']
        if saved['update']!=meta['update'] or jax.tree_util.tree_structure(params)!=jax.tree_util.tree_structure(runtime.params):raise ValueError('Checkpoint structure differs')
        for x,y in zip(jax.tree_util.tree_leaves(params),jax.tree_util.tree_leaves(runtime.params),strict=True):
            if x.shape!=y.shape or x.dtype!=y.dtype or not np.isfinite(x).all():raise ValueError('Invalid parameter array')
        record={k:meta[k] for k in ('id','hardware','mode','update','kind','training_compute_seconds','checkpoint_sha256')}
        record.update(baseline_path=str(baseline_path),eval_identity=identity,origins=origins,pressure_hpa=pressure,lead_hours=list(range(6,121,6)),optimizer_updates=0,job_id=os.environ['SLURM_JOB_ID'],script_sha256=sha256(__file__))
        try:record.update(finite=True,physical_rmse=evaluate(params,index==0),prefix_smoke_passed=index==0)
        except PrefixSmokeError:raise
        except Exception as error:record.update(finite=False,error_type=type(error).__name__,message=str(error))
        write(result_path,record);print(__import__('json').dumps({k:v for k,v in record.items() if k in ('id','finite','update','job_id','error_type','message')}),flush=True)
    if backbone_before!=backbone_parameter_digest(runtime.backbone.model):raise AssertionError('Backbone parameters changed')
    write(a.root/'batch_results'/(a.batch.stem+'.json'),dict(completed=True,backbone_frozen=True,snapshots=batch['snapshots'],job_id=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-started))
    reporter=Path(__file__).with_name('report.py')
    if reporter.exists():subprocess.run([sys.executable,str(reporter),'--root',str(a.root)],check=True)

if __name__=='__main__':main()
