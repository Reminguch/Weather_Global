"""Independent representative ERA5 + real NeuralGCM GPU integration smoke."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resources',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    from src.models.neuralgcm_residual.paper_execution import configure_paper_environment
    configure_paper_environment()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from src.models.neuralgcm_residual.runtime import build_runtime
    from src.models.neuralgcm_residual.config import RunConfig,Architecture,BALANCED_LOSS_NAME
    from src.models.neuralgcm_residual.data import seasonal_order,eligible_origins
    from src.models.neuralgcm_residual.paper_statistics import fit_statistics
    from src.models.neuralgcm_residual.pressure_weighted_loss import PressureWeightedLoss
    from src.models.neuralgcm_residual.checks import backbone_parameter_digest,tree_comparison
    from src.models.neuralgcm_residual.io import read_json,write_json,sha256
    from experimental.adaptive_loss.objective import GroupedLoss,calibrate
    from experimental.adaptive_loss.trainer import AdaptiveTrainer,finite_tree,tree_norm
    from experimental.adaptive_loss.controller import WindowBalancer,BalanceConfig
    if not os.environ.get('SLURM_JOB_ID') or not all(d.platform=='gpu' for d in jax.devices()):
        raise RuntimeError('Slurm GPU allocation required')
    a.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    def log(**x):print(json.dumps(x,allow_nan=False),flush=True)
    resources=read_json(a.resources)
    r=build_runtime(RunConfig(architecture=Architecture(width=128,d_inner=16),
        loss=BALANCED_LOSS_NAME,correction_policy='no_pressure_zero_mean_v2'),resources,stage='finetune')
    before=backbone_parameter_digest(r.backbone.model)
    stats=fit_statistics(r.backbone.model,r.store,10,progress=lambda x:log(**x))
    stats.update(scope='independent adaptive-loss pilot only',checkpoint_sha256=resources['checkpoint_sha256'])
    write_json(a.output/'pilot_statistics.json',stats,immutable=True)
    grouped=GroupedLoss(r.backbone.model,stats,2)
    fixed=PressureWeightedLoss(r.backbone.model,stats,2,upper_pressure_weight=1e-5)
    trainer=AdaptiveTrainer(r,fixed,grouped,peak_learning_rate=1e-4,warmup_steps=4)
    origins=seasonal_order(eligible_origins(r.store.times,'train',2,warm_hours=0,daily=False))
    batch=origins[:2]
    log(stage='real_data_gradient_checks',origins=batch)
    params,key=r.params,jax.random.PRNGKey(22)
    value,detail,key,g=trainer.training_batch(params,key,batch,np.ones(len(grouped.names)))
    np.testing.assert_allclose(value,detail['fixed_loss'],rtol=3e-5)
    cap=trainer.captured
    (ref,_),cot=trainer.loss_grad(cap['prediction'],cap['target'])
    expected=trainer.reverse(params,cot)
    # Real parameter-gradient equivalence (not only output cotangents).
    error=tree_norm(jax.tree_util.tree_map(lambda x,y:x-y,g,expected))/max(tree_norm(expected),1e-20)
    if error>3e-4:raise AssertionError('Grouped gradient disagrees with legacy loss')
    (_, _),double_cot=trainer.weighted_grad(cap['prediction'],cap['target'],jnp.ones(len(grouped.names))*2)
    doubled=trainer.reverse(params,double_cot)
    double_error=tree_norm(jax.tree_util.tree_map(lambda x,y:x-2*y,doubled,g))/max(tree_norm(g),1e-20)
    if double_error>3e-4:raise AssertionError('JIT ignored dynamic coefficients')
    pressure=grouped.names.index('model/log_surface_pressure')
    (_, _),pcot=trainer.weighted_grad(cap['prediction'],cap['target'],jnp.zeros(len(grouped.names)).at[pressure].set(1))
    pressure_norm=tree_norm(trainer.reverse(params,pcot))
    assert pressure_norm==0,'Disconnected native pressure unexpectedly has a gradient'
    calibration=calibrate([detail['groups']],grouped.priors)
    write_json(a.output/'calibration.json',dict(calibration,origins=batch,scope='pilot training only'))
    base=np.asarray(calibration['coefficients'],np.float32)
    controller=WindowBalancer(grouped.names,BalanceConfig(window=4,interval=2),excluded=grouped.excluded)
    opt=trainer.optimizer.init(params)
    rows=[]
    for update in range(6):
        batch=origins[update*2:update*2+2]
        used=controller.weights.copy()
        value,detail,key,g=trainer.training_batch(params,key,batch,base*used,
            probe_coefficients=base if controller.needs_probe() else None)
        params,opt,norm=trainer.apply_update(params,opt,g)
        finite_tree((params,opt),'optimizer result')
        controller.observe(detail['groups']*base,detail['probe_norms'])
        # JSON restoration exercises all history buffers and the probe clock.
        restored=WindowBalancer.from_state(json.loads(json.dumps(controller.state_dict())))
        assert restored.state_dict()==controller.state_dict()
        row=dict(update=update+1,loss=value,fixed_loss=detail['fixed_loss'],gradient_norm=float(norm),
                 weights_used=used.tolist(),weights_next=controller.weights.tolist(),
                 probe_norms=None if detail['probe_norms'] is None else detail['probe_norms'].tolist())
        rows.append(row);log(stage='update',**row)
    delta=tree_norm(jax.tree_util.tree_map(lambda x,y:x-y,params,r.params))
    assert delta>0 and not np.allclose(controller.weights,1),'No real update / no dynamic weight change'
    assert before==backbone_parameter_digest(r.backbone.model),'Frozen backbone changed'
    report=dict(passed=True,pilot=True,K=2,width=128,d_inner=16,updates=6,
        dynamic_jit_relative_error=double_error,fixed_gradient_relative_error=error,
        native_pressure_gradient_norm=pressure_norm,parameter_change_norm=delta,
        controller_json_restore=True,backbone_frozen=True,group_names=grouped.names,
        rows=rows,seconds=time.monotonic()-started,slurm_job_id=os.environ['SLURM_JOB_ID'],
        resources_sha256=sha256(a.resources),pilot_statistics_sha256=sha256(a.output/'pilot_statistics.json'),
        source_hashes={str(q.relative_to(ROOT)):sha256(q) for q in sorted((ROOT/'experimental/adaptive_loss').glob('*.py'))},
        devices=[d.device_kind for d in jax.devices()])
    write_json(a.output/'report.json',report,immutable=True)
    log(passed=True,seconds=report['seconds'])


if __name__=='__main__':main()
