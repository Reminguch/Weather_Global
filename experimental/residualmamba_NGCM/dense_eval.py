#!/usr/bin/env python3
"""Checkpoint monitoring on the same complete tail20 validation set."""
from pathlib import Path
import argparse
import sys
import json
import time
import numpy as np


def configure(arm_root):
    plan=json.loads((arm_root/'plan.json').read_text())
    sys.path.insert(0,str(Path(plan['code_snapshot'])/'experimental/residualmamba_NGCM'))
    import diagnostic_2k as base
    is_tail1='parent_root' in plan
    if is_tail1:
        import tail1_2k as t1
        plan,pp,setup,store,geometry,stats=t1.context(arm_root)
        parent=Path(plan['parent_root']);identity=t1.training_id(arm_root)
        cfg=t1.ext.Tail1Config()
    else:
        base.checked_plan(arm_root)
        parent=arm_root;pp=plan;setup,store,geometry=base.context(parent,pp)
        stats=base.load_arrays(parent/'stats.npz')[0];identity=base.training_identity(parent,pp)
        cfg=base.Config()
    return base,plan,parent,pp,store,geometry,stats,identity,cfg


def evaluate_segment(model,params,initial_memory,segment,load,latitude,seed):
    import jax
    import jax.numpy as jnp
    from residualmamba_ngcm.model import zero_memory
    from residualmamba_ngcm.loss import contributions
    # The caller passes a compiled function so all segments share one executable.
    predict=model
    memory=zero_memory(initial_memory);rows=[];native=[]
    for off in range(0,96,24):
        batch=load(int(segment[off]))
        memory,loss,baseline=predict(params,memory,batch)
        rows.append(np.asarray(loss,dtype=np.float64));native.append(np.asarray(baseline,dtype=np.float64))
    return np.mean(rows,axis=0),np.mean(native,axis=0)


def predictor(model,latitude,seed):
    import jax
    import jax.numpy as jnp
    from residualmamba_ngcm.loss import contributions
    key=jax.random.PRNGKey(seed)
    def predict(params,memory,batch):
        def step(state,data):
            features,target=data
            out,state=model.apply(params,state,key,features)
            return state,(contributions(out,target,latitude),contributions(jnp.zeros_like(out),target,latitude))
        memory,(loss,baseline)=jax.lax.scan(step,memory,(batch['features'],batch['target']))
        return memory,loss,baseline
    return jax.jit(predict)


def evaluate(root,arm_root,checkpoint,deadline):
    import jax
    base,plan,parent,pp,store,geometry,stats,identity,cfg=configure(arm_root)
    from residualmamba_ngcm.artifacts import read_json,write_json,sha256,load_checkpoint,digest
    from residualmamba_ngcm.cache import load_chunk
    from residualmamba_ngcm.config import Config,FIELDS
    from residualmamba_ngcm.model import make_model,read_model_config
    from residualmamba_ngcm.timeline import split_segments
    saved=load_checkpoint(checkpoint,identity);step=saved['step']
    arm='tail1' if 'parent_root' in plan else 'tail20'
    directory=root/arm/f'step_{step:05d}';cksha=sha256(checkpoint)
    protocol={'validation':'all 15 complete 2022 segments, native tail20 inputs, leads 6..126h',
        'cache_ready_sha256':sha256(parent/'cache/READY.json'),'stats_sha256':sha256(parent/'stats.npz'),
        'geometry_sha256':sha256(parent/'geometry.npz'),'seed':22,'memory':'reset each 96-anchor segment, carry all 4 chunks',
        'evaluator_sha256':sha256(Path(__file__))}
    write_json(root/'protocol.json',protocol,immutable=True)
    if (directory/'summary.json').exists():
        if read_json(directory/'summary.json')['checkpoint_sha256']!=cksha:raise ValueError('Checkpoint changed')
        return True
    params,memory=jax.device_put((saved['params'],saved['memory']))
    model=make_model(read_model_config(pp['gc_checkpoint'],base.GC_SHA),cfg,geometry['latitude'],geometry['longitude'])
    predict=predictor(model,geometry['latitude'],cfg.seed)
    load=lambda a0:load_chunk(store,parent/'cache',a0,stats,Config(),geometry['static'],geometry['latitude'],geometry['longitude'])
    segments=split_segments(store.times,'val')[0]
    for i,segment in enumerate(segments):
        path=directory/f'segment_{i:02d}.json'
        if path.exists():
            if read_json(path)['checkpoint_sha256']!=cksha:raise ValueError('Checkpoint changed')
            continue
        if time.monotonic()+180>=deadline:return False
        started=time.monotonic();loss,baseline=evaluate_segment(predict,params,memory,segment,load,geometry['latitude'],cfg.seed)
        if not np.isfinite(loss).all() or not np.isfinite(baseline).all():raise FloatingPointError('Nonfinite validation')
        write_json(path,{'checkpoint_sha256':cksha,'protocol_hash':digest(protocol),
            'loss_by_step_field_level':loss.tolist(),'baseline_by_step_field_level':baseline.tolist(),
            'seconds':time.monotonic()-started},immutable=True)
        print(f'dense eval {arm} step={step} segment={i+1}/15',flush=True)
    rows=[read_json(directory/f'segment_{i:02d}.json') for i in range(15)]
    if any(r['checkpoint_sha256']!=cksha or r['protocol_hash']!=digest(protocol) for r in rows):raise ValueError('Mixed evaluation')
    loss=np.mean([r['loss_by_step_field_level'] for r in rows],axis=0)
    baseline=np.mean([r['baseline_by_step_field_level'] for r in rows],axis=0)
    value=float(loss.sum(axis=(1,2)).mean());ref=float(baseline.sum(axis=(1,2)).mean())
    fields=FIELDS[:loss.shape[1]]
    result={'arm':arm,'step':step,'segments':15,'chunks':60,'checkpoint':str(checkpoint),
        'checkpoint_sha256':cksha,'protocol_hash':digest(protocol),'mean_loss':value,
        'baseline_mean_loss':ref,'improvement_percent':100*(1-value/ref),
        'per_step':loss.sum(axis=(1,2)).tolist(),'baseline_per_step':baseline.sum(axis=(1,2)).tolist(),
        'per_field':{f:float(loss.mean(axis=0).sum(axis=1)[i]) for i,f in enumerate(fields)},
        'baseline_per_field':{f:float(baseline.mean(axis=0).sum(axis=1)[i]) for i,f in enumerate(fields)},
        'completed_unix':time.time(),'parameter_selection':'monitor only; no validation-based tuning'}
    write_json(directory/'summary.json',result,immutable=True)
    plot(root)
    print(json.dumps(result),flush=True)
    return True


def plot(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[json.loads(p.read_text()) for p in root.glob('tail*/step_*/summary.json')]
    if not rows:return
    fields=list(rows[0]['per_field']);fig,axes=plt.subplots(4,2,figsize=(13,12),constrained_layout=True);axes=axes.ravel()
    for arm in ('tail20','tail1'):
        values=sorted([r for r in rows if r['arm']==arm],key=lambda r:r['step'])
        if not values:continue
        steps=[r['step'] for r in values]
        axes[0].plot([0]+steps,[values[0]['baseline_mean_loss']]+[r['mean_loss'] for r in values],marker='o',label=arm)
        for i,f in enumerate(fields,1):
            axes[i].plot([0]+steps,[values[0]['baseline_per_field'][f]]+[r['per_field'][f] for r in values],marker='o',label=arm)
    axes[0].set_title('Full fixed validation objective')
    axes[0].axhline(rows[0]['baseline_mean_loss'],ls='--',color='grey',label='Official NeuralGCM zero head')
    for i,f in enumerate(fields,1):
        axes[i].set_title(f);axes[i].axhline(rows[0]['baseline_per_field'][f],ls='--',color='grey')
    for ax in axes:ax.set_xlabel('Training updates');ax.set_ylabel('Weighted normalized MSE');ax.grid(alpha=.2);ax.legend()
    fig.savefig(root/'eval-loss.png',dpi=160);plt.close(fig)
    (root/'summary.json').write_text(json.dumps(sorted(rows,key=lambda r:(r['arm'],r['step'])),indent=2)+'\n')
    (root/'REPORT.md').write_text('# Dense checkpoint validation\n\nEvery point evaluates all 15 fixed validation segments (60 chunks) on the same tail20 inputs. '
        'Step zero is the zero-head official NeuralGCM reference. Values are normalized weighted loss, not raw physical RMSE. '
        'The per-field panels show contributions to that loss. Training continues independently.\n\n![Validation loss](eval-loss.png)\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm-root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);a=p.parse_args()
    if not str(a.root.resolve()).startswith('/scratch/'):raise ValueError('Scratch required')
    # Configure imports/runtime policy before initializing the GPU backend.
    configure(a.arm_root)
    import jax
    if jax.devices()[0].device_kind!='NVIDIA H200':raise RuntimeError('H200 required')
    evaluate(a.root,a.arm_root,a.checkpoint,time.monotonic()+3000)

if __name__=='__main__':main()
