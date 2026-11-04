"""Cached supervised regression, full 24-step BPTT, exact optimizer/cursor resume."""
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
from .model import zero_memory
from .timeline import TrainingCursor,advance_cursor
from .loss import weighted_mse,contributions
from .artifacts import save_checkpoint,load_checkpoint,write_json,sha256,source_identity
from src.models.neuralgcm_residual.io import append_jsonl
from src.models.mamba.v24_Ilya.training.endpoint_step import build_optimizer


def chunk_loss(apply,params,state,batch,key,latitude,reset_every_step=False):
    # Entry-state is carried data, never a differentiable link to a preceding chunk.
    state = jax.tree_util.tree_map(jax.lax.stop_gradient,state)
    keys = jax.random.split(key,len(batch['features']))
    def step(memory,values):
        features,target,rng = values
        if reset_every_step:
            memory = zero_memory(memory)
        out,memory = apply(params,memory,rng,features)
        return memory,weighted_mse(out,target,latitude)
    state,losses = jax.lax.scan(jax.checkpoint(step),state,(batch['features'],batch['target'],keys))
    return losses.mean(),state


def validate(model,params,initial_state,segments,load,latitude,config):
    totals,baseline = np.zeros(24),np.zeros(24)
    count=0
    predict = jax.jit(model.apply)
    for segment in segments:
        state=zero_memory(initial_state)
        for off in range(0,96,24):
            batch=load(int(segment[off]))
            for i in range(24):
                if config.reset_every_step:
                    state=zero_memory(state)
                out,state=predict(params,state,jax.random.PRNGKey(config.seed),batch['features'][i])
                totals[i]+=float(weighted_mse(out,batch['target'][i],latitude))
                baseline[i]+=float(weighted_mse(jnp.zeros_like(out),batch['target'][i],latitude))
            count+=1
    if not count or not np.isfinite(totals).all():
        raise ValueError('Empty or nonfinite validation')
    return {'per_step':(totals/count).tolist(),'baseline_per_step':(baseline/count).tolist(),
            'one_step_prefix_mean':float(totals[:3].mean()/count),'tail_leads_hours':list(range(6,127,6)),
            'tail':(totals[3:]/count).tolist(),'segments':len(segments),'selection':'monitor_only'}


def census(segments,load,latitude,path,upstream):
    totals=np.zeros((24,7,29),dtype=np.float64)
    n=0
    for segment in segments:
        for off in range(0,96,24):
            batch=load(int(segment[off]))
            for i,target in enumerate(batch['target']):
                totals[i]+=np.asarray(contributions(jnp.zeros_like(target),target,latitude))
            n+=1
    if not n or not np.isfinite(totals).all() or totals.sum()<=0:
        raise ValueError('Invalid zero-head validation census')
    totals/=n
    shares=totals.mean(axis=0)/totals.mean(axis=0).sum()
    hot=[{'field_index':int(v),'level_index':int(p),'share':float(shares[v,p])}
         for v,p in zip(*np.where(shares>.10))]
    classes=np.concatenate((totals[:3].mean(axis=0,keepdims=True),totals[3:]),axis=0)
    denominators=classes.sum(axis=(1,2),keepdims=True)
    class_shares=np.divide(classes,denominators,out=np.zeros_like(classes),where=denominators>0)
    class_names=['one_step_prefix']+[f'tail_{h}h' for h in range(6,127,6)]
    class_hot=[{'step_class':class_names[c],'field_index':int(v),'level_index':int(p),
                'share':float(class_shares[c,v,p])} for c,v,p in zip(*np.where(class_shares>.10))]
    result={'upstream':upstream,'source_identity':source_identity(),'loss_by_step_field_level':totals.tolist(),
            'share_by_field_level':shares.tolist(),'above_ten_percent':hot,'chunks':n,
            'step_class_share':class_shares.tolist(),'step_class_names':class_names,
            'above_ten_percent_by_class':class_hot,'decision_required':bool(hot or class_hot)}
    write_json(path,result,immutable=True)
    return result


def run(model,config,segments,val_segments,load,latitude,root,identity,steps=None,resume=None):
    from .runtime import require_production_policy
    require_production_policy()
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    from dataclasses import asdict
    write_json(root/'run.json',{'identity':identity,'source_identity':source_identity(),
                               'config':asdict(config)},immutable=True)
    optimizer,schedule=build_optimizer(config.v24())
    # A bounded invocation does not alter the predeclared 10k LR schedule.
    endpoint=min(config.max_steps,steps if steps is not None else config.max_steps)
    if resume:
        saved=load_checkpoint(resume,identity)
        params,state,opt_state,key=jax.device_put((saved['params'],saved['memory'],saved['opt_state'],saved['rng']))
        cursor=TrainingCursor.from_mapping(saved['cursor'])
        start=saved['step']
    else:
        if any(root.glob('step_*.pkl')):
            raise FileExistsError('Existing run requires --resume')
        batch=load(int(segments[0][0]))
        key=jax.random.PRNGKey(config.seed)
        params,state=model.init(key,batch['features'][0])
        key,_=jax.random.split(key)
        state=zero_memory(state)
        opt_state=optimizer.init(params)
        cursor=TrainingCursor()
        start=0
    if endpoint < start or endpoint < 1:
        raise ValueError('Requested endpoint precedes the resumed checkpoint or update 1')
    initial_state=zero_memory(state)
    def update(params,state,opt_state,batch,key):
        (loss,memory),grads=jax.value_and_grad(lambda p: chunk_loss(
            model.apply,p,state,batch,key,latitude,config.reset_every_step),has_aux=True)(params)
        updates,opt_state=optimizer.update(grads,opt_state,params)
        return optax.apply_updates(params,updates),memory,opt_state,loss,optax.global_norm(grads)
    update=jax.jit(update)
    for step in range(start+1,endpoint+1):
        clock=time.monotonic()
        if cursor.segment_offset==0:
            state=zero_memory(state)
        a0=int(segments[cursor.segment_index][cursor.segment_offset])
        batch=load(a0)
        key,subkey=jax.random.split(key)
        params,state,opt_state,value,grad_norm=update(params,state,opt_state,batch,subkey)
        value,grad_norm=float(value),float(grad_norm)
        if not np.isfinite(value+grad_norm) or not all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves((params,state,opt_state))):
            raise FloatingPointError(f'Nonfinite update {step}; checkpoint not published')
        state=jax.tree_util.tree_map(jax.lax.stop_gradient,state)
        cursor=advance_cursor(cursor,config.v24(),len(segments))
        record={'step':step,'a0':a0,'loss':value,'grad_norm':grad_norm,'lr':float(schedule(step-1)),
                'seconds':time.monotonic()-clock,'cursor':cursor.to_dict()}
        append_jsonl(root/'training.jsonl',record)
        print(record,flush=True)
        if step%config.checkpoint_every==0 or step==endpoint:
            save_checkpoint(root/f'step_{step:05d}.pkl',{'params':params,'memory':state,'opt_state':opt_state,
                'rng':key,'cursor':cursor.to_dict(),'step':step,'identity':identity})
        if step%config.validation_every==0:
            report=validate(model,params,initial_state,val_segments,load,latitude,config)
            write_json(root/f'validation_{step:05d}.json',report,immutable=True)
    return {'step':endpoint,'identity':identity}
