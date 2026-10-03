"""One atomic checkpoint is the authority for weights, controller and logs.

The JSONL files and best.pkl are materialized views of that checkpoint. Resume
rebuilds them, so a killed writer cannot commit an incomplete log or future best.
"""
from pathlib import Path

import jax
import numpy as np

from src.models.neuralgcm_residual.io import atomic_bytes,write_pickle
from experimental.adaptive_loss.controller import WindowBalancer


def validate(state, identity, params_template, optimizer_template, budget):
    if state.get('schema')!='adaptive_resume_v1' or state.get('identity')!=identity:
        raise ValueError('Checkpoint identity/schema mismatch')
    step=state['update']
    if type(step) is not int or not 0<=step<=budget:
        raise ValueError('Checkpoint update outside budget')
    for name,template in [('params',params_template),('optimizer',optimizer_template),('rng',np.zeros(2,np.uint32))]:
        actual=state[name]
        if jax.tree_util.tree_structure(actual)!=jax.tree_util.tree_structure(template):
            raise ValueError('Checkpoint structure mismatch: '+name)
        for x,y in zip(jax.tree_util.tree_leaves(actual),jax.tree_util.tree_leaves(template),strict=True):
            if x.shape!=y.shape or x.dtype!=y.dtype or not np.isfinite(x).all():
                raise ValueError('Checkpoint shape/dtype/finite mismatch: '+name)
    def counts(value):
        if hasattr(value,'_fields'):
            if 'count' in value._fields:yield int(value.count)
            for x in value:yield from counts(x)
        elif isinstance(value,(list,tuple)):
            for x in value:yield from counts(x)
    counters=list(counts(state['optimizer']))
    if not counters or any(x!=step for x in counters):
        raise ValueError('Optimizer clock mismatch')
    if [x['update'] for x in state['metrics']]!=list(range(1,step+1)):
        raise ValueError('Incomplete committed metrics')
    validation_steps=[x['update'] for x in state['validations']]
    if not validation_steps or validation_steps[0]!=0 or validation_steps!=sorted(set(validation_steps)) or validation_steps[-1]>step:
        raise ValueError('Invalid committed validation history')
    best=state['best']
    candidates=[x for x in state['validations'] if x['eligible']]
    selected=min(candidates,key=lambda x:x['selection_score'])
    if best['update']!=selected['update'] or best['selection_score']!=selected['selection_score']:
        raise ValueError('Best checkpoint does not match fixed-metric history')
    if jax.tree_util.tree_structure(best['params'])!=jax.tree_util.tree_structure(params_template):
        raise ValueError('Best parameter structure mismatch')
    for x,y in zip(jax.tree_util.tree_leaves(best['params']),jax.tree_util.tree_leaves(params_template),strict=True):
        if x.shape!=y.shape or x.dtype!=y.dtype or not np.isfinite(x).all():
            raise ValueError('Best parameters invalid')
    if state['controller'] is not None:
        controller=WindowBalancer.from_state(state['controller'])
        if controller.update!=step:
            raise ValueError('Controller and optimizer clocks differ')
    if not np.isfinite(state['calibration']['coefficients']).all():
        raise ValueError('Calibration invalid')


def materialize(output,state):
    import json
    output=Path(output)
    for name,key in [('metrics.jsonl','metrics'),('validation.jsonl','validations')]:
        body=''.join(json.dumps(row,sort_keys=True,allow_nan=False)+'\n' for row in state[key])
        atomic_bytes(output/name,body.encode())
    write_pickle(output/'best.pkl',state['best'])


def commit(output,state):
    # CPU copies avoid backend-specific device objects in the durable artifact.
    state=jax.device_get(state)
    write_pickle(Path(output)/'last.pkl',state)
    materialize(output,state)
