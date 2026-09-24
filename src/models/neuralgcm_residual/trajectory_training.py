"""K=1/K=2 live training: SG physical feedback and exact memory BPTT.

Loss is differentiated jointly across batch/time before a host-tape reverse
pass, preserving the gradient of coupled batch-bias terms.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .model import zero_memory
from .native_state import stop
from .paper_loss import modal_prediction


def stack(values):
    return jax.tree_util.tree_map(lambda *v: jnp.stack(v), *values)


class TrajectoryTrainer:
    def __init__(self, runtime, loss, *, peak_learning_rate=.002, warmup_steps=2000):
        self.runtime, self.loss = runtime, loss
        self.optimizer = optax.adam(optax.warmup_exponential_decay_schedule(
            init_value=0., peak_value=peak_learning_rate, warmup_steps=warmup_steps,
            transition_steps=10000, decay_rate=.5, transition_begin=15000-warmup_steps),
            b1=.9, b2=.95, eps=1e-6)
        b, a, n = runtime.backbone, runtime.adapter, runtime.normalization

        def step(params, memory, key, features, known, baseline, forcing):
            delta, memory = runtime.branch.apply(params, memory, key, stop(features), stop(known))
            corrected = a.apply_increment(stop(baseline), n.increment(delta))
            return modal_prediction(b, corrected, forcing), memory, corrected

        self.forward = jax.jit(step)
        @jax.jit
        def backward(params, memory, key, record, output_grad, memory_grad):
            def f(p,h):
                out, h, _ = step(p,h,key,*record)
                return out,h
            _, pullback = jax.vjp(jax.checkpoint(f), params, memory)
            return pullback((output_grad,memory_grad))
        self.backward = backward
        self.loss_grad = jax.jit(jax.value_and_grad(loss.terms, has_aux=True))
        self.score = jax.jit(loss.terms)
        self.encode = jax.jit(b.encode)

        @jax.jit
        def apply(params, state, gradient):
            updates, state = self.optimizer.update(gradient,state,params)
            return optax.apply_updates(params,updates),state,optax.global_norm(gradient)
        self.apply_update = apply

    def batch(self, params, key, origins, *, gradients=True, capture=False, initial_memory=None):
        from .config import FIELDS
        from .native_state import NATIVE_FIELDS, field_value
        from .loss import gaussian_weights
        r = self.runtime
        predictions, targets, tapes = [], [], []
        physical_mses = []
        area = gaussian_weights(r.backbone.model.data_coords.horizontal.latitudes) if not gradients else None
        for origin in origins:
            inputs, forcing = r.store.inputs_and_forcing(r.backbone.model, origin)
            physical = self.encode(inputs,forcing)
            memory = zero_memory(r.memory) if initial_memory is None else initial_memory
            forecast, truth, tape, physical_errors = [], [], [], []
            for lead in range(1,self.loss.steps+1):
                baseline = stop(r.backbone.advance_6h(stop(physical),forcing))
                record = (stop(r.normalization.inputs(r.adapter.features(physical))),
                          stop(r.known(physical,forcing)),baseline,forcing)
                key, sub = jax.random.split(key)
                tape.append((jax.device_get(memory),jax.device_get(sub),jax.device_get(record)))
                pred, memory, physical = self.forward(params,memory,sub,*record)
                forecast.append(jax.device_get(pred))
                physical = stop(physical)
                valid = np.datetime64(origin,'h')+np.timedelta64(6*lead,'h')
                data = r.store.frame(valid)
                if not gradients:
                    decoded = jax.device_get(r.backbone.decode(physical,forcing))
                    physical_errors.append({k:np.mean(np.sum(
                        (np.asarray(decoded[k],np.float64)-data[k])**2*area,axis=-1),axis=-1)
                        for k in FIELDS})
                inp = {k:data[k] for k in FIELDS}
                inp['sim_time'] = np.asarray(r.backbone.model.datetime64_to_sim_time(valid),np.float32)
                encoded = self.encode(inp,forcing)
                truth.append(jax.device_get({
                    'data':{k:r.backbone.model.data_coords.horizontal.to_modal(data[k]) for k in FIELDS},
                    'model':{k:field_value(encoded,k) for k,_ in NATIVE_FIELDS}}))
            predictions.append(stack(forecast));targets.append(stack(truth));tapes.append(tape)
            if not gradients:
                physical_mses.append({k:np.stack([x[k] for x in physical_errors]) for k in FIELDS})
        prediction,target = stack(predictions),stack(targets)
        if capture:
            self.captured = dict(prediction=jax.device_get(prediction),target=jax.device_get(target),
                                 tapes=tapes,initial_memory=jax.device_get(
                                     zero_memory(r.memory) if initial_memory is None else initial_memory))
        if not gradients:
            value,details = self.score(prediction,target)
            details = jax.device_get(details)
            details['physical_mse'] = {k:np.mean([x[k] for x in physical_mses],axis=0) for k in FIELDS}
            return float(value),details,key,None
        (value,details),cotangents = self.loss_grad(prediction,target)
        gradient = jax.tree_util.tree_map(jnp.zeros_like,params)
        for i,tape in enumerate(tapes):
            h_grad = jax.tree_util.tree_map(jnp.zeros_like,tape[-1][0])
            for j in reversed(range(len(tape))):
                h,sub,record = tape[j]
                out_grad = jax.tree_util.tree_map(lambda x:x[i,j],cotangents)
                dp,h_grad = self.backward(params,h,sub,record,out_grad,h_grad)
                gradient = jax.tree_util.tree_map(jnp.add,gradient,dp)
        if not np.isfinite(float(value)) or any(not np.isfinite(x).all() for x in
                jax.device_get(jax.tree_util.tree_leaves(gradient))):
            raise FloatingPointError('Nonfinite loss or gradient; no optimizer update applied')
        return float(value),jax.device_get(details),key,gradient
