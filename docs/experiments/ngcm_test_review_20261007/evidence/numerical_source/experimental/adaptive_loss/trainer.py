"""Dynamic loss coefficients over the existing frozen-physics host tape.

Reuses the tested forward solver and memory-only reverse kernel. Probes replay
the same tape, including the joint batch/time bias cotangent. They do not rerun
the physical model or update parameters, optimizer moments, or random keys.
"""
import jax
import jax.numpy as jnp
import numpy as np

from src.models.neuralgcm_residual.trajectory_training import TrajectoryTrainer


def finite_tree(tree, label):
    if any(not np.isfinite(x).all() for x in jax.device_get(jax.tree_util.tree_leaves(tree))):
        raise FloatingPointError('Nonfinite '+label)


def tree_norm(tree):
    # Host float64 avoids overflowing a float32 diagnostic sum of squares.
    return float(np.sqrt(sum(np.sum(np.asarray(x,np.float64)**2)
                            for x in jax.device_get(jax.tree_util.tree_leaves(tree)))))


class AdaptiveTrainer(TrajectoryTrainer):
    def __init__(self, runtime, fixed_loss, grouped_loss, **kwargs):
        super().__init__(runtime,fixed_loss,**kwargs)
        self.grouped = grouped_loss
        self.group_values = jax.jit(grouped_loss.components)
        self.weighted_grad = jax.jit(jax.value_and_grad(grouped_loss.weighted,has_aux=True))

    def reverse(self, params, cotangents, capture=None):
        capture = self.captured if capture is None else capture
        gradient = jax.tree_util.tree_map(jnp.zeros_like,params)
        for i,tape in enumerate(capture['tapes']):
            dh = jax.tree_util.tree_map(jnp.zeros_like,tape[-1][0])
            for j in reversed(range(len(tape))):
                memory,key,record = tape[j]
                out = jax.tree_util.tree_map(lambda x:x[i,j],cotangents)
                dp,dh = self.backward(params,memory,key,record,out,dh)
                gradient = jax.tree_util.tree_map(jnp.add,gradient,dp)
        finite_tree(gradient,'parameter gradient')
        return gradient

    def training_batch(self, params, key, origins, coefficients, *, probe_coefficients=None):
        coefficients = np.asarray(coefficients,np.float32)
        if coefficients.shape != (len(self.grouped.names),) or not np.isfinite(coefficients).all() or np.any(coefficients < 0):
            raise ValueError('Invalid explicit loss coefficients')
        # Capture with the fixed scoring path. It has no parameter backward and
        # gives physical errors plus a complete, reusable recurrent tape.
        fixed, physical, next_key, _ = self.batch(params,key,origins,gradients=False,capture=True)
        cap = self.captured
        finite_tree(cap['prediction'],'forward prediction')
        finite_tree(cap['target'],'target')
        (value,detail),cotangent = self.weighted_grad(cap['prediction'],cap['target'],jnp.asarray(coefficients))
        if not np.isfinite(float(value)):
            raise FloatingPointError('Nonfinite weighted loss')
        finite_tree(cotangent,'loss cotangent')
        gradient = self.reverse(params,cotangent)
        norms = None
        if probe_coefficients is not None:
            base = np.asarray(probe_coefficients,np.float32)
            if base.shape != coefficients.shape or not np.isfinite(base).all() or np.any(base < 0):
                raise ValueError('Invalid probe coefficients')
            norms = np.zeros(len(base),np.float64)
            for i,c in enumerate(base):
                if c == 0:
                    continue
                selector = jnp.zeros(len(base),jnp.float32).at[i].set(c)
                (_, _),cot = self.weighted_grad(cap['prediction'],cap['target'],selector)
                finite_tree(cot,'probe loss cotangent '+self.grouped.names[i])
                norms[i] = tree_norm(self.reverse(params,cot))
        return float(value), dict(groups=np.asarray(detail['groups']), fixed_loss=fixed,
                                   physical=physical, probe_norms=norms), next_key, gradient
