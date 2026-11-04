"""GC sum-over-fields MSE with Gaussian area and the 29-level pressure weights."""
import jax.numpy as jnp
import numpy as np
from .config import P
from src.models.neuralgcm_residual.loss import gaussian_weights


def weights(latitude):
    lat = len(latitude)*gaussian_weights(latitude)
    p = np.asarray(P,dtype=np.float32)
    return jnp.asarray(lat),jnp.asarray(p/p.mean()),jnp.asarray([1,1,1,1,1,.05,.05],dtype=jnp.float32)


def contributions(prediction, target, latitude):
    """Return field/level contributions, whose sum is the single-step loss."""
    lat,p,alpha = weights(latitude)
    mse = jnp.mean(jnp.square(prediction-target)*lat,axis=(-2,-1))
    return mse*alpha[:,None]*p[None,:]/len(P)


def weighted_mse(prediction,target,latitude):
    return jnp.sum(contributions(prediction,target,latitude))


def target_norm(truth,baseline,stats):
    return (truth[:,8:]-baseline[:,8:])/stats['diff_std'][:,8:,None,None]


def correct(baseline,output,stats):
    # Copy upper levels verbatim, even signed zeros. Never add zeros there.
    import jax.numpy as jnp
    delta = output*stats['diff_std'][:,8:,None,None]
    # Preserve signed zeros at zero initialization. This function is the
    # physical output adapter; training differentiates the normalized loss.
    lower = jnp.where(delta==0,baseline[:,8:],baseline[:,8:]+delta)
    return jnp.concatenate((baseline[:,:8],lower),axis=1)
