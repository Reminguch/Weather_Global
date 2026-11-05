"""GC sum-over-fields MSE with Gaussian area and the 29-level pressure weights."""
import jax.numpy as jnp
import numpy as np
from .config import P
from src.models.neuralgcm_residual.loss import gaussian_weights


def weights(latitude):
    lat = len(latitude)*gaussian_weights(latitude)
    p = np.asarray(P,dtype=np.float32)
    return jnp.asarray(lat),jnp.asarray(p/p.mean()),jnp.asarray([1,1,1,1,1],dtype=jnp.float32)


def contributions(prediction, target, latitude):
    """Return field/level contributions, whose sum is the single-step loss."""
    if prediction.shape != target.shape or prediction.shape[-4:-2] != (5, len(P)):
        raise ValueError('v2 loss requires matching [5, 29, lon, lat] arrays')
    lat,p,alpha = weights(latitude)
    mse = jnp.mean(jnp.square(prediction-target)*lat,axis=(-2,-1))
    return mse*alpha[:,None]*p[None,:]/len(P)


def weighted_mse(prediction,target,latitude):
    return jnp.sum(contributions(prediction,target,latitude))


def target_norm(truth,baseline,stats):
    return (truth[:5,8:]-baseline[:5,8:])/stats['diff_std'][:5,8:,None,None]


def correct(baseline,output,stats):
    """Scatter only the five scored fields; clouds and upper levels are bitwise copies."""
    if output.shape != (5, len(P), *baseline.shape[-2:]):
        raise ValueError('v2 correction requires [5, 29, lon, lat]')
    delta = output*stats['diff_std'][:5,8:,None,None]
    # Preserve signed zeros at zero initialization as well as in excluded cells.
    lower = jnp.where(delta==0,baseline[:5,8:],baseline[:5,8:]+delta)
    scored = jnp.concatenate((baseline[:5,:8],lower),axis=1)
    return jnp.concatenate((scored,baseline[5:]),axis=0)
