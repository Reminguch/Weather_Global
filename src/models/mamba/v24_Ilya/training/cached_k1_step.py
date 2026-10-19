"""Masked native-batch k1 training with the maintained host-tape BPTT.

The weather-step model and its parameter names are unchanged. Only reduction
over padded lanes is added; time remains serial and reverse-time VJPs retain
temporal credit. This module neither builds a baseline nor changes JAX runtime
flags. Inputs are physical cached fields and targets are physical residuals.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from graphcast import xarray_jax

from .cached_stepwise import (
    CachedResidualTransforms,
    CachedStepwiseTrainer,
    make_cached_stepwise_train_step,
)
from .endpoint_step import cast_state_boundary_fp32


def _mask_dataset(dataset, valid_mask):
    """Sanitize padding before the model, including non-leading batch axes."""
    def mask_variable(variable):
        if "batch" not in variable.dims:
            return variable
        values = xarray_jax.unwrap_data(variable)
        shape = [1] * values.ndim
        shape[variable.get_axis_num("batch")] = valid_mask.shape[0]
        masked = jnp.where(valid_mask.reshape(shape), values, jnp.zeros_like(values))
        return variable.copy(data=xarray_jax.wrap(masked))

    return dataset.map(mask_variable)


def _mask_state(state, valid_mask):
    # Native Mamba SSM and convolution carries both begin with batch.
    return jax.tree_util.tree_map(
        lambda value: jnp.where(
            valid_mask.reshape((valid_mask.shape[0],) + (1,) * (value.ndim - 1)),
            value,
            jnp.zeros_like(value),
        ), state,
    )


class _MaskedTransform:
    """Pass a dynamic lane mask through the existing one-step kernel API."""

    def __init__(self, transform, *, supervised):
        self.transform = transform
        self.supervised = supervised

    def apply(self, params, state, key, packed_inputs, targets, forcings):
        inputs, valid_mask = packed_inputs
        output, next_state = self.transform.apply(
            params, _mask_state(state, valid_mask), key,
            _mask_dataset(inputs, valid_mask),
            _mask_dataset(targets, valid_mask),
            _mask_dataset(forcings, valid_mask),
        )
        next_state = _mask_state(next_state, valid_mask)
        if not self.supervised:
            return _mask_dataset(output, valid_mask), next_state
        (lane_loss, diagnostics), predictions = output
        if lane_loss.dims != ("batch",):
            raise ValueError("Residual loss must retain exactly the batch dimension")
        values = xarray_jax.unwrap_data(lane_loss).astype(jnp.float32)
        # Reduce in FP32 as maintained scalarization does after BF16 casting.
        loss = jnp.sum(jnp.where(valid_mask, values, 0.0)) / jnp.sum(valid_mask)
        output = ((xr.DataArray(xarray_jax.wrap(loss)), diagnostics),
                  _mask_dataset(predictions, valid_mask))
        return output, next_state


class CachedK1Trainer:
    """Batched cached-k1 adapter with a bounded host recurrent-state tape.

``keys``, ``inputs``, ``targets`` and ``forcings`` are sequences of BPTT length;
each key is one native-batch model key, not a per-lane array of keys. Each
dataset has native ``batch`` dimension B. Every recurrent-state leaf starts
with B. ``valid_mask`` is B booleans (or zero/one values), with at least one
valid lane. Padding contributes neither loss nor gradients and returns zero
state/predictions. The caller carries returned state across chunks of each
segment and resets it when lanes are assigned new segments.

The optimizer update is applied once to the mean over valid lanes and time.
Parameters and optimizer state have no batch dimension and remain compatible
with the native live-k20 checkpoint format. Physical predictions and reported
training loss retain the maintained precision policy; training loss is not a
replacement for the separate exact original GraphCast evaluation metric.
"""

    def __init__(self, *, config, optimizer, transforms: CachedResidualTransforms):
        if config.loss_mode != "all_steps":
            raise ValueError("Cached k1 requires loss_mode='all_steps'")
        self.config = config
        self.optimizer = optimizer
        self.transforms = transforms
        masked = CachedResidualTransforms(
            _MaskedTransform(transforms.residual_predict, supervised=False),
            _MaskedTransform(transforms.residual_loss_and_predictions, supervised=True),
        )
        self._trainer = CachedStepwiseTrainer(
            config=config, optimizer=optimizer, transforms=masked,
        )

    def initialize(self, key, inputs, targets, forcings):
        """Initialize only the native residual model at the requested batch size."""
        params, state = self.transforms.residual_loss_and_predictions.init(
            key, inputs, targets, forcings,
        )
        return params, cast_state_boundary_fp32(state)

    def _validate_batch(self, state, inputs, targets, forcings, valid_mask):
        mask = np.asarray(jax.device_get(valid_mask))
        if mask.ndim != 1 or not mask.size:
            raise ValueError("valid_mask must be a nonempty vector")
        if not np.all((mask == 0) | (mask == 1)) or not np.any(mask):
            raise ValueError("valid_mask must contain zero/one values and a valid lane")
        batch = len(mask)
        for name, values in (("inputs", inputs), ("targets", targets),
                             ("forcings", forcings)):
            for dataset in values:
                if not isinstance(dataset, xr.Dataset):
                    raise TypeError(f"{name} must contain xarray Dataset values")
                if dataset.sizes.get("batch") != batch:
                    raise ValueError(f"{name} batch dimension must match valid_mask ({batch})")
        for leaf in jax.tree_util.tree_leaves(state):
            if leaf.ndim == 0 or leaf.shape[0] != batch:
                raise ValueError("Every recurrent state leaf must start with batch")
        return jnp.asarray(mask, dtype=jnp.bool_)

    def _pack(self, state, keys, inputs, targets, forcings, valid_mask):
        steps = int(self.config.bptt_steps)
        for name, values in (("keys", keys), ("inputs", inputs),
                             ("targets", targets), ("forcings", forcings)):
            if len(values) != steps:
                raise ValueError(f"Expected {steps} {name}, got {len(values)}")
        mask = self._validate_batch(state, inputs, targets, forcings, valid_mask)
        return tuple((dataset, mask) for dataset in inputs)

    def evaluate_step(self, params, state, key, inputs, target, forcing, valid_mask):
        """Return one-step loss, next state, and prediction without a state tape.

        This uses the same compiled masked kernel as chunk evaluation, including
        the native precision boundaries. It is useful for checkpoint roundtrips
        and streaming diagnostics.
        """
        mask = self._validate_batch(state, (inputs,), (target,), (forcing,), valid_mask)
        return self._trainer.evaluate_step(
            params, state, key, (inputs, mask), target, forcing,
        )

    def evaluate(
        self, params, state, keys, inputs, targets, forcings, valid_mask, *,
        retain_predictions=False, prediction_callback=None,
    ):
        """Return loss, next state, per-step losses, and optional predictions."""
        packed = self._pack(state, keys, inputs, targets, forcings, valid_mask)
        return self._trainer.evaluate(
            params, state, keys, packed, targets, forcings,
            retain_predictions=retain_predictions, prediction_callback=prediction_callback,
        )

    def loss_and_grad(
        self, params, state, keys, inputs, targets, forcings, valid_mask, *,
        retain_predictions=False, prediction_callback=None,
    ):
        """Return evaluate's four values followed by FP32 parameter gradients."""
        packed = self._pack(state, keys, inputs, targets, forcings, valid_mask)
        return self._trainer.loss_and_grad(
            params, state, keys, packed, targets, forcings,
            retain_predictions=retain_predictions, prediction_callback=prediction_callback,
        )

    def __call__(self, params, state, opt_state, keys, inputs, targets, forcings, valid_mask):
        """Return the six values of the maintained training-step contract."""
        packed = self._pack(state, keys, inputs, targets, forcings, valid_mask)
        return self._trainer(params, state, opt_state, keys, packed, targets, forcings)


def make_cached_k1_step(*, config, model_config, task_config, stats, optimizer):
    """Build native residual transforms without initializing any model."""
    native = make_cached_stepwise_train_step(
        config=config, model_config=model_config, task_config=task_config,
        stats=stats, optimizer=optimizer,
    )
    return CachedK1Trainer(config=config, optimizer=optimizer, transforms=native.transforms)
