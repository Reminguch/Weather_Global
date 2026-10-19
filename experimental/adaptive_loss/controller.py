"""Host-side, checkpointable EMA gradient balancer. No model autodiff here.

Weights are selected AFTER a successful optimizer update, for the NEXT batch.
Gradient norms describe parameter gradients, not fractions of an Adam update.
"""
from dataclasses import asdict, dataclass
import copy

import numpy as np


@dataclass(frozen=True)
class BalanceConfig:
    window: int = 100
    interval: int = 20
    slow_multiple: int = 4
    alpha: float = .5
    minimum: float = .25
    maximum: float = 4.
    max_change: float = .2
    gradient_floor: float = 1e-12

    def __post_init__(self):
        if any(type(x) is not int or x < 1 for x in (self.window, self.interval, self.slow_multiple)):
            raise ValueError('Window, interval and slow_multiple must be positive integers')
        if not (np.isfinite(self.alpha) and 0 <= self.alpha <= 2):
            raise ValueError('Require finite alpha in [0,2]')
        if not (0 < self.minimum <= 1 <= self.maximum < np.inf):
            raise ValueError('Weight bounds must enclose one')
        if not (0 < self.max_change < 1 and 0 < self.gradient_floor < np.inf):
            raise ValueError('Invalid slew limit or gradient floor')


def bounded_mean_one(raw, lower, upper):
    """Multiplicative projection with bounds AND mean=1, including slew limits."""
    raw, lower, upper = [np.asarray(v, np.float64) for v in (raw, lower, upper)]
    if raw.size == 0:
        return raw.copy()
    if (not np.isfinite(raw).all() or np.any(raw <= 0) or
            np.any(lower > upper) or lower.sum() > raw.size+1e-10 or upper.sum() < raw.size-1e-10):
        raise ValueError('Infeasible bounded normalization')
    # Log space prevents reciprocal tiny gradients overflowing the controller.
    lograw = np.log(raw)
    lo, hi = -1000., 1000.
    for _ in range(100):
        mid = (lo+hi)/2
        value = np.exp(np.clip(lograw+mid, np.log(lower), np.log(upper)))
        if value.sum() < raw.size:
            lo = mid
        else:
            hi = mid
    return np.exp(np.clip(lograw+(lo+hi)/2, np.log(lower), np.log(upper)))


class WindowBalancer:
    schema = 'ema_parameter_gradient_balance_v1'

    def __init__(self, names, config=BalanceConfig(), *, excluded=()):
        self.names = tuple(names)
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError('Group names must be nonempty and unique')
        if not set(excluded) <= set(self.names):
            raise ValueError('Unknown excluded group')
        self.config = config
        self.excluded = tuple(excluded)
        self.weights = np.ones(len(names), np.float64)
        self.fast = self.slow = self.grad_ema = None
        self.update = 0
        self.last_probe = 0
        self.adjustments = 0

    def needs_probe(self):
        return (self.update+1) % self.config.interval == 0

    def observe(self, values, gradient_norms=None):
        """Commit finite pre-update measurements only after optimizer success.

        'values' and norms refer to the fixed, calibrated group objectives,
        without the changing controller weights. Nonfinite input never mutates
        state. Structurally disconnected or currently zero-gradient groups are
        held at one and cannot absorb the budget of trainable groups.
        """
        values = np.asarray(values, np.float64)
        if values.shape != self.weights.shape or not np.isfinite(values).all() or np.any(values < 0):
            raise FloatingPointError('Invalid group losses; controller state unchanged')
        due = self.needs_probe()
        if due != (gradient_norms is not None):
            raise ValueError('Gradient probe must match the configured update cadence')
        norms = None if gradient_norms is None else np.asarray(gradient_norms, np.float64)
        if norms is not None and (norms.shape != values.shape or not np.isfinite(norms).all() or np.any(norms < 0)):
            raise FloatingPointError('Invalid group gradients; controller state unchanged')
        c = self.config
        decay = np.exp(-1./c.window)
        slow_decay = np.exp(-1./(c.window*c.slow_multiple))
        fast = values.copy() if self.fast is None else decay*self.fast+(1-decay)*values
        slow = values.copy() if self.slow is None else slow_decay*self.slow+(1-slow_decay)*values
        weights = self.weights.copy()
        grad_ema = self.grad_ema
        active = np.zeros(len(values), bool)
        if due:
            gd = np.exp(-c.interval/c.window)
            grad_ema = norms.copy() if grad_ema is None else gd*grad_ema+(1-gd)*norms
            active = (norms > c.gradient_floor) & (grad_ema > c.gradient_floor)
            active &= np.array([n not in self.excluded for n in self.names])
            # A group's scalar loss can be positive despite having NO gradient.
            weights[~active] = 1.
            if active.any():
                ratio = (fast[active]+1e-12)/(slow[active]+1e-12)
                ratio = np.clip(ratio/np.mean(ratio), .1, 10.)
                raw = ratio**c.alpha / np.maximum(grad_ema[active], c.gradient_floor)
                # If the active set changed, mean-one and the old slew box can
                # be incompatible. Start that set from feasible unit weights.
                old = weights[active]
                if not np.isclose(old.mean(), 1., rtol=0, atol=1e-10):
                    old = np.ones_like(old)
                lower = np.maximum(c.minimum, old*(1-c.max_change))
                upper = np.minimum(c.maximum, old*(1+c.max_change))
                weights[active] = bounded_mean_one(raw, lower, upper)
        self.fast, self.slow, self.grad_ema = fast, slow, grad_ema
        self.weights = weights
        self.update += 1
        if due:
            self.last_probe = self.update
            self.adjustments += 1
        return dict(update=self.update, adjusted=due, active=active.tolist(), weights=weights.tolist())

    def state_dict(self):
        return dict(schema=self.schema, names=list(self.names), config=asdict(self.config),
                    excluded=list(self.excluded), update=self.update, last_probe=self.last_probe,
                    adjustments=self.adjustments, **{k: None if getattr(self,k) is None else getattr(self,k).tolist()
                    for k in ('weights','fast','slow','grad_ema')})

    @classmethod
    def from_state(cls, state):
        state = copy.deepcopy(state)
        if state.get('schema') != cls.schema:
            raise ValueError('Unknown controller checkpoint schema')
        obj = cls(state['names'], BalanceConfig(**state['config']), excluded=state['excluded'])
        for name in ('weights','fast','slow','grad_ema'):
            value = state[name]
            if value is not None:
                value = np.asarray(value, np.float64)
                if value.shape != obj.weights.shape or not np.isfinite(value).all() or np.any(value < 0):
                    raise ValueError('Invalid saved '+name)
            setattr(obj, name, value)
        if obj.weights is None or np.any(obj.weights < obj.config.minimum-1e-10) or np.any(obj.weights > obj.config.maximum+1e-10):
            raise ValueError('Saved weights outside bounds')
        for name in ('update','last_probe','adjustments'):
            if type(state[name]) is not int or state[name] < 0:
                raise ValueError('Invalid saved clock')
            setattr(obj, name, state[name])
        if obj.last_probe != (obj.update//obj.config.interval)*obj.config.interval or obj.adjustments != obj.update//obj.config.interval:
            raise ValueError('Controller clock mismatch')
        if (obj.fast is None or obj.slow is None) != (obj.update == 0):
            raise ValueError('Controller history missing')
        if (obj.grad_ema is None) != (obj.last_probe == 0):
            raise ValueError('Gradient history missing')
        return obj
