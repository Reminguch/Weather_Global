"""Versioned deterministic GPU execution, configured before JAX initializes."""
from __future__ import annotations
import os
import shlex
import sys

POLICY = 'deterministic_gpu_v1'
REQUIRED_FLAGS = {
    '--xla_gpu_enable_scatter_determinism_expander': 'true',
    '--xla_gpu_exclude_nondeterministic_ops': 'true',
    '--xla_gpu_deterministic_ops': 'true',
    '--xla_gpu_autotune_level': '0',
}


def flag_values(text):
    tokens=shlex.split(text)
    values={}
    i=0
    while i<len(tokens):
        token=tokens[i]
        if token.startswith('--'):
            if '=' in token:
                name,value=token.split('=',1)
            elif i+1<len(tokens) and not tokens[i+1].startswith('--'):
                name,value=token,tokens[i+1];i+=1
            else:
                name,value=token,'true'
            if name in values and values[name]!=value:
                raise ValueError(f'Conflicting XLA option: {name}')
            values[name]=value
        i+=1
    return values


def configure():
    """Set defaults early; a diagnostic subprocess can explicitly keep external flags."""
    mode=os.environ.get('NGCM_GPU_EXECUTION_POLICY',POLICY)
    if mode=='diagnostic_external':return
    if mode!=POLICY:raise ValueError(f'Unknown NeuralGCM execution policy: {mode}')
    existing=os.environ.get('XLA_FLAGS','')
    flags=flag_values(existing)
    for name,expected in REQUIRED_FLAGS.items():
        if name in flags and flags[name]!=expected:
            raise RuntimeError(f'{POLICY} requires {name}={expected}; received {flags[name]}')
    missing={k:v for k,v in REQUIRED_FLAGS.items() if k not in flags}
    bridge=sys.modules.get('jax._src.xla_bridge')
    if missing and bridge is not None and bridge.backends_are_initialized():
        raise RuntimeError('Import residualmamba_ngcm before initializing JAX, so deterministic GPU options take effect')
    additions=' '.join(f'{k}={v}' for k,v in missing.items())
    os.environ['XLA_FLAGS']=' '.join(v for v in (existing,additions) if v)


def identity():
    flags=flag_values(os.environ.get('XLA_FLAGS',''))
    return {'name':os.environ.get('NGCM_GPU_EXECUTION_POLICY',POLICY),
            'flags':{name:flags.get(name) for name in REQUIRED_FLAGS},
            'default_matmul_precision':os.environ.get('JAX_DEFAULT_MATMUL_PRECISION','default')}


def require_production_policy():
    policy=identity()
    if policy['name']!=POLICY or policy['flags']!=REQUIRED_FLAGS:
        raise RuntimeError('Production requires deterministic_gpu_v1; diagnostic execution cannot start a production run')
    return policy
