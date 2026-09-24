"""Pin compiler choices as well as runtime operations for fresh-process replay."""
import os
import shlex

from .numerics import configure_environment


def configure_paper_environment():
    configure_environment()
    flags = shlex.split(os.environ.get('XLA_FLAGS', ''))
    name = '--xla_gpu_autotune_level'
    matches = [x for x in flags if x.split('=', 1)[0] == name]
    if any(x != name+'=0' for x in matches):
        raise ValueError('This experiment pins xla_gpu_autotune_level=0 for process replay')
    if not matches:
        flags.append(name+'=0')
    os.environ['XLA_FLAGS'] = ' '.join(flags)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise ValueError('Launch this experiment with PYTHONHASHSEED=0')


def execution_metadata():
    return {name: os.environ.get(name) for name in (
        'XLA_FLAGS', 'CUBLAS_WORKSPACE_CONFIG', 'PYTHONHASHSEED',
        'JAX_DEFAULT_MATMUL_PRECISION', 'JAX_ENABLE_X64')}
