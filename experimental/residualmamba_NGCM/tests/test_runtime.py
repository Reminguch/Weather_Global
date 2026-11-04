"""Guard the execution policy that makes fresh-process GPU resume reproducible."""
import os
import sys
from types import SimpleNamespace
import pytest
from residualmamba_ngcm import runtime


def test_flags_accept_supported_spellings_and_reject_conflicts():
    assert runtime.flag_values('--a --b 0 --c=true --a=true')=={
        '--a':'true','--b':'0','--c':'true'}
    with pytest.raises(ValueError,match='Conflicting'):
        runtime.flag_values('--a=true --a=false')


def test_defaults_are_early_idempotent_and_preserve_other_flags(monkeypatch):
    monkeypatch.delenv('NGCM_GPU_EXECUTION_POLICY',raising=False)
    monkeypatch.setenv('XLA_FLAGS','--xla_force_host_platform_device_count=1')
    monkeypatch.setitem(sys.modules,'jax._src.xla_bridge',
                        SimpleNamespace(backends_are_initialized=lambda:False))
    runtime.configure()
    first=os.environ['XLA_FLAGS']
    runtime.configure()
    assert os.environ['XLA_FLAGS']==first
    assert runtime.flag_values(first)['--xla_force_host_platform_device_count']=='1'
    assert runtime.require_production_policy()['flags']==runtime.REQUIRED_FLAGS


def test_conflicting_policy_and_late_backend_are_rejected(monkeypatch):
    monkeypatch.delenv('NGCM_GPU_EXECUTION_POLICY',raising=False)
    monkeypatch.setenv('XLA_FLAGS','--xla_gpu_deterministic_ops=false')
    with pytest.raises(RuntimeError,match='requires'):
        runtime.configure()
    monkeypatch.setenv('XLA_FLAGS','')
    monkeypatch.setitem(sys.modules,'jax._src.xla_bridge',
                        SimpleNamespace(backends_are_initialized=lambda:True))
    with pytest.raises(RuntimeError,match='before initializing JAX'):
        runtime.configure()


def test_external_flags_are_diagnostic_only(monkeypatch):
    monkeypatch.setenv('NGCM_GPU_EXECUTION_POLICY','diagnostic_external')
    monkeypatch.setenv('XLA_FLAGS','--xla_gpu_autotune_level=4')
    runtime.configure()
    assert os.environ['XLA_FLAGS']=='--xla_gpu_autotune_level=4'
    with pytest.raises(RuntimeError,match='Production requires'):
        runtime.require_production_policy()
    monkeypatch.setenv('NGCM_GPU_EXECUTION_POLICY','unknown')
    with pytest.raises(ValueError,match='Unknown'):
        runtime.configure()
