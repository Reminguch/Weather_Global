"""The fast scatter lowering retains, rather than disables, determinism."""
import pytest
from src.models.neuralgcm_residual.numerics import configure_environment, REQUIRED_FLAGS


def test_required_flags_are_enabled_once(monkeypatch):
    monkeypatch.setenv('XLA_FLAGS', '--xla_gpu_autotune_level=4')
    configure_environment()
    import os
    first = os.environ['XLA_FLAGS']
    configure_environment()
    assert os.environ['XLA_FLAGS'] == first
    for option in REQUIRED_FLAGS: assert first.split().count(option+'=true') == 1
    assert '--xla_gpu_autotune_level=4' in first


@pytest.mark.parametrize('option', REQUIRED_FLAGS)
def test_cannot_silently_disable_required_options(monkeypatch, option):
    monkeypatch.setenv('XLA_FLAGS', option+'=false')
    with pytest.raises(ValueError, match='requires'): configure_environment()
