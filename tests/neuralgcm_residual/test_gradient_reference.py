import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('detailed', Path(__file__).resolve().parents[2] /
                                             'scripts/training/smoke_neuralgcm_detailed.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_fp32_cancellation_roundoff_has_small_norm_error():
    a = {'head': np.array([1000., 1.0001], np.float32)}
    b = {'head': np.array([1000., 1.], np.float32)}
    result = runner.compare_gradient_reference(a, b)
    assert result['relative_l2'] < 1e-6
    assert result['cosine'] > .99999


@pytest.mark.parametrize('bad', [
    {'head': np.array([999., 1.])},
    {'head': np.array([-1000., -1.])},
    {'head': np.array([1000., np.nan])},
    {'head': np.array([1000.])},
    {'other': np.array([1000., 1.])},
])
def test_rejects_gradient_errors(bad):
    with pytest.raises(AssertionError):
        runner.compare_gradient_reference({'head': np.array([1000., 1.])}, bad)


def test_small_layer_cannot_hide_behind_large_layer():
    a = {'large': np.array([1e8]), 'small': np.array([.01])}
    b = {'large': np.array([1e8]), 'small': np.array([.0101])}
    with pytest.raises(AssertionError): runner.compare_gradient_reference(a, b)
