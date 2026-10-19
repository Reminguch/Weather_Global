import numpy as np
import jax.numpy as jnp

from src.models.mamba.v24_Ilya.spherical_observer import diagnostic_kernel


def test_spherical_projection_separates_unresolved_nyquist_and_obeys_grid_closure():
    lat = np.linspace(-90, 90, 9)
    lon = np.arange(16) * 22.5
    _, calculate = diagnostic_kernel(lat, lon, lmax=5)
    smooth = np.sin(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
    nyquist = .1 * (-1.) ** np.arange(len(lon))[None, :] * np.cos(np.deg2rad(lat))[:, None]
    fields = np.stack([smooth, .8 * smooth, .8 * smooth + nyquist])[:, None]
    d = {k: np.asarray(v) for k, v in calculate(jnp.asarray(fields)).items()}
    np.testing.assert_allclose(d['baseline_error_power'], d['full_error_power'], atol=1e-7)
    assert d['full_unresolved_grid_mse'][0] > .004
    assert d['baseline_unresolved_grid_mse'][0] < 1e-10
    for branch in ('baseline', 'full'):
        np.testing.assert_allclose(d[f'{branch}_grid_mse'],
                                   d[f'{branch}_projected_grid_mse'] + d[f'{branch}_unresolved_grid_mse'],
                                   rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(d['full_error_power'],
                               d['truth_power'] + d['full_power'] - 2 * d['full_cross_power'], atol=1e-7)
