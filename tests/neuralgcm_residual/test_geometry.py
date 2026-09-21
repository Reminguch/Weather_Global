import numpy as np
import pytest

from src.models.neuralgcm_residual.config import Architecture
from src.models.neuralgcm_residual.model import geometry_manifest


@pytest.mark.parametrize("nlat", [64, 128])
def test_common_mesh_covers_both_native_gaussian_grids(nlat):
    latitude = np.arcsin(np.polynomial.legendre.leggauss(nlat)[0])
    longitude = np.arange(2 * nlat) * np.pi / nlat
    geometry = geometry_manifest(Architecture(), latitude, longitude, 193)
    assert geometry["mesh_nodes"] == 2562
    assert geometry["grid_nodes"] == 2 * nlat**2
    assert geometry["edges"]["grid2mesh"]["receiver_degree_min"] > 0
    assert geometry["edges"]["mesh2grid"]["receiver_degree_min"] == 3
    assert geometry["edges"]["mesh2grid"]["receiver_degree_max"] == 3
