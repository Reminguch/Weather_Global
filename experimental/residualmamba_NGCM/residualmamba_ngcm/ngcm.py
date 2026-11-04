"""Load-only NGCM adapter. No training gradients and no corrected-state feedback."""
import numpy as np
from .config import LEVELS
from .timeline import stack_weather, origin_index
from src.models.neuralgcm_residual.backbone import FrozenBackbone
from src.models.neuralgcm_residual.data import PreparedStore, require_complete_experiment_data


def load(checkpoint, expected_sha256, store_path, production=True):
    if not expected_sha256:
        raise ValueError('An explicit NGCM checkpoint SHA256 is required')
    bb = FrozenBackbone.load(checkpoint,expected_sha256)
    store = PreparedStore(store_path)
    if production:
        require_complete_experiment_data(store)
    grid = bb.model.data_coords.horizontal
    assert tuple(grid.nodal_shape) == (128,64)
    assert tuple(bb.model.model_coords.horizontal.nodal_shape) == (128,64)
    assert bb.internal_steps == 6
    np.testing.assert_array_equal(bb.model.data_coords.vertical.centers, LEVELS)
    for name, coordinate in [('latitude',grid.latitudes),('longitude',grid.longitudes),('levels',LEVELS)]:
        np.testing.assert_allclose(store.manifest['data_grid'][name],coordinate,atol=1e-8,rtol=0)
    return bb, store


def rollout(bb, store, origin, steps):
    inputs, forcing = store.inputs_and_forcing(bb.model,origin)
    initial = state = bb.encode(inputs,forcing)
    for k in range(1,steps+1):
        state = bb.advance_6h(state,forcing)
        bb.assert_time(initial,state,k)
        weather = stack_weather(bb.decode(state,forcing))
        if not np.isfinite(weather).all():
            raise FloatingPointError(f'Nonfinite native forecast at {origin}, lead {6*k}h')
        yield weather, forcing


def chunk_baseline(bb, store, times, a0):
    output = [next(rollout(bb,store,times[a0+i],1))[0] for i in range(3)]
    output.extend(b for b,_ in rollout(bb,store,times[a0+3],21))
    return np.stack(output)
