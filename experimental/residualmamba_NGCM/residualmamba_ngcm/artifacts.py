"""Immutable scientific artifacts and authenticated atomic training checkpoints."""
import io
import pickle
from pathlib import Path
import numpy as np
from . import ROOT
from src.models.neuralgcm_residual.io import atomic_bytes, digest, sha256, read_json, write_json, versions as backbone_versions


def versions():
    import importlib.metadata
    result=backbone_versions()
    for name in ('pandas','scipy','chex','jraph','trimesh','rtree','dm-tree',
                 'jax-cuda12-plugin','jax-cuda12-pjrt','gcsfs','zarr'):
        try:result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:result[name]=None
    return result


def source_identity():
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += [ROOT/'experimental/residualmamba_NGCM/README.md']
    for root in ('src/models/mamba/v24_Ilya','src/models/graphcast/training/core',
                 'src/models/neuralgcm_residual','third_party/graphcast/graphcast'):
        paths += list((ROOT/root).rglob('*.py'))
    return digest({str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(paths))})


def save_arrays(path, arrays, manifest):
    path = Path(path)
    if path.exists() or path.with_suffix('.json').exists():
        raise FileExistsError(f'Immutable artifact exists: {path}')
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    atomic_bytes(path, buffer.getvalue())
    write_json(path.with_suffix('.json'), {**manifest,'sha256':sha256(path)},immutable=True)


def load_arrays(path):
    path = Path(path)
    manifest = read_json(path.with_suffix('.json'))
    if sha256(path) != manifest['sha256']:
        raise ValueError(f'Corrupt artifact {path}')
    with np.load(path,allow_pickle=False) as data:
        arrays = {k:data[k] for k in data.files}
    return arrays, manifest


def save_checkpoint(path, value):
    import jax
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    atomic_bytes(path,pickle.dumps(jax.device_get(value),protocol=5))
    write_json(path.with_suffix('.json'), {'sha256':sha256(path),'step':value['step'],
               'identity':value['identity']},immutable=True)


def load_checkpoint(path, identity=None):
    path = Path(path)
    meta = read_json(path.with_suffix('.json'))
    if sha256(path) != meta['sha256'] or (identity is not None and meta['identity'] != identity):
        raise ValueError('Checkpoint integrity or run identity mismatch')
    # Read only locally produced, SHA256-verified checkpoints.
    value = pickle.loads(path.read_bytes())
    if value['identity'] != meta['identity'] or value['step'] != meta['step']:
        raise ValueError('Checkpoint receipt differs from payload')
    return value
