#!/usr/bin/env python3
"""Report local NeuralGCM prerequisites without loading a checkpoint or running JAX."""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resources', type=Path)
    parser.add_argument('--statistics', type=Path, help='Weather loss statistics, separate from native normalization')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    blockers = []
    versions = {}
    # These are the source run's numerical dependencies, not a proposed upgrade
    # of the existing GraphCast environment.
    pins = {}
    for line in (ROOT / 'configs/experiments/neuralgcm_residual/requirements-cuda12.lock.txt').read_text().splitlines():
        if '==' in line and not line.startswith('#'):
            name, version = line.split('==')
            pins[name] = version
    for name in ('jax', 'jaxlib', 'jax-cuda12-plugin', 'jax-cuda12-pjrt',
                 'dinosaur', 'dm-haiku', 'optax', 'numpy', 'gin-config'):
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        versions[name] = dict(installed=installed, expected=pins[name])
        if installed != pins[name]:
            blockers.append(f'{name}: expected {pins[name]}, found {installed}')
    if not (ROOT / '.venv_neuralgcm/bin/python').is_file():
        blockers.append('Local isolated .venv_neuralgcm has not been created')

    def read(path):
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError) as error:
            blockers.append(f'{path}: {error}')
            return None

    resources = read(args.resources) if args.resources else None
    if args.resources is None:
        blockers.append('Provide --resources with checkpoint, checkpoint_sha256, prepared, statistics')
    if resources is not None:
        if not isinstance(resources, dict):
            blockers.append('Resource plan must be a JSON object')
            resources = {}
        for key in ('checkpoint', 'prepared', 'statistics'):
            value = resources.get(key)
            if not isinstance(value, str) or not Path(value).is_absolute():
                blockers.append(f'Resource {key} must be an absolute path')
                continue
            path = Path(value)
            try:
                if key == 'prepared':
                    if not all((path / item).is_file() for item in ('manifest.json', 'READY.json')):
                        blockers.append(f'Prepared store lacks manifest.json or READY.json: {path}')
                elif not path.is_file():
                    blockers.append(f'Missing resource {key}: {path}')
                elif key == 'checkpoint' and sha256(path) != resources.get('checkpoint_sha256'):
                    blockers.append('Backbone checkpoint SHA256 does not match resource plan')
                elif key == 'statistics':
                    stats = read(path)
                    if stats:
                        climate = path.parent / stats.get('climatology_file', '')
                        if not climate.is_file() or sha256(climate) != stats.get('climatology_sha256'):
                            blockers.append('Native normalization climatology is missing or has the wrong hash')
            except OSError as error:
                blockers.append(f'{key}: {error}')
    if args.statistics:
        stats = read(args.statistics)
        if stats and (stats.get('split') != 'train' or stats.get('lag_hours') != 24):
            blockers.append('Weather loss statistics must use training-only 24-hour differences')
    report = dict(source_commit='95b1d2d5a703e29586b716e936dc32eb99adbe86',
                  python=sys.executable, versions=versions, blockers=blockers,
                  prerequisites_present=not blockers,
                  gpu_numerical_and_resume_gates='Must run locally; archived pass records are not authorization gates')
    body = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body)
    print(body, end='')
    return 1 if blockers else 0


if __name__ == '__main__':
    raise SystemExit(main())
