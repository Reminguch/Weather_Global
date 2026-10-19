#!/usr/bin/env python3
"""Bundle the pinned CPU runtime on a responsive host, then stream it to vis1.

Keep the interpreter, standard library and native dependencies together. Copying
only a venv preserves its dependency on the original Conda interpreter. GPU
plugins, NVIDIA libraries, tests and bytecode are unnecessary for ERA5 staging.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--venv', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    interpreter = (args.venv / 'bin/python').resolve()
    base = interpreter.parent.parent
    stdlib = base / 'lib/python3.11'
    site = args.venv / 'lib/python3.11/site-packages'
    manifest = json.loads((args.data_root / 'download_manifest.json').read_text())
    libraries = set()
    # Include the actual dependencies of every CPython extension, recursively.
    pending = [interpreter, *sorted((stdlib / 'lib-dynload').glob('*.so'))]
    inspected = set()
    while pending:
        path = pending.pop()
        if path in inspected:
            continue
        inspected.add(path)
        result = subprocess.run(['ldd', str(path)], capture_output=True, text=True, check=True)
        for name in re.findall(r'=> (/\S+)', result.stdout):
            dep = Path(name)
            if str(dep).startswith(str(base) + '/') and dep not in libraries:
                libraries.add(dep)
                pending.append(dep)

    excluded = {'__pycache__', 'test', 'tests', 'nvidia', 'jax_plugins', 'jax_cuda12_plugin', 'jax_cuda12_pjrt'}

    def keep(info):
        if excluded.intersection(Path(info.name).parts) or info.name.endswith(('.pyc', '.pyo')):
            return None
        return info

    print('Bundling CPU interpreter, dependencies and frozen source', flush=True)
    with tarfile.open(args.output, 'w', dereference=True) as archive:
        archive.add(interpreter, arcname='bin/python3.11')
        for dep in sorted(libraries):
            archive.add(dep, arcname='lib/' + dep.name)
        for path in sorted(stdlib.iterdir()):
            if path.name != 'site-packages':
                archive.add(path, arcname='lib/python3.11/' + path.name, filter=keep)
        for path in sorted(site.iterdir()):
            if path.name not in excluded and not path.name.startswith(('nvidia_', 'jax_cuda12_')):
                archive.add(path, arcname='lib/python3.11/site-packages/' + path.name, filter=keep)
        source = Path(manifest['source_root'])
        for rel, expected in manifest['source_files'].items():
            path = source / rel
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('Frozen source changed: ' + rel)
            archive.add(path, arcname='source/' + rel)
        for rid, checkpoint in manifest['checkpoints'].items():
            path = Path(checkpoint['path'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != checkpoint['sha256']:
                raise ValueError('Checkpoint changed: ' + rid)
            archive.add(path, arcname='checkpoints/' + path.name)
        runner = Path(__file__).with_name('run_neuralgcm_download_local.py')
        archive.add(runner, arcname='run_download.py')
    print(json.dumps({'bundle': str(args.output), 'bytes': args.output.stat().st_size,
                      'source_id': manifest['source_id'], 'shared_libraries': len(libraries)}), flush=True)


if __name__ == '__main__':
    main()
