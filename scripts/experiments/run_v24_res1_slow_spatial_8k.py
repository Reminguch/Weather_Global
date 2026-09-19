"""Run the shared-start di16/di32 spatial-LR sweep using the batch-4 driver."""
import hashlib
import json
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "artifacts/checkpoints/v24_Ilya/res1_batch4_slow_spatial_20260916_8k/source"))
import driver_8k as driver

driver.ROOT = Path(__file__).resolve().parents[2]
driver.EXPERIMENT = driver.ROOT / 'artifacts/checkpoints/v24_Ilya/res1_batch4_slow_spatial_20260916_8k'

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ('train', 'eval', 'summarize'):
        raise SystemExit('Expected train, eval, or summarize')
    manifest = json.loads((driver.EXPERIMENT / 'manifest.json').read_text())
    for entry in manifest['entries']:
        start = driver.Path(entry['init_from'])
        if hashlib.sha256(start.read_bytes()).hexdigest() != entry['init_sha256']:
            raise RuntimeError(f'Start checkpoint changed: {start}')
    driver.main()
