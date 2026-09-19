"""Run the shared-start di16/di32 spatial-LR sweep using the batch-4 driver."""
import hashlib
import json
import sys

import run_v24_res1_batch4_lr as driver

driver.EXPERIMENT = driver.ROOT / 'artifacts/checkpoints/v24_Ilya/res1_batch4_slow_spatial_20260916'

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ('train', 'eval', 'summarize'):
        raise SystemExit('Expected train, eval, or summarize')
    manifest = json.loads((driver.EXPERIMENT / 'manifest.json').read_text())
    for entry in manifest['entries']:
        start = driver.Path(entry['init_from'])
        if hashlib.sha256(start.read_bytes()).hexdigest() != entry['init_sha256']:
            raise RuntimeError(f'Start checkpoint changed: {start}')
    driver.main()
