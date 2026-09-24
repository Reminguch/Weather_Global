#!/usr/bin/env python3
"""Exercise the production CLI, physical validation and interrupted resume."""
import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resource-plan', type=Path, required=True)
    p.add_argument('--statistics', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    # Keep the orchestration process off the GPU; children own each allocation.
    worker_env = dict(os.environ, JAX_PLATFORMS='cuda')
    os.environ['JAX_PLATFORMS'] = 'cpu'
    import numpy as np
    from src.models.neuralgcm_residual.io import read_json, write_json
    from src.models.neuralgcm_residual.checks import tree_comparison
    from src.models.neuralgcm_residual.config import FIELDS
    script = ROOT/'scripts/training/train_neuralgcm_paper_residual.py'
    reports = {}
    for k in (1, 2):
        reference = a.output/f'K{k}_reference'
        resumed = a.output/f'K{k}_resumed'
        common = [sys.executable, '-u', str(script), '--phase', 'train',
                  '--resource-plan', str(a.resource_plan), '--statistics', str(a.statistics),
                  '--steps', str(k), '--width', '128', '--d-inner', '16', '--updates', '12',
                  '--validate-every', '10', '--validation-origins', '2']
        subprocess.run(common+['--output', str(reference)], check=True, env=worker_env)
        subprocess.run(common+['--output', str(resumed), '--stop-after', '10'], check=True, env=worker_env)
        assert read_json(reference/'baseline_validation.json') == read_json(resumed/'baseline_validation.json'), 'Fresh-process baseline differs'
        assert not (resumed/'DONE.json').exists(), 'Interrupted run marked complete'
        # Simulate metrics flushed after the last committed checkpoint.
        for name in ('metrics.jsonl', 'validation.jsonl'):
            with (resumed/name).open('a') as f:
                f.write(json.dumps({'update': 11, 'uncommitted_smoke_tail': True})+'\n')
        subprocess.run(common+['--output', str(resumed), '--resume'], check=True, env=worker_env)
        with (reference/'last.pkl').open('rb') as f:
            expected = pickle.load(f)
        with (resumed/'last.pkl').open('rb') as f:
            actual = pickle.load(f)
        for name in ('params', 'optimizer', 'rng'):
            tree_comparison(actual[name], expected[name], exact=True)
        assert actual['update'] == expected['update'] == 12
        assert actual['identity'] == expected['identity']
        assert actual['best'] == expected['best']
        assert read_json(reference/'baseline_validation.json') == read_json(resumed/'baseline_validation.json')
        for name in ('metrics.jsonl', 'validation.jsonl'):
            def rows(directory):
                return [{key: value for key, value in json.loads(line).items() if key != 'seconds'}
                        for line in (directory/name).read_text().splitlines()]
            assert rows(reference) == rows(resumed), name+' mismatch after resume'
            assert (resumed/(name+'.uncommitted')).exists()
        validation = [json.loads(x) for x in (resumed/'validation.jsonl').read_text().splitlines()]
        assert len(validation) == 2
        for row in validation:
            assert set(row['physical_rmse']) == set(FIELDS)
            for field in FIELDS:
                x = np.asarray(row['physical_rmse'][field])
                assert x.shape == (k, 37) and np.isfinite(x).all() and (x >= 0).all()
        for directory in (reference, resumed):
            assert read_json(directory/'DONE.json')['backbone_frozen']
            assert (directory/'best.pkl').exists()
        reports[f'K{k}'] = dict(passed=True, updates=12, interrupted_at=10,
            exact_checkpoint_and_metric_resume=True, uncommitted_tails_archived=True,
            physical_rmse_shape=[k, 37], physical_fields=list(FIELDS),
            baseline_and_best_checkpoint=True)
        write_json(a.output/'progress.json', reports)
    write_json(a.output/'report.json', dict(passed=True, runs=reports,
        slurm_job_id=os.environ.get('SLURM_JOB_ID')))
    print(json.dumps({'cli_smoke_passed': True}), flush=True)


if __name__ == '__main__':
    main()
