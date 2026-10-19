#!/usr/bin/env python3
"""Paired block intervals for the approved common-grid truth-quality mask."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analyze_models.plot_public_checkpoints_2023_qc import (
    DEFAULT_NAME, DEFAULT_WORKFLOW, eligibility, identity,
)
from scripts.experiments.run_common_grid_2023 import block_counts, gather


def main():
    protocol, values, sources = gather(DEFAULT_WORKFLOW)
    audit_path = DEFAULT_WORKFLOW / 'audits/truth_repeats_20261018.json'
    mask, counts, _ = eligibility(protocol, json.loads(audit_path.read_text()))
    directory = ROOT / 'plots/analyze_models/data/resolution_eval' / DEFAULT_NAME
    summary_path = directory / 'summary.json'
    summary = json.loads(summary_path.read_text())
    if summary['protocol_hash'] != protocol['protocol_hash']:
        raise ValueError('Plot summary uses a different protocol')
    np.testing.assert_array_equal(counts, summary['eligible_initializations_per_lead_diagnostic'])
    settings = protocol['bootstrap']
    times = protocol['initialization_times']
    exposed = set(protocol['prior_exposure']['sampled_initialization_times'])
    cases = [('primary', settings['block_starts'], False)]
    cases += [(f'blocks{length}', length, False) for length in settings['sensitivity_block_starts']]
    cases += [('exclude_pilot', settings['block_starts'], True)]
    analyses = {}
    for name, length, exclude_pilot in cases:
        keep = np.array([not exclude_pilot or stamp not in exposed for stamp in times])
        valid = mask & keep[:, None, None]
        n = valid.sum(axis=0)
        draws = block_counts(len(times), length, settings['replicates'], settings['seed'])
        denominator = draws @ valid.reshape(len(times), -1).astype(np.float64)
        if np.any(denominator <= 0) or np.any(n <= 0):
            raise ValueError('Empty eligible bootstrap cell')
        sums = {m: np.where(valid, cube, 0) for m, cube in values.items()}
        pooled = {m: np.sqrt(cube.sum(axis=0) / n) for m, cube in sums.items()}
        boot = {m: np.sqrt((draws @ cube.reshape(len(times), -1)) / denominator)
                for m, cube in sums.items()}
        comparisons = {}
        for reference in ('gc1', 'gc025'):
            reductions = 100 * (1 - pooled['m1'] / pooled[reference])
            distribution = 100 * (1 - boot['m1'] / boot[reference])
            interval = np.percentile(distribution, [2.5, 97.5], axis=0).reshape((2, 40, 12))
            if name == 'primary':
                np.testing.assert_allclose(reductions, summary['m1_physical_rmse_reduction_pct'][reference],
                                           rtol=1e-12, atol=1e-12)
            comparisons[reference] = {
                'rmse_reduction_pct': reductions.tolist(),
                'ci95_lower': interval[0].tolist(), 'ci95_upper': interval[1].tolist(),
            }
        analyses[name] = {'block_starts': length, 'eligible_initializations': int(keep.sum()),
                          'counts_per_lead_diagnostic': n.tolist(), 'comparisons': comparisons}
        print(f'Completed {name}: counts {n.min()}..{n.max()}', flush=True)
    result = {
        'status': 'complete', 'protocol_hash': protocol['protocol_hash'],
        'bootstrap': settings, 'analyses': analyses,
        'method': 'Same paired nonwrapping block draws as original analysis; invalid truth cells have zero weight for every model, with a separate eligible denominator for each draw, lead, and diagnostic.',
        'interpretation': 'Pointwise 95% percentile intervals conditional on fixed checkpoints, the 2023 sample, and documented truth exclusions; no training-seed or simultaneous-coverage claim.',
        'plot_shading': False,
        'sources': sources + [identity(summary_path), identity(audit_path), identity(__file__),
                              identity(ROOT / 'scripts/experiments/run_common_grid_2023.py')],
    }
    path = directory / 'uncertainty.json'
    with path.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(path, flush=True)


if __name__ == '__main__':
    main()
