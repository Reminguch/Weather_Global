"""Paired-coverage and inferential contracts for the 2023 common-grid report."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json

import numpy as np
import pytest

from scripts.analyze_models.common_grid_metrics import DIAGNOSTICS
from scripts.experiments.run_common_grid_2023 import (
    analyze_values, block_counts, digest, gather, write,
)


def _hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


@pytest.fixture
def campaign(tmp_path):
    start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    stamps = [(start + timedelta(days=3 * index)).isoformat().replace('+00:00', 'Z')
              for index in range(122)]
    manifest = tmp_path / 'initializations.json'
    write(manifest, {'schema_version': 1, 'initializations': stamps})
    protocol = {
        'schema_version': 1, 'initialization_times': stamps, 'target_steps': 40,
        'lead_hours': list(range(6, 241, 6)),
        'verification': {'diagnostics': list(DIAGNOSTICS)},
        'initialization_manifest': str(manifest),
        'initialization_manifest_sha256': digest(manifest),
        'original_protocol_hash': _hash('original'),
        'm1': {'checkpoint_sha256': _hash('m1')},
        'gc1': {'checkpoint_sha256': _hash('gc1')},
        'gc025_checkpoint_sha256': _hash('gc025'),
    }
    protocol['protocol_hash'] = _hash(json.dumps(protocol, sort_keys=True))
    write(tmp_path / 'protocol.json', protocol)
    paths = {}
    for model in ('m1', 'gc025'):
        for shard in range(2):
            records = []
            for index in range(shard, len(stamps), 2):
                origin = datetime.fromisoformat(stamps[index].replace('Z', '+00:00'))
                valid = [(origin + timedelta(hours=hour)).isoformat().replace('+00:00', 'Z')
                         for hour in protocol['lead_hours']]
                branches = {'m1': 1., 'gc1': 4.} if model == 'm1' else {'gc025': 2.25}
                records.append({
                    'initialization_time': stamps[index], 'valid_times': valid,
                    'lead_hours': protocol['lead_hours'],
                    'truth_sha256': [_hash(value) for value in valid],
                    'mse': {branch: {name: [factor * (index + 1) * (lead + 1) * (field + 1)
                                          for lead in range(40)]
                                     for field, name in enumerate(DIAGNOSTICS)}
                            for branch, factor in branches.items()},
                })
            result = {
                'schema_version': 1, 'status': 'complete', 'model': model,
                'protocol_hash': protocol['protocol_hash'],
                'protocol_sha256': digest(tmp_path / 'protocol.json'),
                'original_protocol_hash': protocol['original_protocol_hash'],
                'initialization_manifest_sha256': protocol['initialization_manifest_sha256'],
                'shard_index': shard, 'shard_count': 2,
                'per_initialization': records,
                'evidence': {'checkpoint_sha256': _hash(model)},
            }
            if model == 'm1':
                result['evidence']['baseline_checkpoint_sha256'] = _hash('gc1')
            paths[model, shard] = tmp_path / 'shards' / f'{model}_{shard:02d}.json'
            write(paths[model, shard], result)
    return tmp_path, paths


def _change(path, mutate):
    result = json.loads(path.read_text())
    mutate(result)
    # Deliberately permit NaN here to check that reading untrusted evidence
    # rejects it even when its producer failed to use strict JSON encoding.
    path.write_text(json.dumps(result))


def test_gather_requires_complete_paired_coverage_and_ignores_native_and_partial(campaign):
    root, paths = campaign
    (root / 'shards' / 'm1_00.native.json').write_text('{invalid native output')
    (root / 'shards' / 'gc025_00.partial.json').write_text('{unfinished')
    native = root / 'shards' / 'native'
    native.mkdir()
    (native / 'm1_00.json').write_text('{native exact-loss metrics')
    protocol, values, sources = gather(root)
    assert len(sources) == 4
    assert all(value.shape == (122, 40, 12) for value in values.values())
    np.testing.assert_array_equal(values['gc1'], 4 * values['m1'])
    np.testing.assert_array_equal(values['gc025'], 2.25 * values['m1'])
    assert values['m1'][121, 39, 11] == 122 * 40 * 12
    paths['gc025', 1].unlink()
    with pytest.raises(ValueError, match='All three models require'):
        gather(root)


@pytest.mark.parametrize('fault,pattern', [
    ('truth', 'Different verification truth'),
    ('nonhex_truth', 'Incomplete truth provenance'),
    ('leads', 'Mismatched forecast leads'),
    ('valid_time', 'Mismatched valid times'),
    ('negative', 'Invalid MSE values'),
    ('nan', 'Invalid MSE values'),
    ('short_series', 'Invalid MSE values'),
    ('checkpoint', 'Wrong model checkpoint'),
    ('protocol', 'Wrong protocol'),
    ('manifest', 'Wrong initialization manifest'),
    ('missing_record', 'initialization coverage'),
    ('ownership', 'initialization coverage'),
    ('branches', 'Mismatched model branches'),
])
def test_gather_rejects_unpaired_or_misattributed_evidence(campaign, fault, pattern):
    root, paths = campaign

    def mutate(result):
        record = result['per_initialization'][0]
        if fault == 'truth': record['truth_sha256'][0] = _hash('different truth')
        elif fault == 'nonhex_truth': record['truth_sha256'][0] = 'z' * 64
        elif fault == 'leads': record['lead_hours'][0] = 12
        elif fault == 'valid_time': record['valid_times'][0] = record['valid_times'][1]
        elif fault == 'negative': record['mse']['gc025']['Z500'][0] = -1
        elif fault == 'nan': record['mse']['gc025']['Z500'][0] = float('nan')
        elif fault == 'short_series':
            for series in record['mse']['gc025'].values(): series.pop()
        elif fault == 'checkpoint': result['evidence']['checkpoint_sha256'] = _hash('other checkpoint')
        elif fault == 'protocol': result['protocol_sha256'] = _hash('other protocol')
        elif fault == 'manifest': result['initialization_manifest_sha256'] = _hash('pilot manifest')
        elif fault == 'missing_record': result['per_initialization'].pop()
        elif fault == 'ownership': result['per_initialization'].reverse()
        elif fault == 'branches': record['mse']['m1'] = record['mse'].pop('gc025')

    _change(paths['gc025', 0], mutate)
    with pytest.raises(ValueError, match=pattern):
        gather(root)


def test_duplicate_completed_shard_cannot_double_weight_initializations(campaign):
    root, paths = campaign
    (root / 'shards' / 'm1_duplicate.json').write_bytes(paths['m1', 0].read_bytes())
    with pytest.raises(ValueError, match='Duplicate shard'):
        gather(root)


def test_changed_protocol_and_changed_manifest_are_rejected(campaign):
    root, _ = campaign
    original = (root / 'protocol.json').read_bytes()
    _change(root / 'protocol.json', lambda value: value.update(target_steps=39))
    with pytest.raises(ValueError, match='protocol contents differ'):
        gather(root)
    (root / 'protocol.json').write_bytes(original)
    _change(root / 'initializations.json', lambda value: value['initializations'].pop())
    with pytest.raises(ValueError, match='manifest differs'):
        gather(root)


def test_overlapping_forecasts_share_truth_even_if_all_models_repeat_same_wrong_hash(campaign):
    root, paths = campaign
    # Start zero lead78h has the same valid time as start one lead6h.
    # Altering all models at that start still must disagree with the overlap.
    for model in ('m1', 'gc025'):
        _change(paths[model, 0], lambda value: value['per_initialization'][0]['truth_sha256'].__setitem__(12, _hash('wrong')))
    with pytest.raises(ValueError, match='Different verification truth'):
        gather(root)


def test_bootstrap_pools_mse_before_sqrt_and_keeps_models_paired():
    mse = np.asarray([1., 81., 4., 25., 100., 9., 16., 64.])[:, None, None]
    values = {'m1': mse, 'gc1': 4 * mse, 'gc025': 9 * mse}
    settings = {'replicates': 128, 'seed': 91, 'block_starts': 4,
                'sensitivity_block_starts': [2, 3]}
    times = [f'start{i}' for i in range(len(mse))]
    rmse, analyses = analyze_values(values, times, settings, exposed=['start1'])
    assert rmse['m1'][0, 0] == pytest.approx(np.sqrt(mse.mean()))
    assert rmse['m1'][0, 0] != pytest.approx(np.sqrt(mse).mean())
    assert analyses['primary']['block_starts'] == 4
    assert set(analyses) == {'primary', 'blocks2', 'blocks3', 'exclude_pilot'}
    assert analyses['exclude_pilot']['n_initializations'] == 7
    for analysis in analyses.values():
        for reference, expected in [('gc1', 50.), ('gc025', 200. / 3)]:
            comparison = analysis['comparisons'][reference]
            # Proportional error across every paired start makes the ratio
            # exact for every resample; independent draws would widen the CI.
            for quantity in ('rmse_reduction_pct', 'ci95_lower', 'ci95_upper'):
                np.testing.assert_allclose(comparison[quantity], expected, rtol=1e-13)


def test_bootstrap_excludes_exposed_starts_after_drawing_on_original_grid():
    mse = np.arange(1., 9.)[:, None, None]
    values = {'m1': mse.copy(), 'gc1': mse + 20, 'gc025': mse + 40}
    values['m1'][3] = 10000.
    settings = {'replicates': 32, 'seed': 14, 'block_starts': 8,
                'sensitivity_block_starts': []}
    times = [str(index) for index in range(8)]
    _, analyses = analyze_values(values, times, settings, exposed=['3'])
    eligible = [index for index in range(8) if index != 3]
    expected = 100 * (1 - np.sqrt(values['m1'][eligible].mean() / values['gc025'][eligible].mean()))
    comparison = analyses['exclude_pilot']['comparisons']['gc025']
    for quantity in ('rmse_reduction_pct', 'ci95_lower', 'ci95_upper'):
        assert comparison[quantity][0][0] == pytest.approx(expected)
    assert analyses['primary']['comparisons']['gc025']['rmse_reduction_pct'][0][0] < 0


def test_block_draws_preserve_count_are_deterministic_and_validate_bounds():
    draws = block_counts(122, 5, 64, 20260916)
    np.testing.assert_array_equal(draws.sum(axis=1), 122)
    np.testing.assert_array_equal(draws, block_counts(122, 5, 64, 20260916))
    assert (draws >= 0).all()
    with pytest.raises(ValueError, match='block length'):
        block_counts(4, 5, 64, 0)
    with pytest.raises(ValueError, match='integers'):
        block_counts(4, 2.5, 64, 0)


def test_zero_reference_error_cannot_produce_infinite_relative_improvement():
    values = {'m1': np.ones((8, 1, 1)), 'gc1': np.zeros((8, 1, 1)), 'gc025': np.ones((8, 1, 1))}
    settings = {'replicates': 16, 'seed': 0, 'block_starts': 3, 'sensitivity_block_starts': []}
    with pytest.raises(ValueError, match='undefined for zero'):
        analyze_values(values, list(range(8)), settings)
