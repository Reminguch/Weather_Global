"""Resource and submission guards for the approved paired experiment."""
import json

import pytest

from scripts.experiments import run_v24_amse_pair as pair


def test_budget_uses_measured_time_and_long_run_memory():
    measured = dict(passed=True, completed_steps=64, peak_rss_gib=100,
                    steady_p90_seconds=30, elapsed_seconds=2300, startup_seconds=400)
    result = pair.budget(measured, {'max_rss_gib': 230})
    assert result['memory_gib'] == 336
    assert result['time_minutes'] >= (400 + 30 * 2000 + 1800) * 1.35 / 60


@pytest.mark.parametrize('patch', ({'passed': False}, {'completed_steps': 4},
                                  {'steady_p90_seconds': float('nan')}, {'steady_p90_seconds': 500}))
def test_budget_rejects_missing_invalid_or_impossible_measurements(patch):
    measured = dict(passed=True, completed_steps=64, peak_rss_gib=100,
                    steady_p90_seconds=30, elapsed_seconds=2300, startup_seconds=400)
    measured.update(patch)
    with pytest.raises(ValueError):
        pair.budget(measured, {'max_rss_gib': 230})


def test_preflight_array_is_unthrottled_and_release_cpu_only(tmp_path):
    manifest = {'workspace': str(tmp_path / 'workspace'), 'source_root': str(tmp_path / 'source')}
    args = pair.sbatch(tmp_path, manifest, 'preflight', memory=320, minutes=120, array='0-1')
    assert '--array=0-1' in args and not any('%' in a for a in args if a.startswith('--array'))
    assert f"--chdir={manifest['workspace']}" in args
    args = pair.sbatch(tmp_path, manifest, 'release', memory=4, minutes=15, gpu=False, dependency='1234')
    assert '--dependency=afterok:1234' in args
    assert not any(a.startswith(('--gres', '--qos', '--constraint')) for a in args)


def test_uncertain_submission_is_never_duplicated(tmp_path):
    pair.write(tmp_path / 'submissions/preflight.json', {'command': ['sbatch', '--parsable']})
    with pytest.raises(RuntimeError, match='Unresolved submission'):
        pair.submit_job(tmp_path, 'preflight', ['sbatch', '--parsable'])


def test_existing_submission_reused_only_for_same_command(tmp_path):
    pair.write(tmp_path / 'submissions/preflight.json', {'command': ['sbatch', '--parsable'], 'job_id': '123'})
    assert pair.submit_job(tmp_path, 'preflight', ['sbatch', '--parsable']) == '123'
    with pytest.raises(ValueError, match='differs'):
        pair.submit_job(tmp_path, 'preflight', ['sbatch', '--other'])
