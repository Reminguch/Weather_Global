"""The additional production release gate must fail closed."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[2] / 'scripts/training/smoke_neuralgcm_detailed.py'
SPEC = importlib.util.spec_from_file_location('detailed_smoke', PATH)
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


class DetailedSmokeGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = {'source_id':'frozen', 'configs':{f'arm{i}':{'sha256':f'config{i}'} for i in range(4)}}
        self.reports = {}
        for run, config in self.manifest['configs'].items():
            self.reports[run] = dict(passed=True, run_id=run, source_id='frozen', runner_sha256='runner',
                                    config_sha256=config['sha256'], gpu_measured=True, seconds=1.,
                                    checks={k:{'passed':True} for k in SMOKE.REQUIRED_CHECKS})
        self.save()

    def save(self):
        for run, report in self.reports.items():
            p = self.root/run/'report.json'
            p.parent.mkdir(exist_ok=True)
            p.write_text(json.dumps(report))

    def validate(self):
        return SMOKE.validate_reports(self.manifest, self.root, 'runner')

    def test_all_four_required(self):
        self.assertEqual(len(self.validate()), 4)
        (self.root/'arm3/report.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.validate()

    def test_failed_and_cpu_only_reports_rejected(self):
        for field in ('passed', 'gpu_measured'):
            self.reports['arm0'][field] = False
            self.save()
            with self.assertRaises(ValueError):
                self.validate()
            self.reports['arm0'][field] = True

    def test_source_runner_config_and_arm_must_match(self):
        for field in ('run_id','source_id','runner_sha256','config_sha256'):
            value = self.reports['arm0'][field]
            self.reports['arm0'][field] = 'stale'
            self.save()
            with self.assertRaises(ValueError):
                self.validate()
            self.reports['arm0'][field] = value

    def test_pilot_statistics_cannot_release_production(self):
        self.reports['arm0']['not_a_production_gate'] = True
        self.save()
        with self.assertRaises(ValueError):
            self.validate()

    def test_missing_or_failed_closed_loop_check_rejected(self):
        self.reports['arm0']['checks'].pop('live20_warm')
        self.save()
        with self.assertRaises(ValueError):
            self.validate()
        self.reports['arm0']['checks']['live20_warm'] = {'passed':False}
        self.save()
        with self.assertRaises(ValueError):
            self.validate()


if __name__ == '__main__':
    unittest.main()
