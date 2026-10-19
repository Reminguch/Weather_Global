"""Fast stdlib tests: no JAX, download, or real scheduler submission."""
import importlib.util
from pathlib import Path
import subprocess
from queue import SimpleQueue
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[2] / 'scripts/preprocessing/watch_neuralgcm_smoke.py'
spec = importlib.util.spec_from_file_location('smoke_watcher', MODULE)
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


class WatcherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'res2p8').mkdir()

    def fixture(self):
        months = [dict(id=f'{y}-{m:02}') for y in range(2015, 2024) for m in range(1, 13)]
        w.write(self.root / 'download_manifest.json', dict(source_id='source', months=months))
        w.write(self.root / 'READY.res2p8.json', dict(source_id='source', timestamps=13148, months=108))
        for m in months:
            p = self.root / 'shards' / m['id'] / 'res2p8'
            p.mkdir(parents=True)
            w.write(p / 'READY.json', {})
        self.manifest = dict(records=[dict(time=t) for t in w.expected_times()],
                             data_grid=dict(longitude=list(range(128)), latitude=list(range(64)), levels=list(range(37))))
        self.save_manifest()

    def save_manifest(self):
        p = self.root / 'res2p8/manifest.json'
        w.write(p, self.manifest)
        w.write(p.parent / 'READY.json', dict(manifest_sha256=w.sha(p)))

    def test_incomplete_cannot_trigger(self):
        self.assertFalse(w.ready(self.root))
        with self.assertRaises(ValueError):
            w.metadata(self.root, 'source')

    def test_exact_full_2p8_does_not_need_1p4(self):
        self.fixture()
        self.assertTrue(w.ready(self.root))
        verified = SimpleQueue()
        # Mock.call_count increments are not atomic across worker threads.
        with patch.object(w, 'verify_frame', side_effect=lambda root, record: verified.put(record['time'])):
            report = w.verify_dataset(self.root, 'source')
        checked = [verified.get() for _ in range(verified.qsize())]
        self.assertEqual(sorted(checked), w.expected_times())
        self.assertTrue(report['passed'])

    def test_stage_freezes_vendored_model_dependencies(self):
        project = self.root / 'project'
        for name in ('src/model.py', 'third_party/graphcast/graphcast/__init__.py',
                     'third_party/neuralgcm/neuralgcm/__init__.py',
                     'scripts/preprocessing/watch_neuralgcm_smoke.py',
                     'scripts/training/smoke_neuralgcm_training.py',
                     'scripts/training/smoke_neuralgcm_after_download.sbatch'):
            path = project / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# frozen dependency\n')
        checkpoint = self.root / 'checkpoint.pkl'
        checkpoint.write_bytes(b'checkpoint')
        w.write(self.root / 'download_manifest.json', dict(source_id='source', libraries={},
            checkpoints={'res2p8': dict(path=str(checkpoint), sha256=w.sha(checkpoint))}))
        output = self.root / 'smoke'
        w.stage(project, self.root, output)
        w.check_snapshot(output)
        for vendor in ('graphcast', 'neuralgcm'):
            self.assertTrue((output / 'code/third_party' / vendor / vendor / '__init__.py').is_file())

    def test_wrong_source_and_hash_rejected(self):
        self.fixture()
        with self.assertRaises(ValueError):
            w.metadata(self.root, 'other')
        w.write(self.root / 'res2p8/READY.json', dict(manifest_sha256='bad'))
        with self.assertRaises(ValueError):
            w.metadata(self.root, 'source')

    def test_duplicate_or_missing_timestamp_rejected(self):
        self.fixture()
        self.manifest['records'][1] = self.manifest['records'][0]
        self.save_manifest()
        with self.assertRaises(ValueError):
            w.metadata(self.root, 'source')

    def test_wrong_grid_rejected(self):
        self.fixture()
        self.manifest['data_grid']['longitude'] *= 2
        self.save_manifest()
        with self.assertRaises(ValueError):
            w.metadata(self.root, 'source')

    def test_corrupt_and_out_of_root_frames_rejected(self):
        f = self.root / 'res2p8/frame.npz'
        f.write_bytes(b'abc')
        record = dict(file='frame.npz', bytes=3, sha256=w.sha(f))
        w.verify_frame(self.root, record)
        f.write_bytes(b'xyz')
        with self.assertRaises(ValueError):
            w.verify_frame(self.root, record)
        record['file'] = '/etc/passwd'
        with self.assertRaises(ValueError):
            w.verify_frame(self.root, record)

    def test_submit_once_and_intent_before_side_effect(self):
        state = {}
        cfg = dict(data_root='data', gpu_python='python')
        def sbatch(cmd):
            self.assertIn('submission_intent', state)
            return '1234;cluster'
        with patch.object(w, 'command', side_effect=sbatch) as command:
            w.submit(self.root, cfg, state, lambda **v: state.update(v))
            w.submit(self.root, cfg, state, lambda **v: state.update(v))
        self.assertEqual(command.call_count, 1)
        self.assertEqual(state['job_id'], '1234')

    def test_ambiguous_submission_never_retried(self):
        state = {}
        cfg = dict(data_root='data', gpu_python='python')
        with patch.object(w, 'command', side_effect=subprocess.TimeoutExpired('sbatch', 60)) as command:
            with self.assertRaises(subprocess.TimeoutExpired):
                w.submit(self.root, cfg, state, lambda **v: state.update(v))
            with self.assertRaises(RuntimeError):
                w.submit(self.root, cfg, state, lambda **v: state.update(v))
        self.assertEqual(command.call_count, 1)

    def test_pending_job_and_failed_job(self):
        state = dict(job_id='1234')
        with patch.object(w, 'command', return_value='1234|PENDING|Resources'):
            self.assertFalse(w.track(self.root, state, lambda **v: state.update(v)))
        self.assertEqual(state['phase'], 'smoke_active')
        with patch.object(w, 'command', side_effect=['', '1234|FAILED|1:0|']):
            with self.assertRaises(RuntimeError):
                w.track(self.root, state, lambda **v: state.update(v))

    def test_success_requires_cuda_report_with_correct_identities(self):
        cp = self.root / 'deterministic_2_8_deg.pkl'
        cp.write_bytes(b'checkpoint')
        (self.root / 'result').mkdir()
        w.write(self.root / 'verification.json', dict(dataset_id='verified'))
        report = dict(passed=True, frozen_parameters_unchanged=True, devices=['cuda:0'],
                      kind='real_era5_one_update_smoke', dataset_id='verified', checkpoint_sha256=w.sha(cp))
        w.write(self.root / 'result/report.json', report)
        state = dict(job_id='1234')
        with patch.object(w, 'command', side_effect=['', '1234|COMPLETED|0:0|']):
            self.assertTrue(w.track(self.root, state, lambda **v: state.update(v)))
        report['devices'] = ['cpu:0']
        w.write(self.root / 'result/report.json', report)
        with patch.object(w, 'command', side_effect=['', '1234|COMPLETED|0:0|']):
            with self.assertRaises(RuntimeError):
                w.track(self.root, state, lambda **v: state.update(v))


if __name__ == '__main__':
    unittest.main()
