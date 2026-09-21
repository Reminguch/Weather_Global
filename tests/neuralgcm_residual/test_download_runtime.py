"""Dependency-free regression checks for resume and frozen-source validation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT=Path(__file__).resolve().parents[2]/'scripts/preprocessing/run_neuralgcm_download_local.py'
spec=importlib.util.spec_from_file_location('download_runtime',SCRIPT)
runtime=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class DownloadRuntimeTests(unittest.TestCase):
    def test_one_resolution_can_resume_without_the_other(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            ready=root/'shards/2015-01/res2p8/READY.json'
            ready.parent.mkdir(parents=True)
            ready.write_text('{}')
            self.assertTrue(runtime.month_complete(root,'2015-01',['res2p8']))
            self.assertFalse(runtime.month_complete(root,'2015-01',['res1p4']))
            self.assertFalse(runtime.month_complete(root,'2015-01',['res2p8','res1p4']))

    def test_local_source_and_checkpoint_are_verified(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules):
            root=Path(directory)
            source=root/'source/src/models/neuralgcm_residual/example.py'
            source.parent.mkdir(parents=True)
            source.write_text('frozen = True\n')
            checkpoint=root/'checkpoints/model.pkl'
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b'checkpoint')
            manifest={'source_files':{'src/models/neuralgcm_residual/example.py':hashlib.sha256(source.read_bytes()).hexdigest()},
                      'checkpoints':{'res2p8':{'path':'/old/location/model.pkl','sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
                                     'res1p4':{'path':'/not/staged/model1.pkl','sha256':'unused'}},'libraries':{}}
            (root/'download_manifest.json').write_text(json.dumps(manifest))
            with patch.object(runtime,'RUNTIME',root):
                actual=runtime.checked_manifest(root,['res2p8'])
                self.assertEqual(list(actual['checkpoints']),['res2p8'])
                self.assertEqual(sys.modules['src.models'].__path__,[str(root/'source/src/models')])
                checkpoint.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'checkpoint changed'):
                    runtime.checked_manifest(root,['res2p8'])
                checkpoint.write_bytes(b'checkpoint')
                source.write_text('changed = True\n')
                with self.assertRaisesRegex(ValueError,'source changed'):
                    runtime.checked_manifest(root,['res2p8'])


if __name__=='__main__':
    unittest.main()
