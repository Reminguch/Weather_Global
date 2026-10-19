from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from src.models.neuralgcm_residual import cache
from src.models.neuralgcm_residual.io import digest, read_json, write_json
from src.models.neuralgcm_residual.launcher import prepare_experiment, submit, load_experiment


def record(t):
    return {'origin':t,'valid':str(np.datetime64(t,'h')+np.timedelta64(6,'h')),
            'split':'train','origin_state':{'x':np.array([1.],np.float32)},
            'baseline_state':{'x':np.array([2.],np.float32)},
            'forcing':{'x':np.array([3.],np.float32)},
            'target_time':str(np.datetime64(t,'h')+np.timedelta64(6,'h'))}


def test_cache_requires_all_shards_and_live_parity(tmp_path,monkeypatch):
    producer={**cache.PRODUCER_CONTRACT,'dataset_id':'data','schema':{'channels':[('x','K',1)]}}
    origins=['2020-01-02T00','2020-01-02T06']
    cache.initialize_cache(tmp_path,producer,{'train':origins},shard_records=1)
    cache.initialize_cache(tmp_path,producer,{'train':origins},shard_records=1)  # JSON roundtrip
    monkeypatch.setattr(cache,'producer_identity',lambda *args:producer)
    monkeypatch.setattr(cache,'live_record',lambda b,s,t:record(t))
    cache.produce_shard(tmp_path,'train-00000',None,None,None)
    with pytest.raises(FileNotFoundError):
        cache.verify_cache(tmp_path,fresh_record=record)
    assert not (tmp_path/'READY.json').exists()
    cache.produce_shard(tmp_path,'train-00001',None,None,None)
    with pytest.raises(ValueError,match='Live-versus'):
        cache.verify_cache(tmp_path)
    ready=cache.verify_cache(tmp_path,fresh_record=record)
    assert ready['record_count']==2
    store=SimpleNamespace(identity='data',frame=lambda t:{'temperature':np.array([4.],np.float32)})
    reader=cache.CacheReader(tmp_path,store,'train')
    assert len(reader)==2 and not reader[0]['origin_state']['x'].flags.writeable
    with (tmp_path/'train-00000.zip').open('ab') as f: f.write(b'corrupt')
    with pytest.raises(ValueError,match='changed'):
        cache.CacheReader(tmp_path,store,'train')[0]


def test_prepare_eight_arms_and_dry_run_never_submits(tmp_path,monkeypatch):
    workspace=tmp_path/'workspace'
    source=workspace/'src'
    source.mkdir(parents=True)
    (source/'example.py').write_text('value = 1\n')
    # A partial copied egg-info shadows the installed distribution's version.
    egg=workspace/'third_party/neuralgcm/neuralgcm.egg-info'
    egg.mkdir(parents=True)
    (egg/'SOURCES.txt').write_text('generated build artifact\n')
    root=tmp_path/'experiment'
    manifest=prepare_experiment(root,workspace=workspace,python='/usr/bin/python3')
    assert len(manifest['configs'])==8 and len(manifest['resources'])==2
    assert not (root/'source/third_party/neuralgcm/neuralgcm.egg-info').exists()
    monkeypatch.setattr('subprocess.run',lambda *a,**k:pytest.fail('dry-run submitted a job'))
    plan=submit(root,'pretrain',dry_run=True,memory_gb=64,hours=4,after=['123','456'])
    assert len(plan)==8
    for job in plan:
        assert 'afterok:123:456' in job['sbatch']
        assert '%' not in ' '.join(job['sbatch']).replace('%j','')
        assert '/source/scripts/' in job['command'][1]
    assert not list((root/'submissions').glob('*.json'))
    (root/'source/src/example.py').write_text('value = 2\n')
    with pytest.raises(ValueError,match='changed'):
        load_experiment(root)


def test_selected_resolution_keeps_all_four_arms_and_rejects_other_resolution(tmp_path):
    workspace = tmp_path / 'workspace'
    (workspace / 'src').mkdir(parents=True)
    root = tmp_path / 'experiment'
    manifest = prepare_experiment(root, workspace=workspace, resolutions=['res2p8'])
    assert set(manifest['resources']) == {'res2p8'}
    assert len(manifest['configs']) == 4
    assert len(submit(root, 'pretrain', dry_run=True, memory_gb=32, hours=2)) == 4
    with pytest.raises(ValueError, match='outside'):
        submit(root, 'preflight', resolution='res1p4', dry_run=True)
    manifest['active_resolutions'] = ['res2p8', 'res1p4']
    write_json(root / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='inconsistent'):
        load_experiment(root)


def test_cache_lanes_cover_every_record_once_and_reject_changed_pilot(tmp_path, monkeypatch):
    from src.models.neuralgcm_residual import worker
    producer = {**cache.PRODUCER_CONTRACT, 'dataset_id': 'data'}
    times = np.arange(np.datetime64('2020-01-02T00'), np.datetime64('2020-02-10T00'), np.timedelta64(6, 'h'))
    resources = {'cache_root': str(tmp_path)}
    monkeypatch.setattr(worker, 'resolve_resources', lambda *a: resources)
    monkeypatch.setattr(worker, 'native_setup', lambda r: (None, None, None))
    monkeypatch.setattr(worker, 'producer_identity', lambda *a: producer)
    monkeypatch.setattr(worker, 'valid_k1_origins', lambda store, split: [str(t) for t in times] if split == 'train' else [str(times[-1])])
    monkeypatch.setattr(cache, 'producer_identity', lambda *a: producer)
    monkeypatch.setattr(cache, 'live_record', lambda b, s, t: record(t))
    worker.build_cache({}, 'res2p8', 'plan')
    worker.build_cache({}, 'res2p8', 'train-00000')
    generated = [worker.build_cache({}, 'res2p8', f'lane-{i}-3') for i in range(3)]
    shards = [s['shard_id'] for lane in generated for s in lane['completed_shards']]
    assert len(shards) == len(set(shards)) == len(read_json(tmp_path / 'manifest.json')['shards'])
    with (tmp_path / 'train-00000.zip').open('ab') as f:
        f.write(b'corrupt')
    with pytest.raises(ValueError, match='changed'):
        worker.build_cache({}, 'res2p8', 'lane-0-3')


def test_active_production_gates_still_require_passed_matching_reports(tmp_path):
    from src.models.neuralgcm_residual.worker import require_production_gates
    runtime = SimpleNamespace(backbone=SimpleNamespace(checkpoint_sha256='b'), adapter=SimpleNamespace(identity='s'),
                              store=SimpleNamespace(identity='d'), normalization=SimpleNamespace(identity='n'))
    ids = dict(source='source', backbone='b', native_schema='s', dataset='d', normalization='n')
    manifest = dict(source_id='source', resources={'res2p8': {}})
    write_json(tmp_path / 'checks/numerical_res2p8.json', dict(identities=ids, passed=True))
    with pytest.raises(FileNotFoundError):
        require_production_gates(tmp_path, manifest, 'res2p8', runtime)
    write_json(tmp_path / 'checks/profile_res2p8.json', dict(identities=ids, measured=True, gpu_measured=True))
    require_production_gates(tmp_path, manifest, 'res2p8', runtime)
    write_json(tmp_path / 'checks/numerical_res2p8.json', dict(identities=ids, passed=False))
    with pytest.raises(ValueError, match='passed'):
        require_production_gates(tmp_path, manifest, 'res2p8', runtime)


def test_preflight_always_uses_one_hour_gpu_test(tmp_path):
    workspace = tmp_path / 'workspace'
    (workspace / 'src').mkdir(parents=True)
    root = tmp_path / 'experiment'
    prepare_experiment(root, workspace=workspace, resolutions=['res2p8'])
    plan = submit(root, 'preflight', resolution='res2p8', phase='numerical', dry_run=True)[0]
    args = plan['sbatch']
    assert args[args.index('--qos') + 1] == 'gpu-test'
    assert args[args.index('--time') + 1] == '1:00:00'
    assert '--partition' not in args
    with pytest.raises(ValueError, match='gpu-test'):
        submit(root, 'preflight', resolution='res2p8', dry_run=True, qos='gpu-short')
    with pytest.raises(ValueError, match='one hour'):
        submit(root, 'preflight', resolution='res2p8', dry_run=True, hours=4)
