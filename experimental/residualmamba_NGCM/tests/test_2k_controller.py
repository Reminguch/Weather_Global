"""Scheduling regressions for the user-requested bounded diagnostic."""
import importlib.util
import json
from pathlib import Path
import pytest

PATH=Path(__file__).resolve().parents[3]/'scripts/training/ngcm_2k_controller.py'
SPEC=importlib.util.spec_from_file_location('ngcm_2k_control',PATH)
control=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(control)


def artifact(root,name,value=None):
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps({} if value is None else value))


def prepared(root):
    for name in ('setup.json','stats.json','cache/READY.json','diagnostic-training.json',
                 'cache-lane-0.json','cache-lane-1.json','cache-lane-2.json'):
        artifact(root,name)


def checkpoint(root,step):
    artifact(root,f'checkpoints/step_{step:05d}.json',{'step':step})
    artifact(root,f'checkpoints/step_{step:05d}.pkl')


def test_preparation_does_not_start_training_from_partial_artifacts(tmp_path):
    assert control.next_actions(tmp_path)==[('setup',False,[])]
    artifact(tmp_path,'setup.json')
    actions=control.next_actions(tmp_path)
    assert [a[0] for a in actions]==['stats','cache-0','cache-1','cache-2']
    assert sum(a[1] for a in actions)==3
    for name in ('stats.json','cache-lane-0.json','cache-lane-1.json','cache-lane-2.json'):
        artifact(tmp_path,name)
    assert control.next_actions(tmp_path)==[('cache-finalize',True,[])]


def test_resume_requires_validation_at_each_declared_endpoint(tmp_path):
    prepared(tmp_path)
    assert control.next_actions(tmp_path)[0][0]=='train-1000'
    checkpoint(tmp_path,317)
    assert control.next_actions(tmp_path)[0][0]=='train-1000'
    checkpoint(tmp_path,1000)
    assert control.next_actions(tmp_path)[0][0]=='validate-1000'
    artifact(tmp_path,'checkpoints/validation_01000.json')
    assert control.next_actions(tmp_path)[0][0]=='train-2000'
    checkpoint(tmp_path,2000)
    assert control.next_actions(tmp_path)[0][0]=='validate-2000'
    artifact(tmp_path,'checkpoints/validation_02000.json')
    assert control.next_actions(tmp_path)==[('report',False,[])]
    artifact(tmp_path,'report/COMPLETE.json')
    assert control.next_actions(tmp_path)==[]


def test_receipt_without_checkpoint_does_not_advance_training(tmp_path):
    artifact(tmp_path,'checkpoints/step_01000.json',{'step':1000})
    assert control.latest_step(tmp_path)==0


def test_download_and_gpu_qualification_are_independent_gates(tmp_path):
    assert control.desired_actions(tmp_path,False,False,False)==[]
    assert control.desired_actions(tmp_path,False,False,True)==[('qualify',False,[])]
    assert control.desired_actions(tmp_path,True,False,False)==[('setup',False,[])]
    prepared(tmp_path)
    assert control.desired_actions(tmp_path,True,False,False)==[]
    assert control.desired_actions(tmp_path,True,True,True)[0][0]=='train-1000'


def test_gpu_batches_use_requested_partition_qos_and_hour(tmp_path):
    plan={'root':str(tmp_path),'python':'/scratch/example/python'}
    args,script=control.batch(plan,'cache-2',True,['--lane','2'])
    assert all(flag in args for flag in ('--partition=ailab','--qos=gpu-test','--gres=gpu:h200:1','--time=01:00:00'))
    assert 'diagnostic_2k.py cache --root' in script and '--lane 2' in script
    assert 'NGCM_GPU_EXECUTION_POLICY=deterministic_gpu_v1' in script


def test_ambiguous_submission_cannot_be_repeated(tmp_path,monkeypatch):
    def timeout(*args,**kwargs):raise control.subprocess.TimeoutExpired('sbatch',60)
    monkeypatch.setattr(control.subprocess,'run',timeout)
    plan={'root':str(tmp_path),'python':'/scratch/example/python'}
    with pytest.raises(control.subprocess.TimeoutExpired):control.submit(plan,'setup',False,[],1)
    with pytest.raises(RuntimeError,match='already exists'):control.submit(plan,'setup',False,[],1)
    assert not control.read(tmp_path/'controller/jobs/setup-001.json').get('job_id')


def test_explicit_cluster_capacity_rejection_is_retryable(tmp_path,monkeypatch):
    answers=iter([control.subprocess.CompletedProcess([],1,'','allocation failure: MaxJobCount limit reached'),
                  control.subprocess.CompletedProcess([],0,'12345\n','')])
    monkeypatch.setattr(control.subprocess,'run',lambda *a,**k:next(answers))
    plan={'root':str(tmp_path),'python':'/scratch/example/python'}
    control.submit(plan,'setup',False,[],1)
    assert control.read(tmp_path/'controller/jobs/setup-001.json')['phase']=='rejected'
    control.submit(plan,'setup',False,[],2)
    assert control.read(tmp_path/'controller/jobs/setup-002.json')['job_id']=='12345'


def test_finished_job_can_leave_squeue_before_accounting_lookup(monkeypatch):
    monkeypatch.setattr(control.subprocess,'run',lambda *a,**k:control.subprocess.CompletedProcess(
        [],1,'','slurm_load_jobs error: Invalid job id specified'))
    monkeypatch.setattr(control.subprocess,'check_output',lambda *a,**k:'12345|COMPLETED|0:0|\n')
    assert control.state_of('12345')==('COMPLETED','0:0')
