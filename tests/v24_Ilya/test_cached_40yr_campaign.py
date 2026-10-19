"""The conditional campaign must never release unverified data or duplicate jobs."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from scripts.experiments import run_v24_cached_40yr as campaign


@pytest.fixture
def flow(tmp_path):
    root=tmp_path/"campaign";root.mkdir()
    for name in ("reports","intents","logs"):(root/name).mkdir()
    value={"root":str(root),"workspace":str(tmp_path),"source":str(root/"source"),
           "workflow_id":"identity","storage_reserve_bytes":256*1024**3,
           "paths":{"download_status":str(tmp_path/"download.json"),"download_report":str(tmp_path/"stage_report.json"),
                    "source_zarr":str(tmp_path/"source.zarr")},"download_source_sha256":"producer"}
    campaign.write(root/"state.json",{"workflow_id":"identity","automation_enabled":True,"jobs":{}})
    return value


def qualified(**overrides):
    return {"passed":True,"integrity_passed":True,"bounded_memory_passed":True,"completed_steps":70,
            "full_validation_completed":True,"peak_host_gib":60.,"peak_gpu_gib":20.,
            "p90_end_to_end_seconds":20.,"startup_seconds":600.,"validation_seconds":400.,
            "checkpoint_seconds":1.,**overrides}


def test_matched_scientific_configs_and_exposure(tmp_path):
    template=campaign.read(campaign.ROOT/campaign.TEMPLATE)
    paths={key:str(tmp_path/key) for key in ("prepared_root","anchor_root","baseline_checkpoint","stats_dir")}
    configs=[campaign.training_config(template,paths,tmp_path,w) for w in (128,512)]
    for width,c in zip((128,512),configs):
        assert c["architecture"]["residual_width"]==width
        assert c["architecture"]["residual_initialization"]=="fresh"
        assert c["architecture"]["temporal_dt_rank"]=="32"
        assert c["architecture"]["temporal_d_inner"]==16
        assert c["optimizer"]["max_steps"]*c["distributed"]["per_device_batch_size"]==20000
        assert c["optimizer"]["warmup_steps"]==50
        assert c["optimizer"]["checkpoint_every"]==250
        assert c["sequence"]["feedback_mode"]=="baseline"
        assert c["validation"]["num_segments"] is None
    assert configs[0]["optimizer"]==configs[1]["optimizer"]


def test_download_gate_does_not_accept_coordinates_or_dry_run(flow):
    p=flow["paths"]
    campaign.write(p["download_status"],{"status":"running"})
    assert campaign.require_download(flow) is False
    campaign.write(p["download_status"],{"status":"failed","error":"network"})
    with pytest.raises(RuntimeError,match="Download failed"):campaign.require_download(flow)
    campaign.write(p["download_status"],{"status":"complete","output":p["source_zarr"],"source_sha256":"producer"})
    campaign.write(p["download_report"],{"dry_run":True,"requested_years":campaign.YEARS,"appended_years":campaign.YEARS})
    with pytest.raises(ValueError,match="dry-run"):campaign.require_download(flow)


def test_download_gate_wait_and_timeout(flow,monkeypatch):
    checks=iter([False,False,True]);sleeps=[]
    monkeypatch.setattr(campaign,"require_download",lambda _:next(checks))
    monkeypatch.setattr(campaign.time,"sleep",sleeps.append)
    monkeypatch.setattr(campaign,"sha",lambda _:"report")
    assert campaign.wait_download(flow)["passed"]
    assert sleeps==[60,60]
    monkeypatch.setattr(campaign,"require_download",lambda _:False)
    with pytest.raises(TimeoutError):campaign.wait_download(flow,timeout_seconds=0)


def test_storage_reserves_future_payload_not_only_current_shard(flow,monkeypatch):
    monkeypatch.setattr(campaign.shutil,"disk_usage",lambda _:SimpleNamespace(free=300*1024**3))
    assert campaign.storage_check(flow,40*1024**3)["outstanding_bytes"]==40*1024**3
    with pytest.raises(RuntimeError,match="Insufficient storage"):campaign.storage_check(flow,50*1024**3)


def test_gpu_pool_requires_qualification_and_uses_slowest_measurements():
    reports={"full":qualified(),"mig":qualified(p90_end_to_end_seconds=35.,peak_host_gib=75.)}
    b=campaign.training_budget(reports)
    assert b["constraint"]=="gpu40" and b["memory_gib"]==112
    assert b["p90_end_to_end_seconds"]==35.
    b=campaign.training_budget({"full":qualified(),"mig":{"passed":False,"failure_kind":"resource"}})
    assert b["constraint"]=="gpu40&nomig"
    with pytest.raises(ValueError,match="integrity"):
        campaign.training_budget({"full":qualified(),"mig":{"passed":False,"failure_kind":"integrity"}})
    with pytest.raises(ValueError,match="headroom"):
        campaign.training_budget({"full":qualified(peak_gpu_gib=33.)})
    with pytest.raises(ValueError,match="bounded_memory"):
        campaign.training_budget({"full":qualified(bounded_memory_passed=False)})


def test_checkpoint_aligned_24h_chunks_cover_full_budget():
    chunks=campaign.training_chunks(qualified())
    assert len(chunks)>1 and chunks[-1]["stop_step"]==5000
    assert sum(c["stop_step"]-c["start_step"] for c in chunks)==5000
    assert all(c["time_minutes"]<=1440 and c["stop_step"]%250==0 for c in chunks)
    assert all(a["stop_step"]==b["start_step"] for a,b in zip(chunks,chunks[1:]))
    with pytest.raises(RuntimeError,match="checkpoint interval"):
        campaign.training_chunks(qualified(p90_end_to_end_seconds=1000.))


def test_slurm_dependencies_array_and_pinned_workspace(flow):
    job=campaign.spec("cache_generate",array=40,constraint="gpu40",cpus=4,memory=64,minutes=600)
    args=campaign.sbatch_args(flow,"cache_generate",job,["123"])
    assert "--dependency=afterok:123" in args and "--array=0-39" in args
    assert "--constraint=gpu40" in args and "--gres=gpu:1" in args
    assert args[-1]==flow["workspace"]
    assert not any(a.startswith("--qos=") for a in args)
    assert all("%" not in a for a in args if a.startswith("--array="))


def test_submit_is_idempotent_and_refuses_unresolved_intent(flow,monkeypatch):
    calls=[]
    monkeypatch.setattr(campaign.subprocess,"run",lambda args,**kw:calls.append(args) or SimpleNamespace(stdout="123\n"))
    job=campaign.spec("download_gate")
    assert campaign.submit_job(flow,"download_gate",job)=="123"
    assert campaign.submit_job(flow,"download_gate",job)=="123" and len(calls)==1
    campaign.write(Path(flow["root"])/"intents/second_1.json",{"command":"unknown"})
    with pytest.raises(RuntimeError,match="Unresolved submission"):
        campaign.submit_job(flow,"second",job)
    assert len(calls)==1


def test_dry_run_does_not_enable_automation_write_or_submit(flow,monkeypatch):
    p=Path(flow["root"])/"state.json";before=p.read_bytes()
    monkeypatch.setattr(campaign.subprocess,"run",lambda *a,**kw:pytest.fail("Dry run submitted"))
    assert campaign.submit(flow,dry=True)["submitted"] is False
    assert p.read_bytes()==before


def test_download_gate_budget_follows_observed_slow_download(flow):
    flow['download_wait_hours'] = 168
    assert '--time=10085' in campaign.dry_run(flow)['first_command']
    assert '--qos=vlong' in campaign.dry_run(flow)['first_command']


def test_precompute_endpoint_stops_after_cache_and_rejects_training(flow, monkeypatch):
    flow["stop_after"] = "cache_verify"
    path = Path(flow["root"]) / "state.json"
    state = campaign.read(path)
    state["jobs"]["cache_verify"] = {"job_id": "123"}
    campaign.write(path, state)
    monkeypatch.setattr(campaign.subprocess, "run", lambda *a, **kw: pytest.fail("Submitted past endpoint"))
    campaign.advance(flow, "cache_verify", campaign.spec("cache_verify"), {"chunks": 2492})
    assert campaign.require_report(flow, "precompute_complete")["passed"]
    for stage in ("training_pilot", "training_release", "train", "evaluate", "report"):
        with pytest.raises(ValueError, match="Precompute-only"):
            campaign.submit_job(flow, stage, campaign.spec(stage))
    with pytest.raises(ValueError, match="Precompute-only"):
        campaign.submit_job(flow, "training_collect", campaign.spec("collect", width=128))
    preview = campaign.submit(flow, dry=True)
    assert preview["stages"][-1] == "cache_verify" and "training" not in preview


def test_report_provenance_and_tampering_block(flow):
    campaign.save_report(flow,"cache_verify",{"manifest_sha256":"abc"})
    assert campaign.require_report(flow,"cache_verify")["manifest_sha256"]=="abc"
    path=campaign.report_path(flow,"cache_verify");x=campaign.read(path)
    x["result"]["manifest_sha256"]="tampered";campaign.write(path,x)
    with pytest.raises(ValueError,match="Stale or changed"):campaign.require_report(flow,"cache_verify")


def test_cache_ready_must_match_verified_manifest(flow):
    cache=Path(flow["root"])/"cache";cache.mkdir();flow["paths"]["cache_root"]=str(cache)
    campaign.write(cache/"manifest.json",{"manifest_sha256":"hash"})
    campaign.write(cache/"READY.json",{"manifest_sha256":"hash","partial":True})
    campaign.save_report(flow,"cache_verify",{"manifest_sha256":"hash"})
    with pytest.raises(ValueError,match="complete verified"):campaign.require_cache(flow)
    campaign.write(cache/"READY.json",{"manifest_sha256":"hash","partial":False})
    assert campaign.require_cache(flow)["manifest_sha256"]=="hash"


def test_pilot_collectors_handle_oom_without_releasing_integrity_failures(flow,monkeypatch):
    p=Path(flow["root"])/"state.json";state=campaign.read(p)
    state["jobs"]["cache_full"]={"job_id":"123","status":"failed"};campaign.write(p,state)
    monkeypatch.setattr(campaign.subprocess,"run",lambda *a,**kw:SimpleNamespace(stdout="OUT_OF_MEMORY|\n"))
    result=campaign.collect_pilot(flow,"cache_full_collect")
    assert result["passed"] is False and result["failure_kind"]=="resource"
    campaign.save_report(flow,"cache_full",{"passed":False,"failure_kind":"integrity"})
    assert campaign.collect_pilot(flow,"cache_full_collect")["failure_kind"]=="integrity"


def test_failed_stage_does_not_advance(flow,monkeypatch):
    p=Path(flow["root"])/"state.json";state=campaign.read(p)
    state["jobs"]["download_gate"]={"job_id":"123","spec":campaign.spec("download_gate")};campaign.write(p,state)
    monkeypatch.setattr(campaign,"execute_stage",lambda *a,**kw:(_ for _ in ()).throw(ValueError("bad source")))
    monkeypatch.setattr(campaign,"advance",lambda *a:pytest.fail("Failed stage advanced"))
    with pytest.raises(ValueError,match="bad source"):campaign.execute(flow,"download_gate")
    assert campaign.read(p)["jobs"]["download_gate"]["status"]=="failed"


def test_release_failure_preserves_successful_work_for_resume(flow,monkeypatch):
    p=Path(flow["root"])/"state.json";state=campaign.read(p)
    state["jobs"]["download_gate"]={"job_id":"123","spec":campaign.spec("download_gate")};campaign.write(p,state)
    monkeypatch.setattr(campaign,"execute_stage",lambda *a,**kw:{"passed":True})
    monkeypatch.setattr(campaign,"advance",lambda *a:(_ for _ in ()).throw(RuntimeError("sbatch rejected")))
    assert campaign.execute(flow,"download_gate")["passed"]
    assert campaign.read(p)["jobs"]["download_gate"]["status"]=="release_failed"
    assert campaign.require_report(flow,"download_gate")["passed"]
