#!/usr/bin/env python3
"""Freeze advisory evaluation/report recoveries without rewriting historical inputs."""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.run_v24_2023_eval import (
    check_file, frozen_file, protocol_digest, read_json, snapshot_source, write_json,
    verify_protocol as verify_2023,
)

DEFAULTS = {
    "2023": ROOT / "artifacts/evaluations/v24_res1_2023_stride3_memory_20261004",
    "spectral": ROOT / "plots/analyze_models/data/resolution_eval/res1_spectral_pilot_2023_20260917",
    "amse": ROOT / "artifacts/checkpoints/v24_Ilya/res1_amse_blend_lambda0p1_20260924",
}


def prepare(recovery, family, original, *, separate_generated_inputs=False):
    if recovery.exists():
        raise FileExistsError(f"Refusing to overwrite recovery: {recovery}")
    original = original.resolve(strict=True)
    if family == "2023":
        protocol = verify_2023(original)
        if protocol.get("schema_version") != 2 or len(protocol["models"]) != 7:
            raise ValueError("Recovery requires the prepared seven-policy stride3 protocol")
        recovery.mkdir(parents=True)
        protocol.update(created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                        recovery_of=frozen_file(original / "protocol.json"),
                        numerical_policy="advisory", source_root=str(recovery / "source"))
        protocol["numerical_qualification"] = {
            "path": str(recovery / "numerical_qualification.json"), "required": False,
            "purpose": "Independent advisory diagnostics; never read by production release or report",
        }
        protocol["validation_policy"] = {"integrity": "blocking", "resources": "blocking",
                                         "numerical_repeatability": "advisory", "diagnostic_dependency": "none"}
        if separate_generated_inputs:
            protocol["generated_input_identity"] = "separate_v1"
            protocol["validation_policy"]["generated_inputs"] = (
                "Only finite calendar/solar variables absent from the source store are advisory; "
                "stored weather, coordinates, RNG, truth, parameters and state must match exactly. "
                "Fresh pilot evidence is required; historical whole-input hashes cannot establish this separation.")
        protocol["numerics"].update(xla_flags="--xla_gpu_enable_triton_gemm=false",
                                   environment={"TF_DETERMINISTIC_OPS": "0", "TF_CUDNN_DETERMINISTIC": "0"})
        protocol["resources"]["evidence"]["interpretation"] = (
            "Historical measurements only; fresh 2022 integrity/resource pilots release production. "
            "Finite numerical differences are advisory.")
        protocol["source_hashes"] = snapshot_source(recovery)
        protocol["protocol_hash"] = protocol_digest(protocol)
        write_json(recovery / "protocol.json", protocol)
        verify_2023(recovery)
        return protocol
    if family == "spectral":
        from scripts.experiments.run_v24_spectral_pilot import verify_protocol
        protocol = verify_protocol(original)
        source_manifest = original / "protocol.json"
        files = [source_manifest, *sorted((original / "models").glob("*/evaluation.json")),
                 *sorted((original / "models").glob("*/*.npz"))]
        if len(list((original / "models").glob("*/evaluation.json"))) != 3:
            raise ValueError("Spectral recovery requires three completed evaluations")
    elif family == "amse":
        from scripts.analyze_models.evaluate_v24_amse_pair import load_results
        # Validate all eight saved outputs before freezing references.
        exact, _, _, _, _, _ = load_results(original)
        if len(exact) != 8:
            raise ValueError("AMSE recovery requires eight completed evaluation outputs")
        source_manifest = original / "manifest.json"
        files = [source_manifest, *sorted((original / "eval").glob("*/*/*/specification.json")),
                 *sorted((original / "eval").glob("*/*/*/evaluation.json")),
                 *sorted((original / "eval").glob("*/*/*/*.npz"))]
    else:
        raise ValueError(f"Unknown recovery family: {family}")
    recovery.mkdir(parents=True)
    manifest = {
        "schema_version": 1, "family": family, "numerical_policy": "advisory",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "original_workflow": str(original), "original_manifest": frozen_file(source_manifest),
        "original_protocol_hash": read_json(source_manifest).get("protocol_hash"),
        "reused_artifacts": [frozen_file(path) for path in files],
        "source_root": str(recovery / "source"),
        "output_dir": str(recovery / "report"), "output_image_dir": str(recovery / "images"),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "matplotlib", "xarray")},
        "resources": {"cpus": 2, "mem_gib": 8, "walltime_seconds": 1800},
        "resource_evidence": {"previous_report_jobs": ["14080470", "14397801"],
                              "failure": "logic errors, not resource failures; completed evaluations are reused"},
    }
    manifest["source_hashes"] = snapshot_source(recovery)
    manifest["protocol_hash"] = protocol_digest(manifest)
    write_json(recovery / "recovery.json", manifest)
    return manifest


def verify(recovery):
    if (recovery / "protocol.json").exists():
        return verify_2023(recovery)
    manifest = read_json(recovery / "recovery.json")
    if protocol_digest(manifest) != manifest.get("protocol_hash"):
        raise ValueError("Recovery manifest hash mismatch")
    for record in (manifest["source_hashes"], manifest["original_manifest"], *manifest["reused_artifacts"]):
        check_file(record)
    for relative, digest in read_json(manifest["source_hashes"]["path"]).items():
        check_file({"path": str(Path(manifest["source_root"]) / relative), "sha256": digest})
    return manifest


def report(recovery):
    manifest = verify(recovery)
    if ROOT != Path(manifest["source_root"]):
        raise ValueError("Run reports from the recovery's frozen source")
    for name, version in manifest["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Recovery reporting package changed: {name}")
    original = Path(manifest["original_workflow"])
    outputs = dict(output_dir=Path(manifest["output_dir"]),
                   output_image_dir=Path(manifest["output_image_dir"]))
    completion = recovery / "completion.json"
    if completion.exists():
        previous = read_json(completion)
        if previous.get("recovery_hash") != manifest["protocol_hash"]:
            raise ValueError("Completed report belongs to a different recovery")
        print(f"Report already completed: {completion}")
        return previous
    if manifest["family"] == "spectral":
        from scripts.experiments.run_v24_spectral_pilot import validate_exact_losses
        from scripts.analyze_models.report_v24_spectral_pilot import report as make_report
        validation_path = recovery / "exact_loss_validation.json"
        validate_exact_losses(original, read_json(original / "protocol.json"), numerical_policy="advisory",
                              output_path=validation_path)
        summary = make_report(original, numerical_policy="advisory", validation_path=validation_path, **outputs)
    else:
        from scripts.analyze_models.evaluate_v24_amse_pair import report as make_report
        summary = make_report(original, **outputs)
    result = {"status": summary["status"], "integrity_passed": True,
              "recovery_hash": manifest["protocol_hash"], "original_manifest": manifest["original_manifest"],
              "output_dir": manifest["output_dir"], "output_image_dir": manifest["output_image_dir"],
              "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    write_json(completion, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "verify", "report"))
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--family", choices=tuple(DEFAULTS))
    parser.add_argument("--original", type=Path)
    parser.add_argument("--separate-generated-inputs", action="store_true")
    args = parser.parse_args()
    recovery = args.recovery.resolve()
    if args.phase == "prepare":
        if args.family is None:
            parser.error("prepare requires --family")
        if args.separate_generated_inputs and args.family != "2023":
            parser.error("--separate-generated-inputs applies only to the 2023 campaign")
        result = prepare(recovery, args.family, args.original or DEFAULTS[args.family],
                         separate_generated_inputs=args.separate_generated_inputs)
    elif args.phase == "verify":
        result = verify(recovery)
    else:
        result = report(recovery)
    print(f"{args.phase}: {recovery} ({result.get('protocol_hash', result.get('status'))})")


if __name__ == "__main__":
    main()
