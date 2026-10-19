"""Provenance and evidence checks for the 2023 verification campaign.

Qualification is a derived result, bound to a frozen protocol. Missing GPU
evidence must never be interpreted as a successful numerical qualification.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REQUIRED_CHECKS = (
    "effective_inputs_and_parameters", "within_process_repeatability",
    "fresh_process_repeatability", "matching_gpu_repeatability",
    "standalone_interleaved_baseline", "development_bridge",
)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha256(path)}


def verify_records(value):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            if sha256(value["path"]) != value["sha256"]:
                raise ValueError(f"Qualification/provenance artifact changed: {value['path']}")
        for child in value.values():
            verify_records(child)
    elif isinstance(value, list):
        for child in value:
            verify_records(child)


def _bounds(values):
    values = np.asarray(values).astype("datetime64[ns]")
    if not values.size:
        raise ValueError("Empty provenance time selection")
    return {"first": str(values.min()), "last": str(values.max()), "count": int(values.size)}


def audit_training_provenance(protocol):
    """Audit eligible training windows, rather than calling store extent training.

The segment loader drops an incomplete final segment. Report both the full
manifest split and the complete-segment envelope; neither is a replay of the
individual minibatches visited by a checkpoint's optimization trajectory.
"""
    import xarray as xr

    root = Path(protocol["project_root"])
    manifests, models = {}, []
    for model in protocol["models"]:
        config = json.loads(Path(model["run_config"]["path"]).read_text())
        manifest_root = root / config["data"]["anchor_manifest_root"]
        metadata_path = manifest_root / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        key = str(manifest_root)
        if key not in manifests:
            arrays = {name: np.load(manifest_root / "anchors" / f"{name}.npy", allow_pickle=False)
                      for name in ("anchor_indices", "anchor_times", "split_train", "split_val")}
            indices, times = arrays["anchor_indices"], arrays["anchor_times"]
            if len(indices) != len(times) or np.any(np.diff(indices) <= 0):
                raise ValueError("Invalid training anchor ordering")
            if not np.array_equal(np.diff(times).astype("timedelta64[h]"), np.diff(indices)*np.timedelta64(6, "h")):
                raise ValueError("Training anchor timestamps and indices disagree")
            if np.intersect1d(arrays["split_train"], arrays["split_val"]).size:
                raise ValueError("Training/validation anchor splits overlap")
            for split in (arrays["split_train"], arrays["split_val"]):
                if not len(split) or split.min() < 0 or split.max() >= len(times):
                    raise ValueError("Invalid training/validation split")
            manifests[key] = {
                "metadata": file_record(metadata_path),
                "arrays": {name: file_record(manifest_root / "anchors" / f"{name}.npy") for name in arrays},
                "split_anchor_times": {label: _bounds(times[arrays[split]])
                                       for label, split in (("train", "split_train"), ("validation", "split_val"))},
            }
        times = np.load(manifest_root / "anchors/anchor_times.npy", allow_pickle=False)
        train = np.load(manifest_root / "anchors/split_train.npy", allow_pickle=False)
        validation = np.load(manifest_root / "anchors/split_val.npy", allow_pickle=False)
        segment = int(config["sequence"]["segment_steps"])
        eligible = train[:len(train) // segment * segment]
        if not len(eligible):
            raise ValueError("No complete training segment")
        step = np.timedelta64(6, "h")
        input_steps = int(config.get("derived", {}).get("input_steps", metadata.get("input_steps", 2)))
        target_steps = int(config["data"].get("target_steps", 1))
        train_times = times[eligible]
        last_targets = train_times + target_steps * step
        if last_targets.max() >= np.datetime64("2022-01-01"):
            raise ValueError(f"Complete training segments include 2022 targets: {model['model_id']}")
        if (times[validation] + target_steps * step).max() >= np.datetime64("2023-01-01"):
            raise ValueError("Validation includes test-year targets")
        inherited = config.get("derived", {}).get("parameter_overlay", {})
        parent = None
        if inherited.get("init_from"):
            parent = file_record(root / inherited["init_from"])
            if parent["sha256"] != inherited.get("init_from_sha256"):
                raise ValueError("Inherited checkpoint identity differs from run configuration")
            parent_config = Path(parent["path"]).parent.parent / "run_config.json"
            if parent_config.is_file():
                parent["run_config"] = file_record(parent_config)
                parent_data = json.loads(parent_config.read_text())["data"]
                if (root / parent_data["anchor_manifest_root"]).resolve() != manifest_root.resolve():
                    raise ValueError("Inherited correction uses a different training manifest; audit required")
        configured_baseline = config.get("derived", {}).get("baseline_checkpoint_fingerprint")
        if configured_baseline != protocol["baseline"]["checkpoint_sha256"]:
            raise ValueError("Training baseline fingerprint differs from evaluation baseline")
        models.append({
            "model_id": model["model_id"], "manifest": key,
            "training_inputs": {"first": str(train_times.min() - (input_steps - 1) * step), "last": str(train_times.max())},
            "training_targets": {"first": str(train_times.min() + step), "last": str(last_targets.max())},
            "eligible_complete_segment_anchors": len(eligible), "dropped_incomplete_segment_anchors": len(train) - len(eligible),
            "validation_targets": {"first": str(times[validation].min() + step), "last": str((times[validation] + target_steps * step).max())},
            "source_store_coverage": config.get("derived", {}).get("prepared_store_selection", {
                "time_start": metadata.get("source_time_start"), "time_end": metadata.get("source_time_end")}),
            "inherited_checkpoint": parent,
        })
    statistics = []
    for record in protocol["statistics"]:
        with xr.open_dataset(record["path"]) as dataset:
            attributes = {key: str(value) for key, value in dataset.attrs.items()}
        end = attributes.get("date_end")
        if end is None or int(end[:4]) >= 2023:
            raise ValueError("Normalization fitting coverage is missing or reaches the test year")
        statistics.append({**record, "fitting_attributes": attributes})
    return {
        "schema_version": 1, "status": "passed", "manifests": manifests, "models": models,
        "statistics": statistics,
        "backbone": {"path": protocol["baseline"]["path"], "sha256": protocol["baseline"]["checkpoint_sha256"],
                     "training_years": "1979-2015", "evidence": "official checkpoint filename and training-run fingerprint"},
        "limitations": ["Complete-segment bounds describe eligible data, not a replay of checkpoint minibatch history.",
                        "Historical checkpoint provenance relies on saved run configurations and fingerprints."],
    }


def tree_fingerprint(value):
    """Host-side digest including names, shapes and dtypes (no saved fields)."""
    import jax
    digest = hashlib.sha256()
    leaves, structure = jax.tree_util.tree_flatten_with_path(value)
    description = []
    digest.update(str(structure).encode())
    for path, leaf in leaves:
        array = np.asarray(jax.device_get(leaf))
        name = jax.tree_util.keystr(path)
        shape = list(array.shape)
        dtype = str(array.dtype)
        digest.update(json.dumps([name, shape, dtype], sort_keys=True).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
        description.append({"name": name, "shape": shape, "dtype": dtype})
    return {"sha256": digest.hexdigest(), "leaves": description}


def dataset_fingerprint(dataset):
    return tree_fingerprint({name: variable.values for name, variable in sorted(dataset.variables.items())})


def _require_digest(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Missing or invalid numerical digest: {label}")


def _require_tree_fingerprint(value, label, *, allow_empty=False):
    if not isinstance(value, dict):
        raise ValueError(f"Missing numerical fingerprint: {label}")
    _require_digest(value.get("sha256"), label)
    leaves = value.get("leaves")
    if not isinstance(leaves, list) or (not leaves and not allow_empty):
        raise ValueError(f"Missing numerical fingerprint leaves: {label}")
    for leaf in leaves:
        if (not isinstance(leaf, dict) or not isinstance(leaf.get("name"), str)
                or not isinstance(leaf.get("shape"), list)
                or any(type(size) is not int or size < 0 for size in leaf["shape"])
                or not isinstance(leaf.get("dtype"), str) or not leaf["dtype"]):
            raise ValueError(f"Malformed numerical fingerprint leaf: {label}")


def validate_observation(probe, times):
    """Require complete effective-input and field evidence, including empty-state trees."""
    observation = probe.get("numerical_observation")
    if not isinstance(observation, dict) or not isinstance(observation.get("setup"), dict):
        raise ValueError("Missing numerical observation setup")
    setup = observation["setup"]
    for name in ("baseline_params", "residual_params", "baseline_state", "residual_state"):
        _require_tree_fingerprint(setup.get(name), name,
                                  allow_empty=name.endswith("state") or (name == "residual_params" and probe["diagnostic_mode"] == "standalone"))
    statistics = setup.get("statistics")
    expected_stats = {"mean_by_level", "stddev_by_level", "diffs_stddev_by_level"}
    if not isinstance(statistics, dict) or set(statistics) != expected_stats:
        raise ValueError("Incomplete effective normalization fingerprints")
    for name, value in statistics.items():
        _require_tree_fingerprint(value, name)
    for name in ("baseline_model_config", "task_config"):
        if not isinstance(setup.get(name), dict) or not setup[name]:
            raise ValueError(f"Missing numerical configuration: {name}")
    initializations = observation.get("initializations")
    if not isinstance(initializations, list) or [r.get("initialization_time") for r in initializations] != times:
        raise ValueError("Numerical input fingerprints do not cover every initialization")
    for row in initializations:
        _require_tree_fingerprint(row.get("inputs"), "initialization inputs")
        _require_tree_fingerprint(row.get("rng"), "initialization RNG")
    expected = [(stamp, lead) for stamp in times for lead in range(40)]
    for name in ("predictions", "forcings"):
        records = observation.get(name)
        if not isinstance(records, list) or [(r.get("initialization_time"), r.get("step_index")) for r in records] != expected:
            raise ValueError(f"Numerical {name} evidence must cover every initialization and all 40 leads")
        for row in records:
            if name == "predictions":
                for branch in ("truth", "baseline", "full"):
                    _require_digest(row.get(branch), f"{branch} prediction")
            else:
                _require_tree_fingerprint(row, "forcing")
    return observation


def development_initialization_times(protocol, archived):
    """Map archived positions through the actual historical 2022 time coordinate."""
    from scripts.analyze_models.validate_res1_2023_data import LocalV2
    indices = archived.get("chosen_idx")
    if (not isinstance(indices, list) or len(indices) != 32
            or any(type(i) is not int for i in indices) or indices != sorted(set(indices))
            or archived.get("global_chosen_idx", indices) != indices
            or archived.get("completed_chosen_idx", indices) != indices):
        raise ValueError("Development bridge archived indices are incomplete or inconsistent")
    path = Path(protocol["pilot"]["data_path"])
    store = LocalV2(path)
    if store.attrs("time").get("calendar", "standard") not in ("standard", "gregorian", "proleptic_gregorian"):
        raise ValueError("Development bridge requires a Gregorian calendar")
    time = store.times().astype("datetime64[ns]")
    if time.ndim != 1 or np.isnat(time).any() or not np.all(np.diff(time) == np.timedelta64(6, "h")):
        raise ValueError("Development bridge requires a regular six-hour historical time coordinate")
    time = time[(time >= np.datetime64("2022-01-01")) & (time < np.datetime64("2023-01-01"))]
    if min(indices) < 1 or max(indices) + 40 >= len(time):
        raise ValueError("Development bridge archived indices exceed the historical 2022 slice")
    return [np.datetime_as_string(time[i], unit="s") + "Z" for i in indices]


class NumericalObserver:
    """Collect effective parameter/input identities and lead-wise field digests."""
    def __init__(self, derived_input_variables=()):
        self.derived_input_variables = tuple(sorted(derived_input_variables))
        self.data = {"initializations": [], "predictions": []}

    def setup(self, **values):
        self.data["setup"] = {
            key: tree_fingerprint(values[key]) for key in
            ("baseline_params", "residual_params", "baseline_state", "residual_state")
        }
        self.data["setup"]["statistics"] = {key: dataset_fingerprint(value) for key, value in values["stats"].items()}
        self.data["setup"]["baseline_model_config"] = dataclasses.asdict(values["baseline_model_config"])
        self.data["setup"]["task_config"] = dataclasses.asdict(values["task_config"])

    def initialization(self, record, inputs, key):
        entry = {"initialization_time": record["initialization_time"],
                 "inputs": dataset_fingerprint(inputs), "rng": tree_fingerprint(key)}
        if self.derived_input_variables:
            missing = set(self.derived_input_variables) - set(inputs.data_vars)
            if missing:
                raise ValueError(f"Missing generated input variables: {sorted(missing)}")
            derived = inputs[list(self.derived_input_variables)]
            if any(not np.isfinite(derived[name].values).all() for name in self.derived_input_variables):
                raise ValueError("Nonfinite generated input features")
            entry.update(
                derived_input_variables=list(self.derived_input_variables),
                source_inputs=dataset_fingerprint(inputs.drop_vars(self.derived_input_variables)),
                derived_inputs=dataset_fingerprint(derived), derived_inputs_finite=True,
            )
        self.data["initializations"].append(entry)

    def prediction(self, record, step_index, truth, baseline, full):
        self.data["predictions"].append({"initialization_time": record["initialization_time"], "step_index": step_index,
                                          **{name: dataset_fingerprint(dataset)["sha256"] for name, dataset in
                                             (("truth", truth), ("baseline", baseline), ("full", full))}})


def compare_losses(left, right, *, branches=("baseline", "full"), rtol=1e-3, atol=1e-6):
    left_records = left["original_graphcast_loss_per_initialization"]
    right_records = right["original_graphcast_loss_per_initialization"]
    if [r["initialization_time"] for r in left_records] != [r["initialization_time"] for r in right_records]:
        raise ValueError("Numerical comparison dates differ")
    result = {}
    for branch in branches:
        a = np.asarray([r[f"{branch}_per_step"] for r in left_records], dtype=float)
        b = np.asarray([r[f"{branch}_per_step"] for r in right_records], dtype=float)
        if (a.shape != b.shape or a.ndim != 2 or a.shape[0] == 0 or a.shape[1] != 40
                or not np.isfinite(a).all() or not np.isfinite(b).all()
                or np.any(a < 0) or np.any(b < 0)
                or (branch == "baseline" and (np.any(a <= 0) or np.any(b <= 0)))):
            raise ValueError("Invalid numerical comparison loss arrays")
        difference = np.abs(a-b)
        result[branch] = {"passed": bool(np.all(difference <= atol + rtol*np.abs(a))),
                          "max_absolute_difference": float(difference.max()),
                          "max_relative_difference": float(np.max(difference/np.maximum(np.abs(a), atol))),
                          "first_differing_index": np.argwhere(difference > atol + rtol*np.abs(a))[:1].tolist()}
    return {"passed": all(row["passed"] for row in result.values()), "branches": result, "rtol": rtol, "atol": atol}


def validate_numerical_qualification(workflow, protocol):
    path = Path(protocol.get("numerical_qualification", {}).get("path", Path(workflow)/"numerical_qualification.json"))
    if not path.is_file():
        raise ValueError("Numerical qualification is pending: missing GPU evidence")
    report = json.loads(path.read_text())
    if report.get("status") != "passed" or report.get("protocol_hash") != protocol["protocol_hash"]:
        raise ValueError("Numerical qualification is pending, failed, or belongs to another protocol")
    if report.get("rtol") != 1e-3 or report.get("atol") != 1e-6:
        raise ValueError("Numerical qualification tolerances differ")
    checks = report.get("checks", {})
    for name in REQUIRED_CHECKS:
        check = checks.get(name, {})
        if check.get("passed") is not True or not check.get("artifacts"):
            raise ValueError(f"Missing numerical qualification evidence: {name}")
    verify_records(report)
    # Recompute the scientific checks from the saved evidence. A manually set
    # status/check boolean must not be sufficient to release production.
    rebuilt = build_numerical_qualification(protocol, [r["path"] for r in report.get("probe_artifacts", [])])
    if rebuilt["status"] != "passed" or rebuilt["checks"] != checks or rebuilt["runtime"] != report.get("runtime"):
        raise ValueError("Numerical qualification does not reproduce from its evidence")
    return report


def build_numerical_qualification(protocol, probe_paths):
    """Derive every qualification decision from complete frozen-2022 probes."""
    paths = [Path(p).resolve() for p in probe_paths]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate numerical probe")
    models = {m["model_id"]: m for m in protocol["models"]}
    probes = []
    for path in paths:
        probe = json.loads(path.read_text())
        model = models.get(probe.get("model_id"))
        if model is None or probe.get("protocol_hash") != protocol["protocol_hash"]:
            raise ValueError("Probe identity differs from the frozen protocol")
        if probe.get("checkpoint_sha256") != model["checkpoint_sha256"] or probe.get("baseline_checkpoint_sha256") != protocol["baseline"]["checkpoint_sha256"]:
            raise ValueError("Probe checkpoint identity differs")
        if probe.get("evaluation_status") != "complete" or probe.get("metrics_complete") is not True:
            raise ValueError("Incomplete numerical probe")
        times = [r["initialization_time"] for r in probe["original_graphcast_loss_per_initialization"]]
        if (not times or times != sorted(set(times)) or probe.get("target_steps") != 40
                or probe.get("evaluated_samples") != len(times)):
            raise ValueError("Numerical probe date/lead coverage is incomplete")
        if any(not stamp.startswith("2022-") for stamp in times):
            raise ValueError("Numerical qualification may use only 2022 forecasts")
        if probe.get("diagnostic_mode") not in ("interleaved", "standalone", "bridge"):
            raise ValueError("Unknown numerical diagnostic mode")
        if probe.get("diagnostic_mode") != "bridge" and times != protocol["pilot"]["initialization_times"]:
            raise ValueError("Numerical probe dates differ from the four frozen pilots")
        for field in ("pid", "hostname", "gpu_uuid", "gpu_kind", "driver_version", "process_run_id"):
            if not probe.get(field):
                raise ValueError(f"Probe lacks {field}")
        if type(probe["pid"]) is not int or probe["pid"] < 1 or type(probe.get("repeat_index")) is not int or probe["repeat_index"] < 0:
            raise ValueError("Invalid numerical process/repeat identity")
        if probe.get("rs_reset_every_step") is not model.get("reset_state_every_step", False):
            raise ValueError("Numerical probe memory policy differs from the frozen model")
        validate_observation(probe, times)
        compare_losses(probe, probe)
        probe["_artifact"] = file_record(path)
        probes.append(probe)
    if len({(p["gpu_kind"], p["driver_version"]) for p in probes}) > 1:
        raise ValueError("All qualification probes must use the same GPU kind and driver")
    for model_id in models:
        setups = [p["numerical_observation"]["setup"] for p in probes
                  if p["model_id"] == model_id and p["diagnostic_mode"] != "standalone"]
        if setups and any(any(s[key] != setups[0][key] for key in ("residual_params", "residual_state")) for s in setups[1:]):
            raise ValueError(f"Effective residual parameters/state differ within policy {model_id}")
    checks = {name: {"passed": False, "artifacts": [], "reason": "Required GPU evidence is absent"} for name in REQUIRED_CHECKS}

    def register(name, evidence, comparisons=None, passed=True, reason=None):
        checks[name] = {"passed": bool(passed), "artifacts": [p["_artifact"] for p in evidence]}
        if comparisons is not None:
            checks[name]["comparisons"] = comparisons
        if reason is not None:
            checks[name]["reason"] = reason

    interleaved = [p for p in probes if p.get("diagnostic_mode") == "interleaved"]
    expected_models = set(models)
    identity_keys = ("baseline_params", "baseline_state", "statistics", "baseline_model_config", "task_config")

    def input_identity(observation):
        return {"initializations": observation["initializations"],
                "truth": [(r["initialization_time"], r["step_index"], r["truth"]) for r in observation["predictions"]],
                "forcings": observation["forcings"]}

    if {p["model_id"] for p in interleaved} == expected_models:
        reference = interleaved[0]["numerical_observation"]
        identities_pass = all(
            all(p["numerical_observation"]["setup"][key] == reference["setup"][key] for key in identity_keys)
            and input_identity(p["numerical_observation"]) == input_identity(reference)
            for p in interleaved)
        register("effective_inputs_and_parameters", interleaved, passed=identities_pass)

    def same_process(a, b):
        return (a["hostname"], a["pid"], a["gpu_uuid"], a.get("process_run_id")) == (b["hostname"], b["pid"], b["gpu_uuid"], b.get("process_run_id"))

    within, within_results = [], []
    for model_id in sorted(models):
        members = [p for p in interleaved if p["model_id"] == model_id]
        pairs = [(a, b) for i, a in enumerate(members) for b in members[i+1:]
                 if same_process(a, b) and a["repeat_index"] != b["repeat_index"]]
        if pairs:
            for a, b in pairs:
                within.extend([a, b])
                within_results.append({"model_id": model_id, **compare_losses(a, b)})
    if {r["model_id"] for r in within_results} == expected_models:
        register("within_process_repeatability", within, within_results, all(r["passed"] for r in within_results))
    primary = [p for p in interleaved if p["model_id"] == "M1"]
    for name, predicate in (
        ("fresh_process_repeatability", lambda a, b: not same_process(a, b) and a["gpu_uuid"] == b["gpu_uuid"]),
        ("matching_gpu_repeatability", lambda a, b: a["gpu_uuid"] != b["gpu_uuid"] and a["gpu_kind"] == b["gpu_kind"] and a["driver_version"] == b["driver_version"]),
    ):
        evidence, comparisons = [], []
        for i, a in enumerate(primary):
            for b in primary[i+1:]:
                if predicate(a, b):
                    evidence.extend([a, b]); comparisons.append(compare_losses(a, b))
        if comparisons:
            register(name, evidence, comparisons, all(r["passed"] for r in comparisons))
    standalone = [p for p in probes if p.get("diagnostic_mode") == "standalone" and p["model_id"] == "M1"]
    if standalone and primary:
        evidence, comparisons = [], []
        for a in standalone:
            for b in primary:
                observation_a, observation_b = a["numerical_observation"], b["numerical_observation"]
                same_inputs = input_identity(observation_a) == input_identity(observation_b)
                same_params = all(observation_a["setup"][key] == observation_b["setup"][key] for key in identity_keys)
                comparison = compare_losses(a, b, branches=("baseline",))
                comparison["passed"] &= same_inputs and same_params
                evidence.extend([a, b]); comparisons.append(comparison)
        register("standalone_interleaved_baseline", evidence, comparisons, all(r["passed"] for r in comparisons))
    bridge = [p for p in probes if p.get("diagnostic_mode") == "bridge" and p["model_id"] == "M1"]
    if bridge:
        comparisons = []
        for probe in bridge:
            source = probe.get("bridge_source")
            if not source:
                raise ValueError("Development bridge lacks the archived comparison source")
            verify_records(source)
            old = json.loads(Path(source["path"]).read_text())
            if old.get("evaluated_samples") != 32 or probe.get("evaluated_samples") != 32 or old.get("target_steps") != 40:
                raise ValueError("Development bridge must reproduce all 32 saved development starts")
            old_checkpoint = Path(old["ckpt"])
            if not old_checkpoint.is_absolute():
                old_checkpoint = Path(protocol["project_root"]) / old_checkpoint
            if old_checkpoint.resolve() != Path(models["M1"]["checkpoint"]).resolve():
                raise ValueError("Development bridge source checkpoint differs")
            if probe.get("bridge_source_chosen_indices") != old.get("global_chosen_idx", old.get("chosen_idx")):
                raise ValueError("Development bridge dates do not reproduce the archived selection")
            if [r["initialization_time"] for r in probe["original_graphcast_loss_per_initialization"]] != development_initialization_times(protocol, old):
                raise ValueError("Development bridge UTC dates differ from archived indices in the historical 2022 slice")
            previous, current = old["original_graphcast_loss"], probe["original_graphcast_loss"]
            effects = []
            for label, aggregate in (("archived", previous), ("current", current)):
                baseline = np.asarray(aggregate.get("baseline_per_step"), dtype=float)
                full = np.asarray(aggregate.get("full_per_step"), dtype=float)
                if (baseline.shape != (40,) or full.shape != (40,)
                        or not np.isfinite([baseline, full]).all() or np.any(baseline <= 0) or np.any(full < 0)):
                    raise ValueError(f"Invalid {label} development bridge aggregate losses")
                if label == "current":
                    records = probe["original_graphcast_loss_per_initialization"]
                    for branch, values in (("baseline", baseline), ("full", full)):
                        pooled = np.mean([r[f"{branch}_per_step"] for r in records], axis=0)
                        if not np.allclose(values, pooled, rtol=1e-6, atol=1e-7):
                            raise ValueError("Development bridge aggregate losses disagree with per-initialization losses")
                computed = float(100 * (1 - full.mean() / baseline.mean()))
                saved = aggregate.get("improvement_pct_rollout")
                if not isinstance(saved, (int, float)) or not np.isfinite(saved) or not np.isclose(computed, saved, rtol=1e-6, atol=1e-7):
                    raise ValueError(f"Invalid {label} development bridge reported improvement")
                effects.append(computed)
            comparisons.append({"source": source,
                                "previous_rollout_improvement_pct": effects[0],
                                "current_rollout_improvement_pct": effects[1],
                                "difference_percentage_points": effects[1] - effects[0],
                                "interpretation": "Descriptive bridge; archived numerics are not a repeatability reference."})
        register("development_bridge", bridge, comparisons)
    return {"schema_version": 1, "protocol_hash": protocol["protocol_hash"],
            "status": "passed" if all(c["passed"] for c in checks.values()) else ("pending" if not paths else "failed"),
            "rtol": 1e-3, "atol": 1e-6, "checks": checks,
            "runtime": {"gpu_kind": probes[0]["gpu_kind"], "driver_version": probes[0]["driver_version"]} if probes else None,
            "probe_artifacts": [file_record(path) for path in paths],
            "scope": "2022 numerical qualification only; no 2023 performance is used."}


def main():
    import argparse
    from scripts.experiments.run_v24_2023_eval import verify_protocol, write_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--probe-json", action="append", type=Path, default=[])
    args = parser.parse_args()
    protocol = verify_protocol(args.workflow)
    report = build_numerical_qualification(protocol, args.probe_json)
    output = args.workflow / "numerical_qualification.json"
    if output.exists():
        previous = json.loads(output.read_text())
        if previous.get("status") == "passed" and previous != report:
            raise FileExistsError("Preserve successful qualification; create a new campaign for changed evidence")
    write_json(output, report)
    print(f"Numerical qualification: {report['status']}; evidence files={len(args.probe_json)}; {output}")


if __name__ == "__main__":
    main()
