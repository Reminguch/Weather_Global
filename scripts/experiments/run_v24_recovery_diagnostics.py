"""Independent, bounded numerical diagnostics; never releases production jobs.

The supervisor has no JAX imports. Every runtime variant executes in a fresh
process, while fixed-step repeats reuse one compiled executable and fixed inputs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import resource
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_FLAGS = "--xla_gpu_enable_triton_gemm=false"
MODES = ("ordinary", "deterministic", "autotune_off", "deterministic_autotune_off")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def flags(mode):
    if mode not in MODES:
        raise ValueError("Unknown diagnostic runtime")
    result = BASE_FLAGS
    if mode.startswith("deterministic"):
        result += " --xla_gpu_deterministic_ops=true"
    if mode.endswith("autotune_off"):
        result += " --xla_gpu_autotune_level=0"
    return result


def verify(root):
    from scripts.experiments.frozen_runtime import verify_source
    manifest = read(root / "manifest.json")
    snapshot = verify_source(Path(manifest["source_root"]))
    if snapshot["source_digest"] != manifest["source_digest"]:
        raise ValueError("Diagnostic source identity changed")
    for record in manifest["inputs"]:
        if sha(record["path"]) != record["sha256"]:
            raise ValueError(f"Diagnostic input changed: {record['path']}")
    if manifest["production_dependencies"]:
        raise ValueError("Diagnostics must be independent of production")
    return manifest


def prepare(root, workspace, protocol, mgda):
    from scripts.experiments.frozen_runtime import freeze_source
    if root.exists():
        return verify(root)
    root.mkdir(parents=True)
    p = read(protocol)
    m = read(mgda / "manifest.json")
    model = next(x for x in p["models"] if x["model_id"] == "M1")
    inputs = [pin(protocol), pin(model["checkpoint"]), pin(model["run_config"]["path"]),
              pin(p["baseline"]["path"]), pin(mgda / "manifest.json"),
              pin(mgda / "source/snapshot.json"), pin(m["initialization"]["path"]),
              pin(p["pilot"]["data_metadata"]["path"])]
    inputs += [pin(Path(p["stats_dir"]) / name) for name in
               ("stddev_by_level.nc", "mean_by_level.nc", "diffs_stddev_by_level.nc")]
    inputs += [pin(mgda / "configs" / f"{arm}.json") for arm in ("mgda", "mse")]
    for label, dates in (("january", ["2022-01-15T00:00:00Z"]),
                         ("seasonal", ["2022-01-15T00:00:00Z", "2022-07-15T00:00:00Z"])):
        path = root / f"{label}.json"
        write(path, {"schema_version": 1,
                     "initializations": [{"initialization_time": t} for t in dates]})
        inputs.append(pin(path))
    snapshot = freeze_source(workspace, root / "source")
    manifest = {"format": "v24_recovery_diagnostics_v1", "workspace": str(workspace),
        "source_root": str(root / "source"), "source_digest": snapshot["source_digest"],
        "protocol": str(protocol), "mgda": str(mgda), "inputs": inputs,
        "runtime_modes": {mode: flags(mode) for mode in MODES},
        "production_dependencies": [], "dates_year": 2022,
        "scope": "Numerical investigation only; does not qualify or select production models."}
    write(root / "manifest.json", manifest)
    return manifest


def array_comparison(actual, expected):
    import numpy as np
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("Comparison shapes/dtypes differ")
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite numerical evidence")
    d = a - b
    denom = float(np.sum(b * b))
    return {"elements": int(a.size), "differing_elements": int(np.count_nonzero(actual != expected)),
        "max_absolute": float(np.max(np.abs(d))) if a.size else 0.,
        "squared_difference": float(np.sum(d * d)), "squared_reference": denom,
        "relative_l2": float(np.sqrt(np.sum(d*d) / max(denom, 1e-300)))}


def tree_compare(actual, expected):
    import jax
    if jax.tree_util.tree_structure(actual) != jax.tree_util.tree_structure(expected):
        raise ValueError("Comparison tree structures differ")
    return [array_comparison(a, b) for a, b in zip(jax.tree_util.tree_leaves(actual),
            jax.tree_util.tree_leaves(expected), strict=True)]


def tree_hash(value):
    import jax
    import numpy as np
    h = hashlib.sha256(str(jax.tree_util.tree_structure(value)).encode())
    for leaf in jax.tree_util.tree_leaves(value):
        a = np.asarray(leaf)
        h.update(str((a.shape, a.dtype)).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def config_for(root, manifest, output, *, seasonal=False, steps=1):
    from src.models.mamba.v24_Ilya.config import V24IlyaEvalConfig
    p = read(manifest["protocol"])
    model = next(x for x in p["models"] if x["model_id"] == "M1")
    return V24IlyaEvalConfig(ckpt=Path(model["checkpoint"]), out_json=output,
        ckpt_in=Path(p["baseline"]["path"]), stats_dir=Path(p["stats_dir"]),
        data_path=p["pilot"]["data_path"], val_year=2022, train_start_year=2015,
        train_end_year=2021, target_steps=steps, n_samples=2 if seasonal else 1,
        initialization_manifest=root / ("seasonal.json" if seasonal else "january.json"),
        model_id="M1_numerical_diagnostic", omit_rms_bias=True,
        **p["forecast"], **model["architecture"])


def runtime_identity(mode):
    from scripts.analyze_models.diagnose_v24_2023_numerics import gpu_identity
    import jax
    import jaxlib
    if os.environ.get("XLA_FLAGS") != flags(mode):
        raise ValueError("Unexpected diagnostic runtime flags")
    return {**gpu_identity({"resources": {"gpu": {}}}), "jax": jax.__version__,
        "jaxlib": jaxlib.__version__, "pid": os.getpid(), "xla_flags": flags(mode),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID")}


def localization_context(path):
    """Record streamed aggregation fingerprints, explicitly instrumented.

    Callbacks transfer arrays and can change optimization/scheduling. Their
    evidence locates an observed divergence; it is not a proof of causation.
    """
    from contextlib import contextmanager
    import jax
    import numpy as np
    from graphcast import typed_graph_net
    @contextmanager
    def context():
        original = typed_graph_net._node_update
        trace_count = [0]
        invocation = [0]
        control = {"phase": "initialization"}
        def patched(graph, node_fn, node_key, aggregation_fn):
            def aggregate(data, indices, count):
                site = trace_count[0]
                trace_count[0] += 1
                result = aggregation_fn(data, indices, count)
                def record(x, ids, y):
                    row = {"sequence": invocation[0], "phase": control["phase"], "site": site, "node_set": str(node_key),
                        "input_hash": tree_hash(x), "indices_hash": tree_hash(ids),
                        "output_hash": tree_hash(y), "input_shape": list(x.shape),
                        "output_shape": list(y.shape), "finite": bool(np.isfinite(x).all() and np.isfinite(y).all())}
                    invocation[0] += 1
                    with path.open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
                jax.debug.callback(record, data, indices, result, ordered=True)
                return result
            return original(graph, node_fn, node_key, aggregate)
        typed_graph_net._node_update = patched
        try:
            yield control
        finally:
            typed_graph_net._node_update = original
    return context()


def fixed_worker(root, manifest, label, mode, interleaved=False, instrument=False):
    import dataclasses
    from contextlib import nullcontext
    import jax
    import numpy as np
    from src.models.mamba.v24_Ilya import evaluation as ev
    directory = root / "fixed" / label
    directory.mkdir(parents=True, exist_ok=False)
    result = {"status": "running", "runtime": runtime_identity(mode), "mode": mode,
              "interleaved": interleaved, "instrumented": instrument, "repeats": []}
    write(directory / "result.json", result)
    config = config_for(root, manifest, directory / "unused.json")
    checkpoint = ev.load_graphcast_checkpoint(config.ckpt_in)
    task = dataclasses.replace(checkpoint.task_config, input_duration=config.input_duration)
    stats = ev.load_stats(config.stats_dir)
    data = ev.open_eval_dataset(config, task)
    records = ev.resolve_initialization_manifest(data, config.initialization_manifest, target_steps=1)
    args = tuple(ev.physical_fields_fp32(v) for v in ev.build_eval_batch(data,
        indices=[records[0]["dataset_index"]], target_steps=1, task_config=task))
    keys = jax.random.split(jax.random.PRNGKey(config.seed), 3)
    context = localization_context(directory / "aggregations.jsonl") if instrument else nullcontext()
    with context as control:
        predictors = ev.build_predictors(checkpoint.model_config, task, stats, config)
        params, state = predictors.baseline.init(keys[1], *args)
        params, _ = ev.overlay_frozen_baseline_params(params, checkpoint.params)
        state = ev.state_boundary_fp32(state)
        arguments = (params, state, keys[0], *args)
        lower = jax.jit(predictors.baseline.apply).lower(*arguments)
        compiled = lower.compile()
        result.update(input_hash=tree_hash(jax.device_get(args)), parameter_hash=tree_hash(params),
            state_hash=tree_hash(state), rng_hash=tree_hash(keys[0]),
            hlo_sha256=hashlib.sha256(lower.as_text().encode()).hexdigest())
        residual = residual_args = None
        if interleaved:
            init_params, init_state = predictors.residual.init(keys[2], *args)
            saved = ev.load_v24_Ilya_checkpoint(config.ckpt)
            ev.validate_param_tree_compatible(init_params, saved.residual_params)
            residual_args = (saved.residual_params, ev.state_boundary_fp32(init_state), keys[0], *args)
            residual = jax.jit(predictors.residual.apply).lower(*residual_args).compile()
        jax.effects_barrier()
        if control is not None:
            control["phase"] = "reference"
        expected = jax.device_get(compiled(*arguments))
        jax.effects_barrier()
        np.savez(directory / "baseline_reference.npz", **{
            f"leaf_{i}": np.asarray(x) for i, x in enumerate(jax.tree_util.tree_leaves(expected))})
        for repeat in range(5 if not instrument else 2):
            if control is not None:
                control["phase"] = f"repeat_{repeat}"
            if residual is not None:
                jax.block_until_ready(residual(*residual_args))
            started = time.monotonic()
            actual = jax.device_get(compiled(*arguments))
            jax.effects_barrier()
            result["repeats"].append({"repeat": repeat, "seconds": time.monotonic()-started,
                "comparison": tree_compare(actual, expected), "output_hash": tree_hash(actual)})
            write(directory / "result.json", result)
        jax.effects_barrier()
    data.dataset.close()
    result.update(status="complete", peak_host_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    write(directory / "result.json", result)


def child(root, manifest, arguments, *, mode, device=None, timeout=1800, source=None):
    source = Path(source or manifest["source_root"])
    environment = dict(os.environ)
    environment.update(XLA_FLAGS=flags(mode), PYTHONPATH=f"{source}:{source / 'third_party/graphcast'}",
                       PYTHONSAFEPATH="1", XLA_PYTHON_CLIENT_PREALLOCATE="false",
                       XLA_PYTHON_CLIENT_ALLOCATOR="platform",
                       TF_DETERMINISTIC_OPS="0", TF_CUDNN_DETERMINISTIC="0")
    environment.pop("JAX_PLATFORMS", None)
    if device is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(device)
    command = [sys.executable, "-u", *map(str, arguments)]
    label = hashlib.sha256(json.dumps(command).encode()).hexdigest()[:16]
    log = root / "logs" / f"child_{label}.log"
    log.parent.mkdir(exist_ok=True)
    record = {"command": command, "mode": mode, "gpu_selector": device, "log": str(log)}
    started = time.monotonic()
    print(shlex.join(command), flush=True)
    try:
        with log.open("a") as stream:
            completed = subprocess.run(command, env=environment, cwd=manifest["workspace"],
                stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
        record.update(returncode=completed.returncode, status="complete" if completed.returncode == 0 else "failed")
    except subprocess.TimeoutExpired:
        record.update(status="timeout")
    record["elapsed_seconds"] = time.monotonic()-started
    with (root / "cases.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    return record


def matrix(root, manifest):
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(devices) != 2 or len(set(devices)) != 2:
        raise ValueError("Matrix requires two allocated GPUs")
    script = Path(manifest["source_root"]) / "scripts/experiments/run_v24_recovery_diagnostics.py"
    for mode in MODES:
        for interleaved in (False, True):
            for process, device in (("first", devices[0]), ("fresh", devices[0]), ("other_gpu", devices[1])):
                label = f"{mode}_{'interleaved' if interleaved else 'standalone'}_{process}"
                args = [script, "fixed", "--root", root, "--mode", mode, "--label", label]
                if interleaved:
                    args.append("--interleaved")
                child(root, manifest, args, mode=mode, device=device, timeout=900)
    child(root, manifest, [script, "fixed", "--root", root, "--mode", "ordinary",
          "--label", "ordinary_localization", "--instrument"], mode="ordinary", device=devices[0], timeout=1200)


def forecast_worker(root, manifest, mode, label, seasonal=False):
    from src.models.mamba.v24_Ilya.evaluation import evaluate_v24_Ilya
    output = root / "forecasts" / f"{label}.json"
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    config = config_for(root, manifest, output, seasonal=seasonal, steps=40)
    result = evaluate_v24_Ilya(config)
    result["diagnostic_runtime"] = runtime_identity(mode)
    result["diagnostic_mode"] = mode
    write(output, result)


def forecasts(root, manifest):
    script = Path(manifest["source_root"]) / "scripts/experiments/run_v24_recovery_diagnostics.py"
    for mode, repeats, seasonal in (("ordinary", 3, True), ("deterministic", 2, False)):
        for index in range(repeats):
            args = [script, "forecast", "--root", root, "--mode", mode, "--label", f"{mode}_{index}"]
            if seasonal:
                args.append("--seasonal")
            child(root, manifest, args, mode=mode, timeout=5400 if mode == "deterministic" else 1800)


def compare_checkpoints(left, right):
    from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_training_checkpoint
    a, b = (load_v24_Ilya_training_checkpoint(p) for p in (left, right))
    exact = {k: getattr(a, k) == getattr(b, k) for k in
        ("completed_step", "training_cursor", "baseline_checkpoint_fingerprint", "anchor_manifest_fingerprint",
         "parameter_overlay_metadata")}
    exact["optimizer_count"] = int(a.optimizer_state.count) == int(b.optimizer_state.count) == a.completed_step
    exact["rng"] = tree_hash(a.rng_key) == tree_hash(b.rng_key)
    ac, bc = copy.deepcopy(a.resolved_training_config), copy.deepcopy(b.resolved_training_config)
    ac.pop("output", None)
    bc.pop("output", None)
    exact["config_except_output_directory"] = ac == bc
    return {"identity": exact, "integrity_passed": all(exact.values()),
        "numerical": {k: tree_compare(getattr(a, k), getattr(b, k)) for k in
                      ("residual_params", "optimizer_state", "residual_state")}}


def replay(root, manifest, arm):
    original = Path(manifest["mgda"])
    from scripts.experiments.frozen_runtime import verify_source
    source = original / "source"
    verify_source(source)
    m = read(original / "manifest.json")
    for mode in ("ordinary", "deterministic"):
        completed = {}
        for variant in ("continuous_a", "continuous_b", "restart"):
            directory = root / "replay" / arm / mode / variant
            config = copy.deepcopy(read(original / "configs" / f"{arm}.json"))
            config["optimizer"]["checkpoint_every"] = 3
            config["output"] = {"output_root": str(directory.parent), "run_name": variant}
            config_path = directory.parent / f"{variant}.json"
            write(config_path, config)
            checkpoint = directory / "checkpoints/checkpoint_step00000006.pkl"
            if checkpoint.exists():
                completed[variant] = checkpoint
                continue
            base = [source / "scripts/training/train_v24_Ilya.py", "--config", config_path]
            first = base + ["--init-from", m["initialization"]["path"], "--zero-state-on-init",
                            "--stop-after-steps", "3" if variant == "restart" else "6"]
            record = child(root, manifest, first, mode=mode, source=source, timeout=2400)
            if record["status"] != "complete":
                continue
            if variant == "restart":
                record = child(root, manifest, base + ["--resume", directory / "checkpoints/checkpoint_step00000003.pkl",
                               "--stop-after-steps", "6"], mode=mode, source=source, timeout=2400)
            if record["status"] == "complete" and checkpoint.exists():
                completed[variant] = checkpoint
        # Comparisons are CPU-only and do not claim that a diagnostic failure is a production failure.
        os.environ["JAX_PLATFORMS"] = "cpu"
        result = {"arm": arm, "mode": mode, "completed": {k: str(v) for k, v in completed.items()}}
        for name, candidate in (("repeat", "continuous_b"), ("restart", "restart")):
            if "continuous_a" in completed and candidate in completed:
                result[name] = compare_checkpoints(completed["continuous_a"], completed[candidate])
        write(root / "replay" / arm / mode / "comparison.json", result)


def summarize(root, manifest):
    import numpy as np
    fixed = {p.parent.name: read(p) for p in (root / "fixed").glob("*/result.json")}
    comparisons = []
    def compare_fixed(left, right, kind):
        a, b = fixed.get(left), fixed.get(right)
        if not a or not b or a["status"] != "complete" or b["status"] != "complete":
            return
        with np.load(root / "fixed" / left / "baseline_reference.npz") as av, np.load(root / "fixed" / right / "baseline_reference.npz") as bv:
            metrics = {k: array_comparison(bv[k], av[k]) for k in av.files}
        comparisons.append({"comparison": kind, "left": left, "right": right,
            "same_gpu": a["runtime"]["gpu_uuid"] == b["runtime"]["gpu_uuid"],
            "input_identity": all(a[k] == b[k] for k in ("input_hash", "parameter_hash", "state_hash", "rng_hash")),
            "leaves": metrics})
    for mode in MODES:
        for branch in ("standalone", "interleaved"):
            prefix = f"{mode}_{branch}_"
            first = fixed.get(prefix + "first")
            if not first or first["status"] != "complete":
                continue
            for other in ("fresh", "other_gpu"):
                candidate = fixed.get(prefix + other)
                if not candidate or candidate["status"] != "complete":
                    continue
                same_gpu = first["runtime"]["gpu_uuid"] == candidate["runtime"]["gpu_uuid"]
                identity = all(first[k] == candidate[k] for k in ("input_hash", "parameter_hash", "state_hash", "rng_hash"))
                with np.load(root / "fixed" / (prefix + "first") / "baseline_reference.npz") as a, np.load(root / "fixed" / (prefix + other) / "baseline_reference.npz") as b:
                    metrics = {k: array_comparison(b[k], a[k]) for k in a.files}
                comparisons.append({"mode": mode, "branch": branch, "comparison": other,
                    "same_gpu": same_gpu, "hardware_design_satisfied": same_gpu == (other == "fresh"),
                    "input_identity": identity, "same_hlo": first["hlo_sha256"] == candidate["hlo_sha256"], "leaves": metrics})
        compare_fixed(f"{mode}_standalone_first", f"{mode}_interleaved_first", "standalone_vs_interleaved")
        if mode != "ordinary":
            compare_fixed("ordinary_standalone_first", f"{mode}_standalone_first", "runtime_variant")
    forecast = {p.stem: read(p) for p in (root / "forecasts").glob("*.json") if ".partial" not in p.name}
    impacts = []
    for mode in ("ordinary", "deterministic"):
        baseline = forecast.get(f"{mode}_0")
        if baseline:
            for label, result in forecast.items():
                if not label.startswith(mode + "_") or result is baseline:
                    continue
                a,b=baseline["original_graphcast_loss"],result["original_graphcast_loss"]
                impacts.append({"mode": mode, "label": label,
                    "baseline_loss_change_pct": 100*(b["baseline_rollout"]/a["baseline_rollout"]-1),
                    "loss_reduction_change_pp": b["improvement_pct_rollout"]-a["improvement_pct_rollout"]})
    localization = []
    path = root / "fixed/ordinary_localization/aggregations.jsonl"
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        reference = {r["site"]: r for r in rows if r["phase"] == "reference"}
        for repeat in ("repeat_0", "repeat_1"):
            for row in (r for r in rows if r["phase"] == repeat):
                old = reference.get(row["site"])
                if old and row["output_hash"] != old["output_hash"]:
                    localization.append({"repeat": repeat, "first_different_site": row["site"],
                        "node_set": row["node_set"], "identical_aggregation_input": row["input_hash"] == old["input_hash"],
                        "identical_indices": row["indices_hash"] == old["indices_hash"]})
                    break
    result = {"status": "complete_with_available_evidence", "production_dependencies": [],
        "expected_fixed_cases": 24, "completed_fixed_cases": sum(v["status"] == "complete" and not v["instrumented"] for v in fixed.values()),
        "fixed": fixed, "cross_process_gpu": comparisons, "forecast_impact": impacts,
        "instrumented_first_divergence": localization,
        "replay": {str(p.relative_to(root)): read(p) for p in (root / "replay").glob("*/*/comparison.json")},
        "limitations": ["Missing cases remain incomplete; completion does not certify reproducibility.",
            "Instrumented aggregation callbacks can alter compilation and scheduling.",
            "2022 diagnostic dates do not estimate between-year or training-seed uncertainty."]}
    write(root / "summary.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "verify", "matrix", "fixed", "forecasts", "forecast", "replay", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--mgda", type=Path)
    parser.add_argument("--mode", choices=MODES, default="ordinary")
    parser.add_argument("--label", default="probe")
    parser.add_argument("--arm", choices=("mgda", "mse"), default="mgda")
    parser.add_argument("--interleaved", action="store_true")
    parser.add_argument("--instrument", action="store_true")
    parser.add_argument("--seasonal", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == "prepare":
        if not args.protocol or not args.mgda:
            parser.error("prepare requires --protocol and --mgda")
        prepare(root, args.workspace.resolve(), args.protocol.resolve(), args.mgda.resolve())
        return
    manifest = verify(root)
    if args.stage == "verify":
        print("Verified independent diagnostics", manifest["source_digest"])
    elif args.stage == "matrix":
        matrix(root, manifest)
    elif args.stage == "fixed":
        fixed_worker(root, manifest, args.label, args.mode, args.interleaved, args.instrument)
    elif args.stage == "forecasts":
        forecasts(root, manifest)
    elif args.stage == "forecast":
        forecast_worker(root, manifest, args.mode, args.label, args.seasonal)
    elif args.stage == "replay":
        replay(root, manifest, args.arm)
    else:
        summarize(root, manifest)


if __name__ == "__main__":
    main()
