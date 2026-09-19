#!/usr/bin/env python3
"""Native v24 training with opt-in, host-side fresh-initialization diagnostics.

Diagnostic parameter snapshots are NOT training checkpoints: they contain no
optimizer, recurrent state, data cursor, or RNG state. Exact resume continues to
use the native checkpoints. No training equations or initialization are changed.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import pickle
import re
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EARLY_STEPS = frozenset((1, 10, 100, 500))
SNAPSHOT_FORMAT = "v24_fresh_long_parameter_diagnostics_v1"


def parameter_group(module):
    if module == "temporal_residual_head":
        return "correction_head"
    if module.startswith("mesh_interleaved_temporal_"):
        return module.split("/", 1)[0]
    if module.startswith("grid2mesh_gnn/"):
        return "grid_to_mesh"
    if module.startswith("mesh2grid_gnn/"):
        return "mesh_to_grid"
    if module.startswith("mesh_gnn/"):
        match = re.search(r"processor_(?:edges|nodes)_(\d+)_", module)
        return f"mesh_processor_{match[1]}" if match else "mesh_edge_embedding"
    return "other"


def host_params(params):
    return {module: {name: np.array(value, copy=True) for name, value in values.items()}
            for module, values in params.items()}


def adam_moment(state):
    """Find the single Adam first-moment tree; grouped optimizers are ambiguous."""
    found = []

    def visit(value):
        if hasattr(value, "mu") and hasattr(value, "nu"):
            found.append(value.mu)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)

    visit(state)
    return host_params(found[0]) if len(found) == 1 else None


def parameter_metrics(initial, before, after, *, mu_before=None, mu_after=None, beta1=.9):
    """Measure actual one-update displacement and total movement from step zero."""
    sums = {}
    tensors = {}
    gradient_available = mu_before is not None and mu_after is not None
    for module, values in after.items():
        for name, value in values.items():
            current = np.asarray(value, dtype=np.float64)
            start = np.asarray(initial[module][name], dtype=np.float64)
            previous = np.asarray(before[module][name], dtype=np.float64)
            raw = dict(initial_sq=float(np.sum(start ** 2)),
                       before_sq=float(np.sum(previous ** 2)),
                       current_sq=float(np.sum(current ** 2)),
                       displacement_sq=float(np.sum((current - start) ** 2)),
                       update_sq=float(np.sum((current - previous) ** 2)),
                       parameter_count=int(current.size))
            if gradient_available:
                gradient = (np.asarray(mu_after[module][name], dtype=np.float64)
                            - beta1 * np.asarray(mu_before[module][name], dtype=np.float64)) / (1 - beta1)
                raw["gradient_sq"] = float(np.sum(gradient ** 2))
            category = ("normalization" if "layer_norm" in module or name in ("scale", "offset")
                        else "matrices" if current.ndim >= 2 else "vectors")
            for group in (parameter_group(module), "all_residual"):
                for subset in ("all", category):
                    total = sums.setdefault(group, {}).setdefault(subset, {})
                    for key, number in raw.items():
                        total[key] = total.get(key, 0) + number
            tensors[f"{module}/{name}"] = raw

    def finish(raw):
        result = {key.removesuffix("_sq") + "_norm": float(np.sqrt(value))
                  for key, value in raw.items() if key.endswith("_sq")}
        result["parameter_count"] = raw["parameter_count"]
        result["displacement_over_initial"] = (
            result["displacement_norm"] / result["initial_norm"] if result["initial_norm"] else None)
        result["update_over_before"] = (
            result["update_norm"] / result["before_norm"] if result["before_norm"] else None)
        if "gradient_norm" in result:
            result["post_clip_gradient_norm_estimate"] = result.pop("gradient_norm")
        return result

    return dict(groups={group: {subset: finish(raw) for subset, raw in subsets.items()}
                        for group, subsets in sums.items()},
                tensors={name: finish(raw) for name, raw in tensors.items()},
                gradient_estimate=("(mu_after - beta1 * mu_before) / (1 - beta1); "
                                   "Adam input after clipping, before weight decay; "
                                   "subject to floating-point cancellation" if gradient_available
                                   else "unavailable: no unique Adam first-moment tree"))


class TrainingDiagnostics:
    def __init__(self, config, *, completed_step=0):
        self.config = config
        self.directory = config.run_dir / "diagnostics"
        self.completed_step = completed_step
        self.initial = None
        self.resume_recovered = completed_step == 0
        if completed_step:
            path = self.snapshot_path(0)
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            if payload.get("diagnostic_format") != SNAPSHOT_FORMAT or payload.get("completed_step") != 0:
                raise ValueError(f"Invalid initial-parameter diagnostics: {path}")
            self.initial = payload["residual_params"]

    def recover_resume_history(self):
        """Archive diagnostics from updates absent from the resume checkpoint."""
        if self.resume_recovered:
            return
        metrics_path = self.directory / "parameter_metrics.jsonl"
        lines = metrics_path.read_text().splitlines(keepends=True) if metrics_path.is_file() else []
        kept = []
        for index, line in enumerate(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise ValueError(f"Malformed diagnostic record at line {index + 1}")
                break
            if int(record["step"]) <= self.completed_step:
                kept.append(line.rstrip("\n") + "\n")
        future = [path for path in self.directory.glob("parameters_step????????.pkl")
                  if int(path.stem.removeprefix("parameters_step")) > self.completed_step]
        if kept != lines or future:
            archive = self.directory / "resume_archives" / f"after_step{self.completed_step:08d}_{time.time_ns()}"
            archive.mkdir(parents=True, exist_ok=False)
            if kept != lines:
                shutil.copy2(metrics_path, archive / metrics_path.name)
                temporary = metrics_path.with_suffix(".jsonl.resume_tmp")
                temporary.write_text("".join(kept), encoding="utf-8")
                temporary.replace(metrics_path)
            for path in future:
                path.replace(archive / path.name)
            (archive / "recovery.json").write_text(json.dumps(dict(
                resumed_step=self.completed_step, archived_future_snapshots=[path.name for path in future],
                removed_metric_records=len(lines) - len(kept)), indent=2) + "\n")
            print(f"[fresh_long] archived diagnostics beyond resume checkpoint: {archive}", flush=True)
        self.resume_recovered = True

    def snapshot_path(self, step):
        return self.directory / f"parameters_step{step:08d}.pkl"

    def snapshot(self, step, params):
        from src.models.mamba.v24_Ilya.checkpoint import atomic_pickle_dump
        self.directory.mkdir(parents=True, exist_ok=True)
        atomic_pickle_dump(dict(diagnostic_format=SNAPSHOT_FORMAT,
                                completed_step=step, resumable=False,
                                description="Residual parameters only; use native training checkpoints for exact resume.",
                                residual_params=params), self.snapshot_path(step))

    def append(self, step, metrics, duration):
        record = dict(step=step, update_interval=[max(0, step - 1), step],
                      reference_step=0, diagnostics_seconds=duration, **metrics)
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / "parameter_metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        print(f"[fresh_long] parameter diagnostics step={step} time={duration:.2f}s", flush=True)

    def wrap(self, train_step):
        # The native runner reaches this hook only after validating exact resume.
        self.recover_resume_history()

        def instrumented(params, residual_state, optimizer_state, *args, **kwargs):
            started = time.monotonic()
            if self.initial is None:
                if self.completed_step:
                    raise RuntimeError("Initial parameter reference missing on resume")
                self.initial = host_params(params)
                self.snapshot(0, self.initial)
                self.append(0, parameter_metrics(self.initial, self.initial, self.initial),
                            time.monotonic() - started)
            step = self.completed_step + 1
            selected = (step in EARLY_STEPS or step % self.config.checkpoint_every == 0
                        or step == self.config.max_steps)
            before = host_params(params) if selected else None
            mu_before = adam_moment(optimizer_state) if selected else None
            before_seconds = time.monotonic() - started
            result = train_step(params, residual_state, optimizer_state, *args, **kwargs)
            self.completed_step = step
            if selected:
                started = time.monotonic()
                after = host_params(result[0])
                mu_after = adam_moment(result[2])
                metrics = parameter_metrics(self.initial, before, after, mu_before=mu_before,
                                            mu_after=mu_after, beta1=self.config.adam_beta1)
                if step in EARLY_STEPS:
                    self.snapshot(step, after)
                self.append(step, metrics, before_seconds + time.monotonic() - started)
            return result

        return instrumented


@contextmanager
def instrument_runner(runner, diagnostics):
    original = runner.make_train_step

    def make_train_step(**kwargs):
        return diagnostics.wrap(original(**kwargs))

    runner.make_train_step = make_train_step
    try:
        yield
    finally:
        runner.make_train_step = original


def main(argv=None):
    from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_training_checkpoint
    from src.models.mamba.v24_Ilya.training.config import parse_cli
    from src.models.mamba.v24_Ilya.training import runner
    invocation = parse_cli(argv)
    config = invocation.config
    if (config.distributed.mode != "single" or config.distributed.global_batch_size != 1
            or config.architecture.residual_initialization != "fresh"
            or invocation.init_from is not None or invocation.baseline_validation_only
            or invocation.validation_compare is not None):
        raise ValueError("Fresh-long diagnostics require single-device batch1 fresh residual training")
    if invocation.dry_run:
        return runner.run_training(invocation)
    completed_step = 0
    if invocation.resume is not None:
        checkpoint = load_v24_Ilya_training_checkpoint(invocation.resume)
        completed_step = checkpoint.completed_step
        del checkpoint
    diagnostics = TrainingDiagnostics(config, completed_step=completed_step)
    with instrument_runner(runner, diagnostics):
        return runner.run_training(invocation)


if __name__ == "__main__":
    main()
