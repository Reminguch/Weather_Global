"""Exactly twenty live predictions with SG physical feedback and recurrent BPTT."""
from __future__ import annotations

from pathlib import Path
import time
import jax
import numpy as np
from .data import STEP, seasonal_order, eligible_origins
from .native_state import stop
from .model import zero_memory
from .checkpoint import load_checkpoint, save_checkpoint, transfer_parent, reconcile_metrics
from .io import append_jsonl, write_json


def burn_in(branch, params, memory, rng, backbone, adapter, normalization, known, store, origin):
    """Four observed frames at -24,-18,-12,-6h; t0 is deliberately absent."""
    for lag in (24, 18, 12, 6):
        t = np.datetime64(origin, "h") - np.timedelta64(lag, "h")
        inputs, forcing = store.inputs_and_forcing(backbone.model, t)
        observed = backbone.encode(inputs, forcing)
        rng, key = jax.random.split(rng)
        _, memory = branch.apply(params, memory, key, normalization.inputs(adapter.features(observed)), known(observed, forcing))
    return stop(memory), rng


def episode_gradients(trainer, params, initial_memory, rng, *, backbone, store, known, origin,
                      warm=False, rollout_steps=20):
    if rollout_steps != 20:
        raise ValueError("Fine-tuning requires exactly twenty predictions")
    memory = zero_memory(initial_memory)
    if warm:
        memory, rng = burn_in(trainer.branch, params, memory, rng, backbone, trainer.adapter,
                              trainer.normalization, known, store, origin)
    inputs, forcing = store.inputs_and_forcing(backbone.model, origin)
    physical = backbone.encode(inputs, forcing)  # Exactly one forecast initialization.
    initial_time = physical
    tape, losses, valid_times = [], [], []
    for lead in range(1, rollout_steps + 1):
        physical = stop(physical)
        baseline = stop(backbone.advance_6h(physical, forcing))
        target_time = np.datetime64(origin, "h") + lead * STEP
        target = store.frame(target_time)
        record = trainer.record(physical, baseline, forcing, target, known(physical, forcing))
        rng, key = jax.random.split(rng)
        tape.append((jax.device_get(memory), jax.device_get(key), jax.device_get(record)))
        value, memory, corrected = trainer.forward(params, memory, key, *record)
        physical = stop(corrected)
        backbone.assert_time(initial_time, physical, lead)
        losses.append(float(value))
        valid_times.append(lead * 6)
    if valid_times != list(range(6, 121, 6)):
        raise AssertionError("Forecast valid times differ from 6..120h")
    if not np.isfinite(losses).all():
        raise FloatingPointError(f"Nonfinite forecast at origin {origin}")
    return {"loss": float(np.mean(losses)), "lead_losses": losses, "memory": stop(memory),
            "rng": rng, "gradients": trainer.reverse(params, tape), "physical": physical,
            "valid_hours": valid_times}


def run_finetune(runtime, config, output, identities, parent, *, resume=None, validate=None):
    output = Path(output)
    trainer, store = runtime.trainer, runtime.store
    if resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError("Nonempty fine-tuning output requires explicit --resume")
    output.mkdir(parents=True, exist_ok=True)
    params, parent_info = transfer_parent(parent, {k: identities[k] for k in
        ("architecture", "backbone", "native_schema", "normalization", "dataset")})
    identities = dict(identities, parent=parent_info["sha256"])
    optimizer = trainer.optimizer.init(params)
    memory, rng, start = zero_memory(runtime.memory), jax.random.PRNGKey(config.seed), 0
    if resume:
        state = load_checkpoint(resume, stage="finetune", identities=identities)
        params, optimizer, memory, rng = (state[k] for k in ("params", "optimizer", "memory", "rng"))
        start = state["cursor"]["update"]
        reconcile_metrics(output / "metrics.jsonl", start)
        if validate and start % config.finetune_validate_every == 0:
            validate(params, Path(resume), start)
    write_json(output / "parent.json", parent_info, immutable=True)
    origins = seasonal_order(eligible_origins(store.times, "train", 20), seed=config.seed)
    if not origins:
        raise ValueError("No eligible training episodes")
    write_json(output / "episode_order.json", {"origins": origins, "alternate_cold_warm": True}, immutable=True)
    for update in range(start, config.finetune_updates):
        started = time.perf_counter()
        origin, warm = origins[update % len(origins)], bool(update % 2)
        result = episode_gradients(trainer, params, memory, rng, backbone=runtime.backbone,
                                   store=store, known=runtime.known, origin=origin, warm=warm)
        params, optimizer, norm = trainer.checked_update(params, optimizer, result["gradients"])
        rng = result["rng"]
        memory = zero_memory(runtime.memory)  # Episodes are independent.
        append_jsonl(output / "metrics.jsonl", {"update": update + 1, "origin": origin, "warm": warm,
                     "loss": result["loss"], "gradient_norm": float(norm), "seconds": time.perf_counter() - started})
        if (update + 1) % config.finetune_validate_every == 0:
            checkpoint = output / f"checkpoint_{update + 1:06d}.pkl"
            save_checkpoint(checkpoint, stage="finetune", identities=identities, params=params,
                            optimizer=optimizer, memory=memory, rng=rng,
                            cursor={"update": update + 1, "episode_boundary": True})
            if validate:
                validate(params, checkpoint, update + 1)
    return params
