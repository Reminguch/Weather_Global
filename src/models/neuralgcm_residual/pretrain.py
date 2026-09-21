"""Chronological 96-record segments, carried 24-record decoded-loss TBPTT."""
from __future__ import annotations

from pathlib import Path
import time
import jax
import numpy as np
from .data import complete_segments
from .checkpoint import load_checkpoint, save_checkpoint, reconcile_metrics
from .model import zero_memory
from .io import append_jsonl, write_json


def run_pretrain(runtime, reader, config, output, identities, *, resume=None, validate=None):
    output = Path(output)
    if resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError("Nonempty pretraining output requires explicit --resume")
    output.mkdir(parents=True, exist_ok=True)
    trainer = runtime.trainer
    params, memory = runtime.params, zero_memory(runtime.memory)
    optimizer = trainer.optimizer.init(params)
    rng = jax.random.PRNGKey(config.seed)
    cursor = {"epoch": 0, "segment": 0, "chunk": 0, "update": 0, "processed_timestamps": 0}
    if resume:
        saved = load_checkpoint(resume, stage="pretrain", identities=identities)
        params, optimizer, memory, rng, cursor = (saved[k] for k in ("params", "optimizer", "memory", "rng", "cursor"))
        reconcile_metrics(output / "metrics.jsonl", cursor["update"])
        if validate and cursor["chunk"] == 0 and cursor["segment"] == 0:
            validate(params, Path(resume), cursor["epoch"], ar=cursor["epoch"] % 5 == 0)
    segments, omitted = complete_segments(reader.times, config.segment_steps)
    if not segments:
        raise ValueError("No complete 96-record chronological training segment")
    write_json(output / "sampling.json", {"segments": len(segments), "records_per_epoch": len(segments) * 96,
               "omitted_origins": [reader.times[i] for i in omitted], "shuffle": "segments_only"}, immutable=True)
    if validate and cursor["update"] == 0 and resume is None:
        initial = output / "checkpoint_initial.pkl"
        save_checkpoint(initial, stage="pretrain", identities=identities, params=params, optimizer=optimizer,
                        memory=memory, rng=rng, cursor=cursor)
        validate(params, initial, 0, ar=True)
    for epoch in range(cursor["epoch"], config.pretrain_epochs):
        order = np.random.default_rng(config.seed + epoch).permutation(len(segments))
        segment_start = cursor["segment"] if epoch == cursor["epoch"] else 0
        for position in range(segment_start, len(order)):
            indices = segments[int(order[position])]
            first_chunk = cursor["chunk"] if epoch == cursor["epoch"] and position == cursor["segment"] else 0
            if first_chunk == 0:
                memory = zero_memory(runtime.memory)
            for chunk in range(first_chunk, config.segment_steps // config.bptt_steps):
                started = time.perf_counter()
                records = []
                for i in indices[chunk * config.bptt_steps:(chunk + 1) * config.bptt_steps]:
                    record = reader[i]
                    records.append(trainer.record(record["origin_state"], record["baseline_state"],
                        record["forcing"], record["target"], runtime.known(record["origin_state"], record["forcing"])))
                loss, memory, rng, gradient = trainer.cached_gradients(params, memory, rng, records)
                if not np.isfinite(loss):
                    raise FloatingPointError("Nonfinite cached decoded loss")
                params, optimizer, norm = trainer.checked_update(params, optimizer, gradient)
                cursor = {"epoch": epoch, "segment": position, "chunk": chunk + 1,
                          "update": cursor["update"] + 1,
                          "processed_timestamps": cursor["processed_timestamps"] + len(records)}
                if cursor["chunk"] == 4:
                    cursor.update(segment=position + 1, chunk=0)
                if cursor["segment"] == len(order):
                    cursor.update(epoch=epoch + 1, segment=0)
                append_jsonl(output / "metrics.jsonl", {**cursor, "loss": loss, "gradient_norm": float(norm),
                                                       "seconds": time.perf_counter() - started})
                # Chunk checkpoints retain carried memory and the next exact cursor.
                if cursor["update"] % 200 == 0 and cursor["epoch"] == epoch:
                    save_checkpoint(output / f"checkpoint_update_{cursor['update']:08d}.pkl", stage="pretrain",
                        identities=identities, params=params, optimizer=optimizer, memory=memory, rng=rng, cursor=cursor)
        checkpoint = output / f"checkpoint_pass_{epoch + 1:02d}.pkl"
        save_checkpoint(checkpoint, stage="pretrain", identities=identities, params=params, optimizer=optimizer,
                        memory=memory, rng=rng, cursor=cursor)
        if validate:
            validate(params, checkpoint, epoch + 1, ar=(epoch + 1) % 5 == 0)
    return params
