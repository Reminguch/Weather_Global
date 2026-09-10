"""Instrument a one-segment baseline/checkpoint validation smoke run."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.mamba.v24_Ilya.training import runner
from src.models.mamba.v24_Ilya.training.config import parse_cli

import jax
import numpy as np


def main() -> None:
    invocation = parse_cli()
    config = invocation.config
    if invocation.validation_compare is None:
        raise ValueError("Smoke check requires --validation-compare CHECKPOINT")
    if config.validation.num_segments != 1 or config.bptt_steps != 24:
        raise ValueError("Smoke check expects one segment and 24-step chunks")
    devices = jax.local_devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError(f"Smoke check requires one allocated GPU, got {devices}")

    chunks = []
    original_factory = runner.make_validation_step

    def instrumented_factory(**kwargs):
        validation_step = original_factory(**kwargs)

        def checked_step(*args):
            started = time.monotonic()
            output = validation_step(*args)
            loss, _state, components = output
            loss = float(jax.device_get(loss))
            components = np.asarray(jax.device_get(components))
            if not math.isfinite(loss) or not np.all(np.isfinite(components)):
                raise RuntimeError("Non-finite validation loss")
            np.testing.assert_allclose(
                loss, np.dot(components, config.normalized_supervised_weights),
                rtol=1e-5, atol=1e-6,
            )
            record = {
                "chunk": len(chunks) + 1,
                "loss": loss,
                "seconds": time.monotonic() - started,
                "device_memory": devices[0].memory_stats(),
            }
            chunks.append(record)
            print("[validation-smoke] " + json.dumps(record), flush=True)
            return output

        return checked_step

    runner.make_validation_step = instrumented_factory
    try:
        result_path = runner.run_training(invocation)
    finally:
        runner.make_validation_step = original_factory
    result = json.loads(result_path.read_text())
    expected_chunks = config.segment_steps // config.bptt_steps
    for branch in ("baseline", "checkpoint"):
        record = result[branch]
        assert record["num_chunks"] == expected_chunks
        assert record["num_anchors"] == config.segment_steps
        assert len(record["loss_by_horizon"]) == 24
        assert math.isfinite(record["loss"])
    assert len(chunks) == 2 * expected_chunks
    summary = {
        "status": "passed",
        "scope": "standalone one-GPU validation; no DP training allocations",
        "source_checkpoint": str(invocation.validation_compare),
        "comparison": str(result_path),
        "chunks": chunks,
    }
    summary_path = config.run_dir / "smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[validation-smoke] PASSED summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
