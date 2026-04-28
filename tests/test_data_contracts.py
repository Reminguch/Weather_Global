from __future__ import annotations

import numpy as np
import pytest

from src.data_operations.contracts import CanonicalBatch, validate_canonical_batch


def test_validate_canonical_batch_accepts_minimal_valid_batch() -> None:
    batch = CanonicalBatch(
        inputs={"x": np.zeros((2, 3), dtype=np.float32)},
        targets={"y": np.ones((2, 3), dtype=np.float32)},
        forcings={},
        coords={},
        metadata={},
    )
    validate_canonical_batch(batch)


def test_validate_canonical_batch_rejects_empty_inputs() -> None:
    batch = CanonicalBatch(
        inputs={},
        targets={"y": np.ones((2, 3), dtype=np.float32)},
        forcings={},
        coords={},
        metadata={},
    )
    with pytest.raises(ValueError, match="inputs"):
        validate_canonical_batch(batch)
