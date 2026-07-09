"""Backward-compatibility shim.

The legacy v1 teacher-only MZ module moved to
``src/models/mz/legacy/mz_v1_teacher.py``. This shim re-exports the same
public symbols so old inference scripts (e.g. ``infer_mz_save_tensors.py
--legacy-v1``) keep working.
"""

from .mz.legacy.mz_v1_teacher import (  # noqa: F401
    MZResidualConfig,
    MZResidualMamba,
)
