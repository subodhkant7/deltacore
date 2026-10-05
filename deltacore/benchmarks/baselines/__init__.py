# ==============================================================================
# DeltaCore: deltacore/benchmarks/baselines/__init__.py
# Standardized baseline wrappers for benchmark comparison.
# ==============================================================================

from deltacore.benchmarks.baselines.base import (
    BaseBaseline,
    BaselineStepResult,
    BaselineTrajectoryResult,
)
from deltacore.benchmarks.baselines.factory import get_baseline
from deltacore.benchmarks.baselines.wrappers import (
    AdaptiveDeltaBaseline,
    FixedDeltaBaseline,
    FrozenBaseline,
    SafeSelfReferentialBaseline,
    SelfReferentialBaseline,
)

__all__ = [
    "BaseBaseline",
    "BaselineStepResult",
    "BaselineTrajectoryResult",
    "FrozenBaseline",
    "FixedDeltaBaseline",
    "AdaptiveDeltaBaseline",
    "SelfReferentialBaseline",
    "SafeSelfReferentialBaseline",
    "get_baseline",
]
