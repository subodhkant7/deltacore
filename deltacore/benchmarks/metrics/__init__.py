# ==============================================================================
# DeltaCore: deltacore/benchmarks/metrics/__init__.py
# Standardized metric calculation functions.
# ==============================================================================

from deltacore.benchmarks.metrics.accuracy import (
    compute_adaptation_gain,
    compute_final_error,
    compute_mean_error,
    compute_relative_error,
)
from deltacore.benchmarks.metrics.energy import (
    compute_max_normalized_step,
    compute_min_stability_margin,
    compute_state_growth_ratio,
    compute_step_energy,
    compute_update_energy,
)
from deltacore.benchmarks.metrics.recovery import (
    compute_first_passage_recovery,
    compute_sustained_recovery,
)

__all__ = [
    "compute_final_error",
    "compute_mean_error",
    "compute_relative_error",
    "compute_first_passage_recovery",
    "compute_sustained_recovery",
    "compute_adaptation_gain",
    "compute_state_growth_ratio",
    "compute_update_energy",
    "compute_step_energy",
    "compute_min_stability_margin",
    "compute_max_normalized_step",
]
