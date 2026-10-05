"""Diagnostics, state telemetry, and dynamic observation hooks.

This module hosts non-invasive observers for state norms, step-size trajectories,
update magnitudes, numerical health tracking, self-referential coupling telemetry,
stability metrics (normalized steps, contraction bounds, recovery latency),
and chunked scan observability metrics.
"""

from deltacore.diagnostics.adaptation import (
    check_numerical_health,
    compute_adaptation_gain,
    compute_error_norm,
    compute_memory_norm,
    compute_recovery_steps,
    compute_step_size_stats,
    compute_update_norm,
)
from deltacore.diagnostics.scans import (
    ScanDiagnostics,
    extract_scan_diagnostics,
)
from deltacore.diagnostics.self_reference import (
    compute_coupling_correlation,
    compute_step_size_changes,
    measure_step_telemetry,
)
from deltacore.diagnostics.stability import (
    compute_first_passage_recovery_steps,
    compute_normalized_dynamics_rate,
    compute_normalized_step,
    compute_stability_margin,
    compute_sustained_recovery_steps,
    extract_stability_telemetry,
)

__all__: list[str] = [
    "compute_step_size_stats",
    "compute_memory_norm",
    "compute_update_norm",
    "compute_error_norm",
    "check_numerical_health",
    "compute_adaptation_gain",
    "compute_recovery_steps",
    "compute_step_size_changes",
    "compute_coupling_correlation",
    "measure_step_telemetry",
    "compute_first_passage_recovery_steps",
    "compute_sustained_recovery_steps",
    "compute_normalized_step",
    "compute_normalized_dynamics_rate",
    "compute_stability_margin",
    "extract_stability_telemetry",
    "ScanDiagnostics",
    "extract_scan_diagnostics",
]
