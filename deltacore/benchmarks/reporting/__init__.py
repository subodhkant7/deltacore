# ==============================================================================
# DeltaCore: deltacore/benchmarks/reporting/__init__.py
# Reporting, statistical aggregation, and visualization utilities.
# ==============================================================================

from deltacore.benchmarks.reporting.report import generate_markdown_report
from deltacore.benchmarks.reporting.statistics import (
    aggregate_runs,
    compute_paired_differences,
)
from deltacore.benchmarks.reporting.visualization import (
    plot_error_vs_time,
    plot_key_correlation_vs_error,
    plot_stability_margin_distribution,
    plot_state_norm_vs_time,
)

__all__ = [
    "aggregate_runs",
    "compute_paired_differences",
    "generate_markdown_report",
    "plot_error_vs_time",
    "plot_state_norm_vs_time",
    "plot_key_correlation_vs_error",
    "plot_stability_margin_distribution",
]
