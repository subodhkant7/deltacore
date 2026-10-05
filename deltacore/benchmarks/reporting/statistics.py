# ==============================================================================
# DeltaCore: deltacore/benchmarks/reporting/statistics.py
# Statistical aggregation and paired difference analysis for repeated runs.
# ==============================================================================

import math
from typing import Any

from deltacore.benchmarks.schemas.result import AggregateResult, RunResult, RunStatus


def _compute_stats(values: list[float]) -> dict[str, float]:
    """Compute basic sample summary statistics (mean, std, median, min, max)."""
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "count": 0.0,
        }

    n = len(values)
    mean_val = float(sum(values) / n)
    if n > 1:
        variance = sum((x - mean_val) ** 2 for x in values) / (n - 1)
        std_val = float(math.sqrt(max(0.0, variance)))
    else:
        std_val = 0.0

    sorted_vals = sorted(values)
    mid = n // 2
    if n % 2 == 1:
        median_val = float(sorted_vals[mid])
    else:
        median_val = float(0.5 * (sorted_vals[mid - 1] + sorted_vals[mid]))

    return {
        "mean": mean_val,
        "std": std_val,
        "median": median_val,
        "min": float(min(values)),
        "max": float(max(values)),
        "count": float(n),
    }


def aggregate_runs(runs: list[RunResult]) -> AggregateResult:
    """Aggregate a sequence of repeated benchmark runs across seeds.

    Explicitly counts successes, failures, and per-metric sample statistics.
    Never silently discards recovery failures or non-finite runs.

    Args:
        runs: List of RunResult objects (e.g. across multiple seeds).

    Returns:
        Structured AggregateResult.
    """
    if not runs:
        raise ValueError("Cannot aggregate empty list of runs.")

    experiment = runs[0].experiment
    model = runs[0].model

    num_runs = len(runs)
    num_success = sum(1 for r in runs if r.status == RunStatus.SUCCESS)
    num_failures = num_runs - num_success

    failure_reasons: dict[str, int] = {}
    for r in runs:
        if r.status != RunStatus.SUCCESS:
            reason = (
                r.error_message or r.status.value if r.error_message else r.status.value
            )
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

    # Extract all scalar metric keys
    metric_keys: set[str] = set()
    for r in runs:
        metric_keys.update(r.metrics.keys())

    metric_aggregates: dict[str, dict[str, float]] = {}
    for k in sorted(metric_keys):
        numeric_vals = []
        for r in runs:
            v = r.metrics.get(k)
            if (
                v is not None
                and isinstance(v, (int, float))
                and not isinstance(v, bool)
            ):
                if math.isfinite(v):
                    numeric_vals.append(float(v))
        if numeric_vals:
            metric_aggregates[k] = _compute_stats(numeric_vals)

    return AggregateResult(
        experiment=experiment,
        model=model,
        num_runs=num_runs,
        num_success=num_success,
        num_failures=num_failures,
        metric_aggregates=metric_aggregates,
        failure_reasons=failure_reasons,
    )


def compute_paired_differences(
    runs_a: list[RunResult],
    runs_b: list[RunResult],
    metric_name: str,
) -> dict[str, Any]:
    r"""Compute paired differences D_i = M_i^{(A)} - M_i^{(B)} across matched seeds.

    Args:
        runs_a: RunResults for model A.
        runs_b: RunResults for model B.
        metric_name: Metric key to evaluate.

    Returns:
        Dictionary containing sample statistics of the difference vector D_i.
    """
    map_a = {r.seed: r for r in runs_a}
    map_b = {r.seed: r for r in runs_b}

    common_seeds = sorted(set(map_a.keys()).intersection(map_b.keys()))
    diffs = []
    failed_pairs = 0

    for s in common_seeds:
        ra = map_a[s]
        rb = map_b[s]
        if ra.status != RunStatus.SUCCESS or rb.status != RunStatus.SUCCESS:
            failed_pairs += 1
            continue

        va = ra.metrics.get(metric_name)
        vb = rb.metrics.get(metric_name)
        if (
            va is not None
            and vb is not None
            and isinstance(va, (int, float))
            and isinstance(vb, (int, float))
        ):
            if math.isfinite(va) and math.isfinite(vb):
                diffs.append(float(va - vb))
            else:
                failed_pairs += 1
        else:
            failed_pairs += 1

    stats = _compute_stats(diffs)
    stats["total_pairs"] = len(common_seeds)
    stats["valid_pairs"] = len(diffs)
    stats["failed_pairs"] = failed_pairs
    return stats
