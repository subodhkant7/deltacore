# ==============================================================================
# DeltaCore: tests/benchmarks/test_aggregation.py
# Unit tests for multi-seed aggregation and paired difference computation.
# ==============================================================================


from deltacore.benchmarks.reporting.statistics import (
    aggregate_runs,
    compute_paired_differences,
)
from deltacore.benchmarks.schemas.result import RunResult, RunStatus


def _make_dummy_run(
    seed: int, model: str, final_error: float, status: RunStatus = RunStatus.SUCCESS
) -> RunResult:
    return RunResult(
        experiment="test_task",
        seed=seed,
        model=model,
        status=status,
        config={"seed": seed},
        metrics={"final_error": final_error, "dummy_metric": final_error * 2.0},
        timing_ms={"wall_clock_ms": 10.0},
        device="cpu",
        dtype="float32",
        error_message="Diverged" if status != RunStatus.SUCCESS else None,
    )


def test_aggregation_mean_and_variance() -> None:
    """Verify sample mean, std, median, min, max across runs."""
    runs = [
        _make_dummy_run(0, "Fixed", 1.0),
        _make_dummy_run(1, "Fixed", 2.0),
        _make_dummy_run(2, "Fixed", 3.0),
        _make_dummy_run(3, "Fixed", 4.0),
        _make_dummy_run(4, "Fixed", 5.0),
    ]
    agg = aggregate_runs(runs)

    assert agg.num_runs == 5
    assert agg.num_success == 5
    assert agg.num_failures == 0

    stats = agg.metric_aggregates["final_error"]
    assert abs(stats["mean"] - 3.0) < 1e-6
    assert abs(stats["median"] - 3.0) < 1e-6
    assert abs(stats["min"] - 1.0) < 1e-6
    assert abs(stats["max"] - 5.0) < 1e-6
    # Sample std of [1, 2, 3, 4, 5] is sqrt(2.5) ~= 1.5811
    assert abs(stats["std"] - (2.5**0.5)) < 1e-4


def test_aggregation_with_failures() -> None:
    """Failures are counted and not silently converted into infinite numbers."""
    runs = [
        _make_dummy_run(0, "ModelA", 1.0, RunStatus.SUCCESS),
        _make_dummy_run(1, "ModelA", 2.0, RunStatus.SUCCESS),
        _make_dummy_run(2, "ModelA", 0.0, RunStatus.NUMERICAL_FAILURE),
    ]
    agg = aggregate_runs(runs)

    assert agg.num_runs == 3
    assert agg.num_success == 2
    assert agg.num_failures == 1
    assert "Diverged" in agg.failure_reasons


def test_paired_differences() -> None:
    """Verify paired difference calculation D_i = M_i^{(A)} - M_i^{(B)}."""
    runs_a = [
        _make_dummy_run(0, "ModelA", 1.5),
        _make_dummy_run(1, "ModelA", 2.5),
        _make_dummy_run(2, "ModelA", 3.5),
    ]
    runs_b = [
        _make_dummy_run(0, "ModelB", 1.0),
        _make_dummy_run(1, "ModelB", 2.0),
        _make_dummy_run(2, "ModelB", 3.0),
    ]
    # D_i = A - B = [0.5, 0.5, 0.5]
    diff = compute_paired_differences(runs_a, runs_b, metric_name="final_error")
    assert abs(diff["mean"] - 0.5) < 1e-6
    assert abs(diff["std"] - 0.0) < 1e-6
    assert diff["valid_pairs"] == 3
    assert diff["failed_pairs"] == 0
