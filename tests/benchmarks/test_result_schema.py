# ==============================================================================
# DeltaCore: tests/benchmarks/test_result_schema.py
# Unit tests for RunResult and AggregateResult schemas and JSON serialization.
# ==============================================================================

import json
from pathlib import Path

from deltacore.benchmarks.schemas.result import AggregateResult, RunResult, RunStatus


def test_run_result_creation_and_status() -> None:
    """RunResult should properly store categorical status and metrics."""
    res = RunResult(
        experiment="test_task",
        seed=42,
        model="FixedDelta",
        status=RunStatus.SUCCESS,
        config={"sequence_length": 64},
        metrics={"final_error": 0.123, "state_norm": 4.56},
        timing_ms={"wall_clock_ms": 12.5},
        device="cpu",
        dtype="float32",
    )
    assert res.status == RunStatus.SUCCESS
    assert res.metrics["final_error"] == 0.123
    assert res.error_step is None


def test_run_result_json_roundtrip(tmp_path: Path) -> None:
    """RunResult should serialize to JSON and restore faithfully."""
    res = RunResult(
        experiment="distribution_shift",
        seed=10,
        model="SafeSelfReferential",
        status=RunStatus.SUCCESS,
        config={"key_dim": 16, "value_dim": 16},
        metrics={"recovery_steps": 8, "final_error": 0.05},
        timing_ms={"wall_clock_ms": 25.4},
        device="cpu",
        dtype="float32",
        error_message=None,
        error_step=None,
    )
    json_path = tmp_path / "test_run.json"
    res.save(json_path)

    loaded = RunResult.load(json_path)
    assert loaded.experiment == res.experiment
    assert loaded.seed == res.seed
    assert loaded.model == res.model
    assert loaded.status == res.status
    assert loaded.metrics == res.metrics


def test_no_tensors_in_serialized_result() -> None:
    """Ensure raw PyTorch tensors are not placed in the JSON result."""
    res = RunResult(
        experiment="test_exp",
        seed=0,
        model="fixed",
        status=RunStatus.SUCCESS,
        config={},
        metrics={"float_val": 1.5, "int_val": 10, "bool_val": True},
        timing_ms={},
        device="cpu",
        dtype="float32",
    )
    d = res.to_dict()
    # json.dumps must succeed without custom tensor encoder
    serialized = json.dumps(d)
    assert "float_val" in serialized


def test_aggregate_result_json_roundtrip(tmp_path: Path) -> None:
    """AggregateResult should serialize and load from disk cleanly."""
    agg = AggregateResult(
        experiment="stationary_recall",
        model="FixedDelta",
        num_runs=5,
        num_success=5,
        num_failures=0,
        metric_aggregates={
            "final_error": {
                "mean": 0.1,
                "std": 0.02,
                "median": 0.1,
                "min": 0.08,
                "max": 0.12,
                "count": 5,
            }
        },
        failure_reasons={},
    )
    agg_path = tmp_path / "aggregate.json"
    agg.save(agg_path)

    loaded = AggregateResult.load(agg_path)
    assert loaded.experiment == "stationary_recall"
    assert loaded.num_runs == 5
    assert loaded.num_success == 5
    assert "final_error" in loaded.metric_aggregates
