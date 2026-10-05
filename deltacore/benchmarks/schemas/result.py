# ==============================================================================
# DeltaCore: deltacore/benchmarks/schemas/result.py
# Machine-readable result schema and serialization for benchmark artifacts.
# ==============================================================================

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class RunStatus(str, Enum):
    """Categorical execution status of a single benchmark run."""

    SUCCESS = "SUCCESS"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"
    INVALID_CONFIGURATION = "INVALID_CONFIGURATION"
    TIMEOUT = "TIMEOUT"
    UNSUPPORTED_DEVICE = "UNSUPPORTED_DEVICE"


@dataclass
class RunResult:
    """Complete machine-readable result of a single benchmark run.

    Tensors are NEVER stored in this primary JSON artifact. Trajectories
    and high-dimensional states are saved to dedicated array files if requested.

    Attributes:
        experiment: Name of the task or benchmark.
        seed: Random seed used for generation.
        model: Identifier of the evaluated model/baseline.
        status: Categorical outcome (SUCCESS, NUMERICAL_FAILURE, etc.).
        config: Serialized BenchmarkConfig dictionary.
        metrics: Dictionary of scalar metrics (error, norms, recovery steps, etc.).
        timing_ms: Wall-clock timing breakdown in milliseconds.
        device: Device string on which the run was executed.
        dtype: Dtype string ('float32' or 'float64').
        error_message: Detailed diagnostic error message if run failed.
        error_step: Step index at which numerical failure occurred, if applicable.
        timestamp: ISO 8601 UTC timestamp of execution.
        extra_metadata: Optional dictionary for task-specific diagnostic tags.
    """

    experiment: str
    seed: int
    model: str
    status: RunStatus
    config: dict[str, Any]
    metrics: dict[str, Any]
    timing_ms: dict[str, float]
    device: str
    dtype: str
    error_message: str | None = None
    error_step: int | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to JSON-serializable dictionary."""
        d = asdict(self)
        d["status"] = self.status.value

        def _sanitize(val: Any) -> Any:
            import torch

            if isinstance(val, torch.Tensor):
                return val.tolist()
            if isinstance(val, dict):
                return {k: _sanitize(v) for k, v in val.items()}
            if isinstance(val, (list, tuple)):
                return [_sanitize(v) for v in val]
            return val

        d["extra_metadata"] = _sanitize(d.get("extra_metadata", {}))
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize result to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, filepath: str | Path) -> None:
        """Write JSON result to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunResult":
        """Reconstruct RunResult from dictionary."""
        status_val = data.get("status", "SUCCESS")
        status = (
            RunStatus(status_val)
            if status_val in RunStatus.__members__
            else RunStatus.SUCCESS
        )
        return cls(
            experiment=data["experiment"],
            seed=data["seed"],
            model=data["model"],
            status=status,
            config=data["config"],
            metrics=data["metrics"],
            timing_ms=data.get("timing_ms", {}),
            device=data["device"],
            dtype=data["dtype"],
            error_message=data.get("error_message"),
            error_step=data.get("error_step"),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            extra_metadata=data.get("extra_metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "RunResult":
        """Parse RunResult from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load(cls, filepath: str | Path) -> "RunResult":
        """Load RunResult from disk."""
        return cls.from_json(Path(filepath).read_text(encoding="utf-8"))


@dataclass
class AggregateResult:
    """Aggregated statistical summary across repeated random seeds."""

    experiment: str
    model: str
    num_runs: int
    num_success: int
    num_failures: int
    metric_aggregates: dict[str, dict[str, float]]
    failure_reasons: dict[str, int]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    paired_differences: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert aggregate result to JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize aggregate summary to formatted JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, filepath: str | Path) -> None:
        """Write aggregate JSON summary to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AggregateResult":
        """Reconstruct AggregateResult from dictionary."""
        return cls(
            experiment=data["experiment"],
            model=data["model"],
            num_runs=data["num_runs"],
            num_success=data["num_success"],
            num_failures=data["num_failures"],
            metric_aggregates=data.get("metric_aggregates", {}),
            failure_reasons=data.get("failure_reasons", {}),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            paired_differences=data.get("paired_differences", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "AggregateResult":
        """Parse AggregateResult from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load(cls, filepath: str | Path) -> "AggregateResult":
        """Load AggregateResult from disk."""
        return cls.from_json(Path(filepath).read_text(encoding="utf-8"))
