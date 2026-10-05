# ==============================================================================
# DeltaCore: deltacore/benchmarks/schemas/config.py
# Typed configuration schema and validation for DeltaCore benchmarks.
# ==============================================================================

from dataclasses import asdict, dataclass, field
from typing import Any

VALID_CONTROLLERS = {
    "frozen",
    "fixed",
    "adaptive",
    "self_referential",
    "safe_self_referential",
}

VALID_STABILITY_MODES = {
    "unconstrained",
    "content_safe",
    "dynamics_safe",
    "dual_safe",
}

VALID_DTYPES = {"float32", "float64"}


@dataclass(frozen=True)
class BenchmarkConfig:
    """Strongly typed, validated configuration for benchmark runs.

    Attributes:
        experiment_name: Identifier for the benchmark task/experiment.
        seed: Deterministic random seed.
        sequence_length: Number of time steps T.
        key_dim: Dimension of key vectors K.
        value_dim: Dimension of value vectors V.
        batch_size: Optional batch size for batched evaluation.
        shift_position: Optional step index where distribution shift occurs.
        chunk_size: Optional chunk size C for chunked execution.
        controller: Baseline controller type ('frozen', 'fixed', 'adaptive',
            'self_referential', 'safe_self_referential').
        stability_mode: Stability enforcement mode ('unconstrained',
            'content_safe', 'dynamics_safe', 'dual_safe').
        dtype: Numerical precision ('float32' or 'float64').
        device: Execution device ('cpu' or 'cuda').
        repetitions: Number of timing repetitions.
        warmup_count: Number of untimed warmup iterations.
        task_params: Task-specific hyperparameter dictionary.
    """

    experiment_name: str
    seed: int = 42
    sequence_length: int = 128
    key_dim: int = 16
    value_dim: int = 16
    batch_size: int | None = None
    shift_position: int | None = None
    chunk_size: int | None = None
    controller: str = "fixed"
    stability_mode: str = "unconstrained"
    dtype: str = "float32"
    device: str = "cpu"
    repetitions: int = 1
    warmup_count: int = 0
    task_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration parameters immediately upon construction."""
        self.validate()

    def validate(self) -> None:
        """Enforce strict validation rules on configuration parameters.

        Raises:
            ValueError: If any parameter violates range or type constraints.
        """
        if not self.experiment_name or not isinstance(self.experiment_name, str):
            raise ValueError("experiment_name must be a non-empty string.")

        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {self.seed}.")

        if not isinstance(self.sequence_length, int) or self.sequence_length <= 0:
            raise ValueError(
                f"sequence_length must be positive integer, got {self.sequence_length}."
            )

        if not isinstance(self.key_dim, int) or self.key_dim <= 0:
            raise ValueError(f"key_dim must be positive integer, got {self.key_dim}.")

        if not isinstance(self.value_dim, int) or self.value_dim <= 0:
            raise ValueError(
                f"value_dim must be positive integer, got {self.value_dim}."
            )

        if self.batch_size is not None:
            if not isinstance(self.batch_size, int) or self.batch_size <= 0:
                raise ValueError(
                    f"batch_size must be positive integer or None, got {self.batch_size}."
                )

        if self.shift_position is not None:
            if not isinstance(self.shift_position, int) or not (
                0 <= self.shift_position <= self.sequence_length
            ):
                raise ValueError(
                    f"shift_position must be in [0, {self.sequence_length}], got {self.shift_position}."
                )

        if self.chunk_size is not None:
            if not isinstance(self.chunk_size, int) or self.chunk_size <= 0:
                raise ValueError(
                    f"chunk_size must be positive integer or None, got {self.chunk_size}."
                )

        if self.controller not in VALID_CONTROLLERS:
            raise ValueError(
                f"Unknown controller '{self.controller}'. Must be one of {sorted(VALID_CONTROLLERS)}."
            )

        if self.stability_mode not in VALID_STABILITY_MODES:
            raise ValueError(
                f"Unknown stability_mode '{self.stability_mode}'. Must be one of {sorted(VALID_STABILITY_MODES)}."
            )

        if self.dtype not in VALID_DTYPES:
            raise ValueError(
                f"Unknown dtype '{self.dtype}'. Must be one of {sorted(VALID_DTYPES)}."
            )

        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty string (e.g., 'cpu', 'cuda').")

        if not isinstance(self.repetitions, int) or self.repetitions <= 0:
            raise ValueError(
                f"repetitions must be positive integer, got {self.repetitions}."
            )

        if not isinstance(self.warmup_count, int) or self.warmup_count < 0:
            raise ValueError(
                f"warmup_count must be non-negative integer, got {self.warmup_count}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a plain dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkConfig":
        """Instantiate BenchmarkConfig from a dictionary with validation."""
        valid_keys = set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)
