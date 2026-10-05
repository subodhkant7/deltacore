# ==============================================================================
# DeltaCore: deltacore/observatory/schema.py
# Typed trajectory schema and container with strict missing-data semantics.
# ==============================================================================

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrajectoryStep:
    """Standardized snapshot of internal state and diagnostics at a single step.

    Missing Data & Observation Semantics:
        - observed = True, finite_state = True: Step was computed and all values remained finite.
        - observed = True, finite_state = False: Step was computed and numerical divergence occurred.
        - observed = False, finite_state = None: Step was unobserved / execution terminated beforehand.
        - None: Explicitly unavailable, unsupported, or undefined.
        - 0.0: Measured or derived as exact zero.
        - float('nan'): Numerically invalid result encountered during computation.

    Attributes:
        step: 0-indexed discrete sequence step index.
        observed: True if the step was actually executed during inference; False if execution terminated before this step.
        error_norm: Frobenius/Euclidean norm of the prediction error ||e_t||.
        step_size: Adaptive learning rate eta_t at step t (None if model is non-adaptive/frozen).
        step_size_change: Difference eta_t - eta_{t-1} (None for t=0 or non-adaptive models).
        memory_norm: Frobenius norm of primary associative memory ||M_t||_F.
        dynamics_memory_norm: Frobenius norm of dynamics memory ||C_t||_F (None if unsupported).
        update_norm: Frobenius norm of primary update ||Delta M_t||_F.
        dynamics_update_norm: Frobenius norm of dynamics update ||Delta C_t||_F (None if unsupported).
        normalized_step: Local normalized step magnitude gamma_t = eta_t * ||k_t||^2 (None if non-adaptive).
        stability_margin: Local contraction margin S_t = 2 - gamma_t (None if non-adaptive).
        dynamics_stability_margin: Local margin for dynamics memory S_t^{dyn} (None if unsupported).
        finite_state: True if finite, False if non-finite, None if unobserved.
        clip_event: True if stability controller clipped eta_t or rho_t at this step.
        key_memory_norm: Frobenius norm of key-generation memory ||M_key||_F (Phase 8).
        value_memory_norm: Frobenius norm of value-generation memory ||M_val||_F (Phase 8).
        learning_rate_memory_norm: Frobenius norm of learning-rate memory ||M_lr||_F (Phase 8).
        retention_memory_norm: Frobenius norm of retention memory ||M_ret||_F (Phase 8).
        raw_learning_rate: Unclamped / pre-controller learning rate scalar (Phase 8).
        safe_learning_rate: Effective / post-controller learning rate scalar (Phase 8).
        raw_retention: Unclamped / pre-controller retention decay factor (Phase 8).
        safe_retention: Effective / post-controller retention decay factor (Phase 8).
        extra: Task-specific or diagnostic metadata key-value pairs.
    """

    step: int
    observed: bool = True
    error_norm: float | None = None
    step_size: float | None = None
    step_size_change: float | None = None
    memory_norm: float | None = None
    dynamics_memory_norm: float | None = None
    update_norm: float | None = None
    dynamics_update_norm: float | None = None
    normalized_step: float | None = None
    stability_margin: float | None = None
    dynamics_stability_margin: float | None = None
    finite_state: bool | None = True
    clip_event: bool = False
    key_memory_norm: float | None = None
    value_memory_norm: float | None = None
    learning_rate_memory_norm: float | None = None
    retention_memory_norm: float | None = None
    raw_learning_rate: float | None = None
    safe_learning_rate: float | None = None
    raw_retention: float | None = None
    safe_retention: float | None = None
    route: str | None = None
    route_index: int | None = None
    route_direction: str | None = None
    chunk_size: int | None = None
    chunk_boundary: bool = False
    directional_memory_norm: float | None = None
    directional_learning_rate: float | None = None
    directional_retention: float | None = None
    directional_update_norm: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert step snapshot to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrajectoryStep:
        """Construct TrajectoryStep from dictionary."""
        observed = bool(data.get("observed", True))
        fin = data.get("finite_state")
        finite_val: bool | None
        if not observed:
            finite_val = None
        elif fin is None:
            finite_val = True
        else:
            finite_val = bool(fin)

        return cls(
            step=int(data["step"]),
            observed=observed,
            error_norm=data.get("error_norm"),
            step_size=data.get("step_size"),
            step_size_change=data.get("step_size_change"),
            memory_norm=data.get("memory_norm"),
            dynamics_memory_norm=data.get("dynamics_memory_norm"),
            update_norm=data.get("update_norm"),
            dynamics_update_norm=data.get("dynamics_update_norm"),
            normalized_step=data.get("normalized_step"),
            stability_margin=data.get("stability_margin"),
            dynamics_stability_margin=data.get("dynamics_stability_margin"),
            finite_state=finite_val,
            clip_event=bool(data.get("clip_event", False)),
            key_memory_norm=data.get("key_memory_norm"),
            value_memory_norm=data.get("value_memory_norm"),
            learning_rate_memory_norm=data.get("learning_rate_memory_norm"),
            retention_memory_norm=data.get("retention_memory_norm"),
            raw_learning_rate=data.get("raw_learning_rate"),
            safe_learning_rate=data.get("safe_learning_rate"),
            raw_retention=data.get("raw_retention"),
            safe_retention=data.get("safe_retention"),
            route=data.get("route"),
            route_index=data.get("route_index"),
            route_direction=data.get("route_direction"),
            chunk_size=data.get("chunk_size"),
            chunk_boundary=bool(data.get("chunk_boundary", False)),
            directional_memory_norm=data.get("directional_memory_norm"),
            directional_learning_rate=data.get("directional_learning_rate"),
            directional_retention=data.get("directional_retention"),
            directional_update_norm=data.get("directional_update_norm"),
            extra=data.get("extra", {}),
        )


@dataclass
class StateTrajectory:
    """Standardized representation of a full sequence neural-state trajectory.

    Attributes:
        model: Model identifier (e.g. 'safe_self_referential', 'fixed', 'frozen').
        task: Benchmark task identifier (e.g. 'distribution_shift').
        seed: Random seed used to generate sequence.
        steps: List of TrajectoryStep objects ordered by step index.
        all_states_finite: True iff all steps remained strictly finite.
        first_nonfinite_step: Step index of first numerical overflow/failure, or None.
        terminal_state_finite: True iff the final step was strictly finite.
        metadata: Task hyperparameters and sequence generation metadata.
        metrics: Summary metrics computed over the trajectory.
    """

    model: str
    task: str
    seed: int | None = None
    steps: list[TrajectoryStep] = field(default_factory=list)
    all_states_finite: bool = True
    first_nonfinite_step: int | None = None
    terminal_state_finite: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.steps)

    def get_series(self, metric: str) -> list[float | None]:
        """Extract a single metric across all steps, preserving missing values.

        Args:
            metric: Attribute name of TrajectoryStep (e.g. 'error_norm', 'step_size').

        Returns:
            List of float values or None where metric is unavailable.
        """
        series: list[float | None] = []
        for s in self.steps:
            val = getattr(s, metric, None)
            series.append(val)
        return series

    @property
    def error_norms(self) -> list[float | None]:
        return self.get_series("error_norm")

    @property
    def step_sizes(self) -> list[float | None]:
        return self.get_series("step_size")

    @property
    def memory_norms(self) -> list[float | None]:
        return self.get_series("memory_norm")

    @property
    def update_norms(self) -> list[float | None]:
        return self.get_series("update_norm")

    @property
    def stability_margins(self) -> list[float | None]:
        return self.get_series("stability_margin")

    @property
    def dynamics_memory_norms(self) -> list[float | None]:
        return self.get_series("dynamics_memory_norm")

    @property
    def dynamics_update_norms(self) -> list[float | None]:
        return self.get_series("dynamics_update_norm")

    @property
    def key_memory_norms(self) -> list[float | None]:
        return self.get_series("key_memory_norm")

    @property
    def value_memory_norms(self) -> list[float | None]:
        return self.get_series("value_memory_norm")

    @property
    def learning_rate_memory_norms(self) -> list[float | None]:
        return self.get_series("learning_rate_memory_norm")

    @property
    def retention_memory_norms(self) -> list[float | None]:
        return self.get_series("retention_memory_norm")

    @property
    def raw_learning_rates(self) -> list[float | None]:
        return self.get_series("raw_learning_rate")

    @property
    def safe_learning_rates(self) -> list[float | None]:
        return self.get_series("safe_learning_rate")

    @property
    def raw_retentions(self) -> list[float | None]:
        return self.get_series("raw_retention")

    @property
    def safe_retentions(self) -> list[float | None]:
        return self.get_series("safe_retention")

    def to_dict(self) -> dict[str, Any]:
        """Convert entire trajectory to a serializable dictionary."""
        return {
            "model": self.model,
            "task": self.task,
            "seed": self.seed,
            "all_states_finite": self.all_states_finite,
            "first_nonfinite_step": self.first_nonfinite_step,
            "terminal_state_finite": self.terminal_state_finite,
            "metadata": self.metadata,
            "metrics": self.metrics,
            "steps": [s.to_dict() for s in self.steps],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize trajectory to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, filepath: str | Path) -> None:
        """Write trajectory to disk as JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StateTrajectory:
        """Reconstruct StateTrajectory from dictionary with schema validation."""
        required = ["model", "task", "steps"]
        for req in required:
            if req not in data:
                raise ValueError(
                    f"Malformed trajectory data: missing required field '{req}'"
                )

        first_nonfin = data.get("first_nonfinite_step")
        steps: list[TrajectoryStep] = []
        for s in data["steps"]:
            if (
                "observed" not in s
                and first_nonfin is not None
                and s.get("step", 0) > first_nonfin
            ):
                s_upgraded = dict(s)
                s_upgraded["observed"] = False
                s_upgraded["finite_state"] = None
                steps.append(TrajectoryStep.from_dict(s_upgraded))
            else:
                steps.append(TrajectoryStep.from_dict(s))
        return cls(
            model=str(data["model"]),
            task=str(data["task"]),
            seed=data.get("seed"),
            steps=steps,
            all_states_finite=bool(data.get("all_states_finite", True)),
            first_nonfinite_step=data.get("first_nonfinite_step"),
            terminal_state_finite=data.get("terminal_state_finite"),
            metadata=data.get("metadata", {}),
            metrics=data.get("metrics", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> StateTrajectory:
        """Parse StateTrajectory from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load(cls, filepath: str | Path) -> StateTrajectory:
        """Load StateTrajectory from JSON file."""
        return cls.from_json(Path(filepath).read_text(encoding="utf-8"))
