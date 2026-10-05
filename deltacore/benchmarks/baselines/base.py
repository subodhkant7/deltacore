# ==============================================================================
# DeltaCore: deltacore/benchmarks/baselines/base.py
# Unified abstract interface and result containers for benchmark baselines.
# ==============================================================================

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class BaselineStepResult:
    """Standardized output of a single baseline step."""

    prediction: torch.Tensor
    error: torch.Tensor
    step_size: float
    update_norm: float
    state_norm: float
    stability_margin: float
    is_finite: bool


@dataclass
class BaselineTrajectoryResult:
    """Standardized output of a full sequence evaluation."""

    predictions: torch.Tensor  # [T, V]
    errors: torch.Tensor  # [T, V]
    final_memory: torch.Tensor  # [V, K]
    memory_norms: list[float]
    update_norms: list[float]
    step_sizes: list[float]
    stability_margins: list[float]
    normalized_steps: list[float]
    clip_count: int
    all_states_finite: bool
    terminal_state_finite: bool
    first_nonfinite_step: int | None
    max_state_norm: float
    max_update_norm: float
    final_error_norm: float


class BaseBaseline(ABC):
    """Abstract interface that all benchmark baseline systems must implement."""

    def __init__(
        self,
        name: str,
        key_dim: int,
        value_dim: int,
        dtype: torch.dtype | str = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        self.name = name
        self.key_dim = key_dim
        self.value_dim = value_dim
        if isinstance(dtype, str):
            self.dtype = torch.float64 if dtype == "float64" else torch.float32
        else:
            self.dtype = dtype
        self.device = torch.device(device)

    @abstractmethod
    def reset(self, initial_memory: torch.Tensor | None = None) -> None:
        """Reset internal state of the baseline system."""

    @abstractmethod
    def run_sequence(
        self,
        keys: torch.Tensor,
        targets: torch.Tensor,
        initial_memory: torch.Tensor | None = None,
    ) -> BaselineTrajectoryResult:
        """Execute a full sequence of key-target pairs and return standardized trajectory.

        Args:
            keys: Sequence of keys [T, K].
            targets: Sequence of targets [T, V].
            initial_memory: Optional starting memory [V, K].

        Returns:
            Standardized BaselineTrajectoryResult.
        """
