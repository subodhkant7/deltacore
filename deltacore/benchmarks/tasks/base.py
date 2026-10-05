# ==============================================================================
# DeltaCore: deltacore/benchmarks/tasks/base.py
# Abstract base task for reproducible DeltaCore benchmarks.
# ==============================================================================

import time
from abc import ABC, abstractmethod
from typing import Any

import torch

from deltacore.benchmarks.baselines.base import (
    BaseBaseline,
    BaselineTrajectoryResult,
)
from deltacore.benchmarks.schemas.config import BenchmarkConfig
from deltacore.benchmarks.schemas.result import RunResult, RunStatus


class BaseTask(ABC):
    """Abstract base class for all synthetic benchmark tasks."""

    def __init__(self, name: str = "") -> None:
        self.name = name

    @abstractmethod
    def generate_data(
        self, config: BenchmarkConfig
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Generate deterministic input keys, targets, and task metadata.

        Args:
            config: BenchmarkConfig object specifying dimensions, length, seed, etc.

        Returns:
            Tuple of:
                keys: Tensor [T, K]
                targets: Tensor [T, V]
                metadata: Dictionary containing task-specific ground-truth parameters.
        """

    def run(
        self,
        baseline: BaseBaseline,
        config: BenchmarkConfig,
    ) -> RunResult:
        """Execute the benchmark on a given baseline model.

        Handles timing, execution, numerical error capture, and metric extraction.

        Args:
            baseline: Instantiated BaseBaseline model.
            config: Benchmark configuration.

        Returns:
            Standardized RunResult object.
        """
        try:
            config.validate()
        except ValueError as e:
            return RunResult(
                experiment=self.name,
                seed=config.seed,
                model=baseline.name,
                status=RunStatus.INVALID_CONFIGURATION,
                config=config.to_dict(),
                metrics={},
                timing_ms={},
                device=config.device,
                dtype=config.dtype,
                error_message=str(e),
            )

        # 1. Deterministic data generation
        keys, targets, task_meta = self.generate_data(config)

        # 2. Timing and execution
        dev = torch.device(config.device)
        keys_dev = keys.to(device=dev)
        targets_dev = targets.to(device=dev)

        # Warmup if requested
        for _ in range(config.warmup_count):
            baseline.run_sequence(keys_dev[:10], targets_dev[:10])
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)

        # Timed execution
        t0 = time.perf_counter()
        traj_result: BaselineTrajectoryResult = baseline.run_sequence(
            keys_dev, targets_dev
        )
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        wall_clock_ms = (time.perf_counter() - t0) * 1000.0

        # 3. Determine run status
        if not traj_result.all_states_finite:
            status = RunStatus.NUMERICAL_FAILURE
            error_msg = f"Non-finite state or prediction encountered at step {traj_result.first_nonfinite_step}."
        else:
            status = RunStatus.SUCCESS
            error_msg = None

        # 4. Compute metrics
        metrics = self.compute_metrics(traj_result, task_meta, config)

        timing_dict = {
            "wall_clock_ms": wall_clock_ms,
            "mean_step_ms": wall_clock_ms / max(config.sequence_length, 1),
        }

        return RunResult(
            experiment=self.name,
            seed=config.seed,
            model=baseline.name,
            status=status,
            config=config.to_dict(),
            metrics=metrics,
            timing_ms=timing_dict,
            device=config.device,
            dtype=config.dtype,
            error_message=error_msg,
            error_step=traj_result.first_nonfinite_step,
            extra_metadata=task_meta,
        )

    @abstractmethod
    def compute_metrics(
        self,
        trajectory: BaselineTrajectoryResult,
        metadata: dict[str, Any],
        config: BenchmarkConfig,
    ) -> dict[str, Any]:
        """Compute task-specific scalar metrics from the execution trajectory."""
