# ==============================================================================
# DeltaCore: deltacore/benchmarks/tasks/stationary_recall.py
# Benchmark 1: Stationary associative recall under constant distribution.
# ==============================================================================

from typing import Any

import torch

from deltacore.benchmarks.baselines.base import BaselineTrajectoryResult
from deltacore.benchmarks.metrics.accuracy import (
    compute_final_error,
    compute_mean_error,
)
from deltacore.benchmarks.metrics.energy import (
    compute_step_energy,
    compute_update_energy,
)
from deltacore.benchmarks.schemas.config import BenchmarkConfig
from deltacore.benchmarks.tasks.base import BaseTask


class StationaryRecallTask(BaseTask):
    """Benchmark 1: Stationary Associative Recall.

    Evaluates retrieval accuracy and memory growth on a stationary set of
    associative bindings. Answers: 'Does adaptive state improve a task that
    does not require adaptation?'
    """

    def __init__(self) -> None:
        super().__init__(name="stationary_recall")

    def generate_data(
        self, config: BenchmarkConfig
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        gen = torch.Generator().manual_seed(config.seed)
        dtype = torch.float64 if config.dtype == "float64" else torch.float32

        num_associations = config.task_params.get("num_associations", 8)
        correlation = config.task_params.get("correlation", 0.0)

        # Generate base patterns
        base_keys = torch.randn(
            num_associations, config.key_dim, generator=gen, dtype=dtype
        )
        base_keys = torch.nn.functional.normalize(base_keys, p=2, dim=-1)

        # Inject correlation if requested
        if correlation > 0.0 and num_associations > 1:
            shared = torch.randn(1, config.key_dim, generator=gen, dtype=dtype)
            shared = torch.nn.functional.normalize(shared, p=2, dim=-1)
            base_keys = (1.0 - correlation) * base_keys + correlation * shared
            base_keys = torch.nn.functional.normalize(base_keys, p=2, dim=-1)

        base_targets = torch.randn(
            num_associations, config.value_dim, generator=gen, dtype=dtype
        )

        # Generate sequence of length T by cycling or sampling associations
        seq_indices = torch.randint(
            0,
            num_associations,
            (config.sequence_length,),
            generator=gen,
        )

        keys = base_keys[seq_indices]
        targets = base_targets[seq_indices]

        metadata = {
            "num_associations": num_associations,
            "correlation": correlation,
            "unique_keys_count": num_associations,
        }
        return keys, targets, metadata

    def compute_metrics(
        self,
        trajectory: BaselineTrajectoryResult,
        metadata: dict[str, Any],
        config: BenchmarkConfig,
    ) -> dict[str, Any]:
        final_err = compute_final_error(trajectory.errors)
        mean_err = compute_mean_error(trajectory.errors)
        upd_energy = compute_update_energy(trajectory.update_norms)
        step_energy = compute_step_energy(trajectory.step_sizes)

        return {
            "final_error": final_err,
            "mean_error": mean_err,
            "final_state_norm": float(
                torch.linalg.norm(trajectory.final_memory).item()
            ),
            "max_state_norm": trajectory.max_state_norm,
            "max_update_norm": trajectory.max_update_norm,
            "update_energy": upd_energy,
            "step_energy": step_energy,
            "clip_count": trajectory.clip_count,
            "all_states_finite": trajectory.all_states_finite,
            "terminal_state_finite": trajectory.terminal_state_finite,
            "first_nonfinite_step": trajectory.first_nonfinite_step,
        }
