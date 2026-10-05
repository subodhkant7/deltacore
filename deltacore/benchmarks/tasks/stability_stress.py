# ==============================================================================
# DeltaCore: deltacore/benchmarks/tasks/stability_stress.py
# Benchmark 5: Numerical and adversarial stability stress testing.
# ==============================================================================

from typing import Any

import torch

from deltacore.benchmarks.baselines.base import BaselineTrajectoryResult
from deltacore.benchmarks.metrics.energy import (
    compute_max_normalized_step,
    compute_min_stability_margin,
)
from deltacore.benchmarks.schemas.config import BenchmarkConfig
from deltacore.benchmarks.tasks.base import BaseTask


class StabilityStressTask(BaseTask):
    """Benchmark 5: Adversarial Stability Stress Testing.

    Exposes recurrent memory systems to aggressive regimes designed to induce
    numerical overflow, unbounded state growth, or expansion:
        - repeated conflicting keys
        - large key norms (||k|| >> 1)
        - large target norms (||v|| >> 1)
        - abrupt shifts
        - long sequences

    Evaluates: unconstrained, content-safe, dynamics-safe, and dual-safe modes.
    """

    def __init__(self) -> None:
        super().__init__(name="stability_stress")

    def generate_data(
        self, config: BenchmarkConfig
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        gen = torch.Generator().manual_seed(config.seed)
        dtype = torch.float64 if config.dtype == "float64" else torch.float32

        t_steps = config.sequence_length
        regime = config.task_params.get("regime", "large_key_norms")
        key_scale = config.task_params.get("key_scale", 5.0)
        target_scale = config.task_params.get("target_scale", 10.0)

        if regime == "large_key_norms":
            # Keys with large norm so that eta * ||k||^2 >> 2
            keys = torch.randn(t_steps, config.key_dim, generator=gen, dtype=dtype)
            keys = torch.nn.functional.normalize(keys, p=2, dim=-1) * key_scale
            targets = torch.randn(t_steps, config.value_dim, generator=gen, dtype=dtype)
        elif regime == "large_target_norms":
            keys = torch.randn(t_steps, config.key_dim, generator=gen, dtype=dtype)
            keys = torch.nn.functional.normalize(keys, p=2, dim=-1)
            targets = (
                torch.randn(t_steps, config.value_dim, generator=gen, dtype=dtype)
                * target_scale
            )
        elif regime == "repeated_conflicting_keys":
            # Identical key with opposing targets
            k = torch.randn(config.key_dim, generator=gen, dtype=dtype) * key_scale
            v = torch.randn(config.value_dim, generator=gen, dtype=dtype)
            keys = k.unsqueeze(0).repeat(t_steps, 1)
            targets = torch.zeros((t_steps, config.value_dim), dtype=dtype)
            for t in range(t_steps):
                targets[t] = v if (t % 2 == 0) else -v
        elif regime == "abrupt_shifts":
            keys = torch.randn(t_steps, config.key_dim, generator=gen, dtype=dtype)
            keys = torch.nn.functional.normalize(keys, p=2, dim=-1) * key_scale
            targets = torch.randn(t_steps, config.value_dim, generator=gen, dtype=dtype)
            # Periodic shocks
            period = max(10, t_steps // 4)
            for t in range(0, t_steps, period):
                targets[t : t + period] = targets[t : t + period] * (
                    -1.0 if (t // period) % 2 == 1 else 1.0
                )
        else:
            # Default generic stress
            keys = (
                torch.randn(t_steps, config.key_dim, generator=gen, dtype=dtype)
                * key_scale
            )
            targets = (
                torch.randn(t_steps, config.value_dim, generator=gen, dtype=dtype)
                * target_scale
            )

        metadata = {
            "regime": regime,
            "key_scale": key_scale,
            "target_scale": target_scale,
        }
        return keys, targets, metadata

    def compute_metrics(
        self,
        trajectory: BaselineTrajectoryResult,
        metadata: dict[str, Any],
        config: BenchmarkConfig,
    ) -> dict[str, Any]:
        min_margin = compute_min_stability_margin(trajectory.stability_margins)
        max_step = compute_max_normalized_step(trajectory.normalized_steps)

        # Precise scientific terminology classification
        if not trajectory.all_states_finite:
            classification = "divergent (non-finite)"
        elif min_margin >= 0.0:
            classification = "locally non-expansive and finite in tested horizon"
        else:
            classification = (
                "empirically finite but violates local non-expansion (expansion regime)"
            )

        return {
            "regime": metadata["regime"],
            "all_states_finite": trajectory.all_states_finite,
            "terminal_state_finite": trajectory.terminal_state_finite,
            "first_nonfinite_step": trajectory.first_nonfinite_step,
            "max_state_norm": trajectory.max_state_norm,
            "max_update_norm": trajectory.max_update_norm,
            "min_stability_margin": min_margin,
            "max_normalized_step": max_step,
            "clip_count": trajectory.clip_count,
            "stability_classification": classification,
        }
