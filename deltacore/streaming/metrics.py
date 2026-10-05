"""Adaptation, Recovery, Stability, and Telemetry Metrics for Streaming Benchmark.

Implements all metrics specified in Phase 10 Sections 9 and 10:
    - Pre-shift error (E_pre)
    - Post-shift error (E_post_0)
    - First-passage recovery (t_first)
    - Sustained recovery (t_sust, K=10)
    - Cumulative excess error
    - Forgetting metric (E_return - E_pre)
    - Adaptation energy (sum ||Delta S_t||_F^2)
    - State norms, stability margins, and finite checks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class StreamingTelemetry:
    """Diagnostic tracking container for state trajectory across streaming timesteps."""

    step_errors: list[float] = field(default_factory=list)
    state_norms: list[float] = field(default_factory=list)
    update_norms: list[float] = field(default_factory=list)
    stability_margins: list[float] = field(default_factory=list)
    step_sizes: list[float] = field(default_factory=list)
    retentions: list[float] = field(default_factory=list)
    non_finite_count: int = 0
    adaptation_energy: float = 0.0

    def record_step(
        self,
        error: float,
        state_norm: float,
        update_norm: float,
        stability_margin: float = 2.0,
        step_size: float = 0.0,
        retention: float = 1.0,
        is_finite: bool = True,
    ) -> None:
        self.step_errors.append(error)
        self.state_norms.append(state_norm)
        self.update_norms.append(update_norm)
        self.stability_margins.append(stability_margin)
        self.step_sizes.append(step_size)
        self.retentions.append(retention)
        self.adaptation_energy += float(update_norm**2)
        if not is_finite:
            self.non_finite_count += 1

    def summary(self) -> dict[str, Any]:
        initial_norm = self.state_norms[0] if self.state_norms else 1.0
        max_norm = max(self.state_norms) if self.state_norms else 0.0
        growth_ratio = max_norm / max(initial_norm, 1e-6)

        return {
            "mean_error": float(np.mean(self.step_errors)) if self.step_errors else 0.0,
            "final_error": self.step_errors[-1] if self.step_errors else 0.0,
            "max_state_norm": float(max_norm),
            "state_growth_ratio": float(growth_ratio),
            "min_stability_margin": float(min(self.stability_margins))
            if self.stability_margins
            else 2.0,
            "mean_update_norm": float(np.mean(self.update_norms))
            if self.update_norms
            else 0.0,
            "adaptation_energy": float(self.adaptation_energy),
            "non_finite_count": self.non_finite_count,
            "mean_step_size": float(np.mean(self.step_sizes))
            if self.step_sizes
            else 0.0,
            "mean_retention": float(np.mean(self.retentions))
            if self.retentions
            else 1.0,
        }


def compute_relative_step_error(
    prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-7
) -> float:
    r"""Compute relative error for a single timestep: ||y - \hat{y}|| / \max(||y||, eps)."""
    err = torch.linalg.norm(target - prediction).item()
    norm = max(float(torch.linalg.norm(target).item()), eps)
    return float(err / norm)


def compute_pre_post_errors(
    errors: list[float],
    change_point: int,
    window: int = 15,
) -> tuple[float, float]:
    r"""Compute pre-shift error E_pre and immediate post-shift error E_post,0."""
    start_pre = max(0, change_point - window)
    pre_slice = errors[start_pre:change_point]
    e_pre = float(np.mean(pre_slice)) if pre_slice else errors[change_point]

    e_post_0 = errors[change_point] if change_point < len(errors) else 0.0
    return e_pre, e_post_0


def compute_first_passage_recovery(
    errors: list[float],
    change_point: int,
    threshold: float,
    max_steps: int = 120,
) -> int:
    r"""Find first timestep t >= change_point where error <= threshold.

    Returns relative delay: t - change_point. If never recovered, returns max_steps.
    """
    total = len(errors)
    end = min(total, change_point + max_steps)
    for t in range(change_point, end):
        if errors[t] <= threshold:
            return t - change_point
    return max_steps


def compute_sustained_recovery(
    errors: list[float],
    change_point: int,
    threshold: float,
    consecutive_steps: int = 10,
    max_steps: int = 120,
) -> int:
    r"""Find first timestep where error <= threshold for K=10 consecutive steps.

    Returns relative delay: t - change_point. If never satisfied, returns max_steps.
    """
    total = len(errors)
    end = min(total - consecutive_steps + 1, change_point + max_steps)
    for t in range(change_point, end):
        window = errors[t : t + consecutive_steps]
        if all(e <= threshold for e in window):
            return t - change_point
    return max_steps


def compute_cumulative_excess_error(
    errors: list[float],
    change_point: int,
    e_pre: float,
    window: int = 50,
) -> float:
    r"""Compute cumulative excess error: sum_{t=change_point}^{change_point + window} max(0, E_t - E_pre)."""
    end = min(len(errors), change_point + window)
    excess = [max(0.0, errors[t] - e_pre) for t in range(change_point, end)]
    return float(np.sum(excess))


def compute_adaptation_energy(update_norms: list[float]) -> float:
    r"""Compute cumulative adaptation energy: sum_t ||\Delta S_t||_F^2."""
    return float(np.sum(np.square(update_norms)))


def compute_forgetting(
    errors: list[float],
    initial_steady_point: int,
    return_change_point: int,
    window: int = 15,
) -> float:
    r"""Compute forgetting metric for A -> B -> A:

    E_forget = E_{post, return_A} - E_{pre, initial_A}
    """
    # Pre-shift error on A before entering B
    e_pre_a, _ = compute_pre_post_errors(errors, initial_steady_point, window=window)

    # Post-return error upon re-entering A
    start_ret = return_change_point
    end_ret = min(len(errors), return_change_point + window)
    e_return_slice = errors[start_ret:end_ret]
    e_return_a = float(np.mean(e_return_slice)) if e_return_slice else 0.0

    return float(e_return_a - e_pre_a)


def compute_negative_transfer(error_continuous_b: float, error_reset_b: float) -> float:
    r"""Compute stale-memory negative transfer penalty in Regime B:

    Negative Transfer = Error_continuous(B) - Error_reset(B)
    """
    return float(error_continuous_b - error_reset_b)


def compute_adaptation_efficiency(
    error_reduction: float, adaptation_energy: float, eps: float = 1e-6
) -> float:
    r"""Compute adaptation efficiency: error_reduction / (adaptation_energy + eps)."""
    return float(error_reduction / max(adaptation_energy + eps, eps))
