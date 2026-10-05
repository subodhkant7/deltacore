r"""Diagnostics and observability utilities for adaptive state dynamics.

Provides passive telemetry for measuring step-size trajectories, memory norms,
update magnitudes, numerical health, and recovery dynamics without modifying state.

IMPORTANT: This module provides OBSERVABILITY ONLY. It does NOT provide stability
guarantees or projective clamps (which are reserved for Phase 4).
"""

from typing import Any

import torch

from deltacore.memory.associative import AssociativeMemory


def compute_step_size_stats(step_sizes: torch.Tensor) -> dict[str, float]:
    r"""Compute descriptive statistics over a step-size trajectory $\eta_t$.

    Args:
        step_sizes: 1D or batched tensor of step sizes.

    Returns:
        Dictionary with 'mean', 'min', 'max', and 'std' values.
    """
    s = step_sizes.detach().float()
    return {
        "mean": float(torch.mean(s).item()),
        "min": float(torch.min(s).item()),
        "max": float(torch.max(s).item()),
        "std": float(torch.std(s).item()) if s.numel() > 1 else 0.0,
    }


def compute_memory_norm(memory: AssociativeMemory | torch.Tensor) -> float:
    r"""Compute the Frobenius norm of an associative memory state: $\|M\|_F$.

    Args:
        memory: AssociativeMemory instance or raw tensor.

    Returns:
        Scalar Frobenius norm.
    """
    m_data = memory.data if isinstance(memory, AssociativeMemory) else memory
    return float(torch.linalg.norm(m_data.detach().float()).item())


def compute_update_norm(update: torch.Tensor) -> float:
    r"""Compute the Frobenius norm of a state update tensor: $\|\Delta M\|_F$.

    Args:
        update: State change tensor $\Delta M$.

    Returns:
        Scalar Frobenius norm.
    """
    return float(torch.linalg.norm(update.detach().float()).item())


def compute_error_norm(error: torch.Tensor) -> float:
    r"""Compute the Euclidean $L_2$ norm of a prediction error vector: $\|e\|_2$.

    Args:
        error: Prediction residual vector $e = v - \hat{v}$.

    Returns:
        Scalar Euclidean norm.
    """
    return float(torch.linalg.norm(error.detach().float()).item())


def check_numerical_health(tensor: torch.Tensor) -> dict[str, Any]:
    r"""Inspect tensor for non-finite values ($NaN$, $Inf$).

    Args:
        tensor: Tensor to inspect.

    Returns:
        Dictionary indicating whether the tensor is strictly finite, and count of NaNs/Infs.
    """
    has_nan = bool(torch.isnan(tensor).any().item())
    has_inf = bool(torch.isinf(tensor).any().item())
    is_finite = bool(torch.isfinite(tensor).all().item())
    return {
        "is_finite": is_finite,
        "has_nan": has_nan,
        "has_inf": has_inf,
    }


def compute_adaptation_gain(fixed_error: float, adaptive_error: float) -> float:
    r"""Compute adaptation gain relative to fixed baseline: $G_{\text{adapt}} = E_{\text{fixed}} - E_{\text{adaptive}}$.

    Sign convention:
        Positive gain ($G_{\text{adapt}} > 0$) indicates the adaptive controller
        achieved lower error than the fixed baseline.
        Negative gain indicates the fixed baseline performed better.

    Args:
        fixed_error: Scalar error (e.g. MSE) of fixed baseline.
        adaptive_error: Scalar error of adaptive controller.

    Returns:
        Scalar difference $E_{\text{fixed}} - E_{\text{adaptive}}$.
    """
    return fixed_error - adaptive_error


def compute_recovery_steps(
    errors: torch.Tensor,
    threshold_ratio: float = 0.5,
    baseline_error: float | None = None,
) -> int | None:
    r"""Measure the number of steps required for error to contract below a threshold.

    Definition:
        $$\text{RecoverySteps} = \min \{t \ge 0 : E_t \le \tau E_{\text{baseline}}\}$$

    Args:
        errors: 1D sequence of step-wise scalar errors $E_t = \|e_t\|_2$.
        threshold_ratio: Recovery fraction $\tau \in (0, 1]$.
        baseline_error: Reference baseline error $E_{\text{baseline}}$. If None,
            uses the initial error $E_0 = \text{errors}[0]$.

    Returns:
        Zero-indexed step $t$ where error first recovers below $\tau E_{\text{baseline}}$,
        or None if the condition is never satisfied.
    """
    if errors.numel() == 0:
        return None

    ref_error = (
        baseline_error if baseline_error is not None else float(errors[0].item())
    )
    target_threshold = threshold_ratio * ref_error

    for t in range(len(errors)):
        if float(errors[t].item()) <= target_threshold:
            return t
    return None
