# ==============================================================================
# DeltaCore: deltacore/benchmarks/metrics/accuracy.py
# Retrieval accuracy, error norms, and relative adaptation gain metrics.
# ==============================================================================

import torch


def compute_final_error(errors: torch.Tensor) -> float:
    r"""Compute Euclidean norm of the final error vector \|e_T\|_2."""
    if errors.shape[0] == 0:
        return 0.0
    return float(torch.linalg.norm(errors[-1]).item())


def compute_mean_error(
    errors: torch.Tensor,
    start_idx: int = 0,
    end_idx: int | None = None,
) -> float:
    r"""Compute mean Euclidean error norm across specified step range."""
    t_steps = errors.shape[0]
    if t_steps == 0:
        return 0.0

    end = t_steps if end_idx is None else min(end_idx, t_steps)
    start = max(0, min(start_idx, end))
    if start >= end:
        return 0.0

    sub_errors = errors[start:end]
    norms = torch.linalg.norm(sub_errors, dim=-1)
    return float(torch.mean(norms).item())


def compute_adaptation_gain(
    eval_error: float,
    baseline_error: float,
    eps: float = 1e-6,
) -> float:
    r"""Compute relative adaptation gain with positive-improvement sign convention.

    Definition:
        \text{Gain} = \frac{E_{\text{baseline}} - E_{\text{eval}}}{\max(E_{\text{baseline}}, \epsilon)}

    Returns:
        Positive value if eval_error < baseline_error (improvement).
        Negative value if eval_error > baseline_error (degradation).
        0.0 if equal.
    """
    denom = max(abs(baseline_error), eps)
    return float((baseline_error - eval_error) / denom)


def compute_relative_error(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    r"""Compute normalized relative Frobenius error E_rel = ||Y - \hat{Y}||_F / max(||Y||_F, eps).

    Args:
        predictions: Predicted tensor \hat{Y} of any matching shape.
        targets: Target tensor Y of identical shape.
        eps: Denominator stabilizer (default: 1e-6).

    Returns:
        Relative error scalar E_rel >= 0.0.
    """
    diff_norm = torch.linalg.norm(targets - predictions).item()
    target_norm = torch.linalg.norm(targets).item()
    denom = max(target_norm, eps)
    return float(diff_norm / denom)
