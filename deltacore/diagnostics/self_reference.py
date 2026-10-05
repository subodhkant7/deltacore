r"""Diagnostics and telemetry for self-referential adaptive memory.

Measures coupled state norms, update magnitudes, step-size velocity (\Delta\eta_t),
numerical integrity, and empirical coupling statistics between dynamics updates
and control output changes.
"""

from typing import Any

import torch

from deltacore.updates.self_referential import SelfReferentialStepResult


def compute_step_size_changes(step_sizes: torch.Tensor) -> torch.Tensor:
    r"""Compute step-size variation across sequential transitions: \Delta\eta_t = |\eta_t - \eta_{t-1}|.

    Args:
        step_sizes: 1D tensor of step sizes [T] or batched [B, T].

    Returns:
        Tensor of absolute differences, padded with 0.0 at t=0.
    """
    if step_sizes.numel() <= 1:
        return torch.zeros_like(step_sizes)

    if step_sizes.ndim == 1:
        diffs = torch.abs(step_sizes[1:] - step_sizes[:-1])
        pad = torch.zeros(1, dtype=step_sizes.dtype, device=step_sizes.device)
        return torch.cat([pad, diffs], dim=0)
    elif step_sizes.ndim == 2:
        diffs = torch.abs(step_sizes[:, 1:] - step_sizes[:, :-1])
        pad = torch.zeros(
            (step_sizes.shape[0], 1),
            dtype=step_sizes.dtype,
            device=step_sizes.device,
        )
        return torch.cat([pad, diffs], dim=1)
    else:
        raise ValueError(f"Unsupported step_sizes ndim={step_sizes.ndim}.")


def compute_coupling_correlation(
    dynamics_update_norms: torch.Tensor, step_size_changes: torch.Tensor
) -> float:
    r"""Compute Pearson correlation between dynamics memory update norm and step-size change.

    Note: This is an empirical association statistic; correlation does NOT establish
    causal attribution.

    Args:
        dynamics_update_norms: 1D sequence of \|\Delta C_t\|_F.
        step_size_changes: 1D sequence of \Delta\eta_t.

    Returns:
        Scalar Pearson correlation in [-1.0, 1.0], or 0.0 if variance is zero.
    """
    x = dynamics_update_norms.detach().flatten().float()
    y = step_size_changes.detach().flatten().float()

    if len(x) < 2:
        return 0.0

    x_std = torch.std(x)
    y_std = torch.std(y)

    if x_std < 1e-12 or y_std < 1e-12:
        return 0.0

    x_centered = x - torch.mean(x)
    y_centered = y - torch.mean(y)

    r = torch.sum(x_centered * y_centered) / (
        torch.sqrt(torch.sum(x_centered**2)) * torch.sqrt(torch.sum(y_centered**2))
    )
    return float(r.item())


def measure_step_telemetry(
    step_result: SelfReferentialStepResult,
    prev_step_size: float | None = None,
) -> dict[str, Any]:
    r"""Extract full diagnostic telemetry from a single self-referential step.

    Returns:
        Dictionary recording:
            - content_memory_norm
            - dynamics_memory_norm
            - content_update_norm
            - dynamics_update_norm
            - step_size
            - step_size_change
            - prediction_error_norm
            - control_signal_norm
            - finite_state
    """
    m_data = step_result.new_content_memory.data
    c_data = step_result.new_dynamics_memory.data

    eta_val = float(step_result.step_size.detach().float().item())
    delta_eta = abs(eta_val - prev_step_size) if prev_step_size is not None else 0.0

    is_finite = bool(
        torch.isfinite(m_data).all().item() and torch.isfinite(c_data).all().item()
    )

    return {
        "content_memory_norm": float(torch.linalg.norm(m_data.detach().float()).item()),
        "dynamics_memory_norm": float(
            torch.linalg.norm(c_data.detach().float()).item()
        ),
        "content_update_norm": float(
            torch.linalg.norm(step_result.content_update.detach().float()).item()
        ),
        "dynamics_update_norm": float(
            torch.linalg.norm(step_result.dynamics_update.detach().float()).item()
        ),
        "step_size": eta_val,
        "step_size_change": delta_eta,
        "prediction_error_norm": float(
            torch.linalg.norm(step_result.error.detach().float()).item()
        ),
        "control_signal_norm": float(
            torch.linalg.norm(step_result.dynamics_prediction.detach().float()).item()
        ),
        "finite_state": is_finite,
    }
