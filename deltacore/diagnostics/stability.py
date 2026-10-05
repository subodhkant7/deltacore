r"""Stability telemetry, recovery latency metrics, and contraction diagnostics.

Provides observability functions to assess how close an adaptive memory system
operates to theoretical divergence boundaries:
    - Normalized Step: \gamma = \eta \|k\|^2
    - Stability Margin: \mu = 2.0 - \gamma
    - Normalized Dynamics Rate: \gamma_C = \rho \|z\|^2
    - Dynamics Stability Margin: \mu_C = 2.0 - \gamma_C
    - First-passage recovery latency
    - Sustained recovery latency over window W
"""

from typing import Any

import torch


def compute_normalized_step(
    step_size: torch.Tensor | float, key: torch.Tensor
) -> float:
    r"""Compute normalized content step size: \gamma_t = \eta_t \|k_t\|_2^2.

    Theoretical boundary is \gamma = 2.0 (local contraction requires \gamma < 2.0).
    """
    eta_val = (
        float(step_size.detach().item())
        if isinstance(step_size, torch.Tensor)
        else float(step_size)
    )
    k_norm_sq = float(torch.sum(key.detach().float() ** 2).item())
    return eta_val * k_norm_sq


def compute_normalized_dynamics_rate(
    dynamics_rate: torch.Tensor | float, features: torch.Tensor
) -> float:
    r"""Compute normalized dynamics learning rate: \gamma_{C, t} = \rho_t \|z_t\|_2^2.

    Theoretical boundary is \gamma_C = 2.0.
    """
    rho_val = (
        float(dynamics_rate.detach().item())
        if isinstance(dynamics_rate, torch.Tensor)
        else float(dynamics_rate)
    )
    z_norm_sq = float(torch.sum(features.detach().float() ** 2).item())
    return rho_val * z_norm_sq


def compute_stability_margin(normalized_step: float) -> float:
    r"""Compute distance to theoretical divergence boundary: \mu = 2.0 - \gamma."""
    return 2.0 - float(normalized_step)


def compute_first_passage_recovery_steps(
    errors: torch.Tensor,
    shift_index: int,
    tau: float = 0.5,
) -> int | None:
    r"""Compute first-passage recovery latency following distribution shift.

    $$\text{FirstPassageRecoverySteps} = \min \{ j \ge 0 : \|e_{t_{\text{shift}} + j}\|_2 \le \tau \cdot \|e_{t_{\text{shift}}}\|_2 \}$$

    Args:
        errors: Error tensor of shape `[T, V]`.
        shift_index: Time index $t_{\text{shift}}$ of the distribution shift.
        tau: Contraction ratio threshold (\tau = 0.5 defines 50% shock attenuation).

    Returns:
        Integer steps to first cross recovery threshold, or None if not met within horizon.
    """
    post_shift_errs = errors[shift_index:]
    if len(post_shift_errs) == 0:
        return None

    err_norms = torch.linalg.norm(post_shift_errs, dim=-1)
    e_ref = err_norms[0].item()

    if e_ref <= 1e-12:
        return 0

    threshold = tau * e_ref
    for j in range(len(err_norms)):
        if err_norms[j].item() <= threshold:
            return j

    return None


def compute_sustained_recovery_steps(
    errors: torch.Tensor,
    shift_index: int,
    window: int = 4,
    tau: float = 0.5,
) -> int | None:
    r"""Compute sustained recovery latency following distribution shift.

    $$\text{SustainedRecoverySteps}(W) = \min \{ j \ge 0 : \|e_{t_{\text{shift}} + j + w}\|_2 \le \tau \cdot \|e_{t_{\text{shift}}}\|_2, \; \forall w \in \{0, \dots, W-1\} \}$$

    Requires the error to remain below the threshold for the next $W$ consecutive steps.

    Args:
        errors: Error tensor of shape `[T, V]`.
        shift_index: Time index $t_{\text{shift}}$ of the distribution shift.
        window: Number of consecutive steps $W$ error must remain contracted.
        tau: Contraction ratio threshold.

    Returns:
        Integer steps to enter sustained recovery, or None if not met within horizon.
    """
    post_shift_errs = errors[shift_index:]
    if len(post_shift_errs) < window:
        return None

    err_norms = torch.linalg.norm(post_shift_errs, dim=-1)
    e_ref = err_norms[0].item()

    if e_ref <= 1e-12:
        return 0

    threshold = tau * e_ref
    for j in range(len(err_norms) - window + 1):
        if (err_norms[j : j + window] <= threshold).all():
            return j

    return None


def extract_stability_telemetry(
    step_sizes: torch.Tensor,
    keys: torch.Tensor,
    dynamics_rates: torch.Tensor | float | None = None,
    features: torch.Tensor | None = None,
    content_updates: torch.Tensor | None = None,
    dynamics_updates: torch.Tensor | None = None,
    memory_tensors: list[torch.Tensor] | None = None,
) -> dict[str, Any]:
    r"""Summarize full stability telemetry across an unrolled sequence.

    Returns:
        Dictionary exposing:
            - max_normalized_step
            - min_stability_margin
            - max_normalized_dynamics_rate
            - min_dynamics_stability_margin
            - max_memory_norm
            - max_update_norm
            - finite_state
    """
    # 1. Normalized step sizes
    # Keys [T, K]
    k_sq = torch.sum(keys.detach().float() ** 2, dim=-1)
    s_vals = step_sizes.detach().float().view(-1)
    norm_steps = s_vals * k_sq
    max_norm_step = float(torch.max(norm_steps).item())
    min_margin = float(2.0 - max_norm_step)

    telemetry: dict[str, Any] = {
        "max_normalized_step": max_norm_step,
        "min_stability_margin": min_margin,
    }

    # 2. Dynamics rate telemetry
    if dynamics_rates is not None and features is not None:
        z_sq = torch.sum(features.detach().float() ** 2, dim=-1)
        if isinstance(dynamics_rates, torch.Tensor):
            r_vals = dynamics_rates.detach().float().view(-1)
        else:
            r_vals = torch.full_like(z_sq, float(dynamics_rates))
        norm_rates = r_vals * z_sq
        max_norm_rate = float(torch.max(norm_rates).item())
        telemetry["max_normalized_dynamics_rate"] = max_norm_rate
        telemetry["min_dynamics_stability_margin"] = float(2.0 - max_norm_rate)

    # 3. Update norms
    if content_updates is not None:
        if content_updates.ndim == 3:
            u_norms = torch.linalg.norm(content_updates.detach().float(), dim=(-2, -1))
        else:
            u_norms = torch.linalg.norm(content_updates.detach().float())
        telemetry["max_update_norm"] = float(torch.max(u_norms).item())

    # 4. Memory norms & finiteness tracking
    all_finite = True
    terminal_finite = True
    first_nonfinite: int | None = None
    max_m_norm = 0.0

    if memory_tensors is not None and len(memory_tensors) > 0:
        mem_norms = []
        for step_idx, m in enumerate(memory_tensors):
            m_fl = m.detach().float()
            step_finite = bool(torch.isfinite(m_fl).all().item())
            if not step_finite:
                all_finite = False
                if first_nonfinite is None:
                    first_nonfinite = step_idx
            mem_norms.append(torch.linalg.norm(m_fl).item())
        max_m_norm = float(max(mem_norms)) if mem_norms else 0.0
        terminal_finite = bool(
            torch.isfinite(memory_tensors[-1].detach().float()).all().item()
        )

    telemetry["max_memory_norm"] = max_m_norm
    telemetry["all_states_finite"] = all_finite
    telemetry["terminal_state_finite"] = terminal_finite
    telemetry["first_nonfinite_step"] = first_nonfinite
    telemetry["finite_state"] = all_finite
    return telemetry
