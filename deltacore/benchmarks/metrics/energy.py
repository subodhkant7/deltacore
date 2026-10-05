# ==============================================================================
# DeltaCore: deltacore/benchmarks/metrics/energy.py
# Adaptation budget, state growth, and energy metrics.
# ==============================================================================


def compute_update_energy(update_norms: list[float]) -> float:
    r"""Compute total update energy U_M = \sum_t \|\Delta M_t\|_F."""
    return float(sum(update_norms))


def compute_step_energy(step_sizes: list[float]) -> float:
    r"""Compute total step energy U_\eta = \sum_t \eta_t."""
    return float(sum(step_sizes))


def compute_state_growth_ratio(
    memory_norms: list[float],
    shift_index: int = 0,
    eps: float = 1e-6,
) -> float:
    r"""Compute state growth ratio G_M = \max_{t \ge t_s} \|M_t\|_F / \max(\|M_{t_s}\|_F, \epsilon).

    Args:
        memory_norms: Trajectory of memory Frobenius norms.
        shift_index: Baseline time index t_s.
        eps: Small stabilizer.

    Returns:
        Growth ratio relative to the state norm at t_s.
    """
    if not memory_norms:
        return 1.0

    idx = max(0, min(shift_index, len(memory_norms) - 1))
    base_norm = max(memory_norms[idx], eps)
    post_norms = memory_norms[idx:]
    max_post = max(post_norms) if post_norms else base_norm
    return float(max_post / base_norm)


def compute_min_stability_margin(stability_margins: list[float]) -> float:
    r"""Compute the minimum stability margin \min_t (2 - \eta_t \|k_t\|^2)."""
    return float(min(stability_margins)) if stability_margins else 2.0


def compute_max_normalized_step(normalized_steps: list[float]) -> float:
    r"""Compute the maximum normalized step \max_t (\eta_t \|k_t\|^2)."""
    return float(max(normalized_steps)) if normalized_steps else 0.0
