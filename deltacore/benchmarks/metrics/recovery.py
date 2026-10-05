# ==============================================================================
# DeltaCore: deltacore/benchmarks/metrics/recovery.py
# First-passage and sustained recovery metrics under distribution shifts.
# ==============================================================================

import torch


def compute_first_passage_recovery(
    errors: torch.Tensor,
    shift_index: int,
    tau: float = 0.5,
    eps: float = 1e-8,
) -> int | None:
    r"""Compute first-passage recovery steps following an abrupt distribution shift.

    Definition:
        T_{FP} = \min \{ j \ge 0 : \|e_{t_s + j}\|_2 \le \tau \|e_{t_s}\|_2 \}

    Args:
        errors: Sequence of prediction error vectors [T, V].
        shift_index: Time step t_s at which distribution shift occurred.
        tau: Recovery threshold fraction (default 0.5 = 50% error reduction).
        eps: Small epsilon preventing zero-shock division.

    Returns:
        Number of steps j >= 0 to achieve first passage, or None if recovery
        was not attained before the end of the sequence.
    """
    t_steps = errors.shape[0]
    if shift_index < 0 or shift_index >= t_steps:
        return None

    shock_norm = float(torch.linalg.norm(errors[shift_index]).item())
    target_thresh = tau * max(shock_norm, eps)

    for j in range(t_steps - shift_index):
        step_idx = shift_index + j
        curr_norm = float(torch.linalg.norm(errors[step_idx]).item())
        if curr_norm <= target_thresh:
            return j

    return None


def compute_sustained_recovery(
    errors: torch.Tensor,
    shift_index: int,
    window: int = 4,
    tau: float = 0.5,
    eps: float = 1e-8,
) -> int | None:
    r"""Compute sustained recovery steps following an abrupt distribution shift.

    Definition:
        T_{sustained}(W) = \min \{ j \ge 0 : \|e_{t_s + j + w}\|_2 \le \tau \|e_{t_s}\|_2, \forall w \in [0, W-1] \}

    Args:
        errors: Sequence of prediction error vectors [T, V].
        shift_index: Time step t_s at which distribution shift occurred.
        window: Number of consecutive steps W required to maintain low error.
        tau: Recovery threshold fraction.
        eps: Small epsilon preventing zero-shock division.

    Returns:
        Number of steps j >= 0 to achieve sustained recovery for window W,
        or None if sustained recovery was not achieved.
    """
    t_steps = errors.shape[0]
    if shift_index < 0 or shift_index >= t_steps or window <= 0:
        return None

    shock_norm = float(torch.linalg.norm(errors[shift_index]).item())
    target_thresh = tau * max(shock_norm, eps)

    post_shift_len = t_steps - shift_index
    if post_shift_len < window:
        return None

    for j in range(post_shift_len - window + 1):
        sustained = True
        for w in range(window):
            curr_norm = float(torch.linalg.norm(errors[shift_index + j + w]).item())
            if curr_norm > target_thresh:
                sustained = False
                break
        if sustained:
            return j

    return None
