# ==============================================================================
# DeltaCore: deltacore/scans/prefix.py
# Prefix scan over affine operators and memory state trajectories.
# ==============================================================================

import torch

from deltacore.scans.affine import AffineScanOperator


def affine_prefix_scan(
    operators: list[AffineScanOperator],
) -> list[AffineScanOperator]:
    r"""Compute prefix composition scan over a sequence of affine operators.

    Mathematical definition:
        Given operators $[F_0, F_1, \dots, F_{T-1}]$:
            $$P_0 = F_0$$
            $$P_t = P_{t-1} \otimes F_t = F_0 \otimes F_1 \otimes \dots \otimes F_t \quad \text{for } t \in [1, T-1]$$

    Applying $P_t$ directly to initial state $M_0$ produces:
        $$M_{t+1} = P_t(M_0) = (F_0 \otimes \dots \otimes F_t)(M_0) = F_t(\dots F_0(M_0))$$

    Args:
        operators: List of AffineScanOperator instances $[F_0, \dots, F_{T-1}]$.

    Returns:
        List of prefix AffineScanOperator instances $[P_0, \dots, P_{T-1}]$.
    """
    if not operators:
        return []

    prefix_ops: list[AffineScanOperator] = []
    current_prefix = operators[0]
    prefix_ops.append(current_prefix)

    for op in operators[1:]:
        current_prefix = current_prefix.compose(op)
        prefix_ops.append(current_prefix)

    return prefix_ops


def prefix_scan_memory(
    initial_memory: torch.Tensor,
    operators: list[AffineScanOperator],
    include_initial: bool = False,
) -> list[torch.Tensor]:
    r"""Evaluate state trajectory $M_t$ by applying prefix affine operators to $M_0$.

    Args:
        initial_memory: Memory state $M_0 \in \mathbb{R}^{V \times K}$ or `[B, V, K]`.
        operators: List of AffineScanOperator instances.
        include_initial: If True, prepends $M_0$ to the returned trajectory.

    Returns:
        List of memory tensors:
            If include_initial=True: $[M_0, M_1, \dots, M_T]$
            If include_initial=False: $[M_1, \dots, M_T]$
    """
    prefix_ops = affine_prefix_scan(operators)
    trajectory: list[torch.Tensor] = []

    if include_initial:
        trajectory.append(initial_memory)

    for p_op in prefix_ops:
        trajectory.append(p_op.apply(initial_memory))

    return trajectory
