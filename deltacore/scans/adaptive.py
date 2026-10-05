r"""Adaptive sequential scan executing dynamic associative state evolution.

Iterates over sequence steps using AdaptiveDeltaRule, recording step-size telemetry,
emitted predictions, and terminal memory state.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import torch

from deltacore.memory.associative import AssociativeMemory
from deltacore.updates.adaptive_delta import AdaptiveDeltaRule
from deltacore.updates.controllers import StepSizeController


@dataclass(frozen=True)
class AdaptiveScanResult:
    r"""Diagnostic output of an adaptive sequential scan.

    Attributes:
        predictions: Retrieved predictions emitted before each step's update,
            of shape `[B, T, V]` (or `[T, V]` for unbatched).
        final_memory: The terminal memory state $M_T$ after consuming all $T$ steps.
        step_sizes: The effective learning rate $\eta_t$ used at each step,
            of shape `[B, T]` (or `[T]` for unbatched).
        errors: The residual error vectors $e_t = v_t - \hat{v}_t$, of shape
            `[B, T, V]` (or `[T, V]`).
        update_norms: The Frobenius norm of each update $\|\Delta M_t\|_F$,
            of shape `[B, T]` (or `[T]`).
    """

    predictions: torch.Tensor
    final_memory: AssociativeMemory | torch.Tensor
    step_sizes: torch.Tensor
    errors: torch.Tensor
    update_norms: torch.Tensor

    def __iter__(
        self,
    ) -> Iterator[torch.Tensor | AssociativeMemory]:
        """Allow tuple unpacking: predictions, final_mem, step_sizes = adaptive_scan(...)."""
        yield self.predictions
        yield self.final_memory
        yield self.step_sizes


def adaptive_scan(
    keys: torch.Tensor,
    targets: torch.Tensor,
    initial_memory: AssociativeMemory | torch.Tensor | None = None,
    rule: AdaptiveDeltaRule | None = None,
    controller: StepSizeController | None = None,
) -> AdaptiveScanResult:
    r"""Execute a reference sequential scan with adaptive step-size dynamics.

    Args:
        keys: Key tensor `[B, T, K]` (batched) or `[T, K]` (unbatched).
        targets: Target tensor `[B, T, V]` (batched) or `[T, V]` (unbatched).
        initial_memory: Optional initial state $M_0$. If None, initialized to zeros.
        rule: Optional AdaptiveDeltaRule instance.
        controller: Optional StepSizeController (used if rule is None).

    Returns:
        AdaptiveScanResult containing predictions, final_memory, step_sizes,
        errors, and update_norms.
    """
    if not isinstance(keys, torch.Tensor):
        raise TypeError(f"keys must be a torch.Tensor, got {type(keys).__name__}.")
    if not isinstance(targets, torch.Tensor):
        raise TypeError(
            f"targets must be a torch.Tensor, got {type(targets).__name__}."
        )

    if keys.dtype != targets.dtype:
        raise TypeError(
            f"Dtype mismatch: keys has {keys.dtype}, but targets has {targets.dtype}."
        )
    if keys.device != targets.device:
        raise ValueError(
            f"Device mismatch: keys is on {keys.device}, but targets is on {targets.device}."
        )

    if keys.ndim not in (2, 3):
        raise ValueError(
            f"keys must be 2D [T, K] or 3D [B, T, K], got ndim={keys.ndim}."
        )
    if targets.ndim != keys.ndim:
        raise ValueError(
            f"Dimensionality mismatch: keys ndim={keys.ndim}, targets ndim={targets.ndim}."
        )

    is_unbatched = keys.ndim == 2
    if is_unbatched:
        keys_b = keys.unsqueeze(0)
        targets_b = targets.unsqueeze(0)
    else:
        keys_b = keys
        targets_b = targets

    batch_size, seq_len, k_dim = keys_b.shape
    t_batch, t_seq, v_dim = targets_b.shape

    if batch_size != t_batch:
        raise ValueError(f"Batch size mismatch: keys={batch_size}, targets={t_batch}.")
    if seq_len != t_seq:
        raise ValueError(f"Sequence length mismatch: keys={seq_len}, targets={t_seq}.")

    # Configure rule
    if rule is not None:
        active_rule = rule
    elif controller is not None:
        active_rule = AdaptiveDeltaRule(controller=controller)
    else:
        active_rule = AdaptiveDeltaRule()

    # Initialize memory state
    if initial_memory is None:
        current_memory: AssociativeMemory | torch.Tensor = AssociativeMemory.zeros(
            v_dim=v_dim,
            k_dim=k_dim,
            dtype=keys.dtype,
            device=keys.device,
            batch_size=batch_size,
        )
    else:
        is_obj = isinstance(initial_memory, AssociativeMemory)
        m_data = initial_memory.data if is_obj else initial_memory

        if m_data.ndim == 2:
            if m_data.shape != (v_dim, k_dim):
                raise ValueError(
                    f"initial_memory shape mismatch: expected [{v_dim}, {k_dim}], got {list(m_data.shape)}."
                )
            batched_data = m_data.unsqueeze(0).expand(batch_size, -1, -1).clone()
            current_memory = AssociativeMemory(batched_data) if is_obj else batched_data
        elif m_data.ndim == 3:
            if m_data.shape != (batch_size, v_dim, k_dim):
                raise ValueError(
                    f"initial_memory shape mismatch: expected [{batch_size}, {v_dim}, {k_dim}], "
                    f"got {list(m_data.shape)}."
                )
            current_memory = initial_memory.clone() if is_obj else m_data.clone()
        else:
            raise ValueError(
                f"initial_memory must be 2D or 3D, got ndim={m_data.ndim}."
            )

    predictions_list = []
    step_sizes_list = []
    errors_list = []
    update_norms_list = []

    for t in range(seq_len):
        kt = keys_b[:, t, :]  # [B, K]
        vt = targets_b[:, t, :]  # [B, V]

        step_res = active_rule.step(current_memory, kt, vt)

        predictions_list.append(step_res.prediction)
        errors_list.append(step_res.error)

        # Step size tensor: ensure shape [B]
        eta = step_res.step_size
        if eta.ndim == 0:
            eta = eta.expand(batch_size)
        step_sizes_list.append(eta)

        # Update norm: Frobenius norm over (-2, -1) -> [B]
        delta_m = step_res.update
        if delta_m.ndim == 2:
            norm_val = torch.linalg.norm(delta_m).expand(batch_size)
        else:
            norm_val = torch.linalg.norm(delta_m, dim=(-2, -1))
        update_norms_list.append(norm_val)

        current_memory = step_res.new_memory

    if seq_len > 0:
        stacked_preds = torch.stack(predictions_list, dim=1)  # [B, T, V]
        stacked_etas = torch.stack(step_sizes_list, dim=1)  # [B, T]
        stacked_errs = torch.stack(errors_list, dim=1)  # [B, T, V]
        stacked_norms = torch.stack(update_norms_list, dim=1)  # [B, T]
    else:
        stacked_preds = torch.empty(
            (batch_size, 0, v_dim), dtype=keys.dtype, device=keys.device
        )
        stacked_etas = torch.empty(
            (batch_size, 0), dtype=keys.dtype, device=keys.device
        )
        stacked_errs = torch.empty(
            (batch_size, 0, v_dim), dtype=keys.dtype, device=keys.device
        )
        stacked_norms = torch.empty(
            (batch_size, 0), dtype=keys.dtype, device=keys.device
        )

    if is_unbatched:
        stacked_preds = stacked_preds.squeeze(0)
        stacked_etas = stacked_etas.squeeze(0)
        stacked_errs = stacked_errs.squeeze(0)
        stacked_norms = stacked_norms.squeeze(0)
        if isinstance(current_memory, AssociativeMemory):
            final_memory = AssociativeMemory(current_memory.data.squeeze(0))
        else:
            final_memory = current_memory.squeeze(0)
    else:
        final_memory = current_memory

    return AdaptiveScanResult(
        predictions=stacked_preds,
        final_memory=final_memory,
        step_sizes=stacked_etas,
        errors=stacked_errs,
        update_norms=stacked_norms,
    )
