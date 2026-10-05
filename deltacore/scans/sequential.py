r"""Reference sequential scan for associative state evolution over time.

Iterates sequentially over a sequence of key-target pairs, updating the memory
state at each time step and emitting step-wise predictions:
    For t = 0 to T-1:
        1. read current memory: \hat{v}_t = M_t @ k_t
        2. compute error: e_t = v_t - \hat{v}_t
        3. compute update: \Delta M_t = \eta * e_t (x) k_t
        4. transition state: M_{t+1} = M_t + \Delta M_t
        5. record prediction \hat{v}_t
"""

from collections.abc import Iterator
from dataclasses import dataclass

import torch

from deltacore.memory.associative import AssociativeMemory
from deltacore.memory.read import read
from deltacore.updates.delta import DeltaRule
from deltacore.updates.hebbian import HebbianRule


@dataclass(frozen=True)
class SequentialScanResult:
    r"""Output of a sequential associative scan.

    Attributes:
        predictions: Retrieved predictions emitted before each step's update,
            of shape `[B, T, V]` (or `[T, V]` for unbatched inputs).
        final_memory: The terminal memory state $M_T$ after consuming all $T$ steps.
    """

    predictions: torch.Tensor
    final_memory: AssociativeMemory | torch.Tensor

    def __iter__(self) -> Iterator[torch.Tensor | AssociativeMemory]:
        """Allow tuple unpacking: predictions, final_mem = sequential_scan(...)."""
        yield self.predictions
        yield self.final_memory


def sequential_scan(
    keys: torch.Tensor,
    targets: torch.Tensor,
    initial_memory: AssociativeMemory | torch.Tensor | None = None,
    rule: DeltaRule | HebbianRule | None = None,
    step_size: float | None = None,
) -> SequentialScanResult:
    r"""Execute a reference sequential scan across a sequence of keys and targets.

    At each time step $t \in \{0, \dots, T-1\}$:
        1. Retrieve prediction from current memory: $\hat{v}_t = M_t k_t$.
        2. Compute prediction error: $e_t = v_t - \hat{v}_t$.
        3. Transition memory state via the chosen update rule: $M_{t+1} = M_t + \Delta M_t$.
        4. Emit prediction $\hat{v}_t$.

    This is an explicit reference implementation prioritizing numerical
    correctness and mathematical inspectability over speed.

    Args:
        keys: Key tensor of shape `[B, T, K]` (batched) or `[T, K]` (unbatched).
        targets: Target tensor of shape `[B, T, V]` (batched) or `[T, V]` (unbatched).
        initial_memory: Optional initial state $M_0$. If None, initialized to zeros
            matching keys device and dtype.
        rule: Update rule instance (defaults to DeltaRule()).
        step_size: Optional scalar learning rate override.

    Returns:
        SequentialScanResult containing predictions and final memory state.

    Raises:
        TypeError: If tensor types or dtypes mismatch.
        ValueError: If tensor shapes, devices, or sequence lengths are inconsistent.
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
            f"keys must be 2D [T, K] or 3D [B, T, K], got ndim={keys.ndim} with shape {list(keys.shape)}."
        )

    if targets.ndim != keys.ndim:
        raise ValueError(
            f"Dimensionality mismatch: keys has ndim={keys.ndim}, but targets has ndim={targets.ndim}."
        )

    is_unbatched = keys.ndim == 2
    if is_unbatched:
        # [T, K] -> [1, T, K] and [T, V] -> [1, T, V]
        keys_b = keys.unsqueeze(0)
        targets_b = targets.unsqueeze(0)
    else:
        keys_b = keys
        targets_b = targets

    batch_size, seq_len, k_dim = keys_b.shape
    t_batch, t_seq, v_dim = targets_b.shape

    if batch_size != t_batch:
        raise ValueError(
            f"Batch size mismatch: keys batch={batch_size}, targets batch={t_batch}."
        )
    if seq_len != t_seq:
        raise ValueError(
            f"Sequence length mismatch: keys T={seq_len}, targets T={t_seq}."
        )

    # Initialize rule
    active_rule = DeltaRule() if rule is None else rule

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

        if not isinstance(m_data, torch.Tensor):
            raise TypeError(
                f"initial_memory must be AssociativeMemory or torch.Tensor, got {type(initial_memory).__name__}."
            )
        if m_data.dtype != keys.dtype:
            raise TypeError(
                f"Dtype mismatch: initial_memory has {m_data.dtype}, but keys has {keys.dtype}."
            )
        if m_data.device != keys.device:
            raise ValueError(
                f"Device mismatch: initial_memory is on {m_data.device}, but keys is on {keys.device}."
            )

        if m_data.ndim == 2:
            # Shared 2D memory across batch: expand to [B, V, K]
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

    # Sequential iteration over sequence length T
    for t in range(seq_len):
        kt = keys_b[:, t, :]  # [B, K]
        vt = targets_b[:, t, :]  # [B, V]

        # Step 1: Read current prediction
        m_tensor = (
            current_memory.data
            if isinstance(current_memory, AssociativeMemory)
            else current_memory
        )
        pred_t = read(m_tensor, kt)  # [B, V]
        predictions_list.append(pred_t)

        # Step 2: Update memory for next step
        current_memory = active_rule.update(
            memory=current_memory,
            key=kt,
            target=vt,
            step_size=step_size,
        )

    # Stack emitted predictions: list of [B, V] -> [B, T, V]
    if seq_len > 0:
        stacked_preds = torch.stack(predictions_list, dim=1)
    else:
        stacked_preds = torch.empty(
            (batch_size, 0, v_dim), dtype=keys.dtype, device=keys.device
        )

    # Handle unbatched shape alignment
    if is_unbatched:
        stacked_preds = stacked_preds.squeeze(0)  # [T, V]
        if isinstance(current_memory, AssociativeMemory):
            final_memory = AssociativeMemory(current_memory.data.squeeze(0))
        else:
            final_memory = current_memory.squeeze(0)
    else:
        final_memory = current_memory

    return SequentialScanResult(
        predictions=stacked_preds,
        final_memory=final_memory,
    )
