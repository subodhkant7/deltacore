"""Hebbian outer-product associative update rule.

Implements the classical correlation update:
    Delta M = eta * (v (x) k)
    M' = M + Delta M
"""

from dataclasses import dataclass
from typing import Generic, TypeVar, overload

import torch

from deltacore.memory.associative import AssociativeMemory

MemoryT = TypeVar("MemoryT", AssociativeMemory, torch.Tensor)


@dataclass(frozen=True)
class HebbianStepResult(Generic[MemoryT]):
    r"""Structured diagnostic output for a single Hebbian update step.

    Attributes:
        outer_product: The outer product of target and key: $v k^\top \in \mathbb{R}^{V \times K}$.
        update: The scaled update matrix: $\Delta M = \eta v k^\top \in \mathbb{R}^{V \times K}$.
        new_memory: The updated memory state $M' = M + \Delta M$.
    """

    outer_product: torch.Tensor
    update: torch.Tensor
    new_memory: MemoryT


class HebbianRule:
    r"""Hebbian correlation-based update rule.

    Mathematical formulation:
        $$\Delta M_t = \eta v_t k_t^\top$$
        $$M_{t+1} = M_t + \eta v_t k_t^\top$$

    where $k_t \in \mathbb{R}^K$ is the key, $v_t \in \mathbb{R}^V$ is the target value,
    and $\eta \ge 0$ is the non-negative scalar step size.

    This rule binds key-target pairs via correlation outer products without
    error feedback or residual correction. It preserves functional semantics
    and does NOT mutate the caller's memory state.
    """

    def __init__(self, step_size: float = 1.0) -> None:
        r"""Initialize HebbianRule with a default step size.

        Args:
            step_size: Non-negative scalar learning rate $\eta \ge 0$.

        Raises:
            ValueError: If step_size is negative.
        """
        if step_size < 0.0:
            raise ValueError(
                f"step_size must be non-negative (eta >= 0), got {step_size}."
            )
        self.default_step_size: float = float(step_size)

    @overload
    def step(
        self,
        memory: AssociativeMemory,
        key: torch.Tensor,
        target: torch.Tensor,
        step_size: float | None = None,
    ) -> HebbianStepResult[AssociativeMemory]: ...

    @overload
    def step(
        self,
        memory: torch.Tensor,
        key: torch.Tensor,
        target: torch.Tensor,
        step_size: float | None = None,
    ) -> HebbianStepResult[torch.Tensor]: ...

    @overload
    def step(
        self,
        memory: AssociativeMemory | torch.Tensor,
        key: torch.Tensor,
        target: torch.Tensor,
        step_size: float | None = None,
    ) -> HebbianStepResult[AssociativeMemory] | HebbianStepResult[torch.Tensor]: ...

    def step(
        self,
        memory: AssociativeMemory | torch.Tensor,
        key: torch.Tensor,
        target: torch.Tensor,
        step_size: float | None = None,
    ) -> HebbianStepResult[AssociativeMemory] | HebbianStepResult[torch.Tensor]:
        r"""Compute a Hebbian update and return all intermediate diagnostic quantities.

        Args:
            memory: Current memory state as an AssociativeMemory instance or 2D/3D tensor.
            key: Key vector `[K]` or batched keys `[B, K]`.
            target: Target vector `[V]` or batched targets `[B, V]`.
            step_size: Optional step size $\eta$. If None, uses default_step_size.

        Returns:
            HebbianStepResult containing outer_product, update, and new_memory.

        Raises:
            TypeError: For unexpected types or dtype mismatches.
            ValueError: For shape, device, or negative step size errors.
        """
        eta = self.default_step_size if step_size is None else float(step_size)
        if eta < 0.0:
            raise ValueError(
                f"step_size must be non-negative (eta >= 0), got {eta}. "
                f"Negative step sizes are rejected to prevent divergent dynamics."
            )

        is_state_obj = isinstance(memory, AssociativeMemory)
        m_data = memory.data if is_state_obj else memory

        if not isinstance(m_data, torch.Tensor):
            raise TypeError(
                f"memory must be AssociativeMemory or torch.Tensor, got {type(memory).__name__}."
            )
        if not isinstance(key, torch.Tensor):
            raise TypeError(f"key must be torch.Tensor, got {type(key).__name__}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(
                f"target must be torch.Tensor, got {type(target).__name__}."
            )

        # Dtype checks
        if m_data.dtype != key.dtype or m_data.dtype != target.dtype:
            raise TypeError(
                f"Dtype mismatch: memory is {m_data.dtype}, key is {key.dtype}, "
                f"target is {target.dtype}."
            )

        # Device checks
        if m_data.device != key.device or m_data.device != target.device:
            raise ValueError(
                f"Device mismatch: memory is on {m_data.device}, key is on {key.device}, "
                f"target is on {target.device}."
            )

        # Shape validation
        if m_data.ndim == 2:
            v_dim, k_dim = m_data.shape
            if key.ndim != 1 or target.ndim != 1:
                raise ValueError(
                    f"For 2D memory [V={v_dim}, K={k_dim}], key and target must be 1D, "
                    f"got key.ndim={key.ndim}, target.ndim={target.ndim}."
                )
            if key.shape[0] != k_dim:
                raise ValueError(
                    f"Key dimension mismatch: expected {k_dim}, got {key.shape[0]}."
                )
            if target.shape[0] != v_dim:
                raise ValueError(
                    f"Target dimension mismatch: expected {v_dim}, got {target.shape[0]}."
                )

            # Outer product: [V] (x) [K] -> [V, K]
            outer_product = torch.outer(target, key)
            delta_m = eta * outer_product
            new_m = m_data + delta_m

        elif m_data.ndim == 3:
            b_mem, v_dim, k_dim = m_data.shape
            if key.ndim != 2 or target.ndim != 2:
                raise ValueError(
                    f"For 3D memory [{b_mem}, {v_dim}, {k_dim}], key and target must be 2D [B, K] and [B, V], "
                    f"got key.shape={list(key.shape)}, target.shape={list(target.shape)}."
                )
            if key.shape[0] != b_mem or target.shape[0] != b_mem:
                raise ValueError(
                    f"Batch dimension mismatch: memory batch is {b_mem}, "
                    f"but key batch is {key.shape[0]} and target batch is {target.shape[0]}."
                )
            if key.shape[1] != k_dim:
                raise ValueError(
                    f"Key dimension mismatch: expected {k_dim}, got {key.shape[1]}."
                )
            if target.shape[1] != v_dim:
                raise ValueError(
                    f"Target dimension mismatch: expected {v_dim}, got {target.shape[1]}."
                )

            # Batched outer product: [B, V, 1] @ [B, 1, K] -> [B, V, K]
            outer_product = torch.bmm(target.unsqueeze(-1), key.unsqueeze(-2))
            delta_m = eta * outer_product
            new_m = m_data + delta_m

        else:
            raise ValueError(
                f"Unsupported memory ndim: expected 2 or 3, got {m_data.ndim}."
            )

        if is_state_obj:
            return HebbianStepResult(
                outer_product=outer_product,
                update=delta_m,
                new_memory=AssociativeMemory(new_m),
            )
        return HebbianStepResult(
            outer_product=outer_product,
            update=delta_m,
            new_memory=new_m,
        )

    @overload
    def update(
        self,
        memory: AssociativeMemory,
        key: torch.Tensor,
        target: torch.Tensor,
        step_size: float | None = None,
    ) -> AssociativeMemory: ...

    @overload
    def update(
        self,
        memory: torch.Tensor,
        key: torch.Tensor,
        target: torch.Tensor,
        step_size: float | None = None,
    ) -> torch.Tensor: ...

    @overload
    def update(
        self,
        memory: AssociativeMemory | torch.Tensor,
        key: torch.Tensor,
        target: torch.Tensor,
        step_size: float | None = None,
    ) -> AssociativeMemory | torch.Tensor: ...

    def update(
        self,
        memory: AssociativeMemory | torch.Tensor,
        key: torch.Tensor,
        target: torch.Tensor,
        step_size: float | None = None,
    ) -> AssociativeMemory | torch.Tensor:
        r"""Compute and return the updated memory state directly.

        Args:
            memory: Current memory state.
            key: Key vector.
            target: Target vector.
            step_size: Optional step size.

        Returns:
            New memory state $M' = M + \Delta M$ without mutating input.
        """
        result = self.step(memory, key, target, step_size=step_size)
        return result.new_memory
