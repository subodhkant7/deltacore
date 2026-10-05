r"""Adaptive delta-rule associative update operator.

Composes the core error-correcting update rule with dynamic step-size controllers:
    \hat{v}_t = M_t k_t
    e_t = v_t - \hat{v}_t
    \eta_t = controller(k_t, v_t, \hat{v}_t, e_t, M_t)
    \Delta M_t = \eta_t e_t k_t^\top
    M_{t+1} = M_t + \Delta M_t
"""

from dataclasses import dataclass
from typing import Generic, TypeVar, overload

import torch

from deltacore.memory.associative import AssociativeMemory
from deltacore.memory.read import read
from deltacore.updates.controllers import ConstantStepSize, StepSizeController
from deltacore.updates.delta import DeltaRule

MemoryT = TypeVar("MemoryT", AssociativeMemory, torch.Tensor)


@dataclass(frozen=True)
class AdaptiveDeltaStepResult(Generic[MemoryT]):
    r"""Structured diagnostic output for an adaptive Delta-rule update step.

    Attributes:
        prediction: Retrieved prediction $\hat{v} = M k$.
        error: Residual error $e = v - \hat{v}$.
        step_size: The effective step size $\eta_t \ge 0$ determined by the controller.
        outer_product: Error-key outer product $e k^\top$.
        update: Scaled update matrix $\Delta M = \eta_t e k^\top$.
        new_memory: The updated memory state $M' = M + \Delta M$.
    """

    prediction: torch.Tensor
    error: torch.Tensor
    step_size: torch.Tensor
    outer_product: torch.Tensor
    update: torch.Tensor
    new_memory: MemoryT


class AdaptiveDeltaRule:
    r"""Adaptive Delta-rule operator with independent step-size control.

    This abstraction decouples the update formulation (WHAT the update is)
    from the step-size policy (HOW LARGE the update should be):
        - DeltaRule defines the error-correcting direction: $e_t k_t^\top$.
        - StepSizeController computes the scalar/batched learning rate: $\eta_t$.

    The implementation preserves functional semantics and does NOT mutate the
    caller's memory state.
    """

    def __init__(
        self,
        controller: StepSizeController | None = None,
        base_rule: DeltaRule | None = None,
    ) -> None:
        """Initialize AdaptiveDeltaRule with a step-size controller.

        Args:
            controller: A StepSizeController instance (defaults to ConstantStepSize(1.0)).
            base_rule: Optional DeltaRule instance (for reference).
        """
        self.controller: StepSizeController = (
            controller if controller is not None else ConstantStepSize(1.0)
        )
        self.base_rule: DeltaRule = base_rule if base_rule is not None else DeltaRule()

    @overload
    def step(
        self,
        memory: AssociativeMemory,
        key: torch.Tensor,
        target: torch.Tensor,
    ) -> AdaptiveDeltaStepResult[AssociativeMemory]: ...

    @overload
    def step(
        self,
        memory: torch.Tensor,
        key: torch.Tensor,
        target: torch.Tensor,
    ) -> AdaptiveDeltaStepResult[torch.Tensor]: ...

    @overload
    def step(
        self,
        memory: AssociativeMemory | torch.Tensor,
        key: torch.Tensor,
        target: torch.Tensor,
    ) -> (
        AdaptiveDeltaStepResult[AssociativeMemory]
        | AdaptiveDeltaStepResult[torch.Tensor]
    ): ...

    def step(
        self,
        memory: AssociativeMemory | torch.Tensor,
        key: torch.Tensor,
        target: torch.Tensor,
    ) -> (
        AdaptiveDeltaStepResult[AssociativeMemory]
        | AdaptiveDeltaStepResult[torch.Tensor]
    ):
        r"""Compute an adaptive delta update step exposing all diagnostic quantities.

        Args:
            memory: Current memory state as an AssociativeMemory or 2D/3D tensor.
            key: Key vector `[K]` or batched `[B, K]`.
            target: Target vector `[V]` or batched `[B, V]`.

        Returns:
            AdaptiveDeltaStepResult containing prediction, error, step_size,
            outer_product, update, and new_memory.
        """
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

        # Step 1: Read current prediction: \hat{v} = M @ k
        prediction = read(m_data, key)

        # Step 2: Compute prediction error: e = v - \hat{v}
        error = target - prediction

        # Step 3: Compute dynamic step size via the controller: \eta_t
        step_size = self.controller(
            key=key,
            target=target,
            prediction=prediction,
            error=error,
            memory=memory,
        )

        if (step_size < 0.0).any():
            raise ValueError(
                f"StepSizeController emitted negative step size: {step_size}. "
                f"Negative step sizes are rejected."
            )

        # Step 4: Compute outer product and scale by step_size
        if m_data.ndim == 2:
            # Unbatched: key is [K], error is [V]
            outer_product = torch.outer(error, key)
            delta_m = step_size * outer_product
            new_m = m_data + delta_m

        elif m_data.ndim == 3:
            # Batched: key is [B, K], error is [B, V]
            b_mem, v_dim, k_dim = m_data.shape
            outer_product = torch.bmm(error.unsqueeze(-1), key.unsqueeze(-2))

            # Reshape step_size if needed: [B] -> [B, 1, 1]
            if step_size.ndim == 0:
                scale = step_size
            elif step_size.ndim == 1:
                scale = step_size.view(-1, 1, 1)
            else:
                scale = step_size

            delta_m = scale * outer_product
            new_m = m_data + delta_m
        else:
            raise ValueError(
                f"Unsupported memory dimensions: expected 2D or 3D, got {m_data.ndim}."
            )

        if is_state_obj:
            return AdaptiveDeltaStepResult(
                prediction=prediction,
                error=error,
                step_size=step_size,
                outer_product=outer_product,
                update=delta_m,
                new_memory=AssociativeMemory(new_m),
            )
        return AdaptiveDeltaStepResult(
            prediction=prediction,
            error=error,
            step_size=step_size,
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
    ) -> AssociativeMemory: ...

    @overload
    def update(
        self,
        memory: torch.Tensor,
        key: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor: ...

    @overload
    def update(
        self,
        memory: AssociativeMemory | torch.Tensor,
        key: torch.Tensor,
        target: torch.Tensor,
    ) -> AssociativeMemory | torch.Tensor: ...

    def update(
        self,
        memory: AssociativeMemory | torch.Tensor,
        key: torch.Tensor,
        target: torch.Tensor,
    ) -> AssociativeMemory | torch.Tensor:
        r"""Compute and return updated memory state directly.

        Args:
            memory: Current memory state.
            key: Key vector.
            target: Target vector.

        Returns:
            New memory state $M' = M + \Delta M$ without mutating input.
        """
        result = self.step(memory, key, target)
        return result.new_memory
