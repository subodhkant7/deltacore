"""Delta-rule (error-correcting) associative update rule.

Implements online least-mean-squares associative memory adaptation:
    v_hat = M @ k
    e = v - v_hat
    Delta M = eta * (e (x) k)
    M' = M + Delta M
"""

from dataclasses import dataclass
from typing import Generic, TypeVar, overload

import torch

from deltacore.memory.associative import AssociativeMemory
from deltacore.memory.read import read

MemoryT = TypeVar("MemoryT", AssociativeMemory, torch.Tensor)


@dataclass(frozen=True)
class DeltaStepResult(Generic[MemoryT]):
    r"""Structured diagnostic output for a single Delta-rule update step.

    Attributes:
        prediction: The retrieved prediction $\hat{v} = M k \in \mathbb{R}^V$.
        error: The residual error $e = v - \hat{v} \in \mathbb{R}^V$.
        outer_product: The error-key outer product $e k^\top \in \mathbb{R}^{V \times K}$.
        update: The scaled update matrix $\Delta M = \eta e k^\top \in \mathbb{R}^{V \times K}$.
        new_memory: The updated memory state $M' = M + \Delta M$.
    """

    prediction: torch.Tensor
    error: torch.Tensor
    outer_product: torch.Tensor
    update: torch.Tensor
    new_memory: MemoryT


class DeltaRule:
    r"""Delta-rule (error-correcting) associative update operator.

    Mathematical formulation:
        $$\hat{v}_t = M_t k_t$$
        $$e_t = v_t - \hat{v}_t$$
        $$\Delta M_t = \eta \, e_t k_t^\top$$
        $$M_{t+1} = M_t + \eta \, (v_t - M_t k_t) k_t^\top$$

    where:
        - $M_t \in \mathbb{R}^{V \times K}$ is the current associative memory state.
        - $k_t \in \mathbb{R}^K$ is the input query/key.
        - $v_t \in \mathbb{R}^V$ is the target value.
        - $\eta \ge 0$ is the non-negative scalar step size.

    Unlike Hebbian correlation learning, the delta rule computes directional
    updates proportional to the *prediction error*. If $M_t k_t = v_t$, then
    $e_t = 0$ and $\Delta M_t = 0$, guaranteeing that correctly recalled
    associations remain unaltered.

    The implementation preserves functional semantics and does NOT mutate the
    caller's memory state.
    """

    def __init__(self, step_size: float = 1.0) -> None:
        r"""Initialize DeltaRule with a default step size.

        Args:
            step_size: Non-negative scalar learning rate $\eta \ge 0$.

        Raises:
            ValueError: If step_size is negative.
        """
        if step_size < 0.0:
            raise ValueError(
                f"step_size must be non-negative (eta >= 0), got {step_size}. "
                f"Negative step sizes are rejected to prevent divergent dynamics."
            )
        self.default_step_size: float = float(step_size)

    @overload
    def step(
        self,
        memory: AssociativeMemory,
        key: torch.Tensor,
        target: torch.Tensor,
        step_size: float | torch.Tensor | None = None,
    ) -> DeltaStepResult[AssociativeMemory]: ...

    @overload
    def step(
        self,
        memory: torch.Tensor,
        key: torch.Tensor,
        target: torch.Tensor,
        step_size: float | torch.Tensor | None = None,
    ) -> DeltaStepResult[torch.Tensor]: ...

    @overload
    def step(
        self,
        memory: AssociativeMemory | torch.Tensor,
        key: torch.Tensor,
        target: torch.Tensor,
        step_size: float | torch.Tensor | None = None,
    ) -> DeltaStepResult[AssociativeMemory] | DeltaStepResult[torch.Tensor]: ...

    def step(
        self,
        memory: AssociativeMemory | torch.Tensor,
        key: torch.Tensor,
        target: torch.Tensor,
        step_size: float | torch.Tensor | None = None,
    ) -> DeltaStepResult[AssociativeMemory] | DeltaStepResult[torch.Tensor]:
        r"""Compute a delta-rule update and return full diagnostic intermediate values.

        Intermediate values exposed:
            - prediction: $\hat{v} = M k$
            - error: $e = v - \hat{v}$
            - outer_product: $e k^\top$
            - update: $\Delta M = \eta e k^\top$
            - new_memory: $M' = M + \Delta M$

        Args:
            memory: Current memory state as an AssociativeMemory or 2D/3D tensor.
            key: Key vector `[K]` or batched keys `[B, K]`.
            target: Target vector `[V]` or batched targets `[B, V]`.
            step_size: Optional step size $\eta \ge 0$. If None, uses default_step_size.

        Returns:
            DeltaStepResult with all 5 intermediate quantities.

        Raises:
            TypeError: If types or dtypes mismatch.
            ValueError: If shapes, devices, or step_size are invalid.
        """
        if step_size is None:
            eta = self.default_step_size
        elif isinstance(step_size, torch.Tensor):
            if bool((step_size < 0.0).any()):
                raise ValueError(
                    f"step_size must be non-negative (eta >= 0), got {step_size}. "
                    f"Negative step sizes are rejected."
                )
            eta = step_size
        else:
            eta = float(step_size)
            if eta < 0.0:
                raise ValueError(
                    f"step_size must be non-negative (eta >= 0), got {eta}. "
                    f"Negative step sizes are rejected."
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

        # 1. 2D Memory Case: [V, K]
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

            # Step 1: Read prediction: \hat{v} = M @ k
            prediction = read(m_data, key)

            # Step 2: Compute residual error: e = v - \hat{v}
            error = target - prediction

            # Step 3: Compute outer product: e (x) k
            outer_product = torch.outer(error, key)

            # Step 4: Scale update: \Delta M = \eta * outer_product
            delta_m = eta * outer_product

            # Step 5: Transition state: M' = M + \Delta M (pure functional addition)
            new_m = m_data + delta_m

        # 2. 3D Batched Memory Case: [B, V, K]
        elif m_data.ndim == 3:
            b_mem, v_dim, k_dim = m_data.shape
            if key.ndim != 2 or target.ndim != 2:
                raise ValueError(
                    f"For 3D memory [{b_mem}, {v_dim}, {k_dim}], key and target must be 2D, "
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

            # Step 1: Batched read prediction
            prediction = read(m_data, key)

            # Step 2: Batched error
            error = target - prediction

            # Step 3: Batched outer product: [B, V, 1] @ [B, 1, K] -> [B, V, K]
            outer_product = torch.bmm(error.unsqueeze(-1), key.unsqueeze(-2))

            # Step 4: Scale update
            delta_m = eta * outer_product

            # Step 5: Transition state
            new_m = m_data + delta_m

        else:
            raise ValueError(
                f"Unsupported memory ndim: expected 2 or 3, got {m_data.ndim}."
            )

        if is_state_obj:
            return DeltaStepResult(
                prediction=prediction,
                error=error,
                outer_product=outer_product,
                update=delta_m,
                new_memory=AssociativeMemory(new_m),
            )
        return DeltaStepResult(
            prediction=prediction,
            error=error,
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
        step_size: float | torch.Tensor | None = None,
    ) -> AssociativeMemory: ...

    @overload
    def update(
        self,
        memory: torch.Tensor,
        key: torch.Tensor,
        target: torch.Tensor,
        step_size: float | torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    @overload
    def update(
        self,
        memory: AssociativeMemory | torch.Tensor,
        key: torch.Tensor,
        target: torch.Tensor,
        step_size: float | torch.Tensor | None = None,
    ) -> AssociativeMemory | torch.Tensor: ...

    def update(
        self,
        memory: AssociativeMemory | torch.Tensor,
        key: torch.Tensor,
        target: torch.Tensor,
        step_size: float | torch.Tensor | None = None,
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
