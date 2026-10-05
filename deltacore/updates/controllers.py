r"""Adaptive step-size controllers for DeltaCore state transitions.

Establishes the explicit step-size modulation abstraction:
    \eta_t = f_\theta(z_t)
where z_t is a documented feature of the current input, target, error, or memory.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from deltacore.memory.associative import AssociativeMemory


class StepSizeController(nn.Module, ABC):
    r"""Abstract base class for all step-size controllers.

    Defines the contract for computing dynamic or static learning rates:
        $$\eta_t = f(k_t, v_t, \hat{v}_t, e_t, M_t)$$

    Controllers must be explicit, functional, and non-mutating with respect to
    the memory state.
    """

    @abstractmethod
    def forward(
        self,
        key: torch.Tensor,
        target: torch.Tensor,
        prediction: torch.Tensor,
        error: torch.Tensor,
        memory: AssociativeMemory | torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute the scalar or batched effective step size $\eta_t$.

        Args:
            key: Input query/key vector `[K]` or batched `[B, K]`.
            target: Target value vector `[V]` or batched `[B, V]`.
            prediction: Current retrieved prediction $\hat{v} = M k$.
            error: Residual error $e = v - \hat{v}$.
            memory: Current memory state container or tensor.

        Returns:
            Scalar or batched tensor representing non-negative step size $\eta_t \ge 0$.
        """
        pass


class ConstantStepSize(StepSizeController):
    r"""Controller A: Constant (fixed) step size.

    Mathematical definition:
        $$\eta_t = \eta_0$$

    Input Dependencies:
        None. Step size is fixed regardless of inputs or memory state.

    Output Range:
        $\eta_t = \eta_0$ (strictly non-negative).

    Parameters:
        step_size ($\eta_0$): Scalar non-negative float.
    """

    step_size: torch.Tensor

    def __init__(self, step_size: float = 1.0) -> None:
        super().__init__()
        if step_size < 0.0:
            raise ValueError(
                f"step_size must be non-negative, got {step_size}. "
                f"Negative step sizes are rejected."
            )
        self.register_buffer(
            "step_size", torch.tensor(float(step_size), dtype=torch.float32)
        )

    def forward(
        self,
        key: torch.Tensor,
        target: torch.Tensor,
        prediction: torch.Tensor,
        error: torch.Tensor,
        memory: AssociativeMemory | torch.Tensor,
    ) -> torch.Tensor:
        """Return the constant step size cast to key dtype and device."""
        return self.step_size.to(dtype=key.dtype, device=key.device)


class InputConditionedStepSize(StepSizeController):
    r"""Controller B: Step size conditioned on input/key vector.

    Mathematical definition:
        $$\eta_t = \eta_{\max} \cdot \sigma(w^\top x_t + b)$$

    Input Dependencies:
        Key vector $x_t = k_t \in \mathbb{R}^K$.
        Independent of target, error, or memory state.

    Output Range:
        $$0 < \eta_t < \eta_{\max}$$

    Parameters:
        dim ($K$): Dimensionality of input feature space.
        eta_max ($\eta_{\max}$): Maximum allowable step size ($> 0$).
        weight ($w$): Weight vector in $\mathbb{R}^K$.
        bias ($b$): Scalar bias in $\mathbb{R}$.

    Differentiability:
        Smoothly differentiable with respect to $w, b$, and $x_t$.
    """

    def __init__(
        self,
        dim: int,
        eta_max: float = 1.0,
        weight: torch.Tensor | None = None,
        bias: float = 0.0,
    ) -> None:
        super().__init__()
        if eta_max <= 0.0:
            raise ValueError(f"eta_max must be strictly positive, got {eta_max}.")
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}.")

        self.eta_max = float(eta_max)
        self.weight = nn.Parameter(
            weight.clone().detach() if weight is not None else torch.zeros(dim)
        )
        self.bias = nn.Parameter(torch.tensor(float(bias)))

    def forward(
        self,
        key: torch.Tensor,
        target: torch.Tensor,
        prediction: torch.Tensor,
        error: torch.Tensor,
        memory: AssociativeMemory | torch.Tensor,
    ) -> torch.Tensor:
        """Compute input-conditioned step size."""
        # Align parameter precision with input key
        w = self.weight.to(dtype=key.dtype, device=key.device)
        b = self.bias.to(dtype=key.dtype, device=key.device)

        if key.ndim == 1:
            z = torch.dot(key, w) + b
            return self.eta_max * torch.sigmoid(z)
        elif key.ndim == 2:
            # Batched: [B, K] @ [K] -> [B]
            z = key @ w + b
            return self.eta_max * torch.sigmoid(z)
        else:
            raise ValueError(f"Expected key ndim to be 1 or 2, got ndim={key.ndim}.")


class ErrorConditionedStepSize(StepSizeController):
    r"""Controller C: Step size conditioned on residual prediction error norm.

    Mathematical definition:
        $$\eta_t = \eta_{\max} \cdot \sigma(a \|e_t\|_2 + b)$$

    Input Dependencies:
        Euclidean norm of current residual error $e_t = v_t - \hat{v}_t$.

    Output Range:
        $$0 < \eta_t < \eta_{\max}$$

    Parameters:
        eta_max ($\eta_{\max}$): Maximum allowable step size ($> 0$).
        scale ($a$): Scaling coefficient for error norm.
        bias ($b$): Scalar offset.
        eps: Small non-negative epsilon preventing gradient singularity at zero.

    Differentiability:
        Smoothly differentiable with respect to $a, b$, and $e_t$.
    """

    def __init__(
        self,
        eta_max: float = 1.0,
        scale: float = 1.0,
        bias: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if eta_max <= 0.0:
            raise ValueError(f"eta_max must be strictly positive, got {eta_max}.")

        self.eta_max = float(eta_max)
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.bias = nn.Parameter(torch.tensor(float(bias)))
        self.eps = float(eps)

    def forward(
        self,
        key: torch.Tensor,
        target: torch.Tensor,
        prediction: torch.Tensor,
        error: torch.Tensor,
        memory: AssociativeMemory | torch.Tensor,
    ) -> torch.Tensor:
        """Compute error-conditioned step size."""
        a = self.scale.to(dtype=error.dtype, device=error.device)
        b = self.bias.to(dtype=error.dtype, device=error.device)

        if error.ndim == 1:
            err_norm = torch.sqrt(torch.sum(error**2) + self.eps)
            z = a * err_norm + b
            return self.eta_max * torch.sigmoid(z)
        elif error.ndim == 2:
            # Batched: [B, V] -> norm over dim -1 -> [B]
            err_norm = torch.sqrt(torch.sum(error**2, dim=-1) + self.eps)
            z = a * err_norm + b
            return self.eta_max * torch.sigmoid(z)
        else:
            raise ValueError(
                f"Expected error ndim to be 1 or 2, got ndim={error.ndim}."
            )


class StateConditionedStepSize(StepSizeController):
    r"""Controller D: Step size conditioned on memory Frobenius norm.

    Mathematical definition:
        $$\eta_t = \eta_{\max} \cdot \sigma(a \|M_t\|_F + b)$$

    Input Dependencies:
        Frobenius norm of the current associative memory matrix $M_t$.

    Output Range:
        $$0 < \eta_t < \eta_{\max}$$

    Parameters:
        eta_max ($\eta_{\max}$): Maximum allowable step size ($> 0$).
        scale ($a$): Scaling coefficient for memory norm.
        bias ($b$): Scalar offset.
        eps: Small non-negative epsilon preventing gradient singularity at zero.

    Differentiability:
        Smoothly differentiable with respect to $a, b$, and $M_t$.
    """

    def __init__(
        self,
        eta_max: float = 1.0,
        scale: float = 1.0,
        bias: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if eta_max <= 0.0:
            raise ValueError(f"eta_max must be strictly positive, got {eta_max}.")

        self.eta_max = float(eta_max)
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.bias = nn.Parameter(torch.tensor(float(bias)))
        self.eps = float(eps)

    def forward(
        self,
        key: torch.Tensor,
        target: torch.Tensor,
        prediction: torch.Tensor,
        error: torch.Tensor,
        memory: AssociativeMemory | torch.Tensor,
    ) -> torch.Tensor:
        """Compute state-conditioned step size."""
        m_data = memory.data if isinstance(memory, AssociativeMemory) else memory

        a = self.scale.to(dtype=m_data.dtype, device=m_data.device)
        b = self.bias.to(dtype=m_data.dtype, device=m_data.device)

        if m_data.ndim == 2:
            frob_norm = torch.sqrt(torch.sum(m_data**2) + self.eps)
            z = a * frob_norm + b
            return self.eta_max * torch.sigmoid(z)
        elif m_data.ndim == 3:
            # Batched memory: [B, V, K] -> norm over (-2, -1) -> [B]
            frob_norm = torch.sqrt(torch.sum(m_data**2, dim=(-2, -1)) + self.eps)
            z = a * frob_norm + b
            return self.eta_max * torch.sigmoid(z)
        else:
            raise ValueError(
                f"Expected memory ndim to be 2 or 3, got ndim={m_data.ndim}."
            )
