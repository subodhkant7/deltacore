r"""Stability controllers and non-expansion bounds for adaptive memory systems.

Provides mathematical controllers that bound step sizes and dynamics rates to
guarantee local non-expansion of prediction residuals:
    - Content Memory: \eta_t \|k_t\|_2^2 \le \beta < 2
    - Dynamics Memory: \rho_t \|z_t\|_2^2 \le \beta_C < 2
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class StabilityConstraintResult:
    r"""Structured output of a stability constraint evaluation.

    Attributes:
        raw_value: Unconstrained input step size or rate.
        safe_value: Constrained safe value (\le bound).
        clipped: Boolean tensor indicating whether safety clamping was active.
        normalized_value: Scaled value (\eta \|k\|^2 or \rho \|z\|^2).
        stability_margin: Distance to theoretical divergence boundary (2.0 - normalized_value).
        bound: Theoretical upper bound \beta / (\|x\|^2 + \epsilon).
    """

    raw_value: torch.Tensor
    safe_value: torch.Tensor
    clipped: torch.Tensor
    normalized_value: torch.Tensor
    stability_margin: torch.Tensor
    bound: torch.Tensor


class BaseStabilityController(ABC, nn.Module):
    r"""Abstract base class for memory stability controllers."""

    @abstractmethod
    def safe_step(
        self, raw_step_size: torch.Tensor | float, key: torch.Tensor
    ) -> StabilityConstraintResult:
        r"""Constrain content memory step size \eta_t given query key k_t."""
        pass

    @abstractmethod
    def safe_rate(
        self, raw_rate: torch.Tensor | float, features: torch.Tensor
    ) -> StabilityConstraintResult:
        r"""Constrain dynamics memory learning rate \rho_t given features z_t."""
        pass


class UnconstrainedController(BaseStabilityController):
    r"""Identity controller that passes through raw values without modification.

    Maintains telemetry (normalized step, stability margin) for passive observability
    without enforcing clipping constraints.
    """

    def __init__(self, check_finite: bool = False) -> None:
        super().__init__()
        self.check_finite = check_finite

    def safe_step(
        self, raw_step_size: torch.Tensor | float, key: torch.Tensor
    ) -> StabilityConstraintResult:
        raw_t = (
            torch.as_tensor(raw_step_size, dtype=key.dtype, device=key.device)
            if not isinstance(raw_step_size, torch.Tensor)
            else raw_step_size
        )

        if self.check_finite:
            if not torch.isfinite(raw_t).all() or not torch.isfinite(key).all():
                raise FloatingPointError(
                    f"Non-finite input detected in UnconstrainedController.safe_step: "
                    f"raw_step_size finite={torch.isfinite(raw_t).all().item()}, "
                    f"key finite={torch.isfinite(key).all().item()}."
                )

        if key.ndim == 1:
            k_norm_sq = torch.sum(key**2)
        elif key.ndim == 2:
            k_norm_sq = torch.sum(key**2, dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unsupported key ndim={key.ndim}.")

        norm_step = raw_t * k_norm_sq
        margin = 2.0 - norm_step
        clipped = torch.zeros_like(raw_t, dtype=torch.bool)
        bound = torch.full_like(raw_t, float("inf"))

        return StabilityConstraintResult(
            raw_value=raw_t,
            safe_value=raw_t,
            clipped=clipped,
            normalized_value=norm_step,
            stability_margin=margin,
            bound=bound,
        )

    def safe_rate(
        self, raw_rate: torch.Tensor | float, features: torch.Tensor
    ) -> StabilityConstraintResult:
        raw_t = (
            torch.as_tensor(raw_rate, dtype=features.dtype, device=features.device)
            if not isinstance(raw_rate, torch.Tensor)
            else raw_rate
        )

        if self.check_finite:
            if not torch.isfinite(raw_t).all() or not torch.isfinite(features).all():
                raise FloatingPointError(
                    f"Non-finite input detected in UnconstrainedController.safe_rate: "
                    f"raw_rate finite={torch.isfinite(raw_t).all().item()}, "
                    f"features finite={torch.isfinite(features).all().item()}."
                )

        if features.ndim == 1:
            z_norm_sq = torch.sum(features**2)
        elif features.ndim == 2:
            z_norm_sq = torch.sum(features**2, dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unsupported features ndim={features.ndim}.")

        norm_rate = raw_t * z_norm_sq
        margin = 2.0 - norm_rate
        clipped = torch.zeros_like(raw_t, dtype=torch.bool)
        bound = torch.full_like(raw_t, float("inf"))

        return StabilityConstraintResult(
            raw_value=raw_t,
            safe_value=raw_t,
            clipped=clipped,
            normalized_value=norm_rate,
            stability_margin=margin,
            bound=bound,
        )


class SafeStepSizeController(BaseStabilityController):
    r"""Enforces local non-expansion of content memory residual errors.

    Mathematical formulation:
        $$\eta_t^{\text{bound}} = \frac{\beta}{\|k_t\|_2^2 + \epsilon}$$
        $$\eta_t^{\text{safe}} = \min\left(\eta_t^{\text{raw}}, \eta_t^{\text{bound}}\right)$$

    Guarantees that for any query on key $k_t$:
        $$\|e_{t+1}^{(k_t)}\|_2 = |1 - \eta_t^{\text{safe}} \|k_t\|_2^2| \cdot \|e_t\|_2$$
        $$|1 - \eta_t^{\text{safe}} \|k_t\|_2^2| \le \max(|1 - \beta|, 1) \le 1 \quad \text{for } 0 < \beta < 2$$

    Parameters:
        beta ($\beta$): Safety coefficient in $(0, 2)$. Strict contraction requires $\beta < 2$.
            Default is 1.0 (exact projection onto contractive regime).
        eps ($\epsilon$): Numerical floor preventing division by zero for near-zero keys.
        check_finite: If True, raises FloatingPointError on NaN/Inf inputs.
    """

    def __init__(
        self,
        beta: float = 1.0,
        eps: float = 1e-8,
        check_finite: bool = False,
    ) -> None:
        super().__init__()
        if not (0.0 < beta < 2.0):
            raise ValueError(f"Safety coefficient beta must be in (0, 2), got {beta}.")
        if eps <= 0.0:
            raise ValueError(f"eps must be strictly positive, got {eps}.")

        self.beta = float(beta)
        self.eps = float(eps)
        self.check_finite = check_finite

    def safe_step(
        self, raw_step_size: torch.Tensor | float, key: torch.Tensor
    ) -> StabilityConstraintResult:
        raw_t = (
            torch.as_tensor(raw_step_size, dtype=key.dtype, device=key.device)
            if not isinstance(raw_step_size, torch.Tensor)
            else raw_step_size
        )

        if self.check_finite:
            if not torch.isfinite(raw_t).all() or not torch.isfinite(key).all():
                raise FloatingPointError(
                    f"Non-finite input detected in SafeStepSizeController.safe_step: "
                    f"raw_step_size finite={torch.isfinite(raw_t).all().item()}, "
                    f"key finite={torch.isfinite(key).all().item()}."
                )

        if key.ndim == 1:
            k_norm_sq = torch.sum(key**2)
        elif key.ndim == 2:
            k_norm_sq = torch.sum(key**2, dim=-1, keepdim=True)
        else:
            raise ValueError(
                f"Unsupported key ndim={key.ndim}; expected 1D [K] or 2D [B, K]."
            )

        beta_t = torch.tensor(self.beta, dtype=raw_t.dtype, device=raw_t.device)
        eps_t = torch.tensor(self.eps, dtype=raw_t.dtype, device=raw_t.device)

        bound = beta_t / (k_norm_sq + eps_t)
        safe_val = torch.minimum(raw_t, bound)

        clipped = raw_t > bound
        norm_step = safe_val * k_norm_sq
        margin = 2.0 - norm_step

        return StabilityConstraintResult(
            raw_value=raw_t,
            safe_value=safe_val,
            clipped=clipped,
            normalized_value=norm_step,
            stability_margin=margin,
            bound=bound,
        )

    def safe_rate(
        self, raw_rate: torch.Tensor | float, features: torch.Tensor
    ) -> StabilityConstraintResult:
        """Passthrough for dynamics rate when used as content-only safety controller."""
        raw_t = (
            torch.as_tensor(raw_rate, dtype=features.dtype, device=features.device)
            if not isinstance(raw_rate, torch.Tensor)
            else raw_rate
        )
        if features.ndim == 1:
            z_norm_sq = torch.sum(features**2)
        else:
            z_norm_sq = torch.sum(features**2, dim=-1, keepdim=True)

        return StabilityConstraintResult(
            raw_value=raw_t,
            safe_value=raw_t,
            clipped=torch.zeros_like(raw_t, dtype=torch.bool),
            normalized_value=raw_t * z_norm_sq,
            stability_margin=2.0 - (raw_t * z_norm_sq),
            bound=torch.full_like(raw_t, float("inf")),
        )


class SafeDynamicsRateController(BaseStabilityController):
    r"""Enforces local non-expansion of dynamics memory residual errors.

    Mathematical formulation:
        $$\rho_t^{\text{bound}} = \frac{\beta_C}{\|z_t\|_2^2 + \epsilon}$$
        $$\rho_t^{\text{safe}} = \min\left(\rho_t^{\text{raw}}, \rho_t^{\text{bound}}\right)$$

    Guarantees that for any control feature $z_t$:
        $$|d_{t+1}^{(z_t)}| = |1 - \rho_t^{\text{safe}} \|z_t\|_2^2| \cdot |d_t| \le |d_t| \quad \text{for } 0 < \beta_C < 2$$

    Parameters:
        beta_c ($\beta_C$): Safety coefficient in $(0, 2)$. Default is 1.0.
        eps ($\epsilon$): Numerical floor preventing division by zero for zero feature vectors.
        check_finite: If True, raises FloatingPointError on NaN/Inf inputs.
    """

    def __init__(
        self,
        beta_c: float = 1.0,
        eps: float = 1e-8,
        check_finite: bool = False,
    ) -> None:
        super().__init__()
        if not (0.0 < beta_c < 2.0):
            raise ValueError(
                f"Safety coefficient beta_c must be in (0, 2), got {beta_c}."
            )
        if eps <= 0.0:
            raise ValueError(f"eps must be strictly positive, got {eps}.")

        self.beta_c = float(beta_c)
        self.eps = float(eps)
        self.check_finite = check_finite

    def safe_step(
        self, raw_step_size: torch.Tensor | float, key: torch.Tensor
    ) -> StabilityConstraintResult:
        """Passthrough for content step size when used as dynamics-only safety controller."""
        raw_t = (
            torch.as_tensor(raw_step_size, dtype=key.dtype, device=key.device)
            if not isinstance(raw_step_size, torch.Tensor)
            else raw_step_size
        )
        if key.ndim == 1:
            k_norm_sq = torch.sum(key**2)
        else:
            k_norm_sq = torch.sum(key**2, dim=-1, keepdim=True)

        return StabilityConstraintResult(
            raw_value=raw_t,
            safe_value=raw_t,
            clipped=torch.zeros_like(raw_t, dtype=torch.bool),
            normalized_value=raw_t * k_norm_sq,
            stability_margin=2.0 - (raw_t * k_norm_sq),
            bound=torch.full_like(raw_t, float("inf")),
        )

    def safe_rate(
        self, raw_rate: torch.Tensor | float, features: torch.Tensor
    ) -> StabilityConstraintResult:
        raw_t = (
            torch.as_tensor(raw_rate, dtype=features.dtype, device=features.device)
            if not isinstance(raw_rate, torch.Tensor)
            else raw_rate
        )

        if self.check_finite:
            if not torch.isfinite(raw_t).all() or not torch.isfinite(features).all():
                raise FloatingPointError(
                    f"Non-finite input detected in SafeDynamicsRateController.safe_rate: "
                    f"raw_rate finite={torch.isfinite(raw_t).all().item()}, "
                    f"features finite={torch.isfinite(features).all().item()}."
                )

        if features.ndim == 1:
            z_norm_sq = torch.sum(features**2)
        elif features.ndim == 2:
            z_norm_sq = torch.sum(features**2, dim=-1, keepdim=True)
        else:
            raise ValueError(
                f"Unsupported features ndim={features.ndim}; expected 1D [D] or 2D [B, D]."
            )

        beta_t = torch.tensor(self.beta_c, dtype=raw_t.dtype, device=raw_t.device)
        eps_t = torch.tensor(self.eps, dtype=raw_t.dtype, device=raw_t.device)

        bound = beta_t / (z_norm_sq + eps_t)
        safe_val = torch.minimum(raw_t, bound)

        clipped = raw_t > bound
        norm_rate = safe_val * z_norm_sq
        margin = 2.0 - norm_rate

        return StabilityConstraintResult(
            raw_value=raw_t,
            safe_value=safe_val,
            clipped=clipped,
            normalized_value=norm_rate,
            stability_margin=margin,
            bound=bound,
        )
