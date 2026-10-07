"""Canonical public controller API for DeltaCore adaptive associative state.

Provides the single authoritative stateful controller abstraction:
    - AdaptiveController: Manages explicit internal associative state M_t.
    - ControllerConfig: Typed configuration for dimensions, step rates, and stability.
    - ControllerStepResult: Comprehensive diagnostic returned on each inference step.

Mathematical Contract:
    Given input key x_t in R^K and optional target v_t in R^V (v_t = x_t in auto-associative mode):
    1. Pre-update prediction:
           \\hat{v}_t = M_{t-1} x_t
    2. Residual error:
           e_t = v_t - \\hat{v}_t
           r_t = ||e_t||_2 = ||x_t - M_{t-1} x_t||_2  (auto-associative residual)
    3. Error-modulated candidate step size:
           \\eta_{cand, t} = \\frac{\\eta_0}{1 + \\gamma ||e_t||_2}
    4. Lyapunov contractive safety bound:
           \\eta_t = \\min\\left(\\eta_{cand, t}, \\frac{\\rho}{||x_t||_2^2 + \\epsilon}\\right)
    5. Local stability margin:
           margin_t = 2.0 - \\eta_t ||x_t||_2^2
    6. Selective retention factor:
           \\alpha_t = \\max\\left(\\alpha_{min}, 1.0 - \\eta_t ||x_t||_2^2\\right) if selective_retention else 1.0
    7. Associative state update:
           \\Delta M_t = \\eta_t (e_t \\otimes x_t)
           M_t = \\alpha_t M_{t-1} + \\Delta M_t  (if adapt=True)

Temporal & Gating Invariant:
    All diagnostics (prediction, residual error, reconstruction residual, candidate step size,
    stability margin) are STRICTLY computed against the PRE-UPDATE state M_{t-1}.
    The controller supports separate non-mutating scoring (`score()`), explicit updates (`update()`),
    and unified gated execution (`step(..., adapt=True/False)`).
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

STATE_FORMAT_VERSION = "deltacore.state.v1"


@dataclass(frozen=True)
class ControllerConfig:
    r"""Configuration parameters for AdaptiveController.

    Attributes:
        dim: Dimensionality D of input key vector x_t (and target in auto-associative mode).
        val_dim: Dimensionality V of target vector v_t. Defaults to dim (square matrix D x D).
        eta0: Base nominal learning rate \eta_0 \ge 0.
        rho: Contractive stability threshold \rho \in (0, 2).
        alpha_min: Lower bound on selective retention factor \alpha_t \in [0, 1].
        gamma: Error-sensitivity scaling parameter \gamma \ge 0.
        epsilon: Numerical floor preventing division by zero.
        selective_retention: Whether to modulate M_{t-1} by retention factor \alpha_t.
        initial_scale: Scale factor for initial state (0.0 for zero matrix, >0 for diagonal).
        dtype: PyTorch floating-point data type (default torch.float32).
    """

    dim: int = 64
    val_dim: int | None = None
    eta0: float = 0.015
    rho: float = 1.50
    alpha_min: float = 0.95
    gamma: float = 0.10
    epsilon: float = 1e-6
    selective_retention: bool = False
    initial_scale: float = 0.0
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {self.dim}")
        if self.val_dim is not None and self.val_dim <= 0:
            raise ValueError(f"val_dim must be positive, got {self.val_dim}")
        if self.eta0 < 0.0:
            raise ValueError(f"eta0 must be non-negative, got {self.eta0}")
        if not (0.0 < self.rho < 2.0):
            raise ValueError(
                f"rho must be in (0, 2) for strict contraction, got {self.rho}"
            )
        if not (0.0 <= self.alpha_min <= 1.0):
            raise ValueError(f"alpha_min must be in [0, 1], got {self.alpha_min}")
        if self.gamma < 0.0:
            raise ValueError(f"gamma must be non-negative, got {self.gamma}")
        if self.epsilon <= 0.0:
            raise ValueError(f"epsilon must be strictly positive, got {self.epsilon}")
        if self.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                f"dtype must be torch.float32 or torch.float64, got {self.dtype}"
            )

    @property
    def effective_val_dim(self) -> int:
        return self.val_dim if self.val_dim is not None else self.dim

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["dtype"] = "float32" if self.dtype == torch.float32 else "float64"
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ControllerConfig:
        data = copy.deepcopy(data)
        if "dtype" in data:
            data["dtype"] = (
                torch.float64
                if str(data["dtype"]).lower() in ("float64", "torch.float64")
                else torch.float32
            )
        return cls(**data)


@dataclass(frozen=True)
class ControllerStepResult:
    r"""Structured diagnostic output for a single controller inference step.

    All diagnostic metrics (prediction, residual error, reconstruction residual,
    step size, stability margin) are computed against the PRE-UPDATE state M_{t-1}.

    Attributes:
        prediction: Retrieved associative output \hat{v}_t = M_{t-1} x_t in R^V.
        error: Prediction error vector e_t = v_t - \hat{v}_t in R^V.
        reconstruction_residual: Pre-update L2 norm ||e_t||_2.
        step_size: Effective Lyapunov-bounded step size \eta_t.
        stability_margin: Distance to divergence boundary: 2.0 - \eta_t ||x_t||_2^2.
        retention: Effective retention factor \alpha_t applied to M_{t-1}.
        update_norm: Frobenius norm of applied update ||\Delta M_t||_F (0.0 if adapt=False).
        state_norm: Frobenius norm of active state ||M_t||_F after potential mutation.
        adapted: Whether adaptive state was updated on this step.
    """

    prediction: torch.Tensor
    error: torch.Tensor
    reconstruction_residual: float
    step_size: float
    stability_margin: float
    retention: float
    update_norm: float
    state_norm: float
    adapted: bool


class AdaptiveController(nn.Module):
    r"""Canonical public stateful adaptive controller for DeltaCore.

    Maintains explicit associative memory matrix M in R^{V x K}.
    Guarantees strict score-before-update semantics, deterministic input validation,
    Lyapunov contractive safety bounding, and robust cryptographic state serialization.
    """

    def __init__(self, config: ControllerConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        if config is None:
            self.config = ControllerConfig(**kwargs)
        else:
            if kwargs:
                raise ValueError(
                    "Cannot pass both a ControllerConfig and keyword arguments."
                )
            self.config = config

        self.k_dim = self.config.dim
        self.v_dim = self.config.effective_val_dim

        # Explicit associative state parameter (requires_grad=False for inference)
        self.M = nn.Parameter(
            torch.empty(self.v_dim, self.k_dim, dtype=self.config.dtype),
            requires_grad=False,
        )
        self.step_count: int = 0
        self.reset_state()

    def reset_state(self) -> None:
        """Reset associative state M to initial condition and reset step count."""
        with torch.no_grad():
            if self.config.initial_scale == 0.0:
                self.M.zero_()
            else:
                self.M.zero_()
                min_dim = min(self.v_dim, self.k_dim)
                self.M[:min_dim, :min_dim].copy_(
                    torch.eye(min_dim, dtype=self.config.dtype)
                    * self.config.initial_scale
                )
        self.step_count = 0

    def get_state(self) -> torch.Tensor:
        """Return a detached clone of the current associative memory state M_t."""
        return self.M.detach().clone()

    def set_state(self, new_state: torch.Tensor) -> None:
        """Explicitly set the associative memory state M_t.

        Args:
            new_state: Tensor of shape (V, K) matching controller dtype.

        Raises:
            TypeError: If input is not a torch.Tensor.
            ValueError: If shape is incorrect or contains non-finite values.
        """
        if not isinstance(new_state, torch.Tensor):
            raise TypeError(
                f"new_state must be a torch.Tensor, got {type(new_state).__name__}"
            )
        if new_state.shape != (self.v_dim, self.k_dim):
            raise ValueError(
                f"State shape mismatch: expected ({self.v_dim}, {self.k_dim}), got {tuple(new_state.shape)}"
            )
        if not torch.isfinite(new_state).all():
            raise ValueError("new_state contains non-finite values (NaN or Inf)")
        with torch.no_grad():
            self.M.copy_(new_state.to(dtype=self.config.dtype, device=self.M.device))

    def _validate_inputs(
        self, x: torch.Tensor, target: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate input vector x and optional target vector, ensuring 1D finite floats."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input x must be a torch.Tensor, got {type(x).__name__}")
        if x.dim() != 1:
            raise ValueError(
                f"Input x must be a 1D vector of shape ({self.k_dim},), got ndim={x.dim()} {tuple(x.shape)}"
            )
        if x.shape[0] != self.k_dim:
            raise ValueError(
                f"Input x dimension mismatch: expected ({self.k_dim},), got {tuple(x.shape)}"
            )
        if not torch.isfinite(x).all():
            raise FloatingPointError("Input x contains non-finite values (NaN or Inf)")

        # Target handling
        if target is None:
            if self.v_dim != self.k_dim:
                raise ValueError(
                    f"Auto-associative mode requires v_dim == k_dim, but controller has "
                    f"k_dim={self.k_dim} and v_dim={self.v_dim}. An explicit target must be provided."
                )
            v = x
        else:
            if not isinstance(target, torch.Tensor):
                raise TypeError(
                    f"Target must be a torch.Tensor, got {type(target).__name__}"
                )
            if target.dim() != 1:
                raise ValueError(
                    f"Target must be a 1D vector of shape ({self.v_dim},), got ndim={target.dim()} {tuple(target.shape)}"
                )
            if target.shape[0] != self.v_dim:
                raise ValueError(
                    f"Target dimension mismatch: expected ({self.v_dim},), got {tuple(target.shape)}"
                )
            if not torch.isfinite(target).all():
                raise FloatingPointError(
                    "Target contains non-finite values (NaN or Inf)"
                )
            v = target

        # Ensure correct dtype and device
        x_cast = x.to(dtype=self.config.dtype, device=self.M.device)
        v_cast = v.to(dtype=self.config.dtype, device=self.M.device)
        return x_cast, v_cast

    def _compute_diagnostics(
        self, x: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, float, float, float, float]:
        """Compute pre-update prediction, error, residual, step size, margin, and retention against M_{t-1}."""
        # 1. Pre-update prediction
        pred = self.M @ x
        # 2. Residual error
        err = v - pred
        err_norm = float(torch.linalg.norm(err).item())
        x_norm_sq = float(torch.dot(x, x).item())

        # 3. Candidate adaptive step size
        eta_cand = self.config.eta0 / (1.0 + self.config.gamma * err_norm)

        # 4. Lyapunov contractive bounding
        safe_bound = self.config.rho / (x_norm_sq + self.config.epsilon)
        eta = min(eta_cand, safe_bound)

        # 5. Stability margin: 2.0 - \eta ||x||^2
        margin = 2.0 - (eta * x_norm_sq)

        # 6. Retention factor
        if self.config.selective_retention:
            retention = max(self.config.alpha_min, 1.0 - (eta * x_norm_sq))
        else:
            retention = 1.0

        return pred, err, err_norm, eta, margin, retention

    def score(
        self, x: torch.Tensor, target: torch.Tensor | None = None
    ) -> ControllerStepResult:
        """Evaluate pre-update prediction and residual diagnostics WITHOUT state mutation.

        Guarantees that state M_{t-1} remains completely unmodified.

        Args:
            x: Input key vector in R^K.
            target: Optional target vector in R^V (defaults to x for auto-associative).

        Returns:
            ControllerStepResult with pre-update diagnostics and adapted=False.
        """
        x_val, v_val = self._validate_inputs(x, target)
        with torch.no_grad():
            pred, err, err_norm, eta, margin, retention = self._compute_diagnostics(
                x_val, v_val
            )
            state_norm = float(torch.linalg.norm(self.M).item())

        return ControllerStepResult(
            prediction=pred.detach(),
            error=err.detach(),
            reconstruction_residual=err_norm,
            step_size=eta,
            stability_margin=margin,
            retention=retention,
            update_norm=0.0,
            state_norm=state_norm,
            adapted=False,
        )

    def update(
        self, x: torch.Tensor, target: torch.Tensor | None = None
    ) -> ControllerStepResult:
        """Perform an adaptive update step mutating state from M_{t-1} to M_t.

        All returned diagnostics (prediction, residual, step_size, margin) are
        evaluated against M_{t-1} BEFORE the update is applied.

        Args:
            x: Input key vector in R^K.
            target: Optional target vector in R^V (defaults to x for auto-associative).

        Returns:
            ControllerStepResult with pre-update diagnostics, update_norm, and adapted=True.
        """
        x_val, v_val = self._validate_inputs(x, target)
        with torch.no_grad():
            pred, err, err_norm, eta, margin, retention = self._compute_diagnostics(
                x_val, v_val
            )

            # State transition: M_t = alpha_t M_{t-1} + eta_t (err x^\top)
            delta_m = eta * torch.outer(err, x_val)
            update_norm = float(torch.linalg.norm(delta_m).item())

            if self.config.selective_retention and retention < 1.0:
                self.M.mul_(retention).add_(delta_m)
            else:
                self.M.add_(delta_m)

            self.step_count += 1
            new_state_norm = float(torch.linalg.norm(self.M).item())

        return ControllerStepResult(
            prediction=pred.detach(),
            error=err.detach(),
            reconstruction_residual=err_norm,
            step_size=eta,
            stability_margin=margin,
            retention=retention,
            update_norm=update_norm,
            state_norm=new_state_norm,
            adapted=True,
        )

    def step(
        self,
        x: torch.Tensor,
        target: torch.Tensor | None = None,
        adapt: bool = True,
    ) -> ControllerStepResult:
        """Execute a single inference step with explicit adaptation gating.

        When adapt=True: evaluates pre-update diagnostics then executes state update.
        When adapt=False: evaluates diagnostics without modifying state.

        Args:
            x: Input key vector in R^K.
            target: Optional target vector in R^V. In auto-associative mode, target is omitted.
            adapt: If True, incorporates observation into associative state.

        Returns:
            ControllerStepResult with pre-update diagnostics and active state norm.
        """
        if adapt:
            return self.update(x, target)
        return self.score(x, target)

    def state_dict_payload(self) -> dict[str, Any]:
        """Export serialized state payload with metadata, configuration, and checksum."""
        m_bytes = self.M.detach().cpu().numpy().tobytes()
        checksum = hashlib.sha256(m_bytes).hexdigest()
        return {
            "format_version": STATE_FORMAT_VERSION,
            "config": self.config.to_dict(),
            "k_dim": self.k_dim,
            "v_dim": self.v_dim,
            "step_count": self.step_count,
            "state_matrix": self.M.detach().cpu().tolist(),
            "sha256": checksum,
        }

    def load_state_dict_payload(self, payload: dict[str, Any]) -> None:
        """Restore controller state from serialized payload with strict schema validation.

        Raises:
            ValueError: If payload format version mismatches, dimensions mismatch, or checksum fails.
        """
        if not isinstance(payload, dict):
            raise TypeError("State payload must be a dictionary")
        if payload.get("format_version") != STATE_FORMAT_VERSION:
            raise ValueError(
                f"Incompatible state format version: expected {STATE_FORMAT_VERSION}, "
                f"got {payload.get('format_version')}"
            )
        if payload.get("k_dim") != self.k_dim or payload.get("v_dim") != self.v_dim:
            raise ValueError(
                f"Dimension mismatch in state payload: controller has ({self.v_dim}, {self.k_dim}), "
                f"payload has ({payload.get('v_dim')}, {payload.get('k_dim')})"
            )

        state_tensor = torch.tensor(payload["state_matrix"], dtype=self.config.dtype)
        # Verify checksum
        actual_checksum = hashlib.sha256(state_tensor.numpy().tobytes()).hexdigest()
        expected_checksum = payload.get("sha256")
        if actual_checksum != expected_checksum:
            raise ValueError(
                f"Cryptographic state payload checksum mismatch! Expected {expected_checksum}, "
                f"computed {actual_checksum}. State may be corrupted."
            )

        self.set_state(state_tensor)
        self.step_count = int(payload.get("step_count", 0))

    def save_state(self, filepath: str | Path) -> None:
        """Save controller state and metadata to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.state_dict_payload()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load_state(self, filepath: str | Path) -> None:
        """Load controller state and metadata from a JSON file with integrity validation."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"State file not found: {path}")
        with open(path, encoding="utf-8") as f:
            try:
                payload = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Corrupted state file (invalid JSON): {e}") from e
        self.load_state_dict_payload(payload)
