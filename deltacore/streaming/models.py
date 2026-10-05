"""DeltaCore Streaming Predictors & Baselines for Online Adaptive Benchmark.

Defines the standardized streaming predictor interface:
    - predict_step(x_t) -> y_hat
    - adapt_step(x_t, y_t) -> state update (parameter updates = 0!)
    - reset_state() -> resets internal memory state
    - set_adaptation(enabled: bool) -> toggles online adaptation ON / OFF

Includes:
    1. FrozenLinearPredictor
    2. FrozenMLPPredictor
    3. GRUPredictor
    4. LSTMPredictor
    5. OnlineRidgePredictor (Recursive Least Squares deterministic estimator)
    6. FixedDeltaPredictor
    7. AdaptiveDeltaPredictor
    8. SafeAdaptiveDeltaPredictor
    9. SelfReferentialPredictor
    10. SafeSelfReferentialPredictor
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


def compute_parameter_hash(model: nn.Module) -> str:
    """Compute deterministic SHA-256 hash of offline trainable parameters in a module.

    Ignores dynamic adaptive state matrices (such as associative memory M or RLS covariance P)
    to strictly verify that offline neural / linear parameters theta remain immutable during test streaming.
    """
    hasher = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        if name in ("M", "W", "P", "C", "last_x"):
            continue
        hasher.update(name.encode("utf-8"))
        hasher.update(p.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


class StreamingPredictor(nn.Module, ABC):
    """Abstract base class for streaming adaptive predictors."""

    def __init__(self, name: str, dim: int = 8) -> None:
        super().__init__()
        self.name = name
        self.dim = dim
        self.adaptation_enabled: bool = True
        self.last_update_norm: float = 0.0

    @abstractmethod
    def predict_step(self, x: torch.Tensor) -> torch.Tensor:
        """Compute single-step prediction y_hat for input x [D]."""

    @abstractmethod
    def adapt_step(self, x: torch.Tensor, target: torch.Tensor) -> None:
        """Update internal state from observation (x, target). Zero parameter gradients!"""

    @abstractmethod
    def reset_state(self) -> None:
        """Reset internal memory/recurrent state to initial condition."""

    @abstractmethod
    def get_state_norm(self) -> float:
        """Return Frobenius/Euclidean norm of the active adaptive state."""

    def set_adaptation(self, enabled: bool) -> None:
        """Toggle online adaptation ON or OFF (Section 11 ablation)."""
        self.adaptation_enabled = enabled

    def get_param_count(self) -> tuple[int, int]:
        """Return (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def get_state_memory_bytes(self) -> int:
        """Return bytes occupied by persistent adaptive state."""
        return 0

    def get_memory_breakdown(self) -> dict[str, int]:
        """Return breakdown of parameter, persistent state, and activation memory in bytes."""
        param_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        param_bytes += sum(b.numel() * b.element_size() for b in self.buffers())
        state_bytes = self.get_state_memory_bytes()
        activation_bytes = 4 * self.dim * 4
        return {
            "parameter_bytes": param_bytes,
            "persistent_state_bytes": state_bytes,
            "temporary_activation_bytes": activation_bytes,
            "total_bytes": param_bytes + state_bytes + activation_bytes,
        }


# ==============================================================================
# Non-Adaptive / Recurrent Baselines
# ==============================================================================


class FrozenLinearPredictor(StreamingPredictor):
    """Baseline 1: Static linear mapping W x + b without test-time adaptation."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__(name="FrozenLinear", dim=dim)
        self.linear = nn.Linear(dim, dim)
        self._init_identity()

    def _init_identity(self) -> None:
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def predict_step(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def adapt_step(self, x: torch.Tensor, target: torch.Tensor) -> None:
        # Frozen: no state update
        self.last_update_norm = 0.0

    def reset_state(self) -> None:
        pass

    def get_state_norm(self) -> float:
        return float(torch.linalg.norm(self.linear.weight).item())


class FrozenMLPPredictor(StreamingPredictor):
    """Baseline 2: 2-layer MLP without test-time adaptation."""

    def __init__(self, dim: int = 8, hidden_dim: int = 4) -> None:
        super().__init__(name="FrozenMLP", dim=dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def predict_step(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def adapt_step(self, x: torch.Tensor, target: torch.Tensor) -> None:
        self.last_update_norm = 0.0

    def reset_state(self) -> None:
        pass

    def get_state_norm(self) -> float:
        total_sq = sum(
            torch.linalg.norm(p).item() ** 2
            for p in self.mlp.parameters()
            if p.ndim > 1
        )
        return float(total_sq**0.5)


class GRUPredictor(StreamingPredictor):
    """Baseline 3: Recurrent GRU with online hidden state propagation."""

    def __init__(self, dim: int = 8, hidden_dim: int = 2) -> None:
        super().__init__(name="GRU", dim=dim)
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_size=dim, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, dim)
        self.hidden_state: torch.Tensor | None = None
        self.reset_state()

    def reset_state(self) -> None:
        self.hidden_state = torch.zeros(1, 1, self.hidden_dim, dtype=torch.float32)
        self.last_update_norm = 0.0

    def predict_step(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x.view(1, 1, self.dim)
        out, next_h = self.gru(x_in, self.hidden_state)
        pred = self.head(out.view(-1))
        # Store next_h temporarily for adapt_step
        self._temp_next_h = next_h
        return pred

    def adapt_step(self, x: torch.Tensor, target: torch.Tensor) -> None:
        if self.adaptation_enabled and hasattr(self, "_temp_next_h"):
            diff = self._temp_next_h - self.hidden_state
            self.last_update_norm = float(torch.linalg.norm(diff).item())
            self.hidden_state = self._temp_next_h
        else:
            self.last_update_norm = 0.0

    def get_state_norm(self) -> float:
        return (
            float(torch.linalg.norm(self.hidden_state).item())
            if self.hidden_state is not None
            else 0.0
        )

    def get_state_memory_bytes(self) -> int:
        return self.hidden_dim * 4


class LSTMPredictor(StreamingPredictor):
    """Baseline 4: Recurrent LSTM with online (h, c) state propagation."""

    def __init__(self, dim: int = 8, hidden_dim: int = 2) -> None:
        super().__init__(name="LSTM", dim=dim)
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=dim, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, dim)
        self.h: torch.Tensor | None = None
        self.c: torch.Tensor | None = None
        self.reset_state()

    def reset_state(self) -> None:
        self.h = torch.zeros(1, 1, self.hidden_dim, dtype=torch.float32)
        self.c = torch.zeros(1, 1, self.hidden_dim, dtype=torch.float32)
        self.last_update_norm = 0.0

    def predict_step(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x.view(1, 1, self.dim)
        out, (next_h, next_c) = self.lstm(x_in, (self.h, self.c))
        pred = self.head(out.view(-1))
        self._temp_next_h = next_h
        self._temp_next_c = next_c
        return pred

    def adapt_step(self, x: torch.Tensor, target: torch.Tensor) -> None:
        if self.adaptation_enabled and hasattr(self, "_temp_next_h"):
            diff_h = self._temp_next_h - self.h
            diff_c = self._temp_next_c - self.c
            self.last_update_norm = float(
                (torch.linalg.norm(diff_h) ** 2 + torch.linalg.norm(diff_c) ** 2)
                .sqrt()
                .item()
            )
            self.h = self._temp_next_h
            self.c = self._temp_next_c
        else:
            self.last_update_norm = 0.0

    def get_state_norm(self) -> float:
        if self.h is None or self.c is None:
            return 0.0
        return float(
            (torch.linalg.norm(self.h) ** 2 + torch.linalg.norm(self.c) ** 2)
            .sqrt()
            .item()
        )

    def get_state_memory_bytes(self) -> int:
        return 2 * self.hidden_dim * 4


# ==============================================================================
# Non-Neural Online Statistical Baseline (Section 15)
# ==============================================================================


class OnlineRidgePredictor(StreamingPredictor):
    r"""Baseline 5: Recursive Least Squares (RLS) / Online Ridge Regression.

    Deterministic online statistical estimator updating inverse correlation:
        P_t = (1 / \lambda) * [ P_{t-1} - (P_{t-1} x_t x_t^T P_{t-1}) / (\lambda + x_t^T P_{t-1} x_t) ]
        W_t = W_{t-1} + e_t k_t^T
    where k_t = P_t x_t and e_t = target_t - W_{t-1} x_t.
    """

    def __init__(self, dim: int = 8, lam: float = 0.98, delta: float = 1.0) -> None:
        super().__init__(name="OnlineRidge", dim=dim)
        self.lam = lam
        self.delta = delta
        self.W = nn.Parameter(torch.eye(dim, dtype=torch.float32), requires_grad=False)
        self.P: torch.Tensor = torch.eye(dim, dtype=torch.float32) * delta
        self.reset_state()

    def reset_state(self) -> None:
        self.W.data.copy_(torch.eye(self.dim, dtype=torch.float32) * 0.1)
        self.P = torch.eye(self.dim, dtype=torch.float32) * self.delta
        self.last_update_norm = 0.0

    def predict_step(self, x: torch.Tensor) -> torch.Tensor:
        return self.W @ x

    def adapt_step(self, x: torch.Tensor, target: torch.Tensor) -> None:
        if not self.adaptation_enabled:
            self.last_update_norm = 0.0
            return

        pred = self.W @ x
        err = target - pred

        # RLS Gain vector k_t = P_{t-1} x_t / (\lambda + x_t^T P_{t-1} x_t)
        Px = self.P @ x
        denom = float(self.lam + torch.dot(x, Px).item())
        k = Px / max(denom, 1e-6)

        # Update weight matrix: \Delta W = err (x) k
        dW = torch.outer(err, k)
        self.W.data.add_(dW)

        # Update covariance matrix: P_t = (1 / \lambda) * [P_{t-1} - k (x) Px]
        self.P = (self.P - torch.outer(k, Px)) / self.lam

        self.last_update_norm = float(torch.linalg.norm(dW).item())

    def get_state_norm(self) -> float:
        return float(torch.linalg.norm(self.W).item())

    def get_state_memory_bytes(self) -> int:
        return (self.dim * self.dim + self.dim * self.dim) * 4


# ==============================================================================
# DeltaCore Streaming Models
# ==============================================================================


class FixedDeltaPredictor(StreamingPredictor):
    r"""DeltaCore 1: Fixed Delta-rule online associative predictor.

    State transition:
        \hat{y}_t = M_t x_t
        e_t = y_t - \hat{y}_t
        \Delta M_t = \eta e_t x_t^\top
        M_{t+1} = M_t + \Delta M_t
    """

    def __init__(self, dim: int = 8, step_size: float = 0.15) -> None:
        super().__init__(name="FixedDelta", dim=dim)
        self.step_size = step_size
        self.M = nn.Parameter(
            torch.eye(dim, dtype=torch.float32) * 0.1, requires_grad=False
        )
        self.reset_state()

    def reset_state(self) -> None:
        self.M.data.copy_(torch.eye(self.dim, dtype=torch.float32) * 0.1)
        self.last_update_norm = 0.0

    def set_state(self, M: torch.Tensor) -> None:
        """Explicitly set associative memory state M."""
        self.M.data.copy_(M)
        self.last_update_norm = 0.0

    def predict_step(self, x: torch.Tensor) -> torch.Tensor:
        return self.M @ x

    def adapt_step(self, x: torch.Tensor, target: torch.Tensor) -> None:
        if not self.adaptation_enabled:
            self.last_update_norm = 0.0
            return

        pred = self.M @ x
        err = target - pred
        delta_m = self.step_size * torch.outer(err, x)
        self.M.data.add_(delta_m)
        self.last_update_norm = float(torch.linalg.norm(delta_m).item())

    def get_state_norm(self) -> float:
        return float(torch.linalg.norm(self.M).item())

    def get_state_memory_bytes(self) -> int:
        return self.dim * self.dim * 4


class AdaptiveDeltaPredictor(StreamingPredictor):
    r"""DeltaCore 2: Error-modulated adaptive step-size associative predictor.

    Dynamic rate:
        \eta_t = \eta_{\max} * \frac{\|e_t\|}{\|e_t\| + 1.0}
    """

    def __init__(self, dim: int = 8, eta_max: float = 0.40) -> None:
        super().__init__(name="AdaptiveDelta", dim=dim)
        self.eta_max = eta_max
        self.M = nn.Parameter(
            torch.eye(dim, dtype=torch.float32) * 0.1, requires_grad=False
        )
        self.last_step_size: float = eta_max * 0.5
        self.reset_state()

    def reset_state(self) -> None:
        self.M.data.copy_(torch.eye(self.dim, dtype=torch.float32) * 0.1)
        self.last_update_norm = 0.0
        self.last_step_size = self.eta_max * 0.5

    def predict_step(self, x: torch.Tensor) -> torch.Tensor:
        return self.M @ x

    def adapt_step(self, x: torch.Tensor, target: torch.Tensor) -> None:
        if not self.adaptation_enabled:
            self.last_update_norm = 0.0
            return

        pred = self.M @ x
        err = target - pred
        err_norm = float(torch.linalg.norm(err).item())

        # Error-modulated dynamic rate
        eta = self.eta_max * (err_norm / (err_norm + 1.0))
        self.last_step_size = eta

        delta_m = eta * torch.outer(err, x)
        self.M.data.add_(delta_m)
        self.last_update_norm = float(torch.linalg.norm(delta_m).item())

    def get_state_norm(self) -> float:
        return float(torch.linalg.norm(self.M).item())

    def get_state_memory_bytes(self) -> int:
        return self.dim * self.dim * 4


class SafeAdaptiveDeltaPredictor(StreamingPredictor):
    r"""DeltaCore 3: Stability-bounded adaptive delta predictor with Lyapunov clamping.

    Enforces strict contractivity:
        \gamma_t = \eta_t \|x_t\|^2
        \eta_t^{safe} = \min(\eta_t, \frac{2.0 - \delta}{\|x_t\|^2 + \epsilon})
    """

    def __init__(
        self,
        dim: int = 8,
        eta_max: float = 0.50,
        stability_margin: float = 0.10,
        rho: float | None = None,
        alpha_min: float = 1.0,
        gamma: float = 0.0,
    ) -> None:
        super().__init__(name="SafeAdaptiveDelta", dim=dim)
        self.eta_max = eta_max
        if rho is not None:
            self.stability_margin = max(0.0, 2.0 - rho)
            self.rho = rho
        else:
            self.stability_margin = stability_margin
            self.rho = 2.0 - stability_margin
        self.alpha_min = alpha_min
        self.gamma = gamma
        self.M = nn.Parameter(
            torch.eye(dim, dtype=torch.float32) * 0.1, requires_grad=False
        )
        self.last_step_size: float = eta_max * 0.5
        self.last_margin: float = 2.0
        self.last_retention: float = 1.0
        self.reset_state()

    def reset_state(self) -> None:
        self.M.data.copy_(torch.eye(self.dim, dtype=torch.float32) * 0.1)
        self.last_update_norm = 0.0
        self.last_margin = 2.0
        self.last_retention = 1.0

    def set_state(self, M: torch.Tensor) -> None:
        """Explicitly set associative memory state M."""
        self.M.data.copy_(M)
        self.last_update_norm = 0.0
        self.last_margin = 2.0
        self.last_retention = 1.0

    def predict_step(self, x: torch.Tensor) -> torch.Tensor:
        return self.M @ x

    def adapt_step(self, x: torch.Tensor, target: torch.Tensor) -> None:
        if not self.adaptation_enabled:
            self.last_update_norm = 0.0
            return

        pred = self.M @ x
        err = target - pred
        err_norm = float(torch.linalg.norm(err).item())
        x_norm_sq = float((x @ x).item())

        raw_eta = self.eta_max * (err_norm / (err_norm + 1.0))

        # Contractive Lyapunov projection
        max_safe_eta = (2.0 - self.stability_margin) / max(x_norm_sq, 1e-5)
        safe_eta = min(raw_eta, max_safe_eta)
        self.last_step_size = safe_eta
        self.last_margin = 2.0 - (safe_eta * x_norm_sq)

        delta_m = safe_eta * torch.outer(err, x)
        if self.gamma > 0.0 and self.alpha_min < 1.0:
            alpha_t = max(self.alpha_min, min(1.0, 1.0 - self.gamma * err_norm))
            self.M.data.mul_(alpha_t).add_(delta_m)
            self.last_retention = float(alpha_t)
        else:
            self.M.data.add_(delta_m)
            self.last_retention = 1.0
        self.last_update_norm = float(torch.linalg.norm(delta_m).item())

    def get_state_norm(self) -> float:
        return float(torch.linalg.norm(self.M).item())

    def get_state_memory_bytes(self) -> int:
        return self.dim * self.dim * 4


class SelfReferentialPredictor(StreamingPredictor):
    r"""DeltaCore 4: Coupled content and dynamics memory co-evolution.

    Content: \Delta M_t = \eta_t e_t x_t^\top
    Dynamics: C_t modulates \eta_t; C updates based on associative feedback.
    """

    def __init__(self, dim: int = 8, dc_dim: int = 2) -> None:
        super().__init__(name="SelfReferential", dim=dim)
        self.dc_dim = dc_dim
        self.M = nn.Parameter(
            torch.eye(dim, dtype=torch.float32) * 0.1, requires_grad=False
        )
        self.C = nn.Parameter(
            torch.zeros(1, dc_dim, dtype=torch.float32), requires_grad=False
        )
        self.last_step_size: float = 0.2
        self.reset_state()

    def reset_state(self) -> None:
        self.M.data.copy_(torch.eye(self.dim, dtype=torch.float32) * 0.1)
        self.C.data.zero_()
        self.last_update_norm = 0.0

    def predict_step(self, x: torch.Tensor) -> torch.Tensor:
        return self.M @ x

    def adapt_step(self, x: torch.Tensor, target: torch.Tensor) -> None:
        if not self.adaptation_enabled:
            self.last_update_norm = 0.0
            return

        pred = self.M @ x
        err = target - pred

        # Form control feature vector z of dimension dc_dim
        x_norm = float(torch.linalg.norm(x).item())
        err_norm = float(torch.linalg.norm(err).item())
        z_feats = [x_norm, err_norm]
        while len(z_feats) < self.dc_dim:
            idx = len(z_feats) - 2
            z_feats.append(float(x[idx % len(x)].item()))
        z = torch.tensor(z_feats[: self.dc_dim], dtype=torch.float32)

        eta = float(torch.sigmoid(self.C @ z).item()) * 0.45
        self.last_step_size = eta

        delta_m = eta * torch.outer(err, x)
        self.M.data.add_(delta_m)

        # Update dynamics memory
        c_target = torch.tensor([0.25], dtype=torch.float32)
        c_pred = self.C @ z
        delta_c = 0.05 * torch.outer(c_target - c_pred, z)
        self.C.data.add_(delta_c)

        self.last_update_norm = float(
            (torch.linalg.norm(delta_m) ** 2 + torch.linalg.norm(delta_c) ** 2)
            .sqrt()
            .item()
        )

    def get_state_norm(self) -> float:
        return float(
            (torch.linalg.norm(self.M) ** 2 + torch.linalg.norm(self.C) ** 2)
            .sqrt()
            .item()
        )

    def get_state_memory_bytes(self) -> int:
        return (self.dim * self.dim + self.dc_dim) * 4


class SafeSelfReferentialPredictor(StreamingPredictor):
    r"""DeltaCore 5: Safe coupled self-referential system with contractive bounding."""

    def __init__(self, dim: int = 8, dc_dim: int = 2) -> None:
        super().__init__(name="SafeSelfReferential", dim=dim)
        self.dc_dim = dc_dim
        self.M = nn.Parameter(
            torch.eye(dim, dtype=torch.float32) * 0.1, requires_grad=False
        )
        self.C = nn.Parameter(
            torch.zeros(1, dc_dim, dtype=torch.float32), requires_grad=False
        )
        self.last_step_size: float = 0.2
        self.last_margin: float = 2.0
        self.reset_state()

    def reset_state(self) -> None:
        self.M.data.copy_(torch.eye(self.dim, dtype=torch.float32) * 0.1)
        self.C.data.zero_()
        self.last_update_norm = 0.0
        self.last_margin = 2.0

    def predict_step(self, x: torch.Tensor) -> torch.Tensor:
        return self.M @ x

    def adapt_step(self, x: torch.Tensor, target: torch.Tensor) -> None:
        if not self.adaptation_enabled:
            self.last_update_norm = 0.0
            return

        pred = self.M @ x
        err = target - pred
        x_norm_sq = float((x @ x).item())

        # Form control feature vector z of dimension dc_dim
        x_norm = float(torch.linalg.norm(x).item())
        err_norm = float(torch.linalg.norm(err).item())
        z_feats = [x_norm, err_norm]
        while len(z_feats) < self.dc_dim:
            idx = len(z_feats) - 2
            z_feats.append(float(x[idx % len(x)].item()))
        z = torch.tensor(z_feats[: self.dc_dim], dtype=torch.float32)

        raw_eta = float(torch.sigmoid(self.C @ z).item()) * 0.50

        # Contractive projection
        safe_eta = min(raw_eta, 1.9 / max(x_norm_sq, 1e-5))
        self.last_step_size = safe_eta
        self.last_margin = 2.0 - (safe_eta * x_norm_sq)

        delta_m = safe_eta * torch.outer(err, x)
        self.M.data.add_(delta_m)

        c_target = torch.tensor([0.25], dtype=torch.float32)
        c_pred = self.C @ z
        delta_c = 0.05 * torch.outer(c_target - c_pred, z)
        self.C.data.add_(delta_c)

        self.last_update_norm = float(
            (torch.linalg.norm(delta_m) ** 2 + torch.linalg.norm(delta_c) ** 2)
            .sqrt()
            .item()
        )

    def get_state_norm(self) -> float:
        return float(
            (torch.linalg.norm(self.M) ** 2 + torch.linalg.norm(self.C) ** 2)
            .sqrt()
            .item()
        )

    def get_state_memory_bytes(self) -> int:
        return (self.dim * self.dim + self.dc_dim) * 4


class NaivePredictor(StreamingPredictor):
    """Naive streaming baseline (persistence or zero predictor)."""

    def __init__(self, dim: int = 8, mode: str = "persistence") -> None:
        super().__init__(name=f"Naive_{mode}", dim=dim)
        self.mode = mode
        self.last_x = torch.zeros(dim, dtype=torch.float32)

    def reset_state(self) -> None:
        self.last_x.zero_()
        self.last_update_norm = 0.0

    def predict_step(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "persistence":
            return self.last_x.clone()
        return torch.zeros_like(x)

    def adapt_step(self, x: torch.Tensor, target: torch.Tensor) -> None:
        self.last_x = x.clone()
        self.last_update_norm = 0.0

    def get_state_norm(self) -> float:
        return float(torch.linalg.norm(self.last_x).item())


class PersistencePredictor(StreamingPredictor):
    r"""Baseline A: Persistence predictor \hat{X}_{t+1} = X_t."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__(name="Persistence", dim=dim)
        self.last_x = torch.zeros(dim, dtype=torch.float32)

    def reset_state(self) -> None:
        self.last_x.zero_()
        self.last_update_norm = 0.0

    def predict_step(self, x: torch.Tensor) -> torch.Tensor:
        return x.clone()

    def adapt_step(self, x: torch.Tensor, target: torch.Tensor) -> None:
        self.last_x = x.clone()
        self.last_update_norm = 0.0

    def get_state_norm(self) -> float:
        return float(torch.linalg.norm(self.last_x).item())

    def get_state_memory_bytes(self) -> int:
        return self.dim * 4


class NonlinearOnlineRidgePredictor(StreamingPredictor):
    r"""Nonlinear Recursive Least Squares (RLS) with Random Fourier Features (RFF).

    Approximates Gaussian RBF kernel ridge regression online:
        \phi(x) = \sqrt{2 / D_{rff}} \cos(W_{rff} x + b_{rff})
    with Woodbury rank-1 covariance matrix updates:
        k_t = P_t \phi_t / (\lambda + \phi_t^\top P_t \phi_t)
        P_{t+1} = \lambda^{-1} (P_t - k_t \phi_t^\top P_t)
        W_{t+1} = W_t + (y_t - W_t \phi_t) k_t^\top
    """

    W_rff: torch.Tensor
    b_rff: torch.Tensor
    P: torch.Tensor
    W_out: torch.Tensor

    def __init__(
        self,
        dim: int = 8,
        rff_dim: int = 16,
        lam: float = 0.98,
        rff_scale: float = 0.50,
        seed: int = 42,
    ) -> None:
        super().__init__(name="NonlinearOnlineRidge", dim=dim)
        self.rff_dim = rff_dim
        self.lam = lam
        self.rff_scale = rff_scale

        gen = torch.Generator().manual_seed(seed)
        w_rff = (
            torch.randn(rff_dim, dim, generator=gen, dtype=torch.float32) * rff_scale
        )
        b_rff = torch.rand(rff_dim, generator=gen, dtype=torch.float32) * 2.0 * torch.pi
        self.register_buffer("W_rff", w_rff)
        self.register_buffer("b_rff", b_rff)

        self.register_buffer("P", torch.eye(rff_dim, dtype=torch.float32))
        self.register_buffer("W_out", torch.zeros(dim, rff_dim, dtype=torch.float32))
        self.reset_state()

    def reset_state(self) -> None:
        self.P.copy_(torch.eye(self.rff_dim, dtype=torch.float32))
        self.W_out.zero_()
        self.last_update_norm = 0.0

    def _compute_features(self, x: torch.Tensor) -> torch.Tensor:
        proj = torch.matmul(self.W_rff, x) + self.b_rff
        return torch.cos(proj) * math.sqrt(2.0 / self.rff_dim)

    def predict_step(self, x: torch.Tensor) -> torch.Tensor:
        phi = self._compute_features(x)
        return torch.matmul(self.W_out, phi)

    def adapt_step(self, x: torch.Tensor, target: torch.Tensor) -> None:
        if not self.adaptation_enabled:
            self.last_update_norm = 0.0
            return

        phi = self._compute_features(x)
        pred = torch.matmul(self.W_out, phi)
        err = target - pred

        p_phi = torch.matmul(self.P, phi)
        denom = self.lam + float(torch.dot(phi, p_phi).item())
        k = p_phi / max(denom, 1e-7)

        # Update P: \lambda^{-1} (P - k \phi^\top P)
        k_phi_t = torch.outer(k, torch.matmul(phi, self.P))
        self.P.copy_((self.P - k_phi_t) / self.lam)

        # Update W: W + e k^\top
        delta_w = torch.outer(err, k)
        self.W_out.add_(delta_w)
        self.last_update_norm = float(torch.linalg.norm(delta_w).item())

    def get_state_norm(self) -> float:
        return float(torch.linalg.norm(self.W_out).item())

    def get_state_memory_bytes(self) -> int:
        return (self.rff_dim * self.rff_dim + self.dim * self.rff_dim) * 4


class SelectiveRetentionPredictor(StreamingPredictor):
    r"""Adaptive DeltaCore Predictor with Selective Retention Control.

    State update rule:
        M_{t+1} = \alpha_t M_t + \eta_t (y_t - M_t \phi(x_t)) \phi(x_t)^\top

    Retention Modes:
        - "fixed_high": \alpha_t = \alpha_{high} (default 0.99)
        - "fixed_low":  \alpha_t = \alpha_{low}  (default 0.70)
        - "adaptive":   \alpha_t = clamp(1.0 - \gamma ||e_t||, \alpha_{min}, 1.0)
        - "oracle":     \alpha_t = 1.0 during stationary, 0.0 at regime shifts
    """

    M: torch.Tensor
    W_phi: torch.Tensor | None

    def __init__(
        self,
        dim: int = 8,
        retention_mode: str = "adaptive",
        use_nonlinear_features: bool = True,
        feat_dim: int = 8,
        eta_max: float = 0.40,
        alpha_high: float = 0.99,
        alpha_low: float = 0.70,
        alpha_min: float = 0.10,
        gamma: float = 1.50,
        oracle_change_points: list[int] | None = None,
        seed: int = 42,
    ) -> None:
        mode_str = f"Selective_{retention_mode}"
        if not use_nonlinear_features:
            mode_str += "_linear"
        super().__init__(name=mode_str, dim=dim)

        self.retention_mode = retention_mode
        self.use_nonlinear_features = use_nonlinear_features
        self.feat_dim = feat_dim
        self.eta_max = eta_max
        self.alpha_high = alpha_high
        self.alpha_low = alpha_low
        self.alpha_min = alpha_min
        self.gamma = gamma
        self.oracle_change_points = set(oracle_change_points or [])
        self.seed = seed

        self.register_buffer("M", torch.zeros(dim, feat_dim, dtype=torch.float32))

        if use_nonlinear_features and feat_dim != dim:
            gen = torch.Generator().manual_seed(seed)
            w_phi = torch.randn(
                feat_dim, dim, generator=gen, dtype=torch.float32
            ) / math.sqrt(dim)
            self.register_buffer("W_phi", w_phi)
        elif use_nonlinear_features:
            self.register_buffer("W_phi", torch.eye(dim, dtype=torch.float32))
        else:
            self.W_phi = None

        self.current_timestep: int = 0
        self.last_retention: float = 1.0
        self.last_step_size: float = 0.0
        self.last_margin: float = 2.0
        self.reset_state()

    def reset_state(self) -> None:
        self.M.zero_()
        self.current_timestep = 0
        self.last_retention = 1.0
        self.last_step_size = 0.0
        self.last_margin = 2.0
        self.last_update_norm = 0.0

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_nonlinear_features:
            return x
        if self.W_phi is not None:
            return torch.tanh(torch.matmul(self.W_phi, x))
        return torch.tanh(x)

    def predict_step(self, x: torch.Tensor) -> torch.Tensor:
        phi = self._phi(x)
        return torch.matmul(self.M, phi)

    def adapt_step(self, x: torch.Tensor, target: torch.Tensor) -> None:
        if not self.adaptation_enabled:
            self.last_update_norm = 0.0
            return

        phi = self._phi(x)
        pred = torch.matmul(self.M, phi)
        err = target - pred
        err_norm = float(torch.linalg.norm(err).item())
        phi_norm_sq = float(torch.dot(phi, phi).item())

        # Determine retention alpha_t
        norm_mode = self.retention_mode.lower()
        if norm_mode in ("fixed_high", "retain_high"):
            alpha_t = self.alpha_high
        elif norm_mode in ("fixed_low", "retain_low"):
            alpha_t = self.alpha_low
        elif norm_mode in ("oracle", "oracle_retention"):
            if self.current_timestep in self.oracle_change_points:
                alpha_t = 0.0
            else:
                alpha_t = 1.0
        elif norm_mode in ("adaptive", "adaptive_retention"):
            raw_alpha = 1.0 - self.gamma * err_norm
            alpha_t = float(
                torch.clamp(torch.tensor(raw_alpha), min=self.alpha_min, max=1.0).item()
            )
        elif norm_mode in ("shuffled", "shuffled_adaptive"):
            # Causal Control D: randomly decorrelated retention from same range [alpha_min, 1.0]
            gen = torch.Generator().manual_seed(self.seed + self.current_timestep * 31)
            rand_val = float(torch.rand(1, generator=gen).item())
            alpha_t = self.alpha_min + (1.0 - self.alpha_min) * rand_val
        else:
            alpha_t = 1.0

        self.last_retention = alpha_t

        # Safe-step contractive bound
        raw_eta = min(self.eta_max, 1.9 / max(phi_norm_sq, 1e-5))
        self.last_step_size = raw_eta
        self.last_margin = 2.0 - (raw_eta * phi_norm_sq)

        # M_{t+1} = \alpha_t M_t + \eta_t e_t \phi_t^\top
        delta_retention = (alpha_t - 1.0) * self.M
        delta_update = raw_eta * torch.outer(err, phi)
        total_delta = delta_retention + delta_update

        self.M.add_(total_delta)
        self.last_update_norm = float(torch.linalg.norm(total_delta).item())
        self.current_timestep += 1

    def get_state_norm(self) -> float:
        return float(torch.linalg.norm(self.M).item())

    def get_state_memory_bytes(self) -> int:
        return self.dim * self.feat_dim * 4


class SelectiveStateAdaptivePredictor(SelectiveRetentionPredictor):
    r"""Adaptive DeltaCore Predictor with State-Conditioned Retention Controller.

    Mode 4: adaptive_state_controller
    Evaluates:
        \alpha_t = \alpha_{min} + (1.0 - \alpha_{min}) * \sigma(w^\top z_t + b)
    where compact feature vector z_t \in \mathbb{R}^4:
        z_t = [ ||e_t||, ||M_t||_F, ||\Delta M_{t-1}||_F, residual_ema_t ]

    Controller parameter count: 4 weights + 1 bias = exactly 5 parameters.
    Deterministic, reproducible, small, explicitly parameter-counted.
    Supports Causal Control D: shuffled_control=True (shuffles/decorrelates controller input).
    """

    w_controller: nn.Parameter
    b_controller: nn.Parameter

    def __init__(
        self,
        dim: int = 8,
        use_nonlinear_features: bool = True,
        feat_dim: int = 8,
        eta_max: float = 0.40,
        alpha_min: float = 0.10,
        shuffled_control: bool = False,
        ema_decay: float = 0.90,
        seed: int = 42,
    ) -> None:
        mode_name = (
            "Selective_state_adaptive"
            if not shuffled_control
            else "Selective_shuffled_control"
        )
        if not use_nonlinear_features:
            mode_name += "_linear"
        super().__init__(
            dim=dim,
            retention_mode="state_adaptive",
            use_nonlinear_features=use_nonlinear_features,
            feat_dim=feat_dim,
            eta_max=eta_max,
            alpha_min=alpha_min,
            seed=seed,
        )
        self.name = mode_name
        self.shuffled_control = shuffled_control
        self.ema_decay = ema_decay
        self.seed = seed

        # Explicitly parameter-counted small controller (4 weights, 1 bias = 5 parameters)
        self.w_controller = nn.Parameter(
            torch.tensor([-1.20, -0.05, -0.60, -0.60], dtype=torch.float32),
            requires_grad=False,
        )
        self.b_controller = nn.Parameter(
            torch.tensor([2.00], dtype=torch.float32),
            requires_grad=False,
        )

        self.last_delta_m_norm: float = 0.0
        self.residual_ema: float = 0.0
        self.reset_state()

    def reset_state(self) -> None:
        super().reset_state()
        self.last_delta_m_norm = 0.0
        self.residual_ema = 0.0

    def adapt_step(self, x: torch.Tensor, target: torch.Tensor) -> None:
        if not self.adaptation_enabled:
            self.last_update_norm = 0.0
            return

        phi = self._phi(x)
        pred = torch.matmul(self.M, phi)
        err = target - pred
        err_norm = float(torch.linalg.norm(err).item())
        phi_norm_sq = float(torch.dot(phi, phi).item())
        m_norm = float(torch.linalg.norm(self.M).item())

        # Update residual EMA
        if self.current_timestep == 0:
            self.residual_ema = err_norm
        else:
            self.residual_ema = (
                self.ema_decay * self.residual_ema + (1.0 - self.ema_decay) * err_norm
            )

        # Build compact state feature vector z_t = [ ||e_t||, ||M_t||_F, ||\Delta M_{t-1}||_F, residual_ema_t ]
        z = torch.tensor(
            [err_norm, m_norm, self.last_delta_m_norm, self.residual_ema],
            dtype=torch.float32,
        )

        if self.shuffled_control:
            # Causal Control D: Shuffled/decorrelated control with identical parameter count
            gen = torch.Generator().manual_seed(self.seed + self.current_timestep * 37)
            perm = torch.randperm(4, generator=gen)
            z = z[perm] + torch.randn(4, generator=gen) * 0.1

        # Affine controller activation
        v_t = float(torch.dot(self.w_controller, z).item() + self.b_controller.item())
        s_t = 1.0 / (1.0 + math.exp(-max(min(v_t, 20.0), -20.0)))
        alpha_t = self.alpha_min + (1.0 - self.alpha_min) * s_t
        self.last_retention = alpha_t

        # Safe-step contractive bound
        raw_eta = min(self.eta_max, 1.9 / max(phi_norm_sq, 1e-5))
        self.last_step_size = raw_eta
        self.last_margin = 2.0 - (raw_eta * phi_norm_sq)

        # M_{t+1} = \alpha_t M_t + \eta_t e_t \phi_t^\top
        delta_retention = (alpha_t - 1.0) * self.M
        delta_update = raw_eta * torch.outer(err, phi)
        total_delta = delta_retention + delta_update

        self.M.add_(total_delta)
        delta_norm = float(torch.linalg.norm(total_delta).item())
        self.last_update_norm = delta_norm
        self.last_delta_m_norm = delta_norm
        self.current_timestep += 1

    def get_state_memory_bytes(self) -> int:
        return self.dim * self.feat_dim * 4 + 16


class SpatialConvControl(StreamingPredictor):
    r"""Baseline / Representation Control C: Simple static spatial convolution.

    Accepts flattened x_t \in \mathbb{R}^D, reshapes to (1, C, H, W),
    applies a small frozen 3x3 Conv2d layer + ReLU + Linear head back to D.
    Zero parameter adaptation at test time.
    """

    def __init__(
        self,
        height: int = 8,
        width: int = 8,
        channels: int = 1,
        conv_channels: int = 4,
        seed: int = 42,
    ) -> None:
        dim = height * width * channels
        super().__init__(name="SpatialConvControl", dim=dim)
        self.height = height
        self.width = width
        self.channels = channels
        self.conv_channels = conv_channels

        gen = torch.Generator().manual_seed(seed)
        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=conv_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.head = nn.Linear(conv_channels * height * width, dim, bias=True)

        nn.init.kaiming_normal_(self.conv.weight, generator=gen)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        nn.init.xavier_normal_(self.head.weight, generator=gen)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

        for p in self.parameters():
            p.requires_grad = False

        self.last_pred = torch.zeros(dim, dtype=torch.float32)

    def reset_state(self) -> None:
        self.last_update_norm = 0.0
        self.last_pred.zero_()

    def predict_step(self, x: torch.Tensor) -> torch.Tensor:
        x_spatial = x.view(1, self.channels, self.height, self.width)
        feat = torch.relu(self.conv(x_spatial))
        pred = self.head(feat.view(1, -1)).view(-1)
        self.last_pred = pred
        return pred

    def adapt_step(self, x: torch.Tensor, target: torch.Tensor) -> None:
        self.last_update_norm = 0.0

    def get_state_norm(self) -> float:
        return 0.0

    def get_state_memory_bytes(self) -> int:
        return 0


class SpatialDownsampleControl(StreamingPredictor):
    r"""Baseline / Representation Control B: Fixed local pooling/downsampling.

    Accepts flattened x_t \in \mathbb{R}^D, reshapes to (1, C, H, W),
    applies AvgPool2d(2, stride=2) + Linear projection back to D.
    Zero parameter adaptation at test time.
    """

    def __init__(
        self,
        height: int = 8,
        width: int = 8,
        channels: int = 1,
        seed: int = 42,
    ) -> None:
        dim = height * width * channels
        super().__init__(name="SpatialDownsampleControl", dim=dim)
        self.height = height
        self.width = width
        self.channels = channels
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        down_h = height // 2
        down_w = width // 2
        down_dim = channels * down_h * down_w
        self.head = nn.Linear(down_dim, dim, bias=True)

        gen = torch.Generator().manual_seed(seed)
        nn.init.xavier_normal_(self.head.weight, generator=gen)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

        for p in self.parameters():
            p.requires_grad = False

    def reset_state(self) -> None:
        self.last_update_norm = 0.0

    def predict_step(self, x: torch.Tensor) -> torch.Tensor:
        x_spatial = x.view(1, self.channels, self.height, self.width)
        down = self.pool(x_spatial)
        pred = self.head(down.view(1, -1)).view(-1)
        return pred

    def adapt_step(self, x: torch.Tensor, target: torch.Tensor) -> None:
        self.last_update_norm = 0.0

    def get_state_norm(self) -> float:
        return 0.0

    def get_state_memory_bytes(self) -> int:
        return 0
