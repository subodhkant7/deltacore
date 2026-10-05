"""Phase 18: Classification Model Matrix & Frozen Adaptive Mechanism.

Implements all required models under strict parameter immutability (Delta theta = 0):
    - Frozen / Non-Adaptive: FrozenLinearClassifier18, StateOffAblationClassifier18
    - Classical Online: OnlineLogisticRegression18, OnlineRidgeClassifier18, OnlineMulticlassLinear18
    - DeltaCore: FixedDeltaClassifier18, SafeAdaptiveDeltaClassifier18
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from deltacore.benchmarks.phase_18.generators import StationaryDataset18


def compute_model_parameter_hash_18(model: nn.Module | OnlinePredictor18) -> str:
    """Compute SHA-256 hash over all model parameters to verify test-time immutability."""
    hasher = hashlib.sha256()
    if isinstance(model, nn.Module):
        for p in model.parameters():
            hasher.update(p.detach().cpu().numpy().tobytes())
    elif hasattr(model, "get_parameter_tensors"):
        for t in model.get_parameter_tensors():
            hasher.update(t.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


class OnlinePredictor18(ABC):
    """Abstract base class for Phase 18 online classification models."""

    num_classes: int

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier."""
        ...

    @abstractmethod
    def predict_step(self, x: torch.Tensor) -> tuple[int, torch.Tensor]:
        """Produce predicted class label and predicted probability distribution.

        Args:
            x: Input feature vector of shape (D,).

        Returns:
            Tuple of (predicted_class_int, probability_tensor_of_shape_(K,)).
        """
        ...

    @abstractmethod
    def adapt_step(self, x: torch.Tensor, y: int) -> None:
        """Update adaptive state upon revelation of ground-truth label y.

        Args:
            x: Input feature vector of shape (D,).
            y: Ground-truth integer class label in [0, K-1].
        """
        ...

    @abstractmethod
    def reset_state(self) -> None:
        """Reset internal online adaptive state to initial state (M_0 = 0)."""
        ...

    @abstractmethod
    def get_state_memory_bytes(self) -> int:
        """Return memory footprint of persistent adaptive state in bytes."""
        ...

    @abstractmethod
    def get_parameter_count(self) -> int:
        """Return total parameter count."""
        ...

    def get_parameter_hash(self) -> str:
        """Return SHA-256 hash of offline parameters."""
        return compute_model_parameter_hash_18(self)

    def get_parameter_tensors(self) -> Sequence[torch.Tensor]:
        """Return sequence of parameter tensors."""
        return []


# ==============================================================================
# Frozen Neural Baselines
# ==============================================================================


class FrozenLinearClassifier18(OnlinePredictor18):
    """Offline-fit multinomial linear classifier with zero test-time adaptation."""

    def __init__(
        self, dim: int, num_classes: int, weight: torch.Tensor, bias: torch.Tensor
    ) -> None:
        self.dim = dim
        self.num_classes = num_classes
        self.weight = weight.clone().detach()  # (K, D)
        self.bias = bias.clone().detach()  # (K,)

    @property
    def name(self) -> str:
        return "FrozenLinear"

    def predict_step(self, x: torch.Tensor) -> tuple[int, torch.Tensor]:
        logits = self.weight @ x + self.bias
        probs = F.softmax(logits, dim=-1)
        pred = int(torch.argmax(logits).item())
        return pred, probs

    def adapt_step(self, x: torch.Tensor, y: int) -> None:
        pass  # Frozen: zero adaptation

    def reset_state(self) -> None:
        pass

    def get_state_memory_bytes(self) -> int:
        return 0

    def get_parameter_count(self) -> int:
        return self.weight.numel() + self.bias.numel()

    def get_parameter_tensors(self) -> Sequence[torch.Tensor]:
        return [self.weight, self.bias]


class StateOffAblationClassifier18(OnlinePredictor18):
    """Primary Causal Control: SafeAdaptiveDelta with associative memory clamped to zero (M_t = 0)."""

    def __init__(
        self, dim: int, num_classes: int, weight: torch.Tensor, bias: torch.Tensor
    ) -> None:
        self.dim = dim
        self.num_classes = num_classes
        self.weight = weight.clone().detach()
        self.bias = bias.clone().detach()

    @property
    def name(self) -> str:
        return "AdaptiveStateOFF"

    def predict_step(self, x: torch.Tensor) -> tuple[int, torch.Tensor]:
        # Clamped state: z_t = (I + 0) x_t = x_t
        logits = self.weight @ x + self.bias
        probs = F.softmax(logits, dim=-1)
        pred = int(torch.argmax(logits).item())
        return pred, probs

    def adapt_step(self, x: torch.Tensor, y: int) -> None:
        pass  # Clamped to zero: no state updates

    def reset_state(self) -> None:
        pass

    def get_state_memory_bytes(self) -> int:
        return 0

    def get_parameter_count(self) -> int:
        return self.weight.numel() + self.bias.numel()

    def get_parameter_tensors(self) -> Sequence[torch.Tensor]:
        return [self.weight, self.bias]


# ==============================================================================
# Classical Online Baselines
# ==============================================================================


class OnlineLogisticRegression18(OnlinePredictor18):
    """Multiclass online logistic regression via continuous online gradient descent."""

    def __init__(self, dim: int, num_classes: int, lr: float = 0.05) -> None:
        self.dim = dim
        self.num_classes = num_classes
        self.lr = lr
        self.weight = torch.zeros(num_classes, dim)
        self.bias = torch.zeros(num_classes)
        self._initial_weight = self.weight.clone()
        self._initial_bias = self.bias.clone()

    @property
    def name(self) -> str:
        return "OnlineLogisticRegression"

    def predict_step(self, x: torch.Tensor) -> tuple[int, torch.Tensor]:
        logits = self.weight @ x + self.bias
        probs = F.softmax(logits, dim=-1)
        pred = int(torch.argmax(logits).item())
        return pred, probs

    def adapt_step(self, x: torch.Tensor, y: int) -> None:
        _, probs = self.predict_step(x)
        y_one_hot = torch.zeros(self.num_classes)
        y_one_hot[y] = 1.0
        grad = y_one_hot - probs
        self.weight = self.weight + self.lr * torch.outer(grad, x)
        self.bias = self.bias + self.lr * grad

    def reset_state(self) -> None:
        self.weight = self._initial_weight.clone()
        self.bias = self._initial_bias.clone()

    def get_state_memory_bytes(self) -> int:
        return (self.weight.numel() + self.bias.numel()) * 4

    def get_parameter_count(self) -> int:
        return self.weight.numel() + self.bias.numel()

    def get_parameter_tensors(self) -> Sequence[torch.Tensor]:
        return [self._initial_weight, self._initial_bias]


class OnlineRidgeClassifier18(OnlinePredictor18):
    """Recursive Least Squares (RLS) online ridge classifier on one-hot targets."""

    def __init__(
        self, dim: int, num_classes: int, lam: float = 0.99, delta: float = 50.0
    ) -> None:
        self.dim = dim
        self.num_classes = num_classes
        self.lam = lam
        self.delta = delta
        self.P = torch.eye(dim) * delta
        self.W = torch.zeros(num_classes, dim)
        self._initial_P = self.P.clone()
        self._initial_W = self.W.clone()

    @property
    def name(self) -> str:
        return "OnlineRidge"

    def predict_step(self, x: torch.Tensor) -> tuple[int, torch.Tensor]:
        scores = self.W @ x
        probs = F.softmax(scores, dim=-1)
        pred = int(torch.argmax(scores).item())
        return pred, probs

    def adapt_step(self, x: torch.Tensor, y: int) -> None:
        y_one_hot = torch.zeros(self.num_classes)
        y_one_hot[y] = 1.0
        Px = self.P @ x
        denom = float(self.lam + torch.dot(x, Px).item())
        if abs(denom) < 1e-8:
            return
        k_gain = Px / denom
        error = y_one_hot - self.W @ x
        self.W = self.W + torch.outer(error, k_gain)
        self.P = (self.P - torch.outer(k_gain, Px)) / self.lam

    def reset_state(self) -> None:
        self.P = self._initial_P.clone()
        self.W = self._initial_W.clone()

    def get_state_memory_bytes(self) -> int:
        return (self.P.numel() + self.W.numel()) * 4

    def get_parameter_count(self) -> int:
        return self.W.numel()

    def get_parameter_tensors(self) -> Sequence[torch.Tensor]:
        return [self._initial_W]


class OnlineMulticlassLinear18(OnlinePredictor18):
    """Online multiclass linear error-correcting perceptron / Widrow-Hoff estimator."""

    def __init__(self, dim: int, num_classes: int, step_size: float = 0.05) -> None:
        self.dim = dim
        self.num_classes = num_classes
        self.step_size = step_size
        self.weight = torch.zeros(num_classes, dim)
        self.bias = torch.zeros(num_classes)
        self._initial_weight = self.weight.clone()
        self._initial_bias = self.bias.clone()

    @property
    def name(self) -> str:
        return "OnlineMulticlassLinear"

    def predict_step(self, x: torch.Tensor) -> tuple[int, torch.Tensor]:
        scores = self.weight @ x + self.bias
        probs = F.softmax(scores, dim=-1)
        pred = int(torch.argmax(scores).item())
        return pred, probs

    def adapt_step(self, x: torch.Tensor, y: int) -> None:
        pred, _ = self.predict_step(x)
        if pred != y:
            norm_sq = float(torch.dot(x, x).item() + 1.0)
            eta = self.step_size / norm_sq
            self.weight[y] += eta * x
            self.bias[y] += eta
            self.weight[pred] -= eta * x
            self.bias[pred] -= eta

    def reset_state(self) -> None:
        self.weight = self._initial_weight.clone()
        self.bias = self._initial_bias.clone()

    def get_state_memory_bytes(self) -> int:
        return (self.weight.numel() + self.bias.numel()) * 4

    def get_parameter_count(self) -> int:
        return self.weight.numel() + self.bias.numel()

    def get_parameter_tensors(self) -> Sequence[torch.Tensor]:
        return [self._initial_weight, self._initial_bias]


# ==============================================================================
# DeltaCore Predictors (Frozen Mechanism Section 2)
# ==============================================================================


class SafeAdaptiveDeltaClassifier18(OnlinePredictor18):
    r"""Phase 18 Frozen SafeAdaptiveDelta Classification Model.

    Predictor:
        z_t = x_t + M_t x_t = (I + M_t) x_t
        s_t = W_head z_t + b_head
        \hat{p}_t = softmax(s_t)
        \hat{y}_t = argmax_k s_{t, k}

    Online State Update (Frozen Mechanism Section 2):
        e_{logit, t} = y_{one_hot, t} - \hat{p}_t \in R^K
        e_{x, t} = W_head^\top e_{logit, t} \in R^D
        \eta_t = \min\left(\frac{\eta_0}{1 + \gamma \|e_{x, t}\|_2^2}, \frac{\rho}{\|x_t\|_2^2 + \epsilon}\right)
        \alpha_t = \max(\alpha_{min}, 1 - \eta_t \|x_t\|_2^2)
        M_{t+1} = \alpha_t M_t + \eta_t e_{x, t} x_t^\top

    Invariant: \Delta\theta = 0 (Offline head is strictly immutable).
    """

    def __init__(
        self,
        dim: int,
        num_classes: int,
        weight_head: torch.Tensor,
        bias_head: torch.Tensor,
        eta0: float = 0.015,
        rho: float = 1.50,
        alpha_min: float = 0.95,
        gamma: float = 0.10,
        epsilon: float = 1e-6,
    ) -> None:
        self.dim = dim
        self.num_classes = num_classes
        self.weight_head = weight_head.clone().detach()  # (K, D)
        self.bias_head = bias_head.clone().detach()  # (K,)
        self.eta0 = eta0
        self.rho = rho
        self.alpha_min = alpha_min
        self.gamma = gamma
        self.epsilon = epsilon

        # Persistent associative state (exact 4*D^2 bytes in FP32)
        self.M = torch.zeros(dim, dim)

        # Telemetry
        self.last_eta: float = 0.0
        self.last_safety_margin: float = 2.0
        self.last_retention: float = 1.0
        self.last_update_norm: float = 0.0

    @property
    def name(self) -> str:
        return "SafeAdaptiveDelta"

    def predict_step(self, x: torch.Tensor) -> tuple[int, torch.Tensor]:
        # Associative state projection: z_t = (I + M_t) x_t
        z = x + self.M @ x
        logits = self.weight_head @ z + self.bias_head
        probs = F.softmax(logits, dim=-1)
        pred = int(torch.argmax(logits).item())
        return pred, probs

    def adapt_step(self, x: torch.Tensor, y: int) -> None:
        # Evaluate current prediction
        _, probs = self.predict_step(x)

        # Compute logit prediction error
        y_one_hot = torch.zeros(self.num_classes)
        y_one_hot[y] = 1.0
        e_logit = y_one_hot - probs  # (K,)

        # Backproject logit error into feature coordinate space
        e_x = self.weight_head.T @ e_logit  # (D,)

        # Candidate step size with squared error dampening (Section 2)
        e_x_norm_sq = float(torch.dot(e_x, e_x).item())
        eta_cand = self.eta0 / (1.0 + self.gamma * e_x_norm_sq)

        # Contraction bound enforcing local spectral contractivity: eta_t ||x_t||^2 <= rho
        x_norm_sq = float(torch.dot(x, x).item())
        bound = self.rho / (x_norm_sq + self.epsilon)
        eta_t = min(eta_cand, bound)

        # Local safety margin
        safety_margin = 1.0 - (eta_t * x_norm_sq / self.rho)

        # Adaptive retention factor
        alpha_t = max(self.alpha_min, 1.0 - eta_t * x_norm_sq)

        # Associative state update: M_{t+1} = alpha_t M_t + eta_t e_x x^T
        delta_M = eta_t * torch.outer(e_x, x)
        self.M = alpha_t * self.M + delta_M

        # Telemetry
        self.last_eta = eta_t
        self.last_safety_margin = float(safety_margin)
        self.last_retention = float(alpha_t)
        self.last_update_norm = float(torch.linalg.norm(delta_M).item())

    def reset_state(self) -> None:
        self.M = torch.zeros(self.dim, self.dim)
        self.last_eta = 0.0
        self.last_safety_margin = 2.0
        self.last_retention = 1.0
        self.last_update_norm = 0.0

    def get_state_memory_bytes(self) -> int:
        return self.M.numel() * 4  # Exactly 4*D^2 bytes

    def get_parameter_count(self) -> int:
        return self.weight_head.numel() + self.bias_head.numel()

    def get_parameter_tensors(self) -> Sequence[torch.Tensor]:
        return [self.weight_head, self.bias_head]


class FixedDeltaClassifier18(OnlinePredictor18):
    """Fixed-step associative delta classifier without dynamic contraction bounds."""

    def __init__(
        self,
        dim: int,
        num_classes: int,
        weight_head: torch.Tensor,
        bias_head: torch.Tensor,
        step_size: float = 0.015,
        alpha: float = 0.95,
    ) -> None:
        self.dim = dim
        self.num_classes = num_classes
        self.weight_head = weight_head.clone().detach()
        self.bias_head = bias_head.clone().detach()
        self.step_size = step_size
        self.alpha = alpha

        self.M = torch.zeros(dim, dim)

        self.last_eta: float = step_size
        self.last_safety_margin: float = 2.0
        self.last_retention: float = alpha
        self.last_update_norm: float = 0.0

    @property
    def name(self) -> str:
        return "FixedDelta"

    def predict_step(self, x: torch.Tensor) -> tuple[int, torch.Tensor]:
        z = x + self.M @ x
        logits = self.weight_head @ z + self.bias_head
        probs = F.softmax(logits, dim=-1)
        pred = int(torch.argmax(logits).item())
        return pred, probs

    def adapt_step(self, x: torch.Tensor, y: int) -> None:
        _, probs = self.predict_step(x)
        y_one_hot = torch.zeros(self.num_classes)
        y_one_hot[y] = 1.0
        e_logit = y_one_hot - probs
        e_x = self.weight_head.T @ e_logit
        delta_M = self.step_size * torch.outer(e_x, x)
        self.M = self.alpha * self.M + delta_M
        self.last_update_norm = float(torch.linalg.norm(delta_M).item())

    def reset_state(self) -> None:
        self.M = torch.zeros(self.dim, self.dim)
        self.last_update_norm = 0.0

    def get_state_memory_bytes(self) -> int:
        return self.M.numel() * 4

    def get_parameter_count(self) -> int:
        return self.weight_head.numel() + self.bias_head.numel()

    def get_parameter_tensors(self) -> Sequence[torch.Tensor]:
        return [self.weight_head, self.bias_head]


# ==============================================================================
# Offline Pre-Training & Suite Factory
# ==============================================================================


def fit_offline_linear_head_18(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    epochs: int = 40,
    lr: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Train linear classifier head on stationary split via cross-entropy loss."""
    dim = inputs.size(1)
    linear = nn.Linear(dim, num_classes)
    optimizer = torch.optim.Adam(linear.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        out = linear(inputs)
        loss = loss_fn(out, targets)
        loss.backward()
        optimizer.step()

    weight = linear.weight.detach().clone()
    bias = linear.bias.detach().clone()
    return weight, bias


def build_phase_18_model_suite(
    stat_ds: StationaryDataset18,
    seed: int = 42,
) -> dict[str, OnlinePredictor18]:
    """Instantiate complete Phase 18 model matrix under frozen evaluation configuration."""
    del seed
    dim = stat_ds.dim
    num_classes = stat_ds.num_classes

    # Fit canonical offline linear classifier head on stationary Regime A train split
    w_head, b_head = fit_offline_linear_head_18(
        stat_ds.train_inputs, stat_ds.train_targets, num_classes
    )

    suite: dict[str, OnlinePredictor18] = {
        "FrozenLinear": FrozenLinearClassifier18(dim, num_classes, w_head, b_head),
        "AdaptiveStateOFF": StateOffAblationClassifier18(
            dim, num_classes, w_head, b_head
        ),
        "OnlineLogisticRegression": OnlineLogisticRegression18(
            dim, num_classes, lr=0.05
        ),
        "OnlineRidge": OnlineRidgeClassifier18(dim, num_classes, lam=0.99, delta=50.0),
        "OnlineMulticlassLinear": OnlineMulticlassLinear18(
            dim, num_classes, step_size=0.05
        ),
        "FixedDelta": FixedDeltaClassifier18(
            dim, num_classes, w_head, b_head, step_size=0.015, alpha=0.95
        ),
        "SafeAdaptiveDelta": SafeAdaptiveDeltaClassifier18(
            dim,
            num_classes,
            w_head,
            b_head,
            eta0=0.015,
            rho=1.50,
            alpha_min=0.95,
            gamma=0.10,
        ),
    }

    return suite
