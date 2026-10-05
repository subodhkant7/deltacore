"""Phase 17: Classification Model Matrix & Adaptive State Predictors.

Implements all primary models for online non-stationary classification:
    - Classical: OnlineLogisticRegression, OnlineRidgeClassifier, OnlineMulticlassLinear
    - Frozen Neural: FrozenLinearClassifier, SmallMLPClassifier
    - Recurrent: GRUClassifier, LSTMClassifier
    - DeltaCore: SafeAdaptiveDeltaClassifier, FixedDeltaClassifier, AblationStateOff
Strict online/offline parameter separation: Delta theta = 0 verified via cryptographic hashes.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_model_parameter_hash(
    model: nn.Module | OnlineClassificationPredictor,
) -> str:
    """Compute SHA-256 hash over all model parameters to verify immutability."""
    hasher = hashlib.sha256()
    if isinstance(model, nn.Module):
        for p in model.parameters():
            hasher.update(p.detach().cpu().numpy().tobytes())
    elif hasattr(model, "get_parameter_tensors"):
        for t in model.get_parameter_tensors():
            hasher.update(t.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


class OnlineClassificationPredictor(ABC):
    """Abstract base class for all online classification models."""

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
        """Reset online adaptive state to initial state (M_t = 0 or h_t = 0)."""
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
        """Return hash of offline parameters."""
        return compute_model_parameter_hash(self)

    def get_parameter_tensors(self) -> Sequence[torch.Tensor]:
        """Return list of parameter tensors."""
        return []


class FrozenLinearClassifier(OnlineClassificationPredictor):
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
        pass  # Frozen: no online adaptation

    def reset_state(self) -> None:
        pass

    def get_state_memory_bytes(self) -> int:
        return 0

    def get_parameter_count(self) -> int:
        return self.weight.numel() + self.bias.numel()

    def get_parameter_tensors(self) -> list[torch.Tensor]:
        return [self.weight, self.bias]


class SmallMLPClassifier(OnlineClassificationPredictor):
    """Small 2-layer MLP classifier fit offline with zero test-time adaptation."""

    def __init__(
        self, dim: int, hidden_dim: int, num_classes: int, model: nn.Module
    ) -> None:
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.net = model
        self.net.eval()

    @property
    def name(self) -> str:
        return "SmallMLP"

    def predict_step(self, x: torch.Tensor) -> tuple[int, torch.Tensor]:
        with torch.no_grad():
            logits = self.net(x.unsqueeze(0)).squeeze(0)
            probs = F.softmax(logits, dim=-1)
            pred = int(torch.argmax(logits).item())
            return pred, probs

    def adapt_step(self, x: torch.Tensor, y: int) -> None:
        pass

    def reset_state(self) -> None:
        pass

    def get_state_memory_bytes(self) -> int:
        return 0

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def get_parameter_tensors(self) -> list[torch.Tensor]:
        return list(self.net.parameters())


class OnlineLogisticRegression(OnlineClassificationPredictor):
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
        grad = y_one_hot - probs  # (K,)
        self.weight = self.weight + self.lr * torch.outer(grad, x)
        self.bias = self.bias + self.lr * grad

    def reset_state(self) -> None:
        self.weight = self._initial_weight.clone()
        self.bias = self._initial_bias.clone()

    def get_state_memory_bytes(self) -> int:
        return (self.weight.numel() + self.bias.numel()) * 4

    def get_parameter_count(self) -> int:
        return self.weight.numel() + self.bias.numel()

    def get_parameter_tensors(self) -> list[torch.Tensor]:
        return [self._initial_weight, self._initial_bias]


class OnlineRidgeClassifier(OnlineClassificationPredictor):
    """Recursive Least Squares online ridge classifier on one-hot targets."""

    def __init__(
        self, dim: int, num_classes: int, lam: float = 0.99, delta: float = 100.0
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
        denom = self.lam + float(x @ Px)
        k_t = Px / max(denom, 1e-6)

        err = y_one_hot - (self.W @ x)
        self.W = self.W + torch.outer(err, k_t)
        self.P = (self.P - torch.outer(k_t, Px)) / self.lam

    def reset_state(self) -> None:
        self.P = self._initial_P.clone()
        self.W = self._initial_W.clone()

    def get_state_memory_bytes(self) -> int:
        return (self.P.numel() + self.W.numel()) * 4

    def get_parameter_count(self) -> int:
        return self.W.numel()

    def get_parameter_tensors(self) -> list[torch.Tensor]:
        return [self._initial_W]


class OnlineMulticlassLinear(OnlineClassificationPredictor):
    """Online Multiclass Passive-Aggressive linear estimator."""

    def __init__(self, dim: int, num_classes: int, step_size: float = 0.1) -> None:
        self.dim = dim
        self.num_classes = num_classes
        self.step_size = step_size
        self.weight = torch.zeros(num_classes, dim)
        self._initial_weight = self.weight.clone()

    @property
    def name(self) -> str:
        return "OnlineMulticlassLinear"

    def predict_step(self, x: torch.Tensor) -> tuple[int, torch.Tensor]:
        scores = self.weight @ x
        probs = F.softmax(scores, dim=-1)
        pred = int(torch.argmax(scores).item())
        return pred, probs

    def adapt_step(self, x: torch.Tensor, y: int) -> None:
        pred, _ = self.predict_step(x)
        if pred != y:
            norm_sq = float((x @ x).item()) + 1e-6
            tau = self.step_size / norm_sq
            self.weight[y] += tau * x
            self.weight[pred] -= tau * x

    def reset_state(self) -> None:
        self.weight = self._initial_weight.clone()

    def get_state_memory_bytes(self) -> int:
        return self.weight.numel() * 4

    def get_parameter_count(self) -> int:
        return self.weight.numel()

    def get_parameter_tensors(self) -> list[torch.Tensor]:
        return [self._initial_weight]


class GRUClassifier(OnlineClassificationPredictor):
    """Recurrent GRU classifier with frozen parameters and persistent recurrent hidden state."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_classes: int,
        cell: nn.GRUCell,
        head: nn.Linear,
    ) -> None:
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.cell = cell
        self.head = head
        self.h = torch.zeros(1, hidden_dim)

    @property
    def name(self) -> str:
        return "GRU"

    def predict_step(self, x: torch.Tensor) -> tuple[int, torch.Tensor]:
        with torch.no_grad():
            new_h = self.cell(x.unsqueeze(0), self.h)
            logits = self.head(new_h).squeeze(0)
            probs = F.softmax(logits, dim=-1)
            pred = int(torch.argmax(logits).item())
            return pred, probs

    def adapt_step(self, x: torch.Tensor, y: int) -> None:
        with torch.no_grad():
            self.h = self.cell(x.unsqueeze(0), self.h)

    def reset_state(self) -> None:
        self.h = torch.zeros(1, self.hidden_dim)

    def get_state_memory_bytes(self) -> int:
        return self.h.numel() * 4

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.cell.parameters()) + sum(
            p.numel() for p in self.head.parameters()
        )

    def get_parameter_tensors(self) -> Sequence[torch.Tensor]:
        return list(self.cell.parameters()) + list(self.head.parameters())


class LSTMClassifier(OnlineClassificationPredictor):
    """Recurrent LSTM classifier with frozen parameters and persistent recurrent (h, c) state."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_classes: int,
        cell: nn.LSTMCell,
        head: nn.Linear,
    ) -> None:
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.cell = cell
        self.head = head
        self.h = torch.zeros(1, hidden_dim)
        self.c = torch.zeros(1, hidden_dim)

    @property
    def name(self) -> str:
        return "LSTM"

    def predict_step(self, x: torch.Tensor) -> tuple[int, torch.Tensor]:
        with torch.no_grad():
            new_h, _ = self.cell(x.unsqueeze(0), (self.h, self.c))
            logits = self.head(new_h).squeeze(0)
            probs = F.softmax(logits, dim=-1)
            pred = int(torch.argmax(logits).item())
            return pred, probs

    def adapt_step(self, x: torch.Tensor, y: int) -> None:
        with torch.no_grad():
            self.h, self.c = self.cell(x.unsqueeze(0), (self.h, self.c))

    def reset_state(self) -> None:
        self.h = torch.zeros(1, self.hidden_dim)
        self.c = torch.zeros(1, self.hidden_dim)

    def get_state_memory_bytes(self) -> int:
        return (self.h.numel() + self.c.numel()) * 4

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.cell.parameters()) + sum(
            p.numel() for p in self.head.parameters()
        )

    def get_parameter_tensors(self) -> Sequence[torch.Tensor]:
        return list(self.cell.parameters()) + list(self.head.parameters())


class SafeAdaptiveDeltaClassifier(OnlineClassificationPredictor):
    r"""DeltaCore SafeAdaptiveDelta Classification Model.

    Maintains frozen offline classifier head (W_head, b_head) and online associative
    state matrix M_t in R^{D x D}.

    Prediction:
        z_t = x_t + M_t x_t
        s_t = W_head z_t + b_head
        \hat{p}_t = softmax(s_t)
        \hat{y}_t = argmax_k s_{t, k}

    Adaptive Update:
        e_{logit, t} = y_{one_hot, t} - \hat{p}_t \in R^K
        e_{x, t} = W_head^\top e_{logit, t} \in R^D
        \eta_{cand, t} = \frac{\eta_0}{1 + \gamma \|e_{x, t}\|_2}
        \eta_t = \min(\eta_{cand, t}, \frac{\rho}{\|x_t\|_2^2 + \epsilon})
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
        eta0: float = 0.05,
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

        # Online associative state
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
        # Form adapted feature representation z_t = (I + M_t) x_t
        z = x + self.M @ x
        logits = self.weight_head @ z + self.bias_head
        probs = F.softmax(logits, dim=-1)
        pred = int(torch.argmax(logits).item())
        return pred, probs

    def adapt_step(self, x: torch.Tensor, y: int) -> None:
        _, probs = self.predict_step(x)
        y_one_hot = torch.zeros(self.num_classes)
        y_one_hot[y] = 1.0

        # Gradient of cross-entropy w.r.t logits
        e_logit = y_one_hot - probs  # (K,)
        # Backproject logit error into feature space via fixed head
        e_x = self.weight_head.T @ e_logit  # (D,)

        e_x_norm = float(torch.linalg.norm(e_x).item())
        x_norm_sq = float(torch.dot(x, x).item())

        # 1. Candidate step size
        eta_cand = self.eta0 / (1.0 + self.gamma * e_x_norm)

        # 2. Contractive bound
        eta_safe = self.rho / (x_norm_sq + self.epsilon)
        eta_t = min(eta_cand, eta_safe)

        # 3. Adaptive retention factor
        alpha_t = max(self.alpha_min, 1.0 - eta_t * x_norm_sq)

        # 4. State update: M_{t+1} = \alpha_t M_t + \eta_t e_x x^\top
        delta_M = eta_t * torch.outer(e_x, x)
        self.M = alpha_t * self.M + delta_M

        # Telemetry
        self.last_eta = eta_t
        self.last_safety_margin = max(0.0, 2.0 - eta_t * x_norm_sq)
        self.last_retention = alpha_t
        self.last_update_norm = float(torch.linalg.norm(delta_M).item())

    def reset_state(self) -> None:
        self.M = torch.zeros(self.dim, self.dim)
        self.last_eta = 0.0
        self.last_safety_margin = 2.0
        self.last_retention = 1.0
        self.last_update_norm = 0.0

    def get_state_memory_bytes(self) -> int:
        return self.M.numel() * 4

    def get_parameter_count(self) -> int:
        return self.weight_head.numel() + self.bias_head.numel()

    def get_parameter_tensors(self) -> list[torch.Tensor]:
        return [self.weight_head, self.bias_head]


class FixedDeltaClassifier(OnlineClassificationPredictor):
    """DeltaCore associative classifier with fixed step size and retention (unconstrained baseline)."""

    def __init__(
        self,
        dim: int,
        num_classes: int,
        weight_head: torch.Tensor,
        bias_head: torch.Tensor,
        step_size: float = 0.05,
        alpha: float = 0.99,
    ) -> None:
        self.dim = dim
        self.num_classes = num_classes
        self.weight_head = weight_head.clone().detach()
        self.bias_head = bias_head.clone().detach()
        self.step_size = step_size
        self.alpha = alpha
        self.M = torch.zeros(dim, dim)
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

    def get_parameter_tensors(self) -> list[torch.Tensor]:
        return [self.weight_head, self.bias_head]


class StateOffAblationClassifier(OnlineClassificationPredictor):
    """SafeAdaptiveDelta architecture with internal associative state clamped to zero (M_t = 0)."""

    def __init__(
        self,
        dim: int,
        num_classes: int,
        weight_head: torch.Tensor,
        bias_head: torch.Tensor,
    ) -> None:
        self.dim = dim
        self.num_classes = num_classes
        self.weight_head = weight_head.clone().detach()
        self.bias_head = bias_head.clone().detach()

    @property
    def name(self) -> str:
        return "AdaptiveStateOFF"

    def predict_step(self, x: torch.Tensor) -> tuple[int, torch.Tensor]:
        logits = self.weight_head @ x + self.bias_head
        probs = F.softmax(logits, dim=-1)
        pred = int(torch.argmax(logits).item())
        return pred, probs

    def adapt_step(self, x: torch.Tensor, y: int) -> None:
        pass  # Clamped to zero: exact causal ablation

    def reset_state(self) -> None:
        pass

    def get_state_memory_bytes(self) -> int:
        return 0

    def get_parameter_count(self) -> int:
        return self.weight_head.numel() + self.bias_head.numel()

    def get_parameter_tensors(self) -> list[torch.Tensor]:
        return [self.weight_head, self.bias_head]


def fit_offline_linear_head(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    num_classes: int,
    epochs: int = 50,
    lr: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Train linear classifier head on stationary training split."""
    dim = train_x.shape[-1]
    linear = nn.Linear(dim, num_classes)
    optimizer = torch.optim.Adam(linear.parameters(), lr=lr, weight_decay=1e-4)

    dataset = torch.utils.data.TensorDataset(train_x, train_y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    linear.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            out = linear(bx)
            loss = F.cross_entropy(out, by)
            loss.backward()
            optimizer.step()

    linear.eval()
    return linear.weight.detach().clone(), linear.bias.detach().clone()


def fit_offline_small_mlp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    num_classes: int,
    hidden_dim: int = 16,
    epochs: int = 50,
    lr: float = 0.01,
) -> nn.Module:
    """Train small MLP classifier on stationary training split."""
    dim = train_x.shape[-1]
    mlp = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_classes),
    )
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=1e-4)

    dataset = torch.utils.data.TensorDataset(train_x, train_y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    mlp.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            out = mlp(bx)
            loss = F.cross_entropy(out, by)
            loss.backward()
            optimizer.step()

    mlp.eval()
    return mlp


def fit_offline_gru_cell(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    num_classes: int,
    hidden_dim: int = 16,
    epochs: int = 40,
    lr: float = 0.01,
) -> tuple[nn.GRUCell, nn.Linear]:
    """Train GRU cell and linear projection on stationary training split."""
    dim = train_x.shape[-1]
    cell = nn.GRUCell(dim, hidden_dim)
    head = nn.Linear(hidden_dim, num_classes)
    params = list(cell.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)

    dataset = torch.utils.data.TensorDataset(train_x, train_y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    cell.train()
    head.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            h = torch.zeros(bx.size(0), hidden_dim)
            h = cell(bx, h)
            out = head(h)
            loss = F.cross_entropy(out, by)
            loss.backward()
            optimizer.step()

    cell.eval()
    head.eval()
    return cell, head


def fit_offline_lstm_cell(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    num_classes: int,
    hidden_dim: int = 16,
    epochs: int = 40,
    lr: float = 0.01,
) -> tuple[nn.LSTMCell, nn.Linear]:
    """Train LSTM cell and linear projection on stationary training split."""
    dim = train_x.shape[-1]
    cell = nn.LSTMCell(dim, hidden_dim)
    head = nn.Linear(hidden_dim, num_classes)
    params = list(cell.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)

    dataset = torch.utils.data.TensorDataset(train_x, train_y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    cell.train()
    head.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            h = torch.zeros(bx.size(0), hidden_dim)
            c = torch.zeros(bx.size(0), hidden_dim)
            h, c = cell(bx, (h, c))
            out = head(h)
            loss = F.cross_entropy(out, by)
            loss.backward()
            optimizer.step()

    cell.eval()
    head.eval()
    return cell, head
