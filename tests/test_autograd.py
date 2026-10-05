r"""Autograd compatibility tests for DeltaCore reference primitives.

Gradient Semantics:
1. Read Operation:
   Given \hat{v} = M @ k and scalar loss L = w^T \hat{v}:
   - dL / dM = w (x) k = w @ k^T
   - dL / dk = M^T @ w

2. Delta Update:
   Given M' = M + eta * (v - M @ k) @ k^T:
   - Gradients flow continuously into M, k, and v.

3. Sequential Scan:
   Gradients flow back through the unrolled recurrent state transitions across
   all time steps into initial memory, keys, and targets.
"""

import torch

from deltacore.memory.read import read
from deltacore.scans.sequential import sequential_scan
from deltacore.updates.delta import DeltaRule
from deltacore.updates.hebbian import HebbianRule


class TestAutograd:
    """Verifies that DeltaCore reference primitives maintain clean PyTorch autograd graphs."""

    def test_gradients_through_read(self):
        """Verify analytical gradients through read: dL/dM = w @ k^T, dL/dk = M^T @ w."""
        m = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        k = torch.randn(4, dtype=torch.float64, requires_grad=True)
        w = torch.randn(3, dtype=torch.float64)

        v_hat = read(m, k)
        loss = torch.dot(w, v_hat)
        loss.backward()

        expected_grad_m = torch.outer(w, k)
        expected_grad_k = m.detach().T @ w

        assert m.grad is not None
        assert k.grad is not None
        assert torch.allclose(m.grad, expected_grad_m, atol=1e-10)
        assert torch.allclose(k.grad, expected_grad_k, atol=1e-10)

    def test_read_gradcheck(self):
        """Verify numerical gradient check using torch.autograd.gradcheck."""
        m = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
        k = torch.randn(2, dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradcheck(read, (m, k), eps=1e-6, atol=1e-4)

    def test_gradients_through_hebbian_update(self):
        """Verify backward gradients flow through HebbianRule."""
        rule = HebbianRule(step_size=0.5)
        m = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        k = torch.randn(3, dtype=torch.float64, requires_grad=True)
        v = torch.randn(2, dtype=torch.float64, requires_grad=True)

        m_new = rule.update(m, k, v)
        loss = m_new.sum()
        loss.backward()

        assert m.grad is not None and torch.isfinite(m.grad).all()
        assert k.grad is not None and torch.isfinite(k.grad).all()
        assert v.grad is not None and torch.isfinite(v.grad).all()

    def test_gradients_through_delta_update(self):
        """Verify backward gradients flow through DeltaRule into memory, key, and target."""
        rule = DeltaRule(step_size=0.5)
        m = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        k = torch.randn(3, dtype=torch.float64, requires_grad=True)
        v = torch.randn(2, dtype=torch.float64, requires_grad=True)

        m_new = rule.update(m, k, v)
        loss = (m_new**2).sum()
        loss.backward()

        assert m.grad is not None and torch.isfinite(m.grad).all()
        assert k.grad is not None and torch.isfinite(k.grad).all()
        assert v.grad is not None and torch.isfinite(v.grad).all()

    def test_delta_update_gradcheck(self):
        """Run torch.autograd.gradcheck on DeltaRule functional update."""
        rule = DeltaRule(step_size=0.2)

        def update_fn(m, k, v):
            return rule.update(m, k, v)

        m = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
        k = torch.randn(2, dtype=torch.float64, requires_grad=True)
        v = torch.randn(2, dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradcheck(update_fn, (m, k, v), eps=1e-6, atol=1e-4)

    def test_sequential_scan_unrolling_gradients(self):
        """Verify end-to-end backpropagation through unrolled sequential scan."""
        b, t, k_dim, v_dim = 2, 4, 3, 2
        keys = torch.randn(b, t, k_dim, dtype=torch.float64, requires_grad=True)
        targets = torch.randn(b, t, v_dim, dtype=torch.float64, requires_grad=True)
        init_m = torch.zeros(b, v_dim, k_dim, dtype=torch.float64, requires_grad=True)

        res = sequential_scan(
            keys, targets, initial_memory=init_m, rule=DeltaRule(step_size=0.1)
        )

        # Loss is sum of squared prediction errors
        loss = 0.5 * torch.sum((targets - res.predictions) ** 2)
        loss.backward()

        assert init_m.grad is not None and torch.isfinite(init_m.grad).all()
        assert keys.grad is not None and torch.isfinite(keys.grad).all()
        assert targets.grad is not None and torch.isfinite(targets.grad).all()

        # Non-zero gradients verify signals flowed back through all recurrent steps
        assert torch.norm(init_m.grad) > 0.0
        assert torch.norm(keys.grad) > 0.0
        assert torch.norm(targets.grad) > 0.0
