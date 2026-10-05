r"""Autograd tests for self-referential coupled memory dynamics.

Verifies:
1. Backward gradients flow through the coupled state transitions:
   key, target, content_memory, dynamics_memory, target_generator parameters.
2. Dynamics memory update gradient propagation.
3. Numerical gradient validation using torch.autograd.gradcheck in FP64.
"""

import torch

from deltacore.memory.associative import AssociativeMemory
from deltacore.updates.dynamics_memory import DynamicsMemory
from deltacore.updates.self_referential import (
    CoupledState,
    SelfReferentialSystem,
)


class TestSelfReferentialAutograd:
    """Verifies that the self-referential coupled system maintains a continuous autograd graph."""

    def test_gradients_through_coupled_step(self):
        """Verify backward gradients flow into content memory, dynamics memory, key, and target."""
        v_dim, k_dim, d_c = 2, 2, 3
        system = SelfReferentialSystem(eta_max=1.0, rho=0.2).to(dtype=torch.float64)

        m_data = torch.randn(v_dim, k_dim, dtype=torch.float64, requires_grad=True)
        c_data = torch.randn(1, d_c, dtype=torch.float64, requires_grad=True)

        k = torch.randn(k_dim, dtype=torch.float64, requires_grad=True)
        v = torch.randn(v_dim, dtype=torch.float64, requires_grad=True)

        state = CoupledState(
            content_memory=AssociativeMemory(m_data),
            dynamics_memory=DynamicsMemory(c_data),
        )

        res = system.step(k, v, state=state)

        # Scalar loss combining both new memories and emitted prediction
        loss = (
            (res.new_content_memory.data**2).sum()
            + (res.new_dynamics_memory.data**2).sum()
            + (res.prediction**2).sum()
        )
        loss.backward()

        assert m_data.grad is not None and torch.isfinite(m_data.grad).all()
        assert c_data.grad is not None and torch.isfinite(c_data.grad).all()
        assert k.grad is not None and torch.isfinite(k.grad).all()
        assert v.grad is not None and torch.isfinite(v.grad).all()

        assert torch.norm(m_data.grad) > 0.0
        assert torch.norm(c_data.grad) > 0.0
        assert torch.norm(k.grad) > 0.0
        assert torch.norm(v.grad) > 0.0

    def test_gradcheck_coupled_step(self):
        """Run torch.autograd.gradcheck in FP64 on the functional coupled step."""
        system = SelfReferentialSystem(eta_max=0.5, rho=0.1).to(dtype=torch.float64)

        def step_fn(m, c, k, v):
            state = CoupledState(
                content_memory=AssociativeMemory(m),
                dynamics_memory=DynamicsMemory(c),
            )
            res = system.step(k, v, state=state)
            return (
                res.new_content_memory.data,
                res.new_dynamics_memory.data,
                res.prediction,
            )

        m = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
        c = torch.randn(1, 3, dtype=torch.float64, requires_grad=True)
        k = torch.randn(2, dtype=torch.float64, requires_grad=True)
        v = torch.randn(2, dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradcheck(step_fn, (m, c, k, v), eps=1e-6, atol=1e-4)

    def test_sequential_scan_autograd(self):
        """Verify backpropagation through unrolled self-referential scan."""
        t, k_dim, v_dim = 3, 2, 2
        system = SelfReferentialSystem(eta_max=0.5, rho=0.1).to(dtype=torch.float64)

        keys = torch.randn(t, k_dim, dtype=torch.float64, requires_grad=True)
        targets = torch.randn(t, v_dim, dtype=torch.float64, requires_grad=True)

        m_init = torch.zeros(v_dim, k_dim, dtype=torch.float64, requires_grad=True)
        c_init = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)

        init_state = CoupledState(
            content_memory=AssociativeMemory(m_init),
            dynamics_memory=DynamicsMemory(c_init),
        )

        scan_res = system.scan(keys, targets, initial_state=init_state)

        loss = 0.5 * torch.sum((targets - scan_res.predictions) ** 2)
        loss.backward()

        assert m_init.grad is not None and torch.isfinite(m_init.grad).all()
        assert c_init.grad is not None and torch.isfinite(c_init.grad).all()
        assert keys.grad is not None and torch.isfinite(keys.grad).all()
        assert targets.grad is not None and torch.isfinite(targets.grad).all()

        assert torch.norm(m_init.grad) > 0.0
        assert torch.norm(c_init.grad) > 0.0
